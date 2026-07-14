#!/usr/bin/env python3
"""
calibrate_power.py — sampling library for the calibration/eval tooling.

Provides the shared measurement plumbing: INA3221 power sampling
(PowerSampler), DRAM-bytes sampling via the privileged actmon reader
(BytesSampler), tegrastats GPU-utilization scraping, idle-baseline sampling
(sample_idle), and the instrumented workload runner (run_workload) that
integrates net energy against FlopCounterMode ground truth.

Not a standalone tool: the calibration/accuracy entry point is
eval_power_monitor.py (which fits the active-energy models and prints the
RECOMMENDED CONSTANTS block for detect_flops.py). Other consumers:
drift_test.py, adversarial_probe.py, actmon_scale_bench.py,
writeup/capture_timeseries.py. The legacy standalone calibrator this module
grew out of produced old/power_calibration.txt.
"""

import os
import re
import select
import subprocess
import sys
import threading
import time
from statistics import mean, median, stdev

# ── Sampling ──────────────────────────────────────────────────────────────────
POLL_S = 0.5   # INA3221 sample interval (2 Hz)

# This file lives in power_calibration/; the workload script and the venv are at
# the repo root one level up.
REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKLOAD_SCRIPT = os.path.join(REPO_ROOT, "sample_ml_workload.py")

# ── INA3221 sensor ────────────────────────────────────────────────────────────
# Canonical INA3221 constants/helpers live in detect_flops.py; re-exported here
# so the eval/probe/drift scripts keep importing them from calibrate_power.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from detect_flops import (  # noqa: E402
    INA3221_DRIVER, INA3221_LABEL, actmon_bytes_per_s, find_ina3221_paths,
    read_power_mw,
)


# ── Python interpreter ────────────────────────────────────────────────────────

def find_venv_python():
    """
    Prefer the .venv Python over sys.executable. When the script is invoked via
    sudo, sys.executable is the system Python which lacks the bundled CUDA libs
    that the venv torch ships with.
    """
    for name in ("python3", "python"):
        p = os.path.join(REPO_ROOT, ".venv", "bin", name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return sys.executable


# ── tegrastats helpers ────────────────────────────────────────────────────────

def start_tegrastats(interval_ms):
    # Let a launch failure (tegrastats missing) propagate — surfaced, not masked.
    return subprocess.Popen(
        ["tegrastats", "--interval", str(interval_ms)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1,
    )


def drain_tegrastats(proc):
    """Non-blocking drain; return latest (gpu_pct, emc_pct) or (None, None)."""
    if proc is None:
        return None, None
    gpu = emc = None
    while True:
        ready, _, _ = select.select([proc.stdout], [], [], 0)
        if not ready:
            break
        line = proc.stdout.readline()
        if not line:
            break
        m = re.search(r'GR3D_FREQ\s+(\d+)%', line)
        if m:
            gpu = float(m.group(1))
        m = re.search(r'EMC_FREQ\s+(\d+)%', line)
        if m:
            emc = float(m.group(1))
    return gpu, emc


def stop_tegrastats(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()


# ── Background power sampler ──────────────────────────────────────────────────

class PowerSampler:
    """Samples INA3221 + tegrastats at POLL_S intervals in a daemon thread."""

    def __init__(self, volt_path, curr_path, ts_proc):
        self.volt_path      = volt_path
        self.curr_path      = curr_path
        self.ts_proc        = ts_proc
        self.power_samples  = []   # (monotonic_t, power_mw)
        self.gpu_samples    = []   # (monotonic_t, gpu_pct)
        self.emc_samples    = []   # (monotonic_t, emc_pct)
        self._stop          = threading.Event()
        self._exc           = None
        self._thread        = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        # Surface any failure that happened in the sampling thread — a swallowed
        # thread exception would silently corrupt the energy integral.
        if self._exc is not None:
            raise self._exc

    def _run(self):
        try:
            while not self._stop.is_set():
                t   = time.monotonic()
                mw  = read_power_mw(self.volt_path, self.curr_path)
                gpu, emc = drain_tegrastats(self.ts_proc)
                self.power_samples.append((t, mw))
                # gpu/emc are None only until tegrastats emits its first line — a
                # genuine "no data yet", not a failure, so those stay guarded.
                if gpu is not None:
                    self.gpu_samples.append((t, gpu))
                if emc is not None:
                    self.emc_samples.append((t, emc))
                self._stop.wait(POLL_S)
        except BaseException as e:
            self._exc = e


# ── DRAM bytes sampler (actmon, via sudoers reader) ───────────────────────────

ACTMON_READER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "actmon_reader.py")


class BytesSampler:
    """
    Streams DRAM bandwidth from the privileged actmon reader (root-only debugfs)
    so the unprivileged calibration sweep can integrate bytes moved. Mirrors
    PowerSampler's lifecycle; designed to start/stop on the SAME window as the
    PowerSampler so bytes are commensurable with net_energy_j / duration_s.

    Requires the NOPASSWD sudoers entry for actmon_reader.py (see that file's
    header). A spawn failure raises immediately; if the reader produces no samples
    (e.g. sudoers not configured) the caller is expected to treat that as a hard
    error (see run_workload / require_samples), not silently drop the byte term.
    The reader's stderr is inherited so its diagnostics are visible.
    """

    def __init__(self):
        self._to_bytes_per_s = actmon_bytes_per_s  # single source of truth
        self.samples = []          # (monotonic_t, bytes_per_s)
        self.available = False
        self._stop = threading.Event()
        self._exc = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        # -n: never prompt; fail immediately if NOPASSWD is not configured. A
        # Popen failure (sudo missing) propagates rather than being masked.
        # stderr inherited (not DEVNULL) so the reader's "must run as root" message
        # and any traceback are visible for debugging.
        self.proc = subprocess.Popen(
            ["sudo", "-n", find_venv_python(), ACTMON_READER_SCRIPT,
             "--interval", str(POLL_S)],
            stdout=subprocess.PIPE, text=True, bufsize=1,
        )

    def start(self):
        self._thread.start()

    def stop(self):
        # The reader runs as root (via sudo); a non-root terminate()/kill() would
        # EPERM, so we stop it by CLOSING the read end of its stdout pipe — the
        # reader's next write then raises BrokenPipeError and it self-exits (see
        # actmon_reader.py). This avoids leaking a root process per workload.
        self._stop.set()
        try:
            self.proc.stdout.close()
        except OSError:
            pass
        # No TimeoutExpired swallow: if the reader did NOT exit after we closed the
        # pipe, that is a real problem (leaked root process) we want surfaced.
        self.proc.wait(timeout=5)
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        # Surface any failure that happened in the reader-draining thread.
        if self._exc is not None:
            raise self._exc

    def _run(self):
        try:
            for line in iter(self.proc.stdout.readline, ""):
                if self._stop.is_set():
                    break
                parts = line.split()
                if not parts:
                    continue
                # No try/except around the parse: a malformed line from the reader
                # is unexpected and should surface, not be silently skipped.
                util = float(parts[1])
                emc_rate = float(parts[2])
                self.available = True
                self.samples.append((time.monotonic(), self._to_bytes_per_s(util, emc_rate)))
        except BaseException as e:
            # Only an intentional stop (pipe closed by stop()) is expected here;
            # anything else is a real error to re-raise from stop().
            if not self._stop.is_set():
                self._exc = e

    def require_samples(self):
        """Raise if the reader produced no samples (sudoers/root misconfigured)."""
        if not self.available:
            raise RuntimeError(
                "actmon reader produced no samples — is the NOPASSWD sudoers entry "
                "for actmon_reader.py configured and are you able to sudo -n?")

    def total_tb(self):
        """Trapezoidal integration of bytes/s → terabytes, or None if too few samples."""
        if len(self.samples) < 2:
            return None
        total_bytes = 0.0
        for i in range(1, len(self.samples)):
            t0, b0 = self.samples[i - 1]
            t1, b1 = self.samples[i]
            total_bytes += 0.5 * (b0 + b1) * (t1 - t0)
        return total_bytes / 1e12

    def avg_bytes_per_s(self):
        if not self.samples:
            return None
        return mean(b for _, b in self.samples)


# ── Idle sampling ─────────────────────────────────────────────────────────────

def sample_idle(duration_s, volt_path, curr_path, ts_proc, label):
    """Blocking idle sample; returns list of (mono_t, power_mw)."""
    print(f"  [{label}] sampling idle for {duration_s:.0f}s ...", flush=True)
    sampler = PowerSampler(volt_path, curr_path, ts_proc)
    sampler.start()
    time.sleep(duration_s)
    sampler.stop()
    n = len(sampler.power_samples)
    if n:
        vals = [mw for _, mw in sampler.power_samples]
        print(f"  [{label}] {n} samples  |  median {median(vals):.1f} mW"
              f"  stdev {stdev(vals) if n > 1 else 0:.1f} mW", flush=True)
    return sampler.power_samples


# ── Workload execution ────────────────────────────────────────────────────────

def parse_ground_truth_tflops(text):
    """Extract TFLOPs total from sample_ml_workload.py stdout."""
    m = re.search(r'Ground truth total\s*:\s*([\d.]+)\s*TFLOPs', text)
    return float(m.group(1)) if m else None


def compute_net_energy(power_samples, idle_baseline_mw):
    """Trapezoidal integration of net power over the sample timeseries."""
    net_energy_j   = 0.0
    net_mw_list    = []
    for i in range(1, len(power_samples)):
        t0, mw0 = power_samples[i - 1]
        t1, mw1 = power_samples[i]
        dt      = t1 - t0
        n0      = max(mw0 - idle_baseline_mw, 0.0) / 1000.0  # W
        n1      = max(mw1 - idle_baseline_mw, 0.0) / 1000.0
        net_energy_j += 0.5 * (n0 + n1) * dt
        net_mw_list.append(0.5 * (n0 + n1) * 1000.0)
    avg_net_mw = mean(net_mw_list) if net_mw_list else None
    return net_energy_j, avg_net_mw


def run_workload(config, idle_baseline_mw, volt_path, curr_path, ts_proc):
    """
    Launch sample_ml_workload.py, sample power in parallel, return result dict.
    stdout is echoed live; power is collected by a background thread.
    """
    # "-u" forces the child's stdout unbuffered. Without it, piped stdout is
    # block-buffered and the "[redteam] Starting workload..." trigger below does
    # not reach this parent until the buffer flushes (near exit for short runs),
    # starting the power sampler far too late — the race that corrupted the
    # original calibration (only 3-32 samples on shorter runs).
    #
    # A config may override the workload entirely via {"script": path,
    # "args": [...]} — any script that speaks the same stdout protocol (the
    # "Starting workload" trigger + "Ground truth total : X TFLOPs" line) goes
    # through this exact sampling path, e.g. adversarial_workload.py.
    if "script" in config:
        cmd = ([find_venv_python(), "-u", config["script"]]
               + [str(a) for a in config.get("args", [])])
    else:
        cmd = [
            find_venv_python(), "-u", WORKLOAD_SCRIPT,
            "--steps",      str(config["steps"]),
            "--batch-size", str(config["batch_size"]),
            "--seq-len",    str(config["seq_len"]),
            "--d-model",    str(config["d_model"]),
        ]
        for key, flag in (("num_layers", "--num-layers"),
                          ("nhead", "--nhead"),
                          ("dim_feedforward", "--dim-feedforward"),
                          ("precision", "--precision"),
                          ("optimizer", "--optimizer")):
            if key in config:
                cmd += [flag, str(config[key])]
    print(f"  Launching: {' '.join(cmd[3:])}", flush=True)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    # Power sampling starts only after "[redteam] Starting workload..." so that
    # CUDA init, model creation, the warmup pass, and the FLOPs probe are all
    # excluded from the energy integral (they are not in the GT FLOP count).
    # Read via readline (not `for line in proc.stdout`) to avoid the iterator's
    # read-ahead buffering, which would also delay the trigger.
    sampler = None
    bytes_sampler = None
    t_start = None

    output_lines = []
    for line in iter(proc.stdout.readline, ""):
        print(line, end="", flush=True)
        output_lines.append(line)
        if sampler is None and "[redteam] Starting workload" in line:
            t_start = time.monotonic()
            sampler = PowerSampler(volt_path, curr_path, ts_proc)
            sampler.start()
            # Same window as the power sampler so bytes are commensurable.
            bytes_sampler = BytesSampler()
            bytes_sampler.start()
    proc.wait()
    stdout_text = "".join(output_lines)

    # The trigger gates sampling; if it never appeared the workload did not start
    # as expected — a hard error, not a silent zero-duration run.
    if sampler is None:
        raise RuntimeError(
            f"workload never emitted '[redteam] Starting workload' "
            f"(exit code {proc.returncode}); cannot sample. Last output:\n"
            f"{stdout_text[-800:]}")

    sampler.stop()
    duration_s = time.monotonic() - t_start
    bytes_sampler.stop()          # created alongside sampler, so never None here
    bytes_sampler.require_samples()   # loud if actmon produced nothing
    tb_moved = bytes_sampler.total_tb()
    avg_bytes_per_s = bytes_sampler.avg_bytes_per_s()
    gt_tflops = parse_ground_truth_tflops(stdout_text)

    if proc.returncode != 0:
        print(f"  WARNING: workload exited with code {proc.returncode}", flush=True)

    net_energy_j, avg_net_mw = compute_net_energy(sampler.power_samples, idle_baseline_mw)
    avg_net_w   = avg_net_mw / 1000.0 if avg_net_mw is not None else None
    j_per_tflop = (net_energy_j / gt_tflops
                   if gt_tflops and gt_tflops > 0 and net_energy_j > 0 else None)

    raw_mw = [mw for _, mw in sampler.power_samples]

    return {
        "config":           config,
        "returncode":       proc.returncode,
        "duration_s":       duration_s,
        "idle_baseline_mw": idle_baseline_mw,
        "power_samples":    sampler.power_samples,
        "gpu_samples":      sampler.gpu_samples,
        "emc_samples":      sampler.emc_samples,
        "ground_truth_tf":  gt_tflops,
        "net_energy_j":     net_energy_j,
        "avg_net_power_w":  avg_net_w,
        "j_per_tflop":      j_per_tflop,
        "avg_raw_mw":       mean(raw_mw)    if raw_mw else None,
        "peak_raw_mw":      max(raw_mw)     if raw_mw else None,
        "avg_gpu_pct":      mean(g for _, g in sampler.gpu_samples)
                            if sampler.gpu_samples else None,
        "avg_emc_pct":      mean(e for _, e in sampler.emc_samples)
                            if sampler.emc_samples else None,
        "tb_moved":         tb_moved,
        "avg_bytes_per_s":  avg_bytes_per_s,
        "n_power_samples":  len(sampler.power_samples),
    }
