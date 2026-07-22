#!/usr/bin/env python3
"""
calibrate_power.py — sampling library for the calibration/eval tooling.

Provides the shared measurement plumbing: GPU power sampling via nvidia-smi
(PowerSampler), DRAM-bytes sampling via the DCGM DRAM-active counter
(BytesSampler), idle-baseline sampling (sample_idle), and the instrumented
workload runner (run_workload) that integrates net energy against
FlopCounterMode ground truth.

PORTED from the Jetson Orin Nano (INA3221 + tegrastats + actmon) to an x86
dual-Tesla-V100 box: power/util now come from nvidia-smi (unprivileged) and DRAM
activity from DCGM field 1005 via `dcgmi dmon` (needs a root nv-hostengine, but
the dcgmi client stays unprivileged). See detect_flops.py for the signal map.

Not a standalone tool: the calibration/accuracy entry point is
eval_power_monitor.py (which fits the active-energy models and prints the
RECOMMENDED CONSTANTS block for detect_flops.py). Other consumers:
actmon_scale_bench.py and writeup/capture_timeseries.py.
"""

import os
import re
import subprocess
import sys
import threading
import time
from statistics import mean, median, stdev

# ── Sampling ──────────────────────────────────────────────────────────────────
POLL_S = 0.5   # nvidia-smi / DCGM sample interval (2 Hz)

# This file lives in power_calibration/; the workload script and the venv are at
# the repo root one level up.
REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKLOAD_SCRIPT = os.path.join(REPO_ROOT, "sample_ml_workload.py")

# Canonical hardware primitives live in detect_flops.py; re-exported here so the
# eval/probe/bench scripts keep importing them from calibrate_power.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from detect_flops import (  # noqa: E402
    GPU_INDEX, DRAM_ACTIVE_FIELD, V100_PROFILE, actmon_bytes_per_s,
    read_gpu_sample, read_power_mw, _parse_dcgm_dram_line,
)


# ── Python interpreter ────────────────────────────────────────────────────────

def find_venv_python():
    """
    Prefer the .venv Python over sys.executable, so the launched workload has the
    CUDA-enabled torch from the venv regardless of how this script was invoked.
    """
    for name in ("python3", "python"):
        p = os.path.join(REPO_ROOT, ".venv", "bin", name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return sys.executable


def child_env(gpu_index=GPU_INDEX):
    """Environment for the workload subprocess: pin CUDA to the monitored GPU.
    With CUDA_DEVICE_ORDER=PCI_BUS_ID, nvidia-smi index gpu_index is the same
    physical GPU that CUDA_VISIBLE_DEVICES=gpu_index selects (as cuda:0), so the
    process we measure is the process doing the work."""
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return env


# ── Background power sampler ──────────────────────────────────────────────────

class PowerSampler:
    """Samples nvidia-smi power + GPU utilization at POLL_S intervals in a daemon
    thread. (The Jetson version also scraped tegrastats EMC%; on V100 DRAM
    activity comes from the DCGM BytesSampler instead, so emc_samples is dropped.)"""

    def __init__(self, gpu_index=GPU_INDEX):
        self.gpu_index      = gpu_index
        self.power_samples  = []   # (monotonic_t, power_mw)
        self.gpu_samples    = []   # (monotonic_t, gpu_pct)
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
                t = time.monotonic()
                mw, gpu, _clk = read_gpu_sample(self.gpu_index)
                self.power_samples.append((t, mw))
                self.gpu_samples.append((t, gpu))
                self._stop.wait(POLL_S)
        except BaseException as e:
            self._exc = e


# ── DRAM bytes sampler (DCGM DRAM-active, field 1005) ─────────────────────────

class BytesSampler:
    """
    Streams the DRAM-active fraction from `dcgmi dmon -e 1005` (DCGM field 1005,
    DCGM_FI_PROF_DRAM_ACTIVE) so the calibration sweep can integrate DRAM bytes
    moved. Mirrors PowerSampler's lifecycle; designed to start/stop on the SAME
    window as the PowerSampler so bytes are commensurable with net_energy_j /
    duration_s.

    Requires a running nv-hostengine (root; `sudo nv-hostengine`), but the dcgmi
    client itself is unprivileged — no sudo here. A spawn failure raises
    immediately; if the stream produces no samples (e.g. host engine down) the
    caller is expected to treat that as a hard error (see run_workload /
    require_samples), not silently drop the byte term. The reader's stderr is
    merged into stdout so its diagnostics are visible.
    """

    def __init__(self, gpu_index=GPU_INDEX):
        self.gpu_index = gpu_index
        self.samples = []          # (monotonic_t, bytes_per_s)
        self.available = False
        self._stop = threading.Event()
        self._exc = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.proc = subprocess.Popen(
            ["dcgmi", "dmon", "-e", str(DRAM_ACTIVE_FIELD),
             "-i", str(gpu_index), "-d", str(int(POLL_S * 1000))],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )

    def start(self):
        self._thread.start()

    def stop(self):
        # Stop the stream by closing the read end of stdout and terminating the
        # (unprivileged) dcgmi client.
        self._stop.set()
        try:
            self.proc.terminate()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        try:
            self.proc.stdout.close()
        except OSError:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._exc is not None:
            raise self._exc

    def _run(self):
        try:
            for line in iter(self.proc.stdout.readline, ""):
                if self._stop.is_set():
                    break
                frac = _parse_dcgm_dram_line(line)
                if frac is None:
                    continue
                frac = max(0.0, min(1.0, frac))
                self.available = True
                self.samples.append((time.monotonic(), actmon_bytes_per_s(frac)))
        except BaseException as e:
            # Only an intentional stop is expected here; anything else re-raises.
            if not self._stop.is_set():
                self._exc = e

    def require_samples(self):
        """Raise if the DCGM stream produced no samples (host engine down / field
        1005 unavailable on this GPU/driver)."""
        if not self.available:
            raise RuntimeError(
                "DCGM DRAM-active (field 1005) produced no samples — is "
                "nv-hostengine running (sudo nv-hostengine) and does "
                "`dcgmi dmon -e 1005` work on this GPU?")

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

    def avg_dram_pct(self):
        """Average DRAM-active fraction over the window, as a percent (the V100
        analog of the Jetson EMC%). None if no samples."""
        if not self.samples:
            return None
        peak = V100_PROFILE["PEAK_BW_BYTES_S"]
        return mean(b for _, b in self.samples) / peak * 100.0


# ── Idle sampling ─────────────────────────────────────────────────────────────

def sample_idle(duration_s, gpu_index, label):
    """Blocking idle sample; returns list of (mono_t, power_mw)."""
    print(f"  [{label}] sampling idle for {duration_s:.0f}s ...", flush=True)
    sampler = PowerSampler(gpu_index)
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


def run_workload(config, idle_baseline_mw, gpu_index=GPU_INDEX):
    """
    Launch sample_ml_workload.py (pinned to gpu_index), sample power + DRAM in
    parallel, return result dict. stdout is echoed live; power is collected by a
    background thread.
    """
    # "-u" forces the child's stdout unbuffered so the "[redteam] Starting
    # workload..." trigger below reaches this parent promptly (else piped stdout
    # is block-buffered and the power sampler starts far too late).
    #
    # A config may override the workload entirely via {"script": path,
    # "args": [...]} — any script that speaks the same stdout protocol (the
    # "Starting workload" trigger + "Ground truth total : X TFLOPs" line) goes
    # through this exact sampling path.
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
        env=child_env(gpu_index),
    )

    # Power sampling starts only after "[redteam] Starting workload..." so that
    # CUDA init, model creation, the warmup pass, and the FLOPs probe are all
    # excluded from the energy integral (they are not in the GT FLOP count).
    sampler = None
    bytes_sampler = None
    t_start = None

    output_lines = []
    for line in iter(proc.stdout.readline, ""):
        print(line, end="", flush=True)
        output_lines.append(line)
        if sampler is None and "[redteam] Starting workload" in line:
            t_start = time.monotonic()
            sampler = PowerSampler(gpu_index)
            sampler.start()
            # Same window as the power sampler so bytes are commensurable.
            bytes_sampler = BytesSampler(gpu_index)
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
    bytes_sampler.require_samples()   # loud if DCGM produced nothing
    tb_moved = bytes_sampler.total_tb()
    avg_bytes_per_s = bytes_sampler.avg_bytes_per_s()
    avg_dram_pct = bytes_sampler.avg_dram_pct()
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
        "emc_samples":      [],
        "ground_truth_tf":  gt_tflops,
        "net_energy_j":     net_energy_j,
        "avg_net_power_w":  avg_net_w,
        "j_per_tflop":      j_per_tflop,
        "avg_raw_mw":       mean(raw_mw)    if raw_mw else None,
        "peak_raw_mw":      max(raw_mw)     if raw_mw else None,
        "avg_gpu_pct":      mean(g for _, g in sampler.gpu_samples)
                            if sampler.gpu_samples else None,
        "avg_emc_pct":      avg_dram_pct,
        "tb_moved":         tb_moved,
        "avg_bytes_per_s":  avg_bytes_per_s,
        "n_power_samples":  len(sampler.power_samples),
    }
