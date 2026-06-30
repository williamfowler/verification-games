#!/usr/bin/env python3
"""
calibrate_power.py — Power/FLOP calibration for detect_flops.py

Runs sample_ml_workload.py at several d_model sizes, samples INA3221 power
at 2 Hz in a background thread, and computes J/TFLOP against the
FlopCounterMode ground truth for each run.

Usage:
    python3 calibrate_power.py [--output FILE]

Output:
    calibration_results.txt  (summary + raw timeseries for manual inspection)

Expected runtime: 40–90 min depending on Jetson performance.
Ensure no other GPU workloads are running before starting.
"""

import argparse
import glob
import os
import re
import select
import subprocess
import sys
import threading
import time
from statistics import mean, median, stdev

# ── INA3221 sensor ────────────────────────────────────────────────────────────
INA3221_DRIVER = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon"
INA3221_LABEL  = "VDD_CPU_GPU_CV"

# ── Sampling ──────────────────────────────────────────────────────────────────
POLL_S          = 0.5   # INA3221 sample interval (2 Hz)
IDLE_BASELINE_S = 90    # quiet sampling before first workload (≥ 180 samples)
COOLDOWN_S      = 45    # quiet sampling between workloads

# ── Workload matrix ───────────────────────────────────────────────────────────
# d_model spans memory-bound (128) to compute-bound (1024) on the Orin Nano.
# Steps sized for ~150 s of active training per run (≥300 power samples) based
# on observed step rates (~21 steps/s at d_model=128, ~7 steps/s at d_model=1024).
# d_model=1024 is included but will be skipped gracefully if it OOMs.
CONFIGS = [
    {"d_model": 128,  "steps": 3000, "batch_size": 8, "seq_len": 64},
    {"d_model": 256,  "steps": 2500, "batch_size": 8, "seq_len": 64},
    {"d_model": 512,  "steps": 1500, "batch_size": 8, "seq_len": 64},
    {"d_model": 1024, "steps": 500,  "batch_size": 8, "seq_len": 64},
]

DEFAULT_OUTPUT  = "calibration_results.txt"
# This file lives in power_calibration/; the workload script and the venv are at
# the repo root one level up.
REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKLOAD_SCRIPT = os.path.join(REPO_ROOT, "sample_ml_workload.py")


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


# ── INA3221 helpers ───────────────────────────────────────────────────────────

def find_ina3221_paths():
    """Return (volt_path, curr_path) for INA3221_LABEL, or (None, None)."""
    for hwmon in glob.glob(f"{INA3221_DRIVER}/hwmon*"):
        for i in range(1, 5):
            try:
                with open(f"{hwmon}/in{i}_label") as f:
                    if f.read().strip() == INA3221_LABEL:
                        return f"{hwmon}/in{i}_input", f"{hwmon}/curr{i}_input"
            except OSError:
                continue
    return None, None


def read_power_mw(volt_path, curr_path):
    try:
        with open(volt_path) as f:
            mv = float(f.read().strip())
        with open(curr_path) as f:
            ma = float(f.read().strip())
        return mv * ma / 1000.0
    except OSError:
        return None


# ── tegrastats helpers ────────────────────────────────────────────────────────

def start_tegrastats(interval_ms):
    try:
        return subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
    except OSError:
        return None


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
        self._thread        = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _run(self):
        while not self._stop.is_set():
            t   = time.monotonic()
            mw  = read_power_mw(self.volt_path, self.curr_path)
            gpu, emc = drain_tegrastats(self.ts_proc)
            if mw  is not None:
                self.power_samples.append((t, mw))
            if gpu is not None:
                self.gpu_samples.append((t, gpu))
            if emc is not None:
                self.emc_samples.append((t, emc))
            self._stop.wait(POLL_S)


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
    header). If sudo / the reader is unavailable, no samples are collected and
    total_tb() returns None — the eval then falls back to the 2-parameter fit.
    """

    def __init__(self):
        from detect_flops import actmon_bytes_per_s  # single source of truth
        self._to_bytes_per_s = actmon_bytes_per_s
        self.samples = []          # (monotonic_t, bytes_per_s)
        self.available = False
        self.proc = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        try:
            # -n: never prompt; fail immediately if NOPASSWD is not configured.
            self.proc = subprocess.Popen(
                ["sudo", "-n", find_venv_python(), ACTMON_READER_SCRIPT,
                 "--interval", str(POLL_S)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1,
            )
        except OSError:
            self.proc = None

    def start(self):
        if self.proc is not None:
            self._thread.start()

    def stop(self):
        # The reader runs as root (via sudo); a non-root terminate()/kill() would
        # EPERM, so we stop it by CLOSING the read end of its stdout pipe — the
        # reader's next write then raises BrokenPipeError and it self-exits (see
        # actmon_reader.py). This avoids leaking a root process per workload.
        self._stop.set()
        if self.proc is not None:
            try:
                self.proc.stdout.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=3)   # reader exits via EPIPE within ~1 poll
            except subprocess.TimeoutExpired:
                pass                         # cannot signal a root proc; it should be gone
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self):
        try:
            for line in iter(self.proc.stdout.readline, ""):
                if self._stop.is_set():
                    break
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    util = float(parts[1])
                    emc_rate = float(parts[2])
                except ValueError:
                    continue
                bps = self._to_bytes_per_s(util, emc_rate)
                if bps is not None:
                    self.available = True
                    self.samples.append((time.monotonic(), bps))
        except (ValueError, OSError):
            pass   # stdout closed by stop() while blocked in readline

    def total_tb(self):
        """Trapezoidal integration of bytes/s → terabytes, or None if no samples."""
        if not self.available or len(self.samples) < 2:
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
    cmd = [
        find_venv_python(), "-u", WORKLOAD_SCRIPT,
        "--steps",      str(config["steps"]),
        "--batch-size", str(config["batch_size"]),
        "--seq-len",    str(config["seq_len"]),
        "--d-model",    str(config["d_model"]),
    ]
    for key, flag in (("num_layers", "--num-layers"),
                      ("nhead", "--nhead"),
                      ("dim_feedforward", "--dim-feedforward")):
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

    if sampler is not None:
        sampler.stop()
        duration_s = time.monotonic() - t_start
    else:
        duration_s = 0.0
        sampler = PowerSampler(volt_path, curr_path, ts_proc)  # empty sampler
    if bytes_sampler is not None:
        bytes_sampler.stop()
    tb_moved = bytes_sampler.total_tb() if bytes_sampler is not None else None
    avg_bytes_per_s = (bytes_sampler.avg_bytes_per_s()
                       if bytes_sampler is not None else None)
    stdout_text = "".join(output_lines)
    gt_tflops   = parse_ground_truth_tflops(stdout_text)

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


# ── Report writer ─────────────────────────────────────────────────────────────

def write_report(idle_samples, run_results, idle_baseline_mw, output_path):
    idle_mw = [mw for _, mw in idle_samples]

    with open(output_path, "w") as f:
        def w(s=""):
            f.write(s + "\n")

        w("=" * 72)
        w("detect_flops.py  —  Power Calibration Report")
        w(f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        w(f"Sensor    : {INA3221_LABEL}")
        w(f"Poll rate : {1/POLL_S:.0f} Hz  ({POLL_S}s interval)")
        w("=" * 72)
        w()

        # ── Idle baseline ─────────────────────────────────────────────────
        w("IDLE BASELINE  (pre-workload)")
        w("-" * 40)
        w(f"  Duration  : {IDLE_BASELINE_S} s")
        w(f"  Samples   : {len(idle_mw)}")
        w(f"  Median    : {median(idle_mw):.1f} mW")
        w(f"  Stdev     : {stdev(idle_mw) if len(idle_mw) > 1 else 0:.1f} mW")
        w(f"  Min/Max   : {min(idle_mw):.1f} / {max(idle_mw):.1f} mW")
        w()

        # ── Per-run detail ────────────────────────────────────────────────
        w("WORKLOAD RUNS  (one section per config)")
        w("-" * 72)
        for r in run_results:
            cfg  = r["config"]
            ok   = r["returncode"] == 0
            w(f"  d_model={cfg['d_model']}  steps={cfg['steps']}"
              f"  batch={cfg['batch_size']}  seq={cfg['seq_len']}"
              f"  exit={r['returncode']}")
            w(f"    Duration          : {r['duration_s']:.1f} s")
            w(f"    Power samples     : {r['n_power_samples']}"
              f"  (active training only, excl. CUDA init)")
            w(f"    Idle baseline     : {r['idle_baseline_mw']:.1f} mW")
            if r["avg_raw_mw"] is not None:
                w(f"    Avg raw power     : {r['avg_raw_mw']:.1f} mW")
                w(f"    Peak raw power    : {r['peak_raw_mw']:.1f} mW")
            if r["avg_net_power_w"] is not None:
                w(f"    Avg net power     : {r['avg_net_power_w']*1000:.1f} mW"
                  f"  ({r['avg_net_power_w']:.4f} W)")
            w(f"    Net energy        : {r['net_energy_j']:.4f} J")
            if r["ground_truth_tf"] is not None:
                w(f"    Ground truth      : {r['ground_truth_tf']:.6f} TFLOPs")
            else:
                w("    Ground truth      : N/A  (FlopCounterMode unavailable)")
            if r["j_per_tflop"] is not None:
                w(f"    J / TFLOP         : {r['j_per_tflop']:.4f}")
            else:
                w("    J / TFLOP         : N/A")
            if r["avg_gpu_pct"] is not None:
                w(f"    Avg GPU util      : {r['avg_gpu_pct']:.1f}%")
            if r["avg_emc_pct"] is not None:
                w(f"    Avg EMC util      : {r['avg_emc_pct']:.1f}%")
            w()

        # ── Summary table ─────────────────────────────────────────────────
        w("SUMMARY TABLE")
        w("-" * 72)
        hdr = (f"  {'d_model':>8}  {'steps':>6}  {'avg_net_W':>10}"
               f"  {'net_J':>10}  {'gt_TFLOPs':>10}  {'J/TFLOP':>8}")
        w(hdr)
        w("  " + "-" * (len(hdr) - 2))
        for r in run_results:
            cfg  = r["config"]
            anw  = f"{r['avg_net_power_w']:.4f}" if r['avg_net_power_w']  is not None else "N/A"
            gt   = f"{r['ground_truth_tf']:.6f}"  if r['ground_truth_tf']  is not None else "N/A"
            jpt  = f"{r['j_per_tflop']:.4f}"      if r['j_per_tflop']      is not None else "N/A"
            w(f"  {cfg['d_model']:>8}  {cfg['steps']:>6}  {anw:>10}"
              f"  {r['net_energy_j']:>10.4f}  {gt:>10}  {jpt:>8}")
        w()

        # ── Recommended constants ─────────────────────────────────────────
        valid = [r for r in run_results
                 if r["j_per_tflop"] is not None and r["avg_net_power_w"] is not None]
        w("RECOMMENDED CALIBRATION CONSTANTS")
        w("-" * 40)
        if valid:
            valid_sorted = sorted(valid, key=lambda r: r["avg_net_power_w"])
            low_runs  = [r for r in valid_sorted if r["avg_net_power_w"] <= 2.0]
            high_runs = [r for r in valid_sorted if r["avg_net_power_w"] >  2.0]

            if low_runs:
                low_j = mean(r["j_per_tflop"] for r in low_runs)
                low_w = mean(r["avg_net_power_w"] for r in low_runs)
                w(f"  POWER_CAL_LOW_J_PER_TFLOP  = {low_j:.2f}"
                  f"   # {len(low_runs)} run(s), avg net {low_w:.2f} W")
            if high_runs:
                high_j = mean(r["j_per_tflop"] for r in high_runs)
                high_w = mean(r["avg_net_power_w"] for r in high_runs)
                w(f"  POWER_CAL_HIGH_J_PER_TFLOP = {high_j:.2f}"
                  f"   # {len(high_runs)} run(s), avg net {high_w:.2f} W")

            # Interpolation range endpoints
            if low_runs and high_runs:
                low_max_w  = max(r["avg_net_power_w"] for r in low_runs)
                high_min_w = min(r["avg_net_power_w"] for r in high_runs)
                w(f"  POWER_CAL_LOW_NET_W        = {low_max_w:.1f}")
                w(f"  POWER_CAL_HIGH_NET_W       = {high_min_w:.1f}")
            elif low_runs:
                w("  # All runs fell in low-power regime; collect a heavier workload")
                w("  # before setting POWER_CAL_HIGH_* constants.")
            elif high_runs:
                w("  # All runs fell in high-power regime; collect a lighter workload")
                w("  # before setting POWER_CAL_LOW_* constants.")

            w()
            w(f"  FALLBACK_IDLE_POWER_MW     = {median(idle_mw):.0f}"
              f"   # measured idle baseline")
        else:
            w("  No valid runs (FlopCounterMode unavailable or all runs failed).")
        w()

        # ── Raw timeseries ────────────────────────────────────────────────
        w("=" * 72)
        w("RAW POWER TIMESERIES  (for manual inspection / re-fitting)")
        w("Columns: monotonic_t_s  power_mw  net_power_mw")
        w("=" * 72)
        w()

        w(f"# IDLE PHASE  n={len(idle_samples)}  "
          f"median={median(idle_mw):.1f}mW")
        for t, mw in idle_samples:
            w(f"  {t:.3f}  {mw:.1f}  0.0")
        w()

        for r in run_results:
            cfg = r["config"]
            w(f"# WORKLOAD  d_model={cfg['d_model']}  steps={cfg['steps']}"
              f"  idle_baseline={r['idle_baseline_mw']:.1f}mW"
              f"  n={r['n_power_samples']}")
            for t, mw in r["power_samples"]:
                net = max(mw - r["idle_baseline_mw"], 0.0)
                w(f"  {t:.3f}  {mw:.1f}  {net:.1f}")
            w()

    print(f"\nReport written to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output .txt file path (default: calibration_results.txt)")
    args = parser.parse_args()

    if os.geteuid() == 0:
        print("ERROR: do not run as root. The INA3221 sysfs files are "
              "world-readable (perm=444) and CUDA is only accessible to "
              "the regular user. Run without sudo:")
        print("  python3 calibrate_power.py [--output FILE]")
        sys.exit(1)

    # Sanity checks
    volt_path, curr_path = find_ina3221_paths()
    if volt_path is None:
        print(f"ERROR: INA3221 label '{INA3221_LABEL}' not found under {INA3221_DRIVER}")
        sys.exit(1)
    print(f"INA3221 sensor : {INA3221_LABEL}")
    print(f"  volt_path    : {volt_path}")
    print(f"  curr_path    : {curr_path}")

    if not os.path.exists(WORKLOAD_SCRIPT):
        print(f"ERROR: workload script not found: {WORKLOAD_SCRIPT}")
        sys.exit(1)

    ts_proc = start_tegrastats(int(POLL_S * 1000))
    if ts_proc:
        print("tegrastats started for GPU/EMC utilization.")
    else:
        print("Warning: tegrastats unavailable — GPU/EMC columns will be empty.")

    # Verify sensor is readable
    test_mw = read_power_mw(volt_path, curr_path)
    if test_mw is None:
        print("ERROR: failed to read INA3221 — check that the hwmon sysfs path exists.")
        stop_tegrastats(ts_proc)
        sys.exit(1)
    print(f"Sensor test    : {test_mw:.1f} mW  OK")
    print()

    try:
        # ── Phase 0: idle baseline ────────────────────────────────────────
        print(f"Phase 0: {IDLE_BASELINE_S}s idle baseline"
              f" ({int(IDLE_BASELINE_S / POLL_S)} expected samples)")
        print("  Ensure no GPU workloads are running.", flush=True)
        idle_samples     = sample_idle(IDLE_BASELINE_S, volt_path, curr_path,
                                       ts_proc, "baseline")
        idle_mw_values   = [mw for _, mw in idle_samples]
        idle_baseline_mw = median(idle_mw_values)
        print(f"  Idle baseline set to {idle_baseline_mw:.1f} mW"
              f"  (n={len(idle_samples)}"
              f"  stdev={stdev(idle_mw_values) if len(idle_mw_values) > 1 else 0:.1f} mW)")

        run_results  = []
        all_quiet_mw = list(idle_mw_values)

        # ── Phases 1–N: workload runs ─────────────────────────────────────
        for i, config in enumerate(CONFIGS):
            print(f"\nRun {i+1}/{len(CONFIGS)}: d_model={config['d_model']}"
                  f"  steps={config['steps']}", flush=True)

            if i > 0:
                print(f"  {COOLDOWN_S}s cooldown before next run ...", flush=True)
                cool = sample_idle(COOLDOWN_S, volt_path, curr_path,
                                   ts_proc, "cooldown")
                all_quiet_mw.extend(mw for _, mw in cool)
                idle_baseline_mw = median(all_quiet_mw)
                print(f"  Updated idle baseline: {idle_baseline_mw:.1f} mW"
                      f"  (n={len(all_quiet_mw)} total quiet samples)")

            result = run_workload(config, idle_baseline_mw,
                                  volt_path, curr_path, ts_proc)
            run_results.append(result)

            if result["j_per_tflop"] is not None:
                print(f"\n  -> J/TFLOP : {result['j_per_tflop']:.4f}"
                      f"  |  avg net {result['avg_net_power_w']*1000:.0f} mW"
                      f"  |  GT {result['ground_truth_tf']:.4f} TFLOPs"
                      f"  |  {result['n_power_samples']} power samples", flush=True)
            else:
                print(f"\n  -> J/TFLOP : N/A  (GT missing or net energy zero)",
                      flush=True)

        # ── Write report ──────────────────────────────────────────────────
        print(f"\nWriting report to {args.output} ...", flush=True)
        write_report(idle_samples, run_results, idle_baseline_mw, args.output)

    finally:
        stop_tegrastats(ts_proc)


if __name__ == "__main__":
    main()
