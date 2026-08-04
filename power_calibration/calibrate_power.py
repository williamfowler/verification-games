#!/usr/bin/env python3
"""
calibrate_power.py — sampling library for the calibration/eval tooling (DDP regime).

Provides the shared measurement plumbing for the fp16-AMP DDP experiment on
2× V100:
  - PowerSampler  : per-GPU nvidia-smi power + util; power SUMMED across both GPUs
                    for the energy integral, util kept per-GPU (both-GPU frontier gate).
  - BytesSampler  : per-GPU DRAM-active (DCGM field 1005) → DRAM bytes moved (summed).
  - NvlinkSampler : per-GPU NVLink/PCIe interconnect bytes (Task 1; imported).
  - run_workload  : launches the workload under torchrun --nproc_per_node=2, samples
                    all three on the same window, integrates net energy vs the
                    aggregate FlopCounterMode ground truth.

DDP-only: this branch launches every workload via torchrun on BOTH GPUs (no
single-GPU pinning). Every record is tagged mode="fp16_ddp"/precision so it never
pools with the legacy fp32 single-GPU porting-era fit. DCGM pattern: root
nv-hostengine, unprivileged dcgmi clients.

Entry point for calibration/accuracy is eval_power_monitor.py.
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

REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKLOAD_SCRIPT = os.path.join(REPO_ROOT, "sample_ml_workload.py")

# Both V100s. With CUDA_DEVICE_ORDER=PCI_BUS_ID the nvidia-smi/DCGM index matches
# the torchrun local_rank, so per-GPU samples attribute to the right rank.
DDP_GPUS = (0, 1)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_flops import (  # noqa: E402
    DRAM_ACTIVE_FIELD, V100_PROFILE, actmon_bytes_per_s, read_power_mw,
)
from nvlink_monitor import NvlinkSampler  # noqa: E402


# ── Python / torchrun interpreters ────────────────────────────────────────────

def _venv_bin(name):
    for cand in (name, name + "3"):
        p = os.path.join(REPO_ROOT, ".venv", "bin", cand)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def find_venv_python():
    return _venv_bin("python") or sys.executable


def find_venv_torchrun():
    tr = _venv_bin("torchrun")
    if tr is None:
        raise RuntimeError("torchrun not found in .venv/bin — is torch installed in the venv?")
    return tr


def child_env():
    """Environment for the torchrun workload. Both GPUs must be visible (DDP), so
    unlike the single-GPU porting era we do NOT pin CUDA_VISIBLE_DEVICES.
    PYTHONUNBUFFERED=1 makes the child's stdout line-buffered so the
    '[redteam] Starting workload' trigger reaches the sampler promptly (torchrun
    has no -u passthrough)."""
    env = dict(os.environ)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PYTHONUNBUFFERED"] = "1"
    return env


# ── nvidia-smi dual-GPU read ──────────────────────────────────────────────────

def read_gpu_samples(gpu_indices=DDP_GPUS):
    """One nvidia-smi call for all GPUs → list of (power_mw, util_pct, sm_hz),
    one tuple per index in gpu_indices order. Raises on read error."""
    gpu_arg = ",".join(str(g) for g in gpu_indices)
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,clocks.sm",
         "--format=csv,noheader,nounits", "-i", gpu_arg],
        capture_output=True, text=True, check=True).stdout.strip()
    rows = [ln for ln in out.splitlines() if ln.strip()]
    res = []
    for ln in rows:
        p, u, c = [v.strip() for v in ln.split(",")]
        res.append((float(p) * 1000.0, float(u), float(c) * 1e6))
    return res


# ── Background power sampler (both GPUs) ──────────────────────────────────────

class PowerSampler:
    """nvidia-smi power + util at POLL_S, over both GPUs. power_samples holds the
    SUMMED board power (energy integral); util is kept per-GPU for the both-GPU
    frontier gate."""

    def __init__(self, gpu_indices=DDP_GPUS):
        self.gpu_indices   = tuple(gpu_indices)
        self.power_samples = []                       # (t, total_power_mw)
        self.util_per      = {g: [] for g in self.gpu_indices}   # g -> [(t, util)]
        self._stop         = threading.Event()
        self._exc          = None
        self._thread       = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        if self._exc is not None:
            raise self._exc

    def _run(self):
        try:
            while not self._stop.is_set():
                t = time.monotonic()
                rows = read_gpu_samples(self.gpu_indices)
                self.power_samples.append((t, sum(r[0] for r in rows)))
                for g, r in zip(self.gpu_indices, rows):
                    self.util_per[g].append((t, r[1]))
                self._stop.wait(POLL_S)
        except BaseException as e:
            self._exc = e

    def avg_util_per_gpu(self):
        return {g: (mean(u for _, u in self.util_per[g]) if self.util_per[g] else None)
                for g in self.gpu_indices}


# ── DRAM bytes sampler (DCGM field 1005, both GPUs) ───────────────────────────

def _parse_dram_row(line):
    """`dcgmi dmon -e 1005 -i 0,1` data line → (gpu_idx, fraction) or None."""
    s = line.strip()
    if not s or s.startswith("#") or not s.startswith("GPU"):
        return None
    toks = s.split()
    if len(toks) < 3:
        return None
    try:
        return int(toks[1]), float(toks[-1])
    except ValueError:
        return None


class BytesSampler:
    """Per-GPU DRAM-active fraction (DCGM field 1005) → DRAM bytes/s, summed across
    both GPUs for TB_moved. Requires nv-hostengine (root); dcgmi client unprivileged."""

    def __init__(self, gpu_indices=DDP_GPUS):
        self.gpu_indices = tuple(gpu_indices)
        self.samples = {g: [] for g in self.gpu_indices}   # g -> [(t, bytes_per_s)]
        self.available = False
        self._stop = threading.Event()
        self._exc = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        gpu_arg = ",".join(str(g) for g in self.gpu_indices)
        self.proc = subprocess.Popen(
            ["dcgmi", "dmon", "-e", str(DRAM_ACTIVE_FIELD), "-i", gpu_arg,
             "-d", str(int(POLL_S * 1000))],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )

    def start(self):
        self._thread.start()

    def stop(self):
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
                parsed = _parse_dram_row(line)
                if parsed is None:
                    continue
                idx, frac = parsed
                if idx not in self.samples:
                    continue
                frac = max(0.0, min(1.0, frac))
                self.available = True
                self.samples[idx].append((time.monotonic(), actmon_bytes_per_s(frac)))
        except BaseException as e:
            if not self._stop.is_set():
                self._exc = e

    def require_samples(self):
        if not self.available:
            raise RuntimeError(
                "DCGM DRAM-active (field 1005) produced no samples — is "
                "nv-hostengine running (sudo nv-hostengine)?")

    @staticmethod
    def _integ(samples):
        if len(samples) < 2:
            return 0.0
        tot = 0.0
        for i in range(1, len(samples)):
            t0, b0 = samples[i - 1]; t1, b1 = samples[i]
            tot += 0.5 * (b0 + b1) * (t1 - t0)
        return tot

    def total_tb(self):
        """DRAM bytes moved summed across both GPUs → terabytes."""
        tot = sum(self._integ(self.samples[g]) for g in self.gpu_indices)
        return tot / 1e12 if any(len(self.samples[g]) >= 2 for g in self.gpu_indices) else None

    def avg_bytes_per_s(self):
        allb = [b for g in self.gpu_indices for _, b in self.samples[g]]
        return mean(allb) if allb else None

    def avg_dram_pct(self):
        allb = [b for g in self.gpu_indices for _, b in self.samples[g]]
        if not allb:
            return None
        # mean per-GPU fraction as a percent
        return mean(allb) / V100_PROFILE["PEAK_BW_BYTES_S"] * 100.0


# ── Idle sampling ─────────────────────────────────────────────────────────────

def sample_idle(duration_s, gpu_indices, label):
    """Blocking idle sample of SUMMED both-GPU power; returns [(mono_t, total_mw)]."""
    print(f"  [{label}] sampling idle for {duration_s:.0f}s ...", flush=True)
    sampler = PowerSampler(gpu_indices)
    sampler.start()
    time.sleep(duration_s)
    sampler.stop()
    n = len(sampler.power_samples)
    if n:
        vals = [mw for _, mw in sampler.power_samples]
        print(f"  [{label}] {n} samples  |  median {median(vals):.1f} mW"
              f"  stdev {stdev(vals) if n > 1 else 0:.1f} mW (both GPUs)", flush=True)
    return sampler.power_samples


# ── Workload execution ────────────────────────────────────────────────────────

def parse_ground_truth_tflops(text):
    m = re.search(r'Ground truth total\s*:\s*([\d.]+)\s*TFLOPs', text)
    return float(m.group(1)) if m else None


def parse_n_params(text):
    m = re.search(r'\[redteam\] Params\s*:\s*(\d+)', text)
    return int(m.group(1)) if m else None


def compute_net_energy(power_samples, idle_baseline_mw):
    """Trapezoidal integration of net (summed) power over the sample timeseries."""
    net_energy_j = 0.0
    net_mw_list  = []
    for i in range(1, len(power_samples)):
        t0, mw0 = power_samples[i - 1]
        t1, mw1 = power_samples[i]
        dt      = t1 - t0
        n0      = max(mw0 - idle_baseline_mw, 0.0) / 1000.0
        n1      = max(mw1 - idle_baseline_mw, 0.0) / 1000.0
        net_energy_j += 0.5 * (n0 + n1) * dt
        net_mw_list.append(0.5 * (n0 + n1) * 1000.0)
    avg_net_mw = mean(net_mw_list) if net_mw_list else None
    return net_energy_j, avg_net_mw


def _torchrun_cmd(config, gpu_indices):
    nproc = len(gpu_indices)
    # Workload-override hook (Phase II red team): a config may point at a different
    # entry script and append strategy-specific flags. The core shape args
    # (steps/batch/seq/d_model + optional geometry) are still injected so the
    # step-rate auto-sizer keeps working (it only rewrites config["steps"]); the
    # extra "args" carry things the benign flag set doesn't know about
    # (--strategy, --decoy-gbps, --gap-seconds, ...). Default = benign path.
    script = config.get("script", WORKLOAD_SCRIPT)
    cmd = [find_venv_torchrun(), "--standalone", f"--nproc_per_node={nproc}",
           script,
           "--steps",      str(config["steps"]),
           "--batch-size", str(config["batch_size"]),
           "--seq-len",    str(config["seq_len"]),
           "--d-model",    str(config["d_model"])]
    for key, flag in (("num_layers", "--num-layers"), ("nhead", "--nhead"),
                      ("dim_feedforward", "--dim-feedforward"),
                      ("precision", "--precision"), ("optimizer", "--optimizer")):
        if key in config:
            cmd += [flag, str(config[key])]
    cmd += [str(a) for a in config.get("args", [])]
    return cmd


def measure_step_rate(config, gpu_indices=DDP_GPUS, probe_steps=30):
    """Auto-size helper: run a short torchrun pass and return steps/s (or None).
    Used to size the real run to ~150s active (steps = round(150 x rate))."""
    probe = dict(config); probe["steps"] = probe_steps
    cmd = _torchrun_cmd(probe, gpu_indices)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=child_env())
    out = proc.communicate()[0]
    if proc.returncode != 0:
        return None, out
    m = re.search(r'Done\.\s*\d+\s*steps in [\d.]+s \(([\d.]+) steps/s\)', out)
    return (float(m.group(1)) if m else None), out


def run_workload(config, idle_baseline_mw, gpu_indices=DDP_GPUS):
    """Launch the DDP workload under torchrun on both GPUs; sample summed power,
    per-GPU util, DRAM bytes, and NVLink/PCIe bytes on the same window; integrate
    net energy vs the aggregate FlopCounterMode ground truth. Returns a record."""
    cmd = _torchrun_cmd(config, gpu_indices)
    print(f"  Launching: torchrun x{len(gpu_indices)}  {' '.join(cmd[4:])}", flush=True)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=child_env())

    sampler = bytes_sampler = nvlink_sampler = None
    t_start = None
    output_lines = []
    oom = False
    for line in iter(proc.stdout.readline, ""):
        print(line, end="", flush=True)
        output_lines.append(line)
        if "OutOfMemoryError" in line or "CUDA out of memory" in line:
            oom = True
        if sampler is None and "[redteam] Starting workload" in line:
            t_start = time.monotonic()
            sampler = PowerSampler(gpu_indices);        sampler.start()
            bytes_sampler = BytesSampler(gpu_indices);  bytes_sampler.start()
            nvlink_sampler = NvlinkSampler(gpu_indices); nvlink_sampler.start()
    proc.wait()
    stdout_text = "".join(output_lines)

    if sampler is None:
        # No sampling window opened. OOM or another launch failure — return a
        # returncode!=0 record so the sweep logs it and continues (not a crash).
        note = "OOM" if oom else f"no-start (exit {proc.returncode})"
        print(f"  WARNING: workload never started sampling ({note})", flush=True)
        return {"config": config, "returncode": proc.returncode or 1,
                "error": note, "ddp": True, "world_size": len(gpu_indices),
                "precision": config.get("precision", "fp16"), "mode": "fp16_ddp",
                "ground_truth_tf": None, "net_energy_j": 0.0, "duration_s": 0.0,
                "avg_gpu_pct": None, "tb_moved": None}

    sampler.stop()
    duration_s = time.monotonic() - t_start
    bytes_sampler.stop();  bytes_sampler.require_samples()
    nvlink_sampler.stop(); nvlink_sampler.require_samples()

    tb_moved        = bytes_sampler.total_tb()
    avg_bytes_per_s = bytes_sampler.avg_bytes_per_s()
    avg_dram_pct    = bytes_sampler.avg_dram_pct()
    gt_tflops       = parse_ground_truth_tflops(stdout_text)
    n_params        = parse_n_params(stdout_text)
    util_per        = sampler.avg_util_per_gpu()
    nvl             = nvlink_sampler.record_fields()

    if proc.returncode != 0:
        print(f"  WARNING: workload exited with code {proc.returncode}", flush=True)

    net_energy_j, avg_net_mw = compute_net_energy(sampler.power_samples, idle_baseline_mw)
    avg_net_w   = avg_net_mw / 1000.0 if avg_net_mw is not None else None
    j_per_tflop = (net_energy_j / gt_tflops
                   if gt_tflops and gt_tflops > 0 and net_energy_j > 0 else None)
    raw_mw = [mw for _, mw in sampler.power_samples]

    # both-GPU frontier gate: min per-GPU util (min>=80  <=>  both>=80)
    util_vals = [u for u in util_per.values() if u is not None]
    avg_gpu_min = min(util_vals) if util_vals else None
    steps = config["steps"]

    rec = {
        "config":           config,
        "ddp":              True,
        "world_size":       len(gpu_indices),
        "precision":        config.get("precision", "fp16"),
        "mode":             "fp16_ddp",
        "returncode":       proc.returncode,
        "duration_s":       duration_s,
        "idle_baseline_mw": idle_baseline_mw,
        "power_samples":    sampler.power_samples,
        "ground_truth_tf":  gt_tflops,
        "n_params":         n_params,
        "net_energy_j":     net_energy_j,
        "avg_net_power_w":  avg_net_w,
        "j_per_tflop":      j_per_tflop,
        "avg_raw_mw":       mean(raw_mw) if raw_mw else None,
        "peak_raw_mw":      max(raw_mw)  if raw_mw else None,
        "avg_gpu_pct":      avg_gpu_min,            # min across GPUs (both-GPU gate)
        "avg_gpu_pct_gpu0": util_per.get(gpu_indices[0]),
        "avg_gpu_pct_gpu1": util_per.get(gpu_indices[1]),
        "avg_emc_pct":      avg_dram_pct,
        "tb_moved":         tb_moved,
        "avg_bytes_per_s":  avg_bytes_per_s,
        "n_power_samples":  len(sampler.power_samples),
        # Task 1 interconnect signal
        "nvlink_total_bytes":    nvl["nvlink_total_bytes"],
        "nvlink_bytes_per_step": nvl["nvlink_total_bytes"] / steps if steps else None,
        "pcie_total_bytes":      sum(nvl.get(f"gpu{g}_pcie_tx_bytes", 0.0)
                                     + nvl.get(f"gpu{g}_pcie_rx_bytes", 0.0)
                                     for g in gpu_indices),
        "grad_bytes_pred":       (n_params * 4) if n_params else None,
    }
    return rec
