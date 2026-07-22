#!/usr/bin/env python3
"""
nvlink_monitor.py — per-GPU interconnect byte counting (Task 1 blue-team signal).

Reports, per V100, bytes/s transmitted (egress) and received (ingress) on the
GPU interconnect — NVLink (6 links/card, primary) and PCIe (host↔GPU) — at
~1-2 Hz, in the style of the repo's other blue-team samplers (BytesSampler).
This is the blue team's view of the DDP gradient all-reduce, the observable that
makes distributed training distinctive.

WHICH COUNTER API WORKS ON THIS BOX (driver 580 + V100, empirically checked):
  - Option A  NVML NVLINK_THROUGHPUT_DATA_TX/RX (fields 138/139) → NOT_SUPPORTED
  - Option B  NVML legacy nvmlDeviceGetNvLinkUtilizationCounter  → NOT_SUPPORTED
  - Option C  nvidia-smi nvlink -gt d                            → "Data Tx: N/A"
  - Option D  DCGM DCGM_FI_PROF_NVLINK_TX_BYTES / _RX_BYTES (1011/1012) and
              PCIE_TX/RX_BYTES (1009/1010) via `dcgmi dmon`       → WORKS ✓
So this monitor is built on Option D (DCGM profiling), the same nv-hostengine
path the DRAM-active BytesSampler (field 1005) already uses: root nv-hostengine,
unprivileged dcgmi client. The counters are RATES (bytes/s), so there is no
cumulative-counter wraparound to unwind; cumulative bytes are recovered by
trapezoidal integration of the rate. Garbage/negative samples are flagged and
dropped rather than crashing (see _parse_dcgm_row).

Standalone:
    python3 nvlink_monitor.py --seconds 10                 # sample + report
    python3 nvlink_monitor.py --selftest                   # validations 1 & 2
As a library: NvlinkSampler(gpu_indices=(0,1)) — start()/stop() on the same
window as PowerSampler/BytesSampler so its bytes are commensurable per record.
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from statistics import mean

# DCGM profiling field ids (bytes/s). Order here defines the dcgmi -e column order.
FIELD_PCIE_TX   = 1009
FIELD_PCIE_RX   = 1010
FIELD_NVLINK_TX = 1011
FIELD_NVLINK_RX = 1012
FIELDS = [FIELD_PCIE_TX, FIELD_PCIE_RX, FIELD_NVLINK_TX, FIELD_NVLINK_RX]

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def _parse_dcgm_row(line):
    """Parse one `dcgmi dmon -e 1009,1010,1011,1012` data line.
    Returns (gpu_index, [pcie_tx, pcie_rx, nvlink_tx, nvlink_rx]) in bytes/s, or
    None for headers/blank/short/garbage lines. Values must be finite and
    non-negative (a negative rate is a bad sample → flagged by returning None)."""
    s = line.strip()
    if not s or s.startswith("#") or not s.startswith("GPU"):
        return None
    toks = s.split()
    # "GPU <idx> v1 v2 v3 v4"
    if len(toks) < 2 + len(FIELDS):
        return None
    try:
        idx = int(toks[1])
        vals = [float(t) for t in toks[2:2 + len(FIELDS)]]
    except ValueError:
        return None
    if any(v < 0 for v in vals):
        return None
    return idx, vals


class NvlinkSampler:
    """
    Streams per-GPU PCIe + NVLink byte rates from `dcgmi dmon -e 1009,1010,1011,1012`
    for the given GPUs. Mirrors BytesSampler's lifecycle so it can run on the same
    window as the power/DRAM samplers. Requires a running nv-hostengine (root);
    the dcgmi client itself is unprivileged.

    self.samples[gpu] = list of (mono_t, pcie_tx, pcie_rx, nvlink_tx, nvlink_rx)  [bytes/s]
    """

    def __init__(self, gpu_indices=(0, 1), interval_ms=500):
        self.gpu_indices = tuple(gpu_indices)
        self.samples = {g: [] for g in self.gpu_indices}
        self.flagged = 0            # count of dropped garbage/negative rows
        self.available = False
        self._stop = threading.Event()
        self._exc = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        gpu_arg = ",".join(str(g) for g in self.gpu_indices)
        field_arg = ",".join(str(f) for f in FIELDS)
        self.proc = subprocess.Popen(
            ["dcgmi", "dmon", "-e", field_arg, "-i", gpu_arg,
             "-d", str(int(interval_ms))],
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
                parsed = _parse_dcgm_row(line)
                if parsed is None:
                    if line.strip().startswith("GPU"):
                        self.flagged += 1
                    continue
                idx, vals = parsed
                if idx not in self.samples:
                    continue
                self.available = True
                self.samples[idx].append((time.monotonic(), *vals))
        except BaseException as e:
            if not self._stop.is_set():
                self._exc = e

    def require_samples(self):
        if not self.available:
            raise RuntimeError(
                "DCGM NVLink/PCIe (fields 1009-1012) produced no samples — is "
                "nv-hostengine running (sudo nv-hostengine) and do "
                "`dcgmi dmon -e 1011,1012` work on these GPUs?")

    @staticmethod
    def _integrate(samples, col):
        """Trapezoidal integral of one rate column (index 1..4) over time → bytes."""
        if len(samples) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(samples)):
            t0 = samples[i - 1][0]; t1 = samples[i][0]
            r0 = samples[i - 1][col]; r1 = samples[i][col]
            total += 0.5 * (r0 + r1) * (t1 - t0)
        return total

    def totals(self):
        """Per-GPU integrated bytes over the sampled window:
        {gpu: {pcie_tx, pcie_rx, nvlink_tx, nvlink_rx}} in bytes."""
        cols = {"pcie_tx": 1, "pcie_rx": 2, "nvlink_tx": 3, "nvlink_rx": 4}
        return {g: {k: self._integrate(self.samples[g], c) for k, c in cols.items()}
                for g in self.gpu_indices}

    def avg_rates(self):
        """Per-GPU mean byte rates: {gpu: {pcie_tx, pcie_rx, nvlink_tx, nvlink_rx}} bytes/s."""
        cols = {"pcie_tx": 1, "pcie_rx": 2, "nvlink_tx": 3, "nvlink_rx": 4}
        out = {}
        for g in self.gpu_indices:
            s = self.samples[g]
            out[g] = {k: (mean(row[c] for row in s) if s else 0.0)
                      for k, c in cols.items()}
        return out

    def record_fields(self):
        """Flat, JSON-friendly summary for a sweep record: per-GPU integrated
        NVLink/PCIe bytes + the two-GPU NVLink total (the all-reduce volume)."""
        tot = self.totals()
        rec = {"nvlink_flagged_samples": self.flagged}
        nvlink_sum = 0.0
        for g in self.gpu_indices:
            for k, v in tot[g].items():
                rec[f"gpu{g}_{k}_bytes"] = v
            nvlink_sum += tot[g]["nvlink_tx"] + tot[g]["nvlink_rx"]
        rec["nvlink_total_bytes"] = nvlink_sum
        return rec


# ── Standalone report / validations ──────────────────────────────────────────

def _report(sampler, window_s):
    tot = sampler.totals()
    avg = sampler.avg_rates()
    print(f"\nSampled {window_s:.1f}s  |  flagged/dropped rows: {sampler.flagged}")
    for g in sampler.gpu_indices:
        a = avg[g]; t = tot[g]
        print(f"  GPU{g}: NVLink TX {a['nvlink_tx']/1e9:6.2f} GB/s  RX {a['nvlink_rx']/1e9:6.2f} GB/s"
              f"  | PCIe TX {a['pcie_tx']/1e9:5.2f} RX {a['pcie_rx']/1e9:5.2f} GB/s"
              f"  | NVLink Σ {(t['nvlink_tx']+t['nvlink_rx'])/1e9:.2f} GB")


def selftest(gpu_indices):
    """Validations 1 (known-size transfer) & 2 (symmetry GPU0 TX ≈ GPU1 RX)."""
    import torch
    if len(gpu_indices) < 2:
        raise SystemExit("selftest needs 2 GPUs")
    g0, g1 = gpu_indices[0], gpu_indices[1]
    n_floats = 64 * 1024 * 1024          # 256 MB fp32 per transfer
    xfer_bytes = n_floats * 4
    n_xfers = 200
    expected = xfer_bytes * n_xfers      # GPU0→GPU1 only
    x = torch.ones(n_floats, device=f"cuda:{g0}"); torch.cuda.synchronize(g0)

    sampler = NvlinkSampler(gpu_indices)
    sampler.start()
    time.sleep(1.0)                       # let the stream warm up
    t0 = time.monotonic()
    for _ in range(n_xfers):
        y = x.to(f"cuda:{g1}"); torch.cuda.synchronize(g1)
    dur = time.monotonic() - t0
    time.sleep(1.0)
    sampler.stop()
    sampler.require_samples()

    tot = sampler.totals()
    tx0 = tot[g0]["nvlink_tx"]; rx1 = tot[g1]["nvlink_rx"]
    rx0 = tot[g0]["nvlink_rx"]; tx1 = tot[g1]["nvlink_tx"]
    print(f"\n=== VALIDATION 1: known-size transfer ({n_xfers}×{xfer_bytes/1e6:.0f} MB, GPU{g0}→GPU{g1}) ===")
    print(f"  expected NVLink egress : {expected/1e9:7.2f} GB  over {dur:.1f}s")
    print(f"  GPU{g0} TX (egress)      : {tx0/1e9:7.2f} GB   (ratio {tx0/expected:.2f})")
    print(f"  GPU{g1} RX (ingress)     : {rx1/1e9:7.2f} GB   (ratio {rx1/expected:.2f})")
    print(f"=== VALIDATION 2: symmetry (GPU{g0} TX ≈ GPU{g1} RX, GPU{g0} RX ≈ GPU{g1} TX) ===")
    def sym(a, b, label):
        r = a / b if b else float('inf')
        ok = 0.8 <= r <= 1.25 if b else False
        print(f"  {label}: {a/1e9:.2f} vs {b/1e9:.2f} GB  ratio {r:.2f}  [{'OK' if ok else 'CHECK'}]")
        return ok
    ok1 = sym(tx0, rx1, f"GPU{g0}.TX vs GPU{g1}.RX")
    # reverse direction is ~0 here (one-way transfer); assert the forward symmetry
    within = 0.7 <= (tx0 / expected) <= 1.3 and 0.7 <= (rx1 / expected) <= 1.3
    print(f"\n  known-size within ±30%: {'PASS' if within else 'FAIL'}   symmetry: {'PASS' if ok1 else 'FAIL'}")
    return within and ok1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--interval-ms", type=int, default=500)
    ap.add_argument("--gpus", default="0,1", help="comma-separated GPU indices")
    ap.add_argument("--selftest", action="store_true",
                    help="run validations 1 (known-size) & 2 (symmetry)")
    args = ap.parse_args()
    gpus = tuple(int(g) for g in args.gpus.split(","))

    if args.selftest:
        ok = selftest(gpus)
        sys.exit(0 if ok else 1)

    sampler = NvlinkSampler(gpus, args.interval_ms)
    sampler.start()
    time.sleep(args.seconds)
    sampler.stop()
    sampler.require_samples()
    _report(sampler, args.seconds)


if __name__ == "__main__":
    main()
