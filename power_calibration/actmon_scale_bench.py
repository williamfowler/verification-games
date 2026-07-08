#!/usr/bin/env python3
"""
actmon_scale_bench.py — Calibrate the actmon byte scale against known traffic.

The actmon mc_all counter is linear in DRAM activity but its absolute scale is
unknown (it historically read ~1% where tegrastats claimed ~38%). That is
harmless for the EMC estimator — the scale constant is absorbed into the fitted
E_PER_TB_J — but it leaves E_PER_TB_J physically uninterpretable and
non-portable. This benchmark measures the scale factor directly:

    k = (actmon-integrated bytes − idle background) / analytic bytes moved

by streaming a known byte volume through DRAM (elementwise adds over buffers
far larger than L2; analytic traffic = read + write per element per pass) while
the same BytesSampler used by the calibration sweep integrates the counter.
E_PER_TB_J / k is then the physical J/TB, comparable to LPDDR5 expectations
(~O(100) J/TB). Analytic traffic carries an inherent systematic (write-allocate
/ streaming-store behavior can shift bytes/element by ~1.5x); treat k as an
order-of-magnitude anchor, not a precision constant.

Run on the Jetson, NOT as root (BytesSampler elevates via the sudoers actmon
reader), no other GPU load. ~2 min.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import detect_flops
from calibrate_power import BytesSampler


def sample_window(seconds, work=None):
    """Integrate actmon bytes over a window, optionally running `work` (a
    callable executed repeatedly) during it. Returns (actmon_tb, duration_s,
    iters)."""
    sampler = BytesSampler()
    sampler.start()
    t0 = time.monotonic()
    iters = 0
    if work is None:
        time.sleep(seconds)
    else:
        while time.monotonic() - t0 < seconds:
            work(iters)
            iters += 1
            if iters % 50 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
    duration = time.monotonic() - t0
    sampler.stop()
    sampler.require_samples()
    tb = sampler.total_tb()
    if tb is None:
        raise RuntimeError("too few actmon samples in window")
    return tb, duration, iters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="traffic window duration (default 60)")
    parser.add_argument("--idle-seconds", type=float, default=30.0,
                        help="idle background window (default 30)")
    parser.add_argument("--buffer-mb", type=int, default=64,
                        help="each of 2 fp32 buffers (default 64 MB)")
    args = parser.parse_args()

    if os.geteuid() == 0:
        print("ERROR: do not run as root (CUDA needs the user venv).")
        sys.exit(1)
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found.")

    n = args.buffer_mb * 1024 * 1024 // 4
    bufs = [torch.randn(n, device="cuda") for _ in range(2)]
    bufs[0].add_(1.0)
    torch.cuda.synchronize()
    bytes_per_iter = 2 * 4 * n          # read + write, fp32, per pass

    print(f"Idle background window ({args.idle_seconds:.0f}s, no GPU work)...",
          flush=True)
    idle_tb, idle_t, _ = sample_window(args.idle_seconds)
    idle_rate_tb_s = idle_tb / idle_t
    print(f"  idle actmon traffic : {idle_rate_tb_s * 1e3:.4f} GB/s"
          f"  ({idle_tb:.5f} TB over {idle_t:.1f}s)")

    print(f"Traffic window ({args.seconds:.0f}s, elementwise adds over"
          f" 2x{args.buffer_mb} MB)...", flush=True)
    load_tb, load_t, iters = sample_window(
        args.seconds, work=lambda i: bufs[i % 2].add_(1.0))
    analytic_tb = bytes_per_iter * iters / 1e12
    net_actmon_tb = load_tb - idle_rate_tb_s * load_t

    k = net_actmon_tb / analytic_tb
    # E_PER_TB_J is J per ACTMON-TB; TB_actmon = k * TB_true, so the physical
    # coefficient is J per true TB = E_PER_TB_J * k.
    e_per_tb_phys = detect_flops.E_PER_TB_J * k

    print()
    print("=" * 70)
    print(f"  iters               : {iters}  ({iters / load_t:.1f}/s)")
    print(f"  analytic traffic    : {analytic_tb:.4f} TB"
          f"  ({analytic_tb * 1e3 / load_t:.2f} GB/s)")
    print(f"  actmon (raw)        : {load_tb:.4f} TB")
    print(f"  actmon (net of idle): {net_actmon_tb:.4f} TB")
    print(f"  SCALE FACTOR k      : {k:.4f}   (actmon TB per true TB)")
    print(f"  E_PER_TB_J * k      : {e_per_tb_phys:.1f} J per true TB")
    print(f"    (LPDDR5 ballpark ~50-150 J/TB; shipped E_PER_TB_J"
          f" = {detect_flops.E_PER_TB_J} J per actmon-TB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
