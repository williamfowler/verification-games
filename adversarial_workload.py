"""
adversarial_workload.py — Red Team Deceptive Workload
=====================================================
Workloads deliberately OFF the benign FLOPs<->bytes line, to test whether the
blue team's EMC byte term (estimate_tflops_emc) catches a spoof that the
energy-only estimator cannot.

Modes:
  memory-spoof   : elementwise adds over buffers far larger than L2 — high DRAM
                   traffic and high GPU utilization, but near-zero FLOPs. A red
                   team would run this to make an energy-only monitor "see" a big
                   training run that never happened (burn energy, claim FLOPs are
                   below any treaty threshold — or vice versa, mask a real run's
                   profile). True FLOPs are the analytic elementwise count.
  compute-dense  : small cache-resident matmuls — high FLOPs with minimal DRAM
                   traffic. Control for the opposite direction: the byte term
                   must NOT wreck estimates for high-arithmetic-intensity work.

Emits the same stdout protocol as sample_ml_workload.py so the calibration
sampler (calibrate_power.run_workload) can drive it unchanged:
  "[redteam] Starting workload..."       — gates the power/bytes samplers
  "[redteam] Ground truth total : X TFLOPs"

Usage:
    python3 adversarial_workload.py --mode memory-spoof  --seconds 150
    python3 adversarial_workload.py --mode compute-dense --seconds 150
"""

import argparse
import time

import torch

try:
    from torch.utils.flop_counter import FlopCounterMode
    HAS_FLOP_COUNTER = True
except ImportError:
    HAS_FLOP_COUNTER = False


def counted_flops(fn):
    """FlopCounterMode's count for one call of fn (what a FLOP audit would see),
    or None if unavailable. Elementwise ops are counted as 0 by design — that is
    exactly the point of the memory-spoof mode."""
    if not HAS_FLOP_COUNTER:
        return None
    counter = FlopCounterMode(display=False)
    with counter:
        fn()
    return counter.get_total_flops()


def run(mode, seconds, buffer_mb, mm_size):
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found.")
    device = torch.device("cuda")
    print(f"[redteam] Device : {torch.cuda.get_device_name(0)}")
    print(f"[redteam] Config : mode={mode}, seconds={seconds},"
          f" buffer_mb={buffer_mb}, mm_size={mm_size}")

    if mode == "memory-spoof":
        n = buffer_mb * 1024 * 1024 // 4          # fp32 elements per buffer
        bufs = [torch.randn(n, device=device) for _ in range(2)]
        flops_per_iter = n                         # analytic: 1 add per element
        bytes_per_iter = 2 * 4 * n                 # read + write, 4B each
        step = lambda i: bufs[i % 2].add_(1.0)
        audit = counted_flops(lambda: bufs[0].add_(1.0))
    elif mode == "compute-dense":
        a = torch.randn(mm_size, mm_size, device=device)
        b = torch.randn(mm_size, mm_size, device=device)
        c = torch.empty(mm_size, mm_size, device=device)
        flops_per_iter = 2 * mm_size ** 3          # analytic matmul FLOPs
        bytes_per_iter = 0                         # cache-resident by design
        step = lambda i: torch.mm(a, b, out=c)
        audit = counted_flops(lambda: torch.mm(a, b, out=c))
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # Warmup: bring CUDA fully online before the sampler window opens.
    step(0)
    torch.cuda.synchronize()

    print(f"[redteam] Analytic FLOPs/iter    : {flops_per_iter / 1e9:.4f} GFLOPs"
          f"   (FlopCounterMode sees: "
          f"{'N/A' if audit is None else f'{audit / 1e9:.4f} GFLOPs'})")

    print("[redteam] Starting workload...\n", flush=True)
    t_start = time.monotonic()
    iters = 0
    next_report = 10.0
    while True:
        step(iters)
        iters += 1
        if iters % 50 == 0:
            torch.cuda.synchronize()
            elapsed = time.monotonic() - t_start
            if elapsed >= next_report:
                print(f"  {elapsed:6.1f}s  |  {iters} iters"
                      f"  |  {iters / elapsed:.1f} iters/s", flush=True)
                next_report += 10.0
            if elapsed >= seconds:
                break
    torch.cuda.synchronize()
    total_time = time.monotonic() - t_start

    total_tf = flops_per_iter * iters / 1e12
    total_tb = bytes_per_iter * iters / 1e12
    print(f"\n[redteam] Done. {iters} iters in {total_time:.1f}s")
    print(f"[redteam] Analytic DRAM traffic  : {total_tb:.4f} TB"
          f"  ({total_tb * 1e3 / total_time:.2f} GB/s)")
    print(f"[redteam] Ground truth total : {total_tf:.4f} TFLOPs")
    print(f"[redteam] Ground truth rate  : {total_tf / total_time:.4f} TFLOPS avg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["memory-spoof", "compute-dense"])
    parser.add_argument("--seconds",   type=float, default=150.0)
    parser.add_argument("--buffer-mb", type=int,   default=64,
                        help="memory-spoof buffer size (each of 2, fp32)")
    parser.add_argument("--mm-size",   type=int,   default=512,
                        help="compute-dense square matmul dimension")
    args = parser.parse_args()
    run(args.mode, args.seconds, args.buffer_mb, args.mm_size)
