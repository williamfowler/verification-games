#!/usr/bin/env python3
"""
drift_test.py — Does the multi-minute calibration hold over a sustained run?

Every calibration workload lasts 2-4 minutes, but the actual threat — an
unauthorized frontier training run — occupies the device for hours, long enough
for thermal throttling and baseline drift to move the J/TFLOP operating point.
This runs ONE frontier config scaled to ~50 minutes through the production
sampling path, scores it with the SHIPPED constants, and reports the per-window
power trend plus each window's implied J/TFLOP (assuming the workload's FLOP
rate is uniform, which for a fixed-config training loop it is, up to
throttling — which is exactly what the trend exposes).

Run on the Jetson, NOT as root, via .venv/bin/python3, no other GPU load.
~55 min total.
"""

import argparse
import os
import sys
import time
from statistics import median, stdev

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import detect_flops
from calibrate_power import (
    find_ina3221_paths,
    read_power_mw,
    start_tegrastats,
    stop_tegrastats,
    sample_idle,
    run_workload,
    POLL_S,
)

# d512_b16_s128_L6 ran 1200 steps in ~198s (~6.1 steps/s) in the 2026-06-30
# sweep at 98% util — 18000 steps is ~50 min of the same operating point.
CONFIG = {"d_model": 512, "batch_size": 16, "seq_len": 128, "num_layers": 6,
          "steps": 18000}
WINDOW_S = 300.0
DEFAULT_OUTPUT = "drift_results.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=CONFIG["steps"])
    parser.add_argument("--baseline-seconds", type=int, default=60)
    args = parser.parse_args()

    if os.geteuid() == 0:
        print("ERROR: do not run as root.")
        sys.exit(1)

    volt_path, curr_path = find_ina3221_paths()
    if volt_path is None:
        print("ERROR: INA3221 sensor not found.")
        sys.exit(1)
    read_power_mw(volt_path, curr_path)
    ts_proc = start_tegrastats(int(POLL_S * 1000))

    try:
        print(f"Idle baseline ({args.baseline_seconds}s)...", flush=True)
        idle = sample_idle(args.baseline_seconds, volt_path, curr_path,
                           ts_proc, "baseline")
        idle_mw = [mw for _, mw in idle]
        if not idle_mw:
            raise RuntimeError("no idle samples — cannot baseline")
        baseline_mw = median(idle_mw)
        print(f"Idle baseline: {baseline_mw:.1f} mW", flush=True)

        cfg = dict(CONFIG, steps=args.steps)
        r = run_workload(cfg, baseline_mw, volt_path, curr_path, ts_proc)
        if r["returncode"] != 0:
            raise RuntimeError(f"workload exited {r['returncode']}")
    finally:
        stop_tegrastats(ts_proc)

    gt = r["ground_truth_tf"]
    est2 = detect_flops.estimate_tflops(r["net_energy_j"], r["duration_s"])
    est3 = detect_flops.estimate_tflops_emc(r["net_energy_j"], r["duration_s"],
                                            r["tb_moved"])

    # Per-window analysis over the raw power series. Uniform FLOP rate assumed:
    # window_tflops = gt * (window_time / duration).
    samples = r["power_samples"]
    t0 = samples[0][0]
    gt_rate = gt / r["duration_s"]          # TFLOPs per second
    windows = []
    w_idx, i = 0, 1
    while i < len(samples):
        w_start = t0 + w_idx * WINDOW_S
        w_end = w_start + WINDOW_S
        net_j, seconds, raw = 0.0, 0.0, []
        while i < len(samples) and samples[i][0] <= w_end:
            (ta, mwa), (tb, mwb) = samples[i - 1], samples[i]
            dt = tb - ta
            na = max(mwa - baseline_mw, 0.0) / 1000.0
            nb = max(mwb - baseline_mw, 0.0) / 1000.0
            net_j += 0.5 * (na + nb) * dt
            seconds += dt
            raw.append(mwb)
            i += 1
        if seconds < WINDOW_S * 0.5:      # trailing stub — fold into report anyway
            if not raw:
                break
        w_tf = gt_rate * seconds
        windows.append({
            "idx": w_idx,
            "seconds": seconds,
            "net_j": net_j,
            "avg_net_w": net_j / seconds if seconds else None,
            "j_per_tflop": net_j / w_tf if w_tf else None,
            "est2": detect_flops.estimate_tflops(net_j, seconds),
            "gt": w_tf,
        })
        w_idx += 1

    lines = []

    def w(s=""):
        lines.append(s)

    w("=" * 78)
    w("drift_test.py — sustained-run stability of the shipped calibration")
    w(f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"Config    : d512_b16_s128_L6, steps={args.steps}")
    w(f"Baseline  : {baseline_mw:.1f} mW (fresh, at test start)")
    w("=" * 78)
    w()
    w(f"  duration     : {r['duration_s']:.1f} s   ({r['n_power_samples']} samples)")
    w(f"  gpu util     : {r['avg_gpu_pct']:.1f} %")
    w(f"  ground truth : {gt:.4f} TFLOPs")
    w(f"  est 2-param  : {est2:.4f} TFLOPs  (err {abs(est2 - gt) / gt * 100:.2f}%)")
    w(f"  est 3-param  : {est3:.4f} TFLOPs  (err {abs(est3 - gt) / gt * 100:.2f}%)")
    w(f"  whole-run J/TFLOP: {r['net_energy_j'] / gt:.3f}")
    w()
    w(f"  {'window':>6} {'t_s':>7} {'net_W':>8} {'J/TFLOP':>9} {'est2_TFLOPs':>12}"
      f" {'window_err%':>11}")
    w("  " + "-" * 60)
    for win in windows:
        err = (abs(win["est2"] - win["gt"]) / win["gt"] * 100.0
               if win["est2"] is not None and win["gt"] else None)
        w(f"  {win['idx']:>6} {win['seconds']:>7.1f} {win['avg_net_w']:>8.3f}"
          f" {win['j_per_tflop']:>9.3f} {win['est2']:>12.4f}"
          f" {err if err is None else format(err, '11.2f')}")
    w()
    jpt = [win["j_per_tflop"] for win in windows if win["j_per_tflop"]]
    if len(jpt) > 2:
        drift_pct = (jpt[-1] - jpt[0]) / jpt[0] * 100.0
        w(f"  J/TFLOP first->last window: {jpt[0]:.3f} -> {jpt[-1]:.3f}"
          f"  ({drift_pct:+.2f}%)")
        w(f"  window spread: min {min(jpt):.3f}  max {max(jpt):.3f}"
          f"  stdev {stdev(jpt):.3f}")
    w()

    text = "\n".join(lines)
    print("\n" + text)
    with open(args.output, "w") as f:
        f.write(text + "\n")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
