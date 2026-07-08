#!/usr/bin/env python3
"""
adversarial_probe.py — Does the EMC byte term catch a spoofing workload?

Runs the two adversarial_workload.py modes through the exact production
sampling path (calibrate_power.run_workload: PowerSampler + BytesSampler) and
scores both deployed estimators with the SHIPPED constants in detect_flops.py:

  memory-spoof   : the red team burns energy via DRAM traffic with near-zero
                   FLOPs. The energy-only 2-param estimator should be fooled
                   (large FLOP claim); if the byte term works, the 3-param EMC
                   estimate collapses toward <=0 — i.e. the spoof is flagged.
  compute-dense  : cache-resident matmuls (high arithmetic intensity). Control:
                   the byte term must not wreck a legitimate high-AI estimate.

Writes adversarial_results.txt. Run on the Jetson, NOT as root, via
.venv/bin/python3, with no other GPU load. ~7 min total.
"""

import argparse
import os
import sys
import time
from statistics import median, stdev

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "power_calibration"))

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

ADVERSARIAL_SCRIPT = os.path.join(REPO, "adversarial_workload.py")
DEFAULT_OUTPUT = "adversarial_results.txt"


def probe_configs(seconds):
    return [
        {"label": "memory-spoof",
         "script": ADVERSARIAL_SCRIPT,
         "args": ["--mode", "memory-spoof", "--seconds", seconds]},
        {"label": "compute-dense",
         "script": ADVERSARIAL_SCRIPT,
         "args": ["--mode", "compute-dense", "--seconds", seconds]},
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seconds", type=float, default=150.0,
                        help="active duration per mode (default 150)")
    parser.add_argument("--baseline-seconds", type=int, default=60)
    args = parser.parse_args()

    if os.geteuid() == 0:
        print("ERROR: do not run as root (CUDA needs the user venv). "
              "The actmon reader elevates itself via sudoers.")
        sys.exit(1)

    volt_path, curr_path = find_ina3221_paths()
    if volt_path is None:
        print("ERROR: INA3221 sensor not found.")
        sys.exit(1)
    read_power_mw(volt_path, curr_path)
    ts_proc = start_tegrastats(int(POLL_S * 1000))

    try:
        print(f"Measuring idle baseline ({args.baseline_seconds}s); "
              f"ensure no GPU load.", flush=True)
        idle = sample_idle(args.baseline_seconds, volt_path, curr_path,
                           ts_proc, "baseline")
        idle_mw = [mw for _, mw in idle]
        if not idle_mw:
            raise RuntimeError("no idle power samples — cannot baseline")
        baseline_mw = median(idle_mw)
        sd = stdev(idle_mw) if len(idle_mw) > 1 else 0.0
        print(f"Idle baseline: {baseline_mw:.1f} mW (n={len(idle_mw)},"
              f" stdev={sd:.1f})", flush=True)

        results = []
        for cfg in probe_configs(args.seconds):
            print(f"\n=== {cfg['label']} ===", flush=True)
            r = run_workload(cfg, baseline_mw, volt_path, curr_path, ts_proc)
            r["label"] = cfg["label"]
            if r["returncode"] != 0:
                raise RuntimeError(f"{cfg['label']} exited {r['returncode']}")
            results.append(r)
    finally:
        stop_tegrastats(ts_proc)

    # ── Score with the SHIPPED constants (defaults inside detect_flops) ───────
    lines = []

    def w(s=""):
        lines.append(s)
    w("=" * 82)
    w("adversarial_probe.py — Can the EMC byte term catch a spoofing workload?")
    w(f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"Constants : shipped values in detect_flops.py "
      f"(2p: {detect_flops.E_MARGINAL_J_PER_TFLOP} J/TFLOP,"
      f" {detect_flops.POWER_OVERHEAD_W} W |"
      f" 3p: {detect_flops.E_MARGINAL_EMC_J_PER_TFLOP} J/TFLOP,"
      f" {detect_flops.E_PER_TB_J} J/TB,"
      f" {detect_flops.POWER_OVERHEAD_EMC_W} W)")
    w(f"Baseline  : {baseline_mw:.1f} mW measured at probe start")
    w("=" * 82)
    w()
    w(f"  {'workload':<15} {'gpu%':>5} {'gt_TFLOPs':>10} {'est_2p':>9}"
      f" {'est_EMC':>9} {'tb_TB':>8} {'net_J':>8} {'t_s':>7} {'net_W':>6}")
    w("  " + "-" * 84)
    for r in results:
        est2 = detect_flops.estimate_tflops(r["net_energy_j"], r["duration_s"])
        est3 = detect_flops.estimate_tflops_emc(r["net_energy_j"],
                                                r["duration_s"], r["tb_moved"])
        r["est2"], r["est3"] = est2, est3
        gpu = f"{r['avg_gpu_pct']:.0f}" if r["avg_gpu_pct"] is not None else "NA"
        w(f"  {r['label']:<15} {gpu:>5} {r['ground_truth_tf']:>10.4f}"
          f" {est2:>9.2f} {est3:>9.2f} {r['tb_moved']:>8.4f}"
          f" {r['net_energy_j']:>8.1f} {r['duration_s']:>7.1f}"
          f" {r['avg_net_power_w']:>6.2f}")
    w()

    spoof = next(r for r in results if r["label"] == "memory-spoof")
    dense = next(r for r in results if r["label"] == "compute-dense")

    w("MEMORY-SPOOF  (high bytes, ~zero FLOPs — the attack)")
    w("-" * 60)
    gt = spoof["ground_truth_tf"]
    w(f"  true FLOPs      : {gt:.4f} TFLOPs (analytic elementwise count;"
      f" a FLOP audit via FlopCounterMode sees 0)")
    w(f"  2-param claims  : {spoof['est2']:.2f} TFLOPs"
      f"  ({spoof['est2'] / gt:.0f}x the truth — fooled)" if gt > 0 else
      f"  2-param claims  : {spoof['est2']:.2f} TFLOPs (truth ~0 — fooled)")
    caught = spoof["est3"] < 0.1 * spoof["est2"]
    w(f"  3-param (EMC)   : {spoof['est3']:.2f} TFLOPs"
      f"  -> byte term removed"
      f" {(1 - (spoof['est3'] / spoof['est2'])) * 100:.0f}% of the phantom"
      f" FLOPs" if spoof["est2"] > 0 else
      f"  3-param (EMC)   : {spoof['est3']:.2f} TFLOPs")
    w(f"  VERDICT         : byte term"
      f" {'CATCHES the spoof (EMC estimate <=0 or near-zero)' if caught else 'only partially corrects the spoof'}")
    w()
    w("COMPUTE-DENSE  (high FLOPs, minimal bytes — the control)")
    w("-" * 60)
    gt = dense["ground_truth_tf"]
    e2 = abs(dense["est2"] - gt) / gt * 100
    e3 = abs(dense["est3"] - gt) / gt * 100
    w(f"  true FLOPs      : {gt:.4f} TFLOPs")
    w(f"  2-param         : {dense['est2']:.2f} TFLOPs  (err {e2:.1f}%)")
    w(f"  3-param (EMC)   : {dense['est3']:.2f} TFLOPs  (err {e3:.1f}%)")
    w(f"  NOTE            : errors here reflect pure-matmul J/TFLOP vs the"
      f" transformer-calibrated constants, not the byte term (tb is small).")
    w()

    text = "\n".join(lines)
    print("\n" + text)
    with open(args.output, "w") as f:
        f.write(text + "\n")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
