#!/usr/bin/env python3
"""
eval_power_monitor.py — Accuracy evaluation for the power-based FLOP estimator.

Runs a sweep of transformer workloads (sample_ml_workload.py), measures net GPU
energy with the same offline sampler that calibrate_power.py uses, then scores
detect_flops.py's estimator against FlopCounterMode ground truth.

The sweep is split into a TRAIN set (used to fit the 2-parameter active-energy
model) and a held-out TEST set (used to prove the fit generalizes). Acceptance
target: held-out max error < 10% of ground-truth FLOPs.

Pipeline reuse:
    - calibrate_power.run_workload  : launch workload + sample power, parse GT
    - calibrate_power.sample_idle   : per-workload idle baseline
    - detect_flops.estimate_tflops  : the production estimator under test

Usage:
    python3 eval_power_monitor.py [--output FILE] [--idle-seconds N]

Run on the Jetson, NOT as root, via .venv/bin/python3, with no other GPU load.
Expect roughly 2 min per config (workload + idle baseline).
"""

import argparse
import os
import sys
import time
from statistics import mean, median, stdev

import numpy as np

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

# ── Workload sweep ─────────────────────────────────────────────────────────────
# Transformer variants spanning memory-bound (small d_model) to compute-bound
# (large d_model), plus batch/seq/layer shape variation. `steps` is sized for
# ~80-120 s of active training per run (>=150 power samples at 2 Hz) so every
# point is trustworthy — the corrupted runs behind the old calibration had <40.
# nhead is left at the workload default (4), which divides every d_model here.
#
# TEST configs use shapes absent from TRAIN (interpolated d_model, extra layers,
# different seq) so the held-out error is a genuine generalization check.
TRAIN_CONFIGS = [
    {"d_model": 128, "batch_size": 8,  "seq_len": 64,  "num_layers": 3, "steps": 2500},
    {"d_model": 256, "batch_size": 8,  "seq_len": 64,  "num_layers": 3, "steps": 2500},
    {"d_model": 512, "batch_size": 8,  "seq_len": 64,  "num_layers": 3, "steps": 1200},
    {"d_model": 768, "batch_size": 8,  "seq_len": 64,  "num_layers": 3, "steps": 600},
    {"d_model": 256, "batch_size": 16, "seq_len": 64,  "num_layers": 3, "steps": 1800},
    {"d_model": 256, "batch_size": 8,  "seq_len": 128, "num_layers": 3, "steps": 1600},
]
TEST_CONFIGS = [
    {"d_model": 384, "batch_size": 8,  "seq_len": 64,  "num_layers": 3, "steps": 1200},
    {"d_model": 512, "batch_size": 8,  "seq_len": 64,  "num_layers": 6, "steps": 650},
    {"d_model": 256, "batch_size": 8,  "seq_len": 96,  "num_layers": 3, "steps": 1800},
]

DEFAULT_OUTPUT  = "eval_results.txt"
DEFAULT_IDLE_S  = 20    # per-workload idle baseline (>=40 samples at 2 Hz)
TARGET_ERR_PCT  = 10.0


def config_label(cfg):
    return (f"d{cfg['d_model']}_b{cfg['batch_size']}"
            f"_s{cfg['seq_len']}_L{cfg['num_layers']}")


# ── Model fitting ──────────────────────────────────────────────────────────────

def fit_active_energy_model(records):
    """
    Fit E_net = e_marginal*TFLOPs + p_overhead*t_active, choosing the parameters
    that minimize the worst-case *relative* FLOP error across `records` — the
    metric the accuracy target is stated in. A plain energy least-squares fit
    instead minimizes absolute joule error, which lets high-energy runs dominate
    and leaves the low-intensity workloads (small d_model) badly served.

    For a fixed p_overhead the relative-least-squares-optimal e_marginal has a
    closed form (e_marginal = Σu² / Σu, with u_i = (E_i - p·t_i)/gt_i), so we
    scan p_overhead on a 1-D grid and keep the minimax solution.

    `records`: dicts with keys net_energy_j, duration_s, ground_truth_tf.
    Returns (e_marginal_j_per_tflop, p_overhead_w).
    """
    gt = np.array([r["ground_truth_tf"] for r in records])
    E  = np.array([r["net_energy_j"] for r in records])
    t  = np.array([r["duration_s"] for r in records])
    best = None
    for p in np.linspace(0.0, 1.2, 4801):
        u = (E - p * t) / gt
        s = u.sum()
        if s <= 0:
            continue
        e_marginal = float((u * u).sum() / s)
        if e_marginal <= 0:
            continue
        est = (E - p * t) / e_marginal
        worst = float(np.max(np.abs((est - gt) / gt)))
        if best is None or worst < best[0]:
            best = (worst, e_marginal, float(p))
    _, e_marginal, p_overhead = best
    return e_marginal, p_overhead


def score(records, e_marginal, p_overhead):
    """Attach est_tflops and err_pct to each record using the production estimator."""
    for r in records:
        est = detect_flops.estimate_tflops(
            r["net_energy_j"], r["duration_s"],
            p_overhead_w=p_overhead, e_marginal_j_per_tflop=e_marginal,
        )
        r["est_tflops"] = est
        gt = r["ground_truth_tf"]
        r["err_pct"] = (abs(est - gt) / gt * 100.0
                        if est is not None and gt else None)
    return records


def err_stats(records):
    errs = [r["err_pct"] for r in records if r.get("err_pct") is not None]
    if not errs:
        return None, None
    return max(errs), mean(errs)


# ── Run + report ───────────────────────────────────────────────────────────────

def run_sweep(configs, split, idle_seconds, volt_path, curr_path, ts_proc):
    records = []
    for i, cfg in enumerate(configs):
        print(f"\n[{split}] {i+1}/{len(configs)}: {config_label(cfg)}"
              f"  steps={cfg['steps']}", flush=True)
        idle = sample_idle(idle_seconds, volt_path, curr_path, ts_proc,
                           f"{split}-idle")
        idle_mw = [mw for _, mw in idle]
        idle_baseline_mw = median(idle_mw) if idle_mw else detect_flops.FALLBACK_IDLE_POWER_MW

        result = run_workload(cfg, idle_baseline_mw, volt_path, curr_path, ts_proc)
        result["split"] = split
        result["label"] = config_label(cfg)

        if result["returncode"] != 0:
            print(f"  WARNING: {config_label(cfg)} exited {result['returncode']}"
                  f" — excluded from fit/scoring", flush=True)
        elif result["ground_truth_tf"] is None or result["net_energy_j"] <= 0:
            print(f"  WARNING: {config_label(cfg)} missing GT or zero energy"
                  f" — excluded", flush=True)
        else:
            print(f"  -> GT {result['ground_truth_tf']:.4f} TFLOPs"
                  f"  | net {result['net_energy_j']:.2f} J"
                  f"  | {result['duration_s']:.1f} s"
                  f"  | {result['n_power_samples']} samples", flush=True)
        records.append(result)
    return records


def valid(records):
    return [r for r in records
            if r["returncode"] == 0
            and r["ground_truth_tf"] not in (None, 0)
            and r["net_energy_j"] > 0]


def write_report(train, test, train_fit, ship_fit, idle_seconds, output_path):
    train_v, test_v = valid(train), valid(test)
    all_v = train_v + test_v
    train_e, train_p = train_fit
    ship_e, ship_p = ship_fit

    # Validation: score TRAIN/TEST with the TRAIN-only fit to prove the fit
    # generalizes to held-out workloads.
    score(train_v, train_e, train_p)
    score(test_v, train_e, train_p)
    train_max, train_mean = err_stats(train_v)
    test_max, test_mean = err_stats(test_v)
    passed = test_max is not None and test_max < TARGET_ERR_PCT

    def table_rows(recs, w):
        hdr = (f"  {'config':<18} {'gt_TFLOPs':>10} {'est_TFLOPs':>10}"
               f" {'err_%':>7} {'net_J':>9} {'t_s':>7} {'n':>5}")
        w(hdr)
        w("  " + "-" * (len(hdr) - 2))
        for r in recs:
            est = f"{r['est_tflops']:.4f}" if r.get("est_tflops") is not None else "N/A"
            gt  = f"{r['ground_truth_tf']:.4f}" if r["ground_truth_tf"] is not None else "N/A"
            ep  = f"{r['err_pct']:.2f}" if r.get("err_pct") is not None else "N/A"
            flag = "" if r["returncode"] == 0 else "  [exit!=0]"
            w(f"  {r['label']:<18} {gt:>10} {est:>10} {ep:>7}"
              f" {r['net_energy_j']:>9.2f} {r['duration_s']:>7.1f}"
              f" {r['n_power_samples']:>5}{flag}")
        w()

    with open(output_path, "w") as f:
        def w(s=""):
            f.write(s + "\n")

        w("=" * 78)
        w("eval_power_monitor.py  —  Power Estimator Accuracy Report")
        w(f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        w(f"Estimator : detect_flops.estimate_tflops (2-param active-energy)")
        w(f"Objective : minimax relative FLOP error")
        w(f"Idle base : per-workload median over {idle_seconds}s")
        w("=" * 78)
        w()
        w("VALIDATION FIT  (fit on TRAIN only: E_net = e_marg*TFLOPs + p_oh*t)")
        w("-" * 50)
        w(f"  E_MARGINAL_J_PER_TFLOP = {train_e:.4f}")
        w(f"  POWER_OVERHEAD_W       = {train_p:.4f}")
        w()

        w("TRAIN  (fitted on these)")
        w("-" * 78)
        table_rows(train, w)
        w("TEST   (held out — scored with the TRAIN fit)")
        w("-" * 78)
        table_rows(test, w)

        w("SUMMARY")
        w("-" * 50)
        if train_max is not None:
            w(f"  TRAIN : max err {train_max:5.2f}%   mean err {train_mean:5.2f}%"
              f"   ({len(train_v)} valid)")
        if test_max is not None:
            w(f"  TEST  : max err {test_max:5.2f}%   mean err {test_mean:5.2f}%"
              f"   ({len(test_v)} valid)")
        w()
        if test_max is not None:
            verdict = "PASS" if passed else "FAIL"
            w(f"  {verdict}: held-out max err {test_max:.2f}% "
              f"{'<' if passed else '>='} {TARGET_ERR_PCT:.0f}% target")
        else:
            w("  FAIL: no valid held-out runs")
        w()

        # Shipping fit: refit on ALL valid runs for the constants to deploy, and
        # show every workload's error under those constants.
        score(all_v, ship_e, ship_p)
        ship_max, ship_mean = err_stats(all_v)
        w("SHIP FIT  (refit on ALL valid runs — the constants to deploy)")
        w("-" * 78)
        table_rows(sorted(all_v, key=lambda r: r["label"]), w)
        if ship_max is not None:
            w(f"  ALL workloads: max err {ship_max:.2f}%   mean err {ship_mean:.2f}%"
              f"   ({len(all_v)} runs)")
            w(f"  {'PASS' if ship_max < TARGET_ERR_PCT else 'FAIL'}:"
              f" every workload {'<' if ship_max < TARGET_ERR_PCT else '>='}"
              f" {TARGET_ERR_PCT:.0f}% target")
        w()

        w("RECOMMENDED CONSTANTS  (paste into detect_flops.py)")
        w("-" * 50)
        w(f"  POWER_OVERHEAD_W         = {ship_p:.3f}")
        w(f"  E_MARGINAL_J_PER_TFLOP   = {ship_e:.2f}")
        w()
        w("RAW RECORDS  (for re-fitting: label split net_J t_s gt_TFLOPs "
          "avg_net_W avg_gpu% avg_emc%)")
        w("-" * 78)
        for r in train + test:
            anw = f"{r['avg_net_power_w']:.4f}" if r["avg_net_power_w"] is not None else "NA"
            gpu = f"{r['avg_gpu_pct']:.1f}" if r["avg_gpu_pct"] is not None else "NA"
            emc = f"{r['avg_emc_pct']:.1f}" if r["avg_emc_pct"] is not None else "NA"
            gt  = f"{r['ground_truth_tf']:.6f}" if r["ground_truth_tf"] is not None else "NA"
            w(f"  {r['label']:<18} {r['split']:<5} {r['net_energy_j']:>9.3f}"
              f" {r['duration_s']:>7.1f} {gt:>11} {anw:>9} {gpu:>6} {emc:>6}")

    print(f"\nReport written to {output_path}")
    return passed, test_max


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output report path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--idle-seconds", type=int, default=DEFAULT_IDLE_S,
                        help=f"Idle baseline duration per run (default: {DEFAULT_IDLE_S})")
    args = parser.parse_args()

    if os.geteuid() == 0:
        print("ERROR: do not run as root. INA3221 sysfs is world-readable and "
              "CUDA needs the regular user's venv. Run without sudo.")
        sys.exit(1)

    volt_path, curr_path = find_ina3221_paths()
    if volt_path is None:
        print("ERROR: INA3221 sensor not found.")
        sys.exit(1)
    if read_power_mw(volt_path, curr_path) is None:
        print("ERROR: failed to read INA3221.")
        sys.exit(1)
    print(f"INA3221 sensor OK: {volt_path}")

    ts_proc = start_tegrastats(int(POLL_S * 1000))
    print("tegrastats started." if ts_proc else "Warning: tegrastats unavailable.")

    try:
        train = run_sweep(TRAIN_CONFIGS, "train", args.idle_seconds,
                          volt_path, curr_path, ts_proc)
        test = run_sweep(TEST_CONFIGS, "test", args.idle_seconds,
                         volt_path, curr_path, ts_proc)

        train_v, test_v = valid(train), valid(test)
        if len(train_v) < 2:
            print(f"\nERROR: only {len(train_v)} valid TRAIN runs — cannot fit.")
            sys.exit(1)

        train_fit = fit_active_energy_model(train_v)            # for validation
        ship_fit = (fit_active_energy_model(train_v + test_v)   # for deployment
                    if test_v else train_fit)
        print(f"\nValidation fit (TRAIN): E_MARGINAL={train_fit[0]:.4f} J/TFLOP,"
              f" P_OVERHEAD={train_fit[1]:.4f} W")
        print(f"Ship fit (ALL runs):    E_MARGINAL={ship_fit[0]:.4f} J/TFLOP,"
              f" P_OVERHEAD={ship_fit[1]:.4f} W")

        passed, test_max = write_report(train, test, train_fit, ship_fit,
                                        args.idle_seconds, args.output)
        verdict = "PASS" if passed else "FAIL"
        tm = f"{test_max:.2f}%" if test_max is not None else "N/A"
        print(f"\n{verdict}: held-out max error = {tm} "
              f"(target < {TARGET_ERR_PCT:.0f}%)")
        sys.exit(0 if passed else 2)
    finally:
        stop_tegrastats(ts_proc)


if __name__ == "__main__":
    main()
