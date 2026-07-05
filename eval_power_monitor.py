#!/usr/bin/env python3
"""
eval_power_monitor.py — Accuracy evaluation for the power-based FLOP estimator.

Runs a broad sweep of transformer workloads (sample_ml_workload.py), measures net
GPU energy with the same offline sampler that calibrate_power.py uses, then scores
detect_flops.py's estimator against FlopCounterMode ground truth.

Design choices that mirror a real treaty verifier monitoring a training cluster:

  * Long workloads. Each config trains for several minutes — closer to a genuine
    LLM training run than a short microbenchmark, and the regime where the fixed
    active-overhead term is a small fraction of total energy.
  * One idle baseline. Idle power is sampled ONCE, before any workload. The
    verifier cannot re-measure a clean baseline between jobs on a busy cluster,
    so neither can the eval — baseline drift during the sweep is left uncorrected
    on purpose, the same error the deployed daemon would face.
  * Frontier-like scope. The threat is an unauthorized *frontier* training run,
    which saturates the GPU. Accuracy is therefore scored only on runs whose
    measured GPU utilization clears FRONTIER_MIN_GPU_UTIL — the regime the
    estimator is meant to police. Lower-utilization runs are still executed and
    shown (so the boundary is visible) but excluded from the fit and verdict:
    on this Orin Nano the CPU+GPU share one power rail and no EMC (memory-
    bandwidth) signal is exposed, so a partially-loaded GPU's energy cannot be
    cleanly attributed to FLOPs.
  * Randomized TRAIN/TEST split. Every run randomly partitions the frontier
    workloads into a TRAIN set (fits the 2-parameter active-energy model) and a
    held-out TEST set (proves it generalizes). Pass --seed to reproduce a split.

Acceptance target: held-out frontier max error < 10% of ground-truth FLOPs.

Pipeline reuse:
    - calibrate_power.run_workload  : launch workload + sample power, parse GT
    - calibrate_power.sample_idle   : the single startup idle baseline
    - detect_flops.estimate_tflops  : the production estimator under test

Usage:
    python3 eval_power_monitor.py [--output FILE] [--baseline-seconds N] [--seed S]

Run on the Jetson, NOT as root, via .venv/bin/python3, with no other GPU load.
Expect ~55-70 min total (long workloads + one baseline).
"""

import argparse
import os
import random
import sys
import time
from statistics import mean, median, stdev

import numpy as np

import detect_flops

# calibrate_power.py lives in power_calibration/ (post-refactor); put it on the path
# so the import below resolves regardless of the cwd the eval is launched from.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "power_calibration"))
from calibrate_power import (
    find_ina3221_paths,
    read_power_mw,
    start_tegrastats,
    stop_tegrastats,
    sample_idle,
    run_workload,
    POLL_S,
)

# ── Workload pool ──────────────────────────────────────────────────────────────
# Transformer configs spanning the space. The high-utilization ones (large
# d_model, big batch/seq, or 6 layers) are the frontier-like targets; the small
# batch=8 seq=64 ones at modest d_model deliberately fall below the frontier gate
# so the report shows where the estimator stops being trustworthy. `steps` is
# sized for ~2-4 min of active training per run at observed Orin Nano step rates,
# yielding several hundred power samples. nhead stays at the workload default (4),
# which divides every d_model here. The TRAIN/TEST split is drawn randomly from
# the frontier subset at run time (see main); it is not fixed per config.
CONFIGS = [
    # — frontier-like: large / high-batch / high-seq / deep (expect high util) —
    {"d_model": 256, "batch_size": 16, "seq_len": 128, "num_layers": 3, "steps": 4000},
    {"d_model": 256, "batch_size": 8,  "seq_len": 256, "num_layers": 3, "steps": 3500},
    {"d_model": 384, "batch_size": 16, "seq_len": 128, "num_layers": 3, "steps": 2500},
    {"d_model": 384, "batch_size": 8,  "seq_len": 128, "num_layers": 6, "steps": 2800},
    {"d_model": 512, "batch_size": 8,  "seq_len": 128, "num_layers": 3, "steps": 2600},
    {"d_model": 512, "batch_size": 16, "seq_len": 64,  "num_layers": 6, "steps": 2400},
    {"d_model": 512, "batch_size": 16, "seq_len": 128, "num_layers": 6, "steps": 1200},
    {"d_model": 640, "batch_size": 8,  "seq_len": 64,  "num_layers": 3, "steps": 4000},
    {"d_model": 640, "batch_size": 8,  "seq_len": 128, "num_layers": 3, "steps": 2400},
    {"d_model": 768, "batch_size": 8,  "seq_len": 64,  "num_layers": 3, "steps": 4000},
    {"d_model": 768, "batch_size": 8,  "seq_len": 128, "num_layers": 6, "steps": 1400},
    # — varied arithmetic intensity: spread FLOPs-per-byte so the 3-param/EMC fit
    #   can separate E_PER_TB_J from E_MARGINAL (otherwise gt and tb are collinear
    #   on the frontier line above and E_PER_TB_J is underdetermined). Large
    #   batch*seq with small-ish d_model leans memory-bound (low AI); d512 mid-AI. —
    {"d_model": 256, "batch_size": 16, "seq_len": 256, "num_layers": 3, "steps": 2200},
    {"d_model": 384, "batch_size": 16, "seq_len": 256, "num_layers": 3, "steps": 1600},
    {"d_model": 512, "batch_size": 16, "seq_len": 256, "num_layers": 3, "steps": 1200},
    # — sub-frontier: small, lightly-loaded (expect to fall below the gate) —
    {"d_model": 128, "batch_size": 8,  "seq_len": 64,  "num_layers": 3, "steps": 4800},
    {"d_model": 192, "batch_size": 8,  "seq_len": 64,  "num_layers": 3, "steps": 4800},
    {"d_model": 256, "batch_size": 8,  "seq_len": 64,  "num_layers": 3, "steps": 4800},
    {"d_model": 384, "batch_size": 8,  "seq_len": 64,  "num_layers": 3, "steps": 4500},
    {"d_model": 512, "batch_size": 8,  "seq_len": 64,  "num_layers": 3, "steps": 4500},
]

DEFAULT_OUTPUT        = "eval_results.txt"
BASELINE_SECONDS      = 90       # single startup idle baseline (>=180 samples at 2 Hz)
FRONTIER_MIN_GPU_UTIL = 80.0     # avg GPU util % a run must clear to be "frontier-like"
P_OVERHEAD_MAX_W      = 4.0      # search ceiling for the fixed-overhead power
TRAIN_FRACTION        = 2.0 / 3.0
TARGET_ERR_PCT        = 10.0


def config_label(cfg):
    return (f"d{cfg['d_model']}_b{cfg['batch_size']}"
            f"_s{cfg['seq_len']}_L{cfg['num_layers']}")


# ── Model fitting ──────────────────────────────────────────────────────────────

def _e_marginal_at(E, t, gt, p):
    """Relative-LS-optimal e_marginal for a fixed p_overhead (closed form), or
    None if degenerate. e_marginal = Σu²/Σu with u_i = (E_i - p·t_i)/gt_i."""
    u = (E - p * t) / gt
    s = u.sum()
    if s <= 0:
        return None
    m = float((u * u).sum() / s)
    return m if m > 0 else None


def fit_active_energy_model(records):
    """
    Fit E_net = e_marginal*TFLOPs + p_overhead*t_active.

    Frontier (high-util) runs are near-collinear in (TFLOPs, t) — they all
    saturate the GPU at a similar FLOP rate — so p_overhead and e_marginal trade
    off and any in-sample objective (minimax or plain SSE) just chases the
    p_overhead search bound. We instead pick p_overhead by **leave-one-out
    cross-validation** (the value that best predicts a held-out run), which has a
    genuine interior optimum, then take the relative-LS-optimal e_marginal at
    that p_overhead. Relative (not absolute-energy) least squares is used so the
    fit targets percent error rather than letting high-energy runs dominate.

    `records`: dicts with keys net_energy_j, duration_s, ground_truth_tf.
    Returns (e_marginal_j_per_tflop, p_overhead_w).
    """
    gt = np.array([r["ground_truth_tf"] for r in records])
    E  = np.array([r["net_energy_j"] for r in records])
    t  = np.array([r["duration_s"] for r in records])
    n  = len(gt)
    idx = np.arange(n)

    best = None
    for p in np.linspace(0.0, P_OVERHEAD_MAX_W, 321):
        if n >= 4:
            # LOO: refit e_marginal on n-1 points, score the held-out one.
            sse, ok = 0.0, True
            for i in range(n):
                mask = idx != i
                mi = _e_marginal_at(E[mask], t[mask], gt[mask], p)
                if mi is None:
                    ok = False
                    break
                pred = (E[i] - p * t[i]) / mi
                sse += ((pred - gt[i]) / gt[i]) ** 2
            if not ok:
                continue
            crit = sse
        else:
            # too few points to cross-validate — fall back to in-sample SSE
            m = _e_marginal_at(E, t, gt, p)
            if m is None:
                continue
            crit = float(np.sum((((E - p * t) / m - gt) / gt) ** 2))
        if best is None or crit < best[0]:
            best = (crit, float(p))

    p_overhead = best[1]
    e_marginal = _e_marginal_at(E, t, gt, p_overhead)
    return e_marginal, p_overhead


def _linear2_at(E, t, gt, tb, p):
    """
    Relative-LS-optimal (e_marginal, e_per_tb) for a FIXED p_overhead, from the
    2x2 normal equations of   min Σ((E_i - a*gt_i - c*tb_i - p*t_i)/gt_i)²
    (the same 1/gt² relative weighting the 2-param fit uses, generalized to two
    linear coefficients).  With u_i=(E_i-p·t_i)/gt_i and v_i=tb_i/gt_i:

        det = n·Σv² - (Σv)²
        a   = (Σu·Σv² - Σv·Σuv) / det
        c   = (n·Σuv - Σv·Σu) / det

    Returns (a, c) with the physical constraint c>=0 (negatives clamped and a
    refit), or None if the system is near-singular (gt and tb too collinear to
    separate the two terms) or a<=0.
    """
    u = (E - p * t) / gt
    v = tb / gt
    n = len(u)
    Sv = float(v.sum()); Svv = float((v * v).sum())
    Su = float(u.sum()); Suv = float((u * v).sum())
    det = n * Svv - Sv * Sv
    if abs(det) <= 1e-9 * (n * Svv + 1.0):     # near-singular relative to scale
        return None
    a = (Su * Svv - Sv * Suv) / det
    c = (n * Suv - Sv * Su) / det
    if c < 0.0:                                # memory energy can't be negative
        c = 0.0
        a = Su / n                             # LS-optimal e_marginal with c=0
    if a <= 0.0:
        return None
    return float(a), float(c)


def _fit_linear_at(E, t, gt, tb, p):
    """(e_marginal, e_per_tb) preferring the 2-coefficient solve; falls back to the
    single-coefficient fit with e_per_tb=0 when the 2x2 is degenerate. None if even
    the single-coefficient fit is degenerate."""
    r = _linear2_at(E, t, gt, tb, p)
    if r is not None:
        return r
    m = _e_marginal_at(E, t, gt, p)
    return (m, 0.0) if m is not None else None


def fit_active_energy_emc_model(records):
    """
    Fit E_net = e_marginal*TFLOPs + e_per_tb*TB_moved + p_overhead*t_active.

    Same LOO-over-p_overhead structure as fit_active_energy_model, but each inner
    step solves TWO linear coefficients (via _fit_linear_at) instead of one. The
    byte term separates from the FLOP term only if the calibration pool spans a
    range of arithmetic intensity; on a collinear (benign) pool the 2x2 is
    near-singular and the solve falls back to e_per_tb=0 — see emc_fit_diagnostics.
    Per the adversarial design, e_per_tb is NOT disabled at inference even when
    collinear here; this fit only needs spread to *determine* its value.

    `records`: dicts with net_energy_j, duration_s, ground_truth_tf, tb_moved.
    Returns (e_marginal, e_per_tb, p_overhead).
    """
    gt = np.array([r["ground_truth_tf"] for r in records])
    E  = np.array([r["net_energy_j"] for r in records])
    t  = np.array([r["duration_s"] for r in records])
    tb = np.array([r["tb_moved"] for r in records])
    n  = len(gt)
    idx = np.arange(n)

    best = None
    for p in np.linspace(0.0, P_OVERHEAD_MAX_W, 321):
        if n >= 5:
            # LOO needs n-1 >= 4 points to fit 2 linear params + score; n>=5.
            sse, ok = 0.0, True
            for i in range(n):
                mask = idx != i
                fit = _fit_linear_at(E[mask], t[mask], gt[mask], tb[mask], p)
                if fit is None:
                    ok = False
                    break
                a, c = fit
                pred = (E[i] - c * tb[i] - p * t[i]) / a
                sse += ((pred - gt[i]) / gt[i]) ** 2
            if not ok:
                continue
            crit = sse
        else:
            fit = _fit_linear_at(E, t, gt, tb, p)
            if fit is None:
                continue
            a, c = fit
            pred = (E - c * tb - p * t) / a
            crit = float(np.sum(((pred - gt) / gt) ** 2))
        if best is None or crit < best[0]:
            best = (crit, float(p))

    p_overhead = best[1]
    a, c = _fit_linear_at(E, t, gt, tb, p_overhead)
    return a, c, p_overhead


def emc_fit_diagnostics(records):
    """
    Informational conditioning of the byte term: corr(gt, tb) and the condition
    number of the 2x2 normal matrix [[n, Σv],[Σv, Σv²]] (v=tb/gt). High corr /
    large cond means e_per_tb is poorly determined on this data (collinear), so the
    fitted value is unreliable — but the term still runs at inference by design.
    """
    gt = np.array([r["ground_truth_tf"] for r in records])
    tb = np.array([r["tb_moved"] for r in records])
    n = len(gt)
    corr = float(np.corrcoef(gt, tb)[0, 1]) if n > 1 else float("nan")
    v = tb / gt
    Sv = float(v.sum()); Svv = float((v * v).sum())
    cond = float(np.linalg.cond(np.array([[n, Sv], [Sv, Svv]])))
    return {"corr_gt_tb": corr, "cond": cond}


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


def score_emc(records, e_marginal, e_per_tb, p_overhead):
    """Attach est_tflops_emc / err_pct_emc via the production 3-param estimator."""
    for r in records:
        tb = r.get("tb_moved")
        est = (detect_flops.estimate_tflops_emc(
                   r["net_energy_j"], r["duration_s"], tb,
                   p_overhead_w=p_overhead, e_marginal_j_per_tflop=e_marginal,
                   e_per_tb_j=e_per_tb)
               if tb is not None else None)
        r["est_tflops_emc"] = est
        gt = r["ground_truth_tf"]
        r["err_pct_emc"] = (abs(est - gt) / gt * 100.0
                            if est is not None and gt else None)
    return records


def err_stats(records, key="err_pct"):
    errs = [r[key] for r in records if r.get(key) is not None]
    if not errs:
        return None, None
    return max(errs), mean(errs)


def valid(records):
    return [r for r in records
            if r["returncode"] == 0
            and r["ground_truth_tf"] not in (None, 0)
            and r["net_energy_j"] > 0]


def is_frontier(r):
    return r["avg_gpu_pct"] is not None and r["avg_gpu_pct"] >= FRONTIER_MIN_GPU_UTIL


# ── Run + report ───────────────────────────────────────────────────────────────

def run_sweep(configs, idle_baseline_mw, volt_path, curr_path, ts_proc):
    """Run every config against the single shared idle baseline. The frontier
    gate and TRAIN/TEST assignment are applied afterward in main()."""
    records = []
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {config_label(cfg)}"
              f"  steps={cfg['steps']}", flush=True)
        result = run_workload(cfg, idle_baseline_mw, volt_path, curr_path, ts_proc)
        result["label"] = config_label(cfg)
        result["split"] = "excl"   # overwritten for the frontier runs that get split

        if result["returncode"] != 0:
            print(f"  WARNING: {config_label(cfg)} exited {result['returncode']}"
                  f" — excluded", flush=True)
        elif result["ground_truth_tf"] is None or result["net_energy_j"] <= 0:
            print(f"  WARNING: {config_label(cfg)} missing GT or zero energy"
                  f" — excluded", flush=True)
        else:
            tag = "frontier" if is_frontier(result) else "sub-frontier"
            gpu = result["avg_gpu_pct"]
            gpu_s = f"{gpu:.0f}%" if gpu is not None else "NA"
            print(f"  -> GT {result['ground_truth_tf']:.4f} TFLOPs"
                  f"  | net {result['net_energy_j']:.2f} J"
                  f"  | {result['duration_s']:.1f} s"
                  f"  | gpu {gpu_s}  | {tag}", flush=True)
        records.append(result)
    return records


def write_report(train, test, frontier, sub_frontier, all_records,
                 train_fit, ship_fit, train_fit_emc, ship_fit_emc,
                 baseline_mw, baseline_seconds, seed, output_path):
    train_e, train_p = train_fit
    ship_e, ship_p = ship_fit
    emc_available = train_fit_emc is not None and ship_fit_emc is not None

    # Validation: score TRAIN/TEST with the TRAIN-only fit to prove the fit
    # generalizes to the randomly held-out frontier workloads. The 2-param path
    # is the headline verdict; the 3-param/EMC path is scored alongside for A/B.
    score(train, train_e, train_p)
    score(test, train_e, train_p)
    train_max, train_mean = err_stats(train)
    test_max, test_mean = err_stats(test)
    passed = test_max is not None and test_max < TARGET_ERR_PCT

    if emc_available:
        train_e_emc, train_c_emc, train_p_emc = train_fit_emc
        ship_e_emc, ship_c_emc, ship_p_emc = ship_fit_emc
        score_emc(train, train_e_emc, train_c_emc, train_p_emc)
        score_emc(test, train_e_emc, train_c_emc, train_p_emc)
        train_max_emc, train_mean_emc = err_stats(train, "err_pct_emc")
        test_max_emc, test_mean_emc = err_stats(test, "err_pct_emc")
        # No-regression: EMC held-out must clear the target AND not be worse than
        # the 2-param held-out (small tolerance for fit noise).
        emc_passed = (test_max_emc is not None and test_max_emc < TARGET_ERR_PCT
                      and (test_max is None or test_max_emc <= test_max + 0.5))
        diag = emc_fit_diagnostics(frontier)

    def table_rows(recs, w):
        cols = (f"  {'config':<18} {'gpu%':>5} {'gt_TFLOPs':>10} {'est_2p':>10}"
                f" {'err2p%':>7}")
        if emc_available:
            cols += f" {'est_EMC':>10} {'errEMC%':>7} {'tb_TB':>8}"
        cols += f" {'net_J':>9} {'t_s':>7} {'n':>5}"
        w(cols)
        w("  " + "-" * (len(cols) - 2))
        for r in recs:
            est = f"{r['est_tflops']:.4f}" if r.get("est_tflops") is not None else "N/A"
            gt  = f"{r['ground_truth_tf']:.4f}" if r["ground_truth_tf"] is not None else "N/A"
            ep  = f"{r['err_pct']:.2f}" if r.get("err_pct") is not None else "N/A"
            gpu = f"{r['avg_gpu_pct']:.0f}" if r["avg_gpu_pct"] is not None else "NA"
            row = (f"  {r['label']:<18} {gpu:>5} {gt:>10} {est:>10} {ep:>7}")
            if emc_available:
                este = f"{r['est_tflops_emc']:.4f}" if r.get("est_tflops_emc") is not None else "N/A"
                epe  = f"{r['err_pct_emc']:.2f}" if r.get("err_pct_emc") is not None else "N/A"
                tb   = f"{r['tb_moved']:.4f}" if r.get("tb_moved") is not None else "NA"
                row += f" {este:>10} {epe:>7} {tb:>8}"
            row += (f" {r['net_energy_j']:>9.2f} {r['duration_s']:>7.1f}"
                    f" {r['n_power_samples']:>5}")
            w(row)
        w()

    with open(output_path, "w") as f:
        def w(s=""):
            f.write(s + "\n")

        w("=" * 82)
        w("eval_power_monitor.py  —  Power Estimator Accuracy Report")
        w(f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
        w(f"Estimator : detect_flops.estimate_tflops (2-param) "
          f"+ estimate_tflops_emc (3-param, A/B)")
        w(f"Objective : relative least squares")
        w(f"Baseline  : single startup median over {baseline_seconds}s"
          f" = {baseline_mw:.1f} mW")
        w(f"Frontier  : scored on runs with avg GPU util >= "
          f"{FRONTIER_MIN_GPU_UTIL:.0f}%  ({len(frontier)} of {len(frontier)+len(sub_frontier)} valid)")
        w(f"Split     : random, seed={seed}  ({len(train)} train / {len(test)} test)")
        if not emc_available:
            w("EMC term  : UNAVAILABLE (no actmon tb_moved for frontier runs;"
              " is the sudoers actmon_reader configured?) — 2-param only")
        w("=" * 82)
        w()
        w("VALIDATION FIT  (fit on TRAIN only)")
        w("-" * 50)
        w(f"  2-param:  E_net = e_marg*TFLOPs + p_oh*t")
        w(f"    E_MARGINAL_J_PER_TFLOP = {train_e:.4f}")
        w(f"    POWER_OVERHEAD_W       = {train_p:.4f}")
        if emc_available:
            w(f"  3-param:  E_net = e_marg*TFLOPs + e_per_tb*TB + p_oh*t")
            w(f"    E_MARGINAL_J_PER_TFLOP = {train_e_emc:.4f}")
            w(f"    E_PER_TB_J             = {train_c_emc:.4f}")
            w(f"    POWER_OVERHEAD_W       = {train_p_emc:.4f}")
        w()

        w("TRAIN  (frontier; fitted on these)")
        w("-" * 82)
        table_rows(sorted(train, key=lambda r: r["label"]), w)
        w("TEST   (frontier; held out — scored with the TRAIN fit)")
        w("-" * 82)
        table_rows(sorted(test, key=lambda r: r["label"]), w)

        w("SUMMARY  (frontier only)")
        w("-" * 50)
        if train_max is not None:
            w(f"  2-param TRAIN : max err {train_max:5.2f}%   mean err {train_mean:5.2f}%"
              f"   ({len(train)} runs)")
        if test_max is not None:
            w(f"  2-param TEST  : max err {test_max:5.2f}%   mean err {test_mean:5.2f}%"
              f"   ({len(test)} runs)")
        if emc_available:
            if train_max_emc is not None:
                w(f"  3-param TRAIN : max err {train_max_emc:5.2f}%   mean err"
                  f" {train_mean_emc:5.2f}%   ({len(train)} runs)")
            if test_max_emc is not None:
                w(f"  3-param TEST  : max err {test_max_emc:5.2f}%   mean err"
                  f" {test_mean_emc:5.2f}%   ({len(test)} runs)")
        w()
        if test_max is not None:
            verdict = "PASS" if passed else "FAIL"
            w(f"  [2-param] {verdict}: held-out frontier max err {test_max:.2f}% "
              f"{'<' if passed else '>='} {TARGET_ERR_PCT:.0f}% target")
        else:
            w("  [2-param] FAIL: no valid held-out frontier runs")
        if emc_available and test_max_emc is not None:
            ev = "PASS" if emc_passed else "FAIL"
            w(f"  [3-param] {ev}: held-out EMC max err {test_max_emc:.2f}%"
              f"  (target <{TARGET_ERR_PCT:.0f}% AND no regression vs 2-param)")
            w(f"  [diag] corr(gt,tb)={diag['corr_gt_tb']:.3f}  cond(2x2)={diag['cond']:.1f}"
              f"  — high values mean E_PER_TB_J is weakly determined (collinear)")
        w()

        # Shipping fit: refit on ALL frontier runs for the constants to deploy.
        score(frontier, ship_e, ship_p)
        ship_max, ship_mean = err_stats(frontier)
        if emc_available:
            score_emc(frontier, ship_e_emc, ship_c_emc, ship_p_emc)
            ship_max_emc, ship_mean_emc = err_stats(frontier, "err_pct_emc")
        w("SHIP FIT  (refit on ALL frontier runs — the constants to deploy)")
        w("-" * 82)
        table_rows(sorted(frontier, key=lambda r: r["label"]), w)
        if ship_max is not None:
            w(f"  2-param ALL frontier: max err {ship_max:.2f}%   mean err {ship_mean:.2f}%"
              f"   ({len(frontier)} runs)")
            w(f"  {'PASS' if ship_max < TARGET_ERR_PCT else 'FAIL'}:"
              f" every frontier run {'<' if ship_max < TARGET_ERR_PCT else '>='}"
              f" {TARGET_ERR_PCT:.0f}% target")
        if emc_available and ship_max_emc is not None:
            w(f"  3-param ALL frontier: max err {ship_max_emc:.2f}%   mean err"
              f" {ship_mean_emc:.2f}%   ({len(frontier)} runs)")
        w()

        # Sub-frontier: shown scored with the ship fit to make the out-of-scope
        # degradation visible. NOT part of the verdict. This is where the EMC term
        # is expected to help (off the frontier collinearity line).
        if sub_frontier:
            score(sub_frontier, ship_e, ship_p)
            sf_max, sf_mean = err_stats(sub_frontier)
            if emc_available:
                score_emc(sub_frontier, ship_e_emc, ship_c_emc, ship_p_emc)
                sf_max_emc, sf_mean_emc = err_stats(sub_frontier, "err_pct_emc")
            w(f"SUB-FRONTIER  (avg util < {FRONTIER_MIN_GPU_UTIL:.0f}% — out of scope,"
              f" not scored in verdict)")
            w("-" * 82)
            table_rows(sorted(sub_frontier, key=lambda r: r["label"]), w)
            if sf_max is not None:
                w(f"  (for reference) 2-param max err {sf_max:.2f}%  mean err {sf_mean:.2f}%")
            if emc_available and sf_max_emc is not None:
                w(f"  (for reference) 3-param max err {sf_max_emc:.2f}%  mean err {sf_mean_emc:.2f}%"
                  f"  — does the byte term help off-frontier?")
            w()

        w("RECOMMENDED CONSTANTS  (paste into detect_flops.py)")
        w("-" * 50)
        w(f"  POWER_OVERHEAD_W         = {ship_p:.3f}")
        w(f"  E_MARGINAL_J_PER_TFLOP   = {ship_e:.2f}")
        if emc_available:
            w(f"  # 3-param (EMC) matched set — deploy together with the above only")
            w(f"  # if promoting the EMC estimator; otherwise keep E_PER_TB_J = 0.0.")
            w(f"  POWER_OVERHEAD_W         = {ship_p_emc:.3f}   # (EMC fit)")
            w(f"  E_MARGINAL_J_PER_TFLOP   = {ship_e_emc:.2f}   # (EMC fit)")
            w(f"  E_PER_TB_J               = {ship_c_emc:.3f}")
        else:
            w(f"  # E_PER_TB_J unavailable (no actmon data); leave it at 0.0")
        w()
        w("RAW RECORDS  (label split net_J t_s gt_TFLOPs avg_net_W avg_gpu% avg_emc% tb_TB)")
        w("-" * 82)
        for r in sorted(all_records, key=lambda r: r["label"]):
            anw = f"{r['avg_net_power_w']:.4f}" if r["avg_net_power_w"] is not None else "NA"
            gpu = f"{r['avg_gpu_pct']:.1f}" if r["avg_gpu_pct"] is not None else "NA"
            emc = f"{r['avg_emc_pct']:.1f}" if r["avg_emc_pct"] is not None else "NA"
            tb  = f"{r['tb_moved']:.4f}" if r.get("tb_moved") is not None else "NA"
            gt  = f"{r['ground_truth_tf']:.6f}" if r["ground_truth_tf"] is not None else "NA"
            net = r["net_energy_j"] if r["net_energy_j"] is not None else float("nan")
            dur = r["duration_s"] if r["duration_s"] is not None else float("nan")
            w(f"  {r['label']:<18} {r['split']:<5} {net:>9.3f}"
              f" {dur:>7.1f} {gt:>11} {anw:>9} {gpu:>6} {emc:>6} {tb:>8}")

    print(f"\nReport written to {output_path}")
    return passed, test_max


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output report path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--baseline-seconds", type=int, default=BASELINE_SECONDS,
                        help=f"Single startup idle baseline duration"
                             f" (default: {BASELINE_SECONDS})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the random TRAIN/TEST split"
                             " (default: random each run; the chosen seed is printed)")
    args = parser.parse_args()

    if os.geteuid() == 0:
        print("ERROR: do not run as root. INA3221 sysfs is world-readable and "
              "CUDA needs the regular user's venv. Run without sudo.")
        sys.exit(1)

    volt_path, curr_path = find_ina3221_paths()
    if volt_path is None:
        print("ERROR: INA3221 sensor not found.")
        sys.exit(1)
    read_power_mw(volt_path, curr_path)   # raises loudly if the sensor is unreadable
    print(f"INA3221 sensor OK: {volt_path}")

    ts_proc = start_tegrastats(int(POLL_S * 1000))   # raises if tegrastats is missing
    print("tegrastats started.")

    try:
        # ── Single idle baseline (measured once, before any workload) ─────────
        print(f"\nMeasuring single idle baseline ({args.baseline_seconds}s)."
              f" Ensure no GPU workloads are running.", flush=True)
        idle = sample_idle(args.baseline_seconds, volt_path, curr_path,
                           ts_proc, "baseline")
        idle_mw = [mw for _, mw in idle]
        if not idle_mw:
            # The baseline is critical (it sets the matched set); no samples means
            # something is wrong with sampling — fail loudly rather than fall back.
            raise RuntimeError(
                "no idle power samples collected — cannot establish a baseline "
                "(check the INA3221 sensor and POLL_S timing).")
        idle_baseline_mw = median(idle_mw)
        sd = stdev(idle_mw) if len(idle_mw) > 1 else 0.0
        print(f"Idle baseline: {idle_baseline_mw:.1f} mW"
              f"  (n={len(idle_mw)}, stdev={sd:.1f} mW)"
              f"  — shared by ALL workloads", flush=True)

        # ── Run the full workload pool against that one baseline ──────────────
        records = run_sweep(CONFIGS, idle_baseline_mw, volt_path, curr_path, ts_proc)

        valid_recs = valid(records)
        frontier = [r for r in valid_recs if is_frontier(r)]
        sub_frontier = [r for r in valid_recs if not is_frontier(r)]
        for r in sub_frontier:
            r["split"] = "sub"
        print(f"\nFrontier runs (util >= {FRONTIER_MIN_GPU_UTIL:.0f}%): "
              f"{len(frontier)} of {len(valid_recs)} valid")
        if len(frontier) < 3:
            print(f"ERROR: only {len(frontier)} frontier runs — need >=3 to split"
                  f" and fit. Adjust CONFIGS toward higher utilization.")
            sys.exit(1)

        # ── Randomized TRAIN/TEST split within the frontier subset ────────────
        seed = args.seed if args.seed is not None else random.randrange(1 << 30)
        rng = random.Random(seed)
        shuffled = frontier[:]
        rng.shuffle(shuffled)
        n_train = max(2, round(len(shuffled) * TRAIN_FRACTION))
        n_train = min(n_train, len(shuffled) - 1)   # keep >=1 held-out test
        train, test = shuffled[:n_train], shuffled[n_train:]
        for r in train:
            r["split"] = "train"
        for r in test:
            r["split"] = "test"
        print(f"Random split (seed={seed}): {len(train)} train / {len(test)} test")
        print(f"  TRAIN: {', '.join(sorted(r['label'] for r in train))}")
        print(f"  TEST : {', '.join(sorted(r['label'] for r in test))}")

        train_fit = fit_active_energy_model(train)           # for validation
        ship_fit = fit_active_energy_model(frontier)         # for deployment
        print(f"\nValidation fit (TRAIN): E_MARGINAL={train_fit[0]:.4f} J/TFLOP,"
              f" P_OVERHEAD={train_fit[1]:.4f} W")
        print(f"Ship fit (ALL frontier): E_MARGINAL={ship_fit[0]:.4f} J/TFLOP,"
              f" P_OVERHEAD={ship_fit[1]:.4f} W")

        # 3-param / EMC fit, in parallel. actmon bytes are captured for every run
        # (run_workload raises if the reader yields nothing), so a missing tb_moved
        # here is an unexpected hard error — we do NOT silently report 2-param only.
        missing_tb = [r["label"] for r in frontier if r.get("tb_moved") is None]
        if missing_tb:
            raise RuntimeError(f"frontier runs missing tb_moved: {missing_tb} — "
                               f"actmon capture failed for them.")
        train_fit_emc = fit_active_energy_emc_model(train)
        ship_fit_emc = fit_active_energy_emc_model(frontier)
        print(f"Validation fit EMC (TRAIN): E_MARGINAL={train_fit_emc[0]:.4f},"
              f" E_PER_TB={train_fit_emc[1]:.4f}, P_OVERHEAD={train_fit_emc[2]:.4f}")
        print(f"Ship fit EMC (ALL frontier): E_MARGINAL={ship_fit_emc[0]:.4f},"
              f" E_PER_TB={ship_fit_emc[1]:.4f}, P_OVERHEAD={ship_fit_emc[2]:.4f}")

        passed, test_max = write_report(train, test, frontier, sub_frontier,
                                        records, train_fit, ship_fit,
                                        train_fit_emc, ship_fit_emc,
                                        idle_baseline_mw, args.baseline_seconds,
                                        seed, args.output)
        verdict = "PASS" if passed else "FAIL"
        tm = f"{test_max:.2f}%" if test_max is not None else "N/A"
        print(f"\n{verdict}: held-out frontier max error = {tm} "
              f"(target < {TARGET_ERR_PCT:.0f}%)  [seed={seed}]")
        sys.exit(0 if passed else 2)
    finally:
        stop_tegrastats(ts_proc)


if __name__ == "__main__":
    main()
