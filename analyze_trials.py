#!/usr/bin/env python3
"""
analyze_trials.py — SRF phase-I generalization analysis over the 10 trials.

The SRF outline's Results method: the workloads that clear the 80% frontier gate
were each run 10 times (10 traces of the three estimator inputs — GPU power,
DRAM activity, NVLink traffic). We then draw random calibrate/evaluate splits
200 times; on each split every workload contributes ONE randomly chosen trace of
its 10, the estimator is calibrated on the calibrate workloads and scored on the
held-out evaluate workloads, so it must generalize to workloads it never saw.

This produces:
  * the headline number — median held-out error across the 200 splits;
  * fig_trials_cv_error.png  — per-workload median held-out error (how well the
    estimator generalizes, workload by workload);
  * fig_trials_ablation.png  — the input ablation: held-out error CDF with
    1 signal (power), 2 signals (power+DRAM), 3 signals (power+DRAM+NVLink);
  * trials_cv_results.json    — the raw per-model error arrays, for the writeup.

The three estimator variants map to the production estimators in detect_flops.py:
  1 signal  (power)             -> estimate_tflops       (2-param: E=a·FLOPs+d·t)
  2 signals (power+DRAM)        -> estimate_tflops_emc   (3-param: +b·DRAM)
  3 signals (power+DRAM+NVLink) -> estimate_tflops_nvl   (4-param: +c·NVLink)

    python3 analyze_trials.py [--splits 200] [--seed S]

Frontier pool = fp16 configs that clear the 80% gate in ALL trials (so every one
of a workload's 10 traces is genuinely frontier and drawable).
"""
import argparse
import json
import os
import random
import sys
from statistics import mean, median

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "power_calibration"))

from eval_power_monitor import (
    load_records_json, valid, is_frontier, FRONTIER_MIN_GPU_UTIL, TRAIN_FRACTION,
    fit_active_energy_model, fit_active_energy_emc_model, fit_active_energy_nvl_model,
    score, score_emc, score_nvl,
)
from run_trials import trial_records_path

OUT = os.path.join(REPO, "writeup")

# Report palette (matches make_figs_ddp.py) + one CVD-safe categorical hue.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE, BLUE, AQUA, AMBER = "#e1e0d9", "#c3c2b7", "#2a78d6", "#1baf7a", "#e69f00"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 13,
})

# The three input variants, in fixed categorical order (never cycled). Each entry:
# (key, human label, fit fn, score fn, err key, color, linestyle).
VARIANTS = [
    ("s1", "1 signal · power",            fit_active_energy_model,     score,     "err_pct",     BLUE,  "solid"),
    ("s2", "2 signals · power+DRAM",      fit_active_energy_emc_model, score_emc, "err_pct_emc", AQUA,  (0, (5, 2))),
    ("s3", "3 signals · power+DRAM+NVLink", fit_active_energy_nvl_model, score_nvl, "err_pct_nvl", AMBER, (0, (1, 1.4))),
]


def build_pool(trial_files):
    """label -> [10 trace records], keeping only fp16 configs that are frontier in
    EVERY trial (so all 10 traces are valid, drawable, and above the gate)."""
    per_label = {}
    n_trials = len(trial_files)
    for path in trial_files:
        recs, *_ = load_records_json(path)
        for r in valid(recs):
            if r["config"].get("precision", "fp16") == "fp16" and is_frontier(r):
                per_label.setdefault(r["label"], []).append(r)
    pool = {lab: traces for lab, traces in per_label.items() if len(traces) == n_trials}
    dropped = {lab: len(t) for lab, t in per_label.items() if len(t) != n_trials}
    return pool, dropped, n_trials


def score_variant(fit_fn, score_fn, err_key, cal, ev):
    """Fit on cal, score ev, return list of held-out abs% errors (skip degenerate)."""
    fit = fit_fn(cal)
    if fit[0] is None:
        return None
    score_fn(ev, *fit)
    return [r[err_key] for r in ev if r.get(err_key) is not None]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()

    trial_files = [trial_records_path(k) for k in range(1, args.trials + 1)]
    missing = [p for p in trial_files if not os.path.exists(p)]
    if missing:
        sys.exit(f"missing trial files: {missing}")

    pool, dropped, n_trials = build_pool(trial_files)
    labels = sorted(pool)
    print(f"Trials: {n_trials}   frontier pool (fp16, ≥{FRONTIER_MIN_GPU_UTIL:.0f}%"
          f" in ALL trials): {len(labels)} workloads × {n_trials} traces")
    if dropped:
        print(f"  ({len(dropped)} fp16 configs excluded — not frontier in every"
              f" trial: {', '.join(f'{k}[{v}/{n_trials}]' for k, v in dropped.items())})")
    if len(labels) < 6:
        sys.exit("pool too small to split/fit")

    rng = random.Random(args.seed)
    n_cal = min(len(labels) - 1, max(3, round(len(labels) * TRAIN_FRACTION)))
    print(f"Splits: {args.splits}   per split: {n_cal} calibrate / "
          f"{len(labels) - n_cal} held-out evaluate   (random trace per workload)\n")

    # Held-out errors: pooled across all (split, eval-workload) for each variant,
    # and per-label lists (only the splits where that label was held out). For the
    # primary (3-signal) estimator we also keep the signed estimate per workload
    # for the estimated-vs-truth scatter.
    EST_KEY = {"s1": "est_tflops", "s2": "est_tflops_emc", "s3": "est_tflops_nvl"}
    pooled = {k: [] for k, *_ in VARIANTS}
    per_label = {k: {lab: [] for lab in labels} for k, *_ in VARIANTS}
    est_by_variant = {k: {lab: [] for lab in labels} for k, *_ in VARIANTS}  # held-out estimates
    prim_est = est_by_variant["s3"]                 # 3-signal held-out estimates
    gt_by_label = {lab: pool[lab][0]["ground_truth_tf"] for lab in labels}
    skipped = 0

    for s in range(args.splits):
        chosen = {lab: pool[lab][rng.randrange(n_trials)] for lab in labels}
        order = labels[:]
        rng.shuffle(order)
        cal_labels, ev_labels = order[:n_cal], order[n_cal:]
        cal = [chosen[l] for l in cal_labels]
        ev = [chosen[l] for l in ev_labels]
        for key, _lbl, fit_fn, score_fn, err_key, *_ in VARIANTS:
            errs = score_variant(fit_fn, score_fn, err_key, cal, ev)
            if errs is None:
                skipped += 1
                continue
            pooled[key].extend(errs)
            for lab in ev_labels:
                e = chosen[lab].get(err_key)
                if e is not None:
                    per_label[key][lab].append(e)
                est = chosen[lab].get(EST_KEY[key])
                if est is not None:
                    est_by_variant[key][lab].append(est)
        if (s + 1) % 50 == 0:
            print(f"  {s + 1}/{args.splits} splits done", flush=True)

    # ── Summary table ────────────────────────────────────────────────────────
    def stats(a):
        a = np.array(a, dtype=float)
        return dict(median=float(np.median(a)), mean=float(a.mean()),
                    p90=float(np.percentile(a, 90)), max=float(a.max()), n=len(a))

    print("\n" + "=" * 72)
    print("HELD-OUT ERROR ACROSS %d SPLITS  (abs %% vs ground truth, random trace)" % args.splits)
    print("=" * 72)
    print(f"  {'estimator inputs':<30} {'median':>8} {'mean':>7} {'p90':>7} {'max':>7} {'n':>7}")
    summary = {}
    for key, lbl, *_ in VARIANTS:
        st = stats(pooled[key])
        summary[key] = {"label": lbl, **st}
        print(f"  {lbl:<30} {st['median']:>7.2f}% {st['mean']:>6.2f}% "
              f"{st['p90']:>6.2f}% {st['max']:>6.2f}% {st['n']:>7}")
    if skipped:
        print(f"  (skipped {skipped} degenerate variant-fits)")
    headline = summary["s3"]["median"]
    print(f"\nHeadline: median held-out error = {headline:.2f}%  "
          f"(3-signal estimator, {args.splits} splits × random trace of {n_trials})")

    # Per-workload median held-out error + median held-out estimate (primary = 3-signal).
    lab_median = {lab: median(per_label["s3"][lab])
                  for lab in labels if per_label["s3"][lab]}
    lab_est = {lab: median(prim_est[lab]) for lab in labels if prim_est[lab]}
    # Per-variant per-workload median held-out estimate (for the ablation scatter).
    est_by_variant_median = {
        k: {lab: median(est_by_variant[k][lab])
            for lab in labels if est_by_variant[k][lab]}
        for k, *_ in VARIANTS}

    # ── Persist raw results ──────────────────────────────────────────────────
    with open(os.path.join(REPO, "trials_cv_results.json"), "w") as f:
        json.dump({"splits": args.splits, "seed": args.seed, "n_trials": n_trials,
                   "pool": labels, "n_cal": n_cal, "summary": summary,
                   "pooled_errors": {k: pooled[k] for k in pooled},
                   "per_label_median": lab_median, "gt": gt_by_label,
                   "per_label_median_est": lab_est}, f)
    print("\nwrote trials_cv_results.json")

    make_figs(pooled, lab_median, lab_est, gt_by_label, summary,
              args.splits, n_trials, est_by_variant_median)


def make_figs(pooled, lab_median, lab_est, gt_by_label, summary, n_splits, n_trials,
              est_by_variant_median=None):
    overall = summary["s3"]["median"]

    # ── Fig 1 (headline): held-out estimated vs true TFLOPs (3-signal) ─────────
    labs = [l for l in lab_est if l in gt_by_label]
    gt = np.array([gt_by_label[l] for l in labs])
    est = np.array([lab_est[l] for l in labs])
    lim = (0, max(gt.max(), est.max()) * 1.08)
    xs = np.array(lim)
    fig, ax = plt.subplots(figsize=(5.9, 5.7), dpi=200)
    ax.fill_between(xs, xs * 0.8, xs * 1.2, color=GRID, alpha=0.45, zorder=1,
                    linewidth=0, label="±20%")
    ax.fill_between(xs, xs * 0.9, xs * 1.1, color=BASE, alpha=0.5, zorder=1,
                    linewidth=0, label="±10%")
    ax.plot(xs, xs, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.scatter(gt, est, s=34, color=AMBER, zorder=4, linewidths=0,
               label="one workload (median)")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("Ground-truth TFLOPs")
    ax.set_ylabel("Estimated TFLOPs")
    ax.legend(fontsize=12, frameon=False, loc="upper left")
    ax.set_title(f"3-input estimator \u00b7 median error {overall:.1f}%",
                 fontsize=14.5, color=INK, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_trials_cv_error.png"))
    fig.savefig(os.path.join(OUT, "figure_2.png"))
    plt.close(fig)

    # ── Fig 3 (detail): per-workload median held-out error (horizontal bars) ───
    items = sorted(lab_median.items(), key=lambda kv: kv[1])
    dl = [k for k, _ in items]
    dv = [v for _, v in items]
    fig_h = max(3.2, 0.16 * len(dl) + 1.0)
    fig, ax = plt.subplots(figsize=(7.0, fig_h), dpi=200)
    y = np.arange(len(dl))
    ax.barh(y, dv, color=AMBER, height=0.72, zorder=3)
    ax.axvline(overall, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=4,
               label=f"overall median {overall:.1f}%")
    ax.set_yticks(y)
    ax.set_yticklabels(dl, fontsize=5.2)
    ax.set_ylim(-0.8, len(dl) - 0.2)
    ax.set_xlabel("Median held-out error vs ground truth  (%, 3-signal estimator)")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title(f"Per-workload median held-out error ({n_splits} splits)",
                 fontsize=9.2, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_trials_per_workload.png"))
    plt.close(fig)

    # ── Fig 2: input ablation — held-out error CDF, one line per input set ──────
    fig, ax = plt.subplots(figsize=(5.8, 4.4), dpi=200)
    xmax = max(np.percentile(pooled[k], 98) for k, *_ in VARIANTS)
    for key, lbl, _f, _s, _ek, color, ls in VARIANTS:
        a = np.sort(np.array(pooled[key], dtype=float))
        cdf = np.arange(1, len(a) + 1) / len(a)
        med = summary[key]["median"]
        ax.plot(a, cdf, color=color, lw=2.0, ls=ls, zorder=3,
                label=f"{lbl}   (median {med:.1f}%)")
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=1)
    ax.text(xmax, 0.505, "median", fontsize=7, color=MUTED, va="bottom", ha="right")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Held-out error vs ground truth  (%)")
    ax.set_ylabel("Fraction of held-out evaluations ≤ x")
    ax.legend(fontsize=7.6, frameon=False, loc="lower right")
    ax.set_title("Input ablation: does adding DRAM / NVLink improve accuracy?",
                 fontsize=9.2, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_trials_ablation.png"))
    plt.close(fig)

    # ── Fig 4 (ablation scatter): held-out est vs true TFLOPs for the 2- & 3-param
    #    estimators (power-only, power+DRAM) — same style as Fig 1, one panel each.
    if est_by_variant_median is not None:
        panels = [("s1", "Power only", BLUE),
                  ("s2", "Power + DRAM", AQUA)]
        # Shared limits across both panels (start at 0).
        vmax = 0.0
        for key, _t, _c in panels:
            for lab, e in est_by_variant_median[key].items():
                if lab in gt_by_label:
                    vmax = max(vmax, gt_by_label[lab], e)
        lim = (0, vmax * 1.08)
        xs = np.array(lim)
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.7), dpi=200)
        for ax, (key, title, color) in zip(axes, panels):
            labs = [l for l in est_by_variant_median[key] if l in gt_by_label]
            gt = np.array([gt_by_label[l] for l in labs])
            est = np.array([est_by_variant_median[key][l] for l in labs])
            ax.fill_between(xs, xs * 0.8, xs * 1.2, color=GRID, alpha=0.45, zorder=1,
                            linewidth=0, label="±20%")
            ax.fill_between(xs, xs * 0.9, xs * 1.1, color=BASE, alpha=0.5, zorder=1,
                            linewidth=0, label="±10%")
            ax.plot(xs, xs, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2)
            ax.scatter(gt, est, s=34, color=color, zorder=4, linewidths=0,
                       label="one workload (median)")
            ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
            ax.set_xlabel("Ground-truth TFLOPs")
            ax.set_ylabel("Estimated TFLOPs")
            ax.legend(fontsize=12, frameon=False, loc="upper left")
            ax.set_title(f"{title} \u00b7 median error {summary[key]['median']:.1f}%",
                         fontsize=14.5, color=INK, loc="left", pad=10)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_trials_cv_error_ablation.png"))
        fig.savefig(os.path.join(OUT, "figure_3.png"))
        plt.close(fig)

    print("wrote writeup/fig_trials_cv_error.png, fig_trials_cv_error_ablation.png, "
          "fig_trials_ablation.png, fig_trials_per_workload.png")


if __name__ == "__main__":
    main()
