"""
score_redteam.py — score every adversarial strategy against the Phase I benchmark.

Calibration is FROZEN at Phase I: the estimator constants are fit on the Phase I
benign fp16 frontier (never recalibrated including adversarial runs — that would
launder the attack into the fit, the Phase III anti-pattern the outline warns
against). Each adversarial run is then scored against those constants, exactly as
a deployed blue team would score an unseen tenant.

Produces:
  * a per-strategy table: signed estimator error (2-param power-only AND 3-param
    +DRAM) vs the benign baseline error, paired with the efficiency ratio and the
    2× budget verdict;
  * fig_redteam_error.png     — estimator error under each strategy vs benign;
  * fig_redteam_efficiency.png — efficiency (GPU-hours per real FLOP) vs the 2×
    budget line;
  * redteam_scores.json — the raw per-run scores.

Group A (offline records JSON, from `eval_power_monitor.py --configs-module
red_team.redteam_configs:RED_CONFIGS`) and Group B (live_daemon_probe.py record
JSONs) are both accepted; each is scored on the metric appropriate to it.

    python3 red_team/score_redteam.py --group-a red_records.json \
        --group-b live_none.json live_split.json live_throttle.json
"""
import argparse
import json
import os
import sys
from statistics import mean, median

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "power_calibration"))

import detect_flops
from eval_power_monitor import (
    load_records, valid, is_frontier, fit_active_energy_model,
    fit_active_energy_emc_model,
)
from efficiency import (
    benign_reference_rate, efficiency_ratio, is_budget_legal, BUDGET,
    _default_phase1_paths,
)

OUT = os.path.join(REPO_ROOT, "writeup")
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE, BLUE, AQUA, AMBER, RED = ("#e1e0d9", "#c3c2b7", "#2a78d6", "#1baf7a",
                                      "#e69f00", "#d1495b")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9,
})


def signed_pct(est, gt):
    return (est - gt) / gt * 100.0 if est is not None and gt else None


# ── Phase I calibration (frozen) ─────────────────────────────────────────────
def phase1_calibration(paths):
    recs = []
    for p in paths:
        rlist, *_ = load_records(p)
        recs.extend(valid(rlist))
    frontier = [r for r in recs
                if r["config"].get("precision", "fp16") == "fp16" and is_frontier(r)]
    e2, p2 = fit_active_energy_model(frontier)
    e3, c3, p3 = fit_active_energy_emc_model(frontier)
    r_benign, n_ref = benign_reference_rate(recs)
    # Benign baseline error under the ship fit (reference line on the figures).
    errs = [abs(signed_pct(
                detect_flops.estimate_tflops(r["net_energy_j"], r["duration_s"],
                                             p_overhead_w=p2, e_marginal_j_per_tflop=e2),
                r["ground_truth_tf"])) for r in frontier]
    return {"e2": e2, "p2": p2, "e3": e3, "c3": c3, "p3": p3,
            "r_benign": r_benign, "n_ref": n_ref, "n_frontier": len(frontier),
            "benign_median_err": median(errs)}


# ── Group A scoring (offline, same window as Phase I) ────────────────────────
def score_group_a(records, cal):
    """Signed error vs GT under the frozen 2-param and 3-param fits, + efficiency."""
    out = []
    for r in valid(records):
        gt = r["ground_truth_tf"]
        est2 = detect_flops.estimate_tflops(
            r["net_energy_j"], r["duration_s"],
            p_overhead_w=cal["p2"], e_marginal_j_per_tflop=cal["e2"])
        tb = r.get("tb_moved")
        est3 = (detect_flops.estimate_tflops_emc(
                    r["net_energy_j"], r["duration_s"], tb,
                    p_overhead_w=cal["p3"], e_marginal_j_per_tflop=cal["e3"],
                    e_per_tb_j=cal["c3"]) if tb is not None else None)
        ratio = efficiency_ratio(gt, r["duration_s"], cal["r_benign"])
        out.append({
            "label": r["label"], "strategy": r["config"].get("strategy", "?"),
            "group": "A", "gt": gt,
            "err2_signed": signed_pct(est2, gt), "err3_signed": signed_pct(est3, gt),
            "avg_gpu_pct": r.get("avg_gpu_pct"), "frontier": is_frontier(r),
            "duration_s": r["duration_s"], "eff_ratio": ratio,
            "budget_legal": is_budget_legal(ratio),
        })
    return out


# ── Group B scoring (live daemon: attribution/segmentation) ──────────────────
def score_group_b(records, cal, baseline_none=None):
    """Attribution evasion: the daemon's frontier-attributed estimate vs GT. If a
    `--strategy none` baseline record is present, also express as a fraction of
    what the same (single-GPU) daemon attributes to an unsplit/unthrottled run."""
    out = []
    base_attr = (baseline_none.get("attributable_frontier_tf")
                 if baseline_none else None)
    for r in records:
        gt = r.get("ground_truth_tf")
        attr = r.get("attributable_frontier_tf", 0.0)
        # "Evaded fraction": how much of the true FLOPs the blue team FAILS to
        # attribute as frontier training. Relative to the daemon's own benign
        # attribution when available (cancels the single-GPU calibration offset).
        denom = base_attr if base_attr else (gt or None)
        evaded = (1.0 - attr / denom) * 100.0 if denom else None
        out.append({
            "label": r["label"], "strategy": r.get("strategy", "?"), "group": "B",
            "gt": gt, "attributable_frontier_tf": attr,
            "n_sessions": r.get("n_sessions"),
            "n_frontier_sessions": r.get("n_frontier_sessions"),
            "evaded_pct": evaded, "duration_s": r.get("duration_s"),
        })
    return out


def load_json_records(path):
    with open(path) as f:
        d = json.load(f)
    return d["records"] if isinstance(d, dict) and "records" in d else d


# ── Report ───────────────────────────────────────────────────────────────────
def strategy_summary(scores_a):
    by = {}
    for s in scores_a:
        by.setdefault(s["strategy"], []).append(s)
    rows = []
    for strat, ss in sorted(by.items()):
        e2 = [x["err2_signed"] for x in ss if x["err2_signed"] is not None]
        e3 = [x["err3_signed"] for x in ss if x["err3_signed"] is not None]
        ratios = [x["eff_ratio"] for x in ss if x["eff_ratio"] is not None]
        legal = sum(1 for x in ss if x["budget_legal"])
        rows.append({
            "strategy": strat, "n": len(ss),
            "err2_med": median(e2) if e2 else None,
            "err3_med": median(e3) if e3 else None,
            "eff_med": median(ratios) if ratios else None,
            "legal": legal,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase1", nargs="+", default=None,
                    help="Phase I records for calibration (default: the 10 trials)")
    ap.add_argument("--group-a", default=None,
                    help="offline red records JSON (S1/S2/S3)")
    ap.add_argument("--group-b", nargs="*", default=[],
                    help="live_daemon_probe record JSONs (S4/S5; include a none-baseline)")
    args = ap.parse_args()

    p1 = args.phase1 or _default_phase1_paths()
    cal = phase1_calibration(p1)
    print("=" * 74)
    print("PHASE I CALIBRATION (frozen — adversarial runs scored against this)")
    print("=" * 74)
    print(f"  frontier runs         : {cal['n_frontier']}")
    print(f"  2-param  E_MARGINAL   : {cal['e2']:.3f} J/TFLOP   P_OH {cal['p2']:.3f} W")
    print(f"  3-param  E_MARGINAL   : {cal['e3']:.3f}  E_PER_TB {cal['c3']:.3f}  P_OH {cal['p3']:.3f}")
    print(f"  R_benign              : {cal['r_benign']:.0f} TFLOPs/GPU-hour "
          f"(n={cal['n_ref']})   budget ≤ {BUDGET}×")
    print(f"  benign baseline error : {cal['benign_median_err']:.1f}% (median |err|, 2-param ship fit)")

    scores_a, scores_b = [], []

    if args.group_a:
        recs = load_json_records(args.group_a)
        scores_a = score_group_a(recs, cal)
        print("\n" + "=" * 74)
        print("GROUP A — offline strategies (S1 precision · S2 decoy · S3 atypical)")
        print("=" * 74)
        print(f"  {'strategy':<16} {'n':>2} {'err2p(med)':>11} {'err3p(med)':>11} "
              f"{'eff×(med)':>10} {'budget-legal':>13}")
        for row in strategy_summary(scores_a):
            e2 = f"{row['err2_med']:+.1f}%" if row['err2_med'] is not None else "  --"
            e3 = f"{row['err3_med']:+.1f}%" if row['err3_med'] is not None else "  --"
            ef = f"{row['eff_med']:.2f}" if row['eff_med'] is not None else "  --"
            print(f"  {row['strategy']:<16} {row['n']:>2} {e2:>11} {e3:>11} "
                  f"{ef:>10} {row['legal']:>6}/{row['n']:<6}")
        print("  (err signed: + = OVER-estimate/caught, − = UNDER-estimate/evasion;"
              " budget-legal = eff ratio ≤ 2×)")

    baseline_none = None
    b_records = []
    for path in args.group_b:
        rec = load_json_records(path)
        rec = rec[0] if isinstance(rec, list) else rec
        b_records.append(rec)
        if rec.get("strategy") == "none":
            baseline_none = rec
    if b_records:
        scores_b = score_group_b(b_records, cal, baseline_none)
        print("\n" + "=" * 74)
        print("GROUP B — live-daemon strategies (S4 split · S5 throttle)")
        print("=" * 74)
        print(f"  {'strategy':<12} {'GT_TF':>8} {'#sess':>6} {'#frontier':>9} "
              f"{'attrib_TF':>10} {'evaded%':>8}")
        for s in scores_b:
            ev = f"{s['evaded_pct']:.0f}%" if s['evaded_pct'] is not None else "  --"
            print(f"  {s['strategy']:<12} {s['gt'] or 0:>8.0f} {s['n_sessions']:>6} "
                  f"{s['n_frontier_sessions']:>9} {s['attributable_frontier_tf']:>10.0f} {ev:>8}")

    with open(os.path.join(REPO_ROOT, "redteam_scores.json"), "w") as f:
        json.dump({"calibration": cal, "group_a": scores_a, "group_b": scores_b}, f, indent=1)
    print("\nwrote redteam_scores.json")

    if scores_a:
        make_figs(scores_a, cal)


def make_figs(scores_a, cal):
    rows = strategy_summary(scores_a)
    strategies = [r["strategy"] for r in rows]
    x = np.arange(len(strategies))

    # ── Fig 1: estimator error under each strategy (2p vs 3p) ─────────────────
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=200)
    w = 0.38
    e2 = [r["err2_med"] or 0 for r in rows]
    e3 = [r["err3_med"] or 0 for r in rows]
    ax.bar(x - w/2, e2, w, color=BLUE, label="2-param (power)", zorder=3)
    ax.bar(x + w/2, e3, w, color=AMBER, label="3-param (power+DRAM)", zorder=3)
    b = cal["benign_median_err"]
    ax.axhline(b, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=4,
               label=f"benign |err| {b:.0f}%")
    ax.axhline(-b, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(strategies, rotation=20, ha="right", fontsize=7.5)
    ax.set_ylabel("Signed estimator error vs ground truth  (%)")
    ax.legend(fontsize=7.6, frameon=False, loc="best")
    ax.set_title("Estimator error under each adversarial strategy\n"
                 "(− = under-report / evasion; + = over-report / caught)",
                 fontsize=9.0, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_redteam_error.png"))
    plt.close(fig)

    # ── Fig 2: efficiency vs the 2× budget ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=200)
    eff = [r["eff_med"] or 0 for r in rows]
    colors = [AQUA if (r["eff_med"] is not None and r["eff_med"] <= BUDGET) else RED
              for r in rows]
    ax.bar(x, eff, 0.6, color=colors, zorder=3)
    ax.axhline(BUDGET, color=RED, lw=1.2, ls=(0, (4, 3)), zorder=4,
               label=f"{BUDGET:.0f}× budget (legal ≤ this)")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=2, label="benign (1×)")
    ax.set_xticks(x); ax.set_xticklabels(strategies, rotation=20, ha="right", fontsize=7.5)
    ax.set_ylabel("Efficiency ratio  R_benign / R_adv  (GPU-hours per real FLOP)")
    ax.legend(fontsize=7.6, frameon=False, loc="best")
    ax.set_title("Attack cost vs the 2× GPU-hour budget "
                 "(green = legal, red = over budget)", fontsize=9.0, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_redteam_efficiency.png"))
    plt.close(fig)
    print("wrote writeup/fig_redteam_error.png, fig_redteam_efficiency.png")


if __name__ == "__main__":
    main()
