#!/usr/bin/env python3
"""
score_redteam_nn.py — red-team the NEURAL estimator.

The MLP beat the linear models on benign held-out accuracy (13.1% -> 10.4%). But
accuracy on benign says nothing about ADVERSARIAL robustness: does the intensity-
aware net resist the red-team under-report levers (S4 batch-inflation, S3 atypical
geometry, S5 power-cap) better than the linear estimator — or does it extrapolate
badly and get fooled worse? Batch 64/128 sit OUTSIDE the benign batch range (8-32),
so the net is extrapolating exactly where S4 attacks.

Protocol (deployment-framed, same as the linear scorer): the estimator is FROZEN on
the benign frontier and never sees adversarial data. We train an ENSEMBLE of MLPs
(several seeds, median prediction) on ALL benign frontier records, then score each
red config by the signed error of the median prediction over its trials — alongside
the frozen linear 2-param / 4-param signed errors, against the same benign bands.

    python3 red_team/score_redteam_nn.py
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "power_calibration"))
sys.path.insert(0, os.path.join(REPO, "red_team"))

import detect_flops
from analyze_trials import build_pool
from run_trials import trial_records_path
from nn_estimator import (feat_pure, feat_resid, pure_mlp, residual_mlp)
from score_redteam import phase1_calibration, est4, est2, signed_pct
from eval_power_monitor import fit_active_energy_model
from efficiency import _default_phase1_paths

OUT = os.path.join(REPO, "writeup")
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE, BLUE, AQUA, AMBER, RED, PURPLE = ("#e1e0d9", "#c3c2b7", "#2a78d6",
                                              "#1baf7a", "#e69f00", "#d1495b", "#7b5cd6")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9,
})

N_SEED = 5      # ensemble size for the frozen deployed MLP
EPOCHS = 400


def ensemble_predict(benign, test, base_params):
    """Median prediction over N_SEED MLP fits (both pure and residual)."""
    pure = np.median([pure_mlp(benign, test, EPOCHS, seed=s) for s in range(N_SEED)], axis=0)
    resid = np.median([residual_mlp(benign, test, base_params, EPOCHS, seed=100 + s)
                       for s in range(N_SEED)], axis=0)
    return pure, resid


def load_red(paths, strategy=None):
    recs = []
    for p in paths:
        for r in json.load(open(p))["records"]:
            if strategy is None or r["config"].get("strategy") == strategy:
                recs.append(r)
    return recs


def main():
    cal = phase1_calibration(_default_phase1_paths())
    b4, b2 = cal["band4"], cal["band2"]
    pool, _d, _n = build_pool([trial_records_path(k) for k in range(1, 11)])
    benign = [r for v in pool.values() for r in v]
    base_params = fit_active_energy_model(benign)
    print(f"Frozen MLP ensemble ({N_SEED} seeds) trained on {len(benign)} benign records")
    print(f"benign bands: 2p[{b2['lo']:+.1f},{b2['hi']:+.1f}]  4p[{b4['lo']:+.1f},{b4['hi']:+.1f}]\n")

    trials = sorted(glob.glob(os.path.join(REPO, "red_team/red_v3_trial*_records.json")))
    groups = {
        "S4_batch":    (load_red(trials, "S4_batch"),    "batch_size", "batch"),
        "S3_atypical": (load_red(trials, "S3_atypical"), "nhead",      "nhead"),
        "S5_powercap": (load_red([os.path.join(REPO, "red_team/red_s5_v3_records.json")]),
                        "power_cap_w", "cap_W"),
    }

    all_rows = {}
    for strat, (recs, key, vname) in groups.items():
        if not recs:
            continue
        pure, resid = ensemble_predict(benign, recs, base_params)
        # group by the manipulated variable, median over trials
        by = {}
        for i, r in enumerate(recs):
            v = r["config"].get(key)
            gt = r["ground_truth_tf"]
            by.setdefault(v, {"gt": gt, "e2": [], "e4": [], "ep": [], "er": []})
            by[v]["e2"].append(signed_pct(est2(r, cal), gt))
            by[v]["e4"].append(signed_pct(est4(r, cal), gt))
            by[v]["ep"].append(signed_pct(pure[i], gt))
            by[v]["er"].append(signed_pct(resid[i], gt))
        rows = []
        for v in sorted(by):
            g = by[v]
            rows.append({vname: v, "gt": g["gt"],
                         "err_2p": float(np.median(g["e2"])), "err_4p": float(np.median(g["e4"])),
                         "err_pureMLP": float(np.median(g["ep"])),
                         "err_residMLP": float(np.median(g["er"])), "n": len(g["e2"])})
        all_rows[strat] = rows
        print(f"=== {strat}  (var = {vname}; signed error vs GT, − = under-report) ===")
        print(f"  {vname:>7} {'GT':>6} {'2-param':>9} {'4-param':>9} {'pureMLP':>9} {'residMLP':>9}")
        for r in rows:
            print(f"  {str(r[vname]):>7} {r['gt']:>6.0f} {r['err_2p']:>+8.1f}% {r['err_4p']:>+8.1f}%"
                  f" {r['err_pureMLP']:>+8.1f}% {r['err_residMLP']:>+8.1f}%")
        print()

    with open(os.path.join(REPO, "redteam_nn_scores.json"), "w") as f:
        json.dump({"n_benign": len(benign), "n_seed": N_SEED, "bands": {"b2": b2, "b4": b4},
                   "groups": all_rows}, f, indent=1)
    make_fig(all_rows, b2, b4)
    print("wrote redteam_nn_scores.json, writeup/fig_redteam_nn.png")


def make_fig(all_rows, b2, b4):
    panels = [("S4_batch", "batch", "S4 batch-inflation", "batch size (calib ≤ 32)", True),
              ("S3_atypical", "nhead", "S3 atypical nhead", "nhead (head_dim=1024/nhead)", True),
              ("S5_powercap", "cap_W", "S5 power-cap (v2)", "power cap (W)", False)]
    panels = [p for p in panels if p[0] in all_rows]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.9 * len(panels), 4.5), dpi=200)
    if len(panels) == 1:
        axes = [axes]
    series = [("err_2p", BLUE, "s", "2-param (linear)"),
              ("err_4p", AMBER, "D", "4-param (linear)"),
              ("err_pureMLP", PURPLE, "o", "pure MLP"),
              ("err_residMLP", RED, "^", "residual MLP")]
    for ax, (strat, vname, title, xlab, logx) in zip(axes, panels):
        rows = all_rows[strat]
        xs = [r[vname] for r in rows]
        ax.axhspan(b4["lo"], b4["hi"], color=GRID, alpha=0.45, zorder=1, label="benign band (4p)")
        ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
        for kk, col, mk, lab in series:
            ax.plot(xs, [r[kk] for r in rows], marker=mk, color=col, lw=1.6, ms=6,
                    zorder=4, label=lab)
        if logx:
            ax.set_xscale("log", base=2); ax.set_xticks(xs)
            ax.set_xticklabels([str(x) for x in xs]); ax.minorticks_off()
        if strat == "S5_powercap":
            ax.invert_xaxis()
        ax.set_xlabel(xlab)
        ax.set_ylabel("signed error vs GT (%)   − = under-report")
        ax.set_title(title, fontsize=9.2, color=INK, loc="left")
        ax.legend(fontsize=6.8, frameon=False, loc="best")
    fig.suptitle("Red-team levers vs the NEURAL estimator — does intensity-awareness "
                 "resist the attacks or extrapolate into them?", fontsize=8.8, color=MUTED, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_redteam_nn.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
