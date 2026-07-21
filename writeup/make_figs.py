"""Generate the Results figures for the write-up from real experiment data."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "writeup")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "power_calibration"))

from eval_power_monitor import (fit_active_energy_emc_model, make_split,
                                score_emc, err_stats, load_bias_trials, cfg_val)

# ── palette / chrome (dataviz reference, light mode) ──────────────────────────
SURFACE = "#fcfcfb"
INK     = "#0b0b0b"
INK2    = "#52514e"
MUTED   = "#898781"
GRID    = "#e1e0d9"
BASE    = "#c3c2b7"
BLUE    = "#2a78d6"
AQUA    = "#1baf7a"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9,
})


def load_fp32_frontier():
    p = json.load(open(os.path.join(REPO, "eval_results_v2_records.json")))
    recs = [r for r in p["records"]
            if r["returncode"] == 0 and r["ground_truth_tf"]
            and r["net_energy_j"] > 0]
    fp32 = [r for r in recs if r["config"].get("precision", "fp32") == "fp32"]
    frontier = [r for r in fp32 if r["avg_gpu_pct"] and r["avg_gpu_pct"] >= 80]
    return recs, frontier


_ALL_RECS, FRONTIER = load_fp32_frontier()

# LOO trials (shared by figs 6, 7): every per-run error/estimate drawn from
# these is held out — predicted by a fit that excluded that run.
TRIAL_FILES = [f for f in ("eval_results_v2_records.json",
                           "eval_results_25w_trial2_records.json",
                           "eval_results_25w_trial3_records.json")
               if os.path.exists(os.path.join(REPO, f))]
trials6 = load_bias_trials([os.path.join(REPO, f) for f in TRIAL_FILES])
n_tr = len(trials6)

# ── 200 resampled splits: held-out error stats ───────────────────────────────
# The loop also collects each workload's held-out estimates for fig 7: with a
# 2/3 train fraction each workload lands in the test set ~67/200 times.
base = 1000
maxerrs, pooled_errs = [], []
split_ests = [[] for _ in FRONTIER]
idx7 = {id(r): i for i, r in enumerate(FRONTIER)}
for i in range(200):
    train, test, _ = make_split(FRONTIER, "random", base + i)
    e3, c3, p3 = fit_active_energy_emc_model(train)
    score_emc(test, e3, c3, p3)
    m, _ = err_stats(test, "err_pct_emc")
    maxerrs.append(m)
    for r in test:
        if r["est_tflops_emc"] is not None:
            split_ests[idx7[id(r)]].append(r["est_tflops_emc"])
            pooled_errs.append(r["err_pct_emc"])
maxerrs = np.array(maxerrs)
pooled_errs = np.array(pooled_errs)
fail = float((maxerrs >= 10).mean() * 100)
print(f"200-split held-out (3p): per-workload mean {pooled_errs.mean():.2f}%"
      f" sd {pooled_errs.std(ddof=1):.2f}%  |  per-split max mean"
      f" {maxerrs.mean():.2f}% sd {maxerrs.std(ddof=1):.2f}%")

# ── Figures 6a/6b: signed LOO error by hyperparameter axis (all 25W trials) ───
# One dot per fp32 frontier run (leave-one-out signed error from trials6 —
# the same numbers as the --bias-report output), marker shape = trial, aqua dash =
# mean at each value. Positive = overestimate. Split into two figures:
# (a) model-shape axes, (b) training-setup axes.
from matplotlib.lines import Line2D

PANELS = [  # (config key, panel title, row) — names match the report's
    # "Constructing Sample Workloads" definitions
    ("d_model", "Model Width", 0),
    ("seq_len", "Sequence Length", 0),
    ("dim_feedforward", "Feed-Forward\nWidth", 0),
    ("batch_size", "Batch Size", 1),
    ("num_layers", "Depth", 1),
    ("nhead", "Attention Heads", 1),
    ("optimizer", "Optimizer", 1),
]
OPT_NAMES = {"adamw": "AdamW", "sgd": "SGD"}
levels6 = {key: sorted({cfg_val(r, key) for t in trials6 for r in t["pool"]})
           for key, _title, _row in PANELS}
rows = {0: [p for p in PANELS if p[2] == 0], 1: [p for p in PANELS if p[2] == 1]}
MARKERS = ["o", "s", "^"]

fig = plt.figure(figsize=(7.2, 5.4), dpi=200)
gs = fig.add_gridspec(2, 1, hspace=0.42)
for row_i, panels in rows.items():
    widths = [len(levels6[k]) for k, _t, _r in panels]
    sub = gs[row_i].subgridspec(1, len(panels), width_ratios=widths, wspace=0.14)
    for pi, (key, title, _r) in enumerate(panels):
        ax = fig.add_subplot(sub[pi])
        lv = levels6[key]
        for ti, t in enumerate(trials6):
            xs = [lv.index(cfg_val(r, key)) + (ti - (n_tr - 1) / 2) * 0.22
                  for r in t["pool"]]
            ys = [r["signed_loo_emc"] for r in t["pool"]]
            ax.scatter(xs, ys, s=16, marker=MARKERS[ti], color=BLUE,
                       alpha=0.65, linewidths=0, zorder=3)
        for i, v in enumerate(lv):
            m = float(np.mean([r["signed_loo_emc"] for t in trials6
                               for r in t["pool"] if cfg_val(r, key) == v]))
            ax.hlines(m, i - 0.34, i + 0.34, color=AQUA, lw=2.2, zorder=4)
        ax.axhline(0, color=BASE, lw=1, zorder=2)
        ax.set_xticks(range(len(lv)))
        ax.set_xticklabels([OPT_NAMES.get(str(v), str(v)) for v in lv],
                           fontsize=7,
                           rotation=45 if key == "dim_feedforward" else 0)
        ax.set_xlim(-0.6, len(lv) - 0.4)
        ax.set_ylim(-16.5, 13)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=8, color=INK2)
        if pi == 0:
            ax.set_ylabel("Signed Error (%)\n↑ Over   ↓ Under", fontsize=8)
            if row_i == 0 and n_tr > 1:
                handles = [Line2D([], [], marker=MARKERS[ti], color=BLUE,
                                  ls="none", ms=4.5, label=f"trial {ti + 1}")
                           for ti in range(n_tr)]
                handles.append(Line2D([], [], color=AQUA, lw=2.2,
                                      label="mean error"))
                ax.legend(handles=handles, fontsize=6.8, frameon=False,
                          loc="lower left", handletextpad=0.2,
                          borderaxespad=0.1, labelspacing=0.4)
        else:
            ax.set_yticklabels([])
fig.suptitle("Hyperparameter Bias on FLOP-Estimation Error",
             fontsize=9.5, color=INK, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(OUT, "fig6_bias_by_axis.png"))
plt.close(fig)

# ── Figure 7: estimated vs true TFLOPs, 3-param, spread over 200 splits ──────
# For each of the 21 fp32 frontier workloads: the median and min–max of its
# held-out 3-param estimates collected in the figure-2 loop above. Every
# estimate is from a fit whose train split (14 of the other 20 workloads)
# excluded that workload; the whisker shows how much the estimate moves as the
# train split changes.
gt7 = np.array([r["ground_truth_tf"] for r in FRONTIER])
med7 = np.array([np.median(e) for e in split_ests])
lo7 = np.array([np.min(e) for e in split_ests])
hi7 = np.array([np.max(e) for e in split_ests])

fig, ax = plt.subplots(figsize=(4.8, 4.6), dpi=200)
lim = (70, 190)
xs = np.array(lim)
ax.fill_between(xs, xs * 0.9, xs * 1.1, color=GRID, alpha=0.55, zorder=1,
                linewidth=0, label="±10% error")
ax.plot(xs, xs, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2)
ax.errorbar(gt7, med7, yerr=(med7 - lo7, hi7 - med7), fmt="none",
            ecolor=BLUE, elinewidth=1.1, alpha=0.75, capsize=2.2,
            capthick=1.1, zorder=3, label="min–max error")
ax.scatter(gt7, med7, s=22, color=BLUE, zorder=4, linewidths=0,
           label="median held-out estimate")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect("equal")
ax.set_xlabel("Ground-Truth TFLOPs")
ax.set_ylabel("Estimated TFLOPs")
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
ax.set_title("Estimated vs. True Training TFLOPs",
             fontsize=9.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_est_vs_truth.png"))
plt.close(fig)

# ── Figure 7 (kept alternate): 2-param vs 3-param single-LOO scatter ──────────
# The previous fig7: each workload's estimate from the leave-one-out fit that
# excluded it, for both estimators side by side.
pool7 = trials6[0]["pool"]          # trial 1 = eval_results_v2
gt7a = np.array([r["ground_truth_tf"] for r in pool7])
est2 = gt7a * (1 + np.array([r["signed_loo"] for r in pool7]) / 100)
est3 = gt7a * (1 + np.array([r["signed_loo_emc"] for r in pool7]) / 100)

fig, ax = plt.subplots(figsize=(4.8, 4.6), dpi=200)
ax.fill_between(xs, xs * 0.9, xs * 1.1, color=GRID, alpha=0.55, zorder=1,
                linewidth=0)
ax.plot(xs, xs, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2)
ax.scatter(gt7a, est2, s=26, color=BLUE, zorder=4, linewidths=0,
           label="2-param estimate (excludes bandwidth util %)")
ax.scatter(gt7a, est3, s=30, facecolors="none", edgecolors=AQUA, marker="s",
           linewidths=1.2, zorder=3,
           label="3-param estimate (includes bandwidth util %)")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect("equal")
ax.set_xlabel("Ground-Truth TFLOPs")
ax.set_ylabel("Estimated TFLOPs")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title("Estimated vs. True Training TFLOPs",
             fontsize=9.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_alt_2param_3param.png"))
plt.close(fig)

print("stability fail rate (3p):", fail, "%")
print("fig6 trials:", TRIAL_FILES)
pt3 = [{r["label"]: r["signed_loo_emc"] for r in t["pool"]} for t in trials6]
for i in range(len(pt3)):
    for j in range(i + 1, len(pt3)):
        common = sorted(set(pt3[i]) & set(pt3[j]))
        xa = np.array([pt3[i][k] for k in common])
        ya = np.array([pt3[j][k] for k in common])
        print(f"3p consistency T{i+1} vs T{j+1}: r="
              f"{float(np.corrcoef(xa, ya)[0, 1]):+.2f} same-sign "
              f"{float((xa * ya > 0).mean() * 100):.0f}%")
print("figures written to", OUT)
