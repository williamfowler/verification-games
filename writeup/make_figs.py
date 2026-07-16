"""Generate the three Results figures for the write-up from real experiment data."""
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/home/jetson/verification-games"
OUT = os.path.join(REPO, "writeup")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "power_calibration"))

import detect_flops
from eval_power_monitor import (fit_active_energy_model, make_split, score,
                                err_stats, load_bias_trials, cfg_val)

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


ALL_RECS, FRONTIER = load_fp32_frontier()

# ── Figure 1: per-workload signed error (shipped constants) ───────────────────
rows = []
for r in FRONTIER:
    est = detect_flops.estimate_tflops(r["net_energy_j"], r["duration_s"])
    cfg = r["config"]
    key = (cfg["d_model"], cfg["batch_size"], cfg["seq_len"],
           cfg["num_layers"], r["label"])
    rows.append((key, r["label"], cfg["d_model"],
                 (est - r["ground_truth_tf"]) / r["ground_truth_tf"] * 100))
rows.sort(key=lambda t: t[0])
labels = [t[1] for t in rows]
dmods = [t[2] for t in rows]
errs = [t[3] for t in rows]

fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
ax.bar(range(len(errs)), errs, width=0.62, color=BLUE, zorder=3)
ax.axhline(0, color=BASE, lw=1, zorder=2)
for y in (10, -10):
    ax.axhline(y, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
ax.text(-0.4, 10.4, "±10% target", ha="left", va="bottom",
        color=INK2, fontsize=8)
# separators + captions between d_model families
bounds = [i for i in range(1, len(dmods)) if dmods[i] != dmods[i - 1]]
for i in bounds:
    ax.axvline(i - 0.5, color=GRID, lw=0.8, zorder=1)
starts = [0] + bounds
ends = bounds + [len(dmods)]
for s, e in zip(starts, ends):
    ax.text((s + e - 1) / 2, 12.3, f"d{dmods[s]}", ha="center", fontsize=8,
            color=INK2)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
ax.set_ylabel("estimation error (%)")
ax.set_ylim(-14, 14.5)
ax.set_xlim(-0.7, len(errs) - 0.3)
ax.set_title("FLOP estimation error per workload, grouped by model size\n"
             "(21 FP32 frontier configs, deployed 2-parameter estimator)",
             fontsize=9.5, color=INK, loc="left")
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_per_workload_error.png"))
plt.close(fig)

# ── Figure 2: held-out max error across 200 resampled splits ─────────────────
base = 1000
maxerrs = []
for i in range(200):
    train, test, _ = make_split(FRONTIER, "random", base + i)
    e, pov = fit_active_energy_model(train)
    score(test, e, pov)
    m, _ = err_stats(test)
    maxerrs.append(m)
maxerrs = np.array(maxerrs)
fail = float((maxerrs >= 10).mean() * 100)

fig, ax = plt.subplots(figsize=(5.4, 3.0), dpi=200)
bins = np.arange(3.0, 13.0, 0.5)
ax.hist(maxerrs, bins=bins, color=BLUE, zorder=3)
ax.axvline(10, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=4)
ax.text(10.15, ax.get_ylim()[1] * 0.93, "10% target", color=INK2, fontsize=8,
        va="top")
ax.axvline(8.95, color=AQUA, lw=1.6, zorder=4)
ax.text(8.8, ax.get_ylim()[1] * 0.93, "reported split (8.95%)", color=INK2,
        fontsize=8, va="top", ha="right")
ax.set_xlabel("held-out max error (%)")
ax.set_ylabel("splits (of 200)")
ax.set_title(f"Held-out error across 200 resampled train/test splits\n"
             f"({fail:.1f}% of splits exceed the 10% target)",
             fontsize=9.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_split_stability.png"))
plt.close(fig)

# ── Figure 3: evasion panels ──────────────────────────────────────────────────
# (a) adversarial probe numbers (adversarial_results.txt)
spoof = {"gt": 1.78, "e2": 8.26, "emc": -96.05}
dense = {"gt": 152.08, "e2": 158.00, "emc": 177.71}

# (b) monitor-read fraction per precision (v2 records, deployed constants)
prec_ratio = {}
for prec in ("fp32", "tf32", "fp16", "bf16"):
    rs = [r for r in ALL_RECS
          if r["config"].get("precision", "fp32") == prec
          and (prec == "fp32") == (r in FRONTIER or r["avg_gpu_pct"] < 80)]
    rs = [r for r in ALL_RECS if r["config"].get("precision", "fp32") == prec]
    if prec == "fp32":
        rs = FRONTIER
    ratios = []
    for r in rs:
        est = detect_flops.estimate_tflops(r["net_energy_j"], r["duration_s"])
        ratios.append(est / r["ground_truth_tf"])
    prec_ratio[prec] = float(np.mean(ratios))

fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1), dpi=200,
                           gridspec_kw={"width_ratios": [1.15, 1]})
x = np.arange(2)
w = 0.26
gvals = [spoof["gt"], dense["gt"]]
e2vals = [spoof["e2"], dense["e2"]]
emvals = [spoof["emc"], dense["emc"]]
a.bar(x - w, gvals, w, color=MUTED, label="ground truth", zorder=3)
a.bar(x, e2vals, w, color=BLUE, label="2-param estimate", zorder=3)
a.bar(x + w, emvals, w, color=AQUA, label="EMC estimate", zorder=3)
a.axhline(0, color=BASE, lw=1)
for xi, v in [(x[0] - w, gvals[0]), (x[0], e2vals[0]), (x[0] + w, emvals[0]),
              (x[1] - w, gvals[1]), (x[1], e2vals[1]), (x[1] + w, emvals[1])]:
    a.text(xi, v + (6 if v >= 0 else -6), f"{v:.0f}" if abs(v) > 3 else f"{v:.1f}",
           ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5,
           color=INK2)
a.set_xticks(x)
a.set_xticklabels(["memory spoof\n(~0 true FLOPs)", "cache-resident matmul\n(control)"],
                  fontsize=8)
a.set_ylabel("TFLOPs")
a.set_ylim(-135, 215)
a.legend(fontsize=7.5, frameon=False, loc="upper left")
a.set_title("(a) Adversarial probe — EMC byte term\nflags the spoof",
            fontsize=9.5, color=INK, loc="left")
a.grid(axis="x", visible=False)

precs = ["fp32", "tf32", "fp16", "bf16"]
vals = [prec_ratio[p] for p in precs]
b.bar(range(4), vals, width=0.55, color=BLUE, zorder=3)
b.axhline(0, color=BASE, lw=1, zorder=2)
b.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
b.text(3.4, 1.02, "perfect", ha="right", va="bottom", color=INK2, fontsize=8)
for i, v in enumerate(vals):
    b.text(i, v + 0.04 if v >= 0 else v - 0.04, f"{v:.2f}", ha="center",
           va="bottom" if v >= 0 else "top", fontsize=8, color=INK2)
b.text(2.48, 0.52, "reads as \u2264 0: run missed\nentirely (also drops below\nthe 80% util gate)",
       ha="center", fontsize=7, color=INK2)
b.set_xticks(range(4))
b.set_xticklabels(precs, fontsize=8.5)
b.set_ylim(-0.75, 1.3)
b.set_ylabel("estimated / true FLOPs")
b.set_title("(b) Precision evasion \u2014 estimate as a\nfraction of true FLOPs",
            fontsize=9.5, color=INK, loc="left")
b.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_evasion.png"))
plt.close(fig)

# ── Figure 6: signed LOO error by hyperparameter axis (all 25W trials) ────────
# One dot per fp32 frontier run (leave-one-out signed error, computed by
# eval_power_monitor.load_bias_trials — the same numbers as bias_report.txt),
# marker shape = trial, aqua dash = level mean. Positive = overestimate.
TRIAL_FILES = [f for f in ("eval_results_v2_records.json",
                           "eval_results_25w_trial2_records.json",
                           "eval_results_25w_trial3_records.json")
               if os.path.exists(os.path.join(REPO, f))]
trials6 = load_bias_trials([os.path.join(REPO, f) for f in TRIAL_FILES])
n_tr = len(trials6)

PANELS = [  # (config key, panel title, row)
    ("d_model", "model width\n(d_model)", 0),
    ("seq_len", "sequence\nlength", 0),
    ("dim_feedforward", "FFN width", 0),
    ("batch_size", "batch\nsize", 1),
    ("num_layers", "layers", 1),
    ("nhead", "attention\nheads", 1),
    ("optimizer", "optimizer", 1),
]
levels6 = {key: sorted({cfg_val(r, key) for t in trials6 for r in t["pool"]})
           for key, _title, _row in PANELS}
rows = {0: [p for p in PANELS if p[2] == 0], 1: [p for p in PANELS if p[2] == 1]}

fig = plt.figure(figsize=(7.2, 5.4), dpi=200)
gs = fig.add_gridspec(2, 1, hspace=0.42)
MARKERS = ["o", "s", "^"]
label_means = {"d_model", "dim_feedforward"}   # selective direct labels
for row_i, panels in rows.items():
    widths = [len(levels6[k]) for k, _t, _r in panels]
    sub = gs[row_i].subgridspec(1, len(panels), width_ratios=widths, wspace=0.14)
    for pi, (key, title, _r) in enumerate(panels):
        ax = fig.add_subplot(sub[pi])
        lv = levels6[key]
        for ti, t in enumerate(trials6):
            xs, ys = [], []
            for r in t["pool"]:
                xs.append(lv.index(cfg_val(r, key))
                          + (ti - (n_tr - 1) / 2) * 0.22)
                ys.append(r["signed_loo"])
            ax.scatter(xs, ys, s=16, marker=MARKERS[ti], color=BLUE,
                       alpha=0.65, linewidths=0, zorder=3,
                       label=f"trial {ti + 1}" if (row_i, pi) == (0, 0) else None)
        for i, v in enumerate(lv):
            errs6 = [r["signed_loo"] for t in trials6 for r in t["pool"]
                     if cfg_val(r, key) == v]
            m = float(np.mean(errs6))
            ax.hlines(m, i - 0.34, i + 0.34, color=AQUA, lw=2.2, zorder=4)
            if key in label_means and abs(m) >= 3:   # label only the biased levels
                ax.text(i, m + (1.1 if m >= 0 else -1.1), f"{m:+.1f}",
                        ha="center", va="bottom" if m >= 0 else "top",
                        fontsize=6.8, color=INK2, zorder=5)
        ax.axhline(0, color=BASE, lw=1, zorder=2)
        ax.set_xticks(range(len(lv)))
        ax.set_xticklabels([str(v) for v in lv], fontsize=7,
                           rotation=45 if key == "dim_feedforward" else 0)
        ax.set_xlim(-0.6, len(lv) - 0.4)
        ax.set_ylim(-13, 13)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=8, color=INK2)
        if pi == 0:
            ax.set_ylabel("signed error (%)\n↑ over   ↓ under", fontsize=8)
            if row_i == 0 and n_tr > 1:
                ax.legend(fontsize=6.8, frameon=False, loc="upper right",
                          handletextpad=0.1, borderaxespad=0.1)
        else:
            ax.set_yticklabels([])
fig.suptitle("Signed FLOP-estimation error by workload hyperparameter\n"
             f"(leave-one-out; {sum(len(t['pool']) for t in trials6)} fp32 "
             f"frontier runs, {n_tr} independent 25 W sweeps; "
             "green dash = level mean)",
             fontsize=9.5, color=INK, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(os.path.join(OUT, "fig6_bias_by_axis.png"))
plt.close(fig)

print("stability fail rate:", fail, "%")
print("precision ratios:", {k: round(v, 3) for k, v in prec_ratio.items()})
print("fig6 trials:", TRIAL_FILES)
print("figures written to", OUT)
