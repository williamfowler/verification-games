#!/usr/bin/env python3
"""
make_new_report_figs.py — the report figures not yet generated elsewhere:
  fig_s4_bytes.png     (Figure 7) — NVLink & DRAM bytes vs batch under S4 inflation
  fig_s5_power_time.png(Figure 8) — GPU power over time at each S5 power cap vs stock

    python3 writeup/make_new_report_figs.py
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
OUT = os.path.join(REPO, "writeup")
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE, BLUE, AQUA, AMBER, RED, PURPLE = ("#e1e0d9", "#c3c2b7", "#7b9fd4",
                                              "#7fbfa4", "#e3bc70", "#d29393", "#a795d4")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 13,
})


# ── Figure 7: S4 NVLink & DRAM bytes vs batch ───────────────────────────────
def fig_s4_bytes():
    trials = sorted(glob.glob(os.path.join(REPO, "red_team/red_v3_trial*_records.json")))
    by = {}
    for p in trials:
        for r in json.load(open(p))["records"]:
            if r["config"].get("strategy") != "S4_batch":
                continue
            b = r["config"]["batch_size"]
            by.setdefault(b, {"nv": [], "dram": []})
            by[b]["nv"].append((r.get("nvlink_total_bytes") or 0) / 1e12)   # TB
            by[b]["dram"].append(r.get("tb_moved") or 0)                    # TB
    batches = sorted(by)
    nv = [float(np.median(by[b]["nv"])) for b in batches]
    dram = [float(np.median(by[b]["dram"])) for b in batches]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 4.2), dpi=200)
    for ax, ys, col, ylab, title, fmt in [
            (axL, nv, PURPLE, "NVLink all-reduce traffic (TB, both GPUs)",
             f"NVLink traffic collapses {nv[0]/nv[-1]:.0f}× (∝ 1 / batch)", "{:.2f}"),
            (axR, dram, AQUA, "DRAM traffic (TB moved)",
             f"DRAM traffic falls only {dram[0]/dram[-1]:.1f}×", "{:.0f}")]:
        ax.plot(batches, ys, "-o", color=col, lw=2.2, ms=7, zorder=4)
        for x, y in zip(batches, ys):
            ax.annotate(fmt.format(y), (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, color=col)
        ax.set_xscale("log", base=2); ax.set_xticks(batches)
        ax.set_xticklabels([str(b) for b in batches]); ax.minorticks_off()
        ax.set_xlabel("batch size  (ground truth held constant: batch × steps = 15360)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9.2, color=INK, loc="left")
        ax.set_ylim(0, max(ys) * 1.2)
    fig.suptitle("S4 batch-inflation: the same FLOPs run in fewer, larger steps, so both byte "
                 "signals fall — but NVLink collapses far faster than DRAM,\nso the NVLink / DRAM "
                 "arithmetic-intensity ratio drops below the benign floor (what the consistency "
                 "gate keys on)", fontsize=8.2, color=MUTED, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "old_figures", "fig_s4_bytes.png"))
    plt.close(fig)
    print("wrote writeup/figure_8.png")


# ── Figure 8: S5 GPU power over time at each cap ────────────────────────────
def fig_s5_power_time():
    d = json.load(open(os.path.join(REPO, "red_team/red_s5_v3_records.json")))
    recs = {r["config"]["power_cap_w"]: r for r in d["records"]}
    fig, ax = plt.subplots(figsize=(8.4, 4.7), dpi=200)

    def smooth(y, k=7):
        y = np.asarray(y, float)
        if len(y) < k:
            return y
        pad = np.pad(y, (k // 2, k // 2), mode="edge")
        return np.convolve(pad, np.ones(k) / k, mode="valid")[:len(y)]

    show = [(300, INK2, "300 W (stock)"), (200, AQUA, "200 W cap"), (100, RED, "100 W cap")]
    for cap, col, lbl in show:
        r = recs.get(cap)
        ps = r.get("power_samples") or []
        if not ps:
            continue
        t0 = ps[0][0]
        t = np.array([s[0] - t0 for s in ps])
        w = smooth([s[1] / 1000.0 for s in ps])
        avg = np.mean([s[1] / 1000.0 for s in ps])
        gt = r["ground_truth_tf"]
        ax.plot(t, w, lw=2.0, color=col, alpha=0.9, zorder=4,
                label=f"{lbl}   ·   {r['duration_s']:.0f} s, avg {avg:.0f} W, {r['net_energy_j']/gt:.2f} J/TFLOP")
    ax.set_xlabel("time since workload start (s)   ·   GPUs warmed to steady state before the sweep")
    ax.set_ylabel("total GPU board power, both V100s (W)   ·   7-sample rolling mean")
    ax.set_title("Power-capping does essentially nothing to this workload: it draws ~200 W (below "
                 "the caps),\nso power, runtime, and energy-per-FLOP are the same at every cap — and "
                 "so is the estimate (−13 to −16%, in-band)", fontsize=8.0, color=INK, loc="left")
    ax.legend(fontsize=7.4, frameon=False, loc="upper right", title="all traces overlap:",
              title_fontsize=7.4)
    ax.set_ylim(0, 480)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "old_figures", "fig_s5_power_time.png"))
    plt.close(fig)
    print("wrote writeup/figure_9.png")


# ── Figure 5: S2 throttle — util & power over time vs benign ────────────────
def fig_throttle_time():
    d = json.load(open(os.path.join(REPO, "writeup", "throttle_ts.json")))
    t = np.array(d["t"]); util = np.array(d["util0"]); pw = np.array(d["power_w"])
    ph = np.array(d["phase"]); marks = d["marks"]

    def window(phase):
        m = marks.get(phase, {})
        s = m.get("start"); e = m.get("end")
        sel = (ph == phase)
        if s is not None and e is not None:
            sel &= (t >= s) & (t <= e)
            return t[sel] - s, util[sel], pw[sel]
        return t[sel] - t[sel][0], util[sel], pw[sel]

    bn_t, bn_u, bn_p = window("none")
    th_t, th_u, th_p = window("throttle")

    ZOOM = 12.0     # show the first 12 s so the throttle sawtooth is legible
    fig, (axU, axP) = plt.subplots(2, 1, figsize=(8.4, 6.2), dpi=200, sharex=True)
    for ax, bn, th, ylab in [(axU, bn_u, th_u, "GPU utilization (%)"),
                             (axP, bn_p, th_p, "total board power (W)")]:
        # legend stats over the FULL run; the plot is zoomed for legibility
        lbn = f"benign   ·   full-run avg {np.mean(bn):.0f}, peak {np.max(bn):.0f}"
        lth = f"S2 throttle   ·   full-run avg {np.mean(th):.0f}, peak {np.max(th):.0f}"
        ax.plot(bn_t, bn, color=INK2, lw=1.9, zorder=4, label=lbn)
        ax.plot(th_t, th, color=RED, lw=1.5, zorder=3, label=lth)
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7.6, frameon=False, loc="lower right")
        ax.set_ylim(0, None); ax.set_xlim(0, ZOOM)
    axU.axhline(80, color=AMBER, lw=1.1, ls=(0, (4, 3)), zorder=2)
    axU.annotate("80% frontier gate", xy=(ZOOM * 0.99, 81), fontsize=7.2, color=AMBER,
                 va="bottom", ha="right")
    axP.set_xlabel(f"time since workload start (s)  ·  first {ZOOM:.0f} s shown; "
                   f"legend averages are over the full run")
    axU.set_title("S2 throttling lowers AVERAGE utilization, but every compute step still "
                  "spikes to 100% —\nthe daemon gates on PEAK util, so the run stays flagged "
                  "frontier (attack fails)", fontsize=8.6, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "old_figures", "fig_throttle_time.png"))
    plt.close(fig)
    print("wrote writeup/figure_6.png")


# ── Figure 7: signed error vs hyperparameter, ACROSS estimators (2p/4p/MLP) ──
def fig_estimator_comparison():
    """S4 batch + S3 nhead signed error for the 2-param, 4-param, and MLP estimators
    on the same axes — the red-team's estimator comparison. Data: redteam_nn_scores.json
    (frozen estimators scored on the GT-preserving red records). MLP = residual
    (physics-anchored, the recommended one); the pure MLP is fooled even harder."""
    d = json.load(open(os.path.join(REPO, "redteam_nn_scores.json")))
    # benign band = middle 50% (25th-75th pct) of the 4-param estimator's signed
    # errors on the benign held-out workloads (was min/max before).
    nn = json.load(open(os.path.join(REPO, "nn_estimator_results.json")))
    _gt = nn["gt"]; _e4 = nn["per_label_median_est"]["4-param"]
    _errs = [(_e4[l] - _gt[l]) / _gt[l] * 100 for l in _e4 if l in _gt]
    band_lo, band_hi = np.percentile(_errs, [25, 75])
    # (group, key, title, xlabel, x-axis type). x_type: 'log2' or 'lin_inv' (cap: stock left)
    panels = [("S3_atypical", "nhead", "Atypical n_head",
               "n_head", "log2"),
              ("S4_batch", "batch", "Larger Batch Sizes",
               "Batch Size", "log2"),
              ("S5_powercap", "cap_W", "Power Cap",
               "Power Cap (W)", "lin_inv")]
    series = [("err_2p", BLUE, "s", "1-input"),
              ("err_4p", AMBER, "D", "3-input"),
              ("err_residMLP", RED, "^", "MLP")]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), dpi=200)
    for i, (ax, (grp, key, title, xlab, xtype)) in enumerate(zip(axes, panels)):
        rev = (xtype == "lin_inv")
        rows = sorted(d["groups"][grp], key=lambda r: r[key], reverse=rev)
        xs = [r[key] for r in rows]
        ax.axhspan(band_lo, band_hi, color=GRID, alpha=0.5, zorder=1,
                   label="benign error (middle 50%)")
        ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
        for kk, col, mk, lab in series:
            ax.plot(xs, [r[kk] for r in rows], marker=mk, color=col, lw=1.9, ms=7,
                    zorder=4, label=lab)
        if xtype == "log2":
            ax.set_xscale("log", base=2)
        ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs]); ax.minorticks_off()
        if xtype == "lin_inv":
            ax.invert_xaxis()
        ax.set_xlabel(xlab, fontsize=13.5)
        if i == 0:
            ax.set_ylabel("Signed Error (%)   ↓ = Under-Report", fontsize=13.5)
        if i == 2:
            ax.legend(fontsize=11.5, frameon=False, loc="lower left")
        ax.set_title(title, fontsize=15, color=INK, loc="left", pad=10)
        ax.set_ylim(-48, 25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure_9.png"))
    plt.close(fig)
    print("wrote writeup/figure_9.png  (3-panel estimator comparison: S3 + S4 + S5)")


# ── Figure 4: deployed MLP estimator — strategies vs benign at same GT ───────
def fig4_mlp_vs_benign():
    """Est-vs-truth overview under the DEPLOYED (residual) MLP estimator: benign
    held-out cloud + the point-estimate strategies S3/S4/S5 (est = GT·(1+err)). Shows
    that the more-accurate MLP is pulled further below the truth line by every lever.
    S1/S2 are attribution attacks (no point estimate) → see figure_5."""
    nn = json.load(open(os.path.join(REPO, "nn_estimator_results.json")))
    est = nn["per_label_median_est"]["residual-MLP"]; gtb = nn["gt"]
    labs = [l for l in est if l in gtb]
    bgt = np.array([gtb[l] for l in labs]); best = np.array([est[l] for l in labs])

    sc = json.load(open(os.path.join(REPO, "redteam_nn_scores.json")))["groups"]
    style = {"S4_batch": (AMBER, "D", "S4 batch-inflation", "batch"),
             "S3_atypical": (BLUE, "s", "S3 atypical nhead", "nhead"),
             "S5_powercap": (AQUA, "^", "S5 power-cap", "cap_W")}
    strat_pts = {}
    for grp, (col, mk, lab, key) in style.items():
        strat_pts[grp] = [(r["gt"], r["gt"] * (1 + r["err_residMLP"] / 100.0)) for r in sc[grp]]

    lim = (0, max(bgt.max(), best.max(),
                  max(e for v in strat_pts.values() for _, e in v)) * 1.06)
    xs = np.array(lim)
    fig, ax = plt.subplots(figsize=(6.6, 5.8), dpi=200)
    ax.fill_between(xs, xs * 0.8, xs * 1.2, color=GRID, alpha=0.45, lw=0, zorder=1, label="±20%")
    ax.fill_between(xs, xs * 0.9, xs * 1.1, color=BASE, alpha=0.5, lw=0, zorder=1, label="±10%")
    ax.plot(xs, xs, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.scatter(bgt, best, s=16, color=BASE, alpha=0.45, linewidths=0, zorder=3,
               label="benign held-out (88 workloads)")
    for grp, (col, mk, lab, key) in style.items():
        g = np.array([p[0] for p in strat_pts[grp]]); e = np.array([p[1] for p in strat_pts[grp]])
        err = (np.median(e) - np.median(g)) / np.median(g) * 100
        ax.scatter(g, e, s=95, color=col, marker=mk, edgecolor=INK, linewidths=0.8, zorder=6,
                   label=f"{lab}  ({len(g)} configs, median {err:+.0f}%)")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("Ground-Truth Aggregate TFLOPs")
    ax.set_ylabel("Deployed MLP estimate  (TFLOPs)")
    ax.legend(fontsize=7.2, frameon=False, loc="upper left")
    ax.set_title("Under the deployed (most-accurate) MLP estimator, the atypical-geometry (S3) and\n"
                 "batch-inflation (S4) attacks pull the estimate well below truth; power-cap (S5) "
                 "stays in-band", fontsize=8.6, color=INK, loc="left")
    fig.text(0.5, -0.01, "S1 split / S2 throttle are attribution attacks (no point estimate) — "
             "see figure_5. Compare estimators head-to-head in figure_7.",
             fontsize=6.2, color=MUTED, ha="center", va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "old_figures", "fig_mlp_vs_benign.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote writeup/figure_4.png  (MLP-deployed overview)")


# ── Figure 13: change in median error from benign → each strategy, per estimator ─
def fig_error_change_bars():
    """Grouped bars: for each estimator-based strategy (S3/S4/S5), the change in each
    estimator's median SIGNED error going from benign to that strategy (percentage
    points; ↓ = the strategy pushes the estimate further below truth). Shows the MLP
    is pushed hardest on every strategy. Benign baseline = held-out signed median."""
    nn = json.load(open(os.path.join(REPO, "nn_estimator_results.json")))
    gt = nn["gt"]; estd = nn["per_label_median_est"]
    ESTS = [("2-param", "2-param", "err_2p", BLUE),
            ("4-param", "4-param", "err_4p", AMBER),
            ("MLP", "residual-MLP", "err_residMLP", RED)]
    benign = {}
    for nm, lk, _k, _c in ESTS:
        e = estd[lk]
        benign[nm] = float(np.median([(e[l] - gt[l]) / gt[l] * 100 for l in e if l in gt]))
    sc = json.load(open(os.path.join(REPO, "redteam_nn_scores.json")))["groups"]
    # strongest ATTACK setting per strategy = the non-parent config that under-reports
    # most (by the deployed MLP). Parents (calibration reference): batch 8 / nhead 8 / 300 W.
    spec = [("S3_atypical", "nhead", 8, "Atypical Attention Heads"),
            ("S4_batch", "batch", 8, "Larger Batch Sizes"),
            ("S5_powercap", "cap_W", 300, "Power Cap")]
    groups = []
    delta = {nm: [] for nm, *_ in ESTS}
    for g, key, parent, lab in spec:
        atk = [r for r in sc[g] if r[key] != parent]
        row = min(atk, key=lambda r: r["err_residMLP"])       # most under-report by MLP
        groups.append((g, lab))
        for nm, _lk, ek, _c in ESTS:
            delta[nm].append(row[ek] - benign[nm])

    fig, ax = plt.subplots(figsize=(9.6, 5.6), dpi=200)
    x = np.arange(len(groups)); w = 0.26
    for i, (nm, _lk, _k, col) in enumerate(ESTS):
        xs = x + (i - 1) * w
        bars = ax.bar(xs, delta[nm], width=w, color=col, zorder=3,
                      label={"2-param": "1-input", "4-param": "3-input",
                             "MLP": "MLP"}[nm])
        for xb, v in zip(xs, delta[nm]):
            ax.annotate(f"{v:+.0f}", (xb, v), textcoords="offset points",
                        xytext=(0, -15 if v < 0 else 5), ha="center", fontsize=13,
                        color=INK, fontweight="bold")
    ax.axhline(0, color=INK2, lw=1.0, zorder=2)
    ax.annotate("benign baseline", xy=(len(groups) - 0.55, 0), xytext=(0, 5),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=11.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([g[1] for g in groups], fontsize=13)
    ax.set_ylabel("Change In Error (lower is worse)", fontsize=13)
    ax.set_ylim(min(min(v) for v in delta.values()) - 6, 6)
    ax.legend(fontsize=12, frameon=False, loc="lower left", ncol=1)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figure_7.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote writeup/figure_7.png  (error-change bar chart)")


if __name__ == "__main__":
    fig_s4_bytes()
    fig_s5_power_time()
    if os.path.exists(os.path.join(REPO, "writeup", "throttle_ts.json")):
        fig_throttle_time()
    fig_estimator_comparison()   # now 3-panel: S4 + S3 + S5
    fig4_mlp_vs_benign()
    fig_error_change_bars()
