"""
make_figs_redteam_extra.py — figures for the red-team strategies the two existing
figures (fig_redteam_signed / fig_redteam_evasion, which cover S3 nhead + S4 batch)
leave unillustrated:

  fig_redteam_powercap.png  — S5 DVFS power-cap sweep: J/TFLOP and 4-param error
                              vs power cap, with the ~250 W energy-optimal sweet spot.
  fig_redteam_sessions.png  — S1 split / S2 throttle live-daemon attribution: the
                              session-fragmentation timeline (1 vs 4 vs 1 sessions).
  fig_redteam_gates.png     — the consistency-gate shield: the arith_intensity ratio
                              per run vs the benign floor, showing the gate layer
                              flags S4 batch-inflation the point estimate misses.

Same frozen Phase I calibration and house style as red_team/score_redteam.py.

    python3 writeup/make_figs_redteam_extra.py
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "power_calibration"))
sys.path.insert(0, os.path.join(REPO_ROOT, "red_team"))

import consistency_gates as cg
from score_redteam import phase1_calibration, est4, signed_pct
from efficiency import _default_phase1_paths, efficiency_ratio, BUDGET
from eval_power_monitor import load_records, valid, is_frontier

OUT = os.path.join(REPO_ROOT, "writeup")
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE, BLUE, AMBER, RED, AQUA = ("#e1e0d9", "#c3c2b7", "#2a78d6", "#e69f00",
                                      "#d1495b", "#1baf7a")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9,
})


def load(path):
    return json.load(open(os.path.join(REPO_ROOT, path)))


# ── Fig 1: S5 power-cap (DVFS) sweep ────────────────────────────────────────
def fig_powercap(cal):
    d = load("red_team/red_s5_records.json")
    recs = sorted(d["records"], key=lambda r: r["config"]["power_cap_w"])
    caps = [r["config"]["power_cap_w"] for r in recs]
    jpf = [r["net_energy_j"] / r["ground_truth_tf"] for r in recs]
    err = [signed_pct(est4(r, cal), r["ground_truth_tf"]) for r in recs]
    eff = [efficiency_ratio(r["ground_truth_tf"], r["duration_s"], cal["r_benign"])
           for r in recs]
    b4 = cal["band4"]
    sweet = int(np.argmin(jpf))               # energy-optimal operating point
    stock = caps.index(300) if 300 in caps else int(np.argmax(caps))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 4.3), dpi=200)

    # left: energy per TFLOP vs cap — the non-monotone V with a sweet spot
    axL.plot(caps, jpf, "-o", color=AQUA, lw=2.0, ms=6, zorder=4)
    axL.scatter([caps[sweet]], [jpf[sweet]], s=150, facecolor="none",
                edgecolor=RED, lw=1.8, zorder=5)
    axL.annotate(f"energy-optimal\n~{caps[sweet]} W  ({jpf[sweet]:.2f} J/TFLOP)",
                 xy=(caps[sweet], jpf[sweet]), xytext=(caps[sweet] + 25, jpf[sweet] + 0.18),
                 fontsize=7.2, color=RED, va="bottom",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    axL.axhline(jpf[stock], color=INK2, lw=0.9, ls=(0, (2, 3)), zorder=2)
    axL.annotate(f"stock 300 W = {jpf[stock]:.2f} J/TFLOP", xy=(200, jpf[stock]),
                 fontsize=6.6, color=INK2, va="bottom", ha="center")
    axL.set_xlabel("power cap  (nvidia-smi -pl, W)")
    axL.set_ylabel("energy per TFLOP  (J/TFLOP)   ↓ = cheaper per real FLOP")
    axL.set_title("S5 power-cap: a sub-stock energy sweet spot", fontsize=9.2,
                  color=INK, loc="left")
    axL.invert_xaxis()   # capping harder → rightward, matching "more throttled"

    # right: 4-param signed error vs cap, benign band shaded
    axR.axhspan(b4["lo"], b4["hi"], color=GRID, alpha=0.45, zorder=1,
                label="benign band (4p)")
    axR.axhline(0, color=MUTED, lw=0.8, zorder=2)
    axR.plot(caps, err, "-o", color=AMBER, lw=2.0, ms=6, zorder=4,
             label="4-param signed error")
    axR.axhline(err[stock], color=INK2, lw=0.9, ls=(0, (2, 3)), zorder=2)
    axR.annotate(f"stock baseline {err[stock]:+.0f}%", xy=(caps[-1], err[stock]),
                 fontsize=6.6, color=INK2, va="bottom", ha="right")
    axR.scatter([caps[sweet]], [err[sweet]], s=150, facecolor="none",
                edgecolor=RED, lw=1.8, zorder=5)
    axR.annotate(f"{err[sweet]:+.0f}%  (~{err[sweet]-err[stock]:+.0f} pp vs stock)",
                 xy=(caps[sweet], err[sweet]), xytext=(caps[sweet] + 25, err[sweet] - 4),
                 fontsize=7.2, color=RED, va="top",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
    axR.set_xlabel("power cap  (nvidia-smi -pl, W)")
    axR.set_ylabel("4-param signed error vs GT (%)   ↓ = under-report")
    axR.set_title("…and the estimate reads further under (all eff ≤0.7×, legal)",
                  fontsize=9.2, color=INK, loc="left")
    axR.legend(fontsize=7.2, frameon=False, loc="lower right")
    axR.invert_xaxis()

    fig.suptitle("S5 DVFS power-capping (root): bounded, non-monotone under-report — "
                 "reliable quantity is the delta across caps, not the absolute",
                 fontsize=8.4, color=MUTED, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_redteam_powercap.png"))
    plt.close(fig)
    print("wrote writeup/fig_redteam_powercap.png")


# ── Fig 2: S1 split / S2 throttle live-daemon session timelines ──────────────
def fig_sessions():
    src = {"none": "red_team_old/live_none.json",
           "split": "red_team_old/live_split.json",
           "throttle": "red_team_old/live_throttle.json"}
    runs = {}
    for k, p in src.items():
        d = load(p)
        runs[k] = d[0] if isinstance(d, list) else d

    def t0(run):
        return min(np.datetime64(s["start_time"]) for s in run["sessions"])

    rows = [
        ("none (benign)\ncontrol", runs["none"], BLUE,
         "honest baseline"),
        ("S1 split\n(idle gaps > stop window)", runs["split"], RED,
         "attribution EVADED (4 unlinkable jobs)"),
        ("S2 throttle\n(micro-sleep)", runs["throttle"], AMBER,
         "peak-util still 100% → stays flagged; gate NOT evaded"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.2), dpi=200)
    ylabels = []
    for i, (name, run, col, note) in enumerate(rows):
        base = t0(run)
        for s in run["sessions"]:
            start = (np.datetime64(s["start_time"]) - base) / np.timedelta64(1, "s")
            dur = s["duration_sec"]
            ax.barh(i, dur, left=start, height=0.5, color=col, alpha=0.85,
                    edgecolor=INK, lw=0.6, zorder=3)
            ax.text(start + dur / 2, i, f"{s['peak_gpu_util']:.0f}%\npeak",
                    ha="center", va="center", fontsize=6.2, color="white", zorder=4)
        ns = run["n_sessions"]
        ax.annotate(f"{ns} session{'s' if ns != 1 else ''}  —  {note}",
                    xy=(1.005, i), xycoords=("axes fraction", "data"),
                    fontsize=7.2, color=col, va="center", ha="left", fontweight="bold")
        ylabels.append(name)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("time since first session start  (s)")
    ax.set_xlim(-8, None)
    ax.set_title("S1/S2 live-daemon attribution: one honest run either fragments into "
                 "unlinkable sessions (S1) or stays fully flagged (S2)",
                 fontsize=9.0, color=INK, loc="left")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(right=0.72)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_redteam_sessions.png"))
    plt.close(fig)
    print("wrote writeup/fig_redteam_sessions.png")


# ── Fig 3: the arith_intensity consistency-gate shield ──────────────────────
def fig_gates(cal):
    cgcal = cg.build_calibration(_default_phase1_paths())
    bands = cg.calibrate_bands(cgcal)
    lo, hi, _n = bands["arith_intensity"]
    fp_per, fp_any, _n2 = cg.false_positive_rate(cgcal, bands)

    benign = [v for v in (cg._arith_intensity(r, cgcal) for r in cgcal["_frontier"])
              if v is not None]

    # S4 batch runs across all 10 trials, grouped by batch size
    s4 = {}
    for p in sorted(glob.glob(os.path.join(REPO_ROOT, "red_team/red_v3_trial*_records.json"))):
        for r in json.load(open(p))["records"]:
            if r["config"].get("strategy") != "S4_batch":
                continue
            b = r["config"]["batch_size"]
            v = cg._arith_intensity(r, cgcal)
            if v is not None:
                s4.setdefault(b, []).append(v)

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
    # benign band shaded + benign scatter
    ax.axhspan(lo, hi, color=AQUA, alpha=0.14, zorder=1)
    ax.axhline(lo, color=AQUA, lw=1.3, zorder=3)
    ax.axhline(hi, color=AQUA, lw=0.9, ls=(0, (3, 3)), zorder=3)
    jit = (np.linspace(-0.16, 0.16, len(benign)) if len(benign) > 1 else [0.0])
    ax.scatter(np.zeros(len(benign)) + jit, benign, s=9, color=BASE,
               edgecolor="none", alpha=0.7, zorder=2)
    ax.text(0, hi, "  benign frontier\n  (in-band = OK)", fontsize=7, color=MUTED,
            va="bottom", ha="center")

    batches = sorted(s4)
    for i, b in enumerate(batches, start=1):
        vals = s4[b]
        med = float(np.median(vals))
        caught = med < lo
        col = RED if caught else AMBER
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.1, 0.1, len(vals)), vals,
                   s=26, color=col, edgecolor=INK, lw=0.4, zorder=4)
        ax.scatter([i], [med], marker="_", s=420, color=col, lw=2.2, zorder=5)
        tag = "FLAG" if caught else "OK"
        ax.annotate(tag, xy=(i, max(vals) + (hi - lo) * 0.06),
                    fontsize=7.2, color=col, ha="center", va="bottom",
                    fontweight="bold")

    ax.set_xticks([0] + list(range(1, len(batches) + 1)))
    ax.set_xticklabels(["benign"] + [f"b{b}" for b in batches])
    ax.set_xlabel("S4 batch-inflation (constant ground truth)  →  higher batch")
    ax.set_ylabel("arith_intensity gate  =  NVLink bytes / DRAM bytes  ∝ 1/(batch·seq)")
    ax.set_title(f"The consistency-gate shield: batch-inflation drives NVLink/DRAM below the\n"
                 f"benign floor — flagged where the point estimate can't (benign FP "
                 f"{fp_per['arith_intensity']:.1f}% this gate, {fp_any:.1f}% any gate)",
                 fontsize=8.6, color=INK, loc="left")
    ax.legend(handles=[
        Patch(facecolor=AQUA, alpha=0.14, edgecolor=AQUA, label="benign band"),
        Patch(facecolor=RED, edgecolor=INK, label="below floor → gate FLAG"),
        Patch(facecolor=AMBER, edgecolor=INK, label="in band → gate misses (residual gap)"),
    ], fontsize=7.0, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_redteam_gates.png"))
    plt.close(fig)
    print("wrote writeup/fig_redteam_gates.png")


# ── Fig 4: estimate vs ground truth — red-team strategies over the benign cloud ─
def _benign_frontier(paths):
    recs = []
    for p in paths:
        r, *_ = load_records(p)
        recs.extend(valid(r))
    return [r for r in recs
            if r["config"].get("precision", "fp16") == "fp16" and is_frontier(r)]


def fig_vs_benign(cal):
    paths = _default_phase1_paths()
    frontier = _benign_frontier(paths)
    bgt = np.array([r["ground_truth_tf"] for r in frontier])
    best = np.array([est4(r, cal) for r in frontier])

    # red-team point-estimate runs, grouped by config label → median over trials
    cfg = {}   # label -> {"strategy", "gts":[], "ests":[]}
    for p in sorted(glob.glob(os.path.join(REPO_ROOT, "red_team/red_v3_trial*_records.json"))):
        for r in json.load(open(p))["records"]:
            d = cfg.setdefault(r["label"], {"strategy": r["config"]["strategy"],
                                            "gts": [], "ests": []})
            d["gts"].append(r["ground_truth_tf"]); d["ests"].append(est4(r, cal))
    for r in load("red_team/red_s5_records.json")["records"]:
        d = cfg.setdefault(r["label"], {"strategy": "S5_powercap", "gts": [], "ests": []})
        d["gts"].append(r["ground_truth_tf"]); d["ests"].append(est4(r, cal))

    # collapse each config to one (GT, median-estimate) point
    pts = {}   # strategy -> list of (gt, est_median)
    for lab, d in cfg.items():
        pts.setdefault(d["strategy"], []).append(
            (float(np.median(d["gts"])), float(np.median(d["ests"]))))

    style = {
        "S4_batch":    (AMBER, "D", "S4 batch-inflation"),
        "S3_atypical": (BLUE,  "s", "S3 atypical nhead/opt"),
        "S5_powercap": (AQUA,  "^", "S5 power-cap (DVFS)"),
    }
    order = ["S4_batch", "S3_atypical", "S5_powercap"]

    lim = (0, max(bgt.max(), best.max(),
                  max(e for v in pts.values() for _, e in v)) * 1.06)
    xs = np.array(lim)
    fig, ax = plt.subplots(figsize=(6.6, 5.6), dpi=200)
    ax.fill_between(xs, xs * 0.8, xs * 1.2, color=GRID, alpha=0.45, zorder=1,
                    linewidth=0, label="±20%")
    ax.fill_between(xs, xs * 0.9, xs * 1.1, color=BASE, alpha=0.5, zorder=1,
                    linewidth=0, label="±10%")
    ax.plot(xs, xs, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2)

    # benign context: one faint cloud (shows benign scatter at every GT)
    ax.scatter(bgt, best, s=11, color=BASE, alpha=0.30, linewidths=0, zorder=3,
               label="benign frontier (889 runs)")

    # point-estimate strategies: one dot per config = median over 10 trials
    for s in order:
        if s not in pts:
            continue
        col, mk, lab = style[s]
        g = np.array([x[0] for x in pts[s]])
        e = np.array([x[1] for x in pts[s]])
        err = (np.median(e) - np.median(g)) / np.median(g) * 100
        ax.scatter(g, e, s=90, color=col, marker=mk, edgecolor=INK, linewidths=0.8,
                   zorder=6, label=f"{lab}  ({len(g)} configs, median {err:+.0f}%)")

    # S1 split / S2 throttle — live single-GPU daemon (attribution, not point estimate).
    # The daemon's raw per-session net energy is zeroed by a single-GPU baseline bug,
    # so per-session magnitude is modeled as (benign daemon rate × session work); the
    # measured result is the session count. Robust: S1 → 4 sessions, S2 → 1.
    PURPLE = "#7b5cd6"
    none = load("red_team_old/live_none.json")
    split = load("red_team_old/live_split.json")
    thr = load("red_team_old/live_throttle.json")
    none, split, thr = (d[0] if isinstance(d, list) else d for d in (none, split, thr))
    g0, y0 = none["ground_truth_tf"], none["attributable_frontier_tf"]
    rate = y0 / g0                                   # benign daemon TF-per-true-TF

    ax.scatter([g0], [y0], s=70, color=SURFACE, edgecolor=RED, marker="o",
               linewidths=1.4, zorder=7,
               label=f"S1 benign control — 1 session ({y0:.0f} TF attributed)")
    # S1 split: 4 sessions, each ~¼ the work → ~¼ the attribution
    ns = split["n_sessions"]
    gf, yf = g0 / ns, y0 / ns
    jx = np.linspace(-1, 1, ns) * lim[1] * 0.018
    ax.scatter(gf + jx, np.full(ns, yf), s=95, color=RED, marker="X", edgecolor=INK,
               linewidths=0.8, zorder=7,
               label=f"S1 split — {ns} sessions, each ≈¼ ({yf:.0f} TF) → job hidden")
    ax.annotate("", xy=(gf, yf + lim[1] * 0.015), xytext=(g0, y0),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5,
                                connectionstyle="arc3,rad=0.15"), zorder=6)
    # S2 throttle: stays 1 frontier session → fully attributed (attack fails)
    ax.scatter([thr["ground_truth_tf"]], [thr["ground_truth_tf"] * rate], s=95,
               color=PURPLE, marker="P", edgecolor=INK, linewidths=0.8, zorder=7,
               label="S2 throttle — 1 session, fully attributed (not evaded)")

    # S1/S2/S5 sit below the smallest benign job — flag the extrapolation zone
    ax.axvspan(0, bgt.min(), color=RED, alpha=0.045, zorder=0)
    ax.axvline(bgt.min(), color=RED, lw=0.8, ls=(0, (2, 3)), zorder=1)
    ax.annotate("no benign run below here\n(S1/S2/S5 are smaller jobs\nthan any benign frontier run)",
                xy=(bgt.min() - lim[1] * 0.015, lim[1] * 0.52), fontsize=6.4,
                color=MUTED, va="center", ha="right")

    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("Ground-Truth Aggregate TFLOPs")
    ax.set_ylabel("Blue-team estimate  (TFLOPs)")
    ax.legend(fontsize=6.6, frameon=False, loc="lower right")
    ax.set_title("Red-team strategies vs benign runs at the same ground truth:\n"
                 "point-estimate attacks pull the estimate under y=x; S1 splits the job into ¼-size pieces",
                 fontsize=8.6, color=INK, loc="left")
    fig.text(0.5, -0.01,
             "S1/S2 use the live single-GPU daemon (attribution, not the point estimate); per-session "
             "magnitude = benign daemon rate × session work —\nthe daemon's raw per-session energy is "
             "zeroed by a known single-GPU baseline bug, so the measured result is the session count "
             "(S1→4, S2→1).",
             fontsize=5.6, color=MUTED, ha="center", va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_redteam_vs_benign.png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote writeup/fig_redteam_vs_benign.png")


def main():
    cal = phase1_calibration(_default_phase1_paths())
    fig_powercap(cal)
    fig_sessions()
    fig_gates(cal)
    fig_vs_benign(cal)


if __name__ == "__main__":
    main()
