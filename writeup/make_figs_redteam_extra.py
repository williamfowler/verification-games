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


# ── Fig 1: S5 power-cap (DVFS) — v1 short run vs v2 realistic run ────────────
def fig_powercap(cal):
    """Overlay the original short-run sweep (v1, GT ~977) and the realistic-length
    re-run (v2, GT ~2606). v1 showed a sub-stock energy 'sweet spot'; v2 shows that
    at realistic run length capping only RAISES J/FLOP — the sweet spot was a
    short-run artifact, so DVFS is not a working under-report lever at scale."""
    b4 = cal["band4"]

    def series(path):
        d = load(path)
        recs = sorted(d["records"], key=lambda r: r["config"]["power_cap_w"])
        caps = [r["config"]["power_cap_w"] for r in recs]
        jpf = [r["net_energy_j"] / r["ground_truth_tf"] for r in recs]
        err = [signed_pct(est4(r, cal), r["ground_truth_tf"]) for r in recs]
        gt = recs[0]["ground_truth_tf"]
        return caps, jpf, err, gt

    v1 = series("red_team/red_s5_records.json")
    v2 = series("red_team/red_s5_v2_records.json")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=200)
    lab1 = f"v1 · short run (GT {v1[3]:.0f} TF)"
    lab2 = f"v2 · realistic run (GT {v2[3]:.0f} TF)"

    # left: energy per TFLOP vs cap
    axL.plot(v1[0], v1[1], "--o", color=BASE, lw=1.8, ms=6, zorder=3, label=lab1)
    axL.plot(v2[0], v2[1], "-o", color=AQUA, lw=2.3, ms=7, zorder=4, label=lab2)
    # mark each series' energy-optimal (min J/TFLOP) point
    s1 = int(np.argmin(v1[1])); s2 = int(np.argmin(v2[1]))
    axL.annotate(f"v1 sweet spot\n{v1[0][s1]} W", xy=(v1[0][s1], v1[1][s1]),
                 xytext=(v1[0][s1] - 8, v1[1][s1] - 0.28), fontsize=7, color=MUTED,
                 ha="center", va="top", arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    axL.annotate(f"v2 cheapest at STOCK\n(capping only raises J/FLOP)",
                 xy=(v2[0][s2], v2[1][s2]), xytext=(v2[0][s2] - 20, v2[1][s2] - 0.05),
                 fontsize=7, color=AQUA, ha="left", va="top",
                 arrowprops=dict(arrowstyle="->", color=AQUA, lw=0.9))
    axL.set_xlabel("power cap  (nvidia-smi -pl, W)")
    axL.set_ylabel("energy per TFLOP  (J/TFLOP)   ↓ = cheaper per real FLOP")
    axL.set_title("The 'sweet spot' doesn't survive a realistic run length",
                  fontsize=9.0, color=INK, loc="left")
    axL.legend(fontsize=7.2, frameon=False, loc="upper center")
    axL.invert_xaxis()   # capping harder → rightward

    # right: 4-param signed error vs cap
    axR.axhspan(b4["lo"], b4["hi"], color=GRID, alpha=0.45, zorder=1, label="benign band (4p)")
    axR.axhline(0, color=MUTED, lw=0.8, zorder=2)
    axR.plot(v1[0], v1[2], "--o", color=BASE, lw=1.8, ms=6, zorder=3, label=lab1)
    axR.plot(v2[0], v2[2], "-o", color=AMBER, lw=2.3, ms=7, zorder=4, label=lab2)
    axR.annotate("v2: capping moves the\nestimate TOWARD truth",
                 xy=(v2[0][-2], v2[2][-2]), xytext=(200, -6), fontsize=7, color=AMBER,
                 ha="center", va="bottom", arrowprops=dict(arrowstyle="->", color=AMBER, lw=0.9))
    axR.set_xlabel("power cap  (nvidia-smi -pl, W)")
    axR.set_ylabel("4-param signed error vs GT (%)   ↓ = under-report")
    axR.set_title("v1's extra under-report reverses at realistic scale",
                  fontsize=9.0, color=INK, loc="left")
    axR.legend(fontsize=7.2, frameon=False, loc="lower center")
    axR.invert_xaxis()

    fig.suptitle("S5 DVFS power-capping is NOT a working under-report lever at realistic scale: "
                 "the v1 short-run 'sweet spot' is an artifact",
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


# ── Fig 5: efficiency (cost) vs estimator error, per strategy ───────────────
def fig_eff_vs_error(cal):
    """Cost-vs-effect scatter for the point-estimate strategies (S3/S4/S5). x =
    efficiency ratio R_benign/R_adv (GPU-hours per real FLOP vs benign; LOWER = more
    efficient/cheaper; ≤2 = budget-legal). y = 3-param signed error (down = under-
    report). A real threat sits lower-left: cheap AND under-reporting past the benign
    band. (S1/S2 are attribution attacks — session count, not a point-estimate error —
    so they live in fig_redteam_sessions, not here.)"""
    R = cal["r_benign"]; b4 = cal["band4"]

    def agg(recs, key):
        by = {}
        for r in recs:
            v = r["config"].get(key); gt = r["ground_truth_tf"]
            e = signed_pct(est4(r, cal), gt)
            f = efficiency_ratio(gt, r["duration_s"], R)
            if e is None or f is None:
                continue
            by.setdefault(v, {"e": [], "f": []})
            by[v]["e"].append(e); by[v]["f"].append(f)
        return [(k, float(np.median(x["f"])), float(np.median(x["e"])))
                for k, x in sorted(by.items())]

    trials = sorted(glob.glob(os.path.join(REPO_ROOT, "red_team/red_v3_trial*_records.json")))
    def pull(strat, drop_sgd=False):
        return [r for p in trials for r in json.load(open(p))["records"]
                if r["config"].get("strategy") == strat
                and not (drop_sgd and r["config"].get("optimizer") == "sgd")]
    s4 = agg(pull("S4_batch"), "batch_size")
    s3 = agg(pull("S3_atypical", drop_sgd=True), "nhead")
    s5 = agg(load("red_team/red_s5_v2_records.json")["records"], "power_cap_w")

    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=200)
    # benign scatter band (y) and the evasion floor
    ax.axhspan(b4["lo"], b4["hi"], color=GRID, alpha=0.5, zorder=1, label="benign band (3-param)")
    ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
    xmax = 2.15
    # threat quadrant: budget-legal (x≤2) AND under-reporting past the band (y<lo)
    ax.axhspan(-60, b4["lo"], xmin=0, xmax=(2.0) / xmax, color=RED, alpha=0.06, zorder=0)
    ax.axvline(2.0, color=RED, lw=1.3, ls=(0, (4, 3)), zorder=3, label="2× budget (legal ≤ 2)")
    ax.axvline(1.0, color=MUTED, lw=0.9, ls=(0, (1, 3)), zorder=2)
    ax.annotate("benign parity", xy=(1.0, b4["hi"]), fontsize=6.6, color=MUTED,
                rotation=90, va="top", ha="right")

    series = [("S4 batch-inflation", s4, AMBER, "D", "batch"),
              ("S3 atypical nhead", s3, BLUE, "s", "h"),
              ("S5 power-cap (v2)", s5, AQUA, "^", "W")]
    for name, pts, col, mk, unit in series:
        xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
        ax.plot(xs, ys, "-", color=col, lw=1.0, alpha=0.45, zorder=3)  # sweep trajectory
        ax.scatter(xs, ys, s=70, color=col, marker=mk, edgecolor=INK, linewidths=0.7,
                   zorder=5, label=name)
    # label the two configs that clear the evasion floor + the false-positive over-reporter
    def tag(pts, k, txt, dx, dy, col, ha="left"):
        p = next((q for q in pts if q[0] == k), None)
        if p:
            ax.annotate(txt, xy=(p[1], p[2]), xytext=(p[1] + dx, p[2] + dy),
                        fontsize=6.8, color=col, ha=ha, va="center",
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.7))
    tag(s4, 128, "batch 128", 0.05, -3, AMBER)
    tag(s5, 300, "stock 300 W", 0.06, 0, AQUA)
    tag(s3, 64, "tiny head_dim\n(over-reports)", -0.04, 4, BLUE, ha="right")

    ax.set_xlim(0.35, xmax); ax.set_ylim(-40, b4["hi"] + 3)
    ax.set_xlabel("Efficiency ratio  R_benign / R_adv   (GPU-hours per real FLOP vs benign)\n"
                  "← lower = more efficient / cheaper          higher = wastes GPU-hours →")
    ax.set_ylabel("3-param signed error vs GT (%)    ↓ = under-report")
    ax.annotate("THREAT ZONE\ncheap AND under-reporting", xy=(0.55, -32), fontsize=7.4,
                color=RED, weight="bold", va="center", ha="left")
    ax.set_title("Red-team cost vs. effect: every strategy is budget-legal (all ≤2×);\n"
                 "only S5-stock and S4-batch-128 clear the benign band into evasion",
                 fontsize=8.8, color=INK, loc="left")
    ax.legend(fontsize=7.0, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_redteam_eff_vs_error.png"))
    plt.close(fig)
    print("wrote writeup/fig_redteam_eff_vs_error.png")


def main():
    cal = phase1_calibration(_default_phase1_paths())
    fig_powercap(cal)
    fig_sessions()
    fig_gates(cal)
    fig_vs_benign(cal)
    fig_eff_vs_error(cal)


if __name__ == "__main__":
    main()
