"""
score_redteam.py — Phase II v3 scorer (under-reporting foregrounded).

PRIMARY ESTIMATOR = the branch's full **4-param** form
    TFLOPs_est = (E_net − E_PER_TB·TB_moved − E_PER_NVLINK·NVLink_TB − P_OH·t) / E_MARGINAL
i.e. one coefficient each for compute-energy (FLOPs), memory bandwidth (DRAM),
inter-GPU data transfer (NVLink), and overall-power overhead (detect_flops.
estimate_tflops_nvl). On the benign fp16 frontier the NVLink coefficient fits to
0 (all-reduce volume is collinear with DRAM bytes), so numerically the 4-param
reduces to power+DRAM — but it IS the estimator every strategy is scored against.
The 2-param (power-only) estimator is reported alongside as the accuracy ablation
Phase I flagged (the byte terms don't improve benign accuracy).

Deployment framing: the estimator is the FROZEN Phase I benign fp16 fit; no
adversarial run ever enters a fit. Success for the red team = the 4-param estimate
reads BELOW ground truth (evasion), ranked by under-report beyond the benign band.

    python3 red_team/score_redteam.py --group-a red_team/red_v3_trial1_records.json \
        [--group-b red_team/live_*.json]
"""
import argparse
import json
import os
import sys
from statistics import median

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "power_calibration"))
sys.path.insert(0, os.path.join(REPO_ROOT, "red_team"))

import detect_flops
import consistency_gates as cg
from eval_power_monitor import (load_records, valid, is_frontier,
                                fit_active_energy_model, fit_active_energy_nvl_model)
from efficiency import (benign_reference_rate, efficiency_ratio, is_budget_legal,
                        BUDGET, _default_phase1_paths)

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


def signed_pct(est, gt):
    return (est - gt) / gt * 100.0 if est is not None and gt else None


def est4(r, cal):
    """The primary 4-param estimate for a record (FLOPs+DRAM+NVLink+overhead)."""
    nv = r.get("nvlink_total_bytes")
    tb = r.get("tb_moved")
    if tb is None:
        return None
    return detect_flops.estimate_tflops_nvl(
        r["net_energy_j"], r["duration_s"], tb, (nv / 1e12 if nv is not None else 0.0),
        p_overhead_w=cal["p4"], e_marginal_j_per_tflop=cal["e4"],
        e_per_tb_j=cal["c4tb"], e_per_nvlink_tb_j=cal["c4nv"])


def est2(r, cal):
    """The 2-param power-only estimate (ablation reference)."""
    return detect_flops.estimate_tflops(
        r["net_energy_j"], r["duration_s"],
        p_overhead_w=cal["p2"], e_marginal_j_per_tflop=cal["e2"])


def phase1_calibration(paths):
    recs = []
    for p in paths:
        rlist, *_ = load_records(p)
        recs.extend(valid(rlist))
    frontier = [r for r in recs
                if r["config"].get("precision", "fp16") == "fp16" and is_frontier(r)]
    e4, c4tb, c4nv, p4 = fit_active_energy_nvl_model(frontier)   # PRIMARY
    e2, p2 = fit_active_energy_model(frontier)                   # power-only ablation
    r_benign, n_ref = benign_reference_rate(recs)
    cal = {"e4": e4, "c4tb": c4tb, "c4nv": c4nv, "p4": p4, "e2": e2, "p2": p2,
           "r_benign": r_benign, "n_ref": n_ref, "n_frontier": len(frontier)}
    # Benign noise band per estimator (spread of benign signed errors).
    def band(fn):
        s = np.array([x for x in (signed_pct(fn(r, cal), r["ground_truth_tf"])
                                  for r in frontier) if x is not None])
        return {"mean": float(s.mean()), "sd": float(s.std()),
                "lo": float(np.percentile(s, 5)), "hi": float(np.percentile(s, 95))}
    cal["band4"] = band(est4)
    cal["band2"] = band(est2)
    return cal


def score_offline(records, cal, cgcal, cgbands):
    out = []
    for r in valid(records):
        gt = r["ground_truth_tf"]
        cfg = r["config"]
        gate = cg.check_run(r, cgcal, cgbands)             # cross-signal consistency layer
        out.append({
            "label": r["label"], "strategy": cfg.get("strategy", "?"),
            "parent": cfg.get("parent"), "batch_size": cfg.get("batch_size"),
            "nhead": cfg.get("nhead"), "optimizer": cfg.get("optimizer"),
            "gt": gt, "err4": signed_pct(est4(r, cal), gt),   # PRIMARY
            "err2": signed_pct(est2(r, cal), gt),             # power-only ablation
            "tb_moved": r.get("tb_moved"), "duration_s": r["duration_s"],
            "j_per_tflop": r["net_energy_j"] / gt if gt else None,
            "eff_ratio": efficiency_ratio(gt, r["duration_s"], cal["r_benign"]),
            "avg_gpu_pct": r.get("avg_gpu_pct"),
            "power_cap_w": cfg.get("power_cap_w"), "clock_mhz": cfg.get("clock_mhz"),
            "gate": gate["overall"], "gate_flags": gate["flagged"],
        })
    return out


def evades(err, band):
    """Under-reports beyond benign scatter: signed error below the benign 5th pct."""
    return err is not None and err < band["lo"]


def summarize(scores, cal):
    by = {}
    for s in scores:
        by.setdefault(s["strategy"], []).append(s)
    rows = []
    for strat, ss in sorted(by.items()):
        e4 = [x["err4"] for x in ss if x["err4"] is not None]
        ev4 = [x for x in ss if evades(x["err4"], cal["band4"])]
        legal_ev4 = [x for x in ev4 if is_budget_legal(x["eff_ratio"])]
        rows.append({
            "strategy": strat, "n": len(ss),
            "err4_med": median(e4) if e4 else None, "err4_min": min(e4) if e4 else None,
            "evade4": len(ev4), "evade4_legal": len(legal_ev4),
            "eff_med": median([x["eff_ratio"] for x in ss if x["eff_ratio"]]) if ss else None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase1", nargs="+", default=None)
    ap.add_argument("--group-a", nargs="+", default=None,
                    help="offline v3 red records (S3/S4/S5)")
    ap.add_argument("--group-b", nargs="*", default=[],
                    help="live_daemon_probe records (S1 split / S2 throttle)")
    args = ap.parse_args()

    p1 = args.phase1 or _default_phase1_paths()
    cal = phase1_calibration(p1)
    cgcal = cg.build_calibration(p1)               # cross-signal consistency layer
    cgbands = cg.calibrate_bands(cgcal)
    _fp, cg_any_fp, _n = cg.false_positive_rate(cgcal, cgbands)
    b4, b2 = cal["band4"], cal["band2"]
    print("=" * 80)
    print("FROZEN PHASE I CALIBRATION — PRIMARY ESTIMATOR = 4-param "
          "(FLOPs + DRAM + NVLink + overhead)")
    print("=" * 80)
    print(f"  4-param: E_MARGINAL {cal['e4']:.3f}  E_PER_TB {cal['c4tb']:.2f}  "
          f"E_PER_NVLINK {cal['c4nv']:.4f}  P_OH {cal['p4']:.3f}")
    print(f"           (NVLink coeff = {cal['c4nv']:.4f}: all-reduce volume is collinear"
          f" with DRAM on benign DDP, so it fits to 0 — the 4-param reduces to power+DRAM)")
    print(f"  2-param (power-only ablation): E_MARGINAL {cal['e2']:.3f}  P_OH {cal['p2']:.3f}")
    print(f"  benign signed err  4-param: mean {b4['mean']:+.1f}% band[5,95]=[{b4['lo']:+.1f},{b4['hi']:+.1f}]%"
          f"   power-only: band=[{b2['lo']:+.1f},{b2['hi']:+.1f}]%")
    print(f"  R_benign {cal['r_benign']:.0f} TF/GPU-h  budget ≤ {BUDGET}×   "
          f"(evasion = 4-param signed err below the 5th-pct benign band)")

    scores = []
    if args.group_a:
        recs = []
        for p in args.group_a:
            r, *_ = load_records(p)
            recs.extend(r)
        scores = score_offline(recs, cal, cgcal, cgbands)
        print("\n" + "=" * 80)
        print("OFFLINE STRATEGIES vs the 4-param estimator  (− = UNDER-report = evasion)")
        print("=" * 80)
        print(f"  {'strategy':<12} {'n':>2} {'err4p med/min':>16} {'evade4p':>9} "
              f"{'legal':>6} {'eff×':>6}")
        for row in summarize(scores, cal):
            e4 = f"{row['err4_med']:+.0f}/{row['err4_min']:+.0f}%" if row['err4_med'] is not None else "--"
            ef = f"{row['eff_med']:.1f}" if row['eff_med'] is not None else "--"
            print(f"  {row['strategy']:<12} {row['n']:>2} {e4:>16} "
                  f"{row['evade4']:>4}/{row['n']:<3} {row['evade4_legal']:>5} {ef:>6}")
        print("\n  per-config (var = batch for S4, nhead/opt for S3; err4p = 4-param"
              " primary; gate = cross-signal consistency layer):")
        for s in sorted(scores, key=lambda x: (x["strategy"], x["batch_size"] or 0, x["nhead"] or 0)):
            var = (f"b{s['batch_size']}" if s["strategy"] == "S4_batch"
                   else (f"h{s['nhead']}" if s["optimizer"] != "sgd" else "sgd"))
            jpf = f"{s['j_per_tflop']:.3f}" if s['j_per_tflop'] else "NA"
            ev = "EVADE" if evades(s["err4"], b4) else "     "
            gate = (f"GATE-FLAG[{','.join(s['gate_flags'])}]" if s["gate"] == "FLAG"
                    else "gate-ok")
            print(f"    {s['label']:32s} {var:>5}  J/TF {jpf:>7}  "
                  f"err4p {s['err4']:+6.0f}%  eff {s['eff_ratio']:.1f}×  {ev}  {gate}")

        # Defense-in-depth: for the strongest under-reporters, does the independent
        # consistency layer flag them even where the point estimate can't (they sit
        # inside the wide single-trial benign band)?
        under = sorted([s for s in scores if s["err4"] is not None and s["err4"] < -10],
                       key=lambda s: s["err4"])
        print(f"\n  DEFENSE-IN-DEPTH (consistency layer, benign false-positive rate "
              f"{cg_any_fp:.1f}%):")
        print(f"    strongest under-reporters vs the independent gate layer —")
        for s in under:
            var = (f"b{s['batch_size']}" if s["strategy"] == "S4_batch"
                   else (f"h{s['nhead']}" if s["optimizer"] != "sgd" else "sgd"))
            tag = (f"FLAGGED ({','.join(s['gate_flags'])})" if s["gate"] == "FLAG"
                   else "not flagged  ← residual gap")
            print(f"      {s['strategy']:12s} {var:>4}  err4p {s['err4']:+5.0f}%  → {tag}")
        caught = [s for s in under if s["gate"] == "FLAG"]
        print(f"    → the point estimate reads all of these as ~benign (within the wide"
              f" ±band); the gate layer independently flags {len(caught)}/{len(under)},"
              f" incl. the worst.")

    scores_b = ingest_group_b(args.group_b)
    if scores_b:
        print("\n" + "=" * 80)
        print("LIVE-DAEMON STRATEGIES (S1 split / S2 throttle) — attribution")
        print("  (estimator-agnostic: the attack is session fragmentation / gate, not the point estimate)")
        print("=" * 80)
        for s in scores_b:
            print(f"  {s['strategy']:<10} GT {s['gt'] or 0:>6.0f}  sessions {s['n_sessions']} "
                  f"(frontier {s['n_frontier_sessions']})  attributed {s['attributable_frontier_tf']:.0f} TF")

    with open(os.path.join(REPO_ROOT, "redteam_scores.json"), "w") as f:
        json.dump({"calibration": cal, "offline": scores, "group_b": scores_b}, f, indent=1)
    print("\nwrote redteam_scores.json")
    if scores:
        make_figs(scores, cal)


def ingest_group_b(paths):
    out = []
    for path in paths:
        d = json.load(open(path))
        d = d[0] if isinstance(d, list) else d
        out.append({"strategy": d.get("strategy"), "gt": d.get("ground_truth_tf"),
                    "n_sessions": d.get("n_sessions"),
                    "n_frontier_sessions": d.get("n_frontier_sessions"),
                    "attributable_frontier_tf": d.get("attributable_frontier_tf", 0.0)})
    return out


def _agg_by(scores, strategy, key, ref_val=None):
    """Group a strategy's records by the manipulated variable, returning per-value
    (x, median err4, std err4, median err2, median eff) across trials."""
    groups = {}
    for s in scores:
        if s["strategy"] != strategy or s.get(key) is None or s["err4"] is None:
            continue
        if strategy == "S3_atypical" and s["optimizer"] == "sgd":
            continue
        groups.setdefault(s[key], []).append(s)
    out = []
    for v in sorted(groups):
        g = groups[v]
        out.append({
            "x": v, "n": len(g),
            "err4_med": float(np.median([s["err4"] for s in g])),
            "err4_std": float(np.std([s["err4"] for s in g])),
            "err2_med": float(np.median([s["err2"] for s in g])),
            "eff_med": float(np.median([s["eff_ratio"] for s in g if s["eff_ratio"]])),
        })
    return out


def make_figs(scores, cal):
    b4 = cal["band4"]
    n_trials = max((sum(1 for s in scores if s["label"] == lab)
                    for lab in {s["label"] for s in scores}), default=1)

    # ── Fig 1: signed error vs manipulated variable — median ± std over trials ──
    panels = []
    s4 = _agg_by(scores, "S4_batch", "batch_size")
    s3 = _agg_by(scores, "S3_atypical", "nhead")
    for title, xlab, ref, pts in [
            ("S4 batch-inflation", "batch size (calibration ≤ 32)", 8, s4),
            ("S3 atypical nhead", "nhead (head_dim = d_model/nhead)", 8, s3)]:
        if pts:
            panels.append((title, xlab, ref, pts))
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(4.7 * len(panels), 4.3), dpi=200)
        if len(panels) == 1:
            axes = [axes]
        for ax, (title, xlab, ref, pts) in zip(axes, panels):
            xs = [p["x"] for p in pts]
            ax.axhspan(b4["lo"], b4["hi"], color=GRID, alpha=0.45, zorder=1,
                       label="benign band (4p)")
            ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
            ax.errorbar(xs, [p["err4_med"] for p in pts], yerr=[p["err4_std"] for p in pts],
                        fmt="-o", color=AMBER, lw=2.0, ms=6, capsize=3, zorder=4,
                        label="4-param (±1 sd over 10 trials)")
            ax.plot(xs, [p["err2_med"] for p in pts], "--s", color=BLUE, lw=1.3, ms=4,
                    zorder=3, label="2-param (power-only)")
            # mark the parent reference (the 'one-variable' baseline) — its own error
            rp = [p for p in pts if p["x"] == ref]
            if rp:
                ax.axhline(rp[0]["err4_med"], color=INK2, lw=0.9, ls=(0, (2, 3)), zorder=2)
                ax.annotate(f"parent (var={ref}) baseline {rp[0]['err4_med']:+.0f}%",
                            xy=(xs[0], rp[0]["err4_med"]), fontsize=6.5, color=INK2,
                            va="bottom", ha="left")
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs])
            ax.minorticks_off()
            ax.set_xlabel(xlab); ax.set_ylabel("4-param signed error vs GT (%)  − = under-report")
            ax.set_title(title, fontsize=9.2, color=INK, loc="left")
            ax.legend(fontsize=7.0, frameon=False, loc="best")
        fig.suptitle(f"Per-config error across {n_trials} trials (attack effect = deviation"
                     f" from the dashed parent line, not from 0)", fontsize=8.4, color=MUTED, y=1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_redteam_signed.png"))
        plt.close(fig)

    # ── Fig 2: under-report vs efficiency — per-config median ± std over trials ─
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
    for strat, col, key in [("S4_batch", AMBER, "batch_size"),
                            ("S3_atypical", BLUE, "nhead"), ("S5_powercap", AQUA, "power_cap_w")]:
        pts = _agg_by(scores, strat, key)
        if pts:
            ax.errorbar([p["eff_med"] for p in pts], [p["err4_med"] for p in pts],
                        yerr=[p["err4_std"] for p in pts], fmt="o", color=col, ms=6,
                        capsize=2, lw=1, zorder=4, label=strat)
    ax.axhspan(b4["lo"], b4["hi"], color=GRID, alpha=0.45, zorder=1, label="benign band (4p)")
    ax.axhline(0, color=MUTED, lw=0.8, zorder=2)
    ax.axvline(BUDGET, color=RED, lw=1.2, ls=(0, (4, 3)), zorder=3, label=f"{BUDGET:.0f}× budget")
    ax.set_xlabel("Efficiency ratio  R_benign/R_adv  (≤2 = legal →)")
    ax.set_ylabel("4-param signed error vs GT (%)   ↓ = under-report / evasion")
    ax.set_title(f"Evasion vs cost (4-param, per-config median ±1sd over {n_trials} trials):\n"
                 "a threat sits in the lower-left (under-reports AND within 2× budget)",
                 fontsize=8.8, color=INK, loc="left")
    ax.legend(fontsize=7.4, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_redteam_evasion.png"))
    plt.close(fig)
    print("wrote writeup/fig_redteam_signed.png, fig_redteam_evasion.png")


if __name__ == "__main__":
    main()
