"""Figures for the fp16-AMP DDP regime (2x V100), from the DDP sweep records.

  fig_ddp_est_vs_truth.png : held-out (leave-one-out) estimated vs true aggregate
                             TFLOPs on the fp16 frontier — the accuracy result.
  fig_ddp_nvlink.png       : measured NVLink bytes/step vs predicted gradient
                             bytes (n_params x 4) — the new blue-team observable,
                             i.e. the DDP all-reduce made visible from the outside.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "writeup")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "power_calibration"))
from eval_power_monitor import (load_records, valid, is_frontier,
                                fit_active_energy_model, score)

RECORDS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "eval_results_v100_ddp_records.json")

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE, BLUE, AQUA = "#e1e0d9", "#c3c2b7", "#2a78d6", "#1baf7a"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
})

recs, base, bsec, fp = load_records(RECORDS)
frontier = [r for r in valid(recs)
            if r["config"].get("precision", "fp16") == "fp16" and is_frontier(r)]
print(f"fp16 frontier configs: {len(frontier)}")

# ── Figure: held-out (LOO) estimated vs true TFLOPs ──────────────────────────
gt = np.array([r["ground_truth_tf"] for r in frontier])
est = np.zeros(len(frontier))
for i in range(len(frontier)):
    tr = frontier[:i] + frontier[i+1:]
    e, p = fit_active_energy_model(tr)
    score(frontier[i:i+1], e, p)
    est[i] = frontier[i]["est_tflops"]

lim = (0, max(gt.max(), est.max()) * 1.08)
xs = np.array(lim)
fig, ax = plt.subplots(figsize=(4.8, 4.6), dpi=200)
ax.fill_between(xs, xs * 0.9, xs * 1.1, color=GRID, alpha=0.55, zorder=1,
                linewidth=0, label="±10% error")
ax.plot(xs, xs, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2)
ax.scatter(gt, est, s=24, color=BLUE, zorder=4, linewidths=0,
           label="held-out (LOO) estimate")
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel("Ground-Truth Aggregate TFLOPs (2 GPU)")
ax.set_ylabel("Estimated TFLOPs")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title("fp16-AMP DDP: Estimated vs. True Training TFLOPs",
             fontsize=9.2, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_ddp_est_vs_truth.png"))
plt.close(fig)

# ── Figure: NVLink bytes/step vs predicted gradient bytes ────────────────────
nv = [r for r in frontier if r.get("nvlink_bytes_per_step") and r.get("grad_bytes_pred")]
grad_mb = np.array([r["grad_bytes_pred"] / 1e6 for r in nv])
nvl_mb = np.array([r["nvlink_bytes_per_step"] / 1e6 for r in nv])
order = np.argsort(grad_mb)
gm, nm = grad_mb[order], nvl_mb[order]

fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=200)
gx = np.array([0, gm.max() * 1.08])
# ideal: both-GPU sum of TX+RX per step = 4 x grad_bytes (ring all-reduce, n=2)
ax.plot(gx, 4 * gx, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2,
        label="ring all-reduce prediction (4× grad bytes)")
ax.scatter(gm, nm, s=28, color=AQUA, zorder=4, linewidths=0,
           label="measured NVLink / step")
ax.set_xlim(0, gm.max() * 1.08); ax.set_ylim(0, nm.max() * 1.12)
ax.set_xlabel("Predicted Gradient Bytes / step  (n_params × 4 B, MB)")
ax.set_ylabel("Measured NVLink Bytes / step  (both GPUs, MB)")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.set_title("DDP all-reduce made visible: NVLink traffic vs. model size",
             fontsize=9.2, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_ddp_nvlink.png"))
plt.close(fig)

if len(nv):
    ratio = nm / (4 * gm)
    print(f"NVLink/step vs 4x grad prediction: mean ratio {ratio.mean():.2f}"
          f" (sd {ratio.std():.2f}, n={len(nv)})")
print("wrote fig_ddp_est_vs_truth.png, fig_ddp_nvlink.png")
