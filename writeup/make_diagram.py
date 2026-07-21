"""Framework/pipeline diagram for the write-up: calibration lane feeding the
deployed monitoring loop. Pure layout — no experiment data."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))

# same palette as make_figs.py
SURFACE = "#fcfcfb"
INK     = "#0b0b0b"
INK2    = "#52514e"
MUTED   = "#898781"
GRID    = "#e1e0d9"
BASE    = "#c3c2b7"
BLUE    = "#2a78d6"
AQUA    = "#1baf7a"

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"]})

fig, ax = plt.subplots(figsize=(7.4, 3.9), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

BOX_H = 17


def box(x, y, w, text, accent, fontsize=7.2):
    """Rounded box with a colored left accent bar; (x, y) = center."""
    p = FancyBboxPatch((x - w / 2, y - BOX_H / 2), w, BOX_H,
                       boxstyle="round,pad=0.4,rounding_size=1.6",
                       linewidth=0.9, edgecolor=BASE, facecolor="white",
                       zorder=3)
    ax.add_patch(p)
    ax.plot([x - w / 2 + 0.4, x - w / 2 + 0.4],
            [y - BOX_H / 2 + 1.6, y + BOX_H / 2 - 1.6],
            color=accent, lw=2.6, solid_capstyle="round", zorder=4)
    ax.text(x + 0.9, y, text, ha="center", va="center", fontsize=fontsize,
            color=INK, zorder=5, linespacing=1.25)
    return x - w / 2, x + w / 2


def arrow(x0, x1, y0, y1=None, label=None, ls="-"):
    y1 = y0 if y1 is None else y1
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                        mutation_scale=9, lw=1.0, color=INK2, zorder=2,
                        linestyle=ls, shrinkA=1, shrinkB=1)
    ax.add_patch(a)
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 1.5, label, ha="center",
                va="bottom", fontsize=6.4, color=INK2)


# ── calibration lane ───────────────────────────────────────────────────────────
ax.text(2, 96, "CALIBRATION  (once per device + power mode)", fontsize=8,
        color=INK, fontweight="bold", va="center")
cy = 80
c1 = box(12, cy, 19, "measure idle\nbaseline\n(90 s median)", AQUA)
c2 = box(35.5, cy, 20, "benchmark DRAM\nbyte-scale k\n(known transfer)", AQUA)
c3 = box(60.5, cy, 22, "26-config transformer\nsweep + FlopCounterMode\nground truth", AQUA)
c4 = box(87, cy, 20, "fit constants\nE_PER_TFLOP, E_per_TB,\nE_overhead", AQUA)
arrow(c1[1], c2[0], cy)
arrow(c2[1], c3[0], cy)
arrow(c3[1], c4[0], cy)

# constants flow down into the deployed estimator (top of the eq (1) box)
arrow(87, 87, cy - BOX_H / 2 - 0.7, 43.2, label="")
ax.text(85.5, 59, "calibrated constants\n(matched set)", ha="right",
        fontsize=6.4, color=INK2, va="center")

# ── deployment lane ────────────────────────────────────────────────────────────
ax.text(2, 52, "DEPLOYMENT  (monitoring daemon, no access to workload code)",
        fontsize=8, color=INK, fontweight="bold", va="center")
dy = 33
d1 = box(12, dy, 19, "poll sensors\npower · GPU util ·\nEMC  (every 1.5 s)", BLUE)
d2 = box(35.5, dy, 20, "detect workload\nstart / end\n(util hysteresis)", BLUE)
d3 = box(60.5, dy, 22, "integrate over session t:\nE_net (J), TB_moved", BLUE)
d4 = box(87, dy, 20, "equation (1)\n→ TFLOP estimate\nper workload", BLUE)
arrow(d1[1], d2[0], dy)
arrow(d2[1], d3[0], dy)
arrow(d3[1], d4[0], dy)

# sensor sources under the poll box
ax.text(12, dy - BOX_H / 2 - 3.5, "INA3221 power rail  ·  actmon counters",
        ha="center", fontsize=6.4, color=MUTED)

# scoring
sy = 9
s1 = box(60.5, sy, 22, "compare vs FlopCounterMode\nground truth → % error", BLUE,
         fontsize=7.0)
arrow(87, s1[1] + 0.7, dy - BOX_H / 2 - 0.7, sy + 2, label="")
ax.text(80, 19.5, "evaluation only", ha="left", fontsize=6.4, color=MUTED)

fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(OUT, "fig_pipeline.png"), facecolor=SURFACE)
print("wrote", os.path.join(OUT, "fig_pipeline.png"))
