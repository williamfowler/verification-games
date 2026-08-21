#!/usr/bin/env python3
"""
make_fig_methodology.py — Figure 1: the calibration -> deployment methodology overview.

Regenerates the report's methodology diagram with corrected content (the old hand-made
version said "26-workload sweep" and omitted NVLink). Matches the house figure style.

    python3 writeup/make_fig_methodology.py   ->   writeup/fig_methodology.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE, BLUE, AMBER, AQUA = "#e1e0d9", "#c3c2b7", "#7b9fd4", "#e3bc70", "#7fbfa4"
GREEN_FILL, BLUE_FILL, NEUTRAL_FILL = "#e8f5ef", "#e7f0fb", "#f1f0ea"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]})


def box(ax, cx, cy, w, h, text, fill, edge, fs=12.5, weight="normal", tcol=INK):
    x, y = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=2.2",
                                linewidth=1.4, edgecolor=edge, facecolor=fill, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tcol,
            weight=weight, zorder=4)
    return {"cx": cx, "cy": cy, "top": (cx, cy + h / 2), "bot": (cx, cy - h / 2),
            "l": (cx - w / 2, cy), "r": (cx + w / 2, cy)}


def arrow(ax, p0, p1, color=INK2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, lw=1.5,
                                 color=color, zorder=2, shrinkA=1, shrinkB=1))


def column(ax, cx, w, h, steps, fill, edge):
    ys = [72, 52, 32, 12]
    prev = None
    boxes = []
    for t, y in zip(steps, ys):
        b = box(ax, cx, y, w, h, t, fill, edge)
        if prev is not None:
            arrow(ax, prev["bot"], b["top"])
        prev = b
        boxes.append(b)
    return boxes


def main():
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    fig.patch.set_facecolor(SURFACE)

    # outer containers + labels
    for x0, lbl, col in [(2, "CALIBRATION  ·  run once", AQUA),
                         (55, "DEPLOYMENT  ·  per workload", BLUE)]:
        ax.add_patch(FancyBboxPatch((x0, 4), 30, 88, boxstyle="round,pad=0.6,rounding_size=2.4",
                                    linewidth=1.6, edgecolor=INK, facecolor="none", zorder=1))
        ax.text(x0 + 15, 88, lbl, ha="center", va="center", fontsize=14.5,
                weight="bold", color=col)

    w, h = 26, 15
    column(ax, 17, w, h, [
        "Measure idle\npower baseline",
        "Run 91-config sweep;\nkeep 88 above the\n80% frontier gate",
        "Record energy, DRAM\n& NVLink bytes\nper workload",
        "Fit energy constants\nvs ground truth",
    ], GREEN_FILL, AQUA)

    column(ax, 70, w, h, [
        "Poll power, DRAM &\nNVLink every 1.5 s",
        "Detect workload\nstart / end",
        "Accumulate energy\n& traffic over\nthe session",
        "Estimate FLOPs with\nequation (1)",
    ], BLUE_FILL, BLUE)

    # bridge between containers (mid height)
    bridge = box(ax, 43.5, 50, 15, 15, "Freeze the\nconstants into\nequation (1)",
                 NEUTRAL_FILL, BASE, fs=12.5, weight="bold", tcol=INK2)
    arrow(ax, (32, 50), bridge["l"])          # calibration container -> bridge
    arrow(ax, bridge["r"], (56.5, 50))        # bridge -> deployment container

    # deployment -> evaluation
    ev = box(ax, 92.5, 50, 13, 22,
             "Evaluate vs\nheld-out\nground truth",
             NEUTRAL_FILL, BASE, fs=11.5, tcol=INK2)
    arrow(ax, (85, 50), ev["l"])

    for name in ("fig_methodology.png", "figure_1.png"):
        fig.savefig(os.path.join(OUT, name), facecolor=SURFACE,
                    bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote writeup/fig_methodology.png + writeup/figure_1.png")


if __name__ == "__main__":
    main()
