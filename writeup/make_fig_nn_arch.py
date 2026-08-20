#!/usr/bin/env python3
"""make_fig_nn_arch.py — node diagram of the two neural FLOP estimators
(nn_estimator.py): labeled input features -> 32 -> 32 hidden units -> output,
plus the residual variant's physics backbone. Style matches the other writeup
figures. Writes writeup/fig_nn_arch.png.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE, BLUE, AQUA, AMBER, RED, PURPLE = ("#e1e0d9", "#c3c2b7", "#2a78d6",
                                              "#1baf7a", "#e69f00", "#d1495b", "#7b5cd6")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9,
})

EDGE = "#deddd5"

PURE_FEATS = [r"$\log E$", r"$\log t$", r"$\log TB$", r"$\log NV$",
              r"$\log P$", r"$\log(E/TB)$", r"$\log(NV/TB)$"]
RESID_FEATS = [r"$\log P$", r"$\log(E/TB)$", r"$\log(NV/TB)$",
               r"$\log(TB/t)$", r"$\log(NV/t)$", r"$\log t$"]


def hidden_ys(lo=0.14, hi=0.86, n=8):
    """8 drawn nodes standing in for 32: 4 above and 4 below an ellipsis."""
    ys = list(np.linspace(hi, lo, n))
    mid = (ys[3] + ys[4]) / 2.0
    return ys, mid


def draw_net(ax, feats, color, out_label, title, subtitle):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    x_in, x_h1, x_h2, x_out = 0.30, 0.52, 0.70, 0.88
    r_in, r_h, r_out = 0.014, 0.013, 0.020

    in_ys = list(np.linspace(0.83, 0.17, len(feats)))
    h1_ys, h1_mid = hidden_ys()
    h2_ys, h2_mid = hidden_ys()
    out_y = 0.50

    # edges first (recessive)
    for y0 in in_ys:
        for y1 in h1_ys:
            ax.plot([x_in, x_h1], [y0, y1], color=EDGE, lw=0.45, zorder=1)
    for y0 in h1_ys:
        for y1 in h2_ys:
            ax.plot([x_h1, x_h2], [y0, y1], color=EDGE, lw=0.45, zorder=1)
    for y0 in h2_ys:
        ax.plot([x_h2, x_out], [y0, out_y], color=EDGE, lw=0.45, zorder=1)

    # input nodes + feature labels
    for y, f in zip(in_ys, feats):
        ax.add_patch(Circle((x_in, y), r_in, facecolor=SURFACE, edgecolor=INK2,
                            lw=1.0, zorder=3))
        ax.text(x_in - 0.035, y, f, ha="right", va="center", fontsize=8.2,
                color=INK2, zorder=3)

    # hidden nodes with ellipsis
    for x, ys, mid in ((x_h1, h1_ys, h1_mid), (x_h2, h2_ys, h2_mid)):
        for y in ys:
            ax.add_patch(Circle((x, y), r_h, facecolor=color, edgecolor="none",
                                alpha=0.85, zorder=3))
        ax.text(x, mid, "⋮", ha="center", va="center", fontsize=11,
                color=INK2, zorder=4,
                bbox=dict(boxstyle="round,pad=0.08", fc=SURFACE, ec="none"))

    # output node
    ax.add_patch(Circle((x_out, out_y), r_out, facecolor=INK2, edgecolor="none",
                        zorder=3))
    ax.text(x_out, out_y - 0.065, out_label, ha="center", va="top",
            fontsize=8.4, color=INK, zorder=3)

    # layer captions
    cap_y = 0.935
    ax.text(x_in, cap_y, f"input\n{len(feats)} features", ha="center", va="center",
            fontsize=8.0, color=MUTED, linespacing=1.25)
    ax.text((x_h1 + x_h2) / 2, cap_y, "2 hidden layers · 32 units each\nReLU · dropout 0.1",
            ha="center", va="center", fontsize=8.0, color=MUTED, linespacing=1.25)
    ax.text(x_out, cap_y, "output\n1 unit (linear)", ha="center", va="center",
            fontsize=8.0, color=MUTED, linespacing=1.25)

    ax.set_title(title, loc="left", fontsize=9.4, color=INK, pad=14)
    ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.0,
            color=MUTED, va="bottom")
    return x_out, out_y


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0), dpi=200)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.845, bottom=0.05, wspace=0.06)

    # ── Pure MLP ────────────────────────────────────────────────────────────
    ax = axes[0]
    draw_net(ax, PURE_FEATS, PURPLE, r"$\log\,$TFLOPs",
             "Pure MLP — predicts log(TFLOPs) directly",
             "absolute log-features: sizes E, t, TB, NV plus power & intensity ratios")
    ax.text(0.5, 0.015,
            r"TFLOPs $= \exp(\,\mathrm{MLP}(x)\,)$"
            "\nfeatures z-scored on the training fold; target log-standardized",
            ha="center", va="bottom", fontsize=8.2, color=INK2, linespacing=1.4)

    # ── Residual MLP ───────────────────────────────────────────────────────
    ax = axes[1]
    draw_net(ax, RESID_FEATS, AQUA, r"$g(z)$",
             "Residual MLP — physics-anchored correction",
             "dimensionless intensity ratios only (no job-size features)")
    # backbone box feeding the final multiply
    bb = FancyBboxPatch((0.20, 0.028), 0.44, 0.075,
                        boxstyle="round,pad=0.012", fc=SURFACE, ec=BLUE, lw=1.1,
                        zorder=3)
    ax.add_patch(bb)
    ax.text(0.42, 0.0655, "2-param physics backbone\n"
            r"est$_{2p} = (E - P_{oh}\,t)\,/\,E_{marg}$",
            ha="center", va="center", fontsize=8.0, color=INK2, linespacing=1.3,
            zorder=4)
    ax.text(0.71, 0.0655,
            r"TFLOPs $=$ est$_{2p}\cdot e^{\,g(z)}$",
            ha="left", va="center", fontsize=8.6, color=INK, zorder=4)
    ax.add_patch(FancyArrowPatch((0.648, 0.0655), (0.695, 0.0655),
                                 arrowstyle="-|>", mutation_scale=9,
                                 color=INK2, lw=1.0, zorder=4))
    ax.add_patch(FancyArrowPatch((0.89, 0.42), (0.85, 0.125),
                                 arrowstyle="-|>", mutation_scale=9,
                                 color=INK2, lw=1.0, zorder=4,
                                 connectionstyle="arc3,rad=-0.15"))
    ax.text(0.845, 0.285, r"$g\!=\!0 \Rightarrow$ pure physics" "\nestimate",
            ha="right", va="center", fontsize=7.6, color=MUTED, linespacing=1.25)

    fig.suptitle("Neural FLOP estimator architecture — MLP(d_in, 32, 32, 1) over "
                 "log/ratio features of the sensor aggregates (E, t, DRAM, NVLink)",
                 fontsize=9.6, color=INK, y=0.975)
    path = os.path.join(OUT, "fig_nn_arch.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
