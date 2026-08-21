#!/usr/bin/env python3
"""
make_fig_nn_arch.py — Figure 9: architecture of the two neural estimators.
  Left  — Pure MLP: log(TFLOPs) straight from log/ratio sensor features.
  Right — Residual MLP: TFLOPs = est_2param(physics) × exp(g(z)); the net only
          learns the efficiency correction from dimensionless intensity features.

    python3 writeup/make_fig_nn_arch.py   ->   writeup/fig_nn_arch.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE, BLUE, AQUA, AMBER, RED, PURPLE = ("#e1e0d9", "#c3c2b7", "#7b9fd4",
                                              "#7fbfa4", "#e3bc70", "#d29393", "#a795d4")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"]})


def layer(ax, x, ys, r=0.9, fc="#ffffff", ec=BLUE):
    for y in ys:
        ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=ec, lw=1.3, zorder=4))
    return [(x, y) for y in ys]


def connect(ax, a, b, col=BASE, alpha=0.5):
    for (x0, y0) in a:
        for (x1, y1) in b:
            ax.plot([x0, y0 * 0 + x0 + 0.9, x1 - 0.9, x1][::1], [y0, y0, y1, y1][::1],
                    color=col, lw=0.5, alpha=alpha, zorder=1)


def simple_edges(ax, a, b, col=BASE, alpha=0.45):
    for (x0, y0) in a:
        for (x1, y1) in b:
            ax.plot([x0 + 0.9, x1 - 0.9], [y0, y1], color=col, lw=0.5, alpha=alpha, zorder=1)


def hidden_ys(n_show, cy, gap=1.9):
    ys = [cy + (i - (n_show - 1) / 2) * gap for i in range(n_show)]
    return ys


def pure_panel(ax):
    ax.set_title("Pure MLP", fontsize=17, color=INK, loc="left", weight="bold")
    feats = ["log E", "log t", "log D", "log N", "log P", "log E/D", "log N/D"]
    iy = hidden_ys(len(feats), 0, gap=2.0)
    inp = layer(ax, 0, iy, ec=BLUE)
    for (x, y), t in zip(inp, feats):
        ax.text(x - 1.4, y, t, ha="right", va="center", fontsize=11.5, color=INK2,
                family="monospace")
    h1 = layer(ax, 7, hidden_ys(5, 0), ec=PURPLE)
    h2 = layer(ax, 13, hidden_ys(5, 0), ec=PURPLE)
    out = layer(ax, 19, [0], r=1.0, ec=AMBER)
    simple_edges(ax, inp, h1); simple_edges(ax, h1, h2); simple_edges(ax, h2, out)
    ax.text(6.4, 6.4, "hidden ×32\nReLU", ha="center", fontsize=11, color=PURPLE)
    ax.text(13.6, 6.4, "hidden ×32\nReLU", ha="center", fontsize=11, color=PURPLE)
    ax.text(19, 2.2, "log T̂", ha="center", fontsize=12, color=AMBER, weight="bold")
    ax.annotate("", xy=(23, 0), xytext=(20, 0),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.6))
    ax.text(23.4, 0, "exp(·) → TFLOPs", ha="left", va="center", fontsize=11.5, color=INK)
    ax.set_xlim(-6.5, 31.5); ax.set_ylim(-8.6, 8.6)


def resid_panel(ax):
    ax.set_title("Residual MLP", fontsize=17, color=INK,
                 loc="left", weight="bold")
    feats = ["log P", "log E/D", "log N/D", "log D/t", "log N/t", "log t"]
    iy = hidden_ys(len(feats), 1.5, gap=1.9)
    inp = layer(ax, 0, iy, ec=BLUE)
    for (x, y), t in zip(inp, feats):
        ax.text(x - 1.4, y, t, ha="right", va="center", fontsize=11.5, color=INK2,
                family="monospace")
    h1 = layer(ax, 6.5, hidden_ys(4, 1.5), ec=PURPLE)
    h2 = layer(ax, 12, hidden_ys(4, 1.5), ec=PURPLE)
    gout = layer(ax, 17, [1.5], r=1.0, ec=PURPLE)
    simple_edges(ax, inp, h1); simple_edges(ax, h1, h2); simple_edges(ax, h2, gout)
    ax.text(6.2, 6.2, "hidden ×32", ha="center", fontsize=11, color=PURPLE)
    ax.text(12.4, 6.2, "hidden ×32", ha="center", fontsize=11, color=PURPLE)
    ax.text(17, 3.3, "g(z)", ha="center", fontsize=12.5, color=PURPLE, weight="bold")

    # physics branch
    ax.add_patch(FancyBboxPatch((-4.8, -8.6), 20.4, 3.0, boxstyle="round,pad=0.3,rounding_size=0.6",
                                facecolor="#eef6f2", edgecolor=AQUA, lw=1.3, zorder=3))
    ax.text(5.4, -7.1, "power-only estimator:  (E − p·t) / a", ha="center", va="center",
            fontsize=11, color=AQUA, family="monospace")

    # combine node
    cx, cy = 21.5, -2.0
    ax.add_patch(Circle((cx, cy), 1.15, facecolor="#fff", edgecolor=INK, lw=1.5, zorder=5))
    ax.text(cx, cy, "×", ha="center", va="center", fontsize=19, color=INK, zorder=6)
    ax.annotate("", xy=(cx - 0.9, cy + 0.7), xytext=(17, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.5,
                                connectionstyle="arc3,rad=-0.15"))
    ax.text(17.4, -1.0, "exp(g(z))", ha="center", fontsize=10.5, color=PURPLE)
    ax.annotate("", xy=(cx - 0.9, cy - 0.6), xytext=(15.8, -7.1),
                arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=1.5,
                                connectionstyle="arc3,rad=0.15"))
    ax.annotate("", xy=(cx + 2.8, cy), xytext=(cx + 1.2, cy),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.7))
    ax.text(cx + 3.1, cy, "TFLOPs", ha="left", va="center", fontsize=12.5, color=INK, weight="bold")
    ax.set_xlim(-6.5, 28); ax.set_ylim(-10.4, 8.6)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE); ax.axis("off"); ax.set_aspect("equal")
    pure_panel(axes[0]); resid_panel(axes[1])
    fig.tight_layout()
    for name in ("fig_nn_arch.png", "figure_4.png"):
        fig.savefig(os.path.join(OUT, name), facecolor=SURFACE,
                    bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("wrote writeup/fig_nn_arch.png")


if __name__ == "__main__":
    main()
