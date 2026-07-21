"""Plot the captured idle->workload->idle session: power and DRAM bandwidth
as two separate standalone figures (fig4_power_timeseries.png,
fig5_dram_timeseries.png). Reads timeseries.json from this directory."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from detect_flops import ORIN_PROFILE, FALLBACK_IDLE_POWER_MW

ACTMON_K = 0.0133          # measured actmon-TB per true TB (actmon_scale_bench)
# NB: display denominator only. ORIN_PROFILE's 68 GB/s is the original Orin
# Nano peak (2133 MHz LPDDR5) and is stale for this "Super" board, whose EMC
# runs at 3199 MHz under load -> 102.4 GB/s. The estimator itself never uses
# this (its byte scale is absorbed into E_PER_TB_J); only this axis does.
TRUE_PEAK_BW = 3199e6 * 32          # 102.4 GB/s at the loaded EMC clock
NOMINAL_PEAK_BW = ORIN_PROFILE["PEAK_BW_BYTES_S"]   # actmon-scale conversion

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

d = json.load(open(os.path.join(HERE, "timeseries.json")))
pt = [t / 60 for t, _ in d["power_mw"]]
pw = [mw / 1000 for _, mw in d["power_mw"]]
bt = [t / 60 for t, _ in d["dram_bytes_s"]]
bw = [b / ACTMON_K / TRUE_PEAK_BW * 100 for _, b in d["dram_bytes_s"]]
t_start, t_end = d["t_start"] / 60, d["t_end"] / 60
t_max = pt[-1]


def session_chrome(ax, ymax):
    """Shared idle/workload/idle annotation layer."""
    ax.axvspan(t_start, t_end, color=BLUE, alpha=0.06, zorder=1)
    for x in (t_start, t_end):
        ax.axvline(x, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.text((t_start + t_end) / 2, ymax, "Training Workload", ha="center",
            fontsize=8.5, color=INK2)
    ax.text(t_start / 2, ymax, "Idle", ha="center", fontsize=8.5, color=INK2)
    ax.text((t_end + t_max) / 2, ymax, "Idle", ha="center", fontsize=8.5,
            color=INK2)
    ax.set_xlabel("Time (Minutes)")
    ax.set_xlim(0, t_max)


# ── Figure 4: power ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
ax.plot(pt, pw, color=BLUE, lw=1.4, zorder=3)
ax.axhline(FALLBACK_IDLE_POWER_MW / 1000, color=MUTED, lw=0.9,
           ls=(0, (2, 3)), zorder=2)
ax.text((t_start + t_end) / 2, FALLBACK_IDLE_POWER_MW / 1000 + 0.25,
        "calibrated idle baseline (0.64 W)", ha="center", fontsize=7.5,
        color=INK2)
ax.set_ylabel("Power (W)")
ax.set_ylim(0, max(pw) * 1.18)
session_chrome(ax, max(pw) * 1.10)
ax.set_title("Power over Time During a Sample Workload",
             fontsize=9.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig4_power_timeseries.png"))
plt.close(fig)

# ── Figure 5: DRAM bandwidth ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
ax.plot(bt, bw, color=AQUA, lw=1.4, zorder=3)
ax.set_ylabel("Memory Bandwidth Utilization (%)")
ax.set_ylim(0, max(bw) * 1.25)
session_chrome(ax, max(bw) * 1.16)
ax.set_title("Memory Bandwidth over Time During a Sample Workload",
             fontsize=9.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig5_dram_timeseries.png"))
plt.close(fig)

# ── Figure 5 variant: with a 10 s running average ─────────────────────────────
# Samples arrive every ~0.5 s, so a 10 s window = 20 samples (centered).
import numpy as np
WIN = 20
bw_arr = np.array(bw)
avg = np.convolve(bw_arr, np.ones(WIN) / WIN, mode="valid")
avg_t = bt[WIN // 2 - 1: WIN // 2 - 1 + len(avg)]

fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
ax.plot(bt, bw, color=AQUA, lw=1.0, alpha=0.35, zorder=3,
        label="per-sample (0.5 s)")
ax.plot(avg_t, avg, color=AQUA, lw=1.8, zorder=4, label="10 s running average")
ax.set_ylabel("Memory Bandwidth Utilization (%)")
ax.set_ylim(0, max(bw) * 1.25)
session_chrome(ax, max(bw) * 1.16)
ax.legend(fontsize=7.5, frameon=False, loc="lower center", ncol=2)
ax.set_title("Memory Bandwidth over Time During a Sample Workload",
             fontsize=9.5, color=INK, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig5_dram_timeseries_avg.png"))
plt.close(fig)

print("wrote fig4_power_timeseries.png, fig5_dram_timeseries.png"
      " and fig5_dram_timeseries_avg.png")
