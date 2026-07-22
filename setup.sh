#!/usr/bin/env bash
# One-time environment setup for the FLOP-estimation experiment.
# PORTED target: x86 dual-Tesla-V100 server, Ubuntu 24.04, NVIDIA driver 580 /
# CUDA 13 runtime. (The original targeted a Jetson Orin Nano; see git history.)
# Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"

echo "== 1/4 python venv =="
[ -d .venv ] || python3 -m venv .venv
PIP=".venv/bin/python3 -m pip"

echo "== 2/4 python packages =="
# Standard PyPI torch: the cu12x wheels bundle sm_70 kernels, so they run on the
# V100 (no special index needed, unlike the Jetson's sm_87 wheels).
$PIP install --upgrade pip >/dev/null
$PIP install torch numpy matplotlib

echo "== 3/4 DCGM host engine (root; for the DRAM-active signal) =="
# The DRAM-activity signal is DCGM field 1005 (DCGM_FI_PROF_DRAM_ACTIVE), read by
# unprivileged `dcgmi` clients through a root nv-hostengine. Profiling counters
# are admin-only (RmProfilingAdminOnly=1), so the host engine must run as root.
if pgrep -x nv-hostengine >/dev/null; then
    echo "nv-hostengine already running."
elif dcgmi discovery -l >/dev/null 2>&1; then
    echo "dcgmi can reach a host engine already."
else
    echo "nv-hostengine is not running. Start it as root (one time):"
    echo "    sudo nv-hostengine"
    echo "then re-run this script (or just the sanity checks below)."
fi

echo "== 4/4 sanity checks =="
.venv/bin/python3 -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), torch.cuda.get_arch_list())"
if pgrep -x nv-hostengine >/dev/null || dcgmi discovery -l >/dev/null 2>&1; then
    .venv/bin/python3 power_calibration/power_monitor_test.py
else
    echo "(skipping sensor sanity check — start nv-hostengine first)"
fi

echo "setup complete"
