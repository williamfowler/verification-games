#!/usr/bin/env bash
# One-time environment setup for the FLOP-estimation experiment.
# Target: Jetson Orin Nano 8GB, JetPack 6.2 (L4T R36.5.0), Ubuntu 22.04.
# Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"

echo "== 1/5 system CUDA packages (cupti is required by torch) =="
sudo apt-get update
sudo apt-get install -y cuda-cupti-12-6

echo "== 2/5 python venv =="
[ -d .venv ] || python3 -m venv --system-site-packages .venv
# python3 -m pip (not .venv/bin/pip): the pip script's shebang bakes in the
# repo's absolute path and breaks if the repo is ever moved or renamed.
PIP=".venv/bin/python3 -m pip"

echo "== 3/5 python packages =="
# torch needs the Jetson AI Lab wheel (sm_87). --no-deps on the torch/nvidia
# installs prevents pip pulling standard PyPI nvidia-* packages that conflict
# with the system CUDA 12.6 libraries.
$PIP install --no-deps torch==2.9.1 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/
$PIP install --no-deps nvidia-cudss-cu12
$PIP install jetson-stats numpy matplotlib

echo "== 4/5 LD_LIBRARY_PATH =="
LDLINE="export LD_LIBRARY_PATH=$REPO/.venv/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/local/cuda-12.6/lib64:\$LD_LIBRARY_PATH"
if ! grep -qF "$LDLINE" ~/.bashrc; then
    echo "$LDLINE" >> ~/.bashrc
    echo "appended LD_LIBRARY_PATH export to ~/.bashrc (open a new shell or 'source ~/.bashrc')"
fi

echo "== 5/5 sudoers entry for the actmon DRAM reader =="
# eval_power_monitor.py must run unprivileged (CUDA needs the user venv) but
# reads root-only debugfs counters through actmon_reader.py via 'sudo -n'.
SUDOERS_FILE=/etc/sudoers.d/actmon-reader
SUDOERS_LINE="$USER ALL=(root) NOPASSWD: $REPO/.venv/bin/python3 $REPO/power_calibration/actmon_reader.py *"
echo "$SUDOERS_LINE" | sudo tee "$SUDOERS_FILE" > /dev/null
sudo chmod 440 "$SUDOERS_FILE"
sudo visudo -c -f "$SUDOERS_FILE"

echo "== sanity checks =="
.venv/bin/python3 -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), torch.cuda.get_arch_list())"
.venv/bin/python3 power_calibration/power_monitor_test.py

echo "setup complete"
