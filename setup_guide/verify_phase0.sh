#!/bin/bash
# Phase 0 Verification Script — Verification Games
# Tests all monitoring primitives (except GPS and dark compute)

PASS=0
FAIL=0
WARN=0

pass() { echo "  [PASS] $1"; ((PASS++)); }
fail() { echo "  [FAIL] $1"; ((FAIL++)); }
warn() { echo "  [WARN] $1"; ((WARN++)); }

echo "==========================================="
echo "Phase 0: Environment Setup & Sensor Verification"
echo "==========================================="
echo ""

# --- 0.1 INA3221 Power Monitor ---
echo "--- 0.1 INA3221 Power Monitor ---"

HWMON="/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1"

if [ -d "$HWMON" ]; then
    pass "INA3221 hwmon directory exists"
else
    fail "INA3221 hwmon directory not found at $HWMON"
fi

for label_file in "$HWMON"/in{1,2,3}_label; do
    if [ -f "$label_file" ]; then
        label=$(cat "$label_file")
        input_file="${label_file/_label/_input}"
        if [ -f "$input_file" ]; then
            val=$(cat "$input_file")
            if [ "$val" -gt 0 ] 2>/dev/null; then
                pass "$label voltage: ${val} mV"
            else
                fail "$label voltage reading invalid: $val"
            fi
        else
            fail "$label input file missing"
        fi
    fi
done

for i in 1 2 3; do
    curr_file="$HWMON/curr${i}_input"
    if [ -f "$curr_file" ]; then
        val=$(cat "$curr_file")
        if [ "$val" -ge 0 ] 2>/dev/null; then
            pass "Channel $i current: ${val} mA"
        else
            fail "Channel $i current reading invalid: $val"
        fi
    else
        fail "Channel $i current file missing"
    fi
done

if command -v tegrastats &>/dev/null; then
    ts_output=$(timeout 2 tegrastats 2>&1 | head -1)
    if echo "$ts_output" | grep -q "VDD_IN"; then
        pass "tegrastats reports power data"
    else
        fail "tegrastats output unexpected: $ts_output"
    fi
else
    fail "tegrastats not found"
fi

echo ""

# --- 0.2 CUDA FLOP Profiling ---
echo "--- 0.2 CUDA FLOP Profiling ---"

if command -v ncu &>/dev/null; then
    ncu_ver=$(ncu --version 2>&1 | grep "Version" | head -1)
    pass "ncu installed: $ncu_ver"
else
    fail "ncu (Nsight Compute) not found"
fi

if command -v nvidia-smi &>/dev/null; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "")
    if [ -n "$gpu_name" ]; then
        pass "nvidia-smi reports GPU: $gpu_name"
    else
        # Jetson may not support --query-gpu; try basic nvidia-smi
        if nvidia-smi &>/dev/null; then
            pass "nvidia-smi runs successfully"
        else
            fail "nvidia-smi fails"
        fi
    fi
else
    fail "nvidia-smi not found"
fi

# PyTorch CUDA check
pytorch_result=$(python3 -W ignore -c "
import sys, warnings
warnings.filterwarnings('ignore')
try:
    import torch
except ImportError:
    print('NO_TORCH'); sys.exit()
if not torch.cuda.is_available():
    print('NO_CUDA'); sys.exit()
try:
    a = torch.randn(64, 64, device='cuda')
    b = a @ a
    print('GPU_OK')
except Exception as e:
    print(f'GPU_FAIL:{e}')
" 2>/dev/null)

case "$pytorch_result" in
    GPU_OK)
        pass "PyTorch GPU compute works" ;;
    NO_TORCH)
        fail "PyTorch not installed" ;;
    NO_CUDA)
        fail "torch.cuda.is_available() = False" ;;
    GPU_FAIL*)
        fail "PyTorch CUDA init OK but GPU ops fail: ${pytorch_result#GPU_FAIL:}" ;;
    *)
        fail "PyTorch GPU check unexpected: $pytorch_result" ;;
esac

echo ""

# --- 0.3 Network Traffic Monitor ---
echo "--- 0.3 Network Traffic Monitor ---"

if command -v tcpdump &>/dev/null; then
    td_ver=$(tcpdump --version 2>&1 | head -1)
    pass "tcpdump installed: $td_ver"
else
    fail "tcpdump not installed"
fi

if command -v tc &>/dev/null; then
    tc_ver=$(tc -Version 2>&1)
    pass "tc installed: $tc_ver"
else
    fail "tc (traffic control) not found"
fi

if ip link show wlP1p1s0 &>/dev/null; then
    state=$(ip link show wlP1p1s0 | grep -oP 'state \K\w+')
    if [ "$state" = "UP" ] || [ "$state" = "DORMANT" ]; then
        pass "WiFi interface wlP1p1s0: $state"
    else
        warn "WiFi interface wlP1p1s0 exists but state is $state"
    fi
else
    fail "WiFi interface wlP1p1s0 not found"
fi

echo ""

# --- User & Permission Model ---
echo "--- User & Permission Model ---"

if id redteam &>/dev/null; then
    pass "redteam user exists (uid=$(id -u redteam))"
else
    fail "redteam user does not exist"
fi

if [ -d /home/redteam ]; then
    perms=$(stat -c '%a' /home/redteam)
    if [ "$perms" = "700" ]; then
        pass "redteam home directory permissions: 700"
    else
        warn "redteam home directory permissions: $perms (expected 700)"
    fi
else
    fail "/home/redteam does not exist"
fi

echo ""

# --- Python Dependencies ---
echo "--- Python Dependencies ---"

for pkg in torch flask scapy requests sqlite3; do
    if python3 -c "import $pkg" 2>/dev/null; then
        pass "Python package: $pkg"
    else
        fail "Python package missing: $pkg"
    fi
done

echo ""

# --- LD_LIBRARY_PATH ---
echo "--- Library Path ---"

if echo "$LD_LIBRARY_PATH" | grep -q "cuda"; then
    pass "LD_LIBRARY_PATH includes CUDA: $LD_LIBRARY_PATH"
else
    warn "LD_LIBRARY_PATH does not include CUDA libs (torch may fail to load)"
fi

echo ""

# --- Summary ---
echo "==========================================="
echo "SUMMARY: $PASS passed, $FAIL failed, $WARN warnings"
echo "==========================================="

if [ "$FAIL" -gt 0 ]; then
    exit 1
else
    exit 0
fi
