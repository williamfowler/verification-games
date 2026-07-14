#!/usr/bin/env python3
"""
test_subprocess.py — Diagnostic + subprocess test for calibrate_power.py.

Tests:
  1. Whether the INA3221 sysfs files are readable without root
  2. Whether a workload subprocess launched directly (no su) works

Run as:  python3 test_subprocess.py        (preferred)
         sudo python3 test_subprocess.py   (to compare)

Deliberately self-contained (constants/helpers duplicated from detect_flops.py /
calibrate_power.py): a diagnostic for sensor permissions and subprocess launch
shouldn't depend on the code it sanity-checks.
"""

import glob
import os
import subprocess
import sys

INA3221_DRIVER = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon"
INA3221_LABEL  = "VDD_CPU_GPU_CV"
WORKLOAD_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sample_ml_workload.py")


def find_ina3221_paths():
    for hwmon in glob.glob(f"{INA3221_DRIVER}/hwmon*"):
        for i in range(1, 5):
            try:
                with open(f"{hwmon}/in{i}_label") as f:
                    if f.read().strip() == INA3221_LABEL:
                        return f"{hwmon}/in{i}_input", f"{hwmon}/curr{i}_input"
            except OSError:
                continue
    return None, None


def find_venv_python():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for name in ("python3", "python"):
        p = os.path.join(script_dir, ".venv", "bin", name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return sys.executable


def main():
    target_python = find_venv_python()

    print("=== Identity ===")
    print(f"  uid            : {os.getuid()}  (0 = root)")
    print(f"  sys.executable : {sys.executable}")
    print(f"  target python  : {target_python}")
    print()

    # ── Check 1: INA3221 readability ──────────────────────────────────────
    print("=== INA3221 sensor readability ===")
    volt_path, curr_path = find_ina3221_paths()
    if volt_path is None:
        print("  Could not locate INA3221 paths.")
    else:
        for path in (volt_path, curr_path):
            try:
                with open(path) as f:
                    val = f.read().strip()
                stat = os.stat(path)
                perm = oct(stat.st_mode)[-3:]
                print(f"  OK  {path}  (perm={perm}, value={val})")
            except PermissionError:
                print(f"  DENIED  {path}  — need root or group membership")
            except OSError as e:
                print(f"  ERROR   {path}  — {e}")

    if os.getuid() != 0:
        print()
        print("  Running as non-root and sensor is readable → no sudo needed.")
        print("  Run calibrate_power.py directly: python3 calibrate_power.py")
    print()

    # ── Check 2: direct subprocess (no su) ───────────────────────────────
    print("=== Workload subprocess (direct, no su) ===")
    cmd = [
        target_python, WORKLOAD_SCRIPT,
        "--steps", "10",
        "--batch-size", "4",
        "--seq-len", "32",
        "--d-model", "128",
    ]
    print(f"  cmd: {' '.join(cmd)}")
    print()

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    print()
    print(f"=== Exit code: {proc.returncode} ===")
    print("SUCCESS" if proc.returncode == 0 else "FAILED — see output above.")


if __name__ == "__main__":
    main()
