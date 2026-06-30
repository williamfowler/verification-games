"""
power_monitor_test.py — INA3221 sensor sanity check
Reads VDD_CPU_GPU_CV power (V × I) from hwmon sysfs and prints 10 samples.
"""

import time
import glob

DRIVER_PATH = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon"
TARGET_LABEL = "VDD_CPU_GPU_CV"
POLL_INTERVAL = 1.0


def find_channel():
    """Return (volt_path, curr_path) for the TARGET_LABEL channel, or raise."""
    hwmon_dirs = glob.glob(f"{DRIVER_PATH}/hwmon*")
    if not hwmon_dirs:
        raise FileNotFoundError(f"No hwmon device found under {DRIVER_PATH}")
    hwmon = hwmon_dirs[0]
    for i in range(1, 5):
        label_path = f"{hwmon}/in{i}_label"
        try:
            with open(label_path) as f:
                if f.read().strip() == TARGET_LABEL:
                    return f"{hwmon}/in{i}_input", f"{hwmon}/curr{i}_input"
        except OSError:
            continue
    raise FileNotFoundError(f"Channel '{TARGET_LABEL}' not found in {hwmon}")


def read_power_mw(volt_path, curr_path):
    with open(volt_path) as f:
        mv = float(f.read().strip())
    with open(curr_path) as f:
        ma = float(f.read().strip())
    return mv * ma / 1000.0  # mV × mA / 1000 = mW


def main():
    volt_path, curr_path = find_channel()
    print(f"Found {TARGET_LABEL}:")
    print(f"  voltage : {volt_path}")
    print(f"  current : {curr_path}\n")

    for i in range(10):
        try:
            mw = read_power_mw(volt_path, curr_path)
            print(f"  sample {i+1:>2d}: {mw:>8.1f} mW  ({mw/1000:.3f} W)")
        except OSError as e:
            print(f"  sample {i+1:>2d}: ERROR — {e}")
            break
        time.sleep(POLL_INTERVAL)
    print("\nDone.")


main()
