"""
power_monitor_test.py — sensor sanity check (ported to x86 dual-V100)

The Jetson version read INA3221 VDD_CPU_GPU_CV power from hwmon sysfs. On this
box the two ported signals are checked instead:
  - GPU board power + utilization via nvidia-smi (unprivileged), and
  - DRAM-active fraction (DCGM field 1005) via `dcgmi dmon` — needs a running
    nv-hostengine (start with `sudo nv-hostengine`).

Prints 10 samples of each. Deliberately self-contained apart from the shared
readers in detect_flops / calibrate_power.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detect_flops import GPU_INDEX, read_gpu_sample, V100_PROFILE
from calibrate_power import BytesSampler

POLL_INTERVAL = 1.0


def main():
    print(f"Monitored GPU index: {GPU_INDEX}\n")

    print("nvidia-smi power / utilization:")
    for i in range(10):
        try:
            mw, gpu, clk = read_gpu_sample(GPU_INDEX)
            print(f"  sample {i+1:>2d}: {mw:>9.1f} mW  ({mw/1000:.3f} W)"
                  f"  |  util {gpu:>5.1f}%  |  {clk/1e6:.0f} MHz")
        except Exception as e:  # noqa: BLE001 - smoke test, show the failure
            print(f"  sample {i+1:>2d}: ERROR — {e}")
            break
        time.sleep(POLL_INTERVAL)

    print("\nDCGM DRAM-active (field 1005) via dcgmi dmon:")
    peak = V100_PROFILE["PEAK_BW_BYTES_S"]
    sampler = BytesSampler(GPU_INDEX)
    sampler.start()
    time.sleep(10 * POLL_INTERVAL)
    sampler.stop()
    try:
        sampler.require_samples()
        for i, (_, bps) in enumerate(sampler.samples[:10]):
            print(f"  sample {i+1:>2d}: {bps/peak*100:>6.2f}% of peak"
                  f"  ({bps/1e9:.2f} GB/s)")
    except RuntimeError as e:
        print(f"  ERROR — {e}")

    print("\nDone.")


main()
