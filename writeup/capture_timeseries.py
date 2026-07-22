"""Capture power + DRAM-bandwidth time series across idle -> workload -> idle,
save raw series to JSON for reproducibility."""
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "power_calibration"))
os.chdir(REPO)

from calibrate_power import (GPU_INDEX, PowerSampler, BytesSampler,
                             find_venv_python, child_env, POLL_S)

PRE_S, POST_S = 60.0, 60.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "timeseries.json")

power = PowerSampler(GPU_INDEX)
dram = BytesSampler(GPU_INDEX)
t0 = time.monotonic()
power.start()
dram.start()

print(f"pre-roll idle {PRE_S:.0f}s...", flush=True)
time.sleep(PRE_S)

cmd = [find_venv_python(), "-u", "sample_ml_workload.py",
       "--steps", "9000", "--batch-size", "16", "--seq-len", "128",
       "--d-model", "384"]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True,
                        env=child_env(GPU_INDEX))
t_launch = time.monotonic() - t0
t_start = None
for line in iter(proc.stdout.readline, ""):
    if "[redteam] Starting workload" in line:
        t_start = time.monotonic() - t0
    if "Ground truth total" in line or "steps/s" in line:
        pass
proc.wait()
t_end = time.monotonic() - t0
print(f"workload done (launch {t_launch:.1f}s, start {t_start:.1f}s,"
      f" end {t_end:.1f}s)", flush=True)

print(f"post-roll idle {POST_S:.0f}s...", flush=True)
time.sleep(POST_S)

power.stop()
dram.stop()
dram.require_samples()

t_base = power.power_samples[0][0]
payload = {
    "power_mw": [(t - t_base, mw) for t, mw in power.power_samples],
    "gpu_pct": [(t - t_base, g) for t, g in power.gpu_samples],
    "dram_bytes_s": [(t - t_base, b) for t, b in dram.samples],
    "t_launch": t_launch, "t_start": t_start, "t_end": t_end,
}
json.dump(payload, open(OUT, "w"))
print(f"saved {len(payload['power_mw'])} power / {len(payload['dram_bytes_s'])}"
      f" dram samples -> {OUT}")
