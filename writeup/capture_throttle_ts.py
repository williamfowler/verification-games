#!/usr/bin/env python3
"""
capture_throttle_ts.py — capture GPU util + power over time for a benign run and an
S2-throttle run of the same workload, for Figure 5. High-frequency nvidia-smi polling
in a background thread while each workload runs under torchrun.

    python3 writeup/capture_throttle_ts.py   ->   writeup/throttle_ts.json
"""
import json
import os
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_TORCHRUN = os.path.join(REPO, ".venv", "bin", "torchrun")
ADV = os.path.join(REPO, "red_team", "adversarial_workload.py")
OUT = os.path.join(REPO, "writeup", "throttle_ts.json")

CFG = ["--steps", "150", "--batch-size", "16", "--seq-len", "512", "--d-model", "1024",
       "--num-layers", "12", "--nhead", "8", "--dim-feedforward", "4096", "--precision", "fp16"]

samples = []          # (t, util0, power_total_W, phase)
marks = {}            # phase -> {"start": t, "end": t}  (compute window)
_phase = "idle"
_stop = False


def poller():
    while not _stop:
        try:
            r = subprocess.run(["nvidia-smi",
                                "--query-gpu=utilization.gpu,power.draw",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=1.0)
            lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
            util0 = float(lines[0].split(",")[0])
            pw = sum(float(l.split(",")[1]) for l in lines)
            samples.append((time.monotonic(), util0, pw, _phase))
        except Exception:
            pass
        time.sleep(0.08)


def run(strategy, extra):
    global _phase
    cmd = [VENV_TORCHRUN, "--nproc_per_node=2", ADV, "--strategy", strategy, *CFG, *extra]
    print(f"running {strategy}", flush=True)
    _phase = strategy
    marks[strategy] = {}
    p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    for line in iter(p.stdout.readline, ""):
        if "Starting workload" in line:
            marks[strategy]["start"] = time.monotonic()
    p.wait()
    marks[strategy]["end"] = time.monotonic()
    _phase = "idle"


def main():
    global _stop
    th = threading.Thread(target=poller, daemon=True); th.start()
    time.sleep(3.0)                       # idle baseline
    run("none", [])                       # benign reference
    time.sleep(4.0)
    run("throttle", ["--target-util", "60"])
    time.sleep(2.0)
    _stop = True; th.join(timeout=2.0)
    t0 = samples[0][0]
    data = {"t": [s[0] - t0 for s in samples], "util0": [s[1] for s in samples],
            "power_w": [s[2] for s in samples], "phase": [s[3] for s in samples],
            "marks": {k: {kk: vv - t0 for kk, vv in v.items()} for k, v in marks.items()}}
    json.dump(data, open(OUT, "w"))
    print(f"wrote {OUT}  ({len(samples)} samples)  marks={data['marks']}")


if __name__ == "__main__":
    main()
