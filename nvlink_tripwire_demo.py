#!/usr/bin/env python3
"""
nvlink_tripwire_demo.py — the interconnect signal as an adversarial tripwire.

The energy→FLOP estimator can be fooled by a workload that manipulates its power
signature. The DDP gradient all-reduce is a second, independent observable that a
spoof cannot fake consistently: honest fp16-DDP training pins NVLink volume to
model size, so NVLink_bytes / net_energy_J stays in a bounded band
(detect_flops.nvlink_consistency). This demo takes a real honest run from the
sweep and two synthetic spoofs, and shows the energy-only estimator is fooled
while the NVLink tripwire flags both.

    python3 nvlink_tripwire_demo.py [records.json]
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "power_calibration"))

import detect_flops as d
from eval_power_monitor import valid, is_frontier

path = sys.argv[1] if len(sys.argv) > 1 else "eval_results_v100_ddp_records.json"
recs = json.load(open(path))["records"]
frontier = [r for r in valid(recs)
            if r["config"].get("precision") == "fp16" and is_frontier(r)]
h = max(frontier, key=lambda r: r["ground_truth_tf"])   # a large honest run
E, NV, dur, gt = h["net_energy_j"], h["nvlink_total_bytes"], h["duration_s"], h["ground_truth_tf"]


def show(label, e, nv):
    est = d.estimate_tflops(e, dur)
    verdict, ratio = d.nvlink_consistency(e, nv)
    flag = "" if verdict == "OK" else "   <-- FLAGGED"
    print(f"  {label:42s} energy-est {est:7.0f} TF ({est/gt:4.1f}× true)  "
          f"NVLink/J {ratio:8.2e}  [{verdict}]{flag}")


print(f"Honest reference: {h['label']}  (ground truth {gt:.0f} TFLOPs, aggregate 2-GPU)\n")
print(f"  {'scenario':42s} {'FLOP estimate (energy only)':28s} {'interconnect tripwire'}")
show("honest run",                              E,       NV)
show("Spoof A: 2× energy, no all-reduce",       E * 2.0, NV * 0.001)
show("Spoof B: NVLink flood, 1.5× energy",      E * 1.5, NV * 50)
print("\nThe energy-only estimator over-reports on both spoofs; the NVLink"
      "\nconsistency tripwire (band [%.1e, %.1e] bytes/J) flags both."
      % (d.NVLINK_BYTES_PER_J_LO, d.NVLINK_BYTES_PER_J_HI))
