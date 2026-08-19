"""
collect_live_v2.py — re-run S1 split / S2 throttle / benign-none at a ground truth
comparable to the benign frontier (~2500 TF), so the live-daemon strategies land in
the same GT band as the offline estimator points (not below the benign floor).

Same shape as the S5 parent (d1024_s512_b16_L12) at 160 steps (~2600 TF aggregate),
matching the benign-frontier median. Drives live_daemon_probe.py once per strategy;
writes red_team/live_v2_{none,split,throttle}.json.

    python3 red_team/collect_live_v2.py
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(REPO, "red_team", "live_daemon_probe.py")
PY = os.path.join(REPO, ".venv", "bin", "python")
if not os.path.exists(PY):
    PY = sys.executable

COMMON = ["--d-model", "1024", "--seq-len", "512", "--batch-size", "16",
          "--num-layers", "12", "--nhead", "8", "--dim-feedforward", "4096",
          "--steps", "160"]
RUNS = [
    ("none",     ["--strategy", "none"]),
    ("split",    ["--strategy", "split", "--chunk-steps", "40", "--gap-seconds", "6"]),
    ("throttle", ["--strategy", "throttle", "--target-util", "65"]),
]


def main():
    for name, extra in RUNS:
        out = os.path.join(REPO, "red_team", f"live_v2_{name}.json")
        cmd = [PY, PROBE, *COMMON, *extra, "--label", f"live_v2_{name}", "--out", out]
        print(f"\n===== {name} =====\n{' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd).returncode
        print(f"----- {name} done rc={rc} -> {out}", flush=True)


if __name__ == "__main__":
    main()
