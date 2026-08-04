"""
test_benign_equivalence.py — CI guard (protocol/honesty invariant, §5.3).

Asserts that `adversarial_workload.py --strategy none` computes the SAME ground
truth as the benign `sample_ml_workload.py` for identical shape args. This catches
protocol regressions and accidental GT drift in the red-team wrapper in one check.
GT is a deterministic function of op shapes × steps × world_size, so it must match
exactly. Runs a tiny DDP pass on both GPUs; no DCGM/hostengine needed.

    python3 red_team/test_benign_equivalence.py     # exit 0 = pass, 1 = fail
"""
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TORCHRUN = os.path.join(REPO_ROOT, ".venv", "bin", "torchrun")
SHAPE = ["--steps", "20", "--d-model", "512", "--seq-len", "128",
         "--batch-size", "8", "--num-layers", "4"]


def gt_of(script, extra=()):
    cmd = [TORCHRUN, "--standalone", "--nproc_per_node=2",
           os.path.join(REPO_ROOT, script), *SHAPE, *extra]
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    out = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
    matches = re.findall(r"Ground truth total\s*:\s*([\d.]+)\s*TFLOPs", out)
    return float(matches[-1]) if matches else None


def main():
    benign = gt_of("sample_ml_workload.py")
    adv = gt_of("red_team/adversarial_workload.py", ["--strategy", "none"])
    print(f"benign GT              : {benign}")
    print(f"adversarial(none) GT   : {adv}")
    if benign is None or adv is None:
        print("FAIL: missing ground truth from one entry point")
        return 1
    if abs(benign - adv) > 1e-6:
        print(f"FAIL: GT drift {abs(benign - adv):.6g} TFLOPs — red wrapper changed the compute path")
        return 1
    print("PASS: adversarial_workload --strategy none is GT-equivalent to benign")
    return 0


if __name__ == "__main__":
    sys.exit(main())
