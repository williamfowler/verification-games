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


def check_gt_invariance():
    """v3 one-variable invariant: every RED_CONFIG must match its parent's ground
    truth (<1%). GT is deterministic in shape/steps/batch, so we compute it in-
    process from FlopCounterMode (per-step FLOPs × steps; world_size is constant
    and cancels) — no training, no torchrun. Needs cuda:0 free."""
    import torch
    sys.path.insert(0, REPO_ROOT)
    sys.path.insert(0, os.path.join(REPO_ROOT, "red_team"))
    from sample_ml_workload import TinyTransformer, count_flops_per_step
    from redteam_configs import RED_CONFIGS

    dev = torch.device("cuda:0")
    gt_rel = {}
    for c in RED_CONFIGS:
        m = TinyTransformer(d_model=c["d_model"], nhead=c["nhead"],
                            num_layers=c["num_layers"],
                            dim_feedforward=c["dim_feedforward"]).to(dev)
        fps = count_flops_per_step(m, c["batch_size"], c["seq_len"], c["d_model"],
                                   dev, c["precision"])
        gt_rel[c["label"]] = fps * c["steps"]          # ∝ true GT (world_size cancels)
        del m
        torch.cuda.empty_cache()

    ok = True
    for c in RED_CONFIGS:
        parent = c["parent"]
        if parent not in gt_rel:
            print(f"FAIL: {c['label']} parent {parent} not in config set"); ok = False; continue
        drift = abs(gt_rel[c["label"]] - gt_rel[parent]) / gt_rel[parent]
        tag = "ok" if drift < 0.01 else "FAIL"
        if drift >= 0.01:
            ok = False
        if c["label"] == parent or drift >= 0.01:
            print(f"  {c['label']:34s} vs {parent:28s} GT drift {drift*100:6.3f}%  {tag}")
    print(("PASS" if ok else "FAIL") + ": all RED_CONFIGS GT-match their parent (<1%)")
    return 0 if ok else 1


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
    print("PASS: adversarial_workload --strategy none is GT-equivalent to benign\n")
    print("GT-invariance of the one-variable RED_CONFIGS vs their parents:")
    return check_gt_invariance()


if __name__ == "__main__":
    sys.exit(main())
