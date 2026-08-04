"""
efficiency.py — GPU-hour accounting + the 2× red-team budget gate (Phase II).

The outline caps an attack at **2× the GPU-hours** a benign run of the same
ground-truth FLOP count would spend. Several attacks (throttling, workload
splitting) work only by wasting wall-clock, so this budget is what turns them
from free wins into scored failures. Every adversarial run and every benign
reference is measured on the same rate metric:

    rate R = ground-truth TFLOPs / GPU-hours          (higher = more efficient)
    GPU-hours = wall-clock hours × world_size          (both V100s under DDP)

    R_benign          = median R over the Phase I ≥80%-util frontier pool
    efficiency ratio  = R_benign / R_adv               (GPU-hours per real FLOP,
                                                        relative to benign)
    budget-legal      iff  ratio ≤ 2.0

An attack that halves the blue team's estimate but costs 5× GPU-hours has ratio
5.0 → ILLEGAL, and must be reported as a failed attack. That pairing (error gain
vs efficiency ratio) is the whole point of the constraint.

    python3 red_team/efficiency.py [phase1_records.json ...]   # print R_benign
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "power_calibration"))

from statistics import median

from eval_power_monitor import load_records, valid, is_frontier

WORLD_SIZE = 2          # dual-V100 DDP: 2 GPU-hours per wall-hour
BUDGET = 2.0            # attack is legal iff efficiency ratio ≤ this


def gpu_hours(duration_s, world_size=WORLD_SIZE):
    return (duration_s / 3600.0) * world_size


def rate_tflops_per_gpu_hour(gt_tf, duration_s, world_size=WORLD_SIZE):
    """Ground-truth TFLOPs delivered per GPU-hour. None if not measurable."""
    gh = gpu_hours(duration_s, world_size)
    if not gt_tf or gh <= 0:
        return None
    return gt_tf / gh


def benign_reference_rate(records, world_size=WORLD_SIZE):
    """R_benign = median TFLOPs/GPU-hour over the fp16 ≥80% frontier runs — the
    efficiency a compliant party achieves per real FLOP. `records` is one or more
    loaded Phase I record lists (or pass paths to benign_reference_rate_from)."""
    rates = []
    for r in records:
        if (r.get("config", {}).get("precision", "fp16") == "fp16"
                and is_frontier(r)):
            rate = rate_tflops_per_gpu_hour(r["ground_truth_tf"], r["duration_s"],
                                            world_size)
            if rate is not None:
                rates.append(rate)
    if not rates:
        raise RuntimeError("no fp16 frontier runs to establish R_benign")
    return median(rates), len(rates)


def benign_reference_rate_from(paths, world_size=WORLD_SIZE):
    """R_benign pooled across one or more Phase I records JSON files."""
    recs = []
    for p in paths:
        rlist, *_ = load_records(p)
        recs.extend(valid(rlist))
    return benign_reference_rate(recs, world_size)


def efficiency_ratio(gt_tf, duration_s, r_benign, world_size=WORLD_SIZE):
    """R_benign / R_adv for one adversarial run. >1 means slower-than-benign per
    real FLOP; ≤2 is budget-legal. None if the run's rate isn't measurable."""
    r_adv = rate_tflops_per_gpu_hour(gt_tf, duration_s, world_size)
    if r_adv is None or r_adv <= 0:
        return None
    return r_benign / r_adv


def is_budget_legal(ratio, budget=BUDGET):
    return ratio is not None and ratio <= budget


def _default_phase1_paths():
    """The 10-trial Phase I records (trial 1 base name + trial2..10)."""
    base = os.path.join(REPO_ROOT, "eval_results_v100_ddp_records.json")
    paths = [base] if os.path.exists(base) else []
    for k in range(2, 11):
        p = os.path.join(REPO_ROOT, f"eval_results_v100_ddp_trial{k}_records.json")
        if os.path.exists(p):
            paths.append(p)
    return paths


def main():
    paths = sys.argv[1:] or _default_phase1_paths()
    if not paths:
        sys.exit("no Phase I records found; pass paths explicitly")
    r_benign, n = benign_reference_rate_from(paths)
    print(f"Phase I files            : {len(paths)}")
    print(f"fp16 frontier runs (n)   : {n}")
    print(f"R_benign (median)        : {r_benign:.1f} TFLOPs / GPU-hour")
    print(f"Budget                   : attack legal iff R_benign/R_adv ≤ {BUDGET}")
    print(f"  → an attack is legal only if it delivers ≥ {r_benign / BUDGET:.1f} "
          f"TFLOPs/GPU-hour (spends ≤ {BUDGET}× the GPU-hours per real FLOP).")


if __name__ == "__main__":
    main()
