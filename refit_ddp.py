#!/usr/bin/env python3
"""
refit_ddp.py — fit the power→FLOP estimator on the fp16-AMP DDP frontier subset.

The DDP regime is a SEPARATE calibration from the porting-era fp32 single-GPU fit
(kept as a labeled legacy reference in detect_flops.py). This refits on records
tagged mode="fp16_ddp"/precision="fp16" that clear the both-GPU >=80% gate, and
prints a RECOMMENDED CONSTANTS block for the DDP deployment plus a leave-one-out
held-out error summary. Reuses eval_power_monitor's fit/score functions.

    python3 refit_ddp.py eval_results_v100_ddp_records.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "power_calibration"))

from eval_power_monitor import (
    load_records, valid, is_frontier, FRONTIER_MIN_GPU_UTIL,
    fit_active_energy_model, fit_active_energy_emc_model,
    score, score_emc, err_stats,
)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "eval_results_v100_ddp_records.json"
    recs, base, bsec, fp = load_records(path)

    fp16 = [r for r in valid(recs) if r["config"].get("precision", "fp16") == "fp16"]
    frontier = [r for r in fp16 if is_frontier(r)]
    n_all = len([r for r in recs if r["config"].get("precision", "fp16") == "fp16"])
    print(f"records: {len(recs)} | fp16 valid: {len(fp16)} | "
          f"fp16 frontier (both GPUs >= {FRONTIER_MIN_GPU_UTIL:.0f}%): {len(frontier)}")
    if len(frontier) < 3:
        print("need >=3 frontier runs to fit"); return

    # Ship fits (all fp16 frontier).
    e2, p2 = fit_active_energy_model(frontier)
    e3, c3, p3 = fit_active_energy_emc_model(frontier)
    score(frontier, e2, p2); score_emc(frontier, e3, c3, p3)
    mx2, mn2 = err_stats(frontier, "err_pct")
    mx3, mn3 = err_stats(frontier, "err_pct_emc")

    # Leave-one-out held-out error (2-param).
    loo = []
    for i in range(len(frontier)):
        tr = frontier[:i] + frontier[i+1:]
        ei, pi = fit_active_energy_model(tr)
        score(frontier[i:i+1], ei, pi)
        loo.append(frontier[i]["err_pct"])
    loo_max = max(loo); loo_mean = sum(loo)/len(loo)

    print(f"\n2-param  ship: E_MARGINAL={e2:.3f} J/TFLOP  P_OVERHEAD={p2:.3f} W"
          f"  | all-frontier max {mx2:.2f}% mean {mn2:.2f}%  | LOO held-out max {loo_max:.2f}% mean {loo_mean:.2f}%")
    print(f"3-param  ship: E_MARGINAL={e3:.3f}  E_PER_TB={c3:.3f}  P_OVERHEAD={p3:.3f} W"
          f"  | all-frontier max {mx3:.2f}% mean {mn3:.2f}%")

    print("\n" + "=" * 66)
    print("  RECOMMENDED CONSTANTS (fp16-AMP DDP, 2x V100) — paste into detect_flops.py")
    print("=" * 66)
    print(f"FALLBACK_IDLE_POWER_MW     = {base:.1f}   # both GPUs summed")
    print(f"POWER_OVERHEAD_W           = {p2:.3f}")
    print(f"E_MARGINAL_J_PER_TFLOP     = {e2:.2f}")
    print(f"POWER_OVERHEAD_EMC_W       = {p3:.3f}")
    print(f"E_MARGINAL_EMC_J_PER_TFLOP = {e3:.2f}")
    print(f"E_PER_TB_J                 = {c3:.3f}")
    print("=" * 66)


if __name__ == "__main__":
    main()
