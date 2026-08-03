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
    fit_active_energy_nvl_model, score, score_emc, score_nvl, err_stats,
)


def _loo(frontier, fit_fn, score_fn, key):
    """Leave-one-out held-out (max%, mean%) for a fit/score pair."""
    errs = []
    for i in range(len(frontier)):
        tr = frontier[:i] + frontier[i + 1:]
        score_fn(frontier[i:i + 1], *fit_fn(tr))
        errs.append(frontier[i][key])
    errs = [e for e in errs if e is not None]
    return (max(errs), sum(errs) / len(errs)) if errs else (None, None)


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
    e4, c4tb, c4nv, p4 = fit_active_energy_nvl_model(frontier)
    score(frontier, e2, p2)
    score_emc(frontier, e3, c3, p3)
    score_nvl(frontier, e4, c4tb, c4nv, p4)
    mx2, mn2 = err_stats(frontier, "err_pct")
    mx3, mn3 = err_stats(frontier, "err_pct_emc")
    mx4, mn4 = err_stats(frontier, "err_pct_nvl")

    lmx2, lmn2 = _loo(frontier, fit_active_energy_model, score, "err_pct")
    lmx3, lmn3 = _loo(frontier, fit_active_energy_emc_model, score_emc, "err_pct_emc")
    lmx4, lmn4 = _loo(frontier, fit_active_energy_nvl_model, score_nvl, "err_pct_nvl")

    print(f"\n2-param       : E_MARG={e2:.3f}  P_OH={p2:.3f}"
          f"  | all-frontier max {mx2:.1f}% mean {mn2:.1f}%  | LOO max {lmx2:.1f}% mean {lmn2:.1f}%")
    print(f"3-param (DRAM): E_MARG={e3:.3f}  E_PER_TB={c3:.3f}  P_OH={p3:.3f}"
          f"  | all max {mx3:.1f}% mean {mn3:.1f}%  | LOO max {lmx3:.1f}% mean {lmn3:.1f}%")
    print(f"4-param (+NVL): E_MARG={e4:.3f}  E_PER_TB={c4tb:.3f}  E_PER_NVL={c4nv:.3f}"
          f"  P_OH={p4:.3f}  | all max {mx4:.1f}% mean {mn4:.1f}%  | LOO max {lmx4:.1f}% mean {lmn4:.1f}%")

    print("\n" + "=" * 70)
    print("  RECOMMENDED CONSTANTS (fp16-AMP DDP, 2x V100) — paste into detect_flops.py")
    print("=" * 70)
    print(f"FALLBACK_IDLE_POWER_MW     = {base:.1f}   # both GPUs summed")
    print(f"POWER_OVERHEAD_W           = {p2:.3f}")
    print(f"E_MARGINAL_J_PER_TFLOP     = {e2:.2f}")
    print(f"POWER_OVERHEAD_EMC_W       = {p3:.3f}")
    print(f"E_MARGINAL_EMC_J_PER_TFLOP = {e3:.2f}")
    print(f"E_PER_TB_J                 = {c3:.3f}")
    print("# 4-param NVLink estimator (adversarial tripwire) matched set:")
    print(f"POWER_OVERHEAD_NVL_W         = {p4:.3f}")
    print(f"E_MARGINAL_NVL_J_PER_TFLOP   = {e4:.2f}")
    print(f"E_PER_TB_NVL_J               = {c4tb:.3f}")
    print(f"E_PER_NVLINK_TB_J            = {c4nv:.3f}")
    print("=" * 70)

    # NVLink consistency tripwire: the honest NVLink_bytes/J band + false-positive
    # rate on the frontier (should be 0 — the band is derived from these runs).
    import detect_flops as d
    ratios = sorted(r["nvlink_total_bytes"] / r["net_energy_j"] for r in frontier)
    verdicts = [d.nvlink_consistency(r["net_energy_j"], r["nvlink_total_bytes"])[0]
                for r in frontier]
    n_ok = sum(v == "OK" for v in verdicts)
    print(f"\nNVLink tripwire: honest NVLink/J in [{ratios[0]:.2e}, {ratios[-1]:.2e}]; "
          f"band [{d.NVLINK_BYTES_PER_J_LO:.1e}, {d.NVLINK_BYTES_PER_J_HI:.1e}] "
          f"-> {n_ok}/{len(frontier)} honest runs pass (false-positive rate "
          f"{(len(frontier)-n_ok)/len(frontier)*100:.0f}%). See nvlink_tripwire_demo.py.")


if __name__ == "__main__":
    main()
