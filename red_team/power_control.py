"""
power_control.py — S5 power-cap / S6 clock-lock harness (Phase II v3).

DVFS under-report lever: a stock GPU runs a performance-tuned V/f point, not the
energy optimum. Dynamic energy ~ V², so a lower-voltage operating point can do the
SAME FLOPs at LOWER total energy. If capped J/FLOP falls below the calibrated
E_MARGINAL, the estimator reads BELOW ground truth (evasion). The tension the sweep
quantifies: capping lowers clocks → longer wall-clock → more GPU-hours, so a run is
a real threat only if it under-reports AND stays within the 2× budget.

This wraps a benign workload (code unchanged — one external variable) at each power
cap / clock, records the setting into the run, and **restores defaults on exit and
on crash** (try/finally). GT is preserved (same shape/steps).

PRIVILEGE: `nvidia-smi -pl` / `-lgc` need root on this box (verified: "Insufficient
Permissions" as the tenant). Run the sweep with privilege:
    sudo /home/will/verification-games/.venv/bin/python red_team/power_control.py --sweep-pl 300 250 200 150 100
Modeled as a granted capability (like the blue team's root DCGM access) or, more
realistically, the red team requesting a power-capped allocation. Without privilege
it still runs (at default power) and warns, so the harness itself is testable as the
tenant — the cap just won't apply.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from statistics import median

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "power_calibration"))

from calibrate_power import sample_idle, run_workload, DDP_GPUS
import detect_flops


def _nvidia_smi(args):
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True)


def get_default_limits(gpus):
    out = {}
    for g in gpus:
        r = _nvidia_smi(["--query-gpu=power.default_limit", "--format=csv,noheader,nounits",
                         "-i", str(g)])
        out[g] = float(r.stdout.strip().splitlines()[0])
    return out


def set_power_limit(watts, gpus):
    """Set -pl on each GPU. Returns True if all succeeded (False + warn otherwise)."""
    ok = True
    for g in gpus:
        r = _nvidia_smi(["-i", str(g), "-pl", str(int(watts))])
        if r.returncode != 0:
            ok = False
            print(f"  [warn] could not set -pl {watts}W on GPU {g}: "
                  f"{r.stdout.strip() or r.stderr.strip()}", flush=True)
    return ok


def set_clock(mhz, gpus):
    ok = True
    for g in gpus:
        r = _nvidia_smi(["-i", str(g), "-lgc", str(int(mhz))])
        if r.returncode != 0:
            ok = False
            print(f"  [warn] could not lock clock {mhz}MHz on GPU {g}: "
                  f"{r.stdout.strip() or r.stderr.strip()}", flush=True)
    return ok


def restore(gpus, defaults):
    for g in gpus:
        _nvidia_smi(["-i", str(g), "-pl", str(int(defaults[g]))])
        _nvidia_smi(["-i", str(g), "-rgc"])


def parent_config(steps):
    """Benign S5 parent (a frontier shape); code unchanged, cap is the one variable."""
    return {"family": "S5", "d_model": 1024, "seq_len": 512, "batch_size": 16,
            "num_layers": 12, "nhead": 8, "dim_feedforward": 4096,
            "precision": "fp16", "optimizer": "adamw",
            "fixed_steps": True, "steps": steps}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-pl", nargs="*", type=int, default=None,
                    help="power caps (W) to sweep, e.g. 300 250 200 150 100")
    ap.add_argument("--sweep-lgc", nargs="*", type=int, default=None,
                    help="SM clocks (MHz) to sweep (S6), e.g. 1380 1000 700 500")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--baseline-seconds", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "red_team", "red_s5_records.json"))
    args = ap.parse_args()

    gpus = DDP_GPUS
    defaults = get_default_limits(gpus)
    print(f"Default power limits: {defaults} W")
    baseline_idle = sample_idle(args.baseline_seconds, gpus, "baseline")
    base_mw = median([mw for _, mw in baseline_idle])
    print(f"Idle baseline: {base_mw:.0f} mW")

    records = []
    try:
        settings = [("pl", w) for w in (args.sweep_pl or [])] + \
                   [("lgc", c) for c in (args.sweep_lgc or [])]
        if not settings:
            settings = [("pl", int(median(defaults.values())))]  # default-only smoke
        for kind, val in settings:
            applied = (set_power_limit(val, gpus) if kind == "pl"
                       else set_clock(val, gpus))
            time.sleep(2.0)
            cfg = parent_config(args.steps)
            cfg["strategy"] = "S5_powercap" if kind == "pl" else "S6_clocklock"
            cfg["label"] = f"S5_pl{val}W" if kind == "pl" else f"S6_lgc{val}"
            cfg["parent"] = "S5_pl_default"
            cfg["power_cap_w"] = val if kind == "pl" else None
            cfg["clock_mhz"] = val if kind == "lgc" else None
            cfg["cap_applied"] = applied
            print(f"\n=== {cfg['label']}  (applied={applied}) ===", flush=True)
            r = run_workload(cfg, base_mw, gpus)
            r["label"] = cfg["label"]
            if r["returncode"] == 0:
                print(f"  GT {r['ground_truth_tf']:.0f} TF | net {r['net_energy_j']:.0f} J"
                      f" | {r['duration_s']:.0f}s | J/TF {r['net_energy_j']/r['ground_truth_tf']:.3f}"
                      f" | avg_pwr {r.get('avg_raw_mw',0)/1000:.0f} W | util {r.get('avg_gpu_pct')}%")
            records.append(r)
            with open(args.out, "w") as f:
                json.dump({"records": records, "baseline_mw": base_mw,
                           "baseline_seconds": args.baseline_seconds}, f, indent=1)
    finally:
        restore(gpus, defaults)
        print(f"\nRestored default power limits + clocks. Wrote {args.out}")


if __name__ == "__main__":
    main()
