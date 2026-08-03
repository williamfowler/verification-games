#!/usr/bin/env python3
"""
run_trials.py — collect N independent trials of the full workload sweep.

The SRF outline calls for 10 trials of all workloads so that each of the three
estimator inputs (net energy, DRAM bytes, NVLink bytes) has a distribution
across repeats, and the calibrate/evaluate analysis can draw a random trace per
workload from the 10. Each trial is one full run of eval_power_monitor's sweep
(a fresh idle baseline + all CONFIGS), dumped to its own records JSON:

    eval_results_v100_ddp_records.json          <- trial 1 (already collected)
    eval_results_v100_ddp_trial2_records.json   <- trial 2
    ...
    eval_results_v100_ddp_trial10_records.json  <- trial 10

Each trial re-measures its own idle baseline and re-sizes every config's `steps`
from a fresh step-rate probe — repeats are genuine re-runs, not replays, so
run-to-run variation in power/DRAM/NVLink is captured, which is the whole point
of the 10 traces.

Resumable at two levels so an interruption never loses collected work:
  * whole trials whose JSON already holds every config are skipped;
  * a partially-written trial JSON is resumed config-by-config (the sweep's own
    --resume path reuses the file's baseline and runs only missing configs).
Safe to relaunch after a teardown.

    python3 run_trials.py [--trials 10] [--start 2] [--pool all]

Both V100s are consumed by every DDP run, so trials run SERIALLY (no two at
once). At ~100 s active/config over ~90 frontier configs, budget ~3.5-4 h/trial.
"""
import argparse
import fcntl
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "power_calibration"))

from eval_power_monitor import (
    CONFIGS, config_label, run_sweep_session,
    load_records_json, BASELINE_SECONDS,
)

# Trial 1 lives in the base name (no suffix); trials 2..N get a _trialK suffix.
BASE_STEM = "eval_results_v100_ddp"

# Singleton lock: at most one runner may ever be active. The guard/watchdog/cron
# could — under a pidfile race — try to start a second runner, and two runners
# would fight over the same two V100s and interleave writes to a trial JSON. This
# makes that impossible: a second runner fails the non-blocking flock and exits
# immediately, before touching GPUs or files. The fd is kept open for the whole
# process lifetime (module global) so the lock is held until this runner exits;
# Python opens fds close-on-exec by default, so torchrun subprocesses do NOT
# inherit it (avoiding the fd-leak trap that would otherwise pin the lock).
SINGLETON_LOCK = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              ".run_trials.singleton")
_singleton_fd = None


def acquire_singleton():
    global _singleton_fd
    _singleton_fd = open(SINGLETON_LOCK, "w")
    try:
        fcntl.flock(_singleton_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("run_trials: another runner already holds the singleton lock — "
              "exiting (this is the double-launch guard working).", flush=True)
        sys.exit(0)


def trial_records_path(k):
    return (f"{BASE_STEM}_records.json" if k == 1
            else f"{BASE_STEM}_trial{k}_records.json")


def expected_labels(pool):
    cfgs = CONFIGS
    if pool in ("fp16", "fp32"):
        cfgs = [c for c in CONFIGS if c.get("precision", "fp16") == pool]
    return {config_label(c) for c in cfgs}


def trial_complete(path, want_labels):
    """A trial JSON is complete when it holds a record (any returncode — OOM/
    sizing failures are legitimately-recorded outcomes) for every expected
    config label."""
    if not os.path.exists(path):
        return False
    try:
        recs, *_ = load_records_json(path)
    except Exception:
        return False
    have = {r["label"] for r in recs}
    return want_labels.issubset(have)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=10,
                    help="total trial count to reach (default 10)")
    ap.add_argument("--start", type=int, default=2,
                    help="first trial index to (re)run; 1 already exists (default 2)")
    ap.add_argument("--pool", default="all", choices=["all", "fp16", "fp32"],
                    help="which CONFIGS each trial sweeps (default all = 91)")
    ap.add_argument("--baseline-seconds", type=int, default=BASELINE_SECONDS)
    args = ap.parse_args()

    acquire_singleton()   # at most one runner alive; a duplicate exits here

    want = expected_labels(args.pool)
    print(f"run_trials: pool={args.pool} ({len(want)} configs/trial), "
          f"trials {args.start}..{args.trials}", flush=True)

    for k in range(args.start, args.trials + 1):
        path = trial_records_path(k)
        if trial_complete(path, want):
            print(f"\n=== trial {k}: COMPLETE ({path}) — skipping ===", flush=True)
            continue
        print(f"\n{'='*70}\n=== TRIAL {k}/{args.trials}  ->  {path}\n"
              f"    (started {time.strftime('%Y-%m-%d %H:%M:%S')})\n{'='*70}",
              flush=True)
        sweep_args = SimpleNamespace(
            pool=args.pool,
            resume=True,               # reuse a partial file's baseline + done configs
            records_json=path,
            output=f"eval_results_v100_ddp_trial{k}.txt",
            baseline_seconds=args.baseline_seconds,
        )
        run_sweep_session(sweep_args)
        print(f"=== TRIAL {k}: done ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===",
              flush=True)

    print(f"\nrun_trials: all trials {args.start}..{args.trials} accounted for.",
          flush=True)


if __name__ == "__main__":
    main()
