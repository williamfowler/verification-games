"""
collect_red_trials.py — 10 trials of the v3 offline red strategies (S4 + S3).

Ten independent runs of RED_CONFIGS so each config gets a trial-to-trial
distribution. This is what lets us test whether S4 batch-inflation's under-report is
*systematic* (median across trials clears the config's own noise) rather than a
single-trial fluke, and to tighten the "evasion" threshold from the wide cross-config
band toward each config's own spread. Trial 1 is the existing
red_v3_trial1_records.json; this collects trials 2–10.

Resumable: a complete trial JSON is skipped; a partial one resumes config-by-config
(the sweep's own --resume path). A singleton flock prevents a second copy from
fighting for the two GPUs. Detach with setsid so it survives a teardown.

    python3 red_team/collect_red_trials.py [--trials 10] [--start 2]
"""
import argparse
import fcntl
import os
import sys
import time
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "power_calibration"))
sys.path.insert(0, os.path.join(REPO_ROOT, "red_team"))

from eval_power_monitor import run_sweep_session, load_records_json, config_label
from redteam_configs import RED_CONFIGS

CONFIGS_MODULE = "red_team.redteam_configs:RED_CONFIGS"
WANT = {config_label(c) for c in RED_CONFIGS}
SINGLETON = os.path.join(REPO_ROOT, "red_team", ".red_trials.singleton")
_lock_fd = None


def acquire_singleton():
    global _lock_fd
    _lock_fd = open(SINGLETON, "w")
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("collect_red_trials: another runner holds the lock — exiting.", flush=True)
        sys.exit(0)


def trial_path(k):
    return os.path.join(REPO_ROOT, "red_team", f"red_v3_trial{k}_records.json")


def trial_complete(k):
    p = trial_path(k)
    if not os.path.exists(p):
        return False
    try:
        recs, *_ = load_records_json(p)
    except Exception:
        return False
    return WANT.issubset({r["label"] for r in recs})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--start", type=int, default=2)
    ap.add_argument("--baseline-seconds", type=int, default=90)
    args = ap.parse_args()
    acquire_singleton()

    print(f"collect_red_trials: {len(WANT)} configs/trial, trials {args.start}..{args.trials}",
          flush=True)
    for k in range(args.start, args.trials + 1):
        if trial_complete(k):
            print(f"\n=== trial {k}: COMPLETE — skipping ===", flush=True)
            continue
        print(f"\n{'='*66}\n=== RED TRIAL {k}/{args.trials} -> {trial_path(k)}\n"
              f"    ({time.strftime('%Y-%m-%d %H:%M:%S')})\n{'='*66}", flush=True)
        sweep_args = SimpleNamespace(
            pool="all", resume=True, configs_module=CONFIGS_MODULE,
            records_json=trial_path(k),
            output=os.path.join(REPO_ROOT, "red_team", f"red_v3_trial{k}.txt"),
            baseline_seconds=args.baseline_seconds)
        run_sweep_session(sweep_args)
        print(f"=== RED TRIAL {k}: done ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===",
              flush=True)
    print("\ncollect_red_trials: all trials accounted for.", flush=True)


if __name__ == "__main__":
    main()
