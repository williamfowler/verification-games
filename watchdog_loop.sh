#!/usr/bin/env bash
# watchdog_loop.sh — session-independent watchdog for the 10-trial sweep.
#
# Runs trial_guard.sh every INTERVAL seconds forever. Launched detached via
# setsid so a terminal/session teardown can't SIGHUP it. This is the primary
# keep-alive; the cron entry (if the daemon honors it) is a redundant backup —
# trial_guard.sh uses flock + is idempotent, so both firing is harmless.
#
# It exits on its own once all trials 2..10 are complete (guard reports done and
# this loop notices the sentinel), so it doesn't linger after the sweep.
set -u
export PATH=/usr/sbin:/usr/bin:/bin:/sbin
REPO=/home/will/verification-games
PY=$REPO/.venv/bin/python
GUARD=$REPO/trial_guard.sh
WLOG=$REPO/watchdog_loop.log
INTERVAL=60

echo "[$(date '+%F %T')] watchdog_loop start (pid $$)" >> "$WLOG"
cd "$REPO" || exit 1

while true; do
  "$GUARD"
  # Stop the loop when every trial is complete.
  if "$PY" - <<'PYEOF'
import sys, run_trials as rt
want = rt.expected_labels('all')
sys.exit(0 if all(rt.trial_complete(rt.trial_records_path(k), want)
                  for k in range(2, 11)) else 1)
PYEOF
  then
    echo "[$(date '+%F %T')] all trials complete — watchdog_loop exiting" >> "$WLOG"
    exit 0
  fi
  sleep "$INTERVAL"
done
