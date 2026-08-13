#!/usr/bin/env bash
# red_watchdog.sh — keep the 10-trial red collection alive across teardowns.
# setsid-detached loop; relaunches collect_red_trials.py if it's not running.
# The runner's singleton flock makes a double-launch harmless. Exits when done.
set -u
export PATH=/usr/sbin:/usr/bin:/bin:/sbin
REPO=/home/will/verification-games
PY=$REPO/.venv/bin/python
LOG=$REPO/red_team/collect_red_trials.log
WLOG=$REPO/red_team/red_watchdog.log
INTERVAL=60
cd "$REPO" || exit 1
echo "[$(date '+%F %T')] red_watchdog start (pid $$)" >> "$WLOG"

complete() {
  "$PY" - <<'PYEOF' 2>/dev/null
import sys, os
sys.path.insert(0, "red_team"); sys.path.insert(0, "."); sys.path.insert(0, "power_calibration")
import collect_red_trials as c
sys.exit(0 if all(c.trial_complete(k) for k in range(2, 11)) else 1)
PYEOF
}

while true; do
  if complete; then
    echo "[$(date '+%F %T')] all red trials complete — watchdog exiting" >> "$WLOG"
    exit 0
  fi
  if ! pgrep -f "python -u red_team/collect_red_trials.py" >/dev/null; then
    echo "[$(date '+%F %T')] (re)launching red runner" >> "$WLOG"
    setsid bash -c "exec '$PY' -u red_team/collect_red_trials.py --trials 10 --start 2 >> '$LOG' 2>&1" < /dev/null 9>&- &
  fi
  sleep "$INTERVAL"
done
