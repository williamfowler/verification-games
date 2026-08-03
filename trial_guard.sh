#!/usr/bin/env bash
# trial_guard.sh — keep the 10-trial sweep (run_trials.py) alive.
#
# Run every minute by cron. It is idempotent and safe to run concurrently
# (flock serializes). Three jobs:
#   1. If trials 2..10 are all complete -> do nothing (and stop relaunching).
#   2. If the runner is not alive -> relaunch it (resumes where it left off).
#   3. If the runner is alive but STALLED (run_trials.log hasn't grown in
#      STALL_SECS) -> kill its process group and relaunch (handles a hang/pause,
#      not just a death). A single config takes ~2-3 min and a fresh trial
#      baseline ~90s, so STALL_SECS=1800 only trips on a genuine hang.
#
# The runner itself is fully resumable, so kill+relaunch never loses work.
set -u
export PATH=/usr/sbin:/usr/bin:/bin:/sbin

REPO=/home/will/verification-games
PY=$REPO/.venv/bin/python
LOG=$REPO/run_trials.log
GLOG=$REPO/trial_guard.log
LOCK=$REPO/.trial_guard.lock
PIDFILE=$REPO/.run_trials.pid
STALL_SECS=1800

cd "$REPO" || exit 1

# Serialize guard invocations; if a previous one is still working, bail.
exec 9>"$LOCK"
flock -n 9 || exit 0

ts(){ date '+%Y-%m-%d %H:%M:%S'; }
glog(){ echo "[$(ts)] $*" >> "$GLOG"; }

# Heartbeat: proves cron is invoking the guard even when the runner is healthy
# and the guard is otherwise silent. mtime of this file = last guard run.
HEARTBEAT=$REPO/.trial_guard.heartbeat
touch "$HEARTBEAT"

launch(){
  # 9>&- closes the guard's flock fd in the child so the long-lived runner does
  # NOT inherit (and thus hold) the lock — otherwise every later guard call bails
  # at flock and the watchdog is silently disabled for the whole run.
  setsid bash -c "exec '$PY' -u run_trials.py --trials 10 --start 2 >> '$LOG' 2>&1" < /dev/null 9>&- &
  sleep 2
  local np
  np=$(pgrep -f "run_trials.py --trials 10" | head -1)
  echo "${np:-}" > "$PIDFILE"
  glog "relaunched runner (PID ${np:-unknown})"
}

# 1) Done? Then the guard is finished — remove its own cron line so it stops.
if "$PY" - <<'PYEOF'
import sys
import run_trials as rt
want = rt.expected_labels('all')
done = all(rt.trial_complete(rt.trial_records_path(k), want) for k in range(2, 11))
sys.exit(0 if done else 1)
PYEOF
then
  glog "all trials 2-10 complete — removing guard from crontab"
  crontab -l 2>/dev/null | grep -v 'trial_guard.sh' | crontab - 2>/dev/null || true
  exit 0
fi

# 2/3) Determine liveness.
RPID=""
[ -f "$PIDFILE" ] && RPID=$(cat "$PIDFILE" 2>/dev/null || true)
alive=0
if [ -n "${RPID:-}" ] && kill -0 "$RPID" 2>/dev/null; then
  alive=1
else
  RPID=$(pgrep -f "run_trials.py --trials 10" | head -1)
  [ -n "${RPID:-}" ] && { alive=1; echo "$RPID" > "$PIDFILE"; }
fi

if [ "$alive" -eq 1 ]; then
  # Stall detection: log mtime age.
  if [ -f "$LOG" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
    if [ "$age" -gt "$STALL_SECS" ]; then
      glog "runner PID $RPID STALLED (${age}s since log grew > ${STALL_SECS}s) — killing group"
      kill -TERM "-$RPID" 2>/dev/null || kill -TERM "$RPID" 2>/dev/null || true
      sleep 5
      kill -KILL "-$RPID" 2>/dev/null || true
      pkill -KILL -f "sample_ml_workload.py" 2>/dev/null || true
      pkill -KILL -f "torchrun" 2>/dev/null || true
      launch
    fi
    # else: alive and progressing — nothing to do (stay quiet in the log).
  fi
else
  glog "runner not alive — relaunching"
  launch
fi
