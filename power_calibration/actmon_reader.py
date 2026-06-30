#!/usr/bin/env python3
"""
actmon_reader.py — privileged DRAM-activity sampler for the bytes-energy term.

Reads the Tegra memory-controller activity monitor (actmon) and EMC clock from
debugfs (root-only) and streams one sample per line so an unprivileged process
(the calibration sweep) can integrate DRAM bytes moved over a workload window.

The blue-team daemon (detect_flops.py) runs as root and reads these paths
directly; this standalone reader exists so eval_power_monitor.py / calibrate_power.py,
which MUST run unprivileged (CUDA needs the user venv), can still obtain the signal
via a narrow NOPASSWD sudoers entry, e.g.:

    <user> ALL=(root) NOPASSWD: /home/jetson/verification-games/.venv/bin/python3 \
        /home/jetson/verification-games/power_calibration/actmon_reader.py *

Output (one line per sample, line-buffered, fields space-separated):

    <mono_t> <util_fraction> <emc_rate_hz> <last_prd_activity> <avg_activity>

  mono_t          time.monotonic() seconds (caller aligns to its own window)
  util_fraction   clamp(last_prd_activity / emc_rate_hz, 0, 1)   [hypothesis A]
  emc_rate_hz     current EMC clock (Hz)
  last_prd / avg  raw actmon registers (printed for Phase-0 inspection)

bytes/s is derived caller-side as util_fraction * PEAK_BW_BYTES_S (optionally
* emc_rate/EMC_MAX_HZ to correct for EMC DVFS downclocking). Keeping PEAK_BW out
of this reader means the conversion lives in exactly one place (the estimator).

Exits cleanly on SIGTERM/SIGINT. If the debugfs paths are unreadable (non-root or
absent) it prints a diagnostic to stderr and exits non-zero so the caller can fall
back to the 2-parameter estimator.
"""

import argparse
import signal
import sys
import time

ACTMON_PRD_PATH = "/sys/kernel/debug/bpmp/debug/actmon/mc_all_last_prd_activity"
ACTMON_AVG_PATH = "/sys/kernel/debug/bpmp/debug/actmon/mc_all_avg_activity"
EMC_RATE_PATH   = "/sys/kernel/debug/bpmp/debug/clk/emc/rate"

_running = True


def _stop(signum, frame):
    global _running
    _running = False


def _read_number(path):
    """Return the first number in the file as float, or None on any error."""
    try:
        with open(path) as f:
            return float(f.read().strip().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=0.5,
                    help="sample interval in seconds (default 0.5 = 2 Hz)")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    # Fail fast (non-zero exit) if we cannot read the counters at all, so the
    # caller treats the bytes signal as unavailable rather than silently zero.
    if _read_number(EMC_RATE_PATH) is None or _read_number(ACTMON_PRD_PATH) is None:
        sys.stderr.write(
            "actmon_reader: cannot read debugfs actmon/emc paths "
            "(need root; are you running under sudo?)\n")
        return 1

    while _running:
        emc_rate = _read_number(EMC_RATE_PATH)
        last_prd = _read_number(ACTMON_PRD_PATH)
        avg      = _read_number(ACTMON_AVG_PATH)
        if emc_rate and emc_rate > 0 and last_prd is not None:
            util = last_prd / emc_rate
            util = 0.0 if util < 0.0 else (1.0 if util > 1.0 else util)
        else:
            util = 0.0
        # Line-buffered single-line sample; flush so readline() callers see it promptly.
        try:
            sys.stdout.write(f"{time.monotonic():.6f} {util:.6f} "
                             f"{emc_rate or 0:.0f} {last_prd or 0:.0f} {avg or 0:.0f}\n")
            sys.stdout.flush()
        except BrokenPipeError:
            # Reader (calibration sampler) closed the pipe — exit promptly even if
            # SIGTERM forwarding through sudo did not reach us.
            return 0
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
