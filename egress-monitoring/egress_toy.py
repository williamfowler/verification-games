#!/usr/bin/env python3
"""
egress_toy.py - Phase-gated egress monitoring toy experiment.

A self-contained simulation (no sockets, no real clock) testing whether a
*byte-budget* defense beats a *classifier* defense for catching weight
exfiltration from a monitored compute perimeter.

The perimeter has a developer-declared MODE:

  AIRGAP - no egress is legitimate. Any off-perimeter tx > 0 is a violation.
           (single-perimeter pre-training / inference of a frozen model.)
  OPEN   - egress allowed but capped. Used when tool-use is needed
           (post-training, RL on web search). Binds to a committed model hash
           and accepts a per-window outbound BYTE BUDGET.

Detection in OPEN does NOT try to classify "is this a real search query."
It bounds the channel below W (model size): flag on cumulative outbound
budget, plus a directional asymmetry check (tx/rx).

Run:  python egress_toy.py
This parallels the cumulative-FLOP-threshold result from detect_flops.py:
a budget defense, not a gameable classifier.

----------------------------------------------------------------------------
Known limitations (acknowledged, not solved here):
  * Steganographic content inside the tiny outbound budget is a residual
    bits/sec covert channel. Out of scope -- the budget bounds throughput,
    not information density.
  * Perimeter completeness is assumed: one known set of off-perimeter
    destinations. A real monitor counts ALL egress, not just the known
    dark-compute IP.
  * Budget is per declared tool. A large-upload tool (code exec, file
    upload) would break this and must simply be disallowed in monitored mode.
----------------------------------------------------------------------------
"""

import argparse
import math
import random
import sqlite3

KB = 1024
MB = 1024 * 1024

# --- The asset and the perimeter -------------------------------------------
W = 50 * MB                       # "model" size: bytes the red team wants out

EXFIL_DST = "exfil-host"          # red team's off-perimeter destination
SEARCH_DST = "search-api"         # legitimate web-search endpoint (also external)
OFF_PERIMETER = {EXFIL_DST, SEARCH_DST}

# --- Monitor / simulation constants ----------------------------------------
CHUNK = 8 * KB                    # granularity at which the monitor sees a stream
ASYMMETRY_THRESHOLD = 4.0         # per-window tx/rx ratio above this -> flag
NUM_WINDOWS = 500                 # total simulated windows (the horizon)
OPEN_DUTY = 5                     # phase-gated: 1 of every OPEN_DUTY windows is OPEN
DEFAULT_BUDGET = 64 * KB          # per-window outbound budget in OPEN mode

# --- Legitimate web-search profile (inbound-heavy) -------------------------
LEGIT_CALLS = (5, 15)             # search calls per OPEN window
LEGIT_TX = (200, 800)            # query bytes (small, outbound)
LEGIT_RX = (20 * KB, 80 * KB)     # fetched-page bytes (large, inbound)

AIRGAP, OPEN = "AIRGAP", "OPEN"


def fmt_bytes(n):
    """Human-readable byte count."""
    if n >= MB:
        return f"{n / MB:.2f}MB"
    if n >= KB:
        return f"{n / KB:.1f}KB"
    return f"{n}B"


class Monitor:
    """Blue team. Tracks per-destination tx/rx and flags violations.

    AIRGAP: flag any off-perimeter tx > 0.
    OPEN:   flag if cumulative (per-window) off-perimeter tx exceeds budget,
            OR if a destination's per-window tx/rx ratio exceeds the
            asymmetry threshold (checked at window close).
    """

    def __init__(self, budget, asym=ASYMMETRY_THRESHOLD):
        self.budget = budget
        self.asym = asym
        self.life_tx = {}
        self.life_rx = {}
        self.flags = []            # (window, check, dst)
        self.reset_window()

    def reset_window(self):
        self.win_tx = {}
        self.win_rx = {}
        self.win_tx_total = 0      # cumulative off-perimeter tx this window

    def _flag(self, window, check, dst):
        self.flags.append((window, check, dst))

    def tx(self, window, mode, dst, n):
        self.life_tx[dst] = self.life_tx.get(dst, 0) + n
        self.win_tx[dst] = self.win_tx.get(dst, 0) + n
        if dst in OFF_PERIMETER:
            self.win_tx_total += n
            if mode == AIRGAP and n > 0:
                self._flag(window, "airgap-tx", dst)
            elif mode == OPEN and self.win_tx_total > self.budget:
                self._flag(window, "budget", dst)

    def rx(self, window, mode, dst, n):
        self.life_rx[dst] = self.life_rx.get(dst, 0) + n
        self.win_rx[dst] = self.win_rx.get(dst, 0) + n

    def close_window(self, window, mode):
        if mode == OPEN:
            for dst, t in self.win_tx.items():
                if dst in OFF_PERIMETER:
                    r = self.win_rx.get(dst, 0)
                    if t / max(r, 1) > self.asym:
                        self._flag(window, "asymmetry", dst)
        self.reset_window()

    def flagged(self):
        return bool(self.flags)

    def first_flag(self):
        return self.flags[0] if self.flags else None


def window_mode(perimeter, w):
    """Mode of window `w` under a given perimeter policy."""
    if perimeter == "baseline-open":
        return OPEN                       # gap never closes
    # phase-gated: AIRGAP by default, OPEN only in declared windows
    return OPEN if (w % OPEN_DUTY == 0) else AIRGAP


def open_window_count(perimeter):
    return sum(1 for w in range(NUM_WINDOWS) if window_mode(perimeter, w) == OPEN)


# --- Red team ---------------------------------------------------------------

def run_exfil(perimeter, strategy, budget, seed=0):
    """Simulate one exfiltration attempt. Red stops once detected."""
    rng = random.Random(seed)
    mon = Monitor(budget)
    exfil = 0
    pad_cost = 0

    for w in range(NUM_WINDOWS):
        mode = window_mode(perimeter, w)
        mon.reset_window()

        if strategy == "blast":
            # Dump weights as fast as possible, whatever the mode.
            # AIRGAP trips on the first byte; OPEN trips at the budget boundary.
            remaining = W - exfil
            while remaining > 0:
                n = min(CHUNK, remaining)
                mon.tx(w, mode, EXFIL_DST, n)
                exfil += n
                remaining -= n
                if mon.flagged():
                    break
            mon.close_window(w, mode)

        elif strategy == "slow_leak":
            # Stay under the per-window budget, but no reverse traffic ->
            # the asymmetry check trips at window close. Only acts in OPEN
            # (an AIRGAP tx would be flagged immediately).
            if mode == OPEN:
                leak = min(budget, W - exfil)
                sent = 0
                while sent < leak:
                    n = min(CHUNK, leak - sent)
                    mon.tx(w, mode, EXFIL_DST, n)
                    exfil += n
                    sent += n
                    if mon.flagged():
                        break
                mon.close_window(w, mode)

        elif strategy == "padded":
            # Stay under budget AND pad the reverse channel to fake symmetry,
            # defeating the asymmetry check. Survives -- but is throttled to
            # the budget, so moving W takes (W / budget) OPEN windows.
            if mode == OPEN:
                leak = min(budget, W - exfil)
                pad = math.ceil(leak / ASYMMETRY_THRESHOLD)  # cost of faking rx
                mon.rx(w, mode, EXFIL_DST, pad)
                pad_cost += pad
                sent = 0
                while sent < leak:
                    n = min(CHUNK, leak - sent)
                    mon.tx(w, mode, EXFIL_DST, n)
                    exfil += n
                    sent += n
                    if mon.flagged():
                        break
                mon.close_window(w, mode)
        else:
            raise ValueError(f"unknown strategy {strategy!r}")

        if mon.flagged() or exfil >= W:
            break

    flag = mon.first_flag()
    return {
        "perimeter": perimeter,
        "strategy": strategy,
        "budget": budget,
        "exfil": exfil,
        "frac": exfil / W,
        "detected": flag is not None,
        "detect_window": flag[0] if flag else None,
        "detect_check": flag[1] if flag else None,
        "pad_cost": pad_cost,
    }


# --- Legitimate workload (false-positive measurement) ----------------------

def run_legit(perimeter, budget, seed=0):
    """Simulate the legitimate web-search workload alone; count false positives.

    Legit tool-use only happens in OPEN windows. It is inbound-heavy, so the
    asymmetry check never fires; the only way it gets flagged is a budget set
    tighter than its own outbound footprint.
    """
    rng = random.Random(seed)
    mon = Monitor(budget)
    open_windows = 0
    fp_windows = 0

    for w in range(NUM_WINDOWS):
        if window_mode(perimeter, w) != OPEN:
            continue
        open_windows += 1
        mon.reset_window()
        before = len(mon.flags)
        for _ in range(rng.randint(*LEGIT_CALLS)):
            mon.tx(w, OPEN, SEARCH_DST, rng.randint(*LEGIT_TX))
            mon.rx(w, OPEN, SEARCH_DST, rng.randint(*LEGIT_RX))
        mon.close_window(w, OPEN)
        if len(mon.flags) > before:
            fp_windows += 1

    return {
        "open_windows": open_windows,
        "fp_windows": fp_windows,
        "fp_rate": fp_windows / open_windows if open_windows else 0.0,
    }


def legit_fp_rate(perimeter, budget, seeds):
    """Average legit FP rate over several seeds for a smoother curve."""
    return sum(run_legit(perimeter, budget, s)["fp_rate"] for s in seeds) / len(seeds)


# --- Logging ----------------------------------------------------------------

def init_db(path):
    con = sqlite3.connect(path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            kind TEXT, perimeter TEXT, strategy TEXT, budget INTEGER,
            exfil INTEGER, frac REAL, detected INTEGER, detect_window INTEGER,
            detect_check TEXT, pad_cost INTEGER, fp_rate REAL
        )
    """)
    con.commit()
    return con


def log_exfil(con, r):
    con.execute(
        "INSERT INTO runs (kind, perimeter, strategy, budget, exfil, frac, "
        "detected, detect_window, detect_check, pad_cost) "
        "VALUES ('exfil',?,?,?,?,?,?,?,?,?)",
        (r["perimeter"], r["strategy"], r["budget"], r["exfil"], r["frac"],
         int(r["detected"]), r["detect_window"], r["detect_check"], r["pad_cost"]),
    )


def log_fp(con, perimeter, budget, fp_rate):
    con.execute(
        "INSERT INTO runs (kind, perimeter, budget, fp_rate) VALUES ('legit',?,?,?)",
        (perimeter, budget, fp_rate),
    )


# --- Experiment -------------------------------------------------------------

STRATEGIES = ["blast", "slow_leak", "padded"]
PERIMETERS = ["baseline-open", "phase-gated"]


def verdict_table(con, budget, seeds):
    print(f"\n=== VERDICT @ default budget {fmt_bytes(budget)}/window, "
          f"horizon {NUM_WINDOWS} windows ===")
    print(f"{'perimeter':<14} {'strategy':<10} {'exfil':>9} {'%W':>7} "
          f"{'detected':>9} {'@win':>6} {'via':>10} {'pad-cost':>9}")
    print("-" * 80)
    for perimeter in PERIMETERS:
        for strat in STRATEGIES:
            r = run_exfil(perimeter, strat, budget)
            log_exfil(con, r)
            det = "YES" if r["detected"] else "no"
            win = r["detect_window"] if r["detect_window"] is not None else "-"
            via = r["detect_check"] or "-"
            print(f"{perimeter:<14} {strat:<10} {fmt_bytes(r['exfil']):>9} "
                  f"{r['frac'] * 100:>6.2f}% {det:>9} {str(win):>6} {via:>10} "
                  f"{fmt_bytes(r['pad_cost']):>9}")
        fp = legit_fp_rate(perimeter, budget, seeds)
        log_fp(con, perimeter, budget, fp)
        print(f"{perimeter:<14} {'(legit FP)':<10} {'':>9} {'':>7} "
              f"{'':>9} {'':>6} {'':>10} -> FP rate {fp * 100:.1f}%")
    print("-" * 80)
    print("blast/slow_leak are caught regardless of budget (budget / asymmetry).")
    print("padded survives detection -- only the budget throttles it.")


def sweep_table(con, budgets, seeds):
    perimeter = "phase-gated"
    n_open = open_window_count(perimeter)
    print(f"\n=== BUDGET SWEEP (treatment: phase-gated, {n_open} OPEN windows) ===")
    print("Tradeoff: tighten the budget -> padded exfil falls, but legit FP rises.")
    print(f"{'budget':>9} {'legit-FP':>9} {'padded %W':>10} {'padded exfil':>13} "
          f"{'blast':>10} {'slow_leak':>12}")
    print("-" * 74)
    knee = None
    for b in budgets:
        fp = legit_fp_rate(perimeter, b, seeds)
        log_fp(con, perimeter, b, fp)
        pad = run_exfil(perimeter, "padded", b)
        log_exfil(con, pad)
        blast = run_exfil(perimeter, "blast", b)
        slow = run_exfil(perimeter, "slow_leak", b)
        b_det = f"caught@{blast['detect_window']}"
        s_det = f"caught@{slow['detect_window']}"
        print(f"{fmt_bytes(b):>9} {fp * 100:>8.1f}% {pad['frac'] * 100:>9.2f}% "
              f"{fmt_bytes(pad['exfil']):>13} {b_det:>10} {s_det:>12}")
        if fp == 0.0:
            knee = (b, pad)        # tightest zero-FP budget seen so far
    print("-" * 74)
    if knee:
        b, pad = knee
        print(f"Knee: tightest budget with 0% legit FP = {fmt_bytes(b)} "
              f"-> padded still exfils {pad['frac'] * 100:.2f}% of W "
              f"({fmt_bytes(pad['exfil'])}) undetected.")
        print("Below this the budget keeps strangling exfil but starts "
              "flagging legitimate tool-use.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help="default per-window OPEN budget in bytes")
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed for the legit FP averaging")
    ap.add_argument("--db", default="egress-monitoring/egress_log.db",
                    help="sqlite log path")
    args = ap.parse_args()

    seeds = list(range(args.seed, args.seed + 8))
    con = init_db(args.db)

    print("PHASE-GATED EGRESS MONITORING -- toy experiment")
    print(f"asset W = {fmt_bytes(W)}   asymmetry threshold = {ASYMMETRY_THRESHOLD}x"
          f"   chunk = {fmt_bytes(CHUNK)}")
    print(f"baseline (always-OPEN) OPEN windows = {open_window_count('baseline-open')}"
          f"   phase-gated OPEN windows = {open_window_count('phase-gated')}"
          f"   (1 in {OPEN_DUTY})")

    verdict_table(con, args.budget, seeds)

    budgets = [1 * MB, 256 * KB, 64 * KB, 16 * KB, 12 * KB,
               8 * KB, 6 * KB, 4 * KB, 2 * KB, 1 * KB]
    sweep_table(con, budgets, seeds)

    con.commit()
    con.close()
    print(f"\nLogged to {args.db}")


if __name__ == "__main__":
    main()
