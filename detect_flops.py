import time
import re
import glob
import sqlite3
import subprocess
import select
from collections import deque
from statistics import median
from jtop import jtop

# Frequencies from jtop (confirmed on JetPack 6.2); bandwidth for Orin Nano 8GB.
# TFLOPS_TF32: Ampere Tensor Core peak at half FP16 rate — matches default PyTorch
# behaviour (torch.backends.cuda.matmul.allow_tf32 = True).
ORIN_PROFILE = {
    "MAX_FREQ_HZ":      918_000_000,
    "TFLOPS_FP16":       15.3,
    "TFLOPS_TF32":        7.65,
    "TFLOPS_FP32":        1.5,
    "PEAK_BW_BYTES_S": 68.0e9,
}

# Roofline is an auxiliary signal. PyTorch on Ampere commonly uses TF32 for
# matmul unless explicitly disabled, so default the peak model accordingly.
ASSUMED_PRECISION = "TF32"

# Roofline ridge point: FLOPs/byte where compute and memory-bandwidth bounds
# intersect. Workloads below this AI are memory-bound; above, compute-bound.
RIDGE_AI = (ORIN_PROFILE["TFLOPS_TF32"] * 1e12) / ORIN_PROFILE["PEAK_BW_BYTES_S"]

DB_PATH = "/var/log/flop_log.db"

# INA3221 power sensor — VDD_CPU_GPU_CV rail (CPU + GPU + CV cores).
# hwmonN index can change across reboots; resolved once at startup by label.
INA3221_DRIVER = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon"
INA3221_LABEL  = "VDD_CPU_GPU_CV"

# Power-based FLOP calibration (empirical, Orin Nano, transformer workloads).
#
# Model: net active energy splits into a fixed active-overhead term and a term
# proportional to useful FLOPs:
#
#     E_net = POWER_OVERHEAD_W * t_active + E_MARGINAL_J_PER_TFLOP * TFLOPs
#
# so   TFLOPs = (E_net - POWER_OVERHEAD_W * t_active) / E_MARGINAL_J_PER_TFLOP.
#
# This replaces the old 1-D J/TFLOP-vs-power interpolation, which conflated two
# effects: small workloads look inefficient only because fixed overhead is a
# large fraction of their energy, not because their marginal J/TFLOP differs.
# The two parameters are fit by relative least squares over a sweep of
# transformer configs in eval_power_monitor.py (E_net = a*TFLOPs + b*t_active).
#
# Scope: the estimator is calibrated and validated for FRONTIER-LIKE workloads —
# runs that saturate the GPU (avg util >= 80%), which is the actual threat model
# (an unauthorized frontier training run keeps the cluster busy). On this Orin
# Nano the CPU+GPU share one power rail and no EMC (memory-bandwidth) signal is
# exposed, so a partially-loaded GPU's energy cannot be cleanly attributed to
# FLOPs; the high POWER_OVERHEAD_W below intentionally makes low-intensity
# workloads estimate near-zero (or None when overhead exceeds net energy), which
# is correct behavior for a frontier detector. Do NOT trust the estimate for
# sub-frontier runs (eval_power_monitor.py reports those separately, e.g. a 65%
# util run came out 35% low).
#
# Fit 2026-06-22 via eval_power_monitor.py over 11 frontier transformer configs
# (7 train / 4 held-out test), d_model 256->768 with batch/seq/layer variation,
# 237-451 power samples per run. Objective: relative least squares, p_overhead
# chosen by leave-one-out cross-validation. Results:
#   held-out (TEST) max err 8.14% ; all 11 frontier workloads max err 6.96%
#   (mean 2.80%), well under the 10% target.
# Constants below are the SHIP fit (refit on all 11 frontier runs).
#
# FALLBACK_IDLE_POWER_MW (602.9 mW) is the single startup idle baseline measured
# during this same eval run — it is a MATCHED SET with the constants below: the
# net energy that produced this fit was measured above 602.9 mW, so the live
# daemon's default baseline must match to keep estimates unbiased. Re-run
# eval_power_monitor.py and paste the recommended block (and this baseline) to
# recalibrate together; do not change one without the other.
FALLBACK_IDLE_POWER_MW = 602.9
POWER_OVERHEAD_W         = 3.675
E_MARGINAL_J_PER_TFLOP   = 5.89

ACTIVE_GPU_UTIL_THRESHOLD = 5.0
START_ACTIVE_POLLS = 2
STOP_QUIET_POLLS = 3
IDLE_BASELINE_MIN_SAMPLES = 5
IDLE_BASELINE_WINDOW = 120


def find_ina3221_paths():
    """Return (volt_path, curr_path) for INA3221_LABEL, or (None, None)."""
    for hwmon in glob.glob(f"{INA3221_DRIVER}/hwmon*"):
        for i in range(1, 5):
            try:
                with open(f"{hwmon}/in{i}_label") as f:
                    if f.read().strip() == INA3221_LABEL:
                        return f"{hwmon}/in{i}_input", f"{hwmon}/curr{i}_input"
            except OSError:
                continue
    return None, None


def read_power_mw(volt_path, curr_path):
    """Return VDD_CPU_GPU_CV power in milliwatts, or None on read error."""
    try:
        with open(volt_path) as f:
            mv = float(f.read().strip())
        with open(curr_path) as f:
            ma = float(f.read().strip())
        return mv * ma / 1000.0
    except OSError:
        return None


def add_column_if_missing(conn, table, column, definition):
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS flop_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT NOT NULL,
            tflops_sm        REAL,
            tflops_mem_ceil  REAL,
            estimated_tflops REAL,
            bound_type       TEXT,
            gpu_util         REAL,
            emc_util         REAL,
            power_mw         REAL,
            idle_baseline_mw REAL,
            net_power_mw     REAL,
            dt_sec           REAL,
            roofline_tflops_delta REAL,
            power_tflops_delta    REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS workload_sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time      TEXT NOT NULL,
            end_time        TEXT NOT NULL,
            duration_sec    REAL,
            total_tflops    REAL,
            power_est_tflops REAL,
            peak_gpu_util   REAL,
            poll_count      INTEGER,
            avg_power_mw    REAL,
            peak_power_mw   REAL,
            idle_baseline_mw REAL,
            net_energy_j    REAL,
            avg_net_power_mw REAL,
            avg_emc_util    REAL,
            avg_freq_mhz    REAL,
            estimator       TEXT
        )
    """)
    for column, definition in (
        ("idle_baseline_mw", "REAL"),
        ("net_power_mw", "REAL"),
        ("dt_sec", "REAL"),
        ("roofline_tflops_delta", "REAL"),
        ("power_tflops_delta", "REAL"),
    ):
        add_column_if_missing(conn, "flop_log", column, definition)
    for column, definition in (
        ("idle_baseline_mw", "REAL"),
        ("net_energy_j", "REAL"),
        ("avg_net_power_mw", "REAL"),
        ("avg_emc_util", "REAL"),
        ("avg_freq_mhz", "REAL"),
        ("estimator", "TEXT"),
    ):
        add_column_if_missing(conn, "workload_sessions", column, definition)
    conn.commit()
    return conn


def current_idle_baseline_mw(quiet_power_samples):
    # Use the calibration baseline for repeatable run-to-run estimates. The
    # live quiet samples remain useful diagnostics, but letting them redefine
    # idle between workloads makes identical runs hard to compare.
    return FALLBACK_IDLE_POWER_MW


def observed_idle_baseline_mw(quiet_power_samples):
    if len(quiet_power_samples) >= IDLE_BASELINE_MIN_SAMPLES:
        return median(quiet_power_samples)
    return FALLBACK_IDLE_POWER_MW


def estimate_tflops(net_energy_j, active_time_s,
                    p_overhead_w=POWER_OVERHEAD_W,
                    e_marginal_j_per_tflop=E_MARGINAL_J_PER_TFLOP):
    """
    2-parameter active-energy model — the single canonical FLOP estimator,
    shared by the live daemon and eval_power_monitor.py.

        TFLOPs = (E_net - p_overhead_w * t_active) / e_marginal_j_per_tflop

    Subtracts the fixed active-overhead energy (power drawn while a workload is
    resident but not attributable to useful FLOPs) before converting the
    remaining energy at the marginal J/TFLOP rate. Params are overridable so the
    eval can score candidate fits through this exact code path.

    Returns estimated TFLOPs, or None when inputs are non-positive or the
    overhead term swamps the measured energy (no useful compute detected).
    """
    if net_energy_j <= 0 or active_time_s <= 0 or e_marginal_j_per_tflop <= 0:
        return None
    flop_energy_j = net_energy_j - p_overhead_w * active_time_s
    if flop_energy_j <= 0:
        return None
    return flop_energy_j / e_marginal_j_per_tflop


def parse_emc_util(line):
    m = re.search(r'EMC_FREQ\s+(\d+)%', line)
    return float(m.group(1)) if m else None


class TegrastatsReader:
    def __init__(self, interval_ms):
        self.interval_ms = interval_ms
        self.proc = None
        self.latest_emc_util = None

    def start(self):
        if self.proc is not None and self.proc.poll() is None:
            return
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError:
            self.proc = None

    def read_emc_util(self):
        """
        Return the most recent EMC utilization sample without blocking.
        Falls back to None until tegrastats has emitted at least one line.
        """
        self.start()
        if self.proc is None or self.proc.stdout is None:
            return None

        while True:
            ready, _, _ = select.select([self.proc.stdout], [], [], 0)
            if not ready:
                break
            line = self.proc.stdout.readline()
            if not line:
                break
            emc_util = parse_emc_util(line)
            if emc_util is not None:
                self.latest_emc_util = emc_util
        return self.latest_emc_util

    def close(self):
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None


def get_emc_util():
    """
    EMC bandwidth utilization % via tegrastats.
    jtop 4.x does not expose this on JetPack 6.2 — stats['EMC'] returns 0
    and jetson.emc does not exist. tegrastats EMC_FREQ field gives 'x%@yMHz'
    where x is bandwidth utilization (0-100).
    Returns float or None if tegrastats is unavailable.
    """
    proc = None
    try:
        proc = subprocess.Popen(
            ["tegrastats"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        line = proc.stdout.readline()
        return parse_emc_util(line)
    except BaseException:
        return None
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()


def get_freq_hz(jetson):
    """
    Return (cur_hz, max_hz) from jtop 4.x.
    jetson.gpu['gpu']['freq'] values are in kHz on JetPack 6.2.
    Falls back to ORIN_PROFILE MAX_FREQ_HZ if the key is absent.
    """
    try:
        freq = jetson.gpu.get('gpu', {}).get('freq', {})
        cur, mx = freq.get('cur', 0), freq.get('max', 0)
        if cur > 0 and mx > 0:
            return cur * 1000, mx * 1000
    except Exception:
        pass
    return ORIN_PROFILE["MAX_FREQ_HZ"], ORIN_PROFILE["MAX_FREQ_HZ"]


def run_background_monitor(poll_interval=1.5):
    active_workload_tracked = False
    active_poll_streak = 0
    quiet_poll_streak = 0
    quiet_power_samples = deque(maxlen=IDLE_BASELINE_WINDOW)

    conn = init_db(DB_PATH)
    volt_path, curr_path = find_ina3221_paths()
    if volt_path:
        print(f"INA3221 power sensor: {INA3221_LABEL} found.")
    else:
        print(f"INA3221 power sensor: {INA3221_LABEL} not found; power column will be NULL.")

    print("Daemon initialized. Monitoring Orin Nano hardware channels for new ML workloads...")
    tegrastats_reader = TegrastatsReader(int(poll_interval * 1000))

    session = {}

    def start_session(timestamp, now_mono):
        idle_baseline_mw = current_idle_baseline_mw(quiet_power_samples)
        session.clear()
        session.update({
            "start_time": timestamp,
            "start_mono": now_mono,
            "idle_baseline_mw": idle_baseline_mw,
            "roofline_tflops": 0.0,
            "net_energy_j": 0.0,
            "active_time_s": 0.0,
            "peak_util": 0.0,
            "poll_count": 0,
            "power_sum": 0.0,
            "net_power_sum": 0.0,
            "peak_power": 0.0,
            "emc_sum": 0.0,
            "emc_count": 0,
            "freq_sum_mhz": 0.0,
        })
        print("\n[!] NEW WORKLOAD DETECTED")
        observed_idle_mw = observed_idle_baseline_mw(quiet_power_samples)
        print(f"    Idle baseline : {idle_baseline_mw:.0f} mW calibrated"
              f" | observed quiet median: {observed_idle_mw:.0f} mW"
              f" ({len(quiet_power_samples)} quiet samples)")

    def finish_session(timestamp, now_mono, power_mw):
        duration_sec = now_mono - session["start_mono"]
        poll_count = session["poll_count"]
        avg_power_mw = session["power_sum"] / poll_count if poll_count else None
        avg_net_power_mw = session["net_power_sum"] / poll_count if poll_count else None
        avg_emc_util = (session["emc_sum"] / session["emc_count"]
                        if session["emc_count"] else None)
        avg_freq_mhz = session["freq_sum_mhz"] / poll_count if poll_count else None

        power_est_tflops = estimate_tflops(
            session["net_energy_j"], session["active_time_s"]
        )
        j_per_tflop = (session["net_energy_j"] / power_est_tflops
                       if power_est_tflops else None)

        conn.execute(
            "INSERT INTO workload_sessions"
            " (start_time, end_time, duration_sec, total_tflops, power_est_tflops,"
            "  peak_gpu_util, poll_count, avg_power_mw, peak_power_mw,"
            "  idle_baseline_mw, net_energy_j, avg_net_power_mw, avg_emc_util,"
            "  avg_freq_mhz, estimator)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session["start_time"], timestamp, duration_sec,
             session["roofline_tflops"], power_est_tflops,
             session["peak_util"], poll_count,
             avg_power_mw, session["peak_power"] or None,
             session["idle_baseline_mw"], session["net_energy_j"],
             avg_net_power_mw, avg_emc_util, avg_freq_mhz, "power_energy_v1")
        )
        conn.commit()

        print("\n[---] Workload completed or detached.")
        print(f"      Duration           : {duration_sec:.1f}s  ({poll_count} active polls)")
        if power_est_tflops is not None:
            print(f"      Power est.         : {power_est_tflops:.4f} TFLOPs"
                  f"  ({j_per_tflop:.2f} J/TFLOP)")
        print(f"      Roofline diagnostic: {session['roofline_tflops']:.4f} TFLOPs"
              f"  ({ASSUMED_PRECISION})")
        print(f"      Peak util          : {session['peak_util']:.1f}%")
        if avg_power_mw is not None:
            print(f"      Avg power          : {avg_power_mw:.0f} mW"
                  f"  |  Avg net: {avg_net_power_mw:.0f} mW"
                  f"  |  Peak: {session['peak_power']:.0f} mW")
        print("      Returning to quiet loop.")

        session.clear()
        if power_mw is not None:
            quiet_power_samples.append(power_mw)

    try:
        with jtop() as jetson:
            last_poll_mono = time.monotonic()

            while jetson.ok():
                now_mono = time.monotonic()
                dt_sec = max(now_mono - last_poll_mono, 0.0)
                last_poll_mono = now_mono

                stats = jetson.stats
                gpu_util = stats.get('GPU', 0)
                emc_util = tegrastats_reader.read_emc_util()
                cur_freq_hz, max_freq_hz = get_freq_hz(jetson)
                power_mw = read_power_mw(volt_path, curr_path) if volt_path else None
                timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
                is_active_sample = gpu_util > ACTIVE_GPU_UTIL_THRESHOLD

                if is_active_sample:
                    active_poll_streak += 1
                    quiet_poll_streak = 0
                else:
                    active_poll_streak = 0
                    if active_workload_tracked:
                        quiet_poll_streak += 1
                    elif power_mw is not None:
                        quiet_power_samples.append(power_mw)

                if (not active_workload_tracked
                        and active_poll_streak >= START_ACTIVE_POLLS):
                    active_workload_tracked = True
                    quiet_poll_streak = 0
                    start_session(timestamp, now_mono)

                if active_workload_tracked and is_active_sample:
                    peak_tflops = ORIN_PROFILE[f"TFLOPS_{ASSUMED_PRECISION}"]
                    tflops_sm = peak_tflops * (cur_freq_hz / max_freq_hz) * (gpu_util / 100.0)

                    if emc_util is not None and emc_util > 0:
                        emc_bw_actual = (emc_util / 100.0) * ORIN_PROFILE["PEAK_BW_BYTES_S"]
                        tflops_mem_ceil = (emc_bw_actual * RIDGE_AI) / 1e12
                        estimated_tflops = min(tflops_sm, tflops_mem_ceil)
                        bound_type = "compute" if tflops_sm <= tflops_mem_ceil else "memory"
                    else:
                        emc_bw_actual = None
                        tflops_mem_ceil = None
                        estimated_tflops = tflops_sm
                        bound_type = "unknown"

                    roofline_tflops_delta = estimated_tflops * dt_sec
                    session["roofline_tflops"] += roofline_tflops_delta

                    net_power_mw = None
                    power_tflops_delta = None
                    if power_mw is not None:
                        net_power_mw = max(power_mw - session["idle_baseline_mw"], 0.0)
                        net_power_w = net_power_mw / 1000.0
                        net_energy_delta_j = net_power_w * dt_sec
                        session["net_energy_j"] += net_energy_delta_j
                        # Per-poll FLOP energy after removing this interval's
                        # share of fixed active overhead. Summing these deltas
                        # equals estimate_tflops() over the whole session; an
                        # individual delta may be negative on near-idle polls.
                        flop_energy_delta_j = (net_energy_delta_j
                                               - POWER_OVERHEAD_W * dt_sec)
                        power_tflops_delta = flop_energy_delta_j / E_MARGINAL_J_PER_TFLOP

                        session["power_sum"] += power_mw
                        session["net_power_sum"] += net_power_mw
                        session["peak_power"] = max(session["peak_power"], power_mw)

                    session["peak_util"] = max(session["peak_util"], gpu_util)
                    session["poll_count"] += 1
                    session["active_time_s"] += dt_sec
                    session["freq_sum_mhz"] += cur_freq_hz / 1e6
                    if emc_util is not None:
                        session["emc_sum"] += emc_util
                        session["emc_count"] += 1

                    power_est_so_far = estimate_tflops(
                        session["net_energy_j"], session["active_time_s"]
                    )
                    j_per_tflop_so_far = (
                        session["net_energy_j"] / power_est_so_far
                        if power_est_so_far else None
                    )

                    conn.execute(
                        "INSERT INTO flop_log"
                        " (timestamp, tflops_sm, tflops_mem_ceil, estimated_tflops,"
                        "  bound_type, gpu_util, emc_util, power_mw, idle_baseline_mw,"
                        "  net_power_mw, dt_sec, roofline_tflops_delta, power_tflops_delta)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (timestamp, tflops_sm, tflops_mem_ceil, estimated_tflops,
                         bound_type, gpu_util, emc_util, power_mw,
                         session["idle_baseline_mw"], net_power_mw, dt_sec,
                         roofline_tflops_delta, power_tflops_delta)
                    )
                    conn.commit()

                    print("--------------------------------------------------")
                    print(f"Hardware Clock     : {cur_freq_hz/1e6:.1f} MHz  (max {max_freq_hz/1e6:.0f} MHz)")
                    print(f"GPU / EMC Load     : {gpu_util}% / {emc_util}%")
                    print(f"Poll interval      : {dt_sec:.2f}s observed")
                    if power_mw is not None:
                        print(f"VDD_CPU_GPU_CV     : {power_mw:.0f} mW"
                              f"  |  net {net_power_mw:.0f} mW"
                              f" above idle {session['idle_baseline_mw']:.0f} mW")
                    print(f"SM busy estimate   : {tflops_sm:.3f} TFLOPS ({ASSUMED_PRECISION}, diagnostic)")
                    if tflops_mem_ceil is not None:
                        print(f"Memory BW ceiling  : {tflops_mem_ceil:.3f} TFLOPS"
                              f"  ({emc_bw_actual/1e9:.1f} GB/s x {RIDGE_AI:.0f} FLOPs/byte ridge)")
                        print(f"Roofline diagnostic: {estimated_tflops:.3f} TFLOPS  [{bound_type}-bound]")
                    else:
                        print(f"Roofline diagnostic: {estimated_tflops:.3f} TFLOPS  [EMC unavailable]")
                    if power_est_so_far is not None:
                        print(f"Power estimate     : {power_est_so_far:.4f} TFLOPs"
                              f"  ({j_per_tflop_so_far:.2f} J/TFLOP effective)")
                    print(f"Roofline diagnostic total: {session['roofline_tflops']:.4f} TFLOPs")

                elif active_workload_tracked and quiet_poll_streak >= STOP_QUIET_POLLS:
                    finish_session(timestamp, now_mono, power_mw)
                    active_workload_tracked = False
                    active_poll_streak = 0
                    quiet_poll_streak = 0

                elif not active_workload_tracked:
                    idle_baseline_mw = current_idle_baseline_mw(quiet_power_samples)
                    observed_idle_mw = observed_idle_baseline_mw(quiet_power_samples)
                    power_str = f" | power={power_mw:.0f}mW" if power_mw is not None else ""
                    print(f"[quiet] gpu={gpu_util:.1f}% | emc={emc_util}%"
                          f" | freq={cur_freq_hz/1e6:.0f}MHz"
                          f" | idle_base={idle_baseline_mw:.0f}mW"
                          f" | observed_idle={observed_idle_mw:.0f}mW{power_str}"
                          f" | active_streak={active_poll_streak}/{START_ACTIVE_POLLS}")

                time.sleep(poll_interval)
    finally:
        tegrastats_reader.close()


if __name__ == "__main__":
    run_background_monitor()
