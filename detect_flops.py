import os
import time
import re
import sqlite3
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from statistics import median

# ── PORTED to x86 dual-Tesla-V100 (Ubuntu 24.04) ─────────────────────────────
# This experiment was originally hardwired to a Jetson Orin Nano (INA3221 power
# sensor, Tegra actmon/EMC debugfs counters, tegrastats, jtop, nvpmodel). On this
# discrete-GPU server the same three signals are re-sourced, staying faithful to
# what each measured on the Jetson (see README "Reproduced on dual-V100"):
#   board power (mW)    -> nvidia-smi power.draw for the monitored GPU (unpriv.)
#   GPU utilization (%) -> nvidia-smi utilization.gpu   (the frontier gate)
#   DRAM activity       -> DCGM field 1005 DCGM_FI_PROF_DRAM_ACTIVE (fraction of
#                          cycles the memory interface is active) via `dcgmi dmon`
#                          -- the semantic analog of the Jetson actmon bandwidth
#                          fraction. Requires a root `nv-hostengine`; unprivileged
#                          dcgmi clients read through it, mirroring the Jetson's
#                          privileged-reader / unprivileged-sweep split.
# The estimator math, SQLite schema, and workload-detection logic are unchanged.

# Which physical GPU to monitor. CUDA_DEVICE_ORDER=PCI_BUS_ID (set below) makes
# the nvidia-smi index match the CUDA device index, so the workload pinned to
# cuda:GPU_INDEX is the one we measure.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
GPU_INDEX = int(os.environ.get("FLOP_GPU_INDEX", "0"))

# Tesla V100-SXM2-16GB profile. PEAK_BW_BYTES_S is the HBM2 peak (900 GB/s);
# TFLOPS peaks feed the roofline diagnostic only (not the trusted estimate).
# V100 (Volta) has no TF32 path, so the roofline assumes FP32.
V100_PROFILE = {
    "MAX_FREQ_HZ":     1530_000_000,   # SM boost clock
    "TFLOPS_FP16":      125.0,         # tensor-core peak
    "TFLOPS_FP32":       15.7,
    "PEAK_BW_BYTES_S":  900.0e9,       # HBM2 peak bandwidth
}

# Roofline is an auxiliary signal. On V100 the matmul default is FP32.
ASSUMED_PRECISION = "FP32"

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
# /var/log needs root; on this box the daemon runs unprivileged, so the SQLite
# log lives in the repo. Override with FLOP_DB_PATH.
DB_PATH = os.environ.get("FLOP_DB_PATH", os.path.join(REPO_ROOT, "flop_log.db"))

# DCGM profiling field id for DRAM active (fraction 0..1).
DRAM_ACTIVE_FIELD = 1005

# Power-based FLOP calibration (empirical, transformer workloads).
#
# Model: net active energy splits into a fixed active-overhead term and a term
# proportional to useful FLOPs:
#
#     E_net = POWER_OVERHEAD_W * t_active + E_MARGINAL_J_PER_TFLOP * TFLOPs
#
# so   TFLOPs = (E_net - POWER_OVERHEAD_W * t_active) / E_MARGINAL_J_PER_TFLOP.
#
# The two parameters are fit by relative least squares over a sweep of
# transformer configs in eval_power_monitor.py (E_net = a*TFLOPs + b*t_active).
#
# Scope: the estimator is calibrated and validated for FRONTIER-LIKE workloads —
# runs that saturate the GPU (avg util >= 80%), which is the actual threat model
# (an unauthorized frontier training run keeps the cluster busy). A <=0 /
# sub-frontier estimate just means "below the frontier detection floor", not a
# real FLOP count.
#
# ┌─ RECOMMENDED CONSTANTS (paste block from eval_power_monitor.py) ───────────┐
# Fit 2026-07-22 on this dual-V100 box (eval_results_v2 sweep), fp32-frontier
# subset — 12 transformer configs at avg GPU util >= 80%, matching the report's
# fp32-frontier methodology. 2-param all-frontier max err 13.86% (mean 8.66%);
# 3-param max 14.72% (mean 6.22%). The V100 idles ~49 W (vs the Jetson's 0.64 W)
# and only 12 configs clear the 80% gate (vs 21 on the Orin Nano), so the held-out
# error is a touch looser than the Jetson's <10%. FALLBACK_IDLE_POWER_MW is the
# matched startup idle baseline (median over 90 s); all values are ONE matched set.
FALLBACK_IDLE_POWER_MW = 48930.0
POWER_OVERHEAD_W         = 3.750
E_MARGINAL_J_PER_TFLOP   = 18.88

# EMC / DRAM-bytes term (memory-energy) — its OWN matched constants.
#
# A second, PARALLEL estimator (estimate_tflops_emc) splits net energy into three
# terms, adding a measured memory-energy term proportional to DRAM bytes moved:
#
#     E_net = E_MARGINAL_EMC_J_PER_TFLOP * TFLOPs
#           + E_PER_TB_J * TB_moved          <- observed (not fitted) per run
#           + POWER_OVERHEAD_EMC_W * t_active
#
# so TFLOPs = (E_net - E_PER_TB_J*TB_moved - POWER_OVERHEAD_EMC_W*t) / E_MARGINAL_EMC.
#
# Motivation is adversarial: an energy-only estimator can be spoofed by a workload
# operating OFF the normal FLOPs<->bytes line (low-FLOP/high-memory traffic to
# inflate energy). The byte term, fed by the DCGM DRAM-activity fraction below,
# is what catches that. TB_moved comes from integrating DRAM bandwidth
# utilization over the workload; see DcgmDramReader. The byte scale is absorbed
# into the fitted E_PER_TB_J. PLACEHOLDERS — recalibrate on V100.
POWER_OVERHEAD_EMC_W       = 9.750
E_MARGINAL_EMC_J_PER_TFLOP = 8.21
E_PER_TB_J                 = 188.022   # J per true TB — physically sane (HBM2
                                       # ~50-150 J/TB range); the DCGM DRAM-active
                                       # fraction is a true bandwidth fraction, so
                                       # unlike the Jetson's uncalibrated actmon
                                       # (4621 J/actmon-TB) this is directly J/TB.
# └────────────────────────────────────────────────────────────────────────────┘

# Fingerprint of the device the constants above were calibrated on. On any other
# GPU the numbers are meaningless until recalibrated: re-run eval_power_monitor.py
# and paste its RECOMMENDED CONSTANTS block plus this fingerprint. The daemon
# compares this against the live system at startup and warns loudly on mismatch.
CALIBRATION_FINGERPRINT = {
    "device_model": "Tesla V100-SXM2-16GB",     # nvidia-smi --query-gpu=name
    "driver_version": "580.159.03",             # nvidia-smi --query-gpu=driver_version
    "cuda_version": "13.0",                      # nvidia-smi CUDA Version
    "calibrated": "2026-07-22",
}


def _nvidia_smi(fields, gpu_index=GPU_INDEX):
    """Query per-GPU nvidia-smi fields for one GPU, return a list of string values
    (csv, no header, no units). Raises on failure — nvidia-smi is the primary
    signal, so an unreadable query is a bug to surface, not mask."""
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}",
         "--format=csv,noheader,nounits", "-i", str(gpu_index)],
        capture_output=True, text=True, check=True).stdout.strip()
    return [v.strip() for v in out.split(",")]


def read_power_mw(gpu_index=GPU_INDEX):
    """Return the monitored GPU's board power in milliwatts (nvidia-smi
    power.draw, W -> mW). The discrete-GPU analog of the Jetson INA3221 rail;
    per-GPU here, so no shared-rail attribution problem. Raises on read error."""
    (p,) = _nvidia_smi("power.draw", gpu_index)
    return float(p) * 1000.0


def read_gpu_sample(gpu_index=GPU_INDEX):
    """One nvidia-smi poll: (power_mw, gpu_util_pct, sm_clock_hz)."""
    p, util, clk = _nvidia_smi("power.draw,utilization.gpu,clocks.sm", gpu_index)
    return float(p) * 1000.0, float(util), float(clk) * 1e6


def live_fingerprint(gpu_index=GPU_INDEX):
    """The running system's counterpart to CALIBRATION_FINGERPRINT (nvidia-smi
    GPU name + driver + CUDA version). Raises if a source is unreadable — a
    monitor that can't identify its own hardware should not pretend the
    calibration applies."""
    name, driver = _nvidia_smi("name,driver_version", gpu_index)
    # CUDA version isn't a per-GPU query field; scrape it from the header.
    out = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                         check=True).stdout
    m = re.search(r"CUDA Version:\s*([\d.]+)", out)
    cuda = m.group(1) if m else "unknown"
    return {"device_model": name, "driver_version": driver, "cuda_version": cuda}


def check_calibration_fingerprint():
    """Compare live hardware against what the constants were calibrated on; print
    a loud RECALIBRATE banner on any mismatch. Returns True when everything
    matches."""
    live = live_fingerprint()
    mismatches = [
        (k, CALIBRATION_FINGERPRINT[k], live[k])
        for k in ("device_model", "driver_version", "cuda_version")
        if CALIBRATION_FINGERPRINT[k] != live[k]
    ]
    if not mismatches:
        print(f"Calibration fingerprint OK: {live['device_model']}"
              f" | driver {live['driver_version']} | CUDA {live['cuda_version']}"
              f" (calibrated {CALIBRATION_FINGERPRINT['calibrated']})")
        return True
    print("!" * 78)
    print("!! CALIBRATION MISMATCH — estimates from this daemon are NOT valid !!")
    for key, want, got in mismatches:
        print(f"!!   {key}: calibrated on {want!r}, running on {got!r}")
    print("!! The estimator constants are a matched set for the calibrated")
    print("!! device only. RECALIBRATE: run eval_power_monitor.py on THIS system")
    print("!! and paste its RECOMMENDED CONSTANTS (and matched idle baseline +")
    print("!! this fingerprint) into detect_flops.py.")
    print("!" * 78)
    return False


ACTIVE_GPU_UTIL_THRESHOLD = 5.0
START_ACTIVE_POLLS = 2
STOP_QUIET_POLLS = 3
IDLE_BASELINE_MIN_SAMPLES = 5
IDLE_BASELINE_WINDOW = 120


def add_column_if_missing(conn, table, column, definition):
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


# Single source of truth for the SQLite schema, in exact column order (an
# implicit `id INTEGER PRIMARY KEY AUTOINCREMENT` comes first in each table).
# APPEND ONLY: never reorder or rename — init_db additively migrates existing
# flop_log.db files by adding any missing column at the end.
SCHEMA = {
    "flop_log": [
        ("timestamp",             "TEXT NOT NULL"),
        ("tflops_sm",             "REAL"),
        ("tflops_mem_ceil",       "REAL"),  # always NULL (dead roofline memory ceiling)
        ("estimated_tflops",      "REAL"),
        ("bound_type",            "TEXT"),  # always "unknown" (ditto)
        ("gpu_util",              "REAL"),
        ("emc_util",              "REAL"),  # always NULL (kept for schema compat)
        ("power_mw",              "REAL"),
        ("idle_baseline_mw",      "REAL"),
        ("net_power_mw",          "REAL"),
        ("dt_sec",                "REAL"),
        ("roofline_tflops_delta", "REAL"),
        ("power_tflops_delta",    "REAL"),
        # EMC/bytes term (parallel estimator)
        ("bytes_delta",            "REAL"),
        ("actmon_util",            "REAL"),  # now the DCGM DRAM-active fraction
        ("power_emc_tflops_delta", "REAL"),
    ],
    "workload_sessions": [
        ("start_time",       "TEXT NOT NULL"),
        ("end_time",         "TEXT NOT NULL"),
        ("duration_sec",     "REAL"),
        ("total_tflops",     "REAL"),
        ("power_est_tflops", "REAL"),
        ("peak_gpu_util",    "REAL"),
        ("poll_count",       "INTEGER"),
        ("avg_power_mw",     "REAL"),
        ("peak_power_mw",    "REAL"),
        ("idle_baseline_mw", "REAL"),
        ("net_energy_j",     "REAL"),
        ("avg_net_power_mw", "REAL"),
        ("avg_emc_util",     "REAL"),  # always NULL (kept for schema compat)
        ("avg_freq_mhz",     "REAL"),
        ("estimator",        "TEXT"),
        # EMC/bytes term (parallel estimator)
        ("tb_moved",             "REAL"),
        ("power_est_tflops_emc", "REAL"),
        ("estimator_emc",        "TEXT"),
    ],
}


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    for table, columns in SCHEMA.items():
        cols = ", ".join(
            ["id INTEGER PRIMARY KEY AUTOINCREMENT"]
            + [f"{name} {definition}" for name, definition in columns])
        conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({cols})")
        for name, definition in columns:
            add_column_if_missing(conn, table, name, definition)
    conn.commit()
    return conn


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

    Returns the model's estimated TFLOPs. The value may be <= 0 for sub-frontier /
    low-intensity workloads where the fixed overhead exceeds the net energy — that
    is the model speaking outside its calibrated (frontier) regime, not an error,
    so it is returned rather than clamped to None. Returns None only for a
    degenerate call (no active time, or a non-positive marginal constant).
    """
    # Exactly the 3-param model with a zero byte term (subtracting 0.0 is
    # IEEE-exact), so there is a single implementation of the energy model.
    return estimate_tflops_emc(net_energy_j, active_time_s, 0.0,
                               p_overhead_w=p_overhead_w,
                               e_marginal_j_per_tflop=e_marginal_j_per_tflop,
                               e_per_tb_j=0.0)


def estimate_tflops_emc(net_energy_j, active_time_s, tb_moved,
                        p_overhead_w=POWER_OVERHEAD_EMC_W,
                        e_marginal_j_per_tflop=E_MARGINAL_EMC_J_PER_TFLOP,
                        e_per_tb_j=E_PER_TB_J):
    """
    3-parameter active-energy model — the EMC/bytes-aware estimator, run in
    parallel with estimate_tflops for A/B (NOT a replacement).

        TFLOPs = (E_net - e_per_tb_j*tb_moved - p_overhead_w*t_active)
                 / e_marginal_j_per_tflop

    Subtracts a measured memory-energy term (e_per_tb_j * TB moved) in addition to
    the fixed active overhead before converting the remainder at the marginal
    J/TFLOP rate. With tb_moved=0 and e_per_tb_j=0 this is identical to
    estimate_tflops, so the 2-param behavior is exactly recoverable. Params are
    overridable so the eval can score candidate fits through this exact path.

    Returns the model's estimated TFLOPs (may be <= 0 for sub-frontier workloads
    where the overhead + memory-energy terms exceed the net energy — returned, not
    clamped). Returns None only for a degenerate call (no active time, or a
    non-positive marginal constant).
    """
    if active_time_s <= 0 or e_marginal_j_per_tflop <= 0:
        return None
    flop_energy_j = (net_energy_j
                     - e_per_tb_j * tb_moved
                     - p_overhead_w * active_time_s)
    return flop_energy_j / e_marginal_j_per_tflop


def _tflops_delta(net_energy_delta_j, dt_sec, tb_delta,
                  p_overhead_w, e_marginal_j_per_tflop, e_per_tb_j):
    """
    Per-poll increment of the active-energy model — the same formula as
    estimate_tflops_emc (of which estimate_tflops is the tb=0 special case),
    without the degenerate-input guard so a zero-length poll contributes 0
    rather than None. Summing these deltas over a session reconciles with the
    corresponding estimator total; an individual delta may be negative on
    near-idle polls.
    """
    return ((net_energy_delta_j
             - e_per_tb_j * tb_delta
             - p_overhead_w * dt_sec)
            / e_marginal_j_per_tflop)


def actmon_bytes_per_s(util_fraction, emc_rate_hz=None):
    """
    Convert a DRAM bandwidth-utilization fraction to DRAM bytes/s. Single source
    of truth shared by the daemon's DcgmDramReader and the calibration
    BytesSampler.

        bytes/s = util_fraction * PEAK_BW_BYTES_S

    util_fraction is DCGM field 1005 (DCGM_FI_PROF_DRAM_ACTIVE) — the fraction of
    cycles the HBM2 memory interface was active — the discrete-GPU analog of the
    Jetson actmon bandwidth fraction. V100 memory runs at a fixed clock, so unlike
    the Jetson (whose EMC DVFS-downclocks) there is no emc_rate correction; the
    parameter is accepted and ignored for call-site compatibility.

    Any constant scale error in the fraction is absorbed by the fitted E_PER_TB_J
    (TB_obs = k*TB_true => fitted c = E_PER_TB/k, and c*TB_obs is unchanged), so
    only proportionality matters here, not the absolute byte count.
    """
    return util_fraction * V100_PROFILE["PEAK_BW_BYTES_S"]


def _parse_dcgm_dram_line(line):
    """Parse one `dcgmi dmon -e 1005` data line -> DRAM-active fraction (float),
    or None for headers/blank/non-data lines. Data lines look like:
        GPU 0    0.123456
    (a 'GPU'/'Id' prefix, the entity index, then the field value). Robust to
    column spacing; takes the last float token on a GPU line."""
    s = line.strip()
    if not s or s.startswith("#") or s.startswith("Id"):
        return None
    if not s.startswith("GPU"):
        return None
    toks = s.split()
    for tok in reversed(toks):
        try:
            return float(tok)
        except ValueError:
            continue
    return None


class DcgmDramReader:
    """
    Reads the DRAM-active fraction from a persistent `dcgmi dmon -e 1005` stream
    (DCGM field 1005, DCGM_FI_PROF_DRAM_ACTIVE) and converts it to bytes/s. This
    is the memory-bandwidth signal feeding the EMC estimator — the V100 analog of
    the Jetson actmon counter.

    A background thread keeps latest_util updated from the stream. read_bytes_per_s()
    RAISES if no sample has arrived yet or the stream died (e.g. nv-hostengine not
    running), so a broken DCGM setup is surfaced immediately rather than silently
    disabling the EMC estimate.
    """

    def __init__(self, gpu_index=GPU_INDEX, interval_ms=500, warmup_timeout=8.0):
        self.gpu_index = gpu_index
        self.latest_util = None
        self._exc = None
        self.proc = subprocess.Popen(
            ["dcgmi", "dmon", "-e", str(DRAM_ACTIVE_FIELD),
             "-i", str(gpu_index), "-d", str(interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # The stream is async (background thread) — unlike the Jetson's synchronous
        # actmon file read — so block until the first sample arrives, else the very
        # first poll would race ahead of dcgmi's startup. A stream that never emits
        # (host engine down / field unavailable) is surfaced here, at startup.
        deadline = time.monotonic() + warmup_timeout
        while self.latest_util is None and self._exc is None:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    "dcgmi dmon exited before producing a DRAM-active sample — is "
                    "nv-hostengine running (sudo nv-hostengine) and does "
                    "`dcgmi dmon -e 1005` work on this GPU?")
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "DCGM DRAM-active stream produced no sample within "
                    f"{warmup_timeout:.0f}s — is nv-hostengine running?")
            time.sleep(0.1)
        if self._exc is not None:
            raise self._exc

    def _run(self):
        try:
            for line in iter(self.proc.stdout.readline, ""):
                frac = _parse_dcgm_dram_line(line)
                if frac is not None:
                    self.latest_util = max(0.0, min(1.0, frac))
        except BaseException as e:  # pragma: no cover - surfaced on read
            self._exc = e

    def read_bytes_per_s(self):
        if self._exc is not None:
            raise self._exc
        if self.latest_util is None:
            raise RuntimeError(
                "DCGM DRAM-active stream produced no samples — is nv-hostengine "
                "running (sudo nv-hostengine) and does `dcgmi dmon -e 1005` work?")
        return actmon_bytes_per_s(self.latest_util)


@dataclass
class Session:
    """Accumulator for one tracked workload, from the poll that trips the
    START hysteresis until STOP_QUIET_POLLS quiet polls end it."""
    start_time: str
    start_mono: float
    idle_baseline_mw: float
    roofline_tflops: float = 0.0
    net_energy_j: float = 0.0
    active_time_s: float = 0.0
    peak_util: float = 0.0
    poll_count: int = 0
    power_sum: float = 0.0
    net_power_sum: float = 0.0
    peak_power: float = 0.0
    freq_sum_mhz: float = 0.0
    net_bytes: float = 0.0


def run_background_monitor(poll_interval=1.5, gpu_index=GPU_INDEX):
    active_poll_streak = 0
    quiet_poll_streak = 0
    quiet_power_samples = deque(maxlen=IDLE_BASELINE_WINDOW)

    conn = init_db(DB_PATH)
    # nvidia-smi power is the primary signal; a failing probe read is a hard error.
    read_power_mw(gpu_index)
    print(f"nvidia-smi power probe OK (GPU {gpu_index}). DB: {DB_PATH}")
    check_calibration_fingerprint()

    print(f"Daemon initialized. Monitoring GPU {gpu_index} for new ML workloads...")
    dram_reader = DcgmDramReader(gpu_index)
    max_freq_hz = V100_PROFILE["MAX_FREQ_HZ"]

    # session is None while quiet; a Session object doubles as the
    # "workload currently tracked" state.
    session = None

    def start_session(timestamp, now_mono):
        # Always the calibrated constant, for repeatable run-to-run estimates.
        session = Session(start_time=timestamp, start_mono=now_mono,
                          idle_baseline_mw=FALLBACK_IDLE_POWER_MW)
        print("\n[!] NEW WORKLOAD DETECTED")
        observed_idle_mw = observed_idle_baseline_mw(quiet_power_samples)
        print(f"    Idle baseline : {session.idle_baseline_mw:.0f} mW calibrated"
              f" | observed quiet median: {observed_idle_mw:.0f} mW"
              f" ({len(quiet_power_samples)} quiet samples)")
        return session

    def finish_session(session, timestamp, now_mono, power_mw):
        duration_sec = now_mono - session.start_mono
        poll_count = session.poll_count
        avg_power_mw = session.power_sum / poll_count if poll_count else None
        avg_net_power_mw = session.net_power_sum / poll_count if poll_count else None
        avg_emc_util = None
        avg_freq_mhz = session.freq_sum_mhz / poll_count if poll_count else None

        power_est_tflops = estimate_tflops(
            session.net_energy_j, session.active_time_s
        )

        tb_moved = session.net_bytes / 1e12
        power_est_tflops_emc = estimate_tflops_emc(
            session.net_energy_j, session.active_time_s, tb_moved
        )

        conn.execute(
            "INSERT INTO workload_sessions"
            " (start_time, end_time, duration_sec, total_tflops, power_est_tflops,"
            "  peak_gpu_util, poll_count, avg_power_mw, peak_power_mw,"
            "  idle_baseline_mw, net_energy_j, avg_net_power_mw, avg_emc_util,"
            "  avg_freq_mhz, estimator, tb_moved, power_est_tflops_emc, estimator_emc)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session.start_time, timestamp, duration_sec,
             session.roofline_tflops, power_est_tflops,
             session.peak_util, poll_count,
             avg_power_mw, session.peak_power or None,
             session.idle_baseline_mw, session.net_energy_j,
             avg_net_power_mw, avg_emc_util, avg_freq_mhz, "power_energy_v1",
             tb_moved, power_est_tflops_emc, "power_energy_emc_v1")
        )
        conn.commit()

        print("\n[---] Workload completed or detached.")
        print(f"      Duration           : {duration_sec:.1f}s  ({poll_count} active polls)")
        if power_est_tflops is not None:
            note = (f"{session.net_energy_j / power_est_tflops:.2f} J/TFLOP"
                    if power_est_tflops > 0
                    else "<=0: sub-frontier, overhead exceeds net energy")
            print(f"      Power est.         : {power_est_tflops:.4f} TFLOPs  ({note})")
        if power_est_tflops_emc is not None:
            print(f"      Power est. (EMC)   : {power_est_tflops_emc:.4f} TFLOPs"
                  f"  ({tb_moved:.4f} TB moved)")
        print(f"      Roofline diagnostic: {session.roofline_tflops:.4f} TFLOPs"
              f"  ({ASSUMED_PRECISION})")
        print(f"      Peak util          : {session.peak_util:.1f}%")
        if avg_power_mw is not None:
            print(f"      Avg power          : {avg_power_mw:.0f} mW"
                  f"  |  Avg net: {avg_net_power_mw:.0f} mW"
                  f"  |  Peak: {session.peak_power:.0f} mW")
        print("      Returning to quiet loop.")

        quiet_power_samples.append(power_mw)

    last_poll_mono = time.monotonic()

    while True:
        now_mono = time.monotonic()
        dt_sec = max(now_mono - last_poll_mono, 0.0)
        last_poll_mono = now_mono

        power_mw, gpu_util, cur_freq_hz = read_gpu_sample(gpu_index)
        bytes_per_s_now = dram_reader.read_bytes_per_s()
        actmon_util = dram_reader.latest_util
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        is_active_sample = gpu_util > ACTIVE_GPU_UTIL_THRESHOLD

        if is_active_sample:
            active_poll_streak += 1
            quiet_poll_streak = 0
        else:
            active_poll_streak = 0
            if session is not None:
                quiet_poll_streak += 1
            else:
                quiet_power_samples.append(power_mw)

        if session is None and active_poll_streak >= START_ACTIVE_POLLS:
            quiet_poll_streak = 0
            session = start_session(timestamp, now_mono)

        if session is not None and is_active_sample:
            # Roofline diagnostic: compute ceiling only (auxiliary signal).
            peak_tflops = V100_PROFILE[f"TFLOPS_{ASSUMED_PRECISION}"]
            tflops_sm = peak_tflops * (cur_freq_hz / max_freq_hz) * (gpu_util / 100.0)
            estimated_tflops = tflops_sm

            roofline_tflops_delta = estimated_tflops * dt_sec
            session.roofline_tflops += roofline_tflops_delta

            # DRAM bytes moved this interval (DCGM); read_bytes_per_s() raises if
            # the stream is unreadable, so this is always valid.
            bytes_delta = bytes_per_s_now * dt_sec
            session.net_bytes += bytes_delta

            # power_mw is always valid (read_gpu_sample raises on failure).
            net_power_mw = max(power_mw - session.idle_baseline_mw, 0.0)
            net_power_w = net_power_mw / 1000.0
            net_energy_delta_j = net_power_w * dt_sec
            session.net_energy_j += net_energy_delta_j
            power_tflops_delta = _tflops_delta(
                net_energy_delta_j, dt_sec, 0.0,
                POWER_OVERHEAD_W, E_MARGINAL_J_PER_TFLOP, 0.0)
            power_emc_tflops_delta = _tflops_delta(
                net_energy_delta_j, dt_sec, bytes_delta / 1e12,
                POWER_OVERHEAD_EMC_W, E_MARGINAL_EMC_J_PER_TFLOP, E_PER_TB_J)

            session.power_sum += power_mw
            session.net_power_sum += net_power_mw
            session.peak_power = max(session.peak_power, power_mw)

            session.peak_util = max(session.peak_util, gpu_util)
            session.poll_count += 1
            session.active_time_s += dt_sec
            session.freq_sum_mhz += cur_freq_hz / 1e6

            power_est_so_far = estimate_tflops(
                session.net_energy_j, session.active_time_s
            )
            power_est_emc_so_far = estimate_tflops_emc(
                session.net_energy_j, session.active_time_s,
                session.net_bytes / 1e12
            )

            conn.execute(
                "INSERT INTO flop_log"
                " (timestamp, tflops_sm, tflops_mem_ceil, estimated_tflops,"
                "  bound_type, gpu_util, emc_util, power_mw, idle_baseline_mw,"
                "  net_power_mw, dt_sec, roofline_tflops_delta, power_tflops_delta,"
                "  bytes_delta, actmon_util, power_emc_tflops_delta)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (timestamp, tflops_sm, None, estimated_tflops,
                 "unknown", gpu_util, None, power_mw,
                 session.idle_baseline_mw, net_power_mw, dt_sec,
                 roofline_tflops_delta, power_tflops_delta,
                 bytes_delta, actmon_util, power_emc_tflops_delta)
            )
            conn.commit()

            print("--------------------------------------------------")
            print(f"SM Clock           : {cur_freq_hz/1e6:.1f} MHz  (max {max_freq_hz/1e6:.0f} MHz)")
            print(f"GPU Load           : {gpu_util}%")
            print(f"Poll interval      : {dt_sec:.2f}s observed")
            print(f"GPU power          : {power_mw:.0f} mW"
                  f"  |  net {net_power_mw:.0f} mW"
                  f" above idle {session.idle_baseline_mw:.0f} mW")
            print(f"SM busy estimate   : {tflops_sm:.3f} TFLOPS ({ASSUMED_PRECISION}, diagnostic)")
            print(f"Roofline diagnostic: {estimated_tflops:.3f} TFLOPS  [compute ceiling]")
            if power_est_so_far is not None:
                eff = (f"{session.net_energy_j / power_est_so_far:.2f} J/TFLOP effective"
                       if power_est_so_far > 0
                       else "<=0: sub-frontier, overhead exceeds net energy")
                print(f"Power estimate     : {power_est_so_far:.4f} TFLOPs  ({eff})")
            if power_est_emc_so_far is not None:
                print(f"Power est. (EMC)   : {power_est_emc_so_far:.4f} TFLOPs"
                      f"  ({session.net_bytes/1e12:.4f} TB moved"
                      f"{', DRAM ' + format(actmon_util*100, '.0f') + '%' if actmon_util is not None else ''})")
            print(f"Roofline diagnostic total: {session.roofline_tflops:.4f} TFLOPs")

        elif session is not None and quiet_poll_streak >= STOP_QUIET_POLLS:
            finish_session(session, timestamp, now_mono, power_mw)
            session = None
            active_poll_streak = 0
            quiet_poll_streak = 0

        elif session is None:
            observed_idle_mw = observed_idle_baseline_mw(quiet_power_samples)
            power_str = f" | power={power_mw:.0f}mW"
            print(f"[quiet] gpu={gpu_util:.1f}%"
                  f" | freq={cur_freq_hz/1e6:.0f}MHz"
                  f" | idle_base={FALLBACK_IDLE_POWER_MW:.0f}mW"
                  f" | observed_idle={observed_idle_mw:.0f}mW{power_str}"
                  f" | active_streak={active_poll_streak}/{START_ACTIVE_POLLS}")

        time.sleep(poll_interval)


if __name__ == "__main__":
    run_background_monitor()
