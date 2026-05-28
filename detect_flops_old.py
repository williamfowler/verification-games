import time
import re
import glob
import sqlite3
import subprocess
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

ASSUMED_PRECISION = "FP32"

# Roofline ridge point: FLOPs/byte where compute and memory-bandwidth bounds
# intersect. Workloads below this AI are memory-bound; above, compute-bound.
RIDGE_AI = (ORIN_PROFILE["TFLOPS_TF32"] * 1e12) / ORIN_PROFILE["PEAK_BW_BYTES_S"]

DB_PATH = "/var/log/flop_log.db"

# INA3221 power sensor — VDD_CPU_GPU_CV rail (CPU + GPU + CV cores).
# hwmonN index can change across reboots; resolved once at startup by label.
INA3221_DRIVER = "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon"
INA3221_LABEL  = "VDD_CPU_GPU_CV"

# Power-based FLOP calibration (empirical, Orin Nano, TinyTransformer workloads).
# Derived from four ground-truth runs at d=128/256/512/1024.
# VDD_CPU_GPU_CV idle baseline to subtract before computing net compute power.
IDLE_POWER_MW = 620.0
# Two-bucket J/TFLOP constants split on avg net power:
#   Low-power  (net < 2.5 W): small/memory-bound models — 8.1 J/TFLOP
#   High-power (net >= 2.5 W): larger/compute-bound models — 5.9 J/TFLOP
# Threshold chosen at the natural gap between d=256 (1.0 W net) and d=512 (4.2 W net).
POWER_CAL_LOW_J_PER_TFLOP  = 8.1   # d=128,256 average
POWER_CAL_HIGH_J_PER_TFLOP = 5.9   # d=512,1024 average
POWER_CAL_NET_W_THRESHOLD  = 2.5   # W net above idle


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
            power_mw         REAL
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
            peak_power_mw   REAL
        )
    """)
    conn.commit()
    return conn


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
        m = re.search(r'EMC_FREQ\s+(\d+)%', line)
        return float(m.group(1)) if m else None
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
    active_workload_tracked    = False
    session_start_time         = None
    session_accumulated_tflops = 0.0
    session_peak_util          = 0.0
    session_poll_count         = 0
    session_power_sum          = 0.0
    session_peak_power         = 0.0

    conn = init_db(DB_PATH)

    volt_path, curr_path = find_ina3221_paths()
    if volt_path:
        print(f"INA3221 power sensor: {INA3221_LABEL} found.")
    else:
        print(f"INA3221 power sensor: {INA3221_LABEL} not found — power column will be NULL.")

    print("Daemon initialized. Monitoring Orin Nano hardware channels for new ML workloads...")

    with jtop() as jetson:
        while jetson.ok():
            stats = jetson.stats

            gpu_util                 = stats.get('GPU', 0)
            emc_util                 = get_emc_util()
            cur_freq_hz, max_freq_hz = get_freq_hz(jetson)
            power_mw                 = read_power_mw(volt_path, curr_path) if volt_path else None

            if gpu_util > 5.0:
                timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

                if not active_workload_tracked:
                    print("\n[!] NEW WORKLOAD DETECTED")
                    active_workload_tracked    = True
                    session_start_time         = timestamp
                    session_accumulated_tflops = 0.0
                    session_peak_util          = 0.0
                    session_poll_count         = 0
                    session_power_sum          = 0.0
                    session_peak_power         = 0.0

                peak_tflops = ORIN_PROFILE[f"TFLOPS_{ASSUMED_PRECISION}"]
                tflops_sm   = peak_tflops * (cur_freq_hz / max_freq_hz) * (gpu_util / 100.0)

                if emc_util is not None and emc_util > 0:
                    emc_bw_actual    = (emc_util / 100.0) * ORIN_PROFILE["PEAK_BW_BYTES_S"]
                    tflops_mem_ceil  = (emc_bw_actual * RIDGE_AI) / 1e12
                    estimated_tflops = min(tflops_sm, tflops_mem_ceil)
                    bound_type       = "compute" if tflops_sm <= tflops_mem_ceil else "memory"
                else:
                    emc_bw_actual    = None
                    tflops_mem_ceil  = None
                    estimated_tflops = tflops_sm
                    bound_type       = "unknown"

                session_accumulated_tflops += estimated_tflops * poll_interval
                session_peak_util           = max(session_peak_util, gpu_util)
                session_poll_count         += 1
                if power_mw is not None:
                    session_power_sum  += power_mw
                    session_peak_power  = max(session_peak_power, power_mw)

                conn.execute(
                    "INSERT INTO flop_log"
                    " (timestamp, tflops_sm, tflops_mem_ceil, estimated_tflops,"
                    "  bound_type, gpu_util, emc_util, power_mw)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (timestamp, tflops_sm, tflops_mem_ceil, estimated_tflops,
                     bound_type, gpu_util, emc_util, power_mw)
                )
                conn.commit()

                print("--------------------------------------------------")
                print(f"Hardware Clock     : {cur_freq_hz/1e6:.1f} MHz  (max {max_freq_hz/1e6:.0f} MHz)")
                print(f"GPU / EMC Load     : {gpu_util}% / {emc_util}%")
                if power_mw is not None:
                    print(f"VDD_CPU_GPU_CV     : {power_mw:.0f} mW  ({power_mw/1000:.3f} W)")
                print(f"SM estimate        : {tflops_sm:.3f} TFLOPS ({ASSUMED_PRECISION})")
                if tflops_mem_ceil is not None:
                    print(f"Memory BW ceiling  : {tflops_mem_ceil:.3f} TFLOPS"
                          f"  ({emc_bw_actual/1e9:.1f} GB/s × {RIDGE_AI:.0f} FLOPs/byte ridge)")
                    print(f"Roofline estimate  : {estimated_tflops:.3f} TFLOPS  [{bound_type}-bound]")
                else:
                    print(f"Roofline estimate  : {estimated_tflops:.3f} TFLOPS  [EMC unavailable]")
                print(f"Session Total      : {session_accumulated_tflops:.4f} TFLOPs consumed")

            else:
                if active_workload_tracked:
                    end_time     = time.strftime("%Y-%m-%dT%H:%M:%S")
                    duration_sec = session_poll_count * poll_interval
                    avg_power_mw = (session_power_sum / session_poll_count
                                    if session_poll_count > 0 else None)

                    # Power-based FLOP estimate using empirical J/TFLOP calibration.
                    power_est_tflops = None
                    if avg_power_mw is not None:
                        net_w = (avg_power_mw - IDLE_POWER_MW) / 1000.0
                        if net_w > 0:
                            j_per_tflop  = (POWER_CAL_HIGH_J_PER_TFLOP
                                            if net_w >= POWER_CAL_NET_W_THRESHOLD
                                            else POWER_CAL_LOW_J_PER_TFLOP)
                            power_est_tflops = (net_w * duration_sec) / j_per_tflop

                    conn.execute(
                        "INSERT INTO workload_sessions"
                        " (start_time, end_time, duration_sec, total_tflops, power_est_tflops,"
                        "  peak_gpu_util, poll_count, avg_power_mw, peak_power_mw)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (session_start_time, end_time, duration_sec,
                         session_accumulated_tflops, power_est_tflops,
                         session_peak_util, session_poll_count,
                         avg_power_mw, session_peak_power if session_peak_power > 0 else None)
                    )
                    conn.commit()

                    print("\n[---] Workload completed or detached.")
                    print(f"      Duration     : {duration_sec:.1f}s  ({session_poll_count} polls × {poll_interval}s)")
                    print(f"      Roofline est.: {session_accumulated_tflops:.4f} TFLOPs  ({ASSUMED_PRECISION})")
                    if power_est_tflops is not None:
                        bucket = "high-pwr" if (avg_power_mw - IDLE_POWER_MW)/1000 >= POWER_CAL_NET_W_THRESHOLD else "low-pwr"
                        print(f"      Power est.   : {power_est_tflops:.4f} TFLOPs  [{bucket} bucket]")
                    print(f"      Peak util    : {session_peak_util:.1f}%")
                    if avg_power_mw is not None:
                        print(f"      Avg power    : {avg_power_mw:.0f} mW  |  Peak: {session_peak_power:.0f} mW")
                    print("      Returning to quiet loop.")
                    active_workload_tracked = False
                else:
                    power_str = f" | power={power_mw:.0f}mW" if power_mw is not None else ""
                    print(f"[quiet] gpu={gpu_util:.1f}% | emc={emc_util}%"
                          f" | freq={cur_freq_hz/1e6:.0f}MHz{power_str}")

            time.sleep(poll_interval)


run_background_monitor()
