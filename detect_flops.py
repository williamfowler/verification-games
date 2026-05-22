import time
import re
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

ASSUMED_PRECISION = "TF32"

# Roofline ridge point: FLOPs/byte where compute and memory-bandwidth bounds
# intersect. Workloads below this AI are memory-bound; above, compute-bound.
RIDGE_AI = (ORIN_PROFILE["TFLOPS_TF32"] * 1e12) / ORIN_PROFILE["PEAK_BW_BYTES_S"]

DB_PATH = "/var/log/flop_log.db"


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
            emc_util         REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS workload_sessions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time    TEXT NOT NULL,
            end_time      TEXT NOT NULL,
            duration_sec  REAL,
            total_tflops  REAL,
            peak_gpu_util REAL,
            poll_count    INTEGER
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
    keys_dumped                = False
    session_start_time         = None
    session_accumulated_tflops = 0.0
    session_peak_util          = 0.0
    session_poll_count         = 0

    conn = init_db(DB_PATH)
    print("Daemon initialized. Monitoring Orin Nano hardware channels for new ML workloads...")

    with jtop() as jetson:
        while jetson.ok():
            stats = jetson.stats

            if not keys_dumped:
                print("[DEBUG] jetson.stats keys:", list(stats.keys()))
                try:
                    print("[DEBUG] jetson.gpu:", jetson.gpu)
                except Exception as e:
                    print(f"[DEBUG] jetson.gpu unavailable: {e}")
                print(f"[DEBUG] tegrastats EMC test: {get_emc_util()}%")
                keys_dumped = True

            gpu_util                 = stats.get('GPU', 0)
            emc_util                 = get_emc_util()
            cur_freq_hz, max_freq_hz = get_freq_hz(jetson)

            print(
                f"[debug] gpu_util={gpu_util} | emc_util={emc_util} | "
                f"gpu_freq={cur_freq_hz/1e6:.1f} MHz (max {max_freq_hz/1e6:.0f} MHz)"
            )

            if gpu_util > 5.0:
                timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

                if not active_workload_tracked:
                    print("\n[!] NEW WORKLOAD DETECTED")
                    active_workload_tracked    = True
                    session_start_time         = timestamp
                    session_accumulated_tflops = 0.0
                    session_peak_util          = 0.0
                    session_poll_count         = 0

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

                conn.execute(
                    "INSERT INTO flop_log"
                    " (timestamp, tflops_sm, tflops_mem_ceil, estimated_tflops,"
                    "  bound_type, gpu_util, emc_util)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (timestamp, tflops_sm, tflops_mem_ceil, estimated_tflops,
                     bound_type, gpu_util, emc_util)
                )
                conn.commit()

                print("--------------------------------------------------")
                print(f"Hardware Clock     : {cur_freq_hz/1e6:.1f} MHz  (max {max_freq_hz/1e6:.0f} MHz)")
                print(f"GPU / EMC Load     : {gpu_util}% / {emc_util}%")
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

                    conn.execute(
                        "INSERT INTO workload_sessions"
                        " (start_time, end_time, duration_sec, total_tflops,"
                        "  peak_gpu_util, poll_count)"
                        " VALUES (?, ?, ?, ?, ?, ?)",
                        (session_start_time, end_time, duration_sec,
                         session_accumulated_tflops, session_peak_util, session_poll_count)
                    )
                    conn.commit()

                    print("\n[---] Workload completed or detached.")
                    print(f"      Duration     : {duration_sec:.1f}s  ({session_poll_count} polls × {poll_interval}s)")
                    print(f"      Monitor est. : {session_accumulated_tflops:.4f} TFLOPs  ({ASSUMED_PRECISION})")
                    print(f"      Peak util    : {session_peak_util:.1f}%")
                    print("      Returning to quiet loop.")
                    active_workload_tracked = False

            time.sleep(poll_interval)


run_background_monitor()
