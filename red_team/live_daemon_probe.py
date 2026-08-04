"""
live_daemon_probe.py — Group B harness: drive the LIVE detect_flops.py daemon.

Splitting (S4) and throttling (S5) attack the daemon's *session logic* — start/stop
detection and the 80% frontier gate — which the offline eval bypasses (it brackets
the whole subprocess on the stdout trigger). So they must run against the running
daemon and be scored on the session(s) the daemon actually records.

Flow:
  1. fresh temp DB (FLOP_DB_PATH), launch detect_flops.py, wait until it's polling;
  2. run the adversarial (or benign-baseline) workload via torchrun on both GPUs —
     the daemon watches GPU_INDEX and self-calibrates its idle baseline;
  3. after the workload exits, wait out the stop window (STOP_QUIET_POLLS × poll)
     so the daemon closes and writes its final session;
  4. stop the daemon, read workload_sessions, emit a record in a schema
     score_redteam.py consumes uniformly with the offline records.

The daemon is single-GPU (the both-GPU wiring is a documented follow-on), so
absolute estimates are ~half the DDP truth. Group B is therefore scored RELATIVE
to a `--strategy none` baseline run through the SAME daemon — the calibration
offset cancels; what remains is "did the attack fragment the run into sub-frontier
sessions / hold it under the gate so the blue team can't attribute it?"

    torchrun is invoked internally. Run directly (NOT under torchrun):
    python3 red_team/live_daemon_probe.py --strategy split --chunk-steps 100 \
        --gap-seconds 6 --steps 400 --d-model 1024 --out rec.json
"""
import argparse
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "power_calibration"))

VENV_PY = os.path.join(REPO_ROOT, ".venv", "bin", "python")
VENV_TORCHRUN = os.path.join(REPO_ROOT, ".venv", "bin", "torchrun")
ADV = os.path.join(REPO_ROOT, "red_team", "adversarial_workload.py")
DAEMON = os.path.join(REPO_ROOT, "detect_flops.py")

FRONTIER_UTIL = 80.0          # the daemon's trusted-estimate gate
POLL_INTERVAL = 1.5           # detect_flops run_background_monitor default
STOP_QUIET_POLLS = 3          # daemon closes a session after this many quiet polls


def wait_for_daemon_ready(proc, timeout=40):
    """Block until the daemon prints its readiness line (DCGM warmup done)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            raise RuntimeError("daemon exited during startup:\n" + (line or ""))
        if line:
            print("  [daemon] " + line.rstrip())
            if "power probe OK" in line or "Monitoring" in line:
                return
    raise RuntimeError("daemon did not become ready in time")


def drain_daemon(proc):
    """Non-blocking-ish: print whatever the daemon has emitted (best effort)."""
    import select
    while True:
        r, _, _ = select.select([proc.stdout], [], [], 0)
        if not r:
            break
        line = proc.stdout.readline()
        if not line:
            break
        print("  [daemon] " + line.rstrip())


def run_workload(strategy, args):
    """Launch the adversarial workload under torchrun on both GPUs; stream its
    stdout (so the daemon sees the load in real time); return (returncode, gt_tf)."""
    cmd = [VENV_TORCHRUN, "--standalone", "--nproc_per_node=2", ADV,
           "--strategy", strategy,
           "--steps", str(args.steps), "--batch-size", str(args.batch_size),
           "--seq-len", str(args.seq_len), "--d-model", str(args.d_model),
           "--num-layers", str(args.num_layers), "--nhead", str(args.nhead),
           "--dim-feedforward", str(args.dim_feedforward),
           "--precision", args.precision]
    if strategy == "split":
        cmd += ["--chunk-steps", str(args.chunk_steps),
                "--gap-seconds", str(args.gap_seconds)]
    elif strategy == "throttle":
        cmd += ["--target-util", str(args.target_util)]
    print(f"  workload: {' '.join(cmd[4:])}", flush=True)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUNBUFFERED="1")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env)
    gt = None
    for line in iter(proc.stdout.readline, ""):
        print(line, end="")
        m = re.search(r"Ground truth total\s*:\s*([\d.]+)\s*TFLOPs", line)
        if m:
            gt = float(m.group(1))
    proc.wait()
    return proc.returncode, gt


def read_sessions(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT start_time, end_time, duration_sec, power_est_tflops,"
        " power_est_tflops_emc, peak_gpu_util, poll_count, net_energy_j"
        " FROM workload_sessions ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="split",
                   choices=["none", "split", "throttle"])
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--dim-feedforward", type=int, default=4096)
    p.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--chunk-steps", type=int, default=100)
    p.add_argument("--gap-seconds", type=float, default=6.0)
    p.add_argument("--target-util", type=float, default=70.0)
    p.add_argument("--label", default=None)
    p.add_argument("--out", default=None, help="write the record JSON here")
    args = p.parse_args()

    label = args.label or f"live_{args.strategy}"
    db_path = os.path.join(
        os.environ.get("TMPDIR", "/tmp"),
        f"flop_log_probe_{args.strategy}_{os.getpid()}.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    print(f"=== live_daemon_probe: strategy={args.strategy}  db={db_path} ===")
    env = dict(os.environ, FLOP_DB_PATH=db_path, PYTHONUNBUFFERED="1")
    daemon = subprocess.Popen([VENV_PY, "-u", DAEMON], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, env=env)
    try:
        wait_for_daemon_ready(daemon)
        time.sleep(2.0)                      # let it settle into the poll loop
        rc, gt = run_workload(args.strategy, args)
        # Wait out the stop window so the daemon closes its final session.
        quiet_wait = (STOP_QUIET_POLLS + 2) * POLL_INTERVAL + 3.0
        print(f"  waiting {quiet_wait:.0f}s for the daemon to close sessions...")
        time.sleep(quiet_wait)
        drain_daemon(daemon)
    finally:
        daemon.send_signal(signal.SIGINT)    # let it flush/close cleanly
        try:
            daemon.wait(timeout=10)
        except subprocess.TimeoutExpired:
            daemon.kill()

    sessions = read_sessions(db_path)
    frontier = [s for s in sessions
                if (s["peak_gpu_util"] or 0) >= FRONTIER_UTIL]
    attributable = sum((s["power_est_tflops"] or 0) for s in frontier)
    attributable_all = sum((s["power_est_tflops"] or 0) for s in sessions)
    max_session = max((s["power_est_tflops"] or 0 for s in sessions), default=0.0)

    record = {
        "label": label, "strategy": args.strategy, "group": "B",
        "returncode": rc, "ground_truth_tf": gt,
        "n_sessions": len(sessions), "n_frontier_sessions": len(frontier),
        "attributable_frontier_tf": attributable,
        "attributable_all_tf": attributable_all,
        "max_session_tf": max_session,
        "sessions": sessions,
        "config": {"strategy": args.strategy, "chunk_steps": args.chunk_steps,
                   "gap_seconds": args.gap_seconds, "target_util": args.target_util,
                   "steps": args.steps, "d_model": args.d_model},
    }

    print("\n=== RESULT ===")
    print(f"  honest GT (2-GPU aggregate) : {gt}")
    print(f"  daemon sessions detected    : {len(sessions)} "
          f"({len(frontier)} cleared the {FRONTIER_UTIL:.0f}% gate)")
    for i, s in enumerate(sessions):
        print(f"    session {i}: dur {s['duration_sec']:.0f}s  "
              f"peak_util {s['peak_gpu_util']:.0f}%  "
              f"est {s['power_est_tflops']:.0f} TF"
              f"  {'[frontier]' if (s['peak_gpu_util'] or 0) >= FRONTIER_UTIL else '[sub-gate]'}")
    print(f"  attributable (frontier only): {attributable:.0f} TF")
    print(f"  largest single session      : {max_session:.0f} TF")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(record, f, indent=1)
        print(f"\nwrote {args.out}")
    try:
        os.remove(db_path)
    except OSError:
        pass
    return record


if __name__ == "__main__":
    main()
