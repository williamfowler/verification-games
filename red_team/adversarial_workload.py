"""
adversarial_workload.py — Red-Team scheduling attacks (Phase II v3, dual-V100 DDP)
=================================================================================
One entry point; `--strategy` dispatches the workloads that need to interpose on
the *training loop schedule*. It imports the benign core (sample_ml_workload.
run_training) and interposes ONLY through that function's two no-op hooks
(step_callback / loop_ctx), so the compute path — and therefore the honestly-
printed `Ground truth total : X TFLOPs` — is identical to benign except for the
schedule.

Strategies here (v3):
  none      passthrough — provably GT-identical to sample_ml_workload (CI check).
  split     [live daemon] chunk the run into bursts separated by idle gaps long
            enough to trip the daemon's stop detector (3 quiet polls @1.5s >4.5s),
            so one frontier run is attributed as many sub-frontier sessions.
  throttle  [live daemon] micro-sleep between steps to lower AVERAGE GPU util
            (in-tenant; documented to FAIL the daemon's PEAK-util gate).

NOT here (v3): the other levers are one-variable knobs on the benign script or the
harness, not schedule interposition —
  * batch-inflation (S4), atypical nhead/optimizer (S3): benign script with a
    changed flag, driven from redteam_configs.py (parent-linked, GT-preserving).
  * power-cap / clock-lock (S5/S6): external nvidia-smi settings via
    power_control.py, code unchanged.
The v1 decoy strategies (mem_decoy / nvlink_decoy) are retired to red_team_old/.

DDP-only; launched under torchrun by run_workload / live_daemon_probe.
"""
import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: F401  (imported so a missing-torch launch fails clearly here)

from sample_ml_workload import run_training, require_torchrun


# ── Strategy: throttle (live daemon) ─────────────────────────────────────────
def make_throttle(target_util_pct, max_sleep=2.0):
    """step_callback holding GPU duty cycle near target_util by sleeping
    (1-d)/d × compute-time after each step. Open-loop on measured per-step compute
    time (EMA); all ranks run the same step count so they sleep in lockstep and
    DDP stays synced. NB this lowers AVERAGE util, not PEAK — the daemon gates on
    peak, so this is a documented FAILURE, kept for that negative result."""
    d = max(0.05, min(0.99, target_util_pct / 100.0))
    st = {"t_prev_end": None, "ema": None}

    def cb(step, device):
        now = time.time()
        if st["t_prev_end"] is not None:
            t_compute = now - st["t_prev_end"]
            st["ema"] = t_compute if st["ema"] is None else 0.7 * st["ema"] + 0.3 * t_compute
            t_sleep = min(max_sleep, st["ema"] * (1.0 - d) / d)
            if t_sleep > 0:
                time.sleep(t_sleep)
        st["t_prev_end"] = time.time()
    return cb


# ── Strategy: split (live daemon) ────────────────────────────────────────────
def make_split(chunk_steps, gap_seconds, total_steps):
    """step_callback idling the GPU for gap_seconds every chunk_steps steps (not
    after the final chunk). Gaps must exceed the daemon's stop window
    (STOP_QUIET_POLLS × poll ≈ 4.5 s) to segment the run into separate sessions.
    GT preserved — identical steps, only spaced apart. Ranks idle in lockstep."""
    def cb(step, device):
        n = step + 1
        if n % chunk_steps == 0 and n < total_steps:
            torch.cuda.synchronize()
            time.sleep(gap_seconds)
    return cb


def main():
    require_torchrun()
    p = argparse.ArgumentParser()
    # Benign core flags (mirror sample_ml_workload.py).
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--dim-feedforward", type=int, default=4096)
    p.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    # Schedule-attack dispatch + knobs.
    p.add_argument("--strategy", default="none", choices=["none", "split", "throttle"])
    p.add_argument("--chunk-steps", type=int, default=100)
    p.add_argument("--gap-seconds", type=float, default=6.0)
    p.add_argument("--target-util", type=float, default=70.0)
    a = p.parse_args()

    step_callback = None
    if a.strategy == "throttle":
        step_callback = make_throttle(a.target_util)
    elif a.strategy == "split":
        step_callback = make_split(a.chunk_steps, a.gap_seconds, a.steps)

    if int(os.environ.get("RANK", 0)) == 0 and a.strategy != "none":
        print(f"[redteam] STRATEGY = {a.strategy}  (chunk={a.chunk_steps} "
              f"gap={a.gap_seconds}s target_util={a.target_util}%)", flush=True)

    run_training(a.steps, a.batch_size, a.seq_len, a.d_model,
                 a.num_layers, a.nhead, a.dim_feedforward,
                 a.precision, a.optimizer, step_callback=step_callback)


if __name__ == "__main__":
    main()
