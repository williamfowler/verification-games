"""
adversarial_workload.py — Red-Team Adversarial Workloads (Phase II, dual-V100 DDP)
=================================================================================
One entry point; `--strategy` dispatches. It imports the benign training core
from sample_ml_workload.run_training and interposes ONLY through that function's
two no-op hooks (step_callback / loop_ctx), so the compute path — and therefore
the honestly-printed `Ground truth total : X TFLOPs` — is identical to benign
except where a strategy intends to differ.

Two hard invariants inherited from the design (see red_team/README.md):
  1. Ground truth stays honest. Every strategy still prints the true GT from
     FlopCounterMode; the attack lowers the blue team's *estimate*, never the GT.
  2. The attack lives INSIDE the sampling window — everything after the
     `[redteam] Starting workload...` print (decoy traffic, idle gaps, throttle
     sleeps) is what the monitor integrates.

Strategies:
  none      passthrough — provably identical to sample_ml_workload (CI check).
  mem_decoy [Group A] concurrent compute-free DRAM traffic (device-to-device
            copies on a side stream) to inflate the DRAM-bytes term the estimator
            subtracts; optional [V100] --decoy-allreduce inflates NVLink bytes.
  split     [Group B] chunk the run into bursts separated by idle gaps long enough
            to trip the live daemon's stop detector (3 quiet polls @1.5s > 4.5s),
            so one frontier run is attributed as many sub-frontier sessions.
  throttle  [Group B] micro-sleep between steps to hold GPU util below the 80%
            frontier gate (in-tenant; no clock-cap privilege needed).

DDP-only; launch under torchrun (run_workload / live_daemon_probe do this):
    torchrun --standalone --nproc_per_node=2 red_team/adversarial_workload.py \
        --strategy mem_decoy --decoy-gbps 200 --d-model 1024 --steps 200 ...
"""
import argparse
import os
import sys
import threading
import time
import contextlib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.distributed as dist

from sample_ml_workload import run_training, require_torchrun


# ── Strategy: throttle (Group B) ─────────────────────────────────────────────
def make_throttle(target_util_pct, max_sleep=2.0):
    """A step_callback that holds the GPU duty cycle near target_util by sleeping
    (1-d)/d × compute-time after each step. Open-loop on measured per-step compute
    time (EMA); all ranks run the same step count so they sleep in lockstep and
    DDP stays synced. target_util is a % (nvidia-smi utilization.gpu units)."""
    d = max(0.05, min(0.99, target_util_pct / 100.0))
    st = {"t_prev_end": None, "ema": None}

    def cb(step, device):
        now = time.time()
        if st["t_prev_end"] is not None:
            t_compute = now - st["t_prev_end"]          # this step's compute wall-time
            st["ema"] = t_compute if st["ema"] is None else 0.7 * st["ema"] + 0.3 * t_compute
            t_sleep = min(max_sleep, st["ema"] * (1.0 - d) / d)
            if t_sleep > 0:
                time.sleep(t_sleep)
        st["t_prev_end"] = time.time()
    return cb


# ── Strategy: split (Group B) ────────────────────────────────────────────────
def make_split(chunk_steps, gap_seconds, total_steps):
    """A step_callback that idles the GPU for gap_seconds every chunk_steps steps
    (not after the final chunk). The gap must exceed the daemon's stop window
    (STOP_QUIET_POLLS × poll_interval ≈ 4.5 s) to segment the run into separate
    sessions. Ranks idle in lockstep."""
    def cb(step, device):
        n = step + 1
        if n % chunk_steps == 0 and n < total_steps:
            torch.cuda.synchronize()
            time.sleep(gap_seconds)
    return cb


# ── Strategy: mem_decoy (Group A) ────────────────────────────────────────────
class DecoyTraffic:
    """loop_ctx: concurrent, compute-free memory traffic inside the sampling
    window. A background thread issues device-to-device copies on a side stream,
    paced to ~decoy_gbps of HBM bandwidth (each copy moves 2×buffer: read+write),
    inflating the DCGM DRAM-active signal (field 1005) → the tb_moved the 3-param
    estimator subtracts. Optional --decoy-allreduce adds extra NVLink all-reduce
    traffic; that runs on the MAIN thread via a step_callback (NCCL collectives
    must stay ordered across ranks — never from the background thread)."""
    def __init__(self, gbps, buf_mb):
        self.gbps = gbps
        self.buf_mb = buf_mb
        self._stop = threading.Event()
        self._thread = None
        self._device = None

    def _run(self):
        torch.cuda.set_device(self._device)
        stream = torch.cuda.Stream(self._device)
        n = max(1, int(self.buf_mb * 1024 * 1024 // 4))          # float32 elems
        src = torch.empty(n, device=self._device)
        dst = torch.empty(n, device=self._device)
        bytes_per_copy = 2 * n * 4                                # read + write
        target_s = bytes_per_copy / (self.gbps * 1e9) if self.gbps > 0 else 0.0
        with torch.cuda.stream(stream):
            while not self._stop.is_set():
                t0 = time.time()
                dst.copy_(src)
                stream.synchronize()
                dt = time.time() - t0
                if target_s > dt:                                # pace to target BW
                    time.sleep(target_s - dt)

    def __call__(self, device):                                  # loop_ctx(device)
        self._device = device
        return self

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        return False


def make_decoy_allreduce(mb):
    """step_callback (main thread): one extra all-reduce of an mb-sized dummy per
    step, inflating measured NVLink bytes without adding FLOPs. Ordered across
    ranks (every rank calls it each step)."""
    n = max(1, int(mb * 1024 * 1024 // 4))

    def cb(step, device):
        if dist.is_initialized() and dist.get_world_size() > 1:
            dummy = torch.ones(n, device=device)
            dist.all_reduce(dummy)
    return cb


def _compose(*callbacks):
    cbs = [c for c in callbacks if c is not None]
    if not cbs:
        return None
    if len(cbs) == 1:
        return cbs[0]

    def combined(step, device):
        for c in cbs:
            c(step, device)
    return combined


def main():
    require_torchrun()
    p = argparse.ArgumentParser()
    # Benign core flags (mirror sample_ml_workload.py) so _torchrun_cmd can inject
    # the shape args and the auto-sizer can rewrite --steps.
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--dim-feedforward", type=int, default=4096)
    p.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    # Strategy dispatch + knobs.
    p.add_argument("--strategy", default="none",
                   choices=["none", "mem_decoy", "split", "throttle"])
    p.add_argument("--decoy-gbps", type=float, default=200.0,
                   help="mem_decoy: target device-to-device HBM bandwidth (GB/s)")
    p.add_argument("--decoy-mb", type=float, default=256.0,
                   help="mem_decoy: decoy copy buffer size (MB)")
    p.add_argument("--decoy-allreduce", action="store_true",
                   help="mem_decoy: also inflate NVLink via extra all-reduce [V100]")
    p.add_argument("--decoy-allreduce-mb", type=float, default=64.0)
    p.add_argument("--chunk-steps", type=int, default=100,
                   help="split: steps per burst")
    p.add_argument("--gap-seconds", type=float, default=6.0,
                   help="split: idle gap between bursts (must exceed ~4.5s)")
    p.add_argument("--target-util", type=float, default=70.0,
                   help="throttle: hold GPU util near this %% (below the 80%% gate)")
    a = p.parse_args()

    step_callback = None
    loop_ctx = None

    if a.strategy == "throttle":
        step_callback = make_throttle(a.target_util)
    elif a.strategy == "split":
        step_callback = make_split(a.chunk_steps, a.gap_seconds, a.steps)
    elif a.strategy == "mem_decoy":
        if a.decoy_gbps > 0:                       # DRAM decoy (skip for NVLink-only)
            loop_ctx = DecoyTraffic(a.decoy_gbps, a.decoy_mb)
        if a.decoy_allreduce:                      # NVLink decoy [V100]
            step_callback = make_decoy_allreduce(a.decoy_allreduce_mb)

    rank = int(os.environ.get("RANK", 0))
    if rank == 0 and a.strategy != "none":
        print(f"[redteam] STRATEGY = {a.strategy}  "
              f"(decoy_gbps={a.decoy_gbps} decoy_mb={a.decoy_mb} "
              f"allreduce={a.decoy_allreduce} chunk={a.chunk_steps} "
              f"gap={a.gap_seconds}s target_util={a.target_util}%)", flush=True)

    run_training(a.steps, a.batch_size, a.seq_len, a.d_model,
                 a.num_layers, a.nhead, a.dim_feedforward,
                 a.precision, a.optimizer,
                 step_callback=step_callback, loop_ctx=loop_ctx)


if __name__ == "__main__":
    main()
