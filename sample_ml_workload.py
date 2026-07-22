"""
sample_ml_workload.py — Red Team Workload Simulator (DDP over 2× V100)
=====================================================================
Simulates a *real* V100 training run — DistributedDataParallel over both GPUs,
FP16 AMP with GradScaler (the standard V100 tensor-core training precision) —
and emits an aggregate ground-truth FLOP count for the blue-team monitor to
score against.

DDP-only. Launch under torchrun (never plain python):
    torchrun --standalone --nproc_per_node=2 sample_ml_workload.py --d-model 1024 ...
    torchrun --standalone --nproc_per_node=1 sample_ml_workload.py ...   # debug path

Volta (sm_70) constraints (see ~/ddp_test.py, the reference milestone run):
  - FP16 autocast + GradScaler (NOT bf16 — Volta has no BF16; NOT tf32 — Ampere+)
  - SDPA math / mem-efficient paths only (no flash-attn on sm_70)

Precision (matmul path; the FLOP *count* is precision-independent — FlopCounterMode
counts op shapes):
    fp16 (default, baseline) : torch.autocast(float16) + GradScaler — the regime.
    fp32                     : full precision (contrast family G, ~8× slower matmul).
"""

import os
import time
import argparse
import contextlib

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.utils.flop_counter import FlopCounterMode
    HAS_FLOP_COUNTER = True
except ImportError:
    HAS_FLOP_COUNTER = False

# Volta-safe numerics: TF32 is an Ampere+ feature (silent no-op here) — keep it
# explicitly off so nothing accidentally depends on it.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

AUTOCAST_DTYPES = {"fp16": torch.float16}   # bf16/tf32 removed (unsupported on Volta)


class TinyTransformer(nn.Module):
    """Transformer encoder stack. 'Tiny' is historical — the V100 sweep drives
    d_model up to 2048 / 32 layers. Uses nn.TransformerEncoderLayer, whose
    attention runs through SDPA (math/mem-efficient on sm_70)."""
    def __init__(self, d_model=1024, nhead=8, num_layers=6, dim_feedforward=4096):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=0.1, batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.head(self.encoder(x))


def autocast_ctx(precision):
    """The forward-pass context for this precision (no-op for fp32)."""
    dtype = AUTOCAST_DTYPES.get(precision)
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=dtype)


def count_flops_per_step(model, batch_size, seq_len, d_model, device, precision):
    """
    FLOPs for one forward + backward pass via FlopCounterMode, under the same
    autocast context as training. Counts on the LOCAL (per-GPU) batch — every
    rank runs identical shapes, so the run-total aggregate is this × steps ×
    world_size (see run_training). Optimizer FLOPs are elementwise and <1% —
    not counted. Returns per-rank int, or None if FlopCounterMode is unavailable.
    """
    if not HAS_FLOP_COUNTER:
        return None
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    counter = FlopCounterMode(display=False)
    with counter:
        with autocast_ctx(precision):
            loss = model(x).mean()
        loss.backward()
    model.zero_grad(set_to_none=True)
    return counter.get_total_flops()


def make_optimizer(name, params):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=1e-4)
    if name == "sgd":
        return torch.optim.SGD(params, lr=1e-4, momentum=0.9)
    raise ValueError(f"unknown optimizer {name!r}")


def require_torchrun():
    """This branch is DDP-only. Fail clearly (not with a KeyError) if the script
    was launched with plain `python` instead of torchrun."""
    missing = [k for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if k not in os.environ]
    if missing:
        raise SystemExit(
            "sample_ml_workload.py is DDP-only and must be launched via torchrun, "
            "e.g.\n    torchrun --standalone --nproc_per_node=2 "
            "sample_ml_workload.py [flags]\n"
            f"(missing env: {', '.join(missing)})")


def run_training(steps, batch_size, seq_len, d_model,
                 num_layers=6, nhead=8, dim_feedforward=4096,
                 precision="fp16", optimizer_name="adamw"):
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found.")

    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # set_device BEFORE init_process_group (Volta-safe DDP ordering, per ~/ddp_test.py)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")

    def log(*a, **k):
        if rank == 0:
            print(*a, **k)

    try:
        # Per-rank seed so synthetic data differs across ranks and the gradient
        # all-reduce carries non-degenerate NVLink traffic (see ddp_test.py).
        torch.manual_seed(1234 + rank)

        log(f"[redteam] Device : {torch.cuda.get_device_name(device)} × {world_size} (DDP)")
        log(f"[redteam] Config : steps={steps}, batch/gpu={batch_size},"
            f" global_batch={batch_size * world_size}, seq={seq_len},"
            f" d_model={d_model}, layers={num_layers}, nhead={nhead},"
            f" ff={dim_feedforward}, precision={precision}, opt={optimizer_name}")

        model = TinyTransformer(d_model=d_model, nhead=nhead,
                                num_layers=num_layers,
                                dim_feedforward=dim_feedforward).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        # Param count → predicted grad all-reduce bytes/step (n_params × 4 B, FP32
        # even under AMP); the blue-team NVLink monitor cross-checks against this.
        log(f"[redteam] Params : {n_params} ({n_params/1e6:.1f}M)")

        # Count FLOPs on the raw (unwrapped) model, before the DDP wrap, so the
        # all-reduce hooks don't perturb FlopCounterMode's backward.
        flops_per_step = count_flops_per_step(model, batch_size, seq_len, d_model,
                                              device, precision)

        model     = DDP(model, device_ids=[local_rank])
        optimizer = make_optimizer(optimizer_name, model.parameters())
        scaler    = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
        model.train()

        # Warmup — bring CUDA/NCCL fully online before timing starts.
        with autocast_ctx(precision):
            model(torch.randn(batch_size, seq_len, d_model, device=device)).mean().backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        # Aggregate ground truth = per-rank FLOPs × steps × world_size. All ranks
        # run identical shapes, so the ×world_size is exact, not an approximation.
        if flops_per_step:
            total_tf = flops_per_step * steps * world_size / 1e12
            log(f"[redteam] FLOPs/step/gpu (fwd+bwd) : {flops_per_step/1e9:.3f} GFLOPs")
            log(f"[redteam] Ground truth total    : {total_tf:.4f} TFLOPs"
                f"  ({world_size} ranks)")
        else:
            log("[redteam] FlopCounterMode unavailable — no ground truth FLOP count.")

        log("[redteam] Starting workload...\n")
        t_start          = time.time()
        cumulative_flops = 0

        for step in range(steps):
            x = torch.randn(batch_size, seq_len, d_model, device=device)
            with autocast_ctx(precision):
                loss = model(x).mean()
            scaler.scale(loss).backward()   # DDP all-reduces grads here, over NVLink
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            cumulative_flops += (flops_per_step or 0) * world_size

            if rank == 0 and step % 10 == 0:
                elapsed = time.time() - t_start
                flop_str = f"  |  {cumulative_flops/1e12:.4f} TFLOPs" if flops_per_step else ""
                log(f"  step {step:>4d}/{steps}"
                    f"  |  {(step+1)/elapsed:.2f} steps/s"
                    f"  |  {elapsed:.1f}s elapsed"
                    f"{flop_str}")

        torch.cuda.synchronize()
        total_time = time.time() - t_start
        log(f"\n[redteam] Done. {steps} steps in {total_time:.1f}s ({steps/total_time:.2f} steps/s)")
        if flops_per_step:
            total_tf = cumulative_flops / 1e12
            log(f"[redteam] Ground truth total : {total_tf:.4f} TFLOPs")
            log(f"[redteam] Ground truth rate  : {total_tf/total_time:.4f} TFLOPS avg")
            log(f"[redteam] Compare 'Monitor est.' above to ground truth total.")

        # Prove the collective path end-to-end (as in the reference milestone run).
        probe = torch.ones(1, device=device) * (rank + 1)
        dist.all_reduce(probe)
        expected = world_size * (world_size + 1) / 2
        if rank == 0:
            status = "OK" if probe.item() == expected else f"MISMATCH ({probe.item()})"
            log(f"[redteam] all-reduce probe: {status}")
    finally:
        # destroy in finally so a crashed sweep run never leaves a zombie rank.
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    require_torchrun()
    parser = argparse.ArgumentParser()
    # KEEP IN SYNC: eval_power_monitor.py's CONFIG_DEFAULTS mirrors the
    # nhead/dim_feedforward/precision/optimizer defaults below.
    parser.add_argument("--steps",           type=int, default=150)
    parser.add_argument("--batch-size",      type=int, default=16,
                        help="PER-GPU batch size (global = batch_size × world_size)")
    parser.add_argument("--seq-len",         type=int, default=512)
    parser.add_argument("--d-model",         type=int, default=1024)
    parser.add_argument("--num-layers",      type=int, default=12)
    parser.add_argument("--nhead",           type=int, default=8)
    parser.add_argument("--dim-feedforward", type=int, default=4096)
    parser.add_argument("--precision",       default="fp16",
                        choices=["fp16", "fp32"])
    parser.add_argument("--optimizer",       default="adamw",
                        choices=["adamw", "sgd"])
    args = parser.parse_args()
    run_training(args.steps, args.batch_size, args.seq_len, args.d_model,
                 args.num_layers, args.nhead, args.dim_feedforward,
                 args.precision, args.optimizer)
