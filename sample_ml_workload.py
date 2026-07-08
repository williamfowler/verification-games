"""
sample_ml_workload.py — Red Team Workload Simulator
============================================
Simulates a transformer training workload on the Jetson GPU and emits a
ground truth FLOP count for comparison against the blue team monitor.

Usage:
    python3 sample_ml_workload.py
    python3 sample_ml_workload.py --steps 200 --batch-size 32 --d-model 512
    python3 sample_ml_workload.py --precision fp16 --optimizer sgd

Precision modes (matmul path; the FLOP *count* is precision-independent):
    fp32 (default) : the build's defaults — matmul TF32 OFF, cuDNN TF32 on.
                     This is what every calibration sweep before 2026-07-07 ran
                     (the old "TF32" assumption was wrong for matmul).
    tf32           : tensor-core TF32 matmul explicitly enabled.
    fp16 / bf16    : torch.autocast mixed precision (GradScaler for fp16).
"""

import torch
import torch.nn as nn
import time
import argparse
import contextlib

try:
    from torch.utils.flop_counter import FlopCounterMode
    HAS_FLOP_COUNTER = True
except ImportError:
    HAS_FLOP_COUNTER = False

AUTOCAST_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


class TinyTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3, dim_feedforward=512):
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
    """The forward-pass context for this precision (no-op for fp32/tf32)."""
    dtype = AUTOCAST_DTYPES.get(precision)
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=dtype)


def count_flops_per_step(model, batch_size, seq_len, d_model, device, precision):
    """
    FLOPs for one forward + backward pass via FlopCounterMode, under the same
    autocast context as training (the nominal count is precision-independent —
    FlopCounterMode counts op shapes). Cleans up gradients afterwards.
    Optimizer FLOPs are elementwise and <1% of total — not counted.
    Returns int, or None if FlopCounterMode is unavailable.
    """
    if not HAS_FLOP_COUNTER:
        return None
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    counter = FlopCounterMode(display=False)
    with counter:
        with autocast_ctx(precision):
            loss = model(x).mean()
        loss.backward()
    model.zero_grad()
    return counter.get_total_flops()


def make_optimizer(name, params):
    if name == "adamw":
        return torch.optim.AdamW(params, lr=1e-4)
    if name == "sgd":
        return torch.optim.SGD(params, lr=1e-4, momentum=0.9)
    raise ValueError(f"unknown optimizer {name!r}")


def run_training(steps, batch_size, seq_len, d_model,
                 num_layers=3, nhead=4, dim_feedforward=512,
                 precision="fp32", optimizer_name="adamw"):
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found.")

    if precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # fp32 = leave the build defaults untouched (matmul TF32 off) — this is the
    # historical calibration behavior; fp16/bf16 use autocast per step.

    device = torch.device("cuda")
    print(f"[redteam] Device : {torch.cuda.get_device_name(0)}")
    print(f"[redteam] Config : steps={steps}, batch={batch_size}, seq={seq_len},"
          f" d_model={d_model}, layers={num_layers}, nhead={nhead},"
          f" ff={dim_feedforward}, precision={precision}, opt={optimizer_name}")

    model     = TinyTransformer(d_model=d_model, nhead=nhead,
                                num_layers=num_layers,
                                dim_feedforward=dim_feedforward).to(device)
    optimizer = make_optimizer(optimizer_name, model.parameters())
    scaler    = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
    model.train()

    # Warmup — bring CUDA fully online before timing starts
    with autocast_ctx(precision):
        model(torch.randn(batch_size, seq_len, d_model, device=device))
    torch.cuda.synchronize()

    flops_per_step = count_flops_per_step(model, batch_size, seq_len, d_model,
                                          device, precision)
    if flops_per_step:
        print(f"[redteam] FLOPs/step (fwd+bwd) : {flops_per_step/1e9:.3f} GFLOPs")
        print(f"[redteam] Ground truth total    : {flops_per_step * steps / 1e12:.4f} TFLOPs")
    else:
        print("[redteam] FlopCounterMode unavailable — no ground truth FLOP count.")

    print("[redteam] Starting workload...\n")
    t_start          = time.time()
    cumulative_flops = 0

    for step in range(steps):
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        with autocast_ctx(precision):
            loss = model(x).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        cumulative_flops += flops_per_step or 0

        if step % 10 == 0:
            elapsed = time.time() - t_start
            flop_str = f"  |  {cumulative_flops/1e12:.4f} TFLOPs" if flops_per_step else ""
            print(f"  step {step:>4d}/{steps}"
                  f"  |  {(step+1)/elapsed:.2f} steps/s"
                  f"  |  {elapsed:.1f}s elapsed"
                  f"{flop_str}")

    total_time = time.time() - t_start
    print(f"\n[redteam] Done. {steps} steps in {total_time:.1f}s ({steps/total_time:.2f} steps/s)")
    if flops_per_step:
        total_tf = cumulative_flops / 1e12
        print(f"[redteam] Ground truth total : {total_tf:.4f} TFLOPs")
        print(f"[redteam] Ground truth rate  : {total_tf/total_time:.4f} TFLOPS avg")
        print(f"[redteam] Compare 'Monitor est.' above to ground truth total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",           type=int, default=150)
    parser.add_argument("--batch-size",      type=int, default=8)
    parser.add_argument("--seq-len",         type=int, default=64)
    parser.add_argument("--d-model",         type=int, default=128)
    parser.add_argument("--num-layers",      type=int, default=3)
    parser.add_argument("--nhead",           type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--precision",       default="fp32",
                        choices=["fp32", "tf32", "fp16", "bf16"])
    parser.add_argument("--optimizer",       default="adamw",
                        choices=["adamw", "sgd"])
    args = parser.parse_args()
    run_training(args.steps, args.batch_size, args.seq_len, args.d_model,
                 args.num_layers, args.nhead, args.dim_feedforward,
                 args.precision, args.optimizer)
