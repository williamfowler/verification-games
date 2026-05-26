"""
sample_ml_workload.py — Red Team Workload Simulator
============================================
Simulates a transformer training workload on the Jetson GPU and emits a
ground truth FLOP count for comparison against the blue team monitor.

Usage:
    python3 sample_ml_workload.py
    python3 sample_ml_workload.py --steps 200 --batch-size 32 --d-model 512
"""

import torch
import torch.nn as nn
import time
import argparse

try:
    from torch.utils.flop_counter import FlopCounterMode
    HAS_FLOP_COUNTER = True
except ImportError:
    HAS_FLOP_COUNTER = False


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


def count_flops_per_step(model, batch_size, seq_len, d_model, device):
    """
    FLOPs for one forward + backward pass via FlopCounterMode.
    Runs on the live model and cleans up gradients afterwards.
    Optimizer FLOPs are elementwise and <1% of total — not counted.
    Returns int, or None if FlopCounterMode is unavailable.
    """
    if not HAS_FLOP_COUNTER:
        return None
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    counter = FlopCounterMode(display=False)
    with counter:
        model(x).mean().backward()
    model.zero_grad()
    return counter.get_total_flops()


def run_training(steps, batch_size, seq_len, d_model):
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found.")

    device = torch.device("cuda")
    print(f"[redteam] Device : {torch.cuda.get_device_name(0)}")
    print(f"[redteam] Config : steps={steps}, batch={batch_size}, seq={seq_len}, d_model={d_model}")

    model     = TinyTransformer(d_model=d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    # Warmup — bring CUDA fully online before timing starts
    model(torch.randn(batch_size, seq_len, d_model, device=device))
    torch.cuda.synchronize()

    flops_per_step = count_flops_per_step(model, batch_size, seq_len, d_model, device)
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
        model(x).mean().backward()
        optimizer.step()
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
    parser.add_argument("--steps",      type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len",    type=int, default=64)
    parser.add_argument("--d-model",    type=int, default=128)
    args = parser.parse_args()
    run_training(args.steps, args.batch_size, args.seq_len, args.d_model)
