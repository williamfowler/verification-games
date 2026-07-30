# V100 DDP phase — NVLink monitoring, DDP conversion, and the fp16 sweep

Reproduction of the FLOP-estimation experiment redesigned for what the hardware
is for: **DistributedDataParallel training over 2× Tesla V100-SXM2-16GB (NVLink
NV6), FP16 AMP with GradScaler**, plus a new blue-team signal — per-GPU
interconnect byte counting. Branch `floppy`. This is a distinct calibration
regime from the porting-era fp32 single-GPU fit, which is kept as a labeled
legacy reference (never pooled).

## Task 1 — which NVLink counter API worked

Every candidate was checked empirically on **driver 580 + V100**:

| Option | API | Result |
|---|---|---|
| A | NVML `NVLINK_THROUGHPUT_DATA_TX/RX` (fields 138/139) | **NOT_SUPPORTED** |
| B | NVML legacy `nvmlDeviceGetNvLinkUtilizationCounter` | **NOT_SUPPORTED** |
| C | `nvidia-smi nvlink -gt d` | **"Data Tx: N/A"** |
| **D** | **DCGM `DCGM_FI_PROF_NVLINK_TX/RX_BYTES` (1011/1012)** + **PCIe (1009/1010)** via `dcgmi dmon` | **WORKS ✓** |

So `power_calibration/nvlink_monitor.py` is built on **Option D** — the same
nv-hostengine profiling path the DRAM-active `BytesSampler` (field 1005) already
uses (root host engine, unprivileged `dcgmi` client). Counters are rates
(bytes/s), so there is no cumulative-counter wraparound; cumulative bytes come
from trapezoidal integration. PCIe is captured alongside (fields 1009/1010).

**Validation 1 — known-size transfer** (200 × 256 MB, GPU0→GPU1):
expected 53.69 GB egress → measured **GPU0 TX 53.77 GB (ratio 1.00)**,
**GPU1 RX 53.77 GB (ratio 1.00)**.

**Validation 2 — symmetry:** GPU0.TX = GPU1.RX exactly (ratio **1.00**). Asserted
in `--selftest`; also a spoof-consistency check.

## Task 3 — DDP conversion + NVLink verification

`sample_ml_workload.py` is now **DDP-only** (launched via `torchrun
--nproc_per_node=2`; `require_torchrun()` fails clearly under plain python).
Rank-0-only stdout, aggregate ground truth = per-rank FLOPs × steps × world_size
(exact — ranks run identical shapes). `set_device` before `init_process_group`,
DDP after `.to(device)`, per-rank seed (non-degenerate all-reduce),
`destroy_process_group` in `finally`, per-GPU batch with explicit global batch.
bf16/tf32 removed (unsupported/no-op on Volta); precision = fp16 (baseline) / fp32.

**NCCL topology (`NCCL_DEBUG=INFO`)** — NVLink P2P confirmed, **no SHM/socket
fallback**:

```
Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM      (× 12 channels, both directions)
12 coll channels, 12 collnet channels, 0 nvls channels, 16 p2p channels
nvidia-smi nvlink -s: 6 links/GPU @ 25.781 GB/s   (NV6)
```

**Validation 3 — predicted vs measured all-reduce traffic.** Ring all-reduce,
n=2: each GPU sends ≈ receives ≈ 1× grad_bytes/step, so the both-GPU sum of TX+RX
= **4 × grad_bytes/step** (grads all-reduce in **FP32 = 4 B/param even under
AMP**). Across all 88 fp16 frontier configs:

> measured NVLink/step ÷ (4 × n_params × 4 B) = **1.00 ± 0.01** (n=88)

The DDP all-reduce is predictable from model size alone to within 1%
(`writeup/fig_ddp_nvlink.png`).

## Task 2 — the fp16-AMP DDP sweep

91 configs (families A dense-medium, B long-sequence, C wide-FFN, E deep-narrow,
F Jetson-scale boundary probes, G fp32 contrast, H nhead geometry, I SGD),
`batch_size` per-GPU, `steps` auto-sized to ~100 s active. Dual-GPU sampling
(power summed, util per-GPU, DRAM summed, NVLink/PCIe bytes), OOM→excluded,
incremental record dumps → `eval_results_v100_ddp_records.json`.

**How many cleared the both-GPU ≥80% gate:**

> **88 / 91 fp16 (+ 2 / 2 fp32) = 90 / 91 frontier**, **0 OOM/failures**
> — vs **12 / 38** in the porting-era single-GPU fp32 sweep.

The one sub-frontier config is `d768_b8_s128_L6_ff3072` (fam F, 87/77%) — exactly
the small "boundary probe" family F is designed to map. The fp16-DDP regime
saturates both V100s where the Jetson-scale single-GPU shapes left them idle.

## Refit — the fp16-DDP estimator

Fit on the 88 fp16 frontier runs (`refit_ddp.py`), power summed over both GPUs,
aggregate FLOPs. **A separate matched set from the legacy fp32 single-GPU fit;
never pooled.**

| | 2-param | 3-param (EMC) |
|---|---|---|
| E_MARGINAL (J/TFLOP) | **4.74** | 3.56 |
| E_PER_TB (J/TB) | — | **91.6** (HBM2-plausible) |
| P_OVERHEAD (W) | 0.0 | 0.0 |
| held-out error | **max 32%, mean 13%** (LOO) | max 48%, mean 15% |

`POWER_OVERHEAD_W = 0` reflects the saturated regime — at ~97% util net energy is
almost purely FLOPs. The error is looser than the porting-era fp32 fit (8%/14% on
12 configs) because the 88 configs span a far wider workload space: per-family
mean error runs 9–19% fairly uniformly (not an outlier), so the estimator is a
**rough meter** on this diverse regime, not the tight instrument of the narrow
set. `writeup/fig_ddp_est_vs_truth.png` shows the held-out estimates against
ground truth (aggregate 1651–5192 TFLOPs).

## Artifacts

- `eval_results_v100_ddp_records.json` — 91-config sweep records (per-GPU util,
  DRAM TB, NVLink/PCIe bytes, n_params, mode/precision tags).
- `writeup/fig_ddp_nvlink.png` — NVLink/step vs 4× grad-bytes prediction.
- `writeup/fig_ddp_est_vs_truth.png` — held-out estimated vs true TFLOPs.
- `power_calibration/nvlink_monitor.py` (Task 1), `refit_ddp.py` (refit).
- fp16-DDP constants pasted into `detect_flops.py`; fp32 single-GPU kept as
  labeled legacy.
