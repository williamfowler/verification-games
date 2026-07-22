# Instructions for Claude Code: NVLink traffic monitoring, V100 hyperparameter sweep, and DDP conversion

**Target machine:** floppy (2× Tesla V100-SXM2-16GB on TNS-2SXM2 adapter board, NVLink NV6 confirmed).
**Repo:** github.com/williamfowler/verification-games, branch `floppy`. Read `detect_flops.py`, `power_calibration/calibrate_power.py`, `sample_ml_workload.py`, and `eval_power_monitor.py` before editing anything; the contracts below come from them.

## Framing — read this before touching anything

This phase is **not a port of the Jetson experiment to new hardware**. It is a **reproduction of the overall experiment, redesigned to fit what this hardware is actually for**. Earlier work-in-progress changes assumed the former; that assumption is wrong and should be unwound where it conflicts with the following:

- **The workload is what a real training run on V100s looks like:** DDP over NVLink across both GPUs, **FP16 AMP with GradScaler** (the standard V100 training precision — that's what the tensor cores are for). FP16-AMP-DDP is the *baseline* configuration, not a variant axis.
- **The blue team gains a signal that didn't exist on the Jetson:** per-GPU interconnect ingress/egress (Task 1). This is a designed-in part of the reproduction, not an add-on.
- The existing repo state is mid-port (single-GPU, fp32-calibrated — see the 2026-07-22 fit constants in `detect_flops.py`). That fp32 single-GPU fit stays in the repo as a **legacy reference from the porting era**; the new calibration regime is fp16-AMP DDP, and records from the two regimes must never be pooled into one fit (tag records with mode + precision so the fits stay separable).
- **All of this work happens on its own branch, and Jetson backwards-compatibility is explicitly not a requirement.** The code on this branch will never run on the Jetson again. Do not preserve Jetson code paths, dual-mode plumbing, or Jetson-era config entries for compatibility's sake — git history is the archive. Simplify aggressively where Jetson support was the only reason for complexity.

## Hard constraints

1. **Volta (sm_70) pinned stack:** driver 580-server (apt-mark held), CUDA 12.8, PyTorch 2.7.1+cu126, venv at `~/vg-venv` (`calibrate_power.find_venv_python()` locates it — reuse that). Never `apt upgrade` NVIDIA packages, never install the `cuda` metapackage, never install flash-attn.
2. **Precision on V100:** baseline is `--precision fp16` (autocast + GradScaler, exactly the pattern in `~/ddp_test.py`). `fp32` remains valid (used by the contrast family below and by legacy records). **Remove `bf16` and `tf32` from the argparse choices entirely** — bf16 is unsupported on Volta and tf32 is a silent no-op, and with no Jetson back-compat to preserve there is no reason to keep them even behind a guard. Update `ASSUMED_PRECISION` handling in `detect_flops.py`'s roofline so the fp16 sweep uses the `TFLOPS_FP16` peak (125) — the fp32 assumption there is a porting-era leftover for this regime.
3. **Attention:** SDPA paths only.
4. **Check actual state before overwriting.** Preserve existing CLIs and the stdout protocol (Task 3). Maintain the `# KEEP IN SYNC` contract between `sample_ml_workload.py` argparse defaults and `eval_power_monitor.CONFIG_DEFAULTS` if you touch either.
5. Power limits are 250 W/card; don't change. DCGM pattern on this box: root `nv-hostengine`, unprivileged `dcgmi` clients (see `BytesSampler`) — follow it.

---

## Task 1: Per-GPU ingress/egress byte counting on the GPU interconnect

**Goal:** a monitor reporting, per V100, bytes transmitted (egress) and received (ingress) at ~1–2 Hz, in the style of the existing blue-team samplers. This is the blue team's view of the gradient all-reduce — the observable that makes DDP training distinctive.

**Terminology note:** there is no NIC between the GPUs. GPU↔GPU traffic rides NVLink (6 links/card); host↔GPU traffic rides PCIe. "Ingress/egress per GPU" = **NVLink RX/TX + PCIe RX/TX per device**, NVLink primary.

### Method options, in preferred order — verify each empirically on driver 580 + V100

**Option A — NVML field values (preferred continuous path).** `pip install nvidia-ml-py` into `~/vg-venv` (**not** currently in `requirements.txt` — add it). `nvmlDeviceGetFieldValues` with `NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX` / `_RX` per link (scopeId 0–5, plus the all-links aggregate scope). Cumulative KiB counters — sample and difference; sum links per GPU.

**Option B — legacy Volta NVLink utilization counters (fallback).** If A returns zeros/`NOT_SUPPORTED`: `nvmlDeviceSetNvLinkUtilizationControl` (units = bytes) then `nvmlDeviceGetNvLinkUtilizationCounter` per link. Deprecated on newer architectures, native on Volta. Document which of A/B works in the script docstring.

**Option C — nvidia-smi CLI (validation cross-check only).** `nvidia-smi nvlink -gt d` (may need `nvidia-smi nvlink -sc 0bz` first). Must agree with A/B.

**Option D — DCGM (independent second estimator).** `DCGM_FI_PROF_NVLINK_TX_BYTES` (1011) / `_RX_BYTES` (1012) give bytes/s directly and drop straight into the `BytesSampler` pattern (`dcgmi dmon -e 1011,1012 -i 0,1`). Probe field support before the sweep, exactly as `run_sweep_session` already probes field 1005. If supported, log alongside NVML — measurement redundancy is itself blue-team-relevant.

**PCIe side:** `nvmlDeviceGetPcieThroughput` (20 ms window, KB/s, ~ms cost per call — 1–2 Hz fine) or DCGM 1009/1010.

### Deliverable

`power_calibration/nvlink_monitor.py`, following the repo's sampler conventions (threaded sampler class + standalone CLI, `CUDA_DEVICE_ORDER=PCI_BUS_ID` respected so indices match `GPU_INDEX`/`FLOP_GPU_INDEX` semantics). Log per-GPU timestamp, nvlink_tx/rx (cumulative + rate), pcie_tx/rx rate, to CSV or the existing SQLite pattern. Handle counter wraparound/reset (negative delta → flag, don't crash).

### Validation protocol (must run, must report numbers)

1. **Known-size transfer:** `torch.ones(N, device="cuda:0").to("cuda:1")` with p2p; measured GPU0-TX and GPU1-RX deltas ≈ N × dtype_size within a few percent.
2. **Symmetry invariant:** GPU0 TX ≈ GPU1 RX and vice versa. Assert it — also a blue-team consistency check against spoofing.
3. **DDP cross-check (after Task 3):** measured per-step NVLink traffic ≈ ring all-reduce prediction: n=2 → each GPU sends/receives ≈ 2·(n−1)/n × grad_bytes = 1× grad_bytes/step (DDP grads all-reduce in FP32 → 4 B/param, bucketed; note this stays FP32 even under AMP). Report predicted vs. measured.

---

## Task 2: Hyperparameter sweep for >80% GPU utilization (fp16-AMP DDP)

**Background from the repo:** `eval_power_monitor.py` holds the 38-config Jetson-era `CONFIGS`, measures avg GPU util via `PowerSampler` (nvidia-smi, 2 Hz), and gates the fit on `FRONTIER_MIN_GPU_UTIL = 80.0`. The 2026-07-22 porting-era fit found only **12** of those clear the gate on a single V100 (vs 21 on the Orin Nano): Jetson-sized shapes leave V100 SMs idle between undersized kernels, and fp16 makes kernels *faster*, which makes small shapes even harder to keep resident. The proposed set below pushes shapes into the regime where fp16 tensor-core GEMMs keep both GPUs busy.

**Sweep vehicle:** do not build a new harness. **Replace** the Jetson-era `CONFIGS` list outright with the table below (git history archives the old one) using the same dict keys: `d_model, batch_size, seq_len, num_layers, nhead, dim_feedforward, optimizer, precision, steps`. Optimizer values are **`adamw`/`sgd`**. The `--configs` precision-filter mechanism near line 1223 can be simplified or removed along with the bf16/tf32 entries it existed to manage — keep it only if it stays useful for selecting the fp16 vs fp32-contrast subsets.

**The sweep runs under DDP on both GPUs** (Task 3 must land first — see order of operations). `batch_size` is **per-GPU**; the ground-truth aggregation in Task 3 keeps the FLOP accounting consistent.

**Pass criterion:** avg GPU util ≥ 80% **on both GPUs** over the run (per-GPU util comes from the extended dual-GPU sampling in Task 3's harness work; warmup/FLOP-probe exclusion via the "Starting workload" trigger is unchanged).

**`steps` sizing:** target ~2–4 min active training per config (the harness's design assumption — several hundred power samples). Rates for these shapes on V100 are unknown; auto-size: run ~20 s, measure steps/s, set `steps = round(150 × rate)`, cache. No static guesses.

**Memory estimates** below assume, per GPU (DDP = full replica): AdamW fp32 state + fp16 autocast copies ≈ 18 B/param (SGD ≈ 14), un-checkpointed fp16 activations, mem-efficient SDPA, ×1.3 safety factor; fp32 contrast rows use 4 B activations, no fp16 copies. Rough by construction — **wrap each run in OOM handling** (catch `torch.cuda.OutOfMemoryError` in the harness, log OOM, `empty_cache()`, continue). Everything estimated >13 GiB was pre-filtered for the 16 GiB cards.

### Proposed configurations (91)

Column names map to config keys / CLI flags: `d_model`→`--d-model`, `seq_len`→`--seq-len`, `batch_size`→`--batch-size` (per GPU), `dim_feedforward`→`--dim-feedforward`, `num_layers`→`--num-layers`, `nhead`→`--nhead`.

| # | family | d_model | seq_len | batch_size (per GPU) | dim_feedforward | num_layers | nhead | optimizer | precision | ~params (M) | est. mem/GPU (GiB) |
|---|--------|---------|---------|----------------------|-----------------|------------|-------|-----------|-----------|-------------|--------------------|
| 1 | A | 1024 | 256 | 8 | 4096 | 6 | 8 | adamw | fp16 | 77 | 2.2 |
| 2 | A | 1024 | 256 | 8 | 4096 | 12 | 8 | adamw | fp16 | 152 | 4.4 |
| 3 | A | 1024 | 256 | 8 | 4096 | 16 | 8 | adamw | fp16 | 202 | 5.9 |
| 4 | A | 1024 | 256 | 16 | 4096 | 6 | 8 | adamw | fp16 | 77 | 2.8 |
| 5 | A | 1024 | 256 | 16 | 4096 | 12 | 8 | adamw | fp16 | 152 | 5.5 |
| 6 | A | 1024 | 256 | 16 | 4096 | 16 | 8 | adamw | fp16 | 202 | 7.3 |
| 7 | A | 1024 | 256 | 32 | 4096 | 6 | 8 | adamw | fp16 | 77 | 3.9 |
| 8 | A | 1024 | 256 | 32 | 4096 | 12 | 8 | adamw | fp16 | 152 | 7.7 |
| 9 | A | 1024 | 256 | 32 | 4096 | 16 | 8 | adamw | fp16 | 202 | 10.3 |
| 10 | A | 1024 | 512 | 8 | 4096 | 6 | 8 | adamw | fp16 | 77 | 2.8 |
| 11 | A | 1024 | 512 | 8 | 4096 | 12 | 8 | adamw | fp16 | 152 | 5.5 |
| 12 | A | 1024 | 512 | 8 | 4096 | 16 | 8 | adamw | fp16 | 202 | 7.3 |
| 13 | A | 1024 | 512 | 16 | 4096 | 6 | 8 | adamw | fp16 | 77 | 3.9 |
| 14 | A | 1024 | 512 | 16 | 4096 | 12 | 8 | adamw | fp16 | 152 | 7.7 |
| 15 | A | 1024 | 512 | 16 | 4096 | 16 | 8 | adamw | fp16 | 202 | 10.3 |
| 16 | A | 1024 | 512 | 32 | 4096 | 6 | 8 | adamw | fp16 | 77 | 6.1 |
| 17 | A | 1024 | 512 | 32 | 4096 | 12 | 8 | adamw | fp16 | 152 | 12.1 |
| 18 | A | 1024 | 1024 | 8 | 4096 | 6 | 8 | adamw | fp16 | 77 | 3.9 |
| 19 | A | 1024 | 1024 | 8 | 4096 | 12 | 8 | adamw | fp16 | 152 | 7.7 |
| 20 | A | 1024 | 1024 | 8 | 4096 | 16 | 8 | adamw | fp16 | 202 | 10.3 |
| 21 | A | 1024 | 1024 | 16 | 4096 | 6 | 8 | adamw | fp16 | 77 | 6.1 |
| 22 | A | 1024 | 1024 | 16 | 4096 | 12 | 8 | adamw | fp16 | 152 | 12.1 |
| 23 | A | 1024 | 1024 | 32 | 4096 | 6 | 8 | adamw | fp16 | 77 | 10.4 |
| 24 | A | 1536 | 256 | 8 | 6144 | 6 | 12 | adamw | fp16 | 172 | 4.6 |
| 25 | A | 1536 | 256 | 8 | 6144 | 12 | 12 | adamw | fp16 | 342 | 9.1 |
| 26 | A | 1536 | 256 | 8 | 6144 | 16 | 12 | adamw | fp16 | 455 | 12.1 |
| 27 | A | 1536 | 256 | 16 | 6144 | 6 | 12 | adamw | fp16 | 172 | 5.4 |
| 28 | A | 1536 | 256 | 16 | 6144 | 12 | 12 | adamw | fp16 | 342 | 10.7 |
| 29 | A | 1536 | 256 | 32 | 6144 | 6 | 12 | adamw | fp16 | 172 | 7.0 |
| 30 | A | 1536 | 512 | 8 | 6144 | 6 | 12 | adamw | fp16 | 172 | 5.4 |
| 31 | A | 1536 | 512 | 8 | 6144 | 12 | 12 | adamw | fp16 | 342 | 10.7 |
| 32 | A | 1536 | 512 | 16 | 6144 | 6 | 12 | adamw | fp16 | 172 | 7.0 |
| 33 | A | 1536 | 512 | 32 | 6144 | 6 | 12 | adamw | fp16 | 172 | 10.3 |
| 34 | A | 1536 | 1024 | 8 | 6144 | 6 | 12 | adamw | fp16 | 172 | 7.0 |
| 35 | A | 1536 | 1024 | 16 | 6144 | 6 | 12 | adamw | fp16 | 172 | 10.3 |
| 36 | A | 2048 | 256 | 8 | 8192 | 6 | 16 | adamw | fp16 | 306 | 7.8 |
| 37 | A | 2048 | 256 | 16 | 8192 | 6 | 16 | adamw | fp16 | 306 | 8.9 |
| 38 | A | 2048 | 256 | 32 | 8192 | 6 | 16 | adamw | fp16 | 306 | 11.1 |
| 39 | A | 2048 | 512 | 8 | 8192 | 6 | 16 | adamw | fp16 | 306 | 8.9 |
| 40 | A | 2048 | 512 | 16 | 8192 | 6 | 16 | adamw | fp16 | 306 | 11.1 |
| 41 | A | 2048 | 1024 | 8 | 8192 | 6 | 16 | adamw | fp16 | 306 | 11.1 |
| 42 | B | 768 | 2048 | 2 | 3072 | 6 | 12 | adamw | fp16 | 43 | 1.8 |
| 43 | B | 768 | 2048 | 2 | 3072 | 12 | 12 | adamw | fp16 | 86 | 3.5 |
| 44 | B | 768 | 2048 | 4 | 3072 | 6 | 12 | adamw | fp16 | 43 | 2.6 |
| 45 | B | 768 | 2048 | 4 | 3072 | 12 | 12 | adamw | fp16 | 86 | 5.2 |
| 46 | B | 768 | 2048 | 8 | 3072 | 6 | 12 | adamw | fp16 | 43 | 4.2 |
| 47 | B | 768 | 2048 | 8 | 3072 | 12 | 12 | adamw | fp16 | 86 | 8.4 |
| 48 | B | 768 | 4096 | 2 | 3072 | 6 | 12 | adamw | fp16 | 43 | 2.6 |
| 49 | B | 768 | 4096 | 2 | 3072 | 12 | 12 | adamw | fp16 | 86 | 5.2 |
| 50 | B | 768 | 4096 | 4 | 3072 | 6 | 12 | adamw | fp16 | 43 | 4.2 |
| 51 | B | 768 | 4096 | 4 | 3072 | 12 | 12 | adamw | fp16 | 86 | 8.4 |
| 52 | B | 768 | 4096 | 8 | 3072 | 6 | 12 | adamw | fp16 | 43 | 7.5 |
| 53 | B | 1024 | 2048 | 2 | 4096 | 6 | 16 | adamw | fp16 | 77 | 2.8 |
| 54 | B | 1024 | 2048 | 2 | 4096 | 12 | 16 | adamw | fp16 | 152 | 5.5 |
| 55 | B | 1024 | 2048 | 4 | 4096 | 6 | 16 | adamw | fp16 | 77 | 3.9 |
| 56 | B | 1024 | 2048 | 4 | 4096 | 12 | 16 | adamw | fp16 | 152 | 7.7 |
| 57 | B | 1024 | 2048 | 8 | 4096 | 6 | 16 | adamw | fp16 | 77 | 6.1 |
| 58 | B | 1024 | 2048 | 8 | 4096 | 12 | 16 | adamw | fp16 | 152 | 12.1 |
| 59 | B | 1024 | 4096 | 2 | 4096 | 6 | 16 | adamw | fp16 | 77 | 3.9 |
| 60 | B | 1024 | 4096 | 2 | 4096 | 12 | 16 | adamw | fp16 | 152 | 7.7 |
| 61 | B | 1024 | 4096 | 4 | 4096 | 6 | 16 | adamw | fp16 | 77 | 6.1 |
| 62 | B | 1024 | 4096 | 4 | 4096 | 12 | 16 | adamw | fp16 | 152 | 12.1 |
| 63 | B | 1024 | 4096 | 8 | 4096 | 6 | 16 | adamw | fp16 | 77 | 10.4 |
| 64 | C | 1024 | 512 | 8 | 8192 | 6 | 8 | adamw | fp16 | 127 | 4.3 |
| 65 | C | 1024 | 512 | 16 | 8192 | 6 | 8 | adamw | fp16 | 127 | 5.9 |
| 66 | C | 1024 | 512 | 32 | 8192 | 6 | 8 | adamw | fp16 | 127 | 9.1 |
| 67 | C | 1024 | 1024 | 8 | 8192 | 6 | 8 | adamw | fp16 | 127 | 5.9 |
| 68 | C | 1024 | 1024 | 16 | 8192 | 6 | 8 | adamw | fp16 | 127 | 9.1 |
| 69 | C | 1536 | 512 | 8 | 12288 | 6 | 12 | adamw | fp16 | 286 | 8.6 |
| 70 | C | 1536 | 512 | 16 | 12288 | 6 | 12 | adamw | fp16 | 286 | 11.0 |
| 71 | C | 1536 | 1024 | 8 | 12288 | 6 | 12 | adamw | fp16 | 286 | 11.0 |
| 72 | E | 512 | 512 | 16 | 2048 | 24 | 8 | adamw | fp16 | 76 | 6.0 |
| 73 | E | 512 | 512 | 32 | 2048 | 24 | 8 | adamw | fp16 | 76 | 10.4 |
| 74 | E | 512 | 512 | 16 | 2048 | 32 | 8 | adamw | fp16 | 101 | 8.1 |
| 75 | E | 768 | 512 | 16 | 3072 | 24 | 8 | adamw | fp16 | 171 | 10.3 |
| 76 | F | 512 | 256 | 16 | 2048 | 6 | 8 | adamw | fp16 | 19 | 1.0 |
| 77 | F | 512 | 512 | 8 | 2048 | 3 | 8 | adamw | fp16 | 10 | 0.5 |
| 78 | F | 768 | 256 | 16 | 3072 | 6 | 8 | adamw | fp16 | 43 | 1.8 |
| 79 | F | 768 | 512 | 16 | 3072 | 6 | 8 | adamw | fp16 | 43 | 2.6 |
| 80 | F | 768 | 128 | 8 | 3072 | 6 | 8 | adamw | fp16 | 43 | 1.1 |
| 81 | F | 1024 | 256 | 8 | 4096 | 3 | 8 | adamw | fp16 | 39 | 1.1 |
| 82 | F | 1024 | 128 | 32 | 4096 | 6 | 8 | adamw | fp16 | 77 | 2.8 |
| 83 | F | 512 | 1024 | 16 | 2048 | 12 | 8 | adamw | fp16 | 38 | 5.2 |
| 84 | G | 1024 | 512 | 16 | 4096 | 12 | 8 | adamw | fp32 | 152 | 11.7 |
| 85 | G | 768 | 512 | 16 | 3072 | 6 | 8 | adamw | fp32 | 43 | 4.1 |
| 86 | H | 1024 | 512 | 16 | 4096 | 12 | 4 | adamw | fp16 | 152 | 7.7 |
| 87 | H | 1024 | 512 | 16 | 4096 | 12 | 16 | adamw | fp16 | 152 | 7.7 |
| 88 | H | 1024 | 512 | 16 | 4096 | 12 | 32 | adamw | fp16 | 152 | 7.7 |
| 89 | I | 1024 | 512 | 16 | 4096 | 12 | 8 | sgd | fp16 | 152 | 7.0 |
| 90 | I | 1536 | 512 | 16 | 6144 | 12 | 12 | sgd | fp16 | 342 | 12.4 |
| 91 | I | 1024 | 1024 | 16 | 4096 | 12 | 8 | sgd | fp16 | 152 | 11.4 |

Family key: **A** dense medium fp16 (primary pass candidates — tensor-core GEMMs) · **B** long-sequence attention-heavy · **C** wide-FFN (d_ff = 8×d_model) · **D** large models near the per-GPU memory ceiling (SGD variants for optimizer-state headroom) · **E** deep-narrow (launch-overhead probe — fp16's short kernels stress this hardest) · **F** boundary probes at Jetson-frontier scale (expected marginal/fail — they map where the V100 gate sits, which is itself data) · **G** fp32 contrast twins (precision axis: same shapes, ~8× slower matmul peak — do these clear the gate more easily per FLOP?) · **H** nhead geometry contrast at fixed FLOPs (for the CUPTI kernel-geometry estimator; nhead=32 deliberately drops head_dim below 64) · **I** SGD-vs-AdamW contrast twins.

Sizing rationale to retain in the report: all dims are multiples of 8 (mostly 64+) so FP16 tensor cores engage; head_dim = d_model/nhead ≥ 64 preferred outside family H; `dim_feedforward` carries ~two-thirds of block FLOPs, hence family C; nhead doesn't change FLOPs at fixed d_model. If wall-clock is tight, priority: A → G → I → H → C → D → B → E → F.

---

## Task 3: Convert `sample_ml_workload.py` to DDP over the two V100s

A working reference exists at `~/ddp_test.py` on floppy (aka `ddp_sanity.py`) — read it first; it encodes the Volta-safe DDP patterns. Then produce a **minimal-diff** conversion of `sample_ml_workload.py`. The script is load-bearing for the whole measurement pipeline, so these are contract, not style:

**Preserve the stdout protocol exactly.** `calibrate_power.run_workload` starts power sampling on the literal line `[redteam] Starting workload...` and parses ground truth with the regex `Ground truth total\s*:\s*([\d.]+)\s*TFLOPs`. Under torchrun, **only rank 0 prints**; its `Ground truth total` must be the **aggregate across ranks** (per-rank `flops_per_step × steps × world_size` — ranks run identical shapes so multiplication is exact; state this in a comment rather than silently absorbing it).

**One script, one mode: DDP-only.** With no Jetson back-compat, do not build dual-mode plumbing. The script assumes torchrun (`RANK`/`LOCAL_RANK`/`WORLD_SIZE` from the environment) and fails with a clear message if launched without it. Single-GPU debugging is free anyway: `torchrun --standalone --nproc_per_node=1` runs the same DDP code path on one GPU — no separate branch of logic needed. (The Jetson↔V100 comparison in the write-up compares *results across branches*, not one script running on both machines.)

**Structure:** `torch.cuda.set_device(local_rank)` **before** `dist.init_process_group("nccl")`; `DDP(model, device_ids=[local_rank])` after `.to(device)`; `dist.destroy_process_group()` in a `finally:` so crashed sweep runs don't leave zombies; seed per rank (`base + rank`) so synthetic data differs across ranks and the all-reduce carries non-degenerate traffic; one GradScaler per process (enabled for fp16, as today); `batch_size` stays **per-GPU** — print the global batch explicitly.

**Harness extension for DDP sweeps** (separate commit from the script conversion):
- `run_workload` launches via `[<venv>/bin/torchrun, --standalone, --nproc_per_node=2, WORKLOAD_SCRIPT, ...flags]` unconditionally (this branch is DDP-only), and `child_env` **stops pinning `CUDA_VISIBLE_DEVICES`** — both GPUs must be visible. `GPU_INDEX`/`FLOP_GPU_INDEX` single-GPU selection logic can be retired or repurposed for per-GPU sample attribution. Preserve unbuffered stdout (`-u` equivalent; check whether torchrun needs `PYTHONUNBUFFERED=1` in the child env instead) so the trigger line arrives promptly.
- Measurement covers **both** GPUs: power sampled per GPU and **summed** for the energy integral; util recorded per GPU; frontier gate = both GPUs ≥ 80%; `BytesSampler` covering both (`dcgmi dmon -i 0,1` streams both — parse per-GPU rows). Keep single-GPU records schema-compatible: add fields (`ddp`, per-GPU arrays), don't change existing ones, and tag every record with mode + precision so fp16-DDP records never silently pool with the legacy fp32 single-GPU fit.
- Start Task 1's NVLink monitor on the same sampling window so each sweep record carries interconnect bytes — this is the new blue-team observable and belongs in the record, not a side file.

**Verify NVLink is actually used:** one-time `NCCL_DEBUG=INFO` run; confirm P2P/NVLink channels (not SHM/socket fallback). If NCCL fell back, **stop and report** — Task 1's characterization and the DDP traffic story depend on the NVLink path. Record the log lines.

**FLOP accounting notes for the report (don't silently absorb):** aggregate vs per-GPU FLOPs stay distinct in `detect_flops.py`'s reconstruction; DDP adds ≈ grad_bytes of NVLink traffic per GPU per step (cross-reference the Task 1 monitor); DDP's comm/compute overlap changes the power and DRAM profile relative to single-GPU at identical shapes — a genuine property of the new regime, worth a sentence in the write-up, not a discrepancy to hide.

**Acceptance criteria:** (a) 200 steps under torchrun on both V100s, loss decreasing, no NCCL warnings; (b) `torchrun --nproc_per_node=1` also works (the debug path); (c) all-reduce probe passes as in the reference script; (d) `run_workload` yields a valid record with sane GT parsing, both-GPU power, and NVLink bytes; (e) `nvidia-smi` shows both GPUs loaded, temps in envelope.

---

## Order of operations

1. **Task 3 script conversion** (the reproduction's workload) + `NCCL_DEBUG` NVLink verification.
2. **Task 1 monitor** + validations 1–2, then validation 3 (DDP all-reduce cross-check) against the converted script.
3. **Task 3 harness extension** (DDP launch path, dual-GPU sampling, NVLink bytes in records).
4. **Task 2 sweep** under fp16-AMP DDP with `CONFIGS_V100_DDP`, auto-sized steps.
5. Refit the power→FLOP estimator on the new frontier subset; keep the fp32 single-GPU fit as the labeled legacy reference.

Final report: which NVLink counter API worked (A vs B), known-size-transfer numbers, NCCL topology log lines, predicted-vs-measured all-reduce traffic, sweep records JSON, and how many configs cleared the both-GPU 80% gate (vs 12/38 in the porting-era single-GPU fp32 sweep).
