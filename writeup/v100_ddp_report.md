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

## Ten-trial cross-validation + input ablation (SRF phase-I)

The single-sweep LOO above scores each workload on one trace. The SRF outline
calls for **10 trials of every workload**, so each of the three estimator inputs
(power, DRAM-active, NVLink) has a 10-trace distribution and the accuracy claim
is over resampled splits, not one measurement. All 10 trials are collected
(`run_trials.py`, one records JSON per trial); `analyze_trials.py` runs the
generalization analysis.

**Pool.** 88 fp16 workloads clear the 80% gate in *all* 10 trials (the one
excluded, `d768_b8_s128_L6_ff3072`, is frontier in 9/10 — the family-F boundary
probe, as designed). **200 random calibrate/evaluate splits**, 59 calibrate / 29
held-out each; on every split each workload contributes **one randomly drawn
trace of its 10**, the estimator is fit on the calibrate workloads and scored on
the held-out ones, so it must generalize to workloads it never saw.

**Headline — held-out error over the 200 splits (`fig_trials_cv_error.png`):**

| estimator inputs | median | mean | p90 | max |
|---|---|---|---|---|
| 1 signal · power | **12.9%** | 13.8% | 26.8% | 52.9% |
| 2 signals · power+DRAM | 15.0% | 16.5% | 31.8% | 69.3% |
| 3 signals · power+DRAM+NVLink | 15.0% | 16.5% | 31.8% | 69.3% |

**The input ablation (`fig_trials_ablation.png`) is the notable result: adding
DRAM and NVLink does *not* improve benign accuracy — it slightly hurts.** On the
saturated frontier the byte signals are collinear with the FLOP term (see the
`corr(gt,tb)`/cond diagnostics), so the extra coefficients are poorly determined
and add variance to held-out prediction rather than signal: power-only generalizes
best (12.9% median), power+DRAM is worse (15.0%), and **power+DRAM+NVLink is
identical to power+DRAM to the last decimal** — the NVLink coefficient fits to 0
in every split (`E_PER_NVL → 0`, the collinearity finding, now confirmed across
200 resamples, not one fit). The scatter also shows the estimator **compresses**:
tight to ground truth through the mid-range but under-reporting the largest
wide-FFN / long-sequence workloads (the tail that drives p90/max).

This is the empirical basis for the phase-II/III framing: the byte signals earn
their place through **adversarial robustness** (the NVLink consistency tripwire
below), not benign accuracy. A blue team optimizing purely for held-out error on
benign workloads would ship the 1-signal estimator; it is the *adversary* that
makes the extra inputs worth their variance cost. Reproduce with
`python3 analyze_trials.py` (writes the two figures above + a per-workload
breakdown `fig_trials_per_workload.png` and `trials_cv_results.json`).

## NVLink as an estimator input (adversarial robustness)

The interconnect signal was wired into the estimator two ways, with an honest
split between them:

**1. As a fitted energy term (4-param `estimate_tflops_nvl`):**
`E = a·FLOPs + b·DRAM + c·NVLink + d·t`. Empirically **`c` (E_PER_NVLINK_TB_J)
fits to 0** — the all-reduce volume is **0.89-collinear with the DRAM-bytes
term**, so the energy decomposition assigns the byte-energy to DRAM and the
4-param model collapses to the 3-param one (max 47.6% / mean 15.2%). As an
*accuracy* regressor NVLink is redundant. This is kept and reported, not hidden.

**2. As a consistency tripwire (`nvlink_consistency`) — where the robustness
actually lives.** The all-reduce invariant is razor-tight: NVLink/step = 16·n_params,
measured **1.00 ± 0.01×** (CV 1%), so honest fp16-DDP keeps
`NVLink_bytes / net_energy_J` in a bounded band. A workload that spoofs its
FLOP-energy signature breaks it:

| scenario | energy-only FLOP est | NVLink/J | tripwire |
|---|---|---|---|
| honest run (gt 5192 TF) | 4618 TF (0.9×) | 1.5e8 | **OK** |
| Spoof A: 2× energy, no all-reduce | 9236 TF (**1.8× true**) | 7.7e4 | **COMM_STARVED** |
| Spoof B: NVLink flood, 1.5× energy | 6927 TF (**1.3× true**) | 5.1e9 | **COMM_FLOOD** |

The energy-only estimator is fooled in both directions (a compute-burn faking
training; an interconnect flood inflating power); the NVLink tripwire flags both,
with **0/88 false positives** on the honest frontier. So the interconnect signal
does not improve benign accuracy (it can't — it's collinear with DRAM in energy),
but it makes the FLOP estimate **robust to energy-spoofing adversaries**, which was
the goal. Reproduce with `python3 nvlink_tripwire_demo.py`.

*Live-daemon note:* the tripwire + 4-param estimator are library functions
(`detect_flops.py`) exercised at the sweep/analysis level. Wiring them into the
live `run_background_monitor` loop needs the daemon's single-GPU path converted to
both-GPU DDP monitoring (summed power + a DCGM NVLink reader) — a follow-on, since
the constants and band are both-GPU.

## Artifacts

- `eval_results_v100_ddp_records.json` (trial 1) + `..._trial{2..10}_records.json`
  — the 10-trial sweep records (per-GPU util, DRAM TB, NVLink/PCIe bytes,
  n_params, mode/precision tags), collected by `run_trials.py`.
- `analyze_trials.py` — 10-trial generalization analysis (200 calibrate/evaluate
  splits × random trace) + input ablation; `trials_cv_results.json` = raw errors.
- `writeup/fig_trials_cv_error.png` — held-out estimate vs truth over the splits.
- `writeup/fig_trials_ablation.png` — input ablation (1/2/3 signals) error CDF.
- `writeup/fig_trials_per_workload.png` — per-workload median held-out error.
- `writeup/fig_ddp_nvlink.png` — NVLink/step vs 4× grad-bytes prediction.
- `writeup/fig_ddp_est_vs_truth.png` — held-out estimated vs true TFLOPs.
- `power_calibration/nvlink_monitor.py` (Task 1), `refit_ddp.py` (refit, now also
  fits the 4-param model + reports the tripwire band).
- `nvlink_tripwire_demo.py` — the adversarial demonstration.
- `detect_flops.py`: `estimate_tflops_nvl` (4-param) + `nvlink_consistency`
  (tripwire) + their matched constants; fp16-DDP energy constants; fp32 single-GPU
  kept as labeled legacy.
