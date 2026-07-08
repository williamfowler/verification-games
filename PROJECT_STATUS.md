# Project Status — Verification Games (Power-Based FLOP Monitoring)

_Last updated: 2026-07-07_

## What this project is

A toy model of **AI treaty verification**: can a monitoring body detect an
unauthorized frontier-scale training run on a compute cluster it oversees, using
*only* externally observable hardware signals?

- **Blue team (monitor)** — `detect_flops.py`: a daemon that estimates how many
  FLOPs a GPU workload performed from power draw, GPU/EMC utilization, and clock
  frequency. It never reads the workload's own code or self-reported counts.
- **Red team (workload)** — `sample_ml_workload.py`: a PyTorch transformer
  training run that emits a *ground-truth* FLOP count (via
  `torch.utils.flop_counter.FlopCounterMode`) so the blind estimate can be scored.
  `adversarial_workload.py` adds deliberately deceptive red-team workloads.

The research question is the **gap between the blue team's estimate and the red
team's ground truth** — including when the red team is actively evading.

## Hardware

NVIDIA Jetson Orin Nano 8GB, JetPack 6.2 / L4T R36.5.0, 1024-core Ampere GPU,
running in the **25W nvpmodel mode**. Everything is wired to this one device.

- Power comes from the on-board **INA3221** sensor, `VDD_CPU_GPU_CV` rail — a
  **shared CPU+GPU+CV rail** (there is no GPU-only rail).
- **DRAM traffic** comes from the actmon debugfs counters (root-only); jtop and
  tegrastats still fail to expose EMC on this build. The counter's absolute
  scale was measured on 2026-07-07 (`power_calibration/actmon_scale_bench.py`):
  **k = 0.0133 actmon-TB per true TB** (~1.3%, linear, negligible idle
  background). The scale is absorbed into the fitted `E_PER_TB_J`, whose
  physical equivalent — **61.5 J per true TB** — lands inside the LPDDR5
  ballpark (~50–150 J/TB), a real sanity win for the blind fit.
- **Precision correction (2026-07-07):** this build's torch (2.9.1) defaults
  `matmul.allow_tf32 = False`, so all pre-2026-07 calibration ran **FP32
  matmul** despite older docs saying TF32. `sample_ml_workload.py --precision
  {fp32,tf32,fp16,bf16}` now controls this explicitly (fp32 = the historical
  default).

### Constants are per-device, per-power-mode — recalibrate or they are fiction

Every fitted constant in `detect_flops.py` is a **matched set measured on this
one device in the 25W mode**. `CALIBRATION_FINGERPRINT` records device model /
L4T / nvpmodel mode; the daemon compares it against the live system at startup
and prints a loud RECALIBRATE banner on mismatch. On any new hardware (or after
`nvpmodel -m`): re-run `eval_power_monitor.py` there, then paste its
`RECOMMENDED CONSTANTS` block *and* the matched fingerprint + idle baseline.
The **methodology is the portable artifact; the numbers are not.**

## How the estimator works

The primary estimator is a **2-parameter active-energy model** over net energy
(measured − idle baseline):

```
TFLOPs = (E_net − POWER_OVERHEAD_W · t_active) / E_MARGINAL_J_PER_TFLOP
```

A second, parallel **3-param EMC estimator** (`estimate_tflops_emc`) subtracts a
measured memory-energy term first (`E_PER_TB_J · TB_moved`, from the actmon DRAM
counter). Both run every poll and are logged for A/B; a roofline estimate runs
as a diagnostic only.

**Role split, settled 2026-07-07:** the 2-param estimator is the *meter*; the
EMC estimator is a *tripwire*. On benign data the byte term is weakly determined
and strictly less stable than 2-param (59% of resampled splits fail the 10%
target). But the adversarial probe (`adversarial_results.txt`) shows why it
stays deployed: a **memory-spoof workload at 98% GPU util** (burns energy via
DRAM traffic, ~zero FLOPs) fools the 2-param estimator into claiming
8.3 TFLOPs vs 1.8 true, while the EMC estimate collapses to **−96 TFLOPs** — an
unmistakable spoof flag. Cost: it over-corrects (see the byte-scale nuance
above) and adds ~+17% error on a legitimate pure-matmul control, so its output
is a flag, not a count.

## Current accuracy and scope (v2 sweep, 2026-07-07)

Latest sweep: `eval_results_v2.txt` — 38 configs spanning d_model/batch/seq/
layers **plus nhead {2,8}, ffn width {3072,4096}, seq 512, SGD, and precision
{fp32, tf32, fp16, bf16}**, one shared 642.0 mW baseline, seed 12345.

**FP32 frontier accuracy — PASS.** On the 21 fp32 frontier (≥80% util) runs:
held-out max err **8.95%**, ship-fit max err **8.82%** (mean 4.0%). The new
architecture axes are benign (holdouts: sgd 1.9%, seq512 4.3%, 6-layer 3.5%;
marginal fails at ffn≥3072 10.06% and d_model≤256 10.78%). Constants were
updated to this fit (the old ones let one v2 run breach 10%):

```
FALLBACK_IDLE_POWER_MW      = 642.0
POWER_OVERHEAD_W            = 3.750    # 2-param (meter)
E_MARGINAL_J_PER_TFLOP      = 5.72
POWER_OVERHEAD_EMC_W        = 1.463    # 3-param (tripwire)
E_MARGINAL_EMC_J_PER_TFLOP  = 5.61
E_PER_TB_J                  = 4621.672 # J per ACTMON-TB (≈61.5 J per true TB)
```

**How soft is a PASS?** `eval_generalization.txt` (offline holdout/stability
tooling, `--refit-from` / `--split holdout:<expr>` / `--stability N`): over 200
resampled splits the fp32 fit fails the 10% target on **13.5%** of splits
(improved from 23% pre-diversification), with held-out max err spanning
3.3–11.6%. Honest headline: *typically ~9% held-out max error, occasionally
~11-12%* — not "always under 10%".

**Precision is an open evasion channel — the estimator is precision-blind and
FP32-scoped.** The precision-blind fit over all precisions FAILS (16.4%
held-out; the tf32 outliers also degenerate the fit itself). Physics, not a
bug: a joule meter cannot see precision, and tensor-core modes do 2–3.5× more
FLOPs per joule. Measured evasion on this device:

| red-team choice | true TFLOPs | monitor reads | direction |
|---|---|---|---|
| tf32 (frontier, 87-89% util) | 101–110 | ~54 (−50%) | under-count |
| fp16/bf16 (drops to ~60% util → below the gate) | up to 146 | ~30–44 (−70%) | under-count + gate evasion |

Treat every estimate as a **lower bound with respect to precision**. Closing
this channel needs a precision-revealing signal (e.g. clock/power signatures or
declared-precision audits) — out of MVP scope, documented as the top known gap.

**Sub-frontier remains out of scope** (shared rail; <80% util unattributable;
estimates go ≤0 by design — "below the frontier detection floor").

**Sustained-run drift — PASS with a caveat** (`drift_results.txt`,
`power_calibration/drift_test.py`): a 49-min frontier run at 98.5% util scored
**5.85%** whole-run error, so the 2–4 min calibration does transfer to
sustained load. Caveat: net power declines ~2% as the device heats, so
per-window error creeps monotonically 4.7%→8.6% over the run —
under-estimation (the verifier-dangerous direction). Extrapolated, multi-hour
runs likely drift past 10%; a thermal-equilibrium calibration (or per-window
re-baselining) is the known fix if longer horizons matter.

## The calibration loop

1. `eval_power_monitor.py` sweeps the config pool (~2.5h for all 38), fits by
   relative least squares with `POWER_OVERHEAD` chosen by leave-one-out CV,
   gates to frontier runs, scores through the exact production code path, and
   prints `RECOMMENDED CONSTANTS` + the matched fingerprint.
2. Every sweep dumps per-run records to `<output stem>_records.json`.
   **Refits are free**: `--refit-from FILE` (JSON or an old report's RAW
   RECORDS) reruns gate → split → fit → report offline in seconds;
   `--split holdout:<key><op><val>` tests family extrapolation;
   `--stability N` measures split-dependence of constants and verdicts.
3. Paste the recommended block (constants + fingerprint + baseline) into
   `detect_flops.py`, re-run to confirm.

Sweep hygiene mirrors a real verifier: one startup idle baseline, randomized
train/test split, frontier gate. The EMC fit additionally requires the NOPASSWD
sudoers entry for `power_calibration/actmon_reader.py` (hard requirement — the
eval raises rather than silently reporting 2-param only).

## Files

| File | Role |
|---|---|
| `detect_flops.py` | Blue-team monitor daemon (+ fingerprint check) |
| `sample_ml_workload.py` | Red-team transformer workload (+ precision/optimizer axes) |
| `adversarial_workload.py` | Red-team deceptive workloads (memory-spoof / compute-dense) |
| `adversarial_probe.py` | Scores both estimators against the deceptive workloads |
| `eval_power_monitor.py` | Primary calibration/eval tool (sweep + offline refit/holdout/stability) |
| `power_calibration/calibrate_power.py` | Sampling helpers (PowerSampler, BytesSampler, run_workload) |
| `power_calibration/actmon_scale_bench.py` | Measures the actmon byte scale factor k |
| `power_calibration/drift_test.py` | 50-min sustained-run stability test |
| `eval_results_v2.txt` / `eval_results_v2_records.json` | Latest sweep report + replayable records |
| `eval_generalization.txt` | Holdout/stability analyses of both sweeps |
| `adversarial_results.txt` | Spoof-detection probe results |
| `drift_results.txt` | Sustained-run drift report |

## How to run

```bash
# Red-team workload (prints ground-truth TFLOPs)
.venv/bin/python3 sample_ml_workload.py --steps 150 --batch-size 8 --seq-len 64 --d-model 128

# Blue-team monitor daemon — MUST run as root (actmon read raises without it)
sudo .venv/bin/python3 detect_flops.py

# Full sweep (~2.5h) / offline refit (seconds)
.venv/bin/python3 eval_power_monitor.py --output eval_results.txt
.venv/bin/python3 eval_power_monitor.py --refit-from eval_results_v2_records.json --stability 200

# Adversarial probe (~7 min) and byte-scale benchmark (~2 min)
.venv/bin/python3 adversarial_probe.py
.venv/bin/python3 power_calibration/actmon_scale_bench.py
```

Always use the project venv interpreter; do **not** run the eval/probe as root.

## Notable findings log

- **2026-07-08 — 15W replication: methodology transfers, constants do not.**
  Rerunning the full calibration in nvpmodel 15W mode as pseudo-new-hardware
  reproduced a PASS (held-out 9.20%, all 22 frontier ≤8.92%) with completely
  different constants (E_MARGINAL 3.92 vs 5.72 J/TFLOP). The 25W constants
  scored on 15W runs read **−95% on average** — an unrecalibrated monitor is
  blind, which is what the fingerprint guard is for. New finding: at 15W the
  stability picture inverts — the EMC/3-param estimator becomes the robust
  model (2% split-failure vs 41% for 2-param), i.e. meter-vs-tripwire role
  assignments are power-mode-specific. Details in `eval_generalization.txt`.
- **2026-07-07 — precision evasion quantified:** tf32/fp16/bf16 under-read
  2–3.5×; fp16/bf16 additionally evade the util gate at this model scale. The
  estimator is now explicitly FP32-scoped and lower-bound-framed.
- **2026-07-07 — EMC byte term validated as a tripwire:** catches a 98%-util
  memory spoof decisively (−96 TFLOPs flag); physically plausible coefficient
  (61.5 J/true-TB); not an accuracy improvement on benign data.
- **2026-07-07 — single-split PASSes are soft:** 13.5% (v2) / 23% (v1) of
  resampled splits fail the 10% target; constants trade off across splits
  (E_MARGINAL vs P_OVERHEAD near-collinear on frontier data).
- **2026-06-30 — actmon is the only working EMC signal**; pipe-buffer race and
  boundary-chasing fits fixed earlier (see git history).

## Open items / next steps

- **Precision-revealing signals**: can clock-frequency/power signatures
  distinguish tensor-core vs CUDA-core execution? (Closes the main evasion
  channel if so.)
- **Byte-term as calibrated meter**: with k measured, fit `E_PER_TB_J` on
  intensity-spread data (incl. spoof-like configs) so the tripwire can become a
  quantitative memory-energy correction without over-correcting.
- **Ground-truth bytes from the red team** (parallel to FLOP ground truth) to
  fit the byte term against truth instead of blind.
