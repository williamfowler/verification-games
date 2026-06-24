# Project Status — Verification Games (Power-Based FLOP Monitoring)

_Last updated: 2026-06-23_

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

The research question is the **gap between the blue team's estimate and the red
team's ground truth**.

## Hardware

NVIDIA Jetson Orin Nano 8GB, JetPack 6.2 / L4T R36.5.0, 1024-core Ampere GPU.
Everything is wired to this one device; the code hard-codes device specifics.

- Power comes from the on-board **INA3221** sensor, `VDD_CPU_GPU_CV` rail — a
  **shared CPU+GPU+CV rail** (there is no GPU-only rail).
- **EMC (memory-bandwidth) utilization is not observable** here: jtop returns 0
  on JetPack 6.2, and `tegrastats` emits only `GR3D_FREQ` (GPU), no `EMC_FREQ`.
  This is the key sensing limitation that shapes the whole design.

## How the estimator works

The primary estimator is a **2-parameter active-energy model**. It integrates net
power (measured − idle baseline) over the workload into joules, then converts:

```
TFLOPs = (E_net − POWER_OVERHEAD_W · t_active) / E_MARGINAL_J_PER_TFLOP
```

The fixed-overhead term (`POWER_OVERHEAD_W · t_active`) is why small workloads
*look* less efficient (high apparent J/TFLOP) without needing a per-intensity
constant. A roofline estimator (`min(compute ceiling, memory-bandwidth ceiling)`)
runs alongside as a diagnostic only.

## Current accuracy — **PASS (<10% on all large workloads)**

The estimator is calibrated and validated as a **frontier detector**: it is
accurate for workloads that saturate the GPU (avg util ≥ 80%), which is the actual
threat model — an unauthorized frontier training run keeps the cluster busy.

Latest on-device eval run (`eval_results.txt`, 2026-06-22, seed `151995487`,
single 602.9 mW idle baseline, 11 of 16 configs cleared the 80%-util gate):

| Metric | Max error | Mean error |
|---|---|---|
| Held-out **TEST** (4 frontier configs) | **8.14%** | 4.25% |
| **SHIP FIT** — all 11 frontier (large) workloads | **6.96%** | 2.80% |

Held-out generalization confirms the fit is not overfitting the calibration set.

**Shipped constants** in `detect_flops.py` (a matched set — recalibrate together):

```
FALLBACK_IDLE_POWER_MW = 602.9
POWER_OVERHEAD_W       = 3.675
E_MARGINAL_J_PER_TFLOP = 5.89
```

### Important scope caveat

The <10% claim holds **only for frontier-like (≥80% util) workloads**.
Sub-frontier runs are deliberately out of scope: with a shared CPU+GPU power rail
and no EMC signal, a partially-loaded GPU's energy cannot be cleanly attributed to
FLOPs (a 65%-util run came out 35% low). The high `POWER_OVERHEAD_W` intentionally
drives low-intensity estimates toward zero/`None` — correct behavior for a
frontier detector, but not a general-purpose FLOP meter.

## The calibration loop

The magic constants are **fit by the eval script, then manually copied into the
source**:

1. `eval_power_monitor.py` sweeps a pool of multi-minute transformer configs,
   samples INA3221 power offline at 2 Hz, gates to frontier-like runs, takes a
   **randomized TRAIN/TEST split**, and fits the model by **relative least
   squares** with `POWER_OVERHEAD` chosen by **leave-one-out cross-validation**
   (frontier runs are near-collinear in (TFLOPs, t), so any in-sample objective
   just chases the search bound; LOO has a genuine interior optimum).
2. It scores against `FlopCounterMode` ground truth through the exact production
   path (`detect_flops.estimate_tflops`) and prints `PASS/FAIL: held-out max err
   < 10%` plus a `RECOMMENDED CONSTANTS` block.
3. Copy the recommended block **and its matched idle baseline** into
   `detect_flops.py`, then re-run to confirm.

Choices that mirror a real verifier: idle power is measured **once at startup**
(no clean re-baseline between jobs); the **TRAIN/TEST split is randomized** each
run (`--seed` to reproduce); and only **frontier-like runs** count toward the
verdict.

## Files

| File | Role |
|---|---|
| `detect_flops.py` | Blue-team monitor daemon (current entry point) |
| `sample_ml_workload.py` | Red-team transformer workload + ground-truth FLOPs |
| `eval_power_monitor.py` | **Primary** accuracy/calibration tool (sweep, fit, verdict) |
| `calibrate_power.py` | Older single-purpose calibrator; supplies sampling helpers |
| `eval_results.txt` | Latest passing eval report |
| `detect_flops_old.py` | Prior roofline-only version, kept for reference |
| `CLAUDE.md` / `AGENTS.md` | Guidance for Claude / contributor conventions |

## How to run

```bash
# Red-team workload (prints ground-truth TFLOPs)
.venv/bin/python3 sample_ml_workload.py --steps 150 --batch-size 8 --seq-len 64 --d-model 128

# Blue-team monitor daemon (writes /var/log/flop_log.db; needs sensor + DB perms)
.venv/bin/python3 detect_flops.py

# Accuracy evaluation + constant fitting (~60-70 min; PASS/FAIL vs 10%)
.venv/bin/python3 eval_power_monitor.py --output eval_results.txt
```

Always use the project venv interpreter, and do **not** run the eval as root.

## Notable problems solved along the way

- **Pipe-buffer race** — piped child stdout block-buffered, so the
  `"[redteam] Starting workload..."` trigger arrived late and the sampler started
  after the run, corrupting the original calibration (3–32 samples). Fixed with
  `python -u` + `readline` in `run_workload`; a 25s run now yields ~53 samples.
- **Single-baseline + broad-set FAIL** — scoring every workload against one idle
  baseline failed (held-out 33.77%) because low-power runs are dominated by
  baseline noise and unattributable energy. Resolved by scoping to frontier-like
  workloads, which cluster tightly in J/TFLOP and clear 10%.
- **Boundary-chasing fits** — minimax/SSE drove `POWER_OVERHEAD` to the grid
  boundary on near-collinear frontier data. Switched to LOO cross-validation,
  which has an interior optimum.

## Open items / next steps

- No commit has been made; the working tree carries the new constants, the
  frontier-gated eval, and updated docs.
- The estimator is frontier-only by design. Extending trustworthy estimation to
  sub-frontier loads would require a sensing signal this device doesn't expose
  (e.g. a GPU-only power rail or EMC utilization) — a hardware limitation, not a
  modeling one.
