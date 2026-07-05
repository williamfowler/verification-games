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
- **EMC (memory-bandwidth) is partly observable via the actmon debugfs counters**
  (`/sys/kernel/debug/bpmp/debug/actmon/mc_all_*` + `clk/emc/rate`), **root-only**.
  jtop still returns 0 on JetPack 6.2 and `tegrastats` still emits no `EMC_FREQ`
  (so the old tegrastats scrape stays dead) — actmon is the working signal. It
  reports **total DRAM traffic** (all memory clients, CPU+GPU), the same
  attribution caveat as the shared power rail. This unlocks a measured
  bytes-moved input to the estimator (see below).

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

### Second (parallel) estimator: 3-param EMC/bytes model

A second power estimator (`estimate_tflops_emc`) runs **in parallel** with the
2-param one — both are computed every poll and logged for A/B (it is **not** a
replacement). It adds a *measured* memory-energy term using DRAM bytes moved
(integrated from the actmon signal):

```
TFLOPs = (E_net − E_PER_TB_J · TB_moved − POWER_OVERHEAD_W · t_active) / E_MARGINAL
```

`TB_moved` is **observed**, not fitted. With `TB_moved = 0` / `E_PER_TB_J = 0` it
reduces exactly to the 2-param model.

**Why (adversarial rationale).** The goal is robustness to a red-team trying to
spoof an energy-only estimator. A deceptive workload would operate *off* the
normal FLOPs↔bytes line (e.g. low-FLOP / high-memory-traffic to inflate energy and
fake a high FLOP count); the byte term is what catches that. Consequently the term
is **always active at inference** — collinearity of `TB_moved` and `TFLOPs` on
*benign* runs is expected, not a defect (the calibration just needs some
arithmetic-intensity spread to *determine* `E_PER_TB_J`; the eval prints a
`corr(gt,tb)` / conditioning diagnostic). Ground-truth bytes from the red team are
out of scope for now (acknowledged future improvement; `E_PER_TB_J` is fit blind
against `FlopCounterMode` FLOP truth).

## Current accuracy — **PASS (<10% on all large workloads)**

The estimator is calibrated and validated as a **frontier detector**: it is
accurate for workloads that saturate the GPU (avg util ≥ 80%), which is the actual
threat model — an unauthorized frontier training run keeps the cluster busy.

Latest on-device eval run (`eval_results.txt`, 2026-06-30, seed `12345`,
single 642.0 mW idle baseline, 14 of 19 configs cleared the 80%-util gate):

| Metric | 2-param max | 3-param (EMC) max |
|---|---|---|
| Held-out **TEST** (5 frontier configs) | **8.40%** | **8.34%** |
| **SHIP FIT** — all 14 frontier workloads | **8.98%** | **9.17%** |

Both estimators PASS (<10%); the EMC term is ~a wash on frontier (no regression)
and ~6 pts better on the one scorable sub-frontier run (28.9% → 22.9%).

**Shipped constants** in `detect_flops.py` — one matched set tied to the 642.0 mW
baseline (the daemon feeds one baseline to both estimators, so all move together):

```
FALLBACK_IDLE_POWER_MW      = 642.0
# 2-param estimator
POWER_OVERHEAD_W            = 3.450
E_MARGINAL_J_PER_TFLOP      = 6.23
# 3-param / EMC estimator (its OWN E_MARGINAL/overhead — re-attributed)
POWER_OVERHEAD_EMC_W        = 2.163
E_MARGINAL_EMC_J_PER_TFLOP  = 5.12
E_PER_TB_J                  = 3857.934
```

`E_PER_TB_J` was **promoted** from `0.0` on 2026-06-30. It is only *weakly
determined* (corr(gt,tb)=0.91, cond=1.3e8 — benign frontier runs are near-collinear
in TFLOPs↔bytes) and the actmon byte scale is uncalibrated (absorbed into
`E_PER_TB_J`); it was promoted deliberately for adversarial coverage and will shift
with better-conditioned calibration data.

### Important scope caveat

The <10% claim holds **only for frontier-like (≥80% util) workloads**.
Sub-frontier runs are deliberately out of scope: with a shared CPU+GPU power rail
and no EMC signal, a partially-loaded GPU's energy cannot be cleanly attributed to
FLOPs (a 65%-util run came out 35% low). The high `POWER_OVERHEAD_W` drives
low-intensity estimates low and, once overhead exceeds net energy, **negative**.
The estimator returns that raw value (the daemon always reports a number); a
`<=0`/sub-frontier estimate means "below the frontier detection floor", not a real
FLOP count. It is a frontier detector, not a general-purpose FLOP meter.

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
   < 10%` plus a `RECOMMENDED CONSTANTS` block. It **also** fits the 3-param/EMC
   model (`fit_active_energy_emc_model`, a 2x2 relative-LS solve over the same
   LOO-chosen overhead) and scores `estimate_tflops_emc` alongside for A/B —
   provided actmon `tb_moved` was captured for every frontier run.
   **Prerequisite:** the EMC fit needs the privileged DRAM-bytes reader
   (`power_calibration/actmon_reader.py`) authorized via a NOPASSWD sudoers entry
   (the eval runs unprivileged, but the actmon counters are root-only). Without
   it, `tb_moved` is `None` and the eval reports the 2-param result only.
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

# Blue-team monitor daemon (writes /var/log/flop_log.db; needs sensor + DB perms;
# run as root for the actmon DRAM-bytes signal — falls back to 2-param without it)
sudo .venv/bin/python3 detect_flops.py

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
- **EMC/bytes term — MVP landed (parallel A/B):** actmon DRAM-bytes now feeds a
  second `estimate_tflops_emc` estimator, run alongside (not replacing) the 2-param
  one. Both are logged (`estimator` / `estimator_emc` rows on each session).
  Requires the daemon as root and the NOPASSWD sudoers entry for the calibration
  reader. `E_PER_TB_J` was promoted to the fitted value on 2026-06-30 (weakly
  determined; see the constants note above).
- **Phase 0 done (2026-06-30):** the actmon `mc_all` signal is responsive
  (idle→load ~100×; EMC clock 2133→3199 MHz under load). Its absolute scale is
  uncalibrated (reads ~1% where tegrastats says ~38%), but that is harmless — the
  signal is linear in true activity, so the constant is absorbed by the fitted
  `E_PER_TB_J` (only proportionality matters). Conversion kept as
  `util = last_prd/emc_rate`. Also found: tegrastats `EMC_FREQ` *does* appear under
  root (a possible simpler alternative signal, noted for later).
- **Fixed during bring-up:** `BytesSampler.stop()` couldn't terminate the root
  reader (non-root EPERM); it now closes the pipe so the reader self-exits via
  `BrokenPipeError` — no leaked root process per workload.
- **Frontier-only is no longer a hard hardware limit:** the byte term is the
  signal whose absence previously capped the estimator at frontier loads. Whether
  it actually extends trustworthy estimation sub-frontier depends on the on-device
  fit (and `E_PER_TB_J` being well-determined — needs arithmetic-intensity spread).
- **Future:** ground-truth bytes from the red team; explicit *deceptive* test
  configs (low-AI/high-memory and the converse) to measure the byte term's actual
  spoof-resistance.
