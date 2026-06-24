# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` covers contributor conventions (style, commit/PR norms, smoke-test expectations). This file focuses on what the code *is* and the workflows that span multiple files. Read both.

## What this project is

A toy model of **AI treaty verification**: can a monitoring body detect an unauthorized frontier-scale training run on a compute cluster it oversees? The full scenario and simulation mapping live in `verification_games_implementation_plan.md`.

The mapping that explains the code's vocabulary:

- **Blue team (monitor)** = `detect_flops.py`, a daemon that estimates how many FLOPs a GPU workload performed using *only* externally observable hardware signals (power draw, GPU/EMC utilization, clock frequency) — never the workload's own code.
- **Red team (workload)** = `sample_ml_workload.py`, a PyTorch transformer training run that emits a *ground-truth* FLOP count (via `torch.utils.flop_counter.FlopCounterMode`) so the blue team's blind estimate can be scored against truth.

The research question is the gap between the blue team's estimate and the red team's ground truth. Keep that adversarial framing in mind: the monitor must not "cheat" by reading anything the workload reports.

## Hardware assumptions (everything is wired to one device)

NVIDIA Jetson Orin Nano 8GB, JetPack 6.2 / L4T R36.5.0, 1024-core Ampere GPU. The code hard-codes device specifics; do not assume portability:

- Peak FLOPS/bandwidth and frequencies live in `ORIN_PROFILE` (`detect_flops.py`). Default precision is **TF32** (`ASSUMED_PRECISION`) because PyTorch on Ampere uses TF32 matmul unless disabled.
- Power comes from the on-board **INA3221** sensor, `VDD_CPU_GPU_CV` rail (CPU+GPU+CV cores combined — there is no GPU-only rail). The `hwmonN` index changes across reboots, so paths are resolved at runtime by label (`find_ina3221_paths`).
- **EMC (memory-bandwidth) utilization is read by scraping `tegrastats`**, not jtop: jtop 4.x on JetPack 6.2 returns 0 for EMC. See the comment on `get_emc_util` / `TegrastatsReader`.

## The two FLOP estimators in `detect_flops.py`

Both run simultaneously per workload session; the **power estimator is primary**, roofline is a diagnostic.

1. **Power/energy estimator (primary).** Integrates net power (measured − idle baseline) over the workload into joules, then converts energy to FLOPs with a **2-parameter active-energy model** (`estimate_tflops`): `TFLOPs = (E_net − POWER_OVERHEAD_W·t_active) / E_MARGINAL_J_PER_TFLOP`. The fixed-overhead term is what makes small workloads *look* less efficient (high apparent J/TFLOP) without needing a separate constant per intensity. `estimate_tflops` takes optional param overrides so `eval_power_monitor.py` can score candidate fits through the exact production code path.
2. **Roofline estimator (diagnostic).** `min(SM-busy compute ceiling, memory-bandwidth ceiling)` using `RIDGE_AI` (the FLOPs/byte ridge point). Labeled `bound_type` = compute/memory/unknown.

Session lifecycle: the daemon polls ~every 1.5 s, declares a workload **started** after `START_ACTIVE_POLLS` consecutive samples above `ACTIVE_GPU_UTIL_THRESHOLD`% GPU util, and **ended** after `STOP_QUIET_POLLS` quiet samples. Idle baseline power uses the calibrated constant (`current_idle_baseline_mw` returns `FALLBACK_IDLE_POWER_MW`) for run-to-run repeatability; live quiet samples are kept only as a diagnostic. Per-poll rows go to `flop_log`; one summary row per session goes to `workload_sessions`.

## The calibration loop (how the magic constants are produced)

The `POWER_OVERHEAD_W` / `E_MARGINAL_J_PER_TFLOP` / `FALLBACK_IDLE_POWER_MW` constants in `detect_flops.py` are **fit by the eval/calibration scripts, then manually copied back into the source.** This is the most important cross-file workflow:

1. `eval_power_monitor.py` runs a broad pool of multi-minute transformer configs, samples INA3221 power offline at 2 Hz, and fits the model (`E_net = a·TFLOPs + b·t_active → a=E_MARGINAL, b=POWER_OVERHEAD`) by **relative least squares**, choosing `POWER_OVERHEAD` by **leave-one-out cross-validation** (frontier runs are near-collinear in (TFLOPs, t), so any in-sample objective just chases the search bound; LOO has a genuine interior optimum). It scores runs against `FlopCounterMode` ground truth via `detect_flops.estimate_tflops` and reports per-workload error plus a `PASS/FAIL: held-out max err < 10%` verdict. Three choices mirror a real verifier: idle power is measured **once at startup** (no clean re-baseline between jobs); the **TRAIN/TEST split is randomized each run** (pass `--seed` to reproduce); and **only frontier-like runs are scored** — those with avg GPU util ≥ `FRONTIER_MIN_GPU_UTIL` (80%), the regime of an actual unauthorized frontier training run. Sub-frontier runs are still executed and shown (so the boundary is visible) but excluded from the fit and verdict, because on this device the shared CPU+GPU rail and absent EMC signal make a partially-loaded GPU's energy unattributable to FLOPs. The deployed constants come from a `SHIP FIT` refit on **all frontier runs**, and `FALLBACK_IDLE_POWER_MW` is a **matched set** with them (the same single baseline that produced the fit) — recalibrate both together.
2. Copy the printed `RECOMMENDED CONSTANTS` block into the dated comment block in `detect_flops.py` and re-run to confirm.

`calibrate_power.py` is the older single-purpose calibrator (no train/test split, prints a `POWER_CAL_*`-style block); `eval_power_monitor.py` is the current entry point for both measuring accuracy and producing constants, and imports `calibrate_power`'s sampling helpers (`run_workload`, `PowerSampler`, `sample_idle`).

**Subprocess timing caveat:** workloads are launched with `python -u` and read via `readline` so the `"[redteam] Starting workload..."` trigger (which gates the power sampler, excluding CUDA init/warmup from the energy integral) arrives promptly. Without unbuffered I/O, piped stdout block-buffers and the sampler starts late on short runs — the race that corrupted the original calibration (3–32 samples). Preserve `-u` + `readline` if you touch `run_workload`.

## SQLite schema is migrated in-place

`init_db` creates tables if absent and uses `add_column_if_missing` to additively migrate existing DBs. **Never reorder or rename columns** — add new ones through that helper so existing `/var/log/flop_log.db` files keep working.

## Commands

Always use the project venv interpreter. Under `sudo`, `sys.executable` is the system Python and lacks the venv's bundled CUDA libs — `calibrate_power.py` handles this via `find_venv_python`; do the same in new code.

```bash
# Red-team workload (prints ground-truth TFLOPs)
.venv/bin/python3 sample_ml_workload.py --steps 150 --batch-size 8 --seq-len 64 --d-model 128

# Blue-team monitor daemon (writes /var/log/flop_log.db; needs sensor + DB write perms)
.venv/bin/python3 detect_flops.py

# Accuracy evaluation + constant fitting (transformer sweep, ~20 min; PASS/FAIL vs 10%)
.venv/bin/python3 eval_power_monitor.py --output eval_results.txt

# Full calibration (older single-purpose calibrator; 40–90 min, NO other GPU work)
.venv/bin/python3 calibrate_power.py --output calibration_results.txt
```

`eval_power_monitor.py` is the primary accuracy/calibration tool: it sweeps transformer configs, gates to frontier-like (≥80% GPU util) runs, takes a randomized TRAIN/TEST split, scores `estimate_tflops` against ground truth, and prints a `RECOMMENDED CONSTANTS` block (plus the matched idle baseline) to paste into `detect_flops.py`. As of 2026-06-22 the shipped fit holds all 11 frontier configs to ≤7.0% error (held-out ≤8.1%); sub-frontier runs are out of scope (a 65%-util run came out 35% low) — the estimator is a *frontier* detector by design.

Smoke tests (no formal framework — these are the diagnostics; run before changing power or subprocess code):

```bash
.venv/bin/python3 power_monitor_test.py   # can the INA3221 rail be found + sampled?
.venv/bin/python3 test_subprocess.py      # sensor perms + workload subprocess launch
```

Typical validation: start `detect_flops.py`, run `sample_ml_workload.py` in another shell, compare the monitor's "Power estimate" to the workload's "Ground truth total".

## Note on `detect_flops_old.py`

A prior roofline-only version kept for reference. `detect_flops.py` is the current entry point; don't edit the old file unless explicitly asked.
