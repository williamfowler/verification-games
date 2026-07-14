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

- Peak FLOPS/bandwidth and frequencies live in `ORIN_PROFILE` (`detect_flops.py`). **Precision correction (2026-07-07):** this build's torch (2.9.1) defaults `torch.backends.cuda.matmul.allow_tf32 = False`, so despite older docs claiming TF32, all pre-2026-07 calibration ran **FP32 matmul**. `sample_ml_workload.py --precision {fp32,tf32,fp16,bf16}` now controls this explicitly (fp32 = the historical default); `ASSUMED_PRECISION` in `detect_flops.py` feeds only the roofline *diagnostic*.
- The constants in `detect_flops.py` are matched to one device **and one nvpmodel power mode** (currently 25W "Super"); `CALIBRATION_FINGERPRINT` records which, and the daemon prints a loud RECALIBRATE banner at startup when the live device/L4T/power-mode differs. On any new hardware or after `nvpmodel -m`, re-run `eval_power_monitor.py` and paste both the constants and the fingerprint — the *methodology* is portable, the numbers are not.
- Power comes from the on-board **INA3221** sensor, `VDD_CPU_GPU_CV` rail (CPU+GPU+CV cores combined — there is no GPU-only rail). The `hwmonN` index changes across reboots, so paths are resolved at runtime by label (`find_ina3221_paths`).
- **EMC (memory-bandwidth) is read from the actmon debugfs counters** (`ACTMON_PRD_PATH` etc. in `detect_flops.py`: `/sys/kernel/debug/bpmp/debug/actmon/mc_all_*` + `clk/emc/rate`), **root-only**. Both jtop (returns 0) and `tegrastats` (its `EMC_FREQ` field never matches on this JetPack 6.2 build) fail to expose EMC — the daemon's old tegrastats scrape and roofline memory-ceiling path were removed 2026-07-14 as dead code (their `flop_log` columns `tflops_mem_ceil`/`emc_util`/`bound_type` and `workload_sessions.avg_emc_util` remain in the schema, written as NULL/`"unknown"`). The actmon counter is the working signal, reported as **total DRAM traffic** (all clients, CPU+GPU — same attribution caveat as the shared power rail). `ActmonReader.read_bytes_per_s()` converts it via `actmon_bytes_per_s` (= `util_fraction · PEAK_BW_BYTES_S · emc_rate/EMC_MAX_HZ`); it **raises** when the counters are unreadable (e.g. not root) — the daemon hard-requires root, there is no silent 2-param fallback. Note `calibrate_power.py`'s tegrastats scrape is still live for **GPU utilization** (it feeds the eval's frontier gate); only its EMC field is dead there.

## The FLOP estimators in `detect_flops.py`

All run simultaneously per workload session; the **power estimators are primary**, roofline is a diagnostic. There are **two parallel power estimators** (A/B), both logged:

1. **Power/energy estimator — 2-param (primary, baseline).** Integrates net power (measured − idle baseline) over the workload into joules, then converts with a **2-parameter active-energy model** (`estimate_tflops`): `TFLOPs = (E_net − POWER_OVERHEAD_W·t_active) / E_MARGINAL_J_PER_TFLOP`. The fixed-overhead term is what makes small workloads *look* less efficient without a per-intensity constant. Logged with `estimator = "power_energy_v1"`.
2. **Power/energy estimator — 3-param EMC/bytes (parallel A/B).** `estimate_tflops_emc` adds a measured memory-energy term using integrated DRAM bytes: `TFLOPs = (E_net − E_PER_TB_J·TB_moved − POWER_OVERHEAD_W·t_active) / E_MARGINAL`. `TB_moved` is observed (actmon), not fitted; with `TB_moved=0`/`E_PER_TB_J=0` it reduces to estimator 1. Motivation is **adversarial** — the byte term catches a red-team operating off the normal FLOPs↔bytes line. Runs in parallel (does NOT replace estimator 1); logged with `estimator_emc = "power_energy_emc_v1"` and `tb_moved` on the same session row. It uses its OWN matched constants (`POWER_OVERHEAD_EMC_W`, `E_MARGINAL_EMC_J_PER_TFLOP`, `E_PER_TB_J`), distinct from the 2-param pair — the daemon shares one `FALLBACK_IDLE_POWER_MW` between both, so all constants are one matched set (recalibrate together). **Role split (2026-07-07): 2-param is the meter, EMC is a spoof tripwire** — it decisively flags a memory-spoof (see `adversarial_results.txt`) but is less accurate than 2-param on benign data; treat its output as a flag, not a count. The actmon byte scale was measured (`power_calibration/actmon_scale_bench.py`): k = 0.0133 actmon-TB per true TB, making the fitted `E_PER_TB_J` ≈ 61.5 J per true TB (physically plausible for LPDDR5).
3. **Roofline estimator (diagnostic).** SM-busy **compute ceiling only**: `peak_TFLOPS(ASSUMED_PRECISION) · freq_ratio · gpu_util`. The old memory-bandwidth ceiling depended on the dead tegrastats `emc_util` signal and was removed 2026-07-14; the actmon bytes feed estimator 2, **not** this roofline path.

`estimate_tflops` is implemented as `estimate_tflops_emc` with a zero byte term — there is **one** energy-model implementation. Both take optional param overrides so `eval_power_monitor.py` can score candidate fits through the exact production code path, and both return the **raw model value, which can be ≤ 0** for sub-frontier workloads (overhead term exceeds net energy) — that means "below the frontier detection floor", not an error; `None` only on degenerate inputs. The per-poll loop keeps incremental `power_tflops_delta` / `power_emc_tflops_delta` columns via the shared `_tflops_delta` helper, so their session sums reconcile with the estimator totals by construction — change the model in `estimate_tflops_emc` (and `_tflops_delta`) only.

Session lifecycle: the daemon polls ~every 1.5 s, declares a workload **started** after `START_ACTIVE_POLLS` consecutive samples above `ACTIVE_GPU_UTIL_THRESHOLD`% GPU util, and **ended** after `STOP_QUIET_POLLS` quiet samples. Per-session accumulators live in the `Session` dataclass (`session is None` doubles as the "no workload tracked" state). Idle baseline power always uses the calibrated constant `FALLBACK_IDLE_POWER_MW` for run-to-run repeatability; live quiet samples are kept only as a diagnostic. Per-poll rows go to `flop_log`; one summary row per session goes to `workload_sessions`.

## The calibration loop (how the magic constants are produced)

The `POWER_OVERHEAD_W` / `E_MARGINAL_J_PER_TFLOP` / `FALLBACK_IDLE_POWER_MW` constants in `detect_flops.py` are **fit by the eval/calibration scripts, then manually copied back into the source.** This is the most important cross-file workflow:

1. `eval_power_monitor.py` runs a broad pool of multi-minute transformer configs, samples INA3221 power offline at 2 Hz, and fits the model (`E_net = a·TFLOPs + b·t_active → a=E_MARGINAL, b=POWER_OVERHEAD`) by **relative least squares**, choosing `POWER_OVERHEAD` by **leave-one-out cross-validation** (frontier runs are near-collinear in (TFLOPs, t), so any in-sample objective just chases the search bound; LOO has a genuine interior optimum). It scores runs against `FlopCounterMode` ground truth via `detect_flops.estimate_tflops` and reports per-workload error plus a `PASS/FAIL: held-out max err < 10%` verdict. Three choices mirror a real verifier: idle power is measured **once at startup** (no clean re-baseline between jobs); the **TRAIN/TEST split is randomized each run** (pass `--seed` to reproduce); and **only frontier-like runs are scored** — those with avg GPU util ≥ `FRONTIER_MIN_GPU_UTIL` (80%), the regime of an actual unauthorized frontier training run. Sub-frontier runs are still executed and shown (so the boundary is visible) but excluded from the fit and verdict, because on this device the shared CPU+GPU rail and absent EMC signal make a partially-loaded GPU's energy unattributable to FLOPs. The deployed constants come from a `SHIP FIT` refit on **all frontier runs**, and `FALLBACK_IDLE_POWER_MW` is a **matched set** with them (the same single baseline that produced the fit) — recalibrate both together.
2. Copy the printed `RECOMMENDED CONSTANTS` block into the dated comment block in `detect_flops.py` — **including the matched `CALIBRATION_FINGERPRINT`** it prints — and re-run to confirm.

**Replay/refit without a sweep.** Every sweep dumps its per-run records to `<output stem>_records.json`; `--refit-from FILE` (JSON, or an older report's `RAW RECORDS` section) reruns gate → split → fit → report offline in seconds. `--split holdout:<key><op><val>` (e.g. `holdout:num_layers=6`, `holdout:d_model>=640`, `holdout:precision=fp16`) tests extrapolation to an entire held-out config family; `--stability N` reports the spread of fitted constants and held-out errors over N resampled splits. See `eval_generalization.txt` for both sweeps' analyses (headline: 13.5% of random splits fail the 10% target on the v2 fp32 pool — 23% on v1 — and `E_PER_TB_J` spans 1.2k–6.9k J/TB across splits; treat single-split PASSes and the byte coefficient as soft).

**EMC/3-param fit (parallel).** When actmon `tb_moved` is captured for every frontier run, the eval also fits the 3-param model via `fit_active_energy_emc_model` — same LOO-over-`POWER_OVERHEAD` grid, but each inner step solves **two** linear coefficients (`E_MARGINAL`, `E_PER_TB`) by a 2x2 relative-weighted (1/gt²) least squares (`_linear2_at`), falling back to the 1-coefficient solve when the 2x2 is near-singular. It scores `estimate_tflops_emc` alongside the 2-param estimator for A/B (the report shows both, plus a `corr(gt,tb)`/condition diagnostic and an `E_PER_TB_J` line). Capturing `tb_moved` requires the **privileged actmon reader**: `power_calibration/actmon_reader.py` streams the root-only counters, spawned by `BytesSampler` via `sudo -n` — so a **NOPASSWD sudoers entry** for that script is a hard prerequisite (the eval itself must stay non-root for CUDA). Without it the eval raises (`run_workload` requires actmon samples; a frontier run missing `tb_moved` is a hard error) — there is no silent 2-param-only fallback. `E_PER_TB` is fit blind against `FlopCounterMode` FLOP truth (no ground-truth bytes yet); the byte term stays active at inference regardless of benign-data collinearity (adversarial design) — collinearity only affects whether `E_PER_TB` is well-*determined*, which the diagnostic flags.

`power_calibration/calibrate_power.py` is a **sampling library, not a standalone tool** (its legacy single-purpose calibrator mode was removed 2026-07-14; its last report is `old/power_calibration.txt`). `eval_power_monitor.py` is the sole entry point for measuring accuracy and producing constants; it imports `calibrate_power`'s sampling helpers (`run_workload`, `sample_idle`, `PowerSampler`/`BytesSampler`). The canonical INA3221 helpers (`find_ina3221_paths`, `read_power_mw`) live in `detect_flops.py`; `calibrate_power` re-exports them for its consumers.

**Subprocess timing caveat:** workloads are launched with `python -u` and read via `readline` so the `"[redteam] Starting workload..."` trigger (which gates the power sampler, excluding CUDA init/warmup from the energy integral) arrives promptly. Without unbuffered I/O, piped stdout block-buffers and the sampler starts late on short runs — the race that corrupted the original calibration (3–32 samples). Preserve `-u` + `readline` if you touch `run_workload`.

## SQLite schema is migrated in-place

The schema lives in one ordered `SCHEMA` dict in `detect_flops.py`, from which `init_db` both creates fresh tables and additively migrates existing DBs (`add_column_if_missing` for every column, idempotent). **Never reorder or rename columns — append only** — so existing `/var/log/flop_log.db` files keep working. Columns from removed features (`tflops_mem_ceil`, `emc_util`, `bound_type`, `avg_emc_util`) are retained and written as NULL/`"unknown"` for compatibility.

## Commands

Always use the project venv interpreter. Under `sudo`, `sys.executable` is the system Python and lacks the venv's bundled CUDA libs — `calibrate_power.py` handles this via `find_venv_python`; do the same in new code.

```bash
# Red-team workload (prints ground-truth TFLOPs)
.venv/bin/python3 sample_ml_workload.py --steps 150 --batch-size 8 --seq-len 64 --d-model 128

# Blue-team monitor daemon (writes /var/log/flop_log.db; needs sensor + DB write perms).
# MUST run as root: the actmon DRAM-bytes read raises without it (no 2-param fallback).
sudo .venv/bin/python3 detect_flops.py

# Accuracy evaluation + constant fitting (transformer sweep, ~60-70 min; PASS/FAIL vs 10%)
.venv/bin/python3 eval_power_monitor.py --output eval_results.txt
```

`eval_power_monitor.py` is the primary accuracy/calibration tool: it sweeps transformer configs, gates to frontier-like (≥80% GPU util) runs, takes a randomized TRAIN/TEST split, scores `estimate_tflops` against ground truth, and prints a `RECOMMENDED CONSTANTS` block (plus the matched idle baseline and fingerprint) to paste into `detect_flops.py`. As of 2026-07-07 (v2 sweep) the shipped fit holds all 21 **fp32** frontier configs to ≤8.8% (held-out ≤8.95%); the pool is **fp32-scoped** — tf32/fp16/bf16 under-read 2–3.5× (precision is unobservable from power; estimates are a lower bound w.r.t. precision, see `PROJECT_STATUS.md`). Sub-frontier runs are out of scope — the estimator is a *frontier* detector by design.

Smoke tests (no formal framework — these are the diagnostics; run before changing power or subprocess code):

```bash
.venv/bin/python3 power_calibration/power_monitor_test.py  # can the INA3221 rail be found + sampled?
.venv/bin/python3 misc_debug/test_subprocess.py             # sensor perms + workload subprocess launch
```

Typical validation: start `detect_flops.py`, run `sample_ml_workload.py` in another shell, compare the monitor's "Power estimate" to the workload's "Ground truth total".

## Note on `old/`

Superseded files live in `old/` — `detect_flops_old.py` (prior roofline-only monitor), `eval_results.txt` (the 2026-06-30 v1 sweep report; still replayable via `--refit-from old/eval_results.txt`), and `power_calibration.txt` (a 2026-05-28 report from the legacy calibrator). Don't edit them unless explicitly asked; current reports are `eval_results_v2.txt` / `eval_results_15w.txt` (+ `_records.json`).
