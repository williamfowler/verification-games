# Repository Guidelines

## Project Structure & Module Organization

This repository is a Jetson-focused Python experiment for estimating GPU FLOPs from power and utilization signals.

- `detect_flops.py`: main monitor; reads `jtop`, the actmon DRAM counters, and the INA3221 power sensor, and writes session data to SQLite.
- `sample_ml_workload.py`: CUDA/PyTorch transformer workload used as a repeatable FLOP source.
- `eval_power_monitor.py`: primary accuracy-evaluation and calibration tool; sweeps workloads, fits the power model, and prints the constants to paste into `detect_flops.py`.
- `power_calibration/calibrate_power.py`: sampling library (power + DRAM-bytes samplers, workload runner) used by the eval and probe scripts; not a standalone tool.
- `power_calibration/power_monitor_test.py` and `misc_debug/test_subprocess.py`: hardware and subprocess diagnostics.
- `old/power_calibration.txt`: archived calibration notes/results (see `old/` for other superseded files).
- `verification_games_implementation_plan.md`: design and implementation notes.

Main entry points live at the repo root; calibration helpers live in `power_calibration/` and one-off diagnostics in `misc_debug/`. Avoid committing generated caches such as `__pycache__/`.

## Build, Test, and Development Commands

Always use the project venv interpreter (`.venv/bin/python3`) on a Jetson with CUDA, PyTorch, `jtop`, and `tegrastats`.

```bash
.venv/bin/python3 sample_ml_workload.py --steps 20 --batch-size 4 --seq-len 32 --d-model 128
```
Runs a short GPU workload and prints ground-truth FLOP estimates.

```bash
.venv/bin/python3 power_calibration/power_monitor_test.py
```
Checks that the INA3221 `VDD_CPU_GPU_CV` rail can be located and sampled.

```bash
.venv/bin/python3 misc_debug/test_subprocess.py
```
Verifies sensor permissions and direct workload subprocess execution.

```bash
sudo .venv/bin/python3 detect_flops.py
.venv/bin/python3 eval_power_monitor.py --output eval_results.txt
```
Runs the monitor (root required for the actmon DRAM counters) or the accuracy eval / constant fit. The eval sweep takes ~60-70 minutes; run it without other GPU workloads.

## Coding Style & Naming Conventions

Follow the existing Python style: 4-space indentation, `snake_case` functions and variables, uppercase constants for hardware paths and calibration values, and short docstrings for public helpers. Prefer standard-library modules unless a hardware or ML dependency is already used. Keep comments factual around empirical calibration constants.

## Testing Guidelines

There is no formal test framework in this repo. Treat the diagnostic scripts as smoke tests before changing monitor or calibration behavior. For changes to power sampling, run `.venv/bin/python3 power_calibration/power_monitor_test.py`; for subprocess or workload changes, run `.venv/bin/python3 misc_debug/test_subprocess.py` and a short `sample_ml_workload.py` run. For changes to the fitting code, replay an existing sweep offline with `eval_power_monitor.py --refit-from <records.json> --seed N` and confirm the fitted constants are unchanged. Document any hardware limitations if tests cannot be run locally.

## Commit & Pull Request Guidelines

Recent commits use short, lowercase summaries such as `added calibrate_power.py...` and `improved detect_flops...`. Keep subjects concise and behavior-focused. Pull requests should describe the measurement or workflow change, list commands run, note Jetson model/JetPack version when relevant, and include before/after calibration numbers or log snippets for user-visible output.

## Security & Configuration Tips

Do not hard-code secrets or user-specific paths. Be careful with writes to `/var/log/flop_log.db` and reads from `/sys/bus/i2c/...`; permission differences between root and non-root runs are expected and should be documented.
