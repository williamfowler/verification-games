# Repository Guidelines

## Project Structure & Module Organization

This repository is a Jetson-focused Python experiment for estimating GPU FLOPs from power and utilization signals.

- `detect_flops.py`: main monitor; reads `jtop`, `tegrastats`, INA3221 power sensors, and writes session data to SQLite.
- `sample_ml_workload.py`: CUDA/PyTorch transformer workload used as a repeatable FLOP source.
- `calibrate_power.py`: long-running calibration harness that compares measured energy against workload FLOP counts.
- `power_monitor_test.py` and `test_subprocess.py`: hardware and subprocess diagnostics.
- `power_calibration.txt`: checked-in calibration notes/results.
- `verification_games_implementation_plan.md`: design and implementation notes.

There is no package directory yet; keep new scripts at the repo root unless reusable modules emerge. Avoid committing generated caches such as `__pycache__/`.

## Build, Test, and Development Commands

Use Python 3 on a Jetson with CUDA, PyTorch, `jtop`, and `tegrastats`.

```bash
python3 sample_ml_workload.py --steps 20 --batch-size 4 --seq-len 32 --d-model 128
```
Runs a short GPU workload and prints ground-truth FLOP estimates.

```bash
python3 power_monitor_test.py
```
Checks that the INA3221 `VDD_CPU_GPU_CV` rail can be located and sampled.

```bash
python3 test_subprocess.py
```
Verifies sensor permissions and direct workload subprocess execution.

```bash
python3 detect_flops.py
python3 calibrate_power.py --output calibration_results.txt
```
Runs the monitor or full calibration. Calibration can take 40-90 minutes; run it without other GPU workloads.

## Coding Style & Naming Conventions

Follow the existing Python style: 4-space indentation, `snake_case` functions and variables, uppercase constants for hardware paths and calibration values, and short docstrings for public helpers. Prefer standard-library modules unless a hardware or ML dependency is already used. Keep comments factual around empirical calibration constants.

## Testing Guidelines

There is no formal test framework in this repo. Treat the diagnostic scripts as smoke tests before changing monitor or calibration behavior. For changes to power sampling, run `python3 power_monitor_test.py`; for subprocess or workload changes, run `python3 test_subprocess.py` and a short `sample_ml_workload.py` run. Document any hardware limitations if tests cannot be run locally.

## Commit & Pull Request Guidelines

Recent commits use short, lowercase summaries such as `added calibrate_power.py...` and `improved detect_flops...`. Keep subjects concise and behavior-focused. Pull requests should describe the measurement or workflow change, list commands run, note Jetson model/JetPack version when relevant, and include before/after calibration numbers or log snippets for user-visible output.

## Security & Configuration Tips

Do not hard-code secrets or user-specific paths. Be careful with writes to `/var/log/flop_log.db` and reads from `/sys/bus/i2c/...`; permission differences between root and non-root runs are expected and should be documented.
