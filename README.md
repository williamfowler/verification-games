# Estimating LLM Training FLOPs on the Nvidia Jetson Orin Nano

Compute thresholds (10^25–10^26 training FLOPs) are becoming the legal
definition of a "frontier" AI model, but today the FLOP counts are
self-reported. This project is an MVP of the alternative: estimate the FLOPs of
a training run from the outside, using only system-level hardware signals, with
no access to the workload's code. A monitoring daemon watches two signals on a
Jetson Orin Nano — total board power (the onboard INA3221 sensor) and DRAM
activity (the Tegra actmon counters) — detects when a workload starts and
stops, and converts the net energy it used into a FLOP estimate. The estimate
is scored against ground truth from PyTorch's `FlopCounterMode`, which the
sample workload reports but the monitor never sees.

The full report is in `writeup/` (PDF). Headline result: on 21 "frontier"
transformer training workloads (GPU busy ≥ 80%), the calibrated estimator is
within 10% of ground truth on held-out workloads.

## Reproduced on dual-V100 (x86)

This tree has been **ported from the Jetson Orin Nano to an x86 dual-Tesla-V100
server** (Ubuntu 24.04, driver 580 / CUDA 13, standard PyPI torch). The
estimator math, calibration/fitting, and figure scripts are unchanged; only the
signal-acquisition layer was re-sourced (see the top of `detect_flops.py`):

| Signal | Jetson source | V100 substitute |
|---|---|---|
| Board power | INA3221 `VDD_CPU_GPU_CV` rail | `nvidia-smi power.draw` for the monitored GPU |
| GPU utilization (frontier gate) | `tegrastats GR3D_FREQ` | `nvidia-smi utilization.gpu` |
| DRAM activity → `TB_moved` | Tegra `actmon` bandwidth fraction | **DCGM field 1005 `DCGM_FI_PROF_DRAM_ACTIVE`** via `dcgmi dmon` |

Two device-driven changes are worth calling out:
- **DCGM prerequisite.** The DRAM-active counter is a profiling metric read by
  unprivileged `dcgmi` clients through a **root `nv-hostengine`** (start once with
  `sudo nv-hostengine`). This mirrors the Jetson design, where only the DRAM
  reader was privileged; power/util stay unprivileged.
- **Frontier gate retuned.** These toy transformers are launch/sync-bound on a
  V100 (far more capable than the Orin Nano), so none reach the Jetson's ≥80%
  `utilization.gpu`. To keep the *same workloads* and the same frontier /
  sub-frontier partition, `FRONTIER_MIN_GPU_UTIL` in `eval_power_monitor.py` is
  retuned to the V100 utilization regime; the reproduced figures reflect V100
  data and V100-fit constants (the constants in `detect_flops.py` are re-fit
  here, never reused from the Jetson — as the note below already requires).

The estimator:

```
TFLOPs = (E_net − E_per_TB · TB_moved − P_overhead · t) / E_per_TFLOP
```

`E_net` is measured power minus the idle baseline, integrated over the
workload; `TB_moved` is DRAM traffic integrated from actmon. `E_per_TFLOP`,
`E_per_TB`, and `P_overhead` are constants fit during a calibration phase and
are specific to one device and power mode — reproducing on other hardware means
re-running calibration, not reusing the numbers in `detect_flops.py`.

## Requirements

*(V100 port — the original Jetson requirements are in git history.)*

- x86 server with an NVIDIA GPU (developed on 2× Tesla V100-SXM2-16GB),
  Ubuntu 24.04, driver 580 / CUDA 13. The calibrated constants are device-specific
  and are re-fit here, not reused from the Jetson.
- Python 3.12 with a standard PyPI `torch` (the cu12x wheels include sm_70 and
  run on the V100). Other deps are in `requirements.txt`.
- **DCGM** (`datacenter-gpu-manager` / `dcgmi` + `nv-hostengine`) for the
  DRAM-activity signal. Start the host engine once as root — `sudo nv-hostengine`
  — after which the unprivileged samplers read it. Everything else (power, GPU
  util, the monitor daemon) runs unprivileged.

```bash
./setup.sh   # venv + PyPI wheels, DCGM host-engine check, sensor sanity check
```

## Reproducing the experiment

**1. Calibrate.** Runs a sweep of ~26 transformer training configs (~60–70
min), measures the idle power baseline at startup, fits the estimator
constants to the runs that cleared the 80% GPU-utilization gate, and scores
them on a held-out split:

```bash
.venv/bin/python3 eval_power_monitor.py --output eval_results.txt
```

Paste the printed `RECOMMENDED CONSTANTS` block (including the idle baseline
and calibration fingerprint) into the marked section of `detect_flops.py` —
the constants are a matched set, always replace all of them together.

Optional: measure the actmon byte-scale factor (makes the fitted `E_per_TB`
physically interpretable; it's otherwise absorbed into the fit):

```bash
.venv/bin/python3 power_calibration/actmon_scale_bench.py
```

**2. Run the monitor against a workload.** In one shell start the daemon
(logs to `./flop_log.db`; runs unprivileged — no sudo needed, but
`nv-hostengine` must be up):

```bash
.venv/bin/python3 detect_flops.py
```

In another, run a sample training workload:

```bash
.venv/bin/python3 sample_ml_workload.py --steps 150 --batch-size 8 --seq-len 64 --d-model 128
```

The workload prints its ground-truth TFLOPs; compare with the "Power estimate"
the daemon prints when the session ends.

**3. Figures.** `eval_power_monitor.py` writes its per-run records to
`*_records.json`; the repo's checked-in records (`eval_results_v2`,
`_25w_trial2`, `_25w_trial3`) are the three sweeps behind the report.

```bash
.venv/bin/python3 writeup/make_figs.py           # results figures from the records
.venv/bin/python3 writeup/capture_timeseries.py  # ~12 min: traces one workload -> timeseries.json
.venv/bin/python3 writeup/plot_timeseries.py     # power/bandwidth trace figures
```

## Files

| File | Role |
|---|---|
| `detect_flops.py` | the monitor daemon: sensors, workload detection, estimator, SQLite log |
| `sample_ml_workload.py` | configurable transformer training run; prints `FlopCounterMode` ground truth |
| `eval_power_monitor.py` | calibration sweep, constant fitting, accuracy eval |
| `power_calibration/calibrate_power.py` | shared sampling library (nvidia-smi power/util, DCGM DRAM bytes) |
| `power_calibration/actmon_reader.py` | Jetson-only DRAM reader (unused on V100; DCGM is read directly) |
| `power_calibration/actmon_scale_bench.py` | DRAM byte-scale calibration |
| `power_calibration/power_monitor_test.py` | nvidia-smi + DCGM sensor sanity check |
| `eval_results_*_records.json` | per-run records from the report's three sweeps |
| `writeup/` | the report (PDF), its figures, and the scripts that make them |
