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

- Jetson Orin Nano 8GB developer kit, JetPack 6.2 (L4T R36.5.0), in the
  default 25W power mode. Everything is device-specific: sensor paths, debugfs
  counters, and the calibrated constants.
- Python 3.10 with torch 2.9.1 **from the Jetson AI Lab wheel index**
  (standard PyPI wheels are not built for the Orin's sm_87 GPU and fail at
  runtime). Other deps are in `requirements.txt`.
- Root access: the monitor daemon runs as root (the actmon counters are
  root-only debugfs), and calibration needs a NOPASSWD sudoers entry for the
  small reader script it uses to reach those counters without running the
  whole sweep as root.

```bash
./setup.sh   # apt CUDA pieces, venv + wheels, LD_LIBRARY_PATH, sudoers entry, sanity check
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
(logs to `/var/log/flop_log.db`):

```bash
sudo .venv/bin/python3 detect_flops.py
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
| `power_calibration/calibrate_power.py` | shared sampling library (power, DRAM bytes, GPU util) |
| `power_calibration/actmon_reader.py` | root-only DRAM counter reader (via sudoers) |
| `power_calibration/actmon_scale_bench.py` | actmon byte-scale calibration |
| `power_calibration/power_monitor_test.py` | INA3221 sanity check |
| `eval_results_*_records.json` | per-run records from the report's three sweeps |
| `writeup/` | the report (PDF), its figures, and the scripts that make them |
