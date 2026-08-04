# Red-Team Adversarial Workloads (Phase II)

Adversarial workloads a non-compliant party runs to drive the Phase-I calibrated
FLOP estimator's error up, plus the harness and scoring that quantify each attack
against Phase I's benchmark. Dual-V100 node, `floppy` branch.

## Two invariants (never break these)

1. **Ground truth stays honest.** Every strategy still prints the true
   `[redteam] Ground truth total : X TFLOPs` from `FlopCounterMode`. The attack
   lowers the blue team's *estimate*, never the reported GT — the GT is the
   yardstick error is measured against. Guarded by `test_benign_equivalence.py`.
2. **2× GPU-hour budget.** An attack may not spend more than 2× the GPU-hours a
   benign run of the same GT-FLOP count would (`efficiency.py`). An attack that
   halves the estimate but costs 5× GPU-hours is a **failed** attack and is
   reported as such. Efficiency ratio = `R_benign / R_adv`; legal iff ≤ 2.0.

## Strategies

| ID | strategy | signal gamed | group / path | file |
|----|----------|--------------|--------------|------|
| S1 | precision (fp32 twins) | energy-per-FLOP | A · offline | `redteam_configs.py` |
| S2 | memory decoy | DRAM-bytes term (subtracted) | A · offline | `adversarial_workload.py --strategy mem_decoy` |
| S2′ | NVLink decoy **[V100]** | interconnect term / tripwire | A · offline | `... mem_decoy --decoy-allreduce` |
| S3 | atypical hyperparameters | fit extrapolation | A · offline | `redteam_configs.py` |
| S4 | workload splitting | daemon start/stop detection | B · live daemon | `... --strategy split` |
| S5 | sub-gate throttling | 80% frontier gate | B · live daemon | `... --strategy throttle` |

**Group A** attacks a cleanly-bracketed offline run and is scored through
`eval_power_monitor.py` on the same window as Phase I. **Group B** attacks the
live daemon's *session logic* (which the offline eval bypasses) and must be run
against `detect_flops.py` via `live_daemon_probe.py`.

### S1 caveat — the precision channel is *closed* on Volta (honest result)

The classic precision attack (run a higher-throughput precision than the
calibration assumes, so measured energy-per-true-FLOP drops → under-report) has
**no target here**: on Volta fp16 is already both the DDP baseline *and* the
max-throughput precision (no bf16/tf32/fp8), and Phase I is fp16-calibrated. The
only other precision, fp32, has *higher* energy-per-FLOP, so an fp32 run reads
**high** (over-report) — it can't hide FLOPs. S1's fp32 configs quantify that
precision-blindness (the magnitude of the fp32 over-estimate), and the honest
finding is that the underreport-via-precision channel is closed on this hardware.
On an Ampere+ node with a tf32/bf16 baseline it would reopen.

## Prerequisites

- **`nv-hostengine` must be running** (root) for DCGM DRAM-active (field 1005) and
  NVLink counters — needed by both the offline sampler and the live daemon:
  `! sudo nv-hostengine`. Power/util (nvidia-smi) are unprivileged.
- Both V100s free (each run is DDP over both). The 10-trial Phase-I watchdog infra
  (`run_trials.py` / `trial_guard.sh`) can be reused to babysit a long collection.

## Run order

```bash
# 0. sanity — GT honesty guard (no hostengine needed)
python3 red_team/test_benign_equivalence.py

# 1. efficiency reference (R_benign from the Phase I 10 trials)
python3 red_team/efficiency.py

# 2. Group A collection — reuse the Phase I sweep with the red config list.
#    Repeat into per-trial JSONs for 10-trial distributions (mirror Phase I).
python3 eval_power_monitor.py \
    --configs-module red_team.redteam_configs:RED_CONFIGS \
    --records-json red_team/red_results_trial1_records.json --output /tmp/red1.txt
#    (the sweep dumps records before its own fit/verdict; the FAIL verdict is
#     expected — the offline fit pools mixed strategies and is not used. Scoring
#     is done separately against the FROZEN Phase I calibration, below.)

# 3. Group B collection — live daemon, one record per strategy + a none baseline
python3 red_team/live_daemon_probe.py --strategy none    --out red_team/live_none.json
python3 red_team/live_daemon_probe.py --strategy split   --chunk-steps 100 --gap-seconds 6 --steps 400 --out red_team/live_split.json
python3 red_team/live_daemon_probe.py --strategy throttle --target-util 70 --steps 400 --out red_team/live_throttle.json

# 4. Score everything against the FROZEN Phase I calibration + 2× gate; make figs
python3 red_team/score_redteam.py \
    --group-a red_team/red_results_trial1_records.json \
    --group-b red_team/live_none.json red_team/live_split.json red_team/live_throttle.json
```

## Scoring model (do not launder the attack)

Calibration is **frozen at Phase I**: constants are fit on the Phase I benign fp16
frontier and adversarial runs are scored against them — never recalibrated
including adversarial runs (that is the Phase III anti-pattern). `score_redteam.py`
reports, per strategy, the signed estimator error under both the 2-param
(power-only) and 3-param (power+DRAM) estimators, paired with the efficiency ratio
and the 2× verdict, and emits `writeup/fig_redteam_error.png` and
`writeup/fig_redteam_efficiency.png`.

Note the estimator asymmetry the DRAM decoy exposes: inflating DRAM bytes makes
the **3-param** estimator (which subtracts `E_per_TB·TB_moved`) read low, but the
extra copies also burn power, so the **2-param** (power-only) estimator reads
*high*. Phase I found power-only is the more accurate benign estimator — so the
decoy that beats one estimator backfires on the other. That tension is the Phase
III hook.

## Files

- `adversarial_workload.py` — one entry point, `--strategy {none,mem_decoy,split,throttle}`; imports the benign core and interposes only via `run_training`'s no-op hooks.
- `redteam_configs.py` — `RED_CONFIGS` (Group A: S1/S2/S3), fed via `eval_power_monitor.py --configs-module`.
- `efficiency.py` — GPU-hour accounting + the 2× budget gate; `R_benign`.
- `live_daemon_probe.py` — Group B: drives `detect_flops.py`, reads back session estimates.
- `score_redteam.py` — per-strategy error vs Phase I, efficiency table, the two figures.
- `test_benign_equivalence.py` — CI guard for GT honesty.
