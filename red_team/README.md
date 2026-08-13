# Red-Team Adversarial Workloads — Phase II (v3)

Deployment-phase adversarial workloads run against the **frozen benign-calibrated**
FLOP estimator, scored primarily for **under-reporting**. Dual-V100, `floppy`.
v3 supersedes v1 (retired artifacts in `../red_team_old/`).

## Invariants (enforced by `test_benign_equivalence.py`)

1. **Frozen benign calibration** — the **primary estimator is the 4-param form**
   `(E_net − E_PER_TB·TB − E_PER_NVLINK·NVLink − P_OH·t)/E_MARGINAL` (compute-energy,
   DRAM, NVLink, overhead — `detect_flops.estimate_tflops_nvl`), Phase I benign fp16
   fit `E_MARGINAL=3.54, E_PER_TB=93.4, E_PER_NVLINK=0.0, P_OH=0`. The NVLink
   coefficient fits to 0 (all-reduce collinear with DRAM on benign DDP), so it
   reduces numerically to power+DRAM, but it IS the scored estimator. The 2-param
   power-only fit (`E_MARGINAL=4.78`) is the reported accuracy ablation. No
   adversarial run ever enters a fit.
2. **One variable, GT preserved** — every config declares a benign `parent` and
   changes exactly one GT-invariant knob; `|GT_adv−GT_parent| < 1%` is asserted
   in-process (FLOP-count, no training). Legal one-variables: `nhead`, `optimizer`,
   `batch_size` (with `steps = P/batch` so batch×steps is constant), external
   power/clock caps, idle gaps / throttle sleeps.
3. **Honest GT every run**; the attack moves the estimate, never the reported GT.
4. **2× GPU-hour budget** — `R_benign/R_adv ≤ 2` (`R_benign = 38,540 TF/GPU-h`);
   an under-report that costs >2× the GPU-hours is a failed attack.

## Strategies

| ID | strategy | one variable | targets | path | expectation |
|----|----------|--------------|---------|------|-------------|
| **S1** | workload split | idle-gap schedule | start/stop session detection | live daemon | **evades** (fragmentation) |
| **S2** | throttle | per-step micro-sleep | util gate | live daemon | **fails** (peak gate) |
| **S3** | atypical nhead/optimizer | `nhead`, `optimizer` | kernel efficiency (J/FLOP) | offline | over-report (FP pressure) |
| **S4** | batch inflation | `batch_size` (8→128) | constant-J/FLOP assumption | offline + live | **under-report** (vs 2-param; 3-param DRAM term partly defends) |
| **S5** | power cap | `nvidia-smi -pl` | power/energy signal (DVFS) | offline + live | under-report if efficiency gain beats the slowdown |
| **S6** | clock lock | `nvidia-smi -lgc` | operating point (isolates S5) | offline | optional, only if S5 shows signal |

The three plausible under-report levers are **S1 (split), S4 (batch), S5 (power-cap)**;
S2/S3 are documented negatives (still useful as false-positive / robustness evidence).

## Files

- `adversarial_workload.py` — schedule attacks (`none`/`split`/`throttle`) via the
  benign core's no-op hooks. (Decoy strategies retired to `red_team_old/`.)
- `redteam_configs.py` — parent-linked, GT-preserving offline sets: **S4** batch
  inflation (`batch_inflation_set`), **S3** atypical (`atypical_set`). Fed via
  `eval_power_monitor.py --configs-module red_team.redteam_configs:RED_CONFIGS`.
- `power_control.py` — **S5/S6** harness: sets `-pl`/`-lgc` around a benign run,
  records the setting, restores on exit and crash. **Needs root** (see below).
- `efficiency.py` — 2× budget gate; `R_benign`.
- `live_daemon_probe.py` — S1/S2 against the live `detect_flops.py`.
- `score_redteam.py` — signed-error / under-report-vs-benign-band scorer (2-param
  vs 3-param), figures `writeup/fig_redteam_{signed,evasion}.png`.
- `test_benign_equivalence.py` — GT honesty + one-variable/parent GT-invariance CI.
- `consistency_gates.py` — **Phase III defense-in-depth**: cross-signal consistency
  layer. The point estimate uses power(+DRAM) only (every estimator term is an
  adversary lever); every other signal is a *gate* — an observable ratio that must
  stay in its benign band. `arith_intensity` (NVLink/DRAM) is the shield that catches
  batch-inflation (S4-b128) the point estimate misses; `comm_energy`/`dram_energy`/
  `est_agreement` re-catch the retired comm/decoy attacks. 8.8% benign FP. The scorer
  reports each strategy's point-estimate error AND gate verdict side by side.

## Privilege note (S5/S6)

`nvidia-smi -pl` and `-lgc` require root on this box (verified). `power_control.py`
runs without privilege (at default power, warning + `cap_applied=False`) so the
harness is testable as the tenant, but the real DVFS sweep must be run with
privilege — model it as a granted capability (like the blue team's root DCGM
access) or the red team requesting a power-capped allocation:

```
sudo /home/will/verification-games/.venv/bin/python red_team/power_control.py \
    --sweep-pl 300 250 200 150 100
```

## Run order

```bash
# 0. invariants (GT honesty + one-variable/parent GT-match)
python3 red_team/test_benign_equivalence.py

# 1. offline S4 + S3 (also the S4 J/FLOP-vs-batch go/no-go)
python3 eval_power_monitor.py --configs-module red_team.redteam_configs:RED_CONFIGS \
    --records-json red_team/red_v3_trial1_records.json --output red_team/red_v3_trial1.txt --resume

# 2. S5 power-cap sweep (root)
sudo .../python red_team/power_control.py --sweep-pl 300 250 200 150 100

# 3. live daemon S1 split (gap sweep) + S2 throttle
python3 red_team/live_daemon_probe.py --strategy split --chunk-steps 100 --gap-seconds 6 --steps 400 --out red_team/live_split.json
python3 red_team/live_daemon_probe.py --strategy throttle --target-util 65 --steps 200 --out red_team/live_throttle.json

# 4. score everything (foreground under-reporting) + figures
python3 red_team/score_redteam.py \
    --group-a red_team/red_v3_trial1_records.json red_team/red_s5_records.json \
    --group-b red_team/live_split.json red_team/live_throttle.json
```
