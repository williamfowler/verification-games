# Phase II — Red-Team Adversarial Workloads: findings (v1)

> **v1 — superseded by the v3 plan.** This single-trial run established that the
> energy model is not the attack surface on Volta (all offline hide-FLOPs attacks
> over-report) and that workload-splitting evades. v3 builds on these results with
> under-report levers (batch-inflation, power-capping). Retired v1 code/figures/
> records now live in `red_team_old/` (figures referenced below moved there).

Everything needed to write the outline's "Red-Team Adversarial Workloads" section.
Single-trial collection (2026-08-03/04), dual-V100, `floppy`. Scale to 10 trials
for publication-grade distributions; the direction and magnitude of every result
below are already unambiguous.

## Setup (what the blue team is, frozen)

- **Calibration is frozen at Phase I** and never sees adversarial data (avoids the
  Phase III laundering anti-pattern). Constants fit on the **889 fp16 frontier
  runs** across the 10 Phase I trials:
  - 2-param (power only): `E_MARGINAL = 4.78 J/TFLOP`, `P_OH = 0`
  - 3-param (power+DRAM): `E_MARGINAL = 3.54`, `E_PER_TB = 93.4 J/TB`, `P_OH = 0`
- **Benign baseline error** the attacks must beat: **~13% median |error|** (2-param
  ship fit; Phase I held-out was ~15%).
- **Efficiency reference** `R_benign = 38,540 TFLOPs/GPU-hour` (median over the 889
  frontier runs; GPU-hours = wall-hours × 2). **Budget: an attack is legal iff it
  spends ≤ 2× the GPU-hours per real FLOP**, i.e. ratio `R_benign/R_adv ≤ 2`.
- **Ground truth stays honest** in every run (FlopCounterMode); the attack only
  moves the blue team's *estimate*. Enforced by `test_benign_equivalence.py`.

## Headline result (the section's thesis)

**On this V100 setup the fp16-calibrated energy estimator is robust to every
offline "hide-FLOPs" attack: all three (precision, memory decoy, atypical shape)
make the estimator *over*-report, not under-report — the opposite of hiding
compute — and every one blows the 2× GPU-hour budget. The estimator's real
attack surface is not the energy model but the live daemon's *session logic*:
workload splitting fragments one run into sub-attributed sessions and succeeds;
sub-gate throttling fails against a peak-utilization gate.**

Root cause of the offline robustness: fp16 is already the minimum-energy-per-FLOP
precision on Volta, and the estimator is fp16-calibrated, so there is no way to
*lower* measured energy per true FLOP. Every attack that changes the run either
adds energy (decoy), uses more energy per FLOP (fp32), or just extrapolates the
fit (atypical) — all of which push the estimate **up**.

## Group A — offline strategies (scored on the clean-bracketed window)

Signed error: **+ = over-report (caught); − = under-report (evasion).** `eff×` =
GPU-hours per real FLOP relative to benign (>2 = over budget). Figures:
`writeup/fig_redteam_error.png` (error), `writeup/fig_redteam_efficiency.png` (cost).

| strategy | n | err 2-param | err 3-param | eff× (median) | budget-legal |
|---|---|---|---|---|---|
| **S1** precision (fp32) | 4 | **+357%** | **+450%** | 25.2× | 0/4 |
| **S2** memory decoy | 4 | **+829%** | **+367%** | 17.1× | 0/4 |
| **S2′** NVLink decoy [V100] | 2 | **+201%** | **+277%** | 5.4× | 0/2 |
| **S3** atypical hyperparams | 5 | **+47%** | **+57%** | 8.2× | 0/5 |

Every strategy over-reports and every strategy is over budget. Detail per strategy:

### S1 — precision (fp32 twins) — *channel closed on Volta*
- The classic attack (run a higher-throughput precision so measured energy-per-true-FLOP
  drops → under-report) **has no target here**: fp16 is already the baseline, the
  max-throughput precision (no bf16/tf32/fp8), *and* the calibration precision.
  The only other precision, fp32, has *higher* energy per FLOP → the estimator
  reads **+357% / +450%** high. It cannot hide FLOPs; it exposes them.
- Cost: fp32 is **~22× slower per FLOP** than its fp16 twin (matched-twin: 23.7× and
  21.2× on the two configs with Phase I twins; pool-median 25×). Deeply illegal.
- **Write it as:** the precision evasion channel is *closed* on this hardware; it
  would reopen on an Ampere+ node whose baseline/calibration is tf32 or bf16.

### S2 — memory decoy (DRAM) — *backfires*
- Mechanism works: concurrent device-to-device copies inflate measured DRAM from
  **14 TB → up to 211 TB** (13–15×) while GT stays honest. Intent: inflate the
  `E_per_TB·TB_moved` term the 3-param estimator subtracts.
- **It backfires.** The decoy copies *also* burn power — net energy rises 7.2k →
  ~29k J. The estimator subtracts only `93.4 J/TB` for the extra bytes, but the
  decoy actually costs **~120 J/TB moved**, so it adds more energy than it
  subtracts: 3-param reads **+367%** high, and the power-only 2-param (which
  ignores DRAM entirely) reads **+829%** high. No decoy intensity (150→1200 GB/s)
  reverses this — the sweep shows it fails at every intensity.
- Cost: **~18× slower** per FLOP (matched-twin, vs the benign base). Illegal.

### S2′ — NVLink decoy [V100]
- Extra all-reduce inflates NVLink bytes (97 → 140 GB at `ar256`) cheaply
  (wall-clock ~unchanged). But the interconnect term is `E_PER_NVL ≈ 0` (collinear
  with DRAM — the Phase I finding), so it isn't subtracted; the extra all-reduce
  just adds a little power → **+201% / +277%** over-report. Its real target is the
  Phase-I **NVLink consistency tripwire** (COMM_FLOOD), a Phase III lever, not the
  point estimate. Still ~5.4× over budget from the extra collective.

### S3 — atypical hyperparameters
- Out-of-distribution geometry (ff/d = 21×, seq 8192, nhead 64 / head_dim 16, 48
  layers, d_model 3072) forces the linear energy fit to extrapolate → **+47% /
  +57%** error, ~3–4× the benign 13%. Still positive (over-report), so it degrades
  accuracy but doesn't hide FLOPs. One config (`d3072`) OOM'd and was excluded.
- The 8.2× "efficiency" here is mostly that these odd shapes are intrinsically
  less GPU-efficient than the median frontier — read it as "atypical shapes cost
  more," not deliberate waste.

## Group B — live-daemon strategies (attribution / session logic)

Run against the live `detect_flops.py` (`live_daemon_probe.py`); the daemon polls
GPU 0 every 1.5 s, starts a session after 2 active polls, ends it after 3 quiet
(<5% util) polls (~4.5 s), and gates "frontier" on **peak** util ≥ 80%.

| run | honest GT | daemon sessions | frontier sessions | attributed |
|---|---|---|---|---|
| none (benign) | 537 TF | **1** | 1 | 186 TF |
| **S4 split** | 537 TF | **4** | 4 | **0 TF** |
| **S5 throttle** | 403 TF | 1 | 1 (peak 100%) | 0 TF |

### S4 — workload splitting — *succeeds*
- Chunking the run into bursts separated by 6 s idle gaps (> the 4.5 s stop window)
  makes the daemon record **4 separate sessions** instead of 1. The blue team sees
  four short unrelated jobs, none resembling the real run — a genuine **attribution
  evasion**. This is the one attack that works, and it motivates Phase III
  (harden start/stop detection: minimum-gap coalescing, cross-session correlation).
- Cost: the idle gaps inflate wall-clock — sweep `--gap-seconds` to find the
  budget frontier (6 s gaps on a ~short run already push past 2×; longer real runs
  amortize the gaps better — worth quantifying in the 10-trial pass).

### S5 — sub-gate throttling — *fails the gate*
- Micro-sleep throttling lowers *average* util but **not peak** — each compute step
  still spikes to 100%, so the daemon's peak-util gate still marks the run
  **frontier** (not evaded). Throttling only evades an *average*-util gate (the
  offline calibration scope), not the live daemon's peak gate.
- **Caveat (do not overclaim):** the daemon attributed 0 TF to both the split and
  throttle sessions. That 0 is partly a **single-GPU-daemon artifact** — its
  per-session idle-baseline calibration mis-fires on sessions that don't start
  from clean idle, zeroing the net energy. The *solid* Group B result is the
  session **fragmentation count** (4 vs 1), which is not an artifact. Fixing the
  daemon's baseline (part of the deferred both-GPU rewrite) would firm up the
  attributed-TFLOPs numbers; the fragmentation conclusion stands regardless.

## Caveats to state in the writeup

1. **Single trial.** Directions/magnitudes are clear but variances aren't; the
   10-trial pass gives per-strategy distributions comparable to Phase I.
2. **Single-GPU live daemon.** Group B estimates are ~⅓ of the 2-GPU truth and the
   per-session baseline is fragile (the est→0 artifact above). Group B's robust
   claim is *session structure*, not attributed magnitude.
3. **fp16 calibration is what closes S1/S2.** State the hardware dependence: the
   offline robustness is specific to an fp16-baseline, fp16-calibrated Volta node.
4. **Same-person red/blue.** As the outline already notes, both teams are the same
   author — the attacks are what one author imagined, likely not exhaustive.

## Reproduce

```
python3 red_team/score_redteam.py \
  --group-a red_team/red_results_trial1_records.json \
  --group-b red_team/live_none.json red_team/live_split.json red_team/live_throttle.json
```
Raw scores: `redteam_scores.json`. Records: `red_team/red_results_trial1_records.json`,
`red_team/live_*.json`. Figures: `writeup/fig_redteam_{error,efficiency}.png`.
```
