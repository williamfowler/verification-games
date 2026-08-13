# Phase II v3 — Red-Team findings (deployment-phase, under-report-primary)

Outline-ready results for the "Red-Team Adversarial Workloads" section. Single-trial
collection 2026-08-04, dual-V100, `floppy`. v3 reframes Phase II around **under-
reporting** against the **frozen** benign-calibrated estimator, with every attack
exactly one GT-preserving variable from a benign parent. Supersedes v1 (which
established the energy model isn't the surface for over-report/decoy attacks;
retired to `red_team_old/`).

## Setup (frozen, as a deployed blue team)

- **PRIMARY estimator = the branch's full 4-param form**, one coefficient each for
  compute-energy (FLOPs), memory bandwidth (DRAM), inter-GPU data transfer (NVLink),
  and overall-power overhead:
  `TFLOPs = (E_net − E_PER_TB·TB − E_PER_NVLINK·NVLink − P_OH·t) / E_MARGINAL`
  (`detect_flops.estimate_tflops_nvl`). Frozen Phase I benign fp16 fit, never
  recalibrated on adversarial runs: `E_MARGINAL 3.54, E_PER_TB 93.4,
  E_PER_NVLINK 0.0000, P_OH 0`.
  - **The NVLink coefficient fits to 0**: on benign DDP the all-reduce volume is
    collinear with DRAM bytes (both ∝ model size), so the fit assigns that energy
    to the DRAM term and the 4-param reduces *numerically* to power+DRAM. It is
    still the estimator every strategy is scored against; the only strategy for
    which the NVLink term would matter is one that manipulates the interconnect
    directly (v1's `nvlink_decoy`, retired).
  - The **2-param power-only** estimator (`E_MARGINAL 4.78`) is reported alongside
    as the accuracy ablation Phase I flagged (byte terms don't improve benign
    accuracy — indeed power-only generalizes slightly better).
- **Benign noise band** (defines "evasion" = under-report beyond benign scatter):
  4-param signed error `[5th,95th] = [−24.3%, +37.4%]` over the 889 Phase I frontier
  runs (power-only `[−27.8, +25.1]%`). **Wide** — the estimator is a rough meter —
  so evasion needs a *large*, *systematic* under-report. (This single-trial in-sample
  band is conservative; the 10-trial pass tightens it toward the ~15% held-out spread.)
- `R_benign = 38,540 TFLOPs/GPU-hour`; **budget legal iff eff ratio ≤ 2×**.
- Every config GT-matches its parent (asserted in-process, **0.000% drift**).

## Headline (the section's thesis)

**v3 finds a real energy-model under-report lever that v1 lacked: arithmetic-
intensity via batch-inflation. Running a larger per-step batch than the calibration
saw (8/16/32 → 128) runs bigger GEMMs at higher tensor-core efficiency, so J/FLOP
falls below the fitted constant and the 4-param estimate reads up to −21% below
ground truth (−24% against the power-only ablation) — monotonically, in-tenant (no
privilege), GT-exact, and well within the 2× budget (0.8×).** Unlike every v1 attack
(which over-reported and blew the budget),
the v3 levers are **budget-legal and under-reporting** — the qualitative shift is
visible in `fig_redteam_evasion.png` (all points left of 2×; S4 in the lower-left).
The single caveat: −24% sits just inside the wide single-trial benign band, so
formally clearing "evasion" needs the tighter multi-trial band (and/or higher batch).

## S4 — batch inflation *(the real lever)*

Parent `A:d1024_s256_L6` at batch 8; variable = batch (8→128) with `steps=15360/batch`
so batch×steps — hence GT — is identical. All runs frontier (96–99% util), ~130–184 s.

| batch | J/TFLOP | **err 4-param** (primary) | err 2-param (power-only) | eff× |
|---|---|---|---|---|
| 8 | 4.76 | −17% | −1% | 0.8 |
| 16 | 4.93 | +1% | +3% | 1.1 |
| 32 | 4.33 | −7% | −9% | 0.9 |
| 64 | 3.77 | **−19%** | −21% | 0.8 |
| 128 | 3.65 | **−21%** | −24% | 0.8 |

- **J/FLOP does NOT saturate by batch 32** (the plan's open worry) — it keeps
  falling 4.33 → 3.65 through 64→128, driving the 4-param estimate to **−21%**.
  Monotone from batch 32 up; clean signal, not noise.
- **Budget-positive/neutral:** batch×steps is constant so wall-clock is flat-to-
  shorter (eff 0.8×) — the attack costs *no* extra GPU-hours. In-tenant, no privilege.
- **The DRAM term (inside the 4-param) is a partial, not a full, defense.** Higher
  intensity lowers bytes/FLOP, so the DRAM subtraction shrinks and nudges the
  4-param estimate up vs power-only — but only ~3 pp at batch 128 (−21% vs −24%),
  not enough to stop the evasion. (At *low* batch the DRAM term over-corrects the
  other way: batch 8 reads −17% under the 4-param vs −1% power-only, because
  DRAM/FLOP is high there.) The NVLink term contributes nothing (coeff 0). So the
  full 4-param blunts batch-inflation only marginally.
- **Phase III mitigation:** an arithmetic-intensity-aware or per-run efficiency
  cross-check (the estimator needs to know it's extrapolating below its batch range).

## S3 — atypical nhead / optimizer *(both directions, within band)*

Parent `A:d1024_s512_b16_L12_h8`; variable = nhead (head_dim = 1024/nhead) or sgd.
GT identical (nhead/optimizer don't change FLOPs), all frontier.

- Energy-vs-geometry at fixed GT spans **J/FLOP 4.09 → 5.26** (4-param signed error):
  large head_dim (h1/h2, head_dim 1024/512) is *more* tensor-core efficient → mild
  **under**-report (−13%/−19%); tiny head_dim (h64, head_dim 16) is inefficient →
  **over**-report (+19%). `optimizer=sgd` ≈ −10% (mild under). All within the benign
  band. (Power-only reads a compressed version of the same shape: −11%/−15% … +10%.)
- Read as: geometry-driven kernel efficiency moves the estimate ±~15% at fixed GT
  in *both* directions — false-positive pressure (h64) and a mild under-report edge
  (big head_dim), neither a standalone evasion. Robustness evidence, not a break.

## S1 / S2 — live-daemon strategies (unchanged from v1, still valid)

*Estimator-agnostic: these attack the daemon's session detection / util gate, not
the point estimate, so the 4-param-vs-2-param distinction doesn't apply — the metric
is session fragmentation / gate verdict.*

- **S1 split — evades.** Idle gaps > the 4.5 s stop window fragment one run into
  N sessions (v1: 537 TF → 4 sessions). The one confirmed attribution evasion;
  Phase III = session coalescing / cross-session correlation. (v3 extension:
  sweep gap×run-length for the budget frontier — not yet run.)
- **S2 throttle — fails.** Micro-sleep lowers *average* util but the daemon gates
  on *peak* (still 100%), so it stays flagged frontier. Documented negative.

## S5 — power-cap (DVFS) *(run with root; a working, bounded lever)*

The DVFS under-report bet — lower voltage → same FLOPs at lower energy — **pays off**.
Sweeping `nvidia-smi -pl` on the benign parent (`red_team/power_control.py`,
run with sudo; restores defaults on exit/crash):

| power cap | J/TFLOP | 4-param error | eff× |
|---|---|---|---|
| 300 W (stock) | 3.88 | −20% | 0.4× |
| **250 W** | **3.15** | **−40%** | 0.4× |
| 200 W | 3.62 | −27% | 0.5× |
| 150 W | 3.75 | −23% | 0.6× |
| 100 W | 3.87 | −20% | 0.7× |

- There is an **energy-optimal operating point (~250 W)** below stock where J/FLOP
  drops ~19% (3.88 → 3.15) and the estimate reads ~20 pp *more* under — all
  **budget-legal** (eff 0.4×; capping doesn't cost enough wall-clock to matter here).
- The effect is **bounded and non-monotonic**: capping *past* the sweet spot (100–150 W)
  loses it — static power over the longer runtime pushes J/FLOP back up. So the lever
  is real but has a ceiling.
- **Caveat:** the uncapped 300 W run itself read −20% (a baseline offset for this
  short-duration parent), so the *reliable* quantity is the **delta across caps**
  (~20 pp at the sweet spot), not the absolute error. A cleaner absolute number needs
  the same longer-run sizing as S4.
- **Privilege:** `-pl` needs root ("Insufficient Permissions" as the tenant) — model
  it as a granted capability or a red-team-requested power-capped allocation.
- **Consistency gates do NOT catch it** (all OK): power-capping leaves every byte /
  interconnect ratio in-band because the workload is unchanged. S5 is the attack that
  evades *both* the point estimate *and* the gate layer → its Phase III mitigation is a
  different kind of check: an **energy-floor / clock-and-power-state cross-check** (flag
  a run whose GPUs sit at a non-stock, sub-optimal V/f point). *(S6 clock-lock `-lgc`,
  the DVFS-isolating sibling, is available via `--sweep-lgc` but not yet run.)*

## Efficiency: the qualitative shift from v1

Every v3 lever is **budget-legal** (eff 0.76–1.13×) — vs v1 where every offline
attack was 5–25× over budget. So v3's attacks are cheap *and* (S4) under-reporting;
the danger has moved into the lower-left of `fig_redteam_evasion.png`.

## Toward Phase III — the cross-signal consistency layer (`consistency_gates.py`)

The point estimator uses power(+DRAM) and *nothing else*, because **every term in the
estimate is a lever the adversary can push on** (a nonzero NVLink coefficient would
let an interconnect flood move the estimate — and NVLink is collinear with DRAM
anyway, so it earns no weight). Instead every other signal is a **gate, not a term**:
each benign run keeps a set of observable signal *ratios* in a calibrated band, and an
adversary who moves the estimate must keep **all** of them in band simultaneously —
hard and expensive. A gate has no coefficient to push on.

Four gates, calibrated on the Phase I benign frontier (observables only — deployable):

| gate | ratio | catches | benign FP |
|---|---|---|---|
| `comm_energy` | NVLink / energy | interconnect flood / compute-burn (no all-reduce) | 2.0% |
| **`arith_intensity`** | **NVLink / DRAM** | **batch-inflation** — NVLink∝params·steps, DRAM∝tokens·steps, so the ratio ∝ 1/(batch·seq); inflating batch drops it below the benign floor | 6.1% |
| `dram_energy` | DRAM / energy | memory-vs-compute balance | 2.0% |
| `est_agreement` | power-only vs power+DRAM estimate | DRAM manipulation (splits the two estimators) | 5.2% |

**Result — the shield catches what the point estimate can't.** The point estimate
reads every S4/S3 run as ~benign (all inside the wide ±band). The independent gate
layer, at **8.8% benign false-positive rate**, flags the strongest under-reporter
**S4 batch-128 (−21%) via `arith_intensity`** — this is NVLink finally earning its
keep as a *shield* (it's useless as an energy regressor precisely because NVLink/DRAM
is stable on benign runs, which is exactly what makes a deviation detectable). It also
re-catches every retired v1 comm/memory attack (mem_decoy → `comm_energy`+`dram_energy`
+`est_agreement`; nvlink_decoy → `dram_energy`+`est_agreement`).

**Residual gap (honest):** batch-64 and nhead-2 (both ≈ −19%) slip *both* the point
estimate *and* the gates — the intensity gate's floor is tuned to catch batch-128 at
a 6% FP cost, and −19% sits just inside it. Tightening the floor catches them at more
false positives; the 10-trial pass will sharpen the tradeoff. Defense-in-depth reduces
the attack surface, it doesn't close it — which is the honest Swiss-cheese framing.

Run: `python3 red_team/consistency_gates.py` (bands + FP + catch table).

## Honest caveats

1. **Single trial.** S4's −21% (4-param) is clean and monotone but the wide
   single-trial benign band (4-param lower edge −24.3%) means it doesn't *formally*
   clear "evasion" yet — the 10-trial pass (tighter band) is needed, and/or a higher
   batch / more-inefficient parent.
2. **The 4-param's DRAM defense is weak, not absent** — don't claim the DRAM/NVLink
   terms stop S4; they blunt it ~3 pp. NVLink specifically contributes 0 (collinear).
3. **S5 (power-cap) is now run and is a working lever** (~20 pp extra under-report at
   the ~250 W sweet spot, budget-legal) — but confounded by a baseline offset, so trust
   the delta-across-caps, not the absolute; it needs root; and it evades the gate layer
   (needs an energy-floor cross-check). S6 clock-lock still unrun.
4. **Same-person red/blue.**

## Figures & reproduce

- `writeup/fig_redteam_signed.png` — signed error vs batch (S4) / nhead (S3), benign
  band shaded.
- `writeup/fig_redteam_evasion.png` — under-report vs efficiency, 2× line (threats
  = lower-left).
- Records `red_team/red_v3_trial1_records.json`; scores `redteam_scores.json`.
```
python3 red_team/score_redteam.py --group-a red_team/red_v3_trial1_records.json
```
