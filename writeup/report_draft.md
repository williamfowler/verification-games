# SRF Final Report — drafted sections

Drafted prose to paste into the docx. I kept every point from your stream-of-consciousness
notes and expanded them into report prose; I did **not** remove any of your ideas. Figure
references are descriptive + filename so you can assign final numbers. Places where a
reviewer comment is resolved are tagged `[COMMENT #n]`.

---

## FIGURE MANIFEST — final 13 (renamed `figure_1..13` in `writeup/`)

| # | File | Section | Content |
|---|---|---|---|
| Figure 1 | `figure_1.png` | Methodology | calibration → deployment overview (updated: 91→88, +NVLink) |
| Figure 2 | `figure_2.png` | Phase I Results | held-out estimate vs truth, 3-signal, 88 workloads |
| Figure 3 | `figure_3.png` | Phase I Results | ablation scatter: 2-param & 3-param panels |
| Figure 4 | `figure_4.png` | Red-Team | deployed MLP estimator: S3/S4/S5 vs benign, est-vs-truth (S1/S2 → fig 5) |
| Figure 5 | `figure_5.png` | Red-Team | S1/S2 session timeline (none/split/throttle) |
| Figure 6 | `figure_6.png` | Red-Team | S2 throttle util+power over time vs benign |
| Figure 7 | `figure_7.png` | Red-Team | signed error vs batch/nhead/cap (S4/S3/S5), across **2-param / 4-param / MLP** (3 panels) |
| Figure 8 | `figure_8.png` | Red-Team | S4 NVLink & DRAM bytes vs batch |
| Figure 9 | `figure_9.png` | Red-Team | S5 power over time per cap |
| Figure 10 | `figure_10.png` | MLP | pure + residual architecture diagrams |
| Figure 11 | `figure_11.png` | MLP | median-error bar (2p/3p/4p/pure/residual) |
| Figure 12 | `figure_12.png` | MLP | pure vs residual est-vs-truth, two panels |
| Figure 13 | `figure_13.png` | Red-Team | Δ median error (benign→attack) per estimator, grouped by S3/S4/S5 |

**Archived to `writeup/old_figures/`** — Jetson-era + superseded figures, plus four you
*reference in the text but did not number*. If you want any of these four as real figures,
tell me the slot and I'll renumber:
- `fig_redteam_eff_vs_error.png` — the "how efficiency stacks up vs benign" graph (referenced in
  Red-Team Results).
- `fig_redteam_nn.png` — the MLP-vs-red-team result (your **abstract** says the MLP "sacrifices
  even worse performance against the red-team" — this is its only figure).
- `fig_redteam_powercap.png` — v1-vs-v2 power-cap sweet-spot artifact (only if you discuss it).
- `fig_trials_ablation.png` — the error-CDF companion to the ablation scatter.

---

## Small insertions into the (already-written) Methodology — for the reviewer comments

- **[COMMENT #1 — Daniel: does the run converge / gradients constant?]** After the "randomly
  generated tokens" sentence, add: *"Because the tokens are random, the run does not converge to
  anything meaningful and gradient magnitudes stay roughly stationary throughout — but this is
  irrelevant to what we measure, since the algorithmic FLOP count and the energy drawn per step
  depend only on the tensor shapes, not on whether the loss is going down."*

- **[COMMENT #2 — Madeleine: literature for reasonable hyperparameters?]** Reword *"I was unsure
  which hyperparameters to use…"* → *"There is no standard recipe for which shapes saturate a
  given GPU, so rather than guess, I swept a wide range of configurations and kept the ones that
  cleared the utilization gate."* You can cite the canonical shape conventions your grid sits on:
  the base transformer block (d_ff = 4·d_model, head_dim = 64) from Vaswani et al. 2017, the
  production model dimensions in GPT-3 (Brown et al. 2020, Table 2.1), and the finding that loss
  depends only weakly on width/depth aspect ratio (Kaplan et al. 2020) — which is exactly why
  sweeping shape at fixed compute is reasonable.

- **[COMMENT #5 — Madeleine: make clear which constants are fixed after calibration]** Split the
  estimator's term list into two labeled groups: **Measured per workload:** `E_net`, `TB_moved`,
  `t`. **Fixed at calibration (constants):** `E_PER_TFLOP`, `E_PER_TB`, `P_overhead`.

- **[COMMENT #6 — Daniel: why is only overhead scaled by time / why average FLOPs]** Add: *"`E_net`
  is itself power integrated over the whole run, so the `P_overhead·t` term is simply the energy
  the fixed overhead power accounts for over that same duration. We care about the total FLOP
  count integrated over the run — not an instantaneous rate — because the treaty threshold is a
  total-training-FLOP number, so the estimator's job is to accumulate over the session and report
  one total."*

- **[COMMENT #0 — Daniel: don't call prior work your own in a paper]** In the paper version,
  phrase the Jetson MVP as cited prior work ("earlier work [9] built…") rather than "my earlier
  work"; keep as-is for this report.

- **[COMMENT #4 — Daniel: state target audience]** One line early on: this is written for readers
  at the intersection of AI governance and ML systems — familiar with training at a high level but
  not necessarily with GPU telemetry.

---

## Initial FLOP-estimation model — RESULTS  (drafted)

*(Your stream-of-consciousness paragraphs are preserved and tightened below; the trailing bullet
list that repeated them is folded in.)*

I ran all 91 hyperparameter configurations once and kept the ones that cleared the frontier
gate — each GPU active at least 80% of the time on average. **88 of the 91 configurations cleared
the gate** (`[COMMENT #7: X = 88]`); the three that did not were the two FP32 contrast runs, which
fall outside the FP16 regime the estimator is calibrated on, and one small boundary case that sat
just under 80% utilization. The 80% cutoff is a deliberate but arbitrary line — it just has to be
*some* threshold that isolates the compute-saturated regime a real developer would train in
[a citation on economically-rational utilization could go here]. I then re-ran each of the 88
surviving configurations 9 more times, for 10 trials each, collecting traces of the three sensor
inputs on every run. At this stage nothing is being estimated — we are only gathering traces.

To measure how well the estimator *generalizes*, I split the 88 workloads into random
calibrate/evaluate splits. On each split the estimator's constants are fit (a simple linear
least-squares fit) on the calibration workloads only, and then tested on the held-out evaluation
workloads. Because the configurations span such a wide range of shapes, the evaluation workloads
are genuinely different from the calibration ones — the only thing they have in common is clearing
the 80% gate — so this is a real test of generalization, not memorization. I repeated this split
200 times, each time drawing a random one of the 10 traces for each workload, so the reported
error averages over both the split and the trial-to-trial noise.

The held-out estimate-vs-truth results are shown in **[Figure: `figure_2.png`]**: each
dot is one held-out workload, its position set by the ground-truth FLOPs (x) and the estimator's
median held-out estimate (y), with the shaded bands marking ±10% and ±20% error. `[COMMENT #8:
each dot is one workload — not bars; caption states x = truth, y = estimate.]` The full four-signal
estimator lands at a **median held-out error of 15%**.

Since the estimator is just a linear fit with a coefficient on each input signal plus a background
term, I suspected some of those coefficients might be collinear and contributing little. To test
this I ran an ablation, refitting with only power + memory bandwidth (the 3-parameter estimator)
and with only power (the 2-parameter estimator), shown in
**[Figure: `figure_3.png`]** (and as error CDFs in
`old_figures/fig_trials_ablation.png`). The result was surprising: **dropping down to power alone *improves*
the median error from 15% to 13%.** I had expected memory bandwidth and NVLink traffic to
contribute to an accurate FLOP estimate, but it appears their fitted coefficients were mostly
noise — they aren't good predictors of FLOPs on benign runs, and including them made the estimator
slightly worse. In one sense this is a welcome result: being able to read FLOPs accurately from
**power alone** is powerful, because power is the easiest of the three signals to measure. And the
memory and interconnect signals are not wasted — as the next section shows, they become useful not
as terms in the point estimate but as independent consistency checks against adversarial workloads.

---

## Red-Team Adversarial Workloads — METHODOLOGY  (drafted)

*(Threat-model and scope paragraphs preserved from your notes and tightened; then the strategy
descriptions that were a placeholder are written out.)*

**Threat model.** Up to this point the estimator has only been validated on benign sample
workloads that make no attempt to evade it. In a real deployment — enforcing an actual treaty at a
real data center — this benign calibration is exactly where you would start. But a treaty cannot
assume good faith: no party signs unless they believe the other party *couldn't* cheat undetected,
so we must assume even a compliant-on-paper party will probe for holes. The game is asymmetric in
both directions: the blue team only has to catch the red team once, but the red team only has to
succeed once to train a model that gives them a decisive capability uplift.

**Scope of the red team.** We assume the red team knows roughly what the blue team is doing — not
the exact code, but an educated guess at the method. Crucially, the red team is not free to do
anything: an attack that wildly inflates cost is not economically viable, because a training run
that takes far too long will finish with obsolete, no-longer-frontier capabilities [citation on
the pace of frontier progress vs. training duration would fit here]. We formalize this as a
**budget: an attack may use at most 2× the GPU-hours a benign run of the same FLOP count would
take.** An attack that halves the estimate but costs 5× the GPU-hours is a failed attack.
**Limitation to flag:** I play both red and blue team here. Ideally these are separate teams so
each develops strategies without knowledge of the other; a single author who knows both sides will
overstate robustness (revisited in the Discussion).

**The five strategies.** *(This is the bulleted-list placeholder, written out. The outline also
listed "higher precision" and "adaptive LoRA rank"; I note why those became the set below.)*

1. **S1 — Workload splitting (schedule attack).** Insert idle gaps longer than the daemon's
   stop-detection window so that one continuous run is recorded as several short, unrelated
   sessions. The goal is not to fool the point estimate but to defeat *attribution* — the blue
   team never sees a single job resembling the real training run.
2. **S2 — Throttling (schedule attack).** Insert micro-sleeps between steps to hold the *average*
   GPU utilization below the frontier gate, hoping the run is dismissed as non-frontier.
3. **S3 — Atypical hyperparameter geometry.** Hold the FLOP count fixed but use unusual attention
   geometry (very large or very small head dimension) or swap the optimizer (AdamW → SGD). Neither
   changes ground-truth FLOPs, but both change the energy-per-FLOP the hardware actually spends,
   probing how far the constant-`E_PER_TFLOP` assumption can be pushed.
4. **S4 — Batch inflation.** Hold the FLOP count fixed (batch × steps constant) while inflating the
   per-step batch far above the calibration range (8 → 128). Larger batches run bigger, more
   efficient matrix multiplies, so the true energy-per-FLOP drops below the calibrated constant and
   the estimate reads low.
5. **S5 — Power-capping (DVFS).** Use `nvidia-smi -pl` to run the GPUs at a lower voltage/frequency
   point, betting that lower-voltage operation does the same FLOPs at lower energy and drives the
   estimate down.

*Two strategies from the brainstorm were not pursued as live attacks.* Running at **higher
precision** cannot hide FLOPs on this hardware: FP16 is already the V100's maximum-throughput,
minimum-energy-per-FLOP precision and the calibration precision, so FP32 only *raises* energy per
FLOP and *over*-reports — the channel is closed on Volta (it would reopen on newer GPUs whose
baseline is BF16/TF32). **Adaptive LoRA-rank scaling** is left to future work.

**Experimental setup.** Each strategy modifies the same benign training script from Phase I (or,
for S5, wraps it with an external power cap); the monitoring daemon runs in the background exactly
as in deployment. We keep the same 200 calibrate/evaluate splits and frozen calibration from
Phase I — the estimator never re-fits on adversarial data — and collect 10 trials of each strategy.

---

## Red-Team Adversarial Workloads — RESULTS  (drafted)

We scored every adversarial run against the **frozen** Phase I calibration and its benign error
band. The headline: **most attacks stay within the benign error band while remaining budget-legal;
the strongest legal lever is batch inflation, which pushes the estimate to roughly 25% below
ground truth.**

**Overall picture.** **[Figure: `figure_4.png`]** places the three point-estimate strategies
(S3/S4/S5) on the same estimate-vs-truth axes as Phase I, under the **deployed (most-accurate) MLP
estimator**, with the benign held-out cloud as the reference. Every strategy is comfortably
**budget-legal** — efficiency ratios run 0.4–1.1×, all far inside the 2× budget (the cost-vs-error
view is `old_figures/fig_redteam_eff_vs_error.png` if you want it as a figure). A note on that
efficiency axis (lower = more efficient): most attacks are actually *more* efficient than the
benign median, because "frontier" was defined by GPU utilization, not achieved throughput — the
benign pool spans a 2.5× range of real efficiency, and the attacks are built on dense,
compute-bound parent shapes near the top of that range. (S1 split and S2 throttle are *attribution*
attacks with no point estimate, so they are not on this axis — see figure_5.)

**S1 — splitting succeeds (attribution).** **[Figure: `figure_5.png`]** shows the
daemon's session timeline: the benign run is logged as one session, but the split run fragments
into four short sessions, none of which resembles the real job. This is the one attack that cleanly
works, and it motivates hardening the daemon's start/stop logic (minimum-gap coalescing,
cross-session correlation).

**S2 — throttling fails.** **[Figure: `figure_6.png`]** contrasts utilization and power
over time for a benign run and a throttled run. Throttling does lower the *average* utilization,
but every compute step still spikes to 100%, and the daemon gates on *peak* utilization — so the
run stays flagged as frontier. Throttling would only evade an average-utilization gate.

**S3 — atypical geometry stays within the band (for the linear estimator).** **[Figure:
`figure_7.png`, S3 panel]** shows signed error across attention geometries for all three
estimators. For the linear estimators, kernel efficiency moves the estimate about ±15% at fixed
ground truth in *both* directions: a very large head dimension is more tensor-core efficient and
mildly under-reports, while a tiny head dimension is inefficient and *over*-reports (a
false-positive pressure, not an evasion). All of the linear results sit inside the benign band —
robustness evidence, not a break. **The MLP, however, under-reports by 30–40% across the atypical
geometries** — well past the band (see the estimator comparison below).

**S4 — batch inflation is the real lever.** **[Figure: `figure_7.png`, S4 panel]** shows the
signed error falling monotonically as batch grows past the calibration range, reaching about
**−23% (4-param) / −25% (2-param) at batch 128** — budget-legal, in-tenant, and with the ground
truth held exactly constant. This is the strongest legal under-report against the *linear*
estimator.

**Estimator comparison — the MLP is more accurate on benign runs but *less* robust under attack.**
Because the neural estimator (introduced in the estimator section as the most accurate on benign
workloads, ~10% vs ~13–15%) is a candidate to deploy, we re-ran every lever against it too.
**[Figure: `figure_7.png`]** overlays the **2-param, 4-param, and MLP** signed errors: on the two
*real* attacks the MLP under-reports *harder* than either linear model — batch-128 reaches **−41%**
for the MLP versus −23–25% for the linear estimators, and the atypical-geometry runs reach −30–40%.
**[Figure: `figure_4.png`]** shows the same on the estimate-vs-truth axes for the deployed MLP, and
**[Figure: `figure_13.png`]** summarizes it as the change in each estimator's median error from
benign to each strategy — the MLP bar is the deepest on **S3 and S4**. (S5 power-capping is a null
result once thermally matched — it moves *no* estimator, and the MLP actually reads it slightly
*better* than the linear models, because a capped run stays in-distribution; the MLP is fooled only
where an attack pushes the workload *out* of distribution.) The reason is the mechanism that won it
accuracy: the network learned that energy-per-FLOP varies with workload shape, and an attacker who
pushes the workload into an efficient corner it never saw in training gets that flexibility working
*for* them, extrapolating the correction straight into a larger under-report. The linear model's
rigidity — one constant it cannot be argued out of — is itself a defense; the network's flexibility
is a new attack surface. (The physics-anchored *residual* MLP shown here is the less-foolable of the
two networks; the pure MLP is worse still, reaching −44% at batch 128.) **The headline tradeoff: the
MLP is the better meter on honest workloads and the worse one under the out-of-distribution
attacks.**

**The byte signals earn their keep as a shield.** This is where the memory and interconnect signals
— useless in the point estimate — become valuable. **[Figure: `figure_8.png`]** shows why: as
batch inflates at constant FLOPs, the NVLink all-reduce traffic collapses (it scales as 1/batch)
while DRAM traffic falls only modestly, so the **NVLink/DRAM ratio drops far below anything seen on
benign runs.** Used as an independent *consistency gate* rather than an estimator term, that ratio
flags the strongest batch-inflation run that the point estimate alone reads as benign
(`old_figures/fig_redteam_gates.png`). A gate has no coefficient for the adversary to push on — which is
exactly why these signals are better spent as tripwires than as terms.

**S5 — power-capping does nothing (a clean negative result).** **[Figure: `figure_9.png`]** shows
GPU power over time at every cap from 300 W down to 100 W: the traces **overlap**. This workload
draws only ~200 W (below the caps), so capping never binds — power, runtime (~100 s), and
energy-per-FLOP (~4.0–4.1 J/TFLOP) are the same at every setting, and so is the estimate: a flat
**−13% to −16% across the whole range, all inside the benign band.** The **S5 panel of [Figure: `figure_7.png`]** makes this explicit: on the same y-scale as the S3/S4
panels, all three estimators (2-param, 4-param, and MLP) run flat and in-band across the whole
300→100 W range, exactly where S3 and S4 dive. Power-capping is not a working under-report lever on
this hardware. *(Methodological note worth including: an earlier version of
this sweep showed a dramatic −35% at "stock" that vanished under capping. That was a **thermal
measurement artifact** — the estimator is calibrated on hot, steady-state GPUs from the long Phase I
collection, and a cold first run boosts to higher clocks and lower energy-per-FLOP, reading ~20 pp
low. Warming the GPUs to steady state before the sweep — a `--warmup` pass in `power_control.py` —
removes it entirely and yields the flat result above. This is a good cautionary example of
side-channel calibration being sensitive to operating conditions the workload never advertises.)*

---

## The MLP estimator  (drafted — designed to FOLD INTO the estimator section)

> **Restructure note.** Per your plan to drop the standalone MLP section and move it into the
> first (estimator) section: the **methodology + accuracy** below (Figures 10 pure/residual
> architecture, 11 median-error bar, 12 pure-vs-residual scatter) belong in the estimator section
> as "a third estimator we tried." The **red-team robustness** of the MLP has been moved into the
> Red-Team Results (the "Estimator comparison" paragraph, Figures 4 & 7) so the red-team stands on
> its own and covers all three estimators. What remains here is just the MLP's design and its
> benign accuracy.

*(Your notes preserved; `[COMMENT #9]` — I corrected the two descriptions to match what the code
actually does, and wrote the "math stuff on what we're taking the logs of" you flagged.)*

The linear estimator assumes a single constant energy-per-FLOP. But as S3 and S4 make clear, the
true energy-per-FLOP is not constant — it drifts with the *shape* of the work, because a large
dense matrix multiply uses the tensor cores more efficiently than a small or awkward one. A single
constant cannot track that drift, but a small neural network can. In an effort to improve accuracy,
I trained a small multi-layer perceptron (MLP) on the same benign traces. Like the linear
estimator, the MLP sees only the cumulative sensor totals at the end of a run, and the same
workload-detection logic decides when a run starts and ends. I tried two designs
(**[Figure: `figure_10.png`]**):

- **Pure MLP.** `[COMMENT #9 correction]` Rather than the raw cumulative totals, the network takes
  the **logarithms** of the cumulative sensor values and a few dimensionless ratios of them —
  specifically `log(E_net)`, `log(t)`, `log(DRAM)`, `log(NVLink)`, `log(E_net/DRAM)`, and
  `log(NVLink/DRAM)` — and directly predicts `log(TFLOPs)`, which is exponentiated to give the
  estimate. Working in log space makes the fit target relative (percent) error rather than letting
  the largest runs dominate.
- **Residual MLP (physics-anchored).** `[COMMENT #9 correction]` Instead of learning coefficients
  on each input, this design keeps the 2-parameter power estimate as a fixed physical backbone and
  lets the network learn only a *multiplicative correction* on top of it:
  `TFLOPs = est_2param × exp(g(z))`. The correction `g(z)` is driven by dimensionless *intensity*
  features `z` (`log(power)`, `log(E_net/DRAM)`, `log(NVLink/DRAM)`, `log(DRAM/t)`, `log(NVLink/t)`,
  `log(t)`) that describe efficiency, not job size. When `g(z)=0` the estimator is exactly the
  physics estimate, so the network can only refine it, never diverge wildly.

Both networks are evaluated with the **same workload-held-out cross-validation** as the linear
models, on identical folds, so the comparison is apples-to-apples.

**Results.** **[Figure: `figure_11.png`]** shows median held-out error for every estimator,
and **[Figure: `figure_12.png`]** shows the two networks' held-out estimates against
truth. Both MLPs cut the median held-out error from about **13% to 10%** — a ~20% relative
improvement — and tighten the worst-case tail. The gain comes precisely from learning the
intensity-dependence of energy-per-FLOP that the linear model's single constant cannot capture.

**But the accuracy gain does not survive the red team — it reverses.** That result now lives in the
**Red-Team Results** (the "Estimator comparison" paragraph, Figures 4 & 7), where the MLP is scored
against the same levers as the linear estimators: it under-reports *harder* on every one (−41% at
batch 128 vs −23–25% linear). Summarized here: **the MLP is the better meter on honest workloads
and the worse one under attack** — the accuracy–robustness tradeoff that is the paper's central
finding, and the reason the estimator you *deploy* is not simply the most accurate one.

---

## Discussion / Conclusion  (drafted)

*(Your recap and future-work bullets preserved; I corrected the "Phase III" framing, since the
third phase is now the MLP study — which improved accuracy but did not improve robustness — rather
than a hardening phase.)*

**Recap.** FLOP estimation is a likely prerequisite for verifying international AI agreements, which
increasingly lean on training-FLOP thresholds. In Phase I we built an MVP that estimates training
FLOPs from three external GPU signals, achieving ~15% median held-out error (and a surprising ~13%
from power alone). In Phase II we red-teamed it: most budget-legal attacks stay within benign
scatter, the one clean success is attribution-splitting (S1), the strongest point-estimate lever is
batch inflation (~25% under), power-capping does not work at realistic scale, and the memory /
interconnect signals — weak as estimator terms — are valuable as independent consistency gates. In
Phase III we asked whether a small neural network could do better: it improved benign accuracy to
~10% but **degraded** adversarial robustness, a concrete instance of the accuracy-vs-robustness
tradeoff that a verifier has to weigh.

**Caveat.** These robustness numbers are likely optimistic because I play both red and blue team
and know exactly which attacks were tried. Even with a conscious effort to imagine attacks blind,
one author cannot substitute for adversarial separation. A related caution the Phase III study
makes concrete: **do not calibrate the estimator on adversarial data** — doing so launders the
attack into the "benign" model and hides it. `[COMMENT #9-adjacent: this is the "what could go
wrong" point from the dropped section, worth one line here.]`

**Future work.** Scale to a larger cluster; run red and blue as genuinely separate teams blind to
each other; replace software side-channels with real physical ones; reframe the metric from
percentage error to false-positive / false-negative rates around a fixed FLOP threshold (the form a
treaty actually needs); and extend the approach to distinguishing inference from training. Overall,
FLOP estimation should be treated as one layer of a Swiss-cheese verification stack, not a
standalone guarantee.

**[COMMENT #7 — Will: are synthetic workloads comparable to real ones?]** Add to future work /
limitations: the sample runs are pure compute on random tokens, with no data pipeline, checkpointing,
or evaluation overhead, so they sit at higher utilization than a real training job; validating the
estimator against a handful of real training workloads is important future work.
