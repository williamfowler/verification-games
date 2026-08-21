# SRF Final Report — figure captions, figure updates, references, and factual review

Prepared against `Will Fowler - SRF Final Report(4).docx` (the version with 9 figures and 2 tables).
Figure numbers below are the **document's** numbers; the file each one comes from is noted since the
PNG filenames don't match the doc numbering.

---

## 1. Figure captions (paste under each figure)

**Figure 1** (`figure_1.png`) — *Overview of the estimation pipeline. Calibration (left, run once):
an idle power baseline is measured, 91 candidate workloads are swept and the 88 that clear the 80%
GPU-utilization "frontier" gate are kept, each workload's energy, DRAM, and NVLink signals are
recorded, and the energy constants of equation (1) are fit against FlopCounterMode ground truth.
The fitted constants are then frozen. Deployment (right, per workload): a monitoring daemon polls
power, DRAM activity, and NVLink counters every 1.5 s, detects workload start/end from GPU
utilization, accumulates the signals over the session, and reports a FLOP estimate via equation (1),
which is scored against held-out ground truth.*

**Figure 2** (`figure_2.png`) — *Held-out accuracy of the 3-input estimator (power + DRAM +
NVLink). Each dot is one of the 88 workloads: x is the ground-truth FLOP count (FlopCounterMode,
both GPUs summed) and y is the median held-out estimate over 200 random calibration/evaluation
splits, using all 10 recorded traces per workload. Shaded wedges mark ±10% and ±20% of truth.
Median absolute error: 15.1%.*

**Figure 3** (`figure_3.png`) — *Input ablation under the same protocol as Figure 2. Left: power
as the only input (median error 12.9%). Right: power + DRAM (15.1%, indistinguishable from the
full 3-input estimator). Dropping the memory and interconnect terms makes the estimator more
accurate, not less.*

**Figure 4** (`figure_10.png`) — *The two MLP estimators. Left — pure MLP: seven log-scale
features of the session totals (net energy E, duration t, DRAM terabytes D, NVLink terabytes N,
average power P = E/t, and the ratios E/D and N/D) pass through two 32-unit ReLU hidden layers to
predict log TFLOPs directly. Right — residual MLP: the same signals enter as dimensionless
intensity ratios z, and the network learns only a multiplicative efficiency correction exp(g(z))
applied to the power-only linear estimate; g(z) = 0 exactly recovers the linear estimator.*

**Figure 5** (`figure_12.png`) — *Held-out estimate vs ground truth for the two MLPs under
5-repeat × 5-fold cross-validation (identical folds for both). Each dot is the median held-out
estimate for one of the 88 workloads. Both variants reach 10.4% median absolute error.*

**Figure 6** (`figure_11.png`) — *Median held-out error of all five estimator configurations,
evaluated on identical held-out folds (5 × 5-fold CV): power-only linear 13.1%, power+DRAM and
power+DRAM+NVLink 14.9%, and 10.4% for both MLPs. (The 12.9%/15.1% figures quoted in the text come
from the 200-random-split protocol of Figures 2–3; this chart's matched-folds protocol shifts the
linear numbers slightly but preserves the ranking.)*

**Figure 7** (`figure_13.png`) — *Change in signed error going from benign workloads to each
strategy's strongest attack setting (S3: n_head = 1; S4: batch 128; S5: 250 W cap), in percentage
points; more negative = pushed further toward under-reporting. The MLP is the most accurate
estimator on benign runs but the most gameable: the S3/S4 attacks move it 30–36 pp, roughly 3–6×
the movement of the linear estimators. Power capping moves no estimator.*

**Figure 8** (`figure_5.png`) — *How the monitoring daemon attributed the S1 (splitting) and S2
(throttling) runs of one identical workload, next to a benign control. Each bar is one detected
session. S1's 6-second pauses outlast the daemon's stop window (three quiet 1.5 s polls), so a
single training run is logged as four separate, unlinked jobs. S2's micro-sleeps are shorter than
one poll interval, so the run is still detected — and estimated — as a single session.*

**Figure 9** (`figure_7.png`) — *Signed estimation error for every S3 (atypical attention heads),
S4 (batch inflation), and S5 (power cap) configuration, under each of the three frozen estimators.
The grey band is the range of signed errors observed on benign held-out workloads. Batch inflation
drives all three estimators progressively below truth as batch size grows past the calibration
range (to −41% for the MLP at batch 128); atypical head counts push the MLP far out of band at
small n_head and produce over-reporting at n_head = 64; power capping leaves every estimator inside
the benign band.*

---

## 2. What changed in the figures

All nine document figures were regenerated with the text cut down and the remaining text enlarged
(base font 9 pt → 13 pt; explanatory titles/footnotes removed — their content is now in the
captions above). **Re-insert the new PNGs into the doc.** Specific changes beyond styling:

- **Figure 1**: the stale Jetson-era step "Calibrate memory-traffic scale from a known data
  volume" is replaced with "Record energy, DRAM & NVLink bytes per workload" (on the V100 setup
  nothing is scale-calibrated — see correction #2 below), and the deployment column now mentions
  NVLink.
- **Figure 8**: now plots the v2 live-daemon runs (`red_team/live_v2_*.json`, the re-run at the
  same 2,606-TFLOP workload as S5) instead of the retired v1 runs, and the per-bar "100% peak"
  labels were dropped.
- **Figure 9**: panels reordered S3 → S4 → S5 to match the text's order; the two-sentence
  suptitle was removed.
- Generator scripts edited in place: `analyze_trials.py` (Figs 2–3), `nn_estimator.py` (Figs 5–6,
  replotted from `nn_estimator_results.json` without retraining), `writeup/make_fig_methodology.py`
  (Fig 1), `writeup/make_fig_nn_arch.py` (Fig 4), `writeup/make_figs_redteam_extra.py` (Fig 8),
  `writeup/make_new_report_figs.py` (Figs 7, 9). Each now also writes its `figure_N.png` name
  directly, so a re-run reproduces the doc figures with no manual renaming.

---

## 3. References

Paste-ready list (repeats carried over from the Jetson writeup; every entry verified to exist):

[1] California Senate Bill 53, Transparency in Frontier Artificial Intelligence Act (2025). https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202520260SB53

[2] Regulation (EU) 2024/1689 (EU AI Act), Article 51. https://artificialintelligenceact.eu/article/51/

[3] Y. Shavit. What does it take to catch a Chinchilla? Verifying Rules on Large-Scale Neural Network Training via Compute Monitoring. arXiv:2303.11341, 2023. https://arxiv.org/abs/2303.11341

[4] A. R. Wasil, T. Reed, J. W. Miller, and P. Barnett. Verification methods for international AI agreements. arXiv:2408.16074, 2024. https://arxiv.org/abs/2408.16074

[5] A. Scher and L. Thiergart. Mechanisms to Verify International Agreements About AI Development. arXiv:2506.15867, 2025. https://arxiv.org/abs/2506.15867

[6] Epoch AI. Estimating training compute of deep learning models. 2022. https://epoch.ai/blog/estimating-training-compute

[7] A. Chaudhuri, S. Shukla, S. Bhattacharya, and D. Mukhopadhyay. "Energon": Unveiling Transformers from GPU Power and Thermal Side-Channels. ICCAD 2025; arXiv:2508.01768. https://arxiv.org/abs/2508.01768

[8] R. Rahman and S. Tajdari. Detecting Hidden ML Training With Zero-Overhead Telemetry. arXiv:2606.19262, 2026. https://arxiv.org/abs/2606.19262

[9] W. Fowler. Estimating LLM Training FLOPs on the Nvidia Jetson Orin Nano. Report, University of Chicago Existential Risk Laboratory, July 2026.

[10] NVIDIA. NVIDIA V100 Tensor Core GPU Datasheet (Tesla V100-SXM2). https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf

[11] NVIDIA. Data Center GPU Manager (DCGM) Documentation. https://docs.nvidia.com/datacenter/dcgm/latest/

[12] PyTorch, torch.utils.flop_counter.FlopCounterMode. https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py

[13] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention Is All You Need. NeurIPS 2017; arXiv:1706.03762. https://arxiv.org/abs/1706.03762

[14] T. Brown et al. Language Models are Few-Shot Learners. NeurIPS 2020; arXiv:2005.14165. https://arxiv.org/abs/2005.14165

[15] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei. Scaling Laws for Neural Language Models. arXiv:2001.08361, 2020. https://arxiv.org/abs/2001.08361

[16] Wikipedia. Capture the flag (cybersecurity). https://en.wikipedia.org/wiki/Capture_the_flag_(cybersecurity) (accessed August 2026)

[17] Amodo Design. The Side Channel Cloud (instrumented 32-GPU verification-research cluster). https://amododesign.com/ai-verification/ · https://www.sidechannel.cloud

[18] Lucid Computing. Lucid Labs experimentation cluster (instrumented bare-metal GPU infrastructure for AI verification research). https://lucidcomputing.ai/experimentation-cluster

[19] Wikipedia. Swiss cheese model. https://en.wikipedia.org/wiki/Swiss_cheese_model (accessed August 2026)

**Where the placeholders go:**

- "these transformer/LLM papers (cite, cite, cite)" → **[13][14][15]**. [13] gives the
  d_ff = 4·d_model and head_dim = 64 conventions your grid sits on; [14] (GPT-3, Table 2.1) grounds
  the production model shapes; [15] shows loss depends only weakly on width/depth shape at fixed
  compute, which is what justifies sweeping shapes.
- "Capture The Flag games [cite]" → **[16]**.
- "dedicated clusters [cite Amodo and Lucid]" → **[17][18]**. Both are real: Amodo Design's "Side
  Channel Cloud" (32 instrumented GPUs, power side-channels/network taps, open to researchers) and
  Lucid Computing's "Lucid Labs" (free instrumented bare-metal clusters allocated by the Verifiable
  Compute Foundation).
- "Swiss-Cheese model [cite]" → **[19]**.
- Verified thresholds: SB 53 defines "frontier model" at >10^26 ops; EU AI Act Art. 51(2) presumes
  systemic risk above 10^25 FLOPs — both match your text.
- [8] spelling confirmed: Robi **Rahman** and Sabiha **Tajdari** (not Prof. Shahin Tajik from the
  acknowledgments — different person).
- [9] has no public URL; if the Jetson report is hosted anywhere, add the link.

---

## 4. Factual review

Checked sentence-by-sentence against the code and data in the repo. **Errors first, then
imprecisions where you may just be being concise, then typos.**

### Must fix (factually wrong)

1. **"Each card is capped at 250 W"** (Hardware). These are V100-**SXM2** boards: the power limit
   is **300 W** (verified live: `nvidia-smi` reports current/default/max limit = 300 W on both
   GPUs, and the S5 sweep treats 300 W as the stock setting). 250 W is the PCIe V100's number.

2. **"…scaled from a duty cycle to bytes during calibration"** (TB_moved bullet). Nothing is
   scale-calibrated on this setup (that was the Jetson actmon procedure). The DCGM DRAM-activity
   fraction is converted to bytes/s by multiplying by the fixed HBM2 peak bandwidth (900 GB/s) per
   GPU and summing over both GPUs; any constant error in that conversion is absorbed into the
   fitted E_PER_TB during calibration. Suggested replacement: *"…monitoring the DCGM
   memory-bandwidth-utilization signal on each GPU, converted to bytes at the HBM2 peak bandwidth
   and summed over both GPUs; any constant error in that conversion is absorbed into E_PER_TB
   during calibration."*

3. **"…integrating the DCGM receive byte-rate counters over time"** (NVL_TB bullet). The fit uses
   NVLink **transmit + receive** counters, summed over both GPUs. Say "transmit and receive
   byte-rate counters."

4. **"The specific hyperparameters I altered were the size of the model (d_model) and the number
   of attention heads (n_heads)"** (Strategy 3 description). d_model was never altered — it is held
   at 1024 across the whole S3 set (your own Table 2 lists it under "Values Held-Constant"). The
   set varies **n_heads ∈ {1, 2, 8, 32, 64}** plus one AdamW→SGD twin at n_head = 8. Replace
   d_model with the optimizer swap (or drop it).

5. **"throttling only brings the peak utilization from ~80% to 65%"** (S1/S2 results). Two
   problems, per the live-daemon data: (a) the throttled run's **peak** utilization stays **100%**
   — the micro-sleeps are shorter than the 1.5 s poll interval, so every poll still catches the GPU
   active; it's the **average** utilization the throttle targets (~65%, vs ~97% benign). (b) The
   monitor doesn't key on a threshold the throttle could duck under while running — a session only
   ends after three consecutive polls **below 5%**, and throttling never produces even one.
   Suggested rewrite: *"With a 6-second pause, utilization drops to 0 for multiple polls, ending
   the session. Throttling lowers only the average utilization (~97% → ~65%); each 1.5 s poll still
   sees the GPU active, so the run never produces the three consecutive sub-5% polls needed to
   close a session."*

6. **"both get an average error of 10.4%"** (Improvement #2). It's the **median** error (the same
   statistic used everywhere else). The mean is higher.

7. **"This only nullified 3 of the configurations"** (Results). Of the three dropped configs, only
   **one** failed the 80% utilization gate (a d_model 768 / 6-layer / seq 128 / batch 8 run at
   77%). The other **two were FP32 runs**, excluded because they sit outside the FP16 regime the
   estimator is calibrated on — not because of utilization. One-sentence fix: *"This nullified only
   3 of the configurations: one that fell just under the utilization gate, plus two FP32 runs
   outside the FP16 regime the estimator is calibrated on."* (With that fix, Table 1's counts —
   AdamW n=85, SGD n=3 — check out.)

8. **Abstract: "adversarial workloads that increase the estimator's error to a median of 25%."**
   Pooled across *all* adversarial configurations the deployed MLP's median absolute error is only
   ~13–15% (power capping does nothing and drags the pool median down). The 25% number is
   supportable only for the strongest strategies individually: median |error| on the atypical-head
   set is **25.6%**, on the batch-inflation set **22.4%**, worst single config **−41%** (batch
   128). Suggested precise phrasing: *"…adversarial workloads that increase the estimator's median
   error to ~25% under the strongest strategy, with the worst configuration under-reporting by
   41%."*

9. **"showing an increase in error of 36%"** (Red-team results). Percentage **points**, not
   percent — and it's the change in *signed* error at S4's strongest setting (batch 128), with
   −30 pp on S3. Suggest: *"…was also the most gameable, with its estimate pushed up to 36
   percentage points further below truth."* Similarly, "Figure 7 shows the change in **median**
   error" → the figure shows the change in **signed error at each strategy's strongest setting**.

### Judgment calls (arguably just concision — flag so you can decide)

10. **Text vs Figure 6 numbers.** The text's 12.9%/15.1% come from the 200-random-split protocol
    (Figures 2–3); the MLP's 10.4% and Figure 6's bars (13.1%, 14.9%) come from the 5×5-fold
    matched-folds CV in which every estimator sees identical folds. The prose slides between the
    two without saying so — one sentence fixes it, e.g.: *"Under the matched-folds protocol used to
    compare all five estimators (Figure 6), the power-only estimator scores 13.1% versus the MLPs'
    10.4%."*

11. **"the MLP … takes in the same inputs"** — this is your own Comment 1 ("not actually the
    same"). Precise version: the pure MLP takes **seven log-scale features derived from the same
    session totals** (net energy, duration, DRAM TB, NVLink TB, average power, energy-per-DRAM-TB,
    NVLink-per-DRAM-TB) and predicts log TFLOPs; the residual MLP instead takes six dimensionless
    intensity ratios and multiplies the power-only linear estimate by a learned correction factor
    (Figure 4 shows both).

12. **"I stick with the V100's default, FP16."** FP16 isn't a hardware default — it's the workload
    script's choice (and the precision that engages the V100's tensor cores, which is how mixed-
    precision training was actually run on V100s). Suggest: *"I stick with FP16, the precision that
    drives the V100's tensor cores and was standard for training on this hardware."*

13. **"the 3 main configurations of the estimator (3-inputs, power-only-input, and MLP)"** — worth
    specifying that the deployed MLP is the **residual** variant, since the pure MLP behaves
    differently under attack (it's fooled even harder on S3/S4).

14. **"I start from an existing benign workload in the evaluation set"** — the S3/S4/S5 parents
    are benign configs from the 88-workload pool; under the 200-random-splits protocol each pool
    workload is sometimes calibration, sometimes evaluation. "From the benign workload pool" is the
    accurate phrasing.

15. **Discussion: "most adversarial strategies fail to significantly increase the estimator's
    error apart from batch size inflation."** Atypical attention-head geometry also moves the
    estimators materially (MLP −30 pp at n_head = 1; the linear estimators over-report ~+10/+20% at
    n_head = 64). Suggest "…apart from batch-size inflation and atypical attention-head geometry."

16. **Equation (1) formatting**: mixed `·` and `*` and a stray double space ("E_PER_NVL * NVL_TB -
    P_overhead") — make the operators consistent.

17. **"92.5% … active at least 80% of the time"** — no issue; verified: the frontier gate,
    thresholds (start: util > 5% for 2 polls; stop: 3 polls < 5%), the 1.5 s poll interval, the 2 Hz
    calibration sampling, the ⅔/⅓ split (59/29), 200 splits, 88 workloads × 10 trials, the idle
    baseline being the calibration-phase constant (the daemon logs but does not use a re-measured
    quiet-period baseline), NVLink at ~25 GB/s per direction per link (×6 links), Table 1's value
    sets against the 91-config CSV, Table 2's config sets against `red_team/redteam_configs.py` and
    the S5 records, the split run fragmenting into exactly 4 sessions with 3 pauses, and the 2×
    GPU-hour budget (all S3/S4/S5 attacks land at ≤1.11× benign GPU-hours) — all of these are
    consistent with the code and data. The "<10% on the Jetson" claim is consistent with the Jetson
    report's figures (most workloads inside the ±10% band).

## 5. Resolving your two in-doc comments

**Comment A — "ideally the 'why do these hyperparameter values make sense?' question is answered
in the references"** (anchored on the calibrate/evaluate-split paragraph; the fix belongs at the
"(cite, cite, cite)" sentence just above Table 1). Replace:

> *"I tried to have realistic hyperparameter selections based on what makes sense in these
> transformer/LLM papers (cite, cite, cite), but just to double check, …"*

with:

> *"I based the hyperparameter selections on the shape conventions of the transformer literature:
> the original transformer block sets d_ff = 4·d_model and a per-head dimension of 64 [13], the
> GPT-3 family reports the width/depth/head/batch combinations used in production-scale training
> runs [14], and the scaling-laws literature finds that loss depends only weakly on the exact
> width/depth shape at fixed compute [15] — which is what makes sweeping many shapes at similar
> compute a reasonable stand-in for real training runs. But just to double check, …"*

Every row of Table 1 is then covered: d_ff is 4·d_model throughout [13], head_dim clusters around
64 (32–256 to probe the convention's edges) [13], d_model/n_layers/n_heads/seq_len are scaled-down
points on the GPT-3 grid [14], and the freedom to mix them comes from [15]. If you want the
justification visible at the table itself, add one sentence under Table 1: *"Values follow the
d_ff = 4·d_model and head_dim ≈ 64 conventions of [13], with widths, depths, and sequence lengths
taken from the lower end of the production shapes in [14]."*

**Comment B — "need to specify here because it is not actually the same"** (anchored on "takes in
the same inputs" in Improvement #2). The MLP sees the same three raw signals but not the same
inputs. Replace:

> *"So, I trained a simple Multi-Layer Perceptron (MLP) model which takes in the same inputs and
> runs them through two hidden layers and outputs an estimated FLOP count. I took approaches to
> using the MLP: (1) using the output FLOP directly as the estimate or (2) using the outputted FLOP
> count in combination with the power-only estimator."*

with:

> *"So, I trained a simple Multi-Layer Perceptron (MLP) on features derived from the same three
> signals: rather than the raw session totals the linear estimator uses, the MLP takes log-scale
> features of them — net energy, duration, DRAM terabytes, NVLink terabytes, average power, and
> ratios among these — and runs them through two 32-unit hidden layers. I took two approaches to
> using the MLP: (1) a pure MLP whose output is the FLOP estimate directly, and (2) a residual MLP
> that instead outputs a multiplicative correction applied to the power-only estimator, so that a
> zero correction recovers the linear estimate exactly. Figure 4 shows the architecture of these
> two approaches."*

(The exact feature lists — 7 log/ratio features for the pure MLP, 6 dimensionless ratios for the
residual — are in the Figure 4 caption, so the prose can stay at this level of detail.)

---

### Typos (quick list)

- "employed **be** the model developers" → by
- "a **mico**-pause" → micro-pause
- "full replica **over** the model" → of the model
- "I took **approaches** to using the MLP" → I took **two** approaches
- "**internation** agreement" → international
- "budget **constrains**" → constraints
- "a **trianing** run" → training
- "wide-set of hyperparameters" → wide set
- "The overall results **was**" → were (or "The overall result was")
- "employed … **by** the model developers" sentence also reads better as "…employed by an
  adversarial model developer."

---

## 6. Expanded transition into Figure 6

The single sentence before Figure 6 ("Figure 6 below gives the overview of the different
estimators tried out.") is the natural place to (a) reconcile the two CV protocols (judgment call
#10 above) and (b) close the arc of the two improvements before pivoting to the red team.
Suggested replacement paragraph:

> *"Figure 6 below summarizes all five estimator configurations tried in this work. For this
> comparison, every estimator is scored on identical held-out folds so the errors are directly
> comparable; under this matched-folds protocol the linear estimators' numbers shift slightly from
> the 200-split figures quoted above, to 13.1% for power-only and 14.9% for the two multi-signal
> variants. The ordering tells the story of the two improvements: adding memory and interconnect
> inputs to the linear estimator makes it worse, while replacing the linear form with a learned one
> takes the error from 13.1% down to 10.4% — within half a point of the 10% goal. The pure and
> residual MLPs land at identical accuracy, but they get there differently: the pure MLP is free to
> learn any mapping from the sensor features, while the residual MLP is anchored to the power-only
> physics estimate and only learns a multiplicative efficiency correction on top of it. Since
> accuracy does not separate the two, I carry the residual MLP forward as the deployed neural
> estimator, on the grounds that an estimator which degrades gracefully back to the physics formula
> is easier to reason about — and, as the next section shows, the flexibility that buys the MLP its
> accuracy also opens a new attack surface."*

Trim the last clause if you'd rather not foreshadow the red-team result here; the paragraph stands
without it.
