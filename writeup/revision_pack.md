# Revision pack for report v3 (call feedback, 2026-07-16)

*Each section is drop-in text written in the draft's voice, with its insertion
point named. Citations `[n]` refer to `bibliography.md`. Figure numbers refer
to the docx's own numbering (Fig 1/2 = traces, Fig 3 = per-workload error,
Fig 4 = split histogram, Fig 5 = adversarial panels).*

---

## 1. Background: the hyperparameter dimensions
*(insert into "Constructing Sample Workloads", after the first sentence — and
note the draft's current list omits `d_model`, the most important axis; the
list below fixes that)*

The sweep varies every knob of a transformer training run that plausibly
changes the relationship between FLOPs performed and energy consumed:

- **Model width (d_model):** the size of the vector representing each token
  as it flows through the model. Width sets the size of nearly every matrix
  in the network, so it is the strongest single lever on both FLOPs and
  memory traffic.
- **Depth (number of layers):** how many identical transformer blocks are
  stacked; FLOPs scale linearly with depth.
- **Attention heads:** how many parallel attention operations split up the
  width. Head count barely changes the FLOP count but changes how the work is
  divided into GPU kernels.
- **Feed-forward (MLP) width:** the inner dimension of the two-layer MLP in
  each block — typically the single largest matrix multiplication in the
  model.
- **Sequence length:** how many tokens are processed together; attention cost
  grows quadratically with it, everything else linearly.
- **Batch size:** how many sequences are processed per step; more batch means
  larger, more efficient matrix multiplications.
- **Optimizer (AdamW vs. SGD):** changes the elementwise bookkeeping done
  between steps, a test that the estimator is not sensitive to non-matmul
  overhead.
- **Numeric precision (FP32 / TF32 / FP16 / BF16):** the number format the
  GPU computes in. Lower precisions execute on tensor cores and perform more
  FLOPs per joule — which is exactly why precision turns out to matter (see
  Adversarial Testing).

## 2. Background: pre-training vs. inference
*(insert as a short paragraph in Motivation, after "...verifying the number of
FLOPs used in an LLM training run..." — or as the opening of Constructing
Sample Workloads)*

The verification target here is *training*, not inference, and the two look
very different to a hardware monitor. Training processes large batches
continuously for days or months, and each step runs the model forward, then
backward to compute gradients, then updates the weights — roughly three times
the FLOPs of the forward pass alone, executed at near-constant hardware
saturation. Inference runs only the forward pass, in short bursts that follow
user demand. Compute-threshold laws are written against cumulative *training*
FLOPs [4][5], so the estimator is designed for (and calibrated on) sustained,
GPU-saturating training-shaped workloads; prior work has shown the
training/inference distinction is itself recoverable from GPU telemetry [8].

## 3. Background: workload start/end detection + the explicit time parameter
*(replaces/extends the "Detecting Workloads" paragraph; also makes t explicit
per the call feedback)*

In order to estimate the number of FLOPs in a workload, the estimator needs
to know when a workload begins and ends. The daemon polls GPU utilization
every 1.5 seconds and applies hysteresis in both directions: a workload
*starts* after two consecutive polls above 5% utilization and *ends* after
three consecutive polls below it. The asymmetry is deliberate — a momentary
utilization blip should neither open a phantom session nor split one training
run into several — and everything the estimator integrates is defined over
this detected window. Formally, **t** in equation (1) is the duration of the
detected session in seconds; E_net is the sensor power minus the idle
baseline integrated over exactly this window, and TB_moved is the DRAM
traffic accumulated over the same window. At session end the daemon converts
the accumulated energy and bytes into one FLOP estimate for the workload,
which is then compared against the FlopCounterMode ground truth.

## 4. Equation (1) / figure consistency fix
*(one sentence to add right after equation (1)'s term definitions, plus a
label fix on the accuracy figures)*

Add after the term list:

> In deployment I run two variants of equation (1) side by side as a matched
> set: the full 3-parameter model shown above (fitted constants 1.46 W,
> 4,622 J per actmon-TB, 5.61 J/TFLOP) and a simpler 2-parameter variant with
> the memory term removed (E_per_TB = 0; 3.75 W, 5.72 J/TFLOP). All accuracy
> results in this report (Figures 3, 4, and 6) present the 3-parameter
> estimator — the same model whose byte term catches memory spoofing
> (Figure 5a) — and Figure 7 overlays both variants. Carrying the byte term
> costs some accuracy on benign workloads, because FLOPs and bytes rise
> together on normal training runs and the memory coefficient is therefore
> weakly determined; Figure 4 shows that cost as split sensitivity.

## 4b. Equation (1): end-of-run framing (and the fix for the stray "/s")

Present the end-of-run form; it matches how accuracy is scored and what
compute-threshold policy counts. The unit itch behind v4's "/s" edit is real
but the fix is different: the overhead term is a *power* (W), so it needs
"· t", not everything else needing "/ s". Rename it P_overhead:

> TFLOPs = (E_net − E_per_TB · TB_moved − P_overhead · t) / E_per_TFLOP   (1)
>
> computed once per detected workload over the session window from the
> daemon: t = session duration (s); E_net = sensor power minus idle
> baseline, integrated over the session (J); TB_moved = DRAM traffic
> accumulated over the session; P_overhead (W) = fixed power cost of being
> active at all; E_per_TB, E_per_TFLOP (J) = calibrated memory and compute
> energy costs.

Follow with one sentence on the streaming implementation (true, and a
selling point for a monitor):

> In deployment the daemon evaluates equation (1) incrementally at every
> 1.5-second poll and accumulates the result — mathematically identical
> since the model is linear in its measured inputs, but it means the monitor
> holds a running FLOP estimate throughout the run, so a threshold violation
> can be flagged while the run is still in progress.

## 5. Sensor findings: does the Orin expose separate CPU/GPU power? (call item)
*(insert into "Hardware and Observable Signals", extending the Power bullet)*

A natural question is whether the CPU and GPU can be metered separately
rather than modeled. On this board, they cannot: the Orin Nano exposes
exactly one INA3221 monitor (a three-channel, shunt-based power sensor at I2C
address 0x40 [11]) whose channels are VDD_IN (total module input),
VDD_CPU_GPU_CV (CPU, GPU, and vision accelerators on one shared rail), and
VDD_SOC (memory subsystem and SoC engines) [10]. I verified this by
enumerating every hwmon sensor the kernel exposes — those three rails are all
that exists, so no subtraction of channels can isolate the GPU; the estimator
must attribute the shared rail's energy, which is what the overhead and
memory terms in equation (1) do, and why accuracy is only claimed for
GPU-saturating workloads. Notably this is a limitation of the *small* Orin:
the AGX Orin devkit splits its rails into VDD_GPU_SOC and VDD_CPU_CV [10], so
on larger Jetsons (and on server GPUs with per-board sensors such as the
V100 node planned in Next Steps) the same method gets a strictly cleaner
signal.

## 6. Captions / labeling for Figures 3 and 4 (call item: "clarify")

**Figure 3 (per-workload error; regenerated as `fig1_per_workload_error.png`):**
now shows the **3-parameter estimator** (equation (1) as written), *signed*
(↑ over, ↓ under), and **held out**: each workload is scored by a
leave-one-out fit calibrated on the other 20 workloads (same method as
Figures 6 and 7). Suggested caption:

> **Figure 3.** Signed leave-one-out estimation error of the 3-parameter
> estimator (equation (1)) on all 21 FP32 frontier workloads, grouped by
> model width (d_model); each workload is predicted by constants calibrated
> without it. Positive bars overestimate, negative bars underestimate;
> dashed lines mark the ±10% target. Mean error is 4.0%; 19 of 21 workloads
> land within ±10% (worst −12.9%).

**Figure 4 (split histogram; `fig2_split_stability.png`):** now the
3-parameter estimator; the x-axis reads "max % error over the held-out
workloads of one split". Suggested caption:

> **Figure 4.** Robustness of the calibration to the train/test split
> (3-parameter estimator). The calibration was refit from scratch 200 times,
> each on a different random two-thirds of the 21 frontier workloads and
> scored on the held-out third; each histogram entry is the worst held-out
> error of one refit. 59% of splits exceed the 10% target: on benign
> training runs FLOPs and DRAM bytes rise together, so the memory
> coefficient is weakly pinned by the data and the fit is split-sensitive —
> the accuracy cost of carrying the adversarially-motivated byte term (the
> 2-parameter variant fails only 13.5% of splits).

## 7. Hyperparameter → error breakdown (call item; already produced)

The content exists from the 2026-07-14 replication session:
`fig6_bias_by_axis.png` + the drop-in subsection "Which hyperparameters bias
the estimate, and in which direction?" in `results_section.md` (numbers from
`bias_report.txt`). **NB (2026-07-16): fig6 now shows the 3-parameter
estimator**, so quote the 3-param numbers: per-config signed error remains
reproducible across the three sweeps (config-wise correlation r = 0.72–0.92,
same sign in 76–90% of configs — somewhat noisier than the 2-param variant's
0.96–0.99); d_model trends over→under with width (+4.4% at d256 down to
−2.9% at d768); wide feed-forward underestimates (−4.4% at ff=4096); and two
axes the 2-param model saw as benign are NOT benign under 3-param — SGD reads
+5.8% vs AdamW −0.9%, and 8-head attention reads +4.3% (see the open-items
list in the cover message; the results_section.md subsection still quotes the
2-param numbers and needs the same rewrite if used).

Placement options in the docx:
- **(a) After Figure 4** (flows naturally from "some splits exceed 10%" into
  "and here is *where* that error lives") — the adversarial figure becomes
  Figure 6; or
- **(b) After Figure 5**, keeping existing numbering untouched and the new
  figure becomes Figure 6.
Option (a) reads better; option (b) is zero-renumbering.

## 8. New figures (produced by this revision)

- **`fig7_est_vs_truth.png`** — the actual FLOP numbers (call item): estimated
  vs. ground-truth TFLOPs for the 21 frontier workloads, y = x line with ±10%
  band, both estimator variants shown. Every point is **held out**: each
  workload is predicted by a fit calibrated on the other 20 workloads
  (leave-one-out), so no point was seen by the model that scored it.
  Suggested caption:

  > Estimated vs. true TFLOPs on the 21 FP32 frontier calibration workloads.
  > Each point is predicted by constants calibrated *without* that workload
  > (leave-one-out). Filled circles: 2-parameter estimator; open squares:
  > 3-parameter (EMC) variant. The shaded band is ±10% around perfect
  > estimation.

- **`fig_pipeline.png`** — framework diagram (call item): two lanes,
  one-time calibration (idle baseline → byte-scale benchmark → 26-config
  sweep → constant fitting) feeding its constants into the deployed daemon
  loop (poll sensors → detect session → integrate energy and bytes →
  equation (1) → estimate vs. ground truth). Suggested placement: top of
  Methodology.

## 9. Small corrections noticed while reading v3

1. **"3-fold CV"** (Estimator Accuracy) — what actually runs: the overhead
   parameter is chosen by leave-one-out cross-validation on the training
   workloads, the fit is validated on a randomized ~1/3 held-out split, and
   Figure 4 repeats that with 200 resampled splits. Suggested wording: "I
   validate with a randomized train/test split (and leave-one-out
   cross-validation inside the fit), then refit the final parameters on all
   21 workloads."
2. **"3,000–4,000 steps"** — the configs actually run 900–4,800 steps, sized
   so every workload trains for roughly 2–4 minutes.
3. **"Chaurhuri et al." → "Chaudhuri et al."** [7].
4. **Idle power "642 mW"** — replicate sweeps run a week later measured
   563.6/562.7 mW: the baseline is stable within a day (<1 mW) but drifts by
   up to ~80 mW (two sensor quantization steps) across days. Worth one hedge
   sentence, because it strengthens the story: recalibrating from scratch on
   the drifted baseline moved the fitted energy constant by <3.5% and both
   replicate calibrations passed the same <10% verdict.
5. **Methodology's E_per_TB bullet** says the constant is set "by
   transferring a known amount of bytes and fitting to the measured energy" —
   that describes the *actmon scale factor* benchmark. E_per_TB itself is
   fitted jointly with E_PER_TFLOP against known-FLOP workloads (the
   Calibration section already says this correctly). Suggested Methodology
   wording: "a constant fit during calibration, jointly with E_PER_TFLOP,
   against workloads with known FLOP counts."
6. **"upper limit for the Nano's power consumption is 25 W"** — 25 W is the
   configured power-mode budget (nvpmodel "25W" mode), not a measured
   maximum; suggest "the board runs in its 25 W power mode".
