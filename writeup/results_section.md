# Results (sample draft)

*Drop-in section for the report. Numbers come from the actual experiment
artifacts in the repo (`eval_results_v2.txt`, `eval_generalization.txt`,
`adversarial_results.txt`, `drift_results.txt`, `eval_results_15w.txt`,
`eval_results_25w_trial{2,3}.txt`, `bias_report.txt`); figures
are in `writeup/fig1..6*.png`. One correction to carry into the setup notes:
the Jetson's on-board sensor is an **INA3221** (three-channel), not an INA219 —
the estimator reads its shared `VDD_CPU_GPU_CV` rail.*

---

## Results

### What the monitor sees

Figures 4 and 5 show one complete monitored session: a ~9-minute training run
bracketed by idle periods, as observed through the two signals the estimator
consumes. Both signals switch sharply — rail power (Figure 4) steps from
~0.7 W idle to a steady ~8.5 W, and DRAM bandwidth (Figure 5) from near zero
to ~40% of peak — which is what makes session detection reliable and the
energy integral well-defined. The bandwidth axis is worth a note: the raw
activity counter reads an arbitrary ~0.8% here, and the ~40% comes from
correcting it with the independently measured scale factor (k = 0.0133) and
the 102 GB/s peak at the loaded EMC clock. That corrected value agrees with
the ~39% that tegrastats' own EMC field reports for the same workload — two
independent instruments landing on the same number, cross-validating the
scale calibration. The estimator's inputs are exactly these traces: net
energy is the area between the power curve and the calibrated idle baseline
over the detected session.

**Figure 4.** `fig4_power_timeseries.png` — rail power across
idle → training → idle for a d384_b16_s128_L3 run, with the calibrated idle
baseline marked.

**Figure 5.** `fig5_dram_timeseries.png` — scale-corrected DRAM bandwidth over
the same session.

### Estimation accuracy

The estimator was calibrated and evaluated on a 38-workload sweep of simulated
transformer training runs against a single idle-power baseline measured once at
the start of the sweep, mirroring a verifier that cannot re-baseline a busy
cluster. On the 21 FP32 workloads that saturate the GPU (≥80% average
utilization — the frontier regime the estimator is scoped to), the deployed
2-parameter energy model estimates total training FLOPs to within **8.8% of
ground truth on every workload** (mean error 4.0%), including workloads whose
attention-head count, feed-forward width, sequence length, and optimizer were
never varied in earlier calibrations (Figure 1). On a randomized train/test
split, held-out workloads reach at most 8.95% error, passing the <10% target.

**Figure 1.** `fig1_per_workload_error.png` — signed estimation error per
workload, grouped by model size (d_model), all 21 FP32 frontier configs, with
the ±10% target marked.

### How robust is that number?

A single passing split can be luck, so I resampled the train/test split 200
times and refit from scratch each time (Figure 2). The held-out error
distribution centers near 9%, and **13.5% of splits exceed the 10% target**
(worst: 11.6%). The honest headline is therefore "typically ~9%, occasionally
~12%" rather than "always under 10%". Holding out entire workload *families*
(all 6-layer models, all d_model≥640, etc.) rather than random subsets keeps
errors under 10% for every family except two marginal cases (wide feed-forward
at 10.1%, small d_model at 10.8%), so the model extrapolates across the
transformer family rather than memorizing configurations. The small-model bias
is visible in Figure 1: the d256 family reads uniformly high, a family-level
offset rather than noise. The measurement
pipeline itself is repeatable: on one day the idle baseline reproduced across
three independent sweeps to within 0.6 mW (642.0 / 642.0 / 642.6 mW), though
replicate sweeps a week later measured 563.6 / 562.7 mW — a day-level shift of
~79 mW, two quantization steps of the current sensor. The calibration absorbs
it: the refitted energy constant moved less than 3.5% (5.53–5.72 J/TFLOP
across the three full sweeps) and every sweep passed the same <10% held-out
verdict. A 49-minute sustained run — 15× longer than any calibration
workload — scored 5.9% whole-run error, with a slow thermal drift
(per-5-minute-window error rising from 4.7% to 8.6%) that would matter for
multi-hour runs.

**Figure 2.** `fig2_split_stability.png` — distribution of held-out max error
over 200 resampled calibration splits.

### Which hyperparameters bias the estimate, and in which direction?

The ~10% error budget is not spent on measurement noise — it is spent on
reproducible, hyperparameter-linked bias. To separate the two, the full FP32
calibration sweep was run three times end-to-end (independent runs on separate
days, each with its own idle baseline), and every workload's error was
recomputed leave-one-out within its own sweep: each run is scored by a model
fit to the other 20 runs of the same sweep, so sweep-level effects cancel and
a config's own leverage cannot hide its bias. The result (Figure 6) is that a
config's signed error is essentially a fixed property of the workload: across
trials the 21 configs' errors correlate at r = 0.96–0.99, every config keeps
the same sign in all three sweeps, and the median run-to-run spread within a
config is ~0.5 points, against biases of up to ±9%.

Direction sorts cleanly by hyperparameter. Narrow models **overestimate**:
every d_model=256 run in all three sweeps reads high (+6.8% mean over 12 runs;
the worst config sits at +9.1 ± 0.1%). Wide feed-forward layers
**underestimate**: every run at ff=3072/4096 reads low (−6.1% and −8.9% means,
6 runs, none over). Batch size, depth, and optimizer are benign — level means
within ±1.1 points of zero and over-estimation rates near a coin flip — while
sequence length shows a mild upward gradient (−0.2% at 64 tokens to +4.8% at
512) and the single 8-head config reads +4.9%. Rank correlations point at the
mechanism: signed error tracks d_model (ρ = −0.52), FFN width (ρ = −0.45), and
measured arithmetic intensity, i.e. FLOPs per DRAM byte (ρ = −0.46), while
tracking duration and GPU utilization not at all. That is exactly how a joule
meter should fail: narrow models move more bytes per FLOP, and the extra DRAM
energy is misread as compute (overestimate); wide-FFN matmuls are the most
energy-efficient FLOPs in the pool and read low. Two caveats keep this
honest: the sweep's axes are partially confounded (the smallest models are
also the least arithmetically intense), so per-axis means are descriptive
rather than causal — the family-holdout refits above are the confound-aware
check, and they agree — and the seq-512 and 8-head cells are single configs.
The 15 W replication reproduces the same ordering (d256 +7.3% with every run
over; wide-FFN −5.7/−8.5%), so the pattern belongs to the method, not to one
operating point. For a red team the asymmetry is what matters: the worst
*benign* under-read any hyperparameter buys is ~9%, versus 2–3.5× for the
precision channel below.

**Figure 6.** `fig6_bias_by_axis.png` — signed leave-one-out estimation error
by hyperparameter axis; 63 FP32 frontier runs from three independent 25 W
calibration sweeps (marker = trial), green dash = level mean.

### Unusual and adversarial workloads

Two probes tested behavior off the benign training distribution (Figure 3a). A
*memory-spoof* workload — DRAM traffic with near-zero FLOPs, held at 98% GPU
utilization so it passes the activity gate — fools the energy-only estimator
into reporting 8.3 TFLOPs against 1.8 true. A second estimator that subtracts
measured DRAM-traffic energy (read from the Jetson's memory-controller activity
counters) collapses to −96 TFLOPs on the same run: not a usable count, but an
unmistakable flag that the energy is not FLOPs. On a cache-resident matmul
control it stays sane (17% high vs 4% for the energy-only model), so the byte
term currently serves as a spoof tripwire rather than a meter. Its fitted
coefficient converts to 61.5 J per true TB moved — inside the plausible range
for LPDDR5 — after benchmarking the activity counter's scale against a known
byte volume.

The strongest evasion found is **numeric precision** (Figure 3b). Power reveals
joules, not FLOPs, and tensor-core precisions do 2–3.5× more FLOPs per joule
than the FP32 the estimator was calibrated on: TF32 runs read at 0.26× their
true FLOPs, and FP16/BF16 runs read as *negative* (below the detection floor)
while also dropping below the 80%-utilization gate at this model scale. A
red team can therefore under-report by choosing precision, and no power-only
monitor can see it. Estimates must be read as a **lower bound with respect to
precision**; closing this channel (e.g. via clock/power signatures that
distinguish tensor-core execution) is the top item for future work.

**Figure 3.** `fig3_evasion.png` — (a) adversarial probe: ground truth vs both
estimators on a memory spoof and a compute-dense control; (b) mean estimate as
a fraction of true FLOPs per precision.

### Does the method transfer off this exact setup?

As a proxy for new hardware, I reran the entire calibration from scratch with
the Jetson switched from its 25 W to its 15 W power mode — a different
clock/voltage operating point. The methodology reproduced (held-out max error
9.2%, all 22 frontier workloads ≤8.9%) but with very different fitted
constants (3.92 vs 5.72 J/TFLOP), and the 25 W constants applied blind at 15 W
mis-read workloads by −95% on average. The calibration *procedure* transfers;
the *numbers* do not, so the monitor records a hardware/power-mode fingerprint
at calibration time and refuses to trust constants on a mismatched system.
Replicating the procedure on the V100 setup is the natural next test.

---

*Notes for you (not part of the section):*
- *The bias subsection's numbers come from `bias_report.txt` — regenerate via
  `eval_power_monitor.py --bias-report <the three 25W *_records.json files>`;
  the 15 W comparison is `bias_report_15w.txt`. Figure 6 follows the
  "numbered last, placed where written" convention like figs 4/5.*
- *Figures 4/5 are numbered last to avoid renumbering, but they're written to
  OPEN the Results section — if you renumber, they become Figures 1/2 and the
  rest shift.*
- *Figure 5's DRAM axis is scale-corrected by the measured k=0.0133 and uses
  the 102.4 GB/s peak of the loaded 3199 MHz EMC clock as its denominator (the
  68 GB/s in ORIN_PROFILE is the original non-Super board's peak — stale for
  display, harmless in the estimator where the scale is absorbed into
  E_PER_TB_J). Keep "scale-corrected" in the caption so the axis is honest.*
- *Fig 2's "reported split (8.95%)" marker assumes you quote the seed-12345
  split in the accuracy subsection; if you quote the ship-fit 8.8% instead,
  move the marker.*
- *If the report needs fewer figures, Figure 2 folds into prose most easily.*
- *Your outline's Results placeholder mentioned "making sure the baseline power
  and util reads are good" — that's the baseline-repeatability sentence; say
  the word if you want a small figure for it instead.*
- *The daemon's start/stop detection and the non-ML false-positive question
  from your Methodology notes remain untested — I kept them out of Results so
  the section only claims what was measured.*
