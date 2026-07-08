# Results (sample draft)

*Drop-in section for the report. Numbers come from the actual experiment
artifacts in the repo (`eval_results_v2.txt`, `eval_generalization.txt`,
`adversarial_results.txt`, `drift_results.txt`, `eval_results_15w.txt`); figures
are in `writeup/fig1..3*.png`. One correction to carry into the setup notes:
the Jetson's on-board sensor is an **INA3221** (three-channel), not an INA219 —
the estimator reads its shared `VDD_CPU_GPU_CV` rail.*

---

## Results

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
workload, all 21 FP32 frontier configs, with the ±10% target marked.

### How robust is that number?

A single passing split can be luck, so I resampled the train/test split 200
times and refit from scratch each time (Figure 2). The held-out error
distribution centers near 9%, and **13.5% of splits exceed the 10% target**
(worst: 11.6%). The honest headline is therefore "typically ~9%, occasionally
~12%" rather than "always under 10%". Holding out entire workload *families*
(all 6-layer models, all d_model≥640, etc.) rather than random subsets keeps
errors under 10% for every family except two marginal cases (wide feed-forward
at 10.1%, small d_model at 10.8%), so the model extrapolates across the
transformer family rather than memorizing configurations. The measurement
pipeline itself is repeatable: the idle baseline reproduced across three
independent sweeps to within 0.6 mW (642.0 / 642.0 / 642.6 mW), and a
49-minute sustained run — 15× longer than any calibration workload — scored
5.9% whole-run error, with a slow thermal drift (per-5-minute-window error
rising from 4.7% to 8.6%) that would matter for multi-hour runs.

**Figure 2.** `fig2_split_stability.png` — distribution of held-out max error
over 200 resampled calibration splits.

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
