# Methodology (sample draft)

*Drop-in section. Every experimental design choice is stated with its
rationale, since that's what you wanted ironed out — the "why" clauses are the
load-bearing part; cut detail before cutting those. Corrections carried
through: the sensor is an INA3221 (not INA219), and the board runs in its 25 W
nvpmodel mode.*

---

## Methodology

### Hardware and observable signals

All experiments run on an NVIDIA Jetson Orin Nano 8 GB developer kit (1024-core
Ampere GPU, JetPack 6.2, 25 W power mode) — essentially a Raspberry Pi with a
real CUDA GPU, which makes it a cheap, fully instrumentable stand-in for one
node of a monitored cluster. The blue-team monitor is restricted to signals a
treaty verifier could plausibly obtain without reading the workload's code or
self-reports: power draw, GPU utilization, clock frequencies, and DRAM traffic.

**Power.** The board's INA3221 sensor meters the `VDD_CPU_GPU_CV` rail. A
verifier with physical access would read it over I²C from a peripheral device;
I read the software equivalent, the kernel's hwmon interface — voltage from
`/sys/bus/i2c/drivers/ina3221/…/hwmon/hwmonN/in2_input` (mV) and current from
`curr2_input` (mA), multiplied to milliwatts and polled at 2 Hz. The hwmon
index changes across reboots, so the monitor resolves the path by rail label at
startup rather than hard-coding it. One consequence shapes the whole design:
this rail powers the CPU, GPU, and vision cores together — no GPU-only rail
exists — so the monitor observes *system* energy, not GPU energy. This is why
the estimator is scoped to high-utilization workloads (below), where GPU
compute dominates the rail.

**Memory bandwidth.** None of the standard tools report DRAM utilization on
this JetPack release (jtop returns zero; tegrastats' EMC field never
populates), but the Tegra memory-controller activity monitor is exposed in
debugfs: dividing `/sys/kernel/debug/bpmp/debug/actmon/mc_all_last_prd_activity`
by the current EMC clock (`…/clk/emc/rate`) yields a per-poll bandwidth-
utilization fraction covering all memory clients. The counter's absolute scale
is undocumented, so I calibrated it by streaming a known byte volume (large
tensor copies) past it: it reads a stable, linear 1.33% of true traffic. Only
linearity matters for estimation — the scale constant is absorbed into the
fitted coefficient — but measuring it lets that coefficient be checked against
physical DRAM energy.

### Workloads and ground truth

Red-team samples are simulated LLM training runs: a PyTorch transformer
encoder trained on synthetic data, swept across model width (d_model 128–768),
batch size (8, 16), sequence length (64–512), depth (3 or 6 layers), attention
heads (2, 4, 8), feed-forward width (1–8× d_model), optimizer (AdamW, SGD),
and numeric precision (FP32, TF32, FP16, BF16) — 38 configurations. Each axis
is a knob a training team could turn that plausibly changes the FLOPs↔energy
relationship; sweeping them is the defense against calibrating an estimator
that only works on one shape of workload. Step counts are sized so every run
lasts 2–4 minutes of active training: long enough to be dominated by
steady-state compute rather than startup transients, to yield several hundred
power samples per run, and to resemble a real training job rather than a
microbenchmark.

Ground truth comes from PyTorch's `FlopCounterMode`, a dispatch-mode context
manager that intercepts every operator executed during one forward+backward
step and accumulates an analytic FLOP count from the operator's tensor shapes
(e.g. 2·m·n·k for a matmul, with registered formulas for convolutions and
attention). Because it counts from shapes, the number is the *nominal
algorithmic* FLOP count — independent of numeric precision or kernel choice —
which is exactly the quantity compute-threshold policies are written in.
The per-step count is measured once on the live model and multiplied by the
step count; optimizer arithmetic is elementwise, under 1% of the total, and
excluded. During calibration, power sampling is gated on a start marker the
workload prints after CUDA initialization and warmup, so the measured energy
covers exactly the steps the ground truth counts.

### The estimator

The primary estimator is deliberately minimal — a two-parameter active-energy
model. Net energy (measured power minus an idle baseline, integrated over the
workload) is converted as:

```
TFLOPs = (E_net − POWER_OVERHEAD_W · t_active) / E_MARGINAL_J_PER_TFLOP
```

The physical claim is that at a fixed operating point, each FLOP costs a fixed
marginal energy, on top of a fixed power overhead that the active system burns
regardless of useful work (memory refresh, fans, the CPU feeding the GPU).
Both constants are *fit* to measured runs rather than derived from chip
specifications, because the shared rail and DVFS behavior make first-principles
numbers unreliable — and this is the choice that makes the constants
device-specific by construction (see Results, hardware transfer).

A second estimator runs in parallel and additionally subtracts a memory-energy
term, `E_PER_TB_J · TB_moved`, using the measured DRAM traffic. Its purpose is
adversarial rather than accuracy: an energy-only estimator can be spoofed by a
workload that burns joules on memory traffic instead of FLOPs, and the byte
term is what notices that the energy is in the wrong place. It is evaluated as
an A/B alongside the primary estimator, never silently substituted.

### Calibration and evaluation design

Each design choice mirrors a constraint a real verifier would face:

- **One idle baseline, measured once at sweep start.** A verifier monitoring a
  busy cluster cannot re-measure a clean idle baseline between jobs, so the
  calibration doesn't either; baseline drift over the sweep stays in the error
  budget on purpose.
- **Relative (percent-error) least squares.** The policy question is "how many
  FLOPs, within a tolerance" — absolute-energy least squares would let the
  largest runs dominate the fit and quietly sacrifice accuracy on smaller ones.
- **The overhead parameter is chosen by leave-one-out cross-validation.**
  Frontier runs are near-collinear in (FLOPs, duration), so the two parameters
  trade off almost freely; any in-sample objective just pushes the overhead to
  the edge of its search range. Held-out prediction error has a genuine
  interior optimum.
- **Only high-utilization runs (≥80% average GPU load) count toward the fit
  and verdict.** This is the threat model — an unauthorized frontier training
  run saturates its hardware — and also an honest accounting of the shared
  rail: at partial load, the rail's energy cannot be attributed to GPU FLOPs.
  Low-utilization runs are still executed and reported so the boundary is
  visible, and the scope limit is acknowledged below as an attack surface.
- **A randomized train/test split, re-drawn every calibration run,** so that a
  reported pass is never a memorized split. Because a single split can still
  be lucky, the evaluation also refits across 200 resampled splits (reporting
  the distribution of held-out error and of the constants themselves) and
  holds out entire workload *families* — all 6-layer models, all wide-FFN
  models, all runs at one d_model — to test extrapolation rather than
  interpolation. These three checks exist specifically to answer the
  overfitting question a skeptic should ask of a 38-workload calibration.
- **Constants are treated as a matched set, fingerprinted to the device and
  power mode.** The monitor records the device model, L4T release, and nvpmodel
  power mode at calibration time and refuses to trust constants on a
  mismatched system, because (as Results shows) they are meaningless even one
  power mode away.

### Session detection

The deployed monitor is a daemon that polls the sensors every ~1.5 s. A
workload session opens after two consecutive polls above 5% GPU utilization
and closes after three consecutive quiet polls — hysteresis in both directions
so that momentary blips neither open nor split sessions. Energy, DRAM bytes,
and both FLOP estimates are integrated over the open session and logged per
session. Two aspects are untested and flagged as future work: the false-positive
rate against non-ML GPU activity, and the inverse attack — deliberately pausing
a training run so one job appears as several smaller sessions, none of which
individually crosses a reporting threshold.

---

*Notes for you (not part of the section):*
- *Your outline's "polls for a power spike" is slightly off from the
  implementation — the daemon triggers on GPU utilization, not power; I wrote
  the section to match the code.*
- *The Claude Code acknowledgment from your outline isn't in the draft — it
  reads more naturally in the acknowledgments footer than in Methodology; say
  the word if you want it inline instead.*
- *If the report needs trimming, the sysfs paths can drop to a footnote; keep
  the rationale bullets in "Calibration and evaluation design" intact — they
  are the section's argument.*
