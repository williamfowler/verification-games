# Skeptical review — Phases I & II (what feels off)

A deliberately adversarial read of the experiment as a whole, backing up to first
principles. Written after the 10-trial pass, which sharpened several of these. The
point is not that the work is wrong — the mechanics are sound and honestly reported —
but that a reviewer would push hardest on whether the setup can support the *claims*.
Ranked by how much they'd worry me.

---

## 1. The estimator's own per-config error (~±15–25%) is bigger than most of what's being measured

This is the one that would make me most uneasy, and the 10-trial data makes it
concrete: **both "benign" parent configs under-report by −15%** (batch-8 and nhead-8,
±2 sd), systematically, across all 10 trials. So:

- The estimator isn't a meter that reads ~0 on honest runs and deviates under attack.
  It reads **−15% on honest runs of these shapes** and has a ±15–25% spread *across*
  shapes (the "benign band").
- A verification tool whose per-workload error is ±20% **cannot reliably distinguish a
  run at 0.9× a FLOP threshold from one at 1.3×** — which is the entire point of a
  treaty threshold. The precision is coarse relative to the decision it's meant to
  support. No amount of red/blue refinement fixes a meter this wide; that's a
  calibration-accuracy ceiling, and it's the real headline the project should confront.
- It also means "evasion" is being defined against a band so wide (±25%) that almost
  nothing crosses it — which is why *zero* configs "formally evade." The threshold for
  evasion is set by the estimator's imprecision, not by the attack.

**A reviewer's demand:** report the estimator's absolute per-workload error prominently
as the primary result, and frame the FLOP-verification question as "is this precision
usable for a threshold decision?" — before any red-team story.

## 2. The "attack" is largely the estimator's baseline error, not the adversary

Directly downstream of #1, and the single-trial write-up got this wrong. S4
batch-inflation was reported as "−21% under-report." But the parent (batch-8) already
under-reports −15%, so the **marginal effect of the batch knob is only ~−8%** (b8 −15%
→ b128 −23%). Most of the headline number is the estimator mis-fitting that config
family, not the attack. The updated `fig_redteam_signed.png` now measures the attack as
deviation from the dashed parent line — that's the honest quantity, and it shrinks the
"evasion" to something comparable to trial noise plus the benign spread. Same for S3:
the interesting move (h64 = +20%) is a +35% deviation from the −15% parent, but it's an
*over*-report. The attacks are real but small once you subtract what the estimator was
getting wrong anyway.

## 3. Calibration and evaluation come from one synthetic generator — "held-out" is within-distribution

Every Phase I config (all 91) is the same `TinyTransformer` from one parametric family,
and Phase II attacks are one-variable perturbations of it. So:

- "Generalizes to held-out workloads" means **held out from the same generator**, not
  to genuinely novel workloads. A real unseen run (different architecture, MoE, real
  tokenized data, a framework other than this DDP script) is nowhere in the data, and
  there's no reason to expect the ±20% to hold there — likely worse.
- The 200-split bootstrap resamples the *same 91 points*; its confidence intervals are
  within-distribution and overstate real generalization.
- The workload itself is a weak proxy for real training (encoder-only, no
  embedding/vocab, random inputs, `loss = mean`, DDP-only, ~10⁸ params on 2 GPUs). If
  the proxy's energy/memory/interconnect signature differs from real pre-training, the
  whole calibration is measuring the wrong thing — and none of the experiment can catch
  that, because there's no real-workload holdout.

## 4. Construct validity: energy tracks *hardware* FLOPs; the target is *algorithmic* FLOPs

The estimator predicts FlopCounterMode's algorithmic FLOP count from an energy signal
that physically reflects *hardware* work (tensor-core utilization, non-matmul ops,
recompute, padding). The ratio of the two is **workload-dependent — and that ratio is
exactly what S3/S4/S5 manipulate** (kernel efficiency, arithmetic intensity, V/f point).
So the attacks aren't exotic tricks against a good estimator; they're the estimator's
*definitional* soft spot. A skeptic would say the project has demonstrated that a
single-constant "J per algorithmic FLOP" is not a physical invariant — which is a real
finding, but it reframes the estimator as structurally limited rather than merely
under-hardened.

## 5. The defenses are tuned on the same data as the attacks they catch

The consistency gates are calibrated on the benign frontier *and* validated against the
specific known attacks; the arithmetic-intensity floor was set to catch b128 at the
cost of 6% false positives. That's fitting the defense to the threat model. The 8.8%
FP / "catches the worst" result is honest for *these* attacks, but it says little about
an attack not in the set — and same-person red/blue means the attack set is "what one
author thought of." The gate that matters most (arith_intensity) also rests on the
all-reduce ∝ params invariant, which is a **DDP-specific** artifact; under FSDP /
tensor-parallel / a non-standard sharding it changes, and the shield may evaporate.

## 6. External validity: one node, one GPU, one precision, short runs

- **Scale:** 2 GPUs, ~10⁸ params, 20–180 s runs. Frontier verification is thousands of
  GPUs, 10¹¹⁺ params, days — with thermal throttling, sustained-clock regimes,
  checkpointing, data-loading stalls, and 3D-parallel communication that this setup
  can't produce. The signal relationships are calibrated on short bursts.
- **Hardware/precision:** V100 fp16 only. The "precision channel is closed" result is
  entirely a Volta artifact (no bf16/tf32/fp8); it reopens on an Ampere+/H100 baseline.
  A treaty verifier faces heterogeneous hardware; nothing here speaks to transfer.
- **Measurement drift:** the S5 sweep's uncapped run read −20% weeks after calibration —
  baseline/thermal drift over time is itself an unmodeled error source, and a real
  monitor runs continuously across such drift.

## 7. Smaller things that would still draw a comment

- **Ad-hoc thresholds.** The 80% frontier gate, the 5%/3-poll session rule, the 2×
  budget, R_benign = pool median — all are chosen, not derived. Several attacks (split,
  throttle) target these specific numbers, and the defenses re-tune them; the results
  are somewhat about the arbitrariness of the thresholds.
- **Efficiency metric conflation.** R_benign as a pool median mixes intrinsic shape
  efficiency with attack overhead (already flagged for S1/S3), so "over budget" isn't
  always "wasteful attack."
- **Single-GPU live daemon.** Group B (split/throttle) estimates are ~⅓ of truth and
  the per-session baseline mis-fires (the est→0 artifact). Only the *session-count*
  claim is solid there.

---

## What would actually make me believe it

1. **Lead with the absolute precision** and state whether ±20% is usable for a threshold
   decision — that's the make-or-break, not the red-team drama.
2. **A genuinely out-of-distribution holdout:** at least one real training run (real
   data, a standard GPT/Llama-style decoder, ideally a different framework) never seen
   in calibration, scored cold. If the ±20% survives that, the generalization claim has
   teeth.
3. **Report every attack as a deviation from its own parent baseline** (as the updated
   figures now do), not from zero — so the estimator's inherent error can't be mistaken
   for an evasion.
4. **A blind red team**, or at least attacks designed by someone without the calibration
   data, to break the defense-overfitting critique.
5. **Frame the deliverable as a Swiss-cheese *layer* with a stated precision floor**, not
   a standalone FLOP meter — which the project's own conclusion already gestures at, but
   the results currently read as stronger than the ±20% meter supports.
