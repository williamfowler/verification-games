# Fill-ins for the bracketed comments in the draft

Drop-in text for each `[Claude …]` bracket, in document order, written to match
the draft's style and the notation of equation (1). Values are the deployed
constants (the 3-parameter fit matching equation (1), calibrated 2026-07-07 on
this board in its 25 W power mode).

---

## 1. "[Claude please continue with simple definitions of the equation terms]"

*(continues your E_net bullet; same format)*

**E_per_TB_J:** The calibrated energy cost of moving one terabyte of data
between DRAM and the chip. Multiplied by TB_moved, this removes the
memory-transfer energy from the total before anything is attributed to
compute.

**TB_moved:** The total volume of data transferred during the workload,
obtained by integrating the actmon activity signal over the workload's
duration. Unlike the calibrated parameters, this is measured fresh for every
workload.

**E_overhead:** The extra power the system draws whenever *any* workload is
active, regardless of how much compute it performs — the CPU running the
training loop, raised clock floors, the fan. Multiplied by the workload's
duration, it removes the "cost of being busy at all" from the energy total.

**t:** The active duration of the workload in seconds, as detected by the
daemon (see Detecting Workloads).

**J_PER_TFLOP:** The calibrated marginal energy cost of one TFLOP of compute.
Whatever energy remains after the memory and overhead terms are subtracted is
divided by this to produce the FLOP estimate.

---

## 2. "Idle Power: [Claude, please briefly and simply explain…]"

**Idle Power:** Measured directly: before any calibration workload runs, the
sensor samples a quiet system for 90 seconds at 2 Hz and takes the median.
This came out to **642 mW**, and is frozen as a constant thereafter — the
deployed daemon subtracts this same value rather than re-measuring, so that
estimates are repeatable and don't depend on how quiet the system happened to
be at deployment time.

---

## 3. "Power Overhead: [Claude, please briefly and simply explain…]"

**Power Overhead (E_overhead):** This one cannot be measured directly — there
is no way to run a workload that is "active but computing nothing" to observe
it. Instead it is fitted: the calibration searches candidate values from 0 to
4 W, and scores each candidate by how well the resulting model predicts a
calibration workload that was left out of the fit (leave-one-out
cross-validation). The best-predicting value was **1.46 W**. Of all the
parameters this is the least pinned-down: it trades off against the other two
fitted terms, and only their combined prediction is stable.

---

## 4. "Energy per TB: [Claude, please briefly and simply explain…]"

**Energy per TB (E_per_TB_J):** Fitted jointly with J_PER_TFLOP inside the
same calibration (at each candidate overhead value, the two are solved by
least squares against the known FLOP counts). The fitted value is **4,622 J
per actmon-unit terabyte**, which converts to **≈61.5 J per real terabyte**
after the scale correction below — comfortably inside the plausible range for
LPDDR5 memory (~50–150 J/TB), a useful sanity check that the blindly-fitted
coefficient corresponds to real memory physics.

---

## 5. "TB moved per actmon activity: [Claude, please suggest a better name…]"

Suggested name: **Actmon scale factor (k)** — or, spelled out, "measured bytes
per actmon-reported byte."

**Actmon scale factor (k):** Actmon reports activity in units of undocumented
scale, so I measured the conversion directly: stream a known volume of data
through DRAM (large tensor copies whose byte count is known analytically) and
compare it against what the counter reports. The counter reads a stable,
linear **1.33%** of true traffic, i.e. **k = 0.0133**. Note that the estimator
itself never uses k — E_per_TB_J is fitted in the counter's own units, so the
scale cancels — k exists to translate the fitted coefficient and the figures
into physical units.

---

## 6. "[Claude, please let me know if I've missed anything]"

Two things:

1. **The list is missing the main calibrated constant: J_PER_TFLOP.** It's the
   heart of the estimator and deserves its own bullet alongside the others.
   Suggested text:

   **Energy per TFLOP (J_PER_TFLOP):** Fitted jointly with E_per_TB_J against
   the calibration workloads' known FLOP counts: **5.61 J per TFLOP**. This is
   the constant that converts leftover energy into the FLOP estimate, and the
   one the whole calibration exists to determine.

2. **Worth one sentence at the top or bottom of the list:** these values are a
   *matched set* from a single calibration sweep (26 workload configurations
   against one shared idle baseline; 21 cleared the frontier-utilization gate
   and entered the fit) — they are only valid together, on this device, in
   this power mode, and must all be re-derived at once when recalibrating.
   This backs the sentence you already have about redoing calibration on
   different hardware. (One small consistency note within this scope: since
   equation (1) includes the TB term, the values above are the 3-parameter
   fit's matched set — the repo also carries a 2-parameter variant with its
   own overhead/marginal pair, 3.75 W / 5.72 J-per-TFLOP, which only matters
   if you present that variant.)
