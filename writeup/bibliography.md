# Bibliography for the report (with insertion map)

*Numbered to match the draft's existing `[1][2][3]` style. URLs for [1]–[13]
resolved and checked 2026-07-16; [14]–[21] added and checked 2026-07-21. The
"insert at" notes name the sentence in the draft each reference should attach
to; renumber freely once the set is final.*

---

## References

**[1]** Y. Shavit. *What does it take to catch a Chinchilla? Verifying Rules
on Large-Scale Neural Network Training via Compute Monitoring.* arXiv:2303.11341,
2023. https://arxiv.org/abs/2303.11341

**[2]** A. R. Wasil, T. Reed, J. W. Miller, and P. Barnett. *Verification
methods for international AI agreements.* arXiv:2408.16074, 2024.
https://arxiv.org/abs/2408.16074

**[3]** A. Scher and L. Thiergart. *Mechanisms to Verify International
Agreements About AI Development.* MIRI Technical Governance Team report;
arXiv:2506.15867, 2025. https://arxiv.org/abs/2506.15867

**[4]** California Senate Bill 53, *Transparency in Frontier Artificial
Intelligence Act* (2025). Defines "frontier model" as a foundation model
trained with more than 10^26 integer or floating-point operations.
https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202520260SB53

**[5]** Regulation (EU) 2024/1689 (EU AI Act), Article 51(2): a
general-purpose AI model is presumed to have systemic risk when its
cumulative training compute exceeds 10^25 FLOPs.
https://artificialintelligenceact.eu/article/51/

**[6]** Epoch AI. *Estimating training compute of deep learning models.*
2022. (Describes both estimation approaches: (1) counting operations from
architecture + data, and (2) GPU-time × peak FLOP/s × utilization.)
https://epoch.ai/blog/estimating-training-compute

**[7]** A. Chaudhuri, S. Shukla, S. Bhattacharya, and D. Mukhopadhyay.
*"Energon": Unveiling Transformers from GPU Power and Thermal Side-Channels.*
ICCAD 2025; arXiv:2508.01768. https://arxiv.org/abs/2508.01768
⚠ The draft spells this "Chaurhuri" — correct to **Chaudhuri**.

**[8]** R. Rahman and S. Tajdari. *Detecting Hidden ML Training With
Zero-Overhead Telemetry.* arXiv:2606.19262, 2026.
https://arxiv.org/abs/2606.19262

**[9]** PyTorch, `torch.utils.flop_counter.FlopCounterMode` (the ground-truth
FLOP counter; a `TorchDispatchMode` that intercepts every executed ATen
operator — forward and backward — and applies per-operator formulas, e.g.
2·M·N·K for a matmul, to the runtime tensor shapes; unregistered ops are
decomposed to registered primitives). Source (PyTorch 2.9):
https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py

**[10]** NVIDIA. *Jetson Linux Developer Guide (r36.4): Platform Power and
Performance — Jetson Orin Nano Series, Jetson Orin NX Series and Jetson AGX
Orin Series.* Documents the single 3-channel INA3221 at I2C 0x40 with rails
VDD_IN (total module), VDD_CPU_GPU_CV (CPU+GPU+CV combined), and VDD_SOC —
and, by contrast, the separate VDD_GPU_SOC rail that exists only on AGX Orin.
https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html

**[11]** Texas Instruments. *INA3221 Triple-Channel, High-Side Measurement,
Shunt and Bus Voltage Monitor* (datasheet).
https://www.ti.com/lit/ds/symlink/ina3221.pdf

**[12]** NVIDIA. *NVIDIA Jetson Orin Nano Developer Kit Gets a "Super"
Boost.* Developer blog, Dec 2024. (Source for the 102 GB/s LPDDR5 memory
bandwidth — raised from 68 GB/s by the JetPack 6.2 "Super" clocks used in
this project.)
https://developer.nvidia.com/blog/nvidia-jetson-orin-nano-developer-kit-gets-a-super-boost/

**[13]** M. O'Connor, N. Chatterjee, D. Lee, J. Wilson, A. Agrawal,
S. W. Keckler, and W. J. Dally. *Fine-Grained DRAM: Energy-Efficient DRAM for
Extreme Bandwidth Systems.* MICRO 2017. (Reference point for DRAM access
energies of a few-to-tens of pJ/bit; the calibrated 61.5 J/TB ≈ 7.7 pJ/bit
sits inside the LPDDR-class range.)
https://research.nvidia.com/publication/2017-10_fine-grained-dram-energy-efficient-dram-extreme-bandwidth-systems

**[14]** NVIDIA. *Jetson Orin Nano Super Developer Kit* (product page /
datasheet). Source for the module specs cited in the hardware section:
1024-core Ampere GPU with 32 tensor cores, 6-core Cortex-A78AE CPU, 8 GB
128-bit LPDDR5.
https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/

**[15]** NVIDIA. *Jetson Linux Developer Guide (r36.5): Clocks* — documents
the activity monitor (actmon) / EMC dynamic frequency scaling machinery this
project reads for memory-controller activity. (The Linux kernel devicetree
binding, nvidia,tegra30-actmon.txt, has the concise description: the activity
monitor block "collects statistics about the behaviour of other components in
the system" to derive the required external-memory clock rate.)
https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/Clocks.html
https://www.kernel.org/doc/Documentation/devicetree/bindings/arm/tegra/nvidia,tegra30-actmon.txt

**[16]** A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones,
A. N. Gomez, Ł. Kaiser, and I. Polosukhin. *Attention Is All You Need.*
NeurIPS 2017; arXiv:1706.03762. (Anchors the transformer terminology:
d_model, attention heads, feed-forward width.)
https://arxiv.org/abs/1706.03762

**[17]** J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess,
R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei. *Scaling Laws for
Neural Language Models.* arXiv:2001.08361, 2020. (Standard cite for the
matmul-only FLOP-counting convention / the 6·N·D approximation.)
https://arxiv.org/abs/2001.08361

**[18]** G. Kulp, D. Gonzales, E. Smith, L. Heim, P. Puri, M. J. D. Vermeer,
and Z. Winkelman. *Hardware-Enabled Governance Mechanisms: Developing
Technical Solutions to Exempt Items Otherwise Classified Under Export Control
Classification Numbers 3A090 and 4A090.* RAND working paper WR-A3056-1, 2024.
https://www.rand.org/pubs/working_papers/WRA3056-1.html

**[19]** J. Petrie, O. Aarne, N. Ammann, and D. Dalrymple. *Interim Report:
Mechanisms for Flexible Hardware-Enabled Guarantees.* 2024. (The 2025
follow-on series: *Technical Options for Flexible Hardware-Enabled
Guarantees*, arXiv:2506.03409, and *International Security Applications of
Flexible Hardware-Enabled Guarantees*, arXiv:2506.15100.)
https://yoshuabengio.org/wp-content/uploads/2024/09/FlexHEG-Interim-Report_2024.pdf

**[20]** N. Cankaya, J. Kryś, J. Ng, L. Marks, and F. Krückel.
*Fingerprinting All AI Cluster I/O Without Mutually Trusted Processors.*
arXiv:2606.10724, 2026. (Network taps on all links between an AI cluster and
the outside world, with cryptographic commitments of all ingress/egress data,
for international AI governance verification.)
https://arxiv.org/abs/2606.10724

**[21]** O. Aarne, T. Fist, and C. Withers. *Secure, Governable Chips: Using
On-Chip Mechanisms to Manage National Security Risks from AI & Advanced
Computing.* CNAS report, January 2024. (Tamper-evident and tamper-responsive
AI hardware as a governance mechanism.)
https://www.cnas.org/publications/reports/secure-governable-chips

---

## Insertion map (v3 draft sentence → reference)

| Draft location | Text | Cite |
|---|---|---|
| Motivation, sentence 2 | "California's SB 53 uses 10^26 …" | [4] |
| Motivation, sentence 2 | "…the EU AI Act applies the same categorization at 10^25" | [5] |
| Motivation, sentence 3 | "Proposals for international AI agreements[1][2][3]" | [1][2][3] (as numbered) |
| Related Work, ¶1 | "EpochAI have done work on estimating…" | [6] |
| Related Work, ¶2 | "Chaudhuri et al. demonstrate…" (fix spelling) | [7] |
| Related Work, ¶3 | "Rahman and Tajdari showed…" | [8] |
| Hardware and Observable Signals, ¶1 | "…Nvidia's version of the Raspberry Pi… with an Ampere GPU" | [14] |
| Hardware and Observable Signals, Power bullet | "…onboard INA3221 sensor…" | [10][11] |
| Hardware and Observable Signals, Memory Bandwidth bullet | "Actmon is a system-level process…" | [15] |
| Constructing Sample Workloads, ¶1 | "It initializes a transformer…" (or the hyperparameter list intro) | [16] |
| Constructing Sample Workloads, frontier definition | "…expect an AI developer to want to fully utilize their GPUs…" | [6] |
| Ground Truth Computation | "…PyTorch's FlopCounterMode library…" (rewritten ¶) | [9] |
| Ground Truth Computation | "…the same convention behind the common 6·N·D approximation" | [17] |
| Calibration, EMC Scale Factor bullet | "…the Nano's maximum memory bandwidth is 102 GB/s…" | [12] |
| Calibration, Energy per TB (if the 61.5 J/TB sanity check is added) | "…inside the plausible range for LPDDR5" | [13] |
| New sensor paragraph (see revision_pack.md §5) | rail layout / AGX Orin contrast | [10] |
| Next Steps, "other forms of verification" examples | "verifying training vs inference" | [8] (their paper *is* a training-vs-inference classifier from GPU telemetry) |
| Next Steps, "other forms of verification" examples | "monitoring cluster ingress/egress" | [20] |
| Next Steps, "other forms of verification" examples | "tamper-detection" | [21] |
| Next Steps, limitations ¶ (optional) | "…assume adversarial behavior on behalf of the AI model developers" | [1] or [3] |

*Alternates for [1]–[3] if you'd rather cite different proposals:
Baker, "Nuclear Arms Control Verification and Lessons for AI Treaties"
(arXiv:2304.04123). The three chosen above are the most-cited
compute-verification proposals and each explicitly uses a training-FLOP
threshold. (FlexHEG, previously listed here as an alternate, is now [19].)*
