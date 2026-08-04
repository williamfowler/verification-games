"""
redteam_configs.py — Group A (offline) adversarial workload sweep.

RED_CONFIGS mirrors eval_power_monitor.CONFIGS' shape so it feeds straight through
the existing sweep (`eval_power_monitor.py --configs-module
red_team.redteam_configs:RED_CONFIGS`). Each entry carries a "strategy" tag and a
unique "label"; decoy entries add a "script"/"args" override that points the run
at adversarial_workload.py. These are Group A only — scored on the same cleanly-
bracketed offline window as Phase I. Splitting/throttling (S4/S5) attack the live
daemon's session logic and are NOT here (see live_daemon_probe.py).

Strategies represented:
  S1_precision   — fp32 twins of benign configs. On Volta fp16 is already the
                   baseline AND the max-throughput precision, and Phase I is
                   fp16-calibrated, so the precision *under*-report channel is
                   closed; fp32 runs instead read HIGH. Included to quantify that
                   precision-blindness (the honest V100 result — see README §S1).
  S3_atypical    — out-of-distribution geometry (extreme FFN/seq ratios, nhead,
                   very wide/narrow), where the linear energy fit extrapolates
                   poorly. Signed error per axis is the result (reuse Phase I bias).
  S2_mem_decoy   — concurrent DRAM decoy traffic (device-to-device copies) to
                   inflate the tb_moved the 3-param estimator subtracts; a
                   bandwidth sweep so the efficiency gate marks where it stops
                   being legal.
  S2_nvlink_decoy — [V100] extra NVLink all-reduce to inflate the interconnect
                   signal (targets the tripwire / 4-param term; DRAM decoy off).
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADV = os.path.join(REPO_ROOT, "red_team", "adversarial_workload.py")


def _cfg(strategy, label, d_model, seq_len, batch_size, dim_feedforward,
         num_layers, nhead, precision="fp16", optimizer="adamw",
         family="RED", script=None, args=None):
    c = {"family": family, "strategy": strategy, "label": label,
         "d_model": d_model, "seq_len": seq_len, "batch_size": batch_size,
         "dim_feedforward": dim_feedforward, "num_layers": num_layers,
         "nhead": nhead, "precision": precision, "optimizer": optimizer}
    if script:
        c["script"] = script
    if args:
        c["args"] = args
    return c


def _decoy(label, strategy, decoy_args, d_model=1024, seq_len=512, batch_size=16,
           dim_feedforward=4096, num_layers=12, nhead=8):
    """A benign frontier base shape wrapped by adversarial_workload.py with decoy
    args. Base is d1024_b16_s512_L12 (a Phase I frontier config), so any estimate
    change is attributable to the decoy, not the shape."""
    return _cfg(strategy, label, d_model, seq_len, batch_size, dim_feedforward,
                num_layers, nhead, family="S2", script=ADV,
                args=["--strategy", "mem_decoy"] + decoy_args)


# ── S1 — precision (fp32 twins of benign configs) ─────────────────────────────
S1_CONFIGS = [
    _cfg("S1_precision", "S1_d1024_b16_s512_L12_fp32", 1024, 512, 16, 4096, 12, 8,
         precision="fp32", family="S1"),
    _cfg("S1_precision", "S1_d1024_b16_s256_L6_fp32", 1024, 256, 16, 4096, 6, 8,
         precision="fp32", family="S1"),
    _cfg("S1_precision", "S1_d1536_b16_s512_L6_fp32", 1536, 512, 16, 6144, 6, 12,
         precision="fp32", family="S1"),
    _cfg("S1_precision", "S1_d768_b8_s512_L12_fp32", 768, 512, 8, 3072, 12, 12,
         precision="fp32", family="S1"),
]

# ── S3 — atypical / out-of-distribution hyperparameters ───────────────────────
# Each pushes an axis past the Phase I sweep's range (d≤2048, seq≤4096, nhead≤32,
# ff≤12288·8×d, layers≤32) so the fit must extrapolate.
S3_CONFIGS = [
    _cfg("S3_atypical", "S3_wideFFN_d768_ff16384", 768, 512, 16, 16384, 6, 8,
         family="S3"),                                   # ff/d = 21× (Phase I ≤8×)
    _cfg("S3_atypical", "S3_longseq_d512_s8192", 512, 8192, 2, 2048, 6, 8,
         family="S3"),                                   # seq 8192 (Phase I ≤4096)
    _cfg("S3_atypical", "S3_widemodel_d3072", 3072, 256, 8, 12288, 6, 24,
         family="S3"),                                   # d_model 3072 (≤2048)
    _cfg("S3_atypical", "S3_manyheads_d1024_h64", 1024, 512, 16, 4096, 12, 64,
         family="S3"),                                   # head_dim 16 (nhead≤32)
    _cfg("S3_atypical", "S3_deep_d512_L48", 512, 512, 16, 2048, 48, 8,
         family="S3"),                                   # 48 layers (≤32)
    _cfg("S3_atypical", "S3_skinnyFFN_d2048_ff2048", 2048, 512, 8, 2048, 6, 16,
         family="S3"),                                   # ff/d = 1× (very compute-light FFN)
]

# ── S2 — memory / interconnect decoy (bandwidth + all-reduce sweeps) ──────────
S2_CONFIGS = [
    _decoy("S2_decoy_gbps150",  "S2_mem_decoy", ["--decoy-gbps", "150",  "--decoy-mb", "256"]),
    _decoy("S2_decoy_gbps300",  "S2_mem_decoy", ["--decoy-gbps", "300",  "--decoy-mb", "256"]),
    _decoy("S2_decoy_gbps600",  "S2_mem_decoy", ["--decoy-gbps", "600",  "--decoy-mb", "256"]),
    _decoy("S2_decoy_gbps1200", "S2_mem_decoy", ["--decoy-gbps", "1200", "--decoy-mb", "256"]),
    # [V100] NVLink-only decoy (DRAM decoy off via --decoy-gbps 0): targets the
    # interconnect term / consistency tripwire, not the DRAM subtraction.
    _decoy("S2_nvlink_ar64",  "S2_nvlink_decoy",
           ["--decoy-gbps", "0", "--decoy-allreduce", "--decoy-allreduce-mb", "64"]),
    _decoy("S2_nvlink_ar256", "S2_nvlink_decoy",
           ["--decoy-gbps", "0", "--decoy-allreduce", "--decoy-allreduce-mb", "256"]),
]

RED_CONFIGS = S1_CONFIGS + S3_CONFIGS + S2_CONFIGS

# Strategy → configs, for the scorer's per-strategy grouping.
BY_STRATEGY = {}
for _c in RED_CONFIGS:
    BY_STRATEGY.setdefault(_c["strategy"], []).append(_c["label"])


if __name__ == "__main__":
    print(f"RED_CONFIGS: {len(RED_CONFIGS)} entries")
    for strat, labels in BY_STRATEGY.items():
        print(f"  {strat:16s} {len(labels)}: {', '.join(labels)}")
