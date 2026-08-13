"""
redteam_configs.py — Phase II v3 offline adversarial configs (one variable, GT-preserving).

Every config declares a `parent` (a real benign workload) and changes exactly ONE
variable, holding ground truth constant so any estimator change is attributable to
that variable. Fed through the Phase I sweep unchanged:
    eval_power_monitor.py --configs-module red_team.redteam_configs:RED_CONFIGS

All sets use `fixed_steps` (the sweep must NOT auto-size them — that would break the
GT invariant). GT-preservation is by construction and re-checked at run time by
red_team/test_benign_equivalence.py:

  S4  batch-inflation      variable = batch_size, above the calibration range
                           (8/16/32 → 64/128), with steps = P/batch (P constant),
                           so batch×steps — hence per-step-FLOPs×steps×world — is
                           identical across the set. Attacks the constant-J/FLOP
                           assumption: bigger GEMMs run at higher tensor-core
                           efficiency ⇒ lower J/FLOP than the fitted E_MARGINAL ⇒
                           under-report (vs the 2-param estimator especially).
  S3  atypical geometry    variable = nhead (d_model fixed → head_dim 1024…16) or
                           optimizer (adamw→sgd); neither changes per-step FLOPs,
                           so the whole set shares one fixed step count. Probes
                           kernel-efficiency-driven energy at fixed GT (expected
                           over-report from poor tensor-core utilisation).

Retired to red_team_old/: v1 precision / mem_decoy / nvlink_decoy / GT-altering
atypical.  Split / throttle are schedule attacks (red_team/adversarial_workload.py,
live daemon), not offline configs.
"""

# ── S4 — batch inflation (matched set, GT held constant via batch×steps=P) ─────
# P is divisible by every batch so steps are integers and GT is EXACT (no rounding).
def batch_inflation_set(parent_label, d_model, seq_len, num_layers, nhead,
                        dim_feedforward, batches, P, ref_batch):
    cfgs = []
    for b in batches:
        assert P % b == 0, f"P={P} not divisible by batch {b} (GT would drift)"
        steps = P // b
        label = f"S4_{parent_label}_b{b}"
        cfgs.append({
            "family": "S4", "strategy": "S4_batch",
            "label": label, "parent": f"S4_{parent_label}_b{ref_batch}",
            "d_model": d_model, "seq_len": seq_len, "batch_size": b,
            "num_layers": num_layers, "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "precision": "fp16", "optimizer": "adamw",
            "fixed_steps": True, "steps": steps,
        })
    return cfgs


# ── S3 — atypical nhead / optimizer (fixed steps; nhead|d_model, GT unchanged) ─
def atypical_set(parent_label, d_model, seq_len, batch_size, num_layers,
                 dim_feedforward, ref_nhead, nheads, steps, add_sgd=True):
    cfgs = []
    for h in nheads:
        assert d_model % h == 0, f"nhead {h} must divide d_model {d_model}"
        cfgs.append({
            "family": "S3", "strategy": "S3_atypical",
            "label": f"S3_{parent_label}_h{h}", "parent": f"S3_{parent_label}_h{ref_nhead}",
            "d_model": d_model, "seq_len": seq_len, "batch_size": batch_size,
            "num_layers": num_layers, "nhead": h, "dim_feedforward": dim_feedforward,
            "precision": "fp16", "optimizer": "adamw", "fixed_steps": True, "steps": steps,
        })
    if add_sgd:
        cfgs.append({
            "family": "S3", "strategy": "S3_atypical",
            "label": f"S3_{parent_label}_sgd", "parent": f"S3_{parent_label}_h{ref_nhead}",
            "d_model": d_model, "seq_len": seq_len, "batch_size": batch_size,
            "num_layers": num_layers, "nhead": ref_nhead, "dim_feedforward": dim_feedforward,
            "precision": "fp16", "optimizer": "sgd", "fixed_steps": True, "steps": steps,
        })
    return cfgs


# S4 parent: A:d1024_s256_b8_L6 (a low-batch family-A config with memory headroom
# to grow batch to 128). P=15360 (=128×120) so batch 8→128 gives steps
# 1920/960/480/240/120 — each ~90–120s active (this shape runs ~16 steps/s at
# batch 8), keeping the high-batch attack runs in the frontier/steady-state regime
# the estimator is calibrated for, while GT stays EXACT (batch×steps=15360).
S4_CONFIGS = batch_inflation_set(
    "d1024_s256_L6", d_model=1024, seq_len=256, num_layers=6, nhead=8,
    dim_feedforward=4096, batches=[8, 16, 32, 64, 128], P=15360, ref_batch=8)

# S3 parent: A:d1024_s512_b16_L12_h8. nhead 8 is the reference; 1/2/32/64 are the
# atypical geometries (head_dim 1024/512/32/16); + an sgd twin. steps fixed = 120
# (~40–100s active, all frontier).
S3_CONFIGS = atypical_set(
    "d1024_s512_b16_L12", d_model=1024, seq_len=512, batch_size=16, num_layers=12,
    dim_feedforward=4096, ref_nhead=8, nheads=[1, 2, 8, 32, 64], steps=120)

RED_CONFIGS = S4_CONFIGS + S3_CONFIGS

BY_STRATEGY = {}
for _c in RED_CONFIGS:
    BY_STRATEGY.setdefault(_c["strategy"], []).append(_c["label"])

# parent label -> the set it heads (for the GT-equivalence test).
PARENTS = sorted({c["parent"] for c in RED_CONFIGS})


if __name__ == "__main__":
    print(f"RED_CONFIGS: {len(RED_CONFIGS)} entries; parents: {PARENTS}")
    for strat, labels in BY_STRATEGY.items():
        print(f"  {strat:12s} {len(labels)}: {', '.join(labels)}")
    # GT-invariance sanity: batch×steps constant within the S4 set.
    prods = {c["batch_size"] * c["steps"] for c in S4_CONFIGS}
    print(f"  S4 batch×steps products (must be 1 value): {prods}")
