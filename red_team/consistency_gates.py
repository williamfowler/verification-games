"""
consistency_gates.py — cross-signal consistency layer (Phase III, defense-in-depth).

The point estimator uses the strong signal(s) — power (+DRAM) — and NOTHING else,
because every term in the estimate is a lever the adversary can push on (a nonzero
NVLink coefficient would let an interconnect flood move the estimate; and NVLink is
collinear with DRAM on benign data anyway, so it earns no weight — see the Phase I
ablation). Instead, every *other* signal is used as a GATE, not a term: each benign
run keeps a set of observable signal ratios inside a calibrated band, and an
adversary who moves the point estimate has to keep ALL of those ratios in band at
the same time — which is hard and expensive (it shows up in the 2× GPU-hour budget).

Gates (all from observables — no ground truth, deployable):
  comm_energy      NVLink_bytes / net_energy_J
                   → comm attacks: interconnect flood (HIGH) or compute-burn with
                     no all-reduce (LOW). Loose band (these deviations are egregious).
  arith_intensity  NVLink_bytes / DRAM_bytes
                   → the arithmetic-intensity / batch-inflation gate. NVLink tracks
                     params×steps and DRAM tracks tokens×steps, so their ratio ∝
                     1/(batch·seq); inflating batch past the calibration range drops
                     it below the benign floor. This is the gate that catches S4,
                     which the point estimate misses — NVLink finally earning its
                     keep as a shield rather than a (zeroed) regressor. Tight band.
  dram_energy      DRAM_bytes / net_energy_J
                   → memory-vs-compute balance; a second intensity view.
  est_agreement    (est_power_only − est_power+DRAM) / est_power+DRAM
                   → the two estimators must agree within benign scatter; a DRAM-
                     manipulating attack (the retired mem_decoy) splits them.

Design note (why tight vs loose): subtle attacks (batch-inflation lands just below
the benign 5th pct) need a TIGHT band, which costs ~5%/side false positives; egregious
attacks (comm flood 50×) are caught by a LOOSE band at ~0 FP. Gates FLAG for scrutiny
(combined with the efficiency verdict), they don't hard-reject — Swiss-cheese, not a
single wall.

    python3 red_team/consistency_gates.py    # calibrate on Phase I, report FP + catch
"""
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "power_calibration"))

import detect_flops
from eval_power_monitor import (load_records, valid, is_frontier,
                                fit_active_energy_model, fit_active_energy_nvl_model)
from efficiency import _default_phase1_paths


# ── Observable ratios (each takes a record + the frozen calibration) ─────────
def _dram_bytes(r):
    tb = r.get("tb_moved")
    return tb * 1e12 if tb else None


def _comm_energy(r, cal):
    nv = r.get("nvlink_total_bytes")
    return (nv / r["net_energy_j"]) if nv and r.get("net_energy_j") else None


def _arith_intensity(r, cal):
    nv = r.get("nvlink_total_bytes")
    db = _dram_bytes(r)
    return (nv / db) if nv and db else None


def _dram_energy(r, cal):
    db = _dram_bytes(r)
    return (db / r["net_energy_j"]) if db and r.get("net_energy_j") else None


def _est_agreement(r, cal):
    db = _dram_bytes(r)
    if db is None:
        return None
    est2 = detect_flops.estimate_tflops(
        r["net_energy_j"], r["duration_s"],
        p_overhead_w=cal["p2"], e_marginal_j_per_tflop=cal["e2"])
    nv = r.get("nvlink_total_bytes") or 0.0
    est4 = detect_flops.estimate_tflops_nvl(
        r["net_energy_j"], r["duration_s"], r["tb_moved"], nv / 1e12,
        p_overhead_w=cal["p4"], e_marginal_j_per_tflop=cal["e4"],
        e_per_tb_j=cal["c4tb"], e_per_nvlink_tb_j=cal["c4nv"])
    if not est2 or not est4 or est4 <= 0:
        return None
    return (est2 - est4) / est4


# (name, fn, (lo_pct, hi_pct), one-line description). Tight = intensity/agreement;
# loose = the egregious-deviation comm/memory gates.
GATES = [
    ("comm_energy",     _comm_energy,     (1.0, 99.0),
     "NVLink/energy — interconnect flood / compute-burn (comm attacks)"),
    ("arith_intensity", _arith_intensity, (5.0, 99.0),
     "NVLink/DRAM — batch-inflation / arithmetic intensity (the S4 shield; LOW side"
     " catches high-batch, ceiling kept loose since floods are caught by comm_energy)"),
    ("dram_energy",     _dram_energy,     (1.0, 99.0),
     "DRAM/energy — memory-vs-compute balance"),
    ("est_agreement",   _est_agreement,   (2.5, 97.5),
     "power-only vs power+DRAM estimate agreement"),
]


def build_calibration(phase1_paths=None):
    """Frozen 2-param + 4-param constants (for est_agreement) from Phase I benign."""
    recs = []
    for p in (phase1_paths or _default_phase1_paths()):
        r, *_ = load_records(p)
        recs.extend(valid(r))
    frontier = [r for r in recs
                if r["config"].get("precision", "fp16") == "fp16" and is_frontier(r)]
    e2, p2 = fit_active_energy_model(frontier)
    e4, c4tb, c4nv, p4 = fit_active_energy_nvl_model(frontier)
    return {"e2": e2, "p2": p2, "e4": e4, "c4tb": c4tb, "c4nv": c4nv, "p4": p4,
            "_frontier": frontier}


def calibrate_bands(cal):
    """Benign band [lo, hi] per gate at its configured percentiles."""
    bands = {}
    for name, fn, (lo_p, hi_p), _desc in GATES:
        vals = np.array([v for v in (fn(r, cal) for r in cal["_frontier"]) if v is not None])
        bands[name] = (float(np.percentile(vals, lo_p)), float(np.percentile(vals, hi_p)),
                       int(len(vals)))
    return bands


def check_run(record, cal, bands):
    """Per-gate verdict for one run. Returns {gate: (value, lo, hi, verdict)} and an
    overall 'FLAG' if any gate is out of band ('OK' otherwise). verdict ∈
    {OK, LOW, HIGH, NA}."""
    result, flagged = {}, []
    for name, fn, _pct, _desc in GATES:
        v = fn(record, cal)
        lo, hi, _n = bands[name]
        if v is None:
            verdict = "NA"
        elif v < lo:
            verdict = "LOW"; flagged.append(name)
        elif v > hi:
            verdict = "HIGH"; flagged.append(name)
        else:
            verdict = "OK"
        result[name] = (v, lo, hi, verdict)
    return {"gates": result, "flagged": flagged, "overall": "FLAG" if flagged else "OK"}


def false_positive_rate(cal, bands):
    """Fraction of benign frontier runs flagged by each gate and by ANY gate."""
    per_gate = {name: 0 for name, *_ in GATES}
    any_flag = 0
    F = cal["_frontier"]
    for r in F:
        res = check_run(r, cal, bands)
        for g in res["flagged"]:
            per_gate[g] += 1
        if res["flagged"]:
            any_flag += 1
    n = len(F)
    return {g: c / n * 100 for g, c in per_gate.items()}, any_flag / n * 100, n


def main():
    cal = build_calibration()
    bands = calibrate_bands(cal)
    print("=" * 78)
    print("CONSISTENCY GATES — benign bands (Phase I frontier) + false-positive rate")
    print("=" * 78)
    fp, any_fp, n = false_positive_rate(cal, bands)
    for name, fn, pct, desc in GATES:
        lo, hi, nn = bands[name]
        print(f"  {name:16s} band[{lo:.3e}, {hi:.3e}]  FP {fp[name]:4.1f}%   {desc}")
    print(f"  {'ANY gate':16s} benign flagged by ≥1 gate: {any_fp:.1f}%  (n={n})")

    # Catch rate on the v3 offline strategies.
    path = os.path.join(REPO_ROOT, "red_team", "red_v3_trial1_records.json")
    if os.path.exists(path):
        rv = json.load(open(path))["records"]
        print("\n" + "=" * 78)
        print("CATCH — which gate flags each adversarial run (point-estimate evasion in ())")
        print("=" * 78)
        for strat in ["S4_batch", "S3_atypical"]:
            print(f"  {strat}:")
            for r in sorted([x for x in rv if x["returncode"] == 0
                             and x["config"].get("strategy") == strat],
                            key=lambda r: (r["config"].get("batch_size", 0),
                                           r["config"].get("nhead", 0))):
                res = check_run(r, cal, bands)
                cfg = r["config"]
                var = (f"b{cfg['batch_size']}" if strat == "S4_batch"
                       else (f"h{cfg['nhead']}" if cfg["optimizer"] != "sgd" else "sgd"))
                flags = ", ".join(f"{g}={res['gates'][g][3]}" for g in res["flagged"]) or "—"
                print(f"    {var:>5}: {res['overall']:4s}  [{flags}]")

    # Retired v1 comm/memory attacks — show the OTHER gates catch their intended attack.
    v1 = os.path.join(REPO_ROOT, "red_team_old", "red_results_trial1_records.json")
    if os.path.exists(v1):
        rv1 = json.load(open(v1))["records"]
        v1strats = ("S2_mem_decoy", "S2_nvlink_decoy")
        rows = [r for r in rv1 if r["returncode"] == 0
                and r["config"].get("strategy") in v1strats]
        if rows:
            print("\n" + "=" * 78)
            print("CATCH — retired v1 comm/memory attacks (each caught by its matched gate)")
            print("=" * 78)
            for r in rows:
                res = check_run(r, cal, bands)
                flags = ", ".join(f"{g}={res['gates'][g][3]}" for g in res["flagged"]) or "—"
                print(f"    {r['label']:22s} ({r['config']['strategy']:15s}): "
                      f"{res['overall']:4s}  [{flags}]")


if __name__ == "__main__":
    main()
