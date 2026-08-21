#!/usr/bin/env python3
"""
nn_estimator.py — a neural-net FLOP estimator, benchmarked against the linear
2-/3-/4-param energy models on the SAME workload-held-out cross-validation.

Motivation
----------
The production estimators invert a linear energy balance
    E_net = a·FLOPs + c1·DRAM_TB + c2·NVLink_TB + p·t
so they assume a CONSTANT marginal J/TFLOP `a`. Real J/FLOP varies with
arithmetic intensity / kernel efficiency (this is exactly the lever S4
batch-inflation exploits). A neural net over the three sensor signals can learn
that intensity-dependence and — if the effect is real and learnable — predict
FLOPs more accurately than a single global `a`.

Two neural estimators (both small MLPs, torch):
  * PURE-MLP     : log(TFLOPs) ≈ MLP(log/ratio features of E,t,DRAM,NVLink).
                   The literal "use a NN instead of linear regression" ask.
  * RESIDUAL-MLP : TFLOPs = est_2param · exp(g(z))  — a physics-anchored net.
                   The 2-param power-only estimate is the backbone; the MLP g
                   learns only the multiplicative efficiency correction from
                   dimensionless intensity features z. Data-efficient and can't
                   blow up (g=0 ⇒ the physics estimate), so it is the recommended
                   estimator.

Evaluation (apples-to-apples with analyze_trials.py's linear CV)
----------
Repeated k-fold with WORKLOADS as the fold unit (a held-out workload's 10 traces
are never seen in training), IDENTICAL folds for every estimator. Per held-out
workload we take the median predicted TFLOPs over its 10 traces (as
fig_trials_cv_error.png does) and score abs% error vs ground truth. We report the
median held-out error pooled over all (workload, repeat), and draw held-out
estimated-vs-truth scatters + a comparison bar.

    python3 nn_estimator.py [--repeats 5 --folds 5 --epochs 400 --seed S]

CPU-only by default (tiny net; leaves the GPUs free for data collection).
"""
import argparse
import json
import os
import sys
from statistics import median

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "power_calibration"))

import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import detect_flops
from analyze_trials import build_pool
from run_trials import trial_records_path
from eval_power_monitor import (fit_active_energy_model, fit_active_energy_emc_model,
                                fit_active_energy_nvl_model)

OUT = os.path.join(REPO, "writeup")
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE, BLUE, AQUA, AMBER, RED, PURPLE = ("#e1e0d9", "#c3c2b7", "#2a78d6",
                                              "#1baf7a", "#e69f00", "#d1495b", "#7b5cd6")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 13,
})

EPS = 1e-9


# ── Features (observable-only; no ground truth) ──────────────────────────────
def _base(r):
    E = float(r["net_energy_j"]); t = float(r["duration_s"])
    TB = max(float(r.get("tb_moved") or 0.0), EPS)
    NV = max(float(r.get("nvlink_total_bytes") or 0.0) / 1e12, EPS)
    return E, t, TB, NV


def feat_pure(r):
    """Absolute log-features for the pure MLP (predicts log TFLOPs)."""
    E, t, TB, NV = _base(r)
    P = E / t
    return np.array([np.log(E), np.log(t), np.log(TB), np.log(NV), np.log(P),
                     np.log(E / TB), np.log(NV / TB)], dtype=np.float64)


def feat_resid(r):
    """Dimensionless intensity features for the residual correction g(z).
    Scale-invariant ratios that track J/FLOP efficiency (not job size)."""
    E, t, TB, NV = _base(r)
    P = E / t
    return np.array([np.log(P), np.log(E / TB), np.log(NV / TB),
                     np.log(TB / t), np.log(NV / t), np.log(t)], dtype=np.float64)


# ── Linear baselines (production estimators), fit on train records ───────────
def linear_estimates(train, test):
    """Return {name: [est per test record]} for the 2/3/4-param linear models,
    each fit on `train` and applied to `test` via the production estimate fns."""
    out = {}
    a2, p2 = fit_active_energy_model(train)
    out["2-param"] = [detect_flops.estimate_tflops(r["net_energy_j"], r["duration_s"],
                      p_overhead_w=p2, e_marginal_j_per_tflop=a2) for r in test]
    a3, c3, p3 = fit_active_energy_emc_model(train)
    out["3-param"] = [detect_flops.estimate_tflops_emc(r["net_energy_j"], r["duration_s"],
                      r["tb_moved"], p_overhead_w=p3, e_marginal_j_per_tflop=a3,
                      e_per_tb_j=c3) for r in test]
    a4, c4a, c4b, p4 = fit_active_energy_nvl_model(train)
    out["4-param"] = [detect_flops.estimate_tflops_nvl(r["net_energy_j"], r["duration_s"],
                      r["tb_moved"], (r.get("nvlink_total_bytes") or 0.0) / 1e12,
                      p_overhead_w=p4, e_marginal_j_per_tflop=a4, e_per_tb_j=c4a,
                      e_per_nvlink_tb_j=c4b) for r in test]
    return out, (a2, p2)


# ── Neural nets ──────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, d_in, hidden=(32, 32), p_drop=0.1):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _standardize(train_X):
    mu = train_X.mean(0); sd = train_X.std(0) + 1e-8
    return mu, sd


def _train_mlp(X, y, epochs, wd, seed, lr=3e-3):
    torch.manual_seed(seed)
    net = MLP(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(net(Xt), yt)
        loss.backward()
        opt.step()
    net.eval()
    return net


def pure_mlp(train, test, epochs, seed):
    """Predict log(TFLOPs) directly from absolute log-features."""
    Xtr = np.stack([feat_pure(r) for r in train])
    Xte = np.stack([feat_pure(r) for r in test])
    ytr = np.log(np.array([r["ground_truth_tf"] for r in train]))
    mu, sd = _standardize(Xtr)
    ymu, ysd = ytr.mean(), ytr.std() + 1e-8
    net = _train_mlp((Xtr - mu) / sd, (ytr - ymu) / ysd, epochs, wd=1e-3, seed=seed)
    with torch.no_grad():
        z = net(torch.tensor((Xte - mu) / sd, dtype=torch.float32)).numpy()
    return np.exp(z * ysd + ymu)


def residual_mlp(train, test, base_params, epochs, seed):
    """TFLOPs = est_2param · exp(g(z)). g regresses the log-residual of the
    2-param estimate from dimensionless intensity features."""
    a2, p2 = base_params
    def base_est(r):
        e = detect_flops.estimate_tflops(r["net_energy_j"], r["duration_s"],
                                         p_overhead_w=p2, e_marginal_j_per_tflop=a2)
        return max(e, EPS)
    Xtr = np.stack([feat_resid(r) for r in train])
    Xte = np.stack([feat_resid(r) for r in test])
    btr = np.array([base_est(r) for r in train])
    resid = np.log(np.array([r["ground_truth_tf"] for r in train])) - np.log(btr)
    mu, sd = _standardize(Xtr)
    rmu, rsd = resid.mean(), resid.std() + 1e-8
    net = _train_mlp((Xtr - mu) / sd, (resid - rmu) / rsd, epochs, wd=3e-3, seed=seed)
    with torch.no_grad():
        g = net(torch.tensor((Xte - mu) / sd, dtype=torch.float32)).numpy()
    g = g * rsd + rmu
    return np.array([base_est(r) for r in test]) * np.exp(g)


# ── Cross-validation harness ─────────────────────────────────────────────────
ESTIMATORS = ["2-param", "3-param", "4-param", "pure-MLP", "residual-MLP"]


def run_cv(pool, repeats, folds, epochs, seed):
    labels = sorted(pool)
    rng = np.random.default_rng(seed)
    # per-estimator: pooled abs% errors, and per-workload list of held-out estimates
    err = {e: [] for e in ESTIMATORS}
    est_by_lab = {e: {l: [] for l in labels} for e in ESTIMATORS}
    gt = {l: pool[l][0]["ground_truth_tf"] for l in labels}

    for rep in range(repeats):
        order = labels[:]
        rng.shuffle(order)
        fold_of = {lab: i % folds for i, lab in enumerate(order)}
        for f in range(folds):
            te_labs = [l for l in labels if fold_of[l] == f]
            tr_labs = [l for l in labels if fold_of[l] != f]
            train = [r for l in tr_labs for r in pool[l]]     # all traces of train workloads
            # held-out: score each workload by the MEDIAN prediction over its 10 traces
            test = [r for l in te_labs for r in pool[l]]
            test_lab = [l for l in te_labs for _ in pool[l]]

            lin, base_params = linear_estimates(train, test)
            preds = dict(lin)
            preds["pure-MLP"] = pure_mlp(train, test, epochs, seed=1000 * rep + f)
            preds["residual-MLP"] = residual_mlp(train, test, base_params, epochs,
                                                 seed=2000 * rep + f)

            for e in ESTIMATORS:
                per_lab = {}
                for lab, p in zip(test_lab, preds[e]):
                    if p is not None and np.isfinite(p):
                        per_lab.setdefault(lab, []).append(float(p))
                for lab, ps in per_lab.items():
                    m = float(np.median(ps))
                    est_by_lab[e][lab].append(m)
                    err[e].append(abs(m - gt[lab]) / gt[lab] * 100.0)
        print(f"  repeat {rep + 1}/{repeats} done", flush=True)
    return err, est_by_lab, gt


def summarize(err):
    rows = {}
    for e in ESTIMATORS:
        a = np.array(err[e], dtype=float)
        rows[e] = {"median": float(np.median(a)), "mean": float(a.mean()),
                   "p90": float(np.percentile(a, 90)), "n": len(a)}
    return rows


# ── Figures ──────────────────────────────────────────────────────────────────
def _scatter(ax, gt, est_by_lab, name, color, sub):
    labs = [l for l in est_by_lab if est_by_lab[l]]
    x = np.array([gt[l] for l in labs])
    y = np.array([np.median(est_by_lab[l]) for l in labs])
    lim = (0, max(x.max(), y.max()) * 1.08)
    xs = np.array(lim)
    ax.fill_between(xs, xs * 0.8, xs * 1.2, color=GRID, alpha=0.45, lw=0, zorder=1, label="±20%")
    ax.fill_between(xs, xs * 0.9, xs * 1.1, color=BASE, alpha=0.5, lw=0, zorder=1, label="±10%")
    ax.plot(xs, xs, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.scatter(x, y, s=34, color=color, zorder=4, linewidths=0,
               label="one workload (median)")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("Ground-truth TFLOPs")
    ax.set_ylabel("Estimated TFLOPs")
    ax.set_title(f"{name} \u00b7 median error {sub:.1f}%", fontsize=14.5, color=INK,
                 loc="left", pad=10)
    ax.legend(fontsize=12, frameon=False, loc="upper left")


def make_figs(rows, est_by_lab, gt, n_workloads, repeats, folds):
    # scatter: pure-MLP and residual-MLP
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.7), dpi=200)
    _scatter(axes[0], gt, est_by_lab["pure-MLP"], "Pure MLP", PURPLE, rows["pure-MLP"]["median"])
    _scatter(axes[1], gt, est_by_lab["residual-MLP"], "Residual MLP",
             AQUA, rows["residual-MLP"]["median"])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_nn_est_vs_truth.png"))
    fig.savefig(os.path.join(OUT, "figure_5.png"))
    plt.close(fig)

    # comparison bar: median held-out error across all estimators
    fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=200)
    cols = {"2-param": BLUE, "3-param": AQUA, "4-param": AMBER,
            "pure-MLP": PURPLE, "residual-MLP": RED}
    xs = np.arange(len(ESTIMATORS))
    meds = [rows[e]["median"] for e in ESTIMATORS]
    ax.bar(xs, meds, color=[cols[e] for e in ESTIMATORS], zorder=3, width=0.66)
    for i, m in enumerate(meds):
        ax.annotate(f"{m:.1f}%", (i, m), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=13, color=INK, fontweight="bold")
    best = ESTIMATORS[int(np.argmin(meds))]
    ax.axhline(rows["2-param"]["median"], color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=2)
    ax.set_xticks(xs); ax.set_xticklabels(ESTIMATORS, fontsize=12.5)
    ax.set_ylabel("Median held-out error  (%)")
    ax.set_title("Estimator accuracy, identical held-out folds",
                 fontsize=14.5, color=INK, loc="left", pad=10)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_nn_compare.png"))
    fig.savefig(os.path.join(OUT, "figure_6.png"))
    plt.close(fig)
    print("wrote writeup/fig_nn_est_vs_truth.png, writeup/fig_nn_compare.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260816)
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    trial_files = [trial_records_path(k) for k in range(1, args.trials + 1)]
    pool, _dropped, n_trials = build_pool(trial_files)
    print(f"Frontier pool: {len(pool)} workloads × {n_trials} traces "
          f"({sum(len(v) for v in pool.values())} records)")
    print(f"CV: {args.repeats} repeats × {args.folds}-fold (workload-held-out), "
          f"MLP epochs {args.epochs}\n")

    err, est_by_lab, gt = run_cv(pool, args.repeats, args.folds, args.epochs, args.seed)
    rows = summarize(err)

    print("\n" + "=" * 66)
    print("MEDIAN HELD-OUT ERROR  (abs %, identical folds, per-workload median)")
    print("=" * 66)
    print(f"  {'estimator':<16}{'median':>9}{'mean':>8}{'p90':>8}{'n':>7}")
    for e in ESTIMATORS:
        r = rows[e]
        print(f"  {e:<16}{r['median']:>8.2f}%{r['mean']:>7.2f}%{r['p90']:>7.2f}%{r['n']:>7}")
    base = rows["2-param"]["median"]
    for e in ("pure-MLP", "residual-MLP"):
        d = rows[e]["median"] - base
        print(f"  {e} vs 2-param: {d:+.2f} pp "
              f"({'better' if d < 0 else 'worse'})")

    with open(os.path.join(REPO, "nn_estimator_results.json"), "w") as f:
        json.dump({"config": vars(args), "n_workloads": len(pool),
                   "summary": rows,
                   "per_label_median_est": {e: {l: float(np.median(v)) for l, v in est_by_lab[e].items() if v}
                                            for e in ESTIMATORS},
                   "gt": gt}, f, indent=1)
    print("\nwrote nn_estimator_results.json")
    make_figs(rows, est_by_lab, gt, len(pool), args.repeats, args.folds)


if __name__ == "__main__":
    main()
