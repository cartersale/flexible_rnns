"""Does the WIRING-ONLY group-constant subspace predict dynamics / memory?

Per network, from the consensus signed-WL partition + W_rec ALONE (no activity),
we extract properties of the group-constant (piecewise-constant / lumpable)
subspace:

  DIMENSION        k (subspace dim) and effective dim (#|lambda(Q)|>0.5).
  QUOTIENT SPECTRUM eigenvalues of the k x k quotient matrix Q = D^-1 H^T W H:
                   spectral radius, #near-marginal (|lambda|>0.9 / >0.99;
                   line-attractor candidates), #complex modes near 1 (rotational
                   / ring candidates), spectral gap.
  SUBSPACE CAPTURE how well the subspace captures W's leading structure:
                   linear step-R^2, subspace invariance, eigenvector leakage.

We then ask whether these predict (a) per-network memory-task performance (from
log.json) and, downstream, (b) the fixed-point inventory (line-attractor count /
ring presence; separate script once FPs exist for all nets).

Wiring-only: everything here is a function of (W_rec, partition). The partition is
the consensus over 20 WL restarts, so predictors are on the identifiability-honest
partition; a --per-restart mode reports predictor stability across the ensemble.
"""

from __future__ import annotations

from flexsym import paths

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np


from flexsym.structural.signed_input_wl_structural_roles import (  # noqa: E402
    apply_matrix_variant,
    labels_to_H,
    load_w_rec,
    quotient_matrix,
)
from flexsym.run.lumpability_check import eig_leakage, pw_constant_projector, subspace_invariance  # noqa: E402
from flexsym.run.motif_jacobian_lumpability import best_r2  # noqa: E402

REPO = paths.REPO_ROOT
ENS = paths.RESULTS_ROOT / "signed_input_wl_paper" / "restart_ensemble"
ORIG = paths.ORIGINAL_STORE
STAGED = paths.STAGED_STORE
OUT = paths.RESULTS_ROOT / "signed_input_wl_paper" / "subspace_dynamics"

MEMORY_TASKS = ["delaygo", "delayanti", "delaydm1", "delaydm2", "contextdelaydm1",
                "contextdelaydm2", "multidelaydm", "dmsgo", "dmsnogo", "dmcgo", "dmcnogo"]
NONMEM_TASKS = ["fdgo", "reactgo", "fdanti", "reactanti"]
ALPHA = 0.2  # leak; J = (1-alpha) + alpha*eig(W) maps W-spectrum -> discrete-time map


def resolve(model: str) -> Path | None:
    for store in (ORIG, STAGED):
        if (store / model).exists():
            return store / model
    return None


def quotient_spectrum_features(W_sub: np.ndarray, labels: np.ndarray, k: int) -> dict:
    H = labels_to_H(labels, k)
    Q = quotient_matrix(W_sub, labels_to_H(labels, k))
    w = np.linalg.eigvals(Q)
    absw = np.abs(w)
    order = np.argsort(-absw)
    absw_sorted = absw[order]
    # map to leaky discrete-time eigenvalues so "marginal" means memory/line-attractor
    Jw = (1 - ALPHA) + ALPHA * w
    absJ = np.abs(Jw)
    is_complex = np.abs(w.imag) > 1e-6
    return {
        "dim_k": int(k),
        "dim_eff_half": int(np.sum(absw > 0.5)),
        "q_spectral_radius": float(absw_sorted[0]),
        "q_gap_12": float(absw_sorted[0] - absw_sorted[1]) if k > 1 else float("nan"),
        # near-marginal quotient modes on the leaky map (line-attractor candidates)
        "qJ_n_marg_090": int(np.sum(absJ > 0.90)),
        "qJ_n_marg_099": int(np.sum(absJ > 0.99)),
        "qJ_max_abs": float(absJ.max()),
        # rotational / ring candidates: complex quotient modes with large modulus
        "q_n_complex": int(is_complex.sum()),
        "q_n_complex_marg": int(np.sum(is_complex & (absJ > 0.90))),
        "q_max_complex_abs": float(absJ[is_complex].max()) if is_complex.any() else 0.0,
    }


def subspace_capture_features(W_sub: np.ndarray, labels: np.ndarray, k: int) -> dict:
    P = pw_constant_projector(labels, k)
    return {
        "cap_step_r2_linear": float(best_r2(W_sub, labels, k)),
        "cap_invariance": float(subspace_invariance(W_sub, P)),
        "cap_leakage": float(eig_leakage(W_sub, P, k)),
    }


def memory_targets(model_dir: Path) -> dict:
    log = json.loads((model_dir / "log.json").read_text())

    def last(task: str) -> float:
        v = log.get(f"perf_{task}")
        return float(v[-1]) if v else float("nan")

    mem = np.array([last(t) for t in MEMORY_TASKS], dtype=float)
    non = np.array([last(t) for t in NONMEM_TASKS], dtype=float)
    out = {
        "perf_memory_mean": float(np.nanmean(mem)),
        "perf_memory_min": float(np.nanmin(mem)),
        "perf_nonmem_mean": float(np.nanmean(non)),
        "perf_mem_minus_nonmem": float(np.nanmean(mem) - np.nanmean(non)),
        "perf_avg": float(log.get("perf_avg", [np.nan])[-1]),
        "perf_min": float(log.get("perf_min", [np.nan])[-1]),
    }
    for t in MEMORY_TASKS:
        out[f"perf_{t}"] = last(t)
    return out


def run_net(seed: int) -> dict | None:
    model = f"LeakyRNN_softplus_diag_256_all_{seed}"
    md = resolve(model)
    ens = ENS / f"{model}_ensemble.npz"
    if md is None or not ens.exists():
        print(f"SKIP {model}")
        return None
    z = np.load(ens, allow_pickle=True)
    k = int(z["k"])
    ind = np.asarray(z["ind_active"], dtype=np.int64)
    labels = np.asarray(z["consensus_labels"], dtype=np.int64)
    W_sub = apply_matrix_variant(load_w_rec(md), "transpose")[np.ix_(ind, ind)]

    row = {"model": model, "seed": seed, "n_active": int(len(ind))}
    row.update(quotient_spectrum_features(W_sub, labels, k))
    row.update(subspace_capture_features(W_sub, labels, k))
    row.update(memory_targets(md))
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = [r for s in args.seeds if (r := run_net(s)) is not None]
    fieldnames = sorted({k for r in rows for k in r})
    with (OUT / "subspace_predictors_256.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # correlations: each wiring predictor vs memory performance across nets
    from scipy.stats import spearmanr, pearsonr

    predictors = ["dim_k", "dim_eff_half", "q_spectral_radius", "q_gap_12",
                  "qJ_n_marg_090", "qJ_n_marg_099", "qJ_max_abs",
                  "q_n_complex", "q_n_complex_marg", "q_max_complex_abs",
                  "cap_step_r2_linear", "cap_invariance", "cap_leakage"]
    targets = ["perf_memory_mean", "perf_memory_min", "perf_mem_minus_nonmem", "perf_min"]
    print(f"\n=== {len(rows)} nets: wiring predictor -> memory performance ===")
    print("target variances:", {t: round(float(np.nanvar([r[t] for r in rows])), 6) for t in targets})
    for tgt in targets:
        y = np.array([r[tgt] for r in rows], dtype=float)
        print(f"\n-- {tgt} (range {np.nanmin(y):.3f}..{np.nanmax(y):.3f}) --")
        res = []
        for pr in predictors:
            x = np.array([r[pr] for r in rows], dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 4 or np.nanstd(x[ok]) == 0:
                continue
            rho, pval = spearmanr(x[ok], y[ok])
            res.append((abs(rho), pr, rho, pval))
        for _, pr, rho, pval in sorted(res, reverse=True)[:6]:
            print(f"   {pr:22} spearman rho={rho:+.3f}  p={pval:.3f}")
    print(f"\nwrote {OUT/'subspace_predictors_256.csv'}")


if __name__ == "__main__":
    main()
