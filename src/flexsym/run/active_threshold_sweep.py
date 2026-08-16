"""Is the lumpability an artefact of where we cut the active-unit threshold?

Every Act-1 analysis restricts W_rec to "active" units -- those whose summed task-epoch
variance exceeds tau = 1e-3 (`cluster_epoch_variance.py`). That threshold is a choice, and
a sceptical reading of the result is that it selects a well-behaved sub-network and the
piecewise-constant eigenvectors follow from the selection rather than from training.

This sweeps tau over five orders of magnitude and reports, for each value:

  n_active(tau)   how many units survive
  r2_topk(tau)    the lumpability statistic on the surviving sub-matrix, with its own
                  size-matched random-partition null and z

at two coarse-graining conventions, because they answer different objections:

  fixed k         the same k at every tau. Isolates the effect of *which units* are kept,
                  holding granularity constant -- but as n_active falls, k/n rises, which
                  inflates R^2 on its own (the matched null absorbs this).
  proportional k  k = round(0.20 * n_active), the policy the paper's operating k uses.
                  Holds k/n constant instead.

If R^2 tracks tau, the result is threshold-dependent. If the excess over the matched null
is flat across tau, it is not.

Needs the checkpoint (for the full W_rec) and variance_epoch.pkl (for the per-unit
variance profile), so it covers the networks whose weights are still on the data volume.

Writes results/signed_input_wl_paper/lumpability/active_threshold_sweep.csv.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path

import numpy as np

from flexsym import paths
from flexsym.run.lumpability_r2_ksweep import GRID, leading_eigvecs, scores
from flexsym.structural.signed_input_wl_structural_roles import (
    apply_matrix_variant,
    approx_wl_signed,
    load_w_rec,
    random_labels_same_sizes,
)

ACT_ROOT = paths.REPO_ROOT / "results" / "activity_clusters_paper"
OUT = paths.REPO_ROOT / "results/signed_input_wl_paper/lumpability/active_threshold_sweep.csv"

TAUS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
FIXED_K = 20
FRACTION = 0.20
N_NULL = 30
TOP_N = 12


def resolve(model: str) -> Path | None:
    for store in (paths.ORIGINAL_STORE, paths.STAGED_STORE):
        if (store / model).exists():
            return store / model
    return None


def unit_variance_profile(model: str) -> np.ndarray | None:
    """Per-unit summed task-epoch variance -- the quantity the threshold is applied to."""
    p = ACT_ROOT / model / "variance_epoch.pkl"
    if not p.exists():
        return None
    with p.open("rb") as f:
        try:
            d = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            d = pickle.load(f, encoding="latin1")
    # h_var_all is units x task_epochs; the threshold is applied to the row sum
    return np.asarray(d["h_var_all"]).sum(axis=1)


def score_at(W: np.ndarray, k: int, n_null: int, top_n: int) -> dict | None:
    Wp, Wn = np.maximum(W, 0.0), np.maximum(-W, 0.0)
    labels, _ = approx_wl_signed(Wp, Wn, k, n_iter=20, seed=0, n_init=20)
    kk = int(np.unique(labels).size)
    if kk < 2:
        return None
    V = leading_eigvecs(W, W.shape[0])
    Vb, Vk = V[:top_n], V[:kk]
    best, topk = scores(Vb, Vk, labels)
    nb = np.empty(n_null)
    nk = np.empty(n_null)
    for b in range(n_null):
        rl = random_labels_same_sizes(labels, seed=b)
        nb[b], nk[b] = scores(Vb, Vk, rl)
    sk = nk.std(ddof=1)
    return {"k": kk, "r2_best": best, "r2_best_null_mean": float(nb.mean()),
            "r2_topk": topk, "r2_topk_null_mean": float(nk.mean()),
            "z_topk": float((topk - nk.mean()) / sk) if sk > 0 else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", default=",".join(str(t) for t in TAUS))
    ap.add_argument("--fixed-k", type=int, default=FIXED_K)
    ap.add_argument("--fraction", type=float, default=FRACTION)
    ap.add_argument("--n-null", type=int, default=N_NULL)
    args = ap.parse_args()
    taus = [float(x) for x in args.taus.split(",")]

    rows, skipped = [], []
    t0 = time.time()
    for size, seed in GRID:
        model = f"LeakyRNN_softplus_diag_{size}_all_{seed}"
        md = resolve(model)
        prof = unit_variance_profile(model)
        if md is None or prof is None:
            skipped.append(f"{model}({'no weights' if md is None else 'no variance'})")
            continue
        W_full = apply_matrix_variant(load_w_rec(md), "transpose")

        for tau in taus:
            ind = np.where(prof > tau)[0]
            if ind.size < 10:
                continue
            W = W_full[np.ix_(ind, ind)]
            for policy, k in (("fixed", args.fixed_k),
                              ("proportional", max(3, int(round(args.fraction * ind.size))))):
                if k >= ind.size:
                    continue
                res = score_at(W, k, args.n_null, TOP_N)
                if res is None:
                    continue
                rows.append({"model": model, "size": size, "seed": seed, "tau": tau,
                             "n_active": int(ind.size),
                             "active_frac": float(ind.size / prof.size),
                             "k_policy": policy, **res})
        got = [r for r in rows if r["model"] == model and r["k_policy"] == "fixed"]
        print(f"[{time.time()-t0:5.0f}s] {model}: "
              + "  ".join(f"tau={r['tau']:g} n={r['n_active']} R2={r['r2_topk']:.3f}"
                          f"(null {r['r2_topk_null_mean']:.3f})" for r in got), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows)} rows)")
    if skipped:
        print("skipped: " + ", ".join(skipped))


if __name__ == "__main__":
    main()
