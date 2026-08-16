"""Surrogate-W controls for the mesoscopic-lumpability claim, across the k grid.

Section 1.2 shows that the signed-WL partition of the trained W_rec makes the leading
eigenvectors piecewise-constant far beyond a size-matched random partition, at every k.
That is a claim about the *partition*. This script asks the complementary question about
the *matrix*: which properties of the trained W_rec are responsible?

For each surrogate matrix type we run the FULL pipeline on the surrogate -- WL-optimize a
fresh partition on it at each k, then score its eigenvectors against ITS OWN size-matched
random-partition null. Each surrogate therefore gets its best shot; a low score means the
structure genuinely is not there, not that we handed it the wrong partition.

Surrogate types (all built from M = the transpose variant of W_rec on active units):

  trained        the real matrix (reference curve)
  row-shuffle    permute the entries within each row -> preserves every unit's incoming
                 strength profile (sum, distribution, sparsity) and kills only the pattern
                 of who-connects-to-whom.  [strength-preserving surrogate]
  full-shuffle   permute all entries -> preserves the global value distribution only
  gaussian       i.i.d. normal matched in Frobenius norm.  This is also the *untrained*
                 control: the `diag` initialisation is exactly 1.0 * I, for which every
                 partition is trivially invariant, so the identity is a degenerate null
                 and a matched Gaussian draw is the meaningful untrained geometry.
  lowrank-r<R>   keep the exact rank-R SVD component of M, replace the residual with a
                 Frobenius-matched Gaussian.  Asks whether the lumpability is carried by
                 the leading low-rank component alone.
  spectrum       real Schur M = U T U^T with U replaced by a random orthogonal Q.
                 Preserves the ENTIRE spectrum (hence spectral radius and all decay
                 times) exactly, and randomises only the basis alignment.  Asks whether
                 piecewise-constant eigenvectors follow from the spectrum or from where
                 the eigenvectors sit relative to the units.

Statistics per (network, type, k) are the same two as run/lumpability_r2_ksweep.py:
`r2_best` (max step-R^2 over the top-12 eigenvectors) and `r2_topk` (mean over the top-k),
each with a 30-draw size-matched random-partition null and a z.

Pure connectivity: reads the cached sweep npz (which stores the active-unit matrix), so
no checkpoint volume and no TF.

Writes results/signed_input_wl_paper/lumpability/control_ksweep.csv.
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np
from scipy.linalg import schur

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.run.lumpability_r2_ksweep import GRID, leading_eigvecs, scores
from flexsym.structural.signed_input_wl_structural_roles import (
    approx_wl_signed,
    random_labels_same_sizes,
)

WL_ROOT = paths.REPO_ROOT / "results" / "signed_input_wl_paper"
OUT = WL_ROOT / "lumpability" / "control_ksweep.csv"

K_GRID = [3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60]
TYPES = ["trained", "row-shuffle", "full-shuffle", "gaussian",
         "lowrank-r5", "lowrank-r20", "spectrum"]
TOP_N = 12
N_NULL = 30


def make_matrix(W: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    n = W.shape[0]
    if kind == "trained":
        return W
    if kind == "row-shuffle":
        out = np.empty_like(W)
        for i in range(n):
            out[i] = W[i, rng.permutation(n)]
        return out
    if kind == "full-shuffle":
        return rng.permutation(W.flatten()).reshape(n, n)
    if kind == "gaussian":
        g = rng.normal(0, 1, size=(n, n))
        return g * (np.linalg.norm(W) / np.linalg.norm(g))
    if kind.startswith("lowrank-r"):
        r = int(kind.split("r")[-1])
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        keep = U[:, :r] @ np.diag(S[:r]) @ Vt[:r]
        resid = W - keep
        g = rng.normal(0, 1, size=(n, n))
        nr, ng = np.linalg.norm(resid), np.linalg.norm(g)
        return keep + (g * (nr / ng) if ng > 0 else g)
    if kind == "spectrum":
        # real Schur: W = U T U^T with T quasi-upper-triangular. Replacing U by a random
        # orthogonal Q leaves the eigenvalues (which live in T) untouched.
        T, U = schur(W, output="real")
        Q, _ = np.linalg.qr(rng.normal(0, 1, size=(n, n)))
        return Q @ T @ Q.T
    raise ValueError(kind)


def analyse(M: np.ndarray, k: int, n_null: int, top_n: int, wl_seed: int) -> dict | None:
    """WL-optimize a partition on M at k, then score M's eigenvectors on it."""
    Wp, Wn = np.maximum(M, 0.0), np.maximum(-M, 0.0)
    labels, _ = approx_wl_signed(Wp, Wn, k, n_iter=20, seed=wl_seed, n_init=20)
    k_real = int(np.unique(labels).size)
    if k_real < 2:
        return None
    V = leading_eigvecs(M, M.shape[0])
    Vb, Vk = V[:top_n], V[:k_real]
    best, topk = scores(Vb, Vk, labels)
    nb = np.empty(n_null)
    nk = np.empty(n_null)
    for b in range(n_null):
        rl = random_labels_same_sizes(labels, seed=b)
        nb[b], nk[b] = scores(Vb, Vk, rl)
    sb, sk = nb.std(ddof=1), nk.std(ddof=1)
    return {
        "k": k_real,
        "r2_best": best, "r2_best_null_mean": float(nb.mean()),
        "z_best": float((best - nb.mean()) / sb) if sb > 0 else np.nan,
        "r2_topk": topk, "r2_topk_null_mean": float(nk.mean()),
        "z_topk": float((topk - nk.mean()) / sk) if sk > 0 else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=N_NULL)
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--types", default=",".join(TYPES))
    ap.add_argument("--k-grid", default=",".join(str(k) for k in K_GRID))
    ap.add_argument("--sizes", default="128,256,512,1024")
    args = ap.parse_args()

    types = args.types.split(",")
    kg = [int(x) for x in args.k_grid.split(",")]
    sizes = {int(x) for x in args.sizes.split(",")}

    rows, missing = [], []
    t0 = time.time()
    for size, seed in GRID:
        if size not in sizes:
            continue
        model = f"LeakyRNN_softplus_diag_{size}_all_{seed}"
        npz = WL_ROOT / model / "transpose" / "signed_input_wl_sweep.npz"
        if not npz.exists():
            missing.append(model)
            continue
        W = np.asarray(npload(npz)["W_rec"], dtype=float)
        for kind in types:
            # one rng per (network, type) so surrogates are reproducible and
            # independent of which k values were requested
            rng = np.random.default_rng(abs(hash((seed, size, kind))) % (2**32))
            M = make_matrix(W, kind, rng)
            for k in kg:
                res = analyse(M, k, args.n_null, args.top_n, wl_seed=0)
                if res is None:
                    continue
                rows.append({"model": model, "size": size, "seed": seed, "type": kind,
                             "k_req": k, "n_active": W.shape[0], **res})
            done = [r for r in rows if r["model"] == model and r["type"] == kind]
            print(f"  {model} {kind:13s} "
                  + " ".join(f"k{r['k']}:{r['r2_best']:.2f}/{r['z_topk']:+.0f}" for r in done),
                  flush=True)
        print(f"[{time.time() - t0:6.0f}s] {model} done", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows)} rows)")
    if missing:
        print("no cached sweep: " + ", ".join(missing))


if __name__ == "__main__":
    main()
