"""Eigenvector step-R^2 as a function of the coarse-graining scale k.

The headline lumpability number in the paper is reported at each network's single
"operating" k (results/.../lumpability/step_r2_all.csv). That leaves k as a free
parameter chosen by a separate policy -- and the policy is not the same for the
released Driscoll seeds (0-2, k = activity-cluster count) and our own seeds (3-9,
k = 20% of active units). This script removes k as a free parameter: it recomputes
the same measurement at EVERY k in the cached WL sweep, for every network whose
signed-WL sweep is cached.

Measurement (identical to run/training_control.py, applied to the transpose variant
of W_rec restricted to active units):

  step_r2(v, labels) = 1 - SS_res / SS_tot  of approximating an eigenvector v by its
                       per-cell mean; 1 = perfectly piecewise-constant on cells.

  r2_best   : max step_r2 over the TOP_N eigenvectors of largest |lambda|
              (this is the statistic behind the published numbers)
  r2_topk   : mean step_r2 over the k eigenvectors of largest |lambda|
              (no best-of selection; reported alongside so the headline number
               cannot be read as a selection artefact)

Both are compared against `n_null` size-matched random partitions of the same cell-size
profile -> z-score. Positive z = the structural partition makes eigenvectors more
piecewise-constant than a partition with identical granularity.

Inputs are the cached sweeps only -- W_rec (transpose variant, active units) and
labels_struct_by_k both live in the npz -- so this runs without the checkpoint volume.

Writes results/signed_input_wl_paper/lumpability/step_r2_ksweep.csv (one row per
network x k).
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.structural.signed_input_wl_structural_roles import random_labels_same_sizes

WL_ROOT = paths.RESULTS_ROOT / "signed_input_wl_paper"
OUT = WL_ROOT / "lumpability" / "step_r2_ksweep.csv"

#: The original 33-network cohort: seeds 0-9 at 128/256/512 plus seeds 0-2 at 1024.
#: Seeds 0-2 at every size are the RELEASED Driscoll et al. (2024) networks, so all
#: three 1024s are released nets; seeds 3-9 were trained here.
GRID_DRISCOLL33 = [(s, i) for s in (128, 256, 512) for i in range(10)] + [(1024, i) for i in range(3)]

#: Our own training grid (MODEL_ROOT/flexsym_grid), no released networks: seeds 3-12
#: at 128/256/512, 30 networks.
#:
#: 512 seeds 3-9 were excluded until 2026-08-20 because the cached results under those
#: names came from an older, discarded 512 set (MODEL_ROOT/old_new_runs). Those networks
#: have since been retrained into flexsym_grid, the staged symlinks repointed, and the
#: stale results quarantined in results/retired/, so the exclusion is lifted.
GRID_FLEXSYM = [(s, i) for s in (128, 256, 512) for i in range(3, 13)]

COHORTS = {"driscoll33": GRID_DRISCOLL33, "flexsym": GRID_FLEXSYM}

#: Default kept at the published cohort so existing outputs stay reproducible.
GRID = GRID_DRISCOLL33

TOP_N = 12          # matches best_eig_r2 in run/training_control.py
N_NULL = 30         # matches the null size used there


def step_r2(v: np.ndarray, labels: np.ndarray) -> float:
    """R^2 of approximating v by its per-cell mean (1 = perfect step function)."""
    pred = np.zeros_like(v)
    for c in np.unique(labels):
        m = labels == c
        pred[m] = v[m].mean()
    denom = ((v - v.mean()) ** 2).sum()
    return float(1.0 - ((v - pred) ** 2).sum() / denom) if denom > 0 else 0.0


def leading_blocks(M: np.ndarray, n: int) -> list[np.ndarray]:
    """The leading real spectral blocks of M, as orthonormal bases, until n dimensions.

    Eigenvalues are taken in order of decreasing |lambda|. A real eigenvalue gives a
    one-column block, its unit eigenvector; a complex-conjugate pair gives a two-column
    orthonormal basis of its real invariant plane, and the partner is not visited again.
    A block is never split, so the last one may carry the total past n.

    Blocks rather than individual vectors because Re(z) and Im(z) of a complex
    eigenvector depend on the phase the eigensolver happens to return, while the plane
    they span does not.
    """
    w, V = np.linalg.eig(M)
    order = np.argsort(-np.abs(w))
    blocks, used, dims = [], set(), 0
    for i in order:
        if dims >= n:
            break
        if i in used:
            continue
        used.add(i)
        if abs(w[i].imag) < 1e-9:
            v = np.real(V[:, i])
            nv = np.linalg.norm(v)
            if not nv:
                continue
            B = (v / nv)[:, None]
        else:
            used.add(int(np.argmin(np.abs(w - np.conj(w[i])))))   # conjugate partner
            B, _ = np.linalg.qr(np.column_stack([np.real(V[:, i]), np.imag(V[:, i])]))
        blocks.append(B)
        dims += B.shape[1]
    return blocks


def step_r2_block(B: np.ndarray, labels: np.ndarray) -> float:
    """Piecewise-constant R^2 of a real spectral block.

    1 - ||(I-P)B||_F^2 / ||(I-P_1)B||_F^2 with P the projector onto the role-constant
    subspace and P_1 the projector onto the constant vector. For a one-column block this
    is exactly step_r2 of that eigenvector; for a two-column block it scores the whole
    invariant plane and is unchanged by any rotation of Re(z) and Im(z) within it.
    """
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B[:, None]
    res = B.copy()
    for c in np.unique(labels):
        m = labels == c
        res[m] -= B[m].mean(axis=0)
    tot = B - B.mean(axis=0)
    denom = float((tot ** 2).sum())
    return float(1.0 - (res ** 2).sum() / denom) if denom > 0 else 0.0


def block_prefix(blocks: list[np.ndarray], n: int) -> list[np.ndarray]:
    """The leading blocks covering at least n real dimensions, never splitting one."""
    out, dims = [], 0
    for B in blocks:
        if dims >= n:
            break
        out.append(B)
        dims += B.shape[1]
    return out


def leading_eigvecs(M: np.ndarray, n: int) -> np.ndarray:
    """The columns of the leading real spectral blocks, unit-normalised, as rows.

    For display and for code that wants individual directions. The reported statistic is
    computed per block (see step_r2_block), because a single column of a complex block
    turns with the eigenvector's phase.
    """
    out = []
    for B in leading_blocks(M, n):
        out.extend(B.T)
    return np.asarray(out[:n])


def scores(blocks_best: list[np.ndarray], blocks_topk: list[np.ndarray],
           labels: np.ndarray) -> tuple[float, float]:
    """Best and mean block R^2. The mean weights each block by its dimension, so it is
    still an average over real directions and reduces to the per-eigenvector mean when
    every leading eigenvalue is real."""
    best = max(step_r2_block(B, labels) for B in blocks_best)
    vals = [(step_r2_block(B, labels), B.shape[1]) for B in blocks_topk]
    topk = float(sum(v * d for v, d in vals) / sum(d for _, d in vals))
    return best, topk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=N_NULL)
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--models", default=None,
                    help="Comma-separated model names, overriding the softplus/diag GRID. "
                         "Use to score other architectures trained on the same 15 tasks.")
    ap.add_argument("--cohort", choices=sorted(COHORTS), default="driscoll33",
                    help="Network cohort: 'driscoll33' is the published 33-network grid "
                         "(includes the released seeds 0-2); 'flexsym' is our own "
                         "training grid only.")
    ap.add_argument("--out", type=str, default=None, help="Override the output CSV path.")
    args = ap.parse_args()

    if args.models:
        grid = []
        for m in args.models.split(","):
            parts = m.split("_")
            grid.append((int(parts[-3]), int(parts[-1]), m))
    else:
        grid = [(size, seed, f"LeakyRNN_softplus_diag_{size}_all_{seed}")
                for size, seed in COHORTS[args.cohort]]
    out_path = paths.REPO_ROOT / args.out if args.out else OUT

    rows, missing = [], []
    for size, seed, model in grid:
        npz = WL_ROOT / model / "transpose" / "signed_input_wl_sweep.npz"
        if not npz.exists():
            missing.append(model)
            print(f"MISSING sweep cache: {model}", flush=True)
            continue
        d = npload(npz)
        W = np.asarray(d["W_rec"], dtype=float)          # transpose variant, active units
        ls = np.asarray(d["labels_struct_by_k"], dtype=int)
        n_active = W.shape[0]
        # one eigendecomposition per network; every k reuses it
        B_all = leading_blocks(W, n_active)

        for row in range(ls.shape[0]):
            labels = ls[row]
            k = int(np.unique(labels).size)
            if k < 2:
                continue
            k_req = row + 3                               # sweep rows are k = 3..60
            # blocks are added until at least the requested number of real dimensions
            # is covered, so a complex pair is never split at the cutoff
            Vb = block_prefix(B_all, args.top_n)
            Vk = block_prefix(B_all, k)
            best, topk = scores(Vb, Vk, labels)
            nb = np.empty(args.n_null)
            nk = np.empty(args.n_null)
            for b in range(args.n_null):
                rl = random_labels_same_sizes(labels, seed=b)
                nb[b], nk[b] = scores(Vb, Vk, rl)
            rows.append({
                "model": model, "size": size, "seed": seed, "arch": model.rsplit("_", 3)[0],
                "k_req": k_req, "k": k, "n_active": n_active,
                "r2_best": best,
                "r2_best_null_mean": float(nb.mean()), "r2_best_null_sd": float(nb.std(ddof=1)),
                "z_best": float((best - nb.mean()) / nb.std(ddof=1)) if nb.std(ddof=1) > 0 else np.nan,
                "r2_topk": topk,
                "r2_topk_null_mean": float(nk.mean()), "r2_topk_null_sd": float(nk.std(ddof=1)),
                "z_topk": float((topk - nk.mean()) / nk.std(ddof=1)) if nk.std(ddof=1) > 0 else np.nan,
            })
        print(f"{model}: n_active={n_active}, {ls.shape[0]} k values", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path} ({len(rows)} rows, {len(grid) - len(missing)}/{len(grid)} networks)")
    if missing:
        print("networks with no cached WL sweep: " + ", ".join(missing))


if __name__ == "__main__":
    main()
