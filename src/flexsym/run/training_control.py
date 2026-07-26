"""Training control: is the equitable/lumpable structure CREATED BY TRAINING?

The within-analysis null (random partitions of the same matrix) shows the WL
optimization finds a good partition — but any matrix might optimize to something
better-than-random. This control runs the IDENTICAL pipeline (WL-optimize a
partition at operating k, then measure equitability z and eigenvector step-R^2)
on the trained weights vs structure-destroying baselines:

  trained       : the real recurrent matrix (transpose variant, active units)
  row-shuffle   : permute each row's entries  -> preserves each unit's OUT-strength
                  (the degree budget) but destroys the wiring pattern
  full-shuffle  : permute all entries         -> preserves value distribution only
  gaussian      : i.i.d. normal matched to ||W||_F -> an untrained/random network

If trained >> all baselines, training created the structure (not optimization +
degree). Pure connectivity — no TF model load.
"""

from __future__ import annotations

from flexsym import paths

import csv
import sys
from pathlib import Path

import numpy as np


from flexsym.structural.signed_input_wl_structural_roles import (  # noqa: E402
    apply_matrix_variant, approx_wl_signed, gamma_signed, labels_to_H,
    load_w_rec, random_labels_same_sizes,
)

REPO = paths.REPO_ROOT
ACT = REPO / "results/activity_clusters_paper"
ORIG = paths.ORIGINAL_STORE
STAGED = paths.STAGED_STORE
OPK = {0: 41, 1: 51, 2: 38, 3: 14, 4: 14, 5: 14, 6: 14, 7: 14, 8: 14, 9: 15}
TYPES = ["trained", "row-shuffle", "full-shuffle", "gaussian"]


def resolve(m):
    for s in (ORIG, STAGED):
        if (s / m).exists():
            return s / m
    return None


def make_matrix(W, kind, rng):
    n = W.shape[0]
    if kind == "trained":
        return W
    if kind == "row-shuffle":
        out = np.empty_like(W)
        for i in range(n):
            out[i] = W[i, rng.permutation(n)]
        return out
    if kind == "full-shuffle":
        flat = W.flatten()
        return rng.permutation(flat).reshape(n, n)
    if kind == "gaussian":
        g = rng.normal(0, 1, size=(n, n))
        return g * (np.linalg.norm(W) / np.linalg.norm(g))
    raise ValueError(kind)


def step_r2(v, labels):
    pred = np.zeros_like(v)
    for c in np.unique(labels):
        pred[labels == c] = v[labels == c].mean()
    return 1 - ((v - pred) ** 2).sum() / ((v - v.mean()) ** 2).sum()


def best_eig_r2(M, labels, topn=12):
    w, V = np.linalg.eig(M)
    order = np.argsort(-np.abs(w))[:topn]
    best = -1
    for i in order:
        v = np.real(V[:, i]); nv = np.linalg.norm(v)
        if nv:
            best = max(best, step_r2(v / nv, labels))
    return best


def analyse_matrix(M, k, rng, n_null=60):
    Wp, Wn = np.maximum(M, 0), np.maximum(-M, 0)
    labels, H = approx_wl_signed(Wp, Wn, k, n_iter=20, seed=0, n_init=20)
    gamma = gamma_signed(Wp, Wn, H)
    null_g = np.array([gamma_signed(Wp, Wn, labels_to_H(random_labels_same_sizes(labels, seed=b), k))
                       for b in range(n_null)])
    z_eq = (gamma - null_g.mean()) / null_g.std()
    r2 = best_eig_r2(M, labels)
    null_r2 = np.array([best_eig_r2(M, random_labels_same_sizes(labels, seed=b)) for b in range(30)])
    z_lump = (r2 - null_r2.mean()) / null_r2.std()
    return z_eq, r2, z_lump


def main():
    rows = []
    for seed, k in OPK.items():
        model = f"LeakyRNN_softplus_diag_256_all_{seed}"
        md = resolve(model); act = ACT / model / "epoch_variance_unit_clusters_paper_max.npz"
        if md is None or not act.exists():
            continue
        ind = np.asarray(np.load(act, allow_pickle=True)["ind_active"], dtype=np.int64)
        W = apply_matrix_variant(load_w_rec(md), "transpose")[np.ix_(ind, ind)]
        rng = np.random.default_rng(seed)
        rec = {"seed": seed, "k": k}
        for kind in TYPES:
            M = make_matrix(W, kind, rng)
            z_eq, r2, z_lump = analyse_matrix(M, k, rng)
            rec[f"{kind}_equit_z"] = z_eq
            rec[f"{kind}_lump_r2"] = r2
            rec[f"{kind}_lump_z"] = z_lump
        rows.append(rec)
        print(f"seed {seed}: equit_z " +
              " ".join(f"{t[:4]}={rec[f'{t}_equit_z']:+.1f}" for t in TYPES) +
              "  | lump_R² " + " ".join(f"{t[:4]}={rec[f'{t}_lump_r2']:.2f}" for t in TYPES), flush=True)

    out = REPO / "results/signed_input_wl_paper/training_control.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    import numpy as _np
    print("\n=== across 10 seeds (256): mean ± sd ===")
    print(f"{'type':13} {'equitability z':>18} {'lumpability step-R²':>20} {'lumpability z':>15}")
    for t in TYPES:
        ez = _np.array([r[f'{t}_equit_z'] for r in rows])
        r2 = _np.array([r[f'{t}_lump_r2'] for r in rows])
        lz = _np.array([r[f'{t}_lump_z'] for r in rows])
        print(f"{t:13} {ez.mean():>8.1f} ± {ez.std(ddof=1):>4.1f}   {r2.mean():>9.2f} ± {r2.std(ddof=1):>4.2f}   {lz.mean():>7.1f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
