"""Lumpability across a k-sweep (rather than at a single 20%-of-active operating scale).

For each k in a grid we take the signed-WL structural partition at that k (from the cached
labels_struct_by_k of the transpose sweep), and compute the eigenvector-leakage and
subspace-invariance scores of lumpability_check against size-matched random partitions.
One row per (seed, k) for the ten 256-unit networks. Writes lumpability_ksweep.csv.
"""
from __future__ import annotations
import argparse, csv
import numpy as np

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.run.lumpability_check import (
    resolve, ACT_ROOT, ACT_NAME, pw_constant_projector, eig_leakage, subspace_invariance,
)
from flexsym.structural.signed_input_wl_structural_roles import load_w_rec, random_labels_same_sizes

OUT = paths.REPO_ROOT / "results/signed_input_wl_paper/lumpability/lumpability_ksweep.csv"
K_GRID = [8, 12, 16, 20, 25, 30, 40, 50]


def score(M, labels, k, n_null, rng_seed=0):
    P = pw_constant_projector(labels, k)
    leak = eig_leakage(M, P, k)
    inv = subspace_invariance(M, P)
    nl_leak = np.empty(n_null); nl_inv = np.empty(n_null)
    for b in range(n_null):
        nl = random_labels_same_sizes(labels, seed=b)
        Pn = pw_constant_projector(nl, k)
        nl_leak[b] = eig_leakage(M, Pn, k)
        nl_inv[b] = subspace_invariance(M, Pn)
    lz = (leak - nl_leak.mean()) / nl_leak.std(ddof=1) if nl_leak.std(ddof=1) > 0 else np.nan
    iz = (inv - nl_inv.mean()) / nl_inv.std(ddof=1) if nl_inv.std(ddof=1) > 0 else np.nan
    return -lz, -iz    # scores: positive = less leakage / more invariant than chance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--n-null", type=int, default=200)
    ap.add_argument("--k-grid", default=None)
    args = ap.parse_args()
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-"); seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    kg = [int(x) for x in args.k_grid.split(",")] if args.k_grid else K_GRID

    rows = []
    for s in seeds:
        model = f"LeakyRNN_softplus_diag_{args.size}_all_{s}"
        md = resolve(model)
        ind = np.asarray(np.load(ACT_ROOT / model / ACT_NAME, allow_pickle=True)["ind_active"], int)
        W = load_w_rec(md)[np.ix_(ind, ind)]
        wl = npload(WL_ROOT / model / "transpose" / "signed_input_wl_sweep.npz")
        ls = wl["labels_struct_by_k"]
        for k in kg:
            labels = np.asarray(ls[_row_for_k(ls, k)], int)
            kk = int(labels.max() + 1)
            if kk < 2:
                continue
            lzT, izT = score(W.T, labels, kk, args.n_null)
            lzW, _ = score(W, labels, kk, args.n_null)
            rows.append({"seed": s, "k_req": k, "k": kk, "leak_score_T": lzT,
                         "leak_score_W": lzW, "inv_score_T": izT})
            print(f"  seed {s} k={kk}: leakT {lzT:+.1f} leakW {lzW:+.1f} inv {izT:+.1f}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
