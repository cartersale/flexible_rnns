"""Equitability defect of trained vs row-shuffled W_rec, across the k grid.

The signed equitability defect is

    Gamma_signed(H) = ||W+ H - H D^-1 H^T W+ H||_F + ||W- H - H D^-1 H^T W- H||_F.

For each network, k, and matrix in {trained, row-shuffle}, a partition is optimised
on that matrix and scored against size-matched random partitions as

    S = (mu_null - Gamma_obs) / sigma_null      (positive = more equitable than chance)

Writes results/signed_input_wl_paper/lumpability/equitability_control_ksweep.csv.
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.run.control_ksweep import K_GRID, make_matrix
from flexsym.run.lumpability_r2_ksweep import COHORTS
from flexsym.structural.signed_input_wl_structural_roles import (
    approx_wl_signed,
    gamma_signed,
    labels_to_H,
    random_labels_same_sizes,
)

WL_ROOT = paths.RESULTS_ROOT / "signed_input_wl_paper"
OUT = WL_ROOT / "lumpability" / "equitability_control_ksweep.csv"
TYPES = ["trained", "row-shuffle"]
N_NULL = 200


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-grid", default=",".join(str(k) for k in K_GRID))
    ap.add_argument("--n-null", type=int, default=N_NULL)
    ap.add_argument("--types", default=",".join(TYPES))
    ap.add_argument("--cohort", choices=sorted(COHORTS), default="driscoll33",
                    help="Network cohort (see run/lumpability_r2_ksweep.py).")
    ap.add_argument("--out", type=str, default=None, help="Override the output CSV path.")
    args = ap.parse_args()
    kg = [int(x) for x in args.k_grid.split(",")]
    types = args.types.split(",")

    rows = []
    t0 = time.time()
    for size, seed in COHORTS[args.cohort]:
        model = f"LeakyRNN_softplus_diag_{size}_all_{seed}"
        npz = WL_ROOT / model / "transpose" / "signed_input_wl_sweep.npz"
        if not npz.exists():
            print(f"MISSING {model}", flush=True)
            continue
        W = np.asarray(npload(npz)["W_rec"], dtype=float)

        for kind in types:
            rng = np.random.default_rng(abs(hash((seed, size, kind))) % (2**32))
            M = make_matrix(W, kind, rng)
            Mp, Mn = np.maximum(M, 0.0), np.maximum(-M, 0.0)
            for k in kg:
                labels, H = approx_wl_signed(Mp, Mn, k, n_iter=20, seed=0, n_init=20)
                kk = int(np.unique(labels).size)
                if kk < 2:
                    continue
                g = gamma_signed(Mp, Mn, H)
                null = np.array([
                    gamma_signed(Mp, Mn,
                                 labels_to_H(random_labels_same_sizes(labels, seed=b), k))
                    for b in range(args.n_null)
                ])
                sd = null.std(ddof=1)
                rows.append({
                    "model": model, "size": size, "seed": seed, "type": kind,
                    "k_req": k, "k": kk, "n_active": W.shape[0],
                    "gamma": float(g), "gamma_null_mean": float(null.mean()),
                    "gamma_null_sd": float(sd),
                    # orientation-corrected: positive = more equitable than chance
                    "equit_score": float((null.mean() - g) / sd) if sd > 0 else np.nan,
                })
            done = [r for r in rows if r["model"] == model and r["type"] == kind]
            print(f"  {model} {kind:12s} "
                  + " ".join(f"k{r['k']}:{r['equit_score']:+.0f}" for r in done), flush=True)
        print(f"[{time.time()-t0:6.0f}s] {model} done", flush=True)

    out_path = paths.REPO_ROOT / args.out if args.out else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
