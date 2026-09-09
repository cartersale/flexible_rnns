"""Is the *partition* reproducible, or only the *subspace* it spans?

The signed-WL refinement is a k-means-based approximation, so different random restarts
land on different labellings. If we reported only the labels, that instability would be a
problem. The claim in Act 1 is not about labels, though -- it is about the group-constant
subspace col(H) being approximately invariant. This script separates the two:

  ARI        Adjusted Rand Index between every pair of restart partitions. Low = the
             restarts disagree about which unit goes in which cell.
  overlap    ||P_a P_b||_F^2 / k for the two piecewise-constant projectors. This is the
             mean squared cosine of the principal angles between col(H_a) and col(H_b),
             i.e. how much of the same k-dimensional subspace the two restarts span.
             1 = identical subspace, k/n = orthogonal-as-possible.
  r2_topk    the lumpability statistic itself, per restart. Low spread = the measurement
             does not depend on which restart you happen to get.

Each is compared against the matched quantity for size-matched RANDOM partitions built to
mirror each restart's own cell-size profile, so "restarts agree more than chance" is a
statement about structure and not about granularity.

The expected pattern (and the one the paper claims): ARI is low -- restarts genuinely
disagree about labels -- while overlap and r2_topk are tightly concentrated and far above
their random counterparts. Label identity is not the carrier of the result; the subspace
is.

Pure connectivity: reads the cached sweep npz. Writes
results/signed_input_wl_paper/lumpability/restart_ari_vs_r2.csv (one row per network x k)
and ..._pairs.csv (one row per network x k x restart pair, for plotting distributions).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.run.lumpability_r2_ksweep import COHORTS, GRID, block_prefix, leading_blocks, scores
from flexsym.structural.signed_input_wl_structural_roles import (
    approx_wl_signed,
    random_labels_same_sizes,
)

WL_ROOT = paths.RESULTS_ROOT / "signed_input_wl_paper"
OUT = WL_ROOT / "lumpability" / "restart_ari_vs_r2.csv"
OUT_PAIRS = WL_ROOT / "lumpability" / "restart_ari_vs_r2_pairs.csv"

K_GRID = [10, 25, 40]
N_RESTARTS = 20


def projector(labels: np.ndarray) -> np.ndarray:
    """P = H (H^T H)^-1 H^T for the piecewise-constant subspace of a labelling."""
    cells = np.unique(labels)
    H = np.zeros((labels.size, cells.size))
    for j, c in enumerate(cells):
        H[labels == c, j] = 1.0
    return H @ np.diag(1.0 / H.sum(axis=0)) @ H.T


def overlap(Pa: np.ndarray, Pb: np.ndarray, k: int) -> float:
    """||Pa Pb||_F^2 / k = mean squared cosine of the principal angles."""
    return float(np.sum((Pa @ Pb) ** 2) / k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=sorted(COHORTS), default=None,
                    help="network cohort; default keeps the published grid")
    ap.add_argument("--k-grid", default=",".join(str(k) for k in K_GRID))
    ap.add_argument("--n-restarts", type=int, default=N_RESTARTS)
    ap.add_argument("--top-n", type=int, default=12)
    args = ap.parse_args()
    kg = [int(x) for x in args.k_grid.split(",")]
    grid = COHORTS[args.cohort] if args.cohort else GRID

    rows, pair_rows = [], []
    t0 = time.time()
    for size, seed in grid:
        model = f"LeakyRNN_softplus_diag_{size}_all_{seed}"
        npz = WL_ROOT / model / "transpose" / "signed_input_wl_sweep.npz"
        if not npz.exists():
            print(f"MISSING {model}", flush=True)
            continue
        W = np.asarray(npload(npz)["W_rec"], dtype=float)
        Wp, Wn = np.maximum(W, 0.0), np.maximum(-W, 0.0)
        B_all = leading_blocks(W, W.shape[0])

        for k in kg:
            labs, r2b, r2t = [], [], []
            for r in range(args.n_restarts):
                # a different WL seed per restart: different k-means initialisation
                lab, _ = approx_wl_signed(Wp, Wn, k, n_iter=20, seed=r, n_init=1)
                if np.unique(lab).size < 2:
                    continue
                labs.append(np.asarray(lab, int))
                kk = int(np.unique(lab).size)
                b, t = scores(block_prefix(B_all, args.top_n), block_prefix(B_all, kk), lab)
                r2b.append(b)
                r2t.append(t)
            if len(labs) < 2:
                continue
            projs = [projector(l) for l in labs]
            # matched random partitions: one per restart, same cell-size profile
            rlabs = [random_labels_same_sizes(l, seed=1000 + i) for i, l in enumerate(labs)]
            rprojs = [projector(l) for l in rlabs]
            rr2t = [scores(block_prefix(B_all, args.top_n),
                           block_prefix(B_all, int(np.unique(l).size)), l)[1]
                    for l in rlabs]

            aris, ovs, r_aris, r_ovs = [], [], [], []
            for a, b in itertools.combinations(range(len(labs)), 2):
                ari = adjusted_rand_score(labs[a], labs[b])
                ov = overlap(projs[a], projs[b], k)
                aris.append(ari)
                ovs.append(ov)
                r_aris.append(adjusted_rand_score(rlabs[a], rlabs[b]))
                r_ovs.append(overlap(rprojs[a], rprojs[b], k))
                pair_rows.append({"model": model, "size": size, "seed": seed, "k": k,
                                  "a": a, "b": b, "ari": ari, "overlap": ov,
                                  "ari_rand": r_aris[-1], "overlap_rand": r_ovs[-1]})

            rows.append({
                "model": model, "size": size, "seed": seed, "k": k,
                "n_active": W.shape[0], "n_restarts": len(labs),
                "ari_mean": float(np.mean(aris)), "ari_sd": float(np.std(aris)),
                "ari_rand_mean": float(np.mean(r_aris)),
                "overlap_mean": float(np.mean(ovs)), "overlap_sd": float(np.std(ovs)),
                "overlap_rand_mean": float(np.mean(r_ovs)),
                "r2_topk_mean": float(np.mean(r2t)), "r2_topk_sd": float(np.std(r2t)),
                "r2_topk_cv": float(np.std(r2t) / np.mean(r2t)) if np.mean(r2t) else np.nan,
                "r2_topk_rand_mean": float(np.mean(rr2t)),
                "r2_best_mean": float(np.mean(r2b)), "r2_best_sd": float(np.std(r2b)),
            })
            r = rows[-1]
            print(f"[{time.time()-t0:5.0f}s] {model} k={k}: ARI {r['ari_mean']:.3f}"
                  f" (rand {r['ari_rand_mean']:+.3f})  overlap {r['overlap_mean']:.3f}"
                  f" (rand {r['overlap_rand_mean']:.3f})  R2_topk {r['r2_topk_mean']:.3f}"
                  f" +-{r['r2_topk_sd']:.3f} (rand {r['r2_topk_rand_mean']:.3f})", flush=True)

    for path, data in ((OUT, rows), (OUT_PAIRS, pair_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
        print(f"wrote {path} ({len(data)} rows)")


if __name__ == "__main__":
    main()
