"""Restart ensembles for ALL network sizes (128/256/512/1024), not just 256.

Same procedure as restart_ensemble_partitions.py -- R independent WL restarts at the
network's operating k, plus a size-matched random ensemble mirroring each restart's
group-size profile -- but driven off lumpability.csv so it covers all 33 networks.

Writes one npz per network to results/signed_input_wl_paper/restart_ensemble_all_sizes/
containing restart_labels, restart_gamma, consensus_labels and random_labels, matching
the schema the 256-unit ensembles already use.
"""

from __future__ import annotations

from flexsym import paths

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


from flexsym.structural.restart_ensemble_partitions import resolve, size_matched_random  # noqa: E402
from flexsym.structural.signed_input_wl_structural_roles import (  # noqa: E402
    apply_matrix_variant, approx_wl_signed, gamma_signed, labels_to_H, load_w_rec,
)
from flexsym.structural.structural_partition_stability import _consensus_partition  # noqa: E402

REPO = paths.REPO_ROOT
ACT = paths.RESULTS_ROOT / "activity_clusters_paper"
LUMP = paths.RESULTS_ROOT / "signed_input_wl_paper/lumpability/lumpability.csv"
OUT = paths.RESULTS_ROOT / "signed_input_wl_paper/restart_ensemble_all_sizes"


def build(model: str, k: int, seed: int, n_restarts: int, variant: str) -> dict | None:
    md = resolve(model)
    act = ACT / model / "epoch_variance_unit_clusters_paper_max.npz"
    if md is None or not act.exists():
        print(f"SKIP {model}", flush=True)
        return None
    ind = np.asarray(np.load(act, allow_pickle=True)["ind_active"], dtype=np.int64)
    W = apply_matrix_variant(load_w_rec(md), variant)[np.ix_(ind, ind)]
    Wp, Wn = np.maximum(W, 0.0), np.maximum(-W, 0.0)
    n = len(ind)

    restart_labels = np.empty((n_restarts, n), dtype=np.int64)
    restart_gamma = np.empty(n_restarts, dtype=float)
    for r in range(n_restarts):
        labels, H = approx_wl_signed(Wp, Wn, k, n_iter=20, seed=r, n_init=20)
        restart_labels[r] = labels
        restart_gamma[r] = float(gamma_signed(Wp, Wn, H))

    consensus = _consensus_partition([restart_labels[r] for r in range(n_restarts)], k)
    consensus_gamma = float(gamma_signed(Wp, Wn, labels_to_H(consensus, k)))

    rng = np.random.default_rng(1234 + seed)
    random_labels = np.empty((n_restarts, n), dtype=np.int64)
    for r in range(n_restarts):
        _, sizes = np.unique(restart_labels[r], return_counts=True)
        random_labels[r] = size_matched_random(sizes, n, rng)

    cv = restart_gamma.std(ddof=1) / restart_gamma.mean()
    print(f"{model}  k={k}  n={n}  gamma={restart_gamma.mean():.2f} (cv={cv:.3f})",
          flush=True)
    return {"model": model, "seed": seed, "k": k, "variant": variant, "ind_active": ind,
            "restart_labels": restart_labels, "restart_gamma": restart_gamma,
            "consensus_labels": consensus, "consensus_gamma": consensus_gamma,
            "random_labels": random_labels}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-restarts", type=int, default=20)
    p.add_argument("--matrix-variant", default="transpose")
    p.add_argument("--sizes", type=int, nargs="*", default=[128, 256, 512, 1024])
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    todo = [r for r in csv.DictReader(open(LUMP)) if int(r["size"]) in args.sizes]
    for i, r in enumerate(todo, 1):
        dest = OUT / f"{r['model']}_ensemble.npz"
        if dest.exists():
            print(f"[{i}/{len(todo)}] cached {r['model']}", flush=True)
            continue
        rec = build(r["model"], int(r["k"]), int(r["seed"]),
                    args.n_restarts, args.matrix_variant)
        if rec is None:
            continue
        print(f"[{i}/{len(todo)}] done {r['model']}", flush=True)
        np.savez(dest, **rec)
    print("all done")


if __name__ == "__main__":
    main()
