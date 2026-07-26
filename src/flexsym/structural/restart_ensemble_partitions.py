"""Generate and persist the restart-partition ensemble for the 256-unit nets.

Reviewer concern #5: the signed-WL partition is non-unique (restart ARI ~0.4).
Rather than collapse to a single/consensus partition, we persist the full
ensemble so every downstream analysis can be replicated across it and we can ask
the identifiability question directly (between-network vs between-restart
variance).

Per 256-unit network, at its operating k, we persist:
  * ``restart``   : R=20 hard signed-WL partitions from independent random inits
                    (seeds 0..R-1), each with its Gamma_signed cost.
  * ``consensus`` : co-association consensus over the R restarts.
  * ``random``    : R size-matched random partitions -- partition r matches the
                    cell-size profile of restart r, giving a chance reference
                    whose size variation mirrors the restart ensemble.

Everything is over the active-unit submatrix (transpose variant), the exact
objects the lesion / reorg / connectivity analyses consume.
"""

from __future__ import annotations

from flexsym import paths

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np


from flexsym.structural.signed_input_wl_structural_roles import (  # noqa: E402
    apply_matrix_variant,
    approx_wl_signed,
    gamma_signed,
    labels_to_H,
    load_w_rec,
)
from flexsym.structural.structural_partition_stability import _consensus_partition  # noqa: E402

REPO = paths.REPO_ROOT
ACTIVITY_ROOT = REPO / "results" / "activity_clusters_paper"
ACTIVITY_NAME = "epoch_variance_unit_clusters_paper_max.npz"
ORIG = paths.ORIGINAL_STORE
STAGED = paths.STAGED_STORE
OUT = REPO / "results" / "signed_input_wl_paper" / "restart_ensemble"

# Operating k, as used by the downstream lesion / reorg / connectivity scripts.
#
# This was previously a hardcoded table. That silently went stale when seeds 3-9
# were retrained on 2026-07-25: the table still said k=14 (from checkpoints where
# only ~70 of 256 units were active) while the current nets want k=42-44. Deriving
# it from the activity file instead means it cannot drift again.
#
# Rule (matches configs/batch_rnn_softplus_diag*.json, policy "active_fraction"):
#     k = clamp(min(activity_best_k, round(0.20 * n_active)), K_MIN, n_active - 1)
# It reproduces the released seeds 0-2 values exactly (41, 51, 38).
K_MIN = 3
ACTIVE_FRACTION = 0.20


def operating_k(model: str) -> int | None:
    """Derive the operating k for a model from its activity clustering."""
    path = ACTIVITY_ROOT / model / ACTIVITY_NAME
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    n_active = int(np.asarray(z["ind_active"], dtype=np.int64).size)
    structural_k = int(round(ACTIVE_FRACTION * n_active))
    structural_k = max(K_MIN, min(structural_k, n_active - 1))
    k = min(int(z["best_k"]), structural_k)
    return max(K_MIN, min(k, n_active - 1))


def resolve(model: str) -> Path | None:
    for store in (ORIG, STAGED):
        if (store / model).exists():
            return store / model
    return None


def size_matched_random(sizes: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random partition of ``n`` units with the given (exact) cell-size profile."""
    assert int(sizes.sum()) == n, (int(sizes.sum()), n)
    base = np.concatenate([np.full(int(s), i, dtype=np.int64) for i, s in enumerate(sizes)])
    return base[rng.permutation(n)]


def build_net(seed: int, k: int, *, n_restarts: int, variant: str) -> dict | None:
    model = f"LeakyRNN_softplus_diag_256_all_{seed}"
    md = resolve(model)
    activity_path = ACTIVITY_ROOT / model / ACTIVITY_NAME
    if md is None or not activity_path.exists():
        print(f"SKIP {model}: dir={md} activity={activity_path.exists()}")
        return None

    ind_active = np.asarray(
        np.load(activity_path, allow_pickle=True)["ind_active"], dtype=np.int64
    )
    W_rec = apply_matrix_variant(load_w_rec(md), variant)[np.ix_(ind_active, ind_active)]
    W_pos = np.maximum(W_rec, 0.0)
    W_neg = np.maximum(-W_rec, 0.0)
    n = len(ind_active)

    restart_labels = np.empty((n_restarts, n), dtype=np.int64)
    restart_gamma = np.empty(n_restarts, dtype=float)
    for r in range(n_restarts):
        labels, H = approx_wl_signed(W_pos, W_neg, k, n_iter=20, seed=r, n_init=20)
        restart_labels[r] = labels
        restart_gamma[r] = float(gamma_signed(W_pos, W_neg, H))

    consensus = _consensus_partition([restart_labels[r] for r in range(n_restarts)], k)
    consensus_gamma = float(gamma_signed(W_pos, W_neg, labels_to_H(consensus, k)))

    # size-matched random ensemble: partition r mirrors restart r's size profile
    rng = np.random.default_rng(1234 + seed)
    random_labels = np.empty((n_restarts, n), dtype=np.int64)
    for r in range(n_restarts):
        _, sizes = np.unique(restart_labels[r], return_counts=True)
        random_labels[r] = size_matched_random(sizes, n, rng)

    print(
        f"{model}  k={k}  n_active={n}  "
        f"Gamma restart mean={restart_gamma.mean():.2f} "
        f"(cv={restart_gamma.std(ddof=1)/restart_gamma.mean():.3f})  "
        f"consensus={consensus_gamma:.2f}",
        flush=True,
    )
    return {
        "model": model,
        "seed": seed,
        "k": k,
        "variant": variant,
        "ind_active": ind_active,
        "restart_labels": restart_labels,
        "restart_gamma": restart_gamma,
        "consensus_labels": consensus,
        "consensus_gamma": consensus_gamma,
        "random_labels": random_labels,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-restarts", type=int, default=20)
    p.add_argument("--matrix-variant", default="transpose")
    p.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    p.add_argument("--k", type=int, default=None,
                   help="Override the derived operating k (applies to all seeds).")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        model = f"LeakyRNN_softplus_diag_256_all_{seed}"
        k = args.k if args.k is not None else operating_k(model)
        if k is None:
            print(f"SKIP {model}: no activity clustering at "
                  f"{ACTIVITY_ROOT / model / ACTIVITY_NAME}")
            continue
        rec = build_net(
            seed, k, n_restarts=args.n_restarts, variant=args.matrix_variant
        )
        if rec is None:
            continue
        np.savez(
            OUT / f"{rec['model']}_ensemble.npz",
            seed=rec["seed"],
            k=rec["k"],
            variant=rec["variant"],
            ind_active=rec["ind_active"],
            restart_labels=rec["restart_labels"],
            restart_gamma=rec["restart_gamma"],
            consensus_labels=rec["consensus_labels"],
            consensus_gamma=rec["consensus_gamma"],
            random_labels=rec["random_labels"],
        )
    print(f"wrote ensembles to {OUT}")


if __name__ == "__main__":
    main()
