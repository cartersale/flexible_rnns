"""ARI between the production signed-WL partition and a node-strength quantile
partition, for the 256-unit LeakyRNN softplus-diag networks.

Rationale: a common trivial explanation for the WL cells is that they merely
re-express node strength (the signed cost is degree sensitive). If the WL
partition is basically "strength bins", then binning nodes into k equal-count
quantiles of |W| strength should recover it (high ARI). We compute, per net:

  1. WL partition at operating k (production ``approx_wl_signed``, seed 0).
  2. Strength quantile partition: rank the active units by node strength
     (|W_rec| row-sum + col-sum), cut into k equal-count bins.
  3. ARI(WL, strength-bins).

For a fair baseline we also report ARI(WL, random k-partition) averaged over
restarts (same class-size profile as the quantile bins by construction).
"""

from __future__ import annotations

from flexsym import paths

import csv
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


from flexsym.structural.signed_input_wl_structural_roles import (  # noqa: E402
    apply_matrix_variant,
    approx_wl_signed,
    load_w_rec,
)

REPO_DIR = paths.REPO_ROOT
RESULTS_ROOT = paths.RESULTS_ROOT
ACTIVITY_ROOT = RESULTS_ROOT / "activity_clusters_paper"
ACTIVITY_NAME = "epoch_variance_unit_clusters_paper_max.npz"
ORIGINAL_STORE = paths.ORIGINAL_STORE
STAGED_STORE = paths.STAGED_STORE

# operating k per 256 net (active_fraction selected_k from batch k-selection summaries)
OPERATING_K = {
    "LeakyRNN_softplus_diag_256_all_0": 33,
    "LeakyRNN_softplus_diag_256_all_1": 51,
    "LeakyRNN_softplus_diag_256_all_2": 38,
    "LeakyRNN_softplus_diag_256_all_3": 14,
    "LeakyRNN_softplus_diag_256_all_4": 14,
    "LeakyRNN_softplus_diag_256_all_5": 14,
    "LeakyRNN_softplus_diag_256_all_6": 14,
    "LeakyRNN_softplus_diag_256_all_7": 14,
    "LeakyRNN_softplus_diag_256_all_8": 14,
    "LeakyRNN_softplus_diag_256_all_9": 15,
}


def resolve_model_dir(model: str) -> Path | None:
    for store in (ORIGINAL_STORE, STAGED_STORE):
        candidate = store / model
        if candidate.exists():
            return candidate
    return None


def strength_quantile_labels(strength: np.ndarray, k: int) -> np.ndarray:
    """k equal-count bins of node strength (ties broken by stable rank order)."""
    order = np.argsort(strength, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(strength))
    # np.array_split gives near-equal bin sizes; map each rank to its bin index
    edges = np.linspace(0, len(strength), k + 1)
    labels = np.searchsorted(edges[1:-1], ranks, side="right")
    return labels.astype(np.int64)


def random_partition_ari(
    wl_labels: np.ndarray, sizes: np.ndarray, k: int, n_rep: int = 200
) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    n = len(wl_labels)
    base = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    aris = np.empty(n_rep)
    for r in range(n_rep):
        perm = rng.permutation(n)
        aris[r] = adjusted_rand_score(wl_labels, base[perm])
    return float(aris.mean()), float(aris.std(ddof=1))


def main() -> None:
    rows = []
    for model, k in OPERATING_K.items():
        model_dir = resolve_model_dir(model)
        activity_path = ACTIVITY_ROOT / model / ACTIVITY_NAME
        if model_dir is None or not activity_path.exists():
            print(f"SKIP {model}: dir={model_dir} activity={activity_path.exists()}")
            continue

        activity = np.load(activity_path, allow_pickle=True)
        ind_active = np.asarray(activity["ind_active"], dtype=np.int64)

        W_rec = apply_matrix_variant(load_w_rec(model_dir), "transpose")
        W_rec = W_rec[np.ix_(ind_active, ind_active)]
        W_pos = np.maximum(W_rec, 0.0)
        W_neg = np.maximum(-W_rec, 0.0)
        strength = np.abs(W_rec).sum(axis=1) + np.abs(W_rec).sum(axis=0)

        wl_labels, _ = approx_wl_signed(W_pos, W_neg, k, n_iter=20, seed=0, n_init=20)
        q_labels = strength_quantile_labels(strength, k)

        ari = float(adjusted_rand_score(wl_labels, q_labels))
        nmi = float(normalized_mutual_info_score(wl_labels, q_labels))
        # null: random partition with the same (equal-count) size profile as quantiles
        _, q_sizes = np.unique(q_labels, return_counts=True)
        rand_ari_mean, rand_ari_sd = random_partition_ari(wl_labels, q_sizes, k)
        z = (ari - rand_ari_mean) / rand_ari_sd if rand_ari_sd > 0 else float("nan")

        n_eff = int(len(np.unique(wl_labels)))
        rows.append(
            {
                "model": model,
                "k": k,
                "n_active": int(len(ind_active)),
                "n_eff_wl_cells": n_eff,
                "ari_wl_vs_strength": ari,
                "nmi_wl_vs_strength": nmi,
                "rand_ari_mean": rand_ari_mean,
                "rand_ari_sd": rand_ari_sd,
                "ari_z_vs_random": z,
            }
        )
        print(
            f"{model}  k={k:>3}  ARI={ari:+.3f}  NMI={nmi:.3f}  "
            f"rand_ARI={rand_ari_mean:+.4f}+/-{rand_ari_sd:.4f}  z={z:+.2f}"
        )

    if not rows:
        raise SystemExit("No models analysed (external drive mounted?).")

    aris = np.array([r["ari_wl_vs_strength"] for r in rows])
    print(
        f"\n256-unit nets (n={len(rows)}): "
        f"ARI(WL, strength-quantiles) mean={aris.mean():.3f} "
        f"median={np.median(aris):.3f} range=[{aris.min():.3f}, {aris.max():.3f}]"
    )

    out_dir = RESULTS_ROOT / "signed_input_wl_paper" / "wl_vs_strength"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "wl_vs_strength_ari_256.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
