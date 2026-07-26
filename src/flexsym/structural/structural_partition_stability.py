"""Partition stability diagnostics for the signed-WL structural partition.

Addresses reviewer concern #5 (method fragility). For a given model and
operating k, this script quantifies whether the signed-WL partition is
*reproducible* before anything downstream is trusted:

  1. Restart stability. The production ``approx_wl_signed`` runs from a single
     random start (one ``seed`` drives both the initial labels and the KMeans
     ``random_state``). Here we re-run the refinement from many independent
     restarts and report the full pairwise ARI/NMI distribution across restart
     partitions, plus a co-association consensus and each restart's agreement
     with it.
  2. Cell-size / degree artefact. The signed cost is known to be degree
     sensitive (Scholkemper & Schaub): it can peel high-strength units into
     singletons and dump the rest into one giant cell. We report the cell-size
     distribution (max-cell fraction, singleton count/mass, normalised entropy,
     Gini) and test whether singleton membership correlates with node strength.
  3. Soft back-end. We repeat (1) with a fuzzy/GMM refinement to test whether
     hard k-means assignment is the source of any instability (the reviewer's
     proposed fix for the 1024 reversal and the k wobble).

The script re-uses the *exact* production pipeline (``load_w_rec`` ->
``apply_matrix_variant`` -> active-unit restriction -> positive/negative split)
so the diagnostic partitions are the same objects the rest of the analysis
consumes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

STEP_DIR = Path(__file__).resolve().parent
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from signed_input_wl_structural_roles import (  # noqa: E402
    _canonicalize_labels,
    apply_matrix_variant,
    approx_wl_signed,
    checked_matmul,
    gamma_signed,
    labels_to_H,
    load_w_rec,
)


# --------------------------------------------------------------------------- #
# Soft / fuzzy WL refinement (GMM back-end)
# --------------------------------------------------------------------------- #
def soft_wl_signed(
    W_pos: np.ndarray,
    W_neg: np.ndarray,
    k: int,
    *,
    n_iter: int = 20,
    seed: int = 0,
    n_init: int = 5,
    reg_covar: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuzzy analogue of ``approx_wl_signed`` using soft responsibilities.

    Identical WL feature construction (``[W_pos @ R, W_neg @ R]``) but the cell
    membership ``R`` is a soft responsibility matrix from a Gaussian mixture and
    is fed back softly on each iteration. Returns the hardened labels (argmax)
    and the final responsibility matrix.
    """
    n = W_pos.shape[0]
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    R = labels_to_H(labels, k).astype(float)

    prev_hard: np.ndarray | None = None
    for _ in range(n_iter):
        X = np.concatenate(
            [
                checked_matmul(W_pos, R, "W_pos @ R"),
                checked_matmul(W_neg, R, "W_neg @ R"),
            ],
            axis=1,
        )
        gm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            n_init=n_init,
            reg_covar=reg_covar,
            random_state=seed,
        )
        gm.fit(X)
        R = gm.predict_proba(X)
        hard = _canonicalize_labels(gm.predict(X))
        if prev_hard is not None and np.array_equal(hard, prev_hard):
            break
        prev_hard = hard

    return hard, R


# --------------------------------------------------------------------------- #
# Restart stability
# --------------------------------------------------------------------------- #
def _pairwise_agreement(partitions: list[np.ndarray]) -> dict[str, Any]:
    r = len(partitions)
    ari_vals: list[float] = []
    nmi_vals: list[float] = []
    ari_mat = np.eye(r)
    for i in range(r):
        for j in range(i + 1, r):
            ari = float(adjusted_rand_score(partitions[i], partitions[j]))
            nmi = float(normalized_mutual_info_score(partitions[i], partitions[j]))
            ari_vals.append(ari)
            nmi_vals.append(nmi)
            ari_mat[i, j] = ari_mat[j, i] = ari
    ari_arr = np.asarray(ari_vals, dtype=float)
    nmi_arr = np.asarray(nmi_vals, dtype=float)
    return {
        "n_pairs": len(ari_vals),
        "ari_mean": float(ari_arr.mean()) if ari_arr.size else float("nan"),
        "ari_median": float(np.median(ari_arr)) if ari_arr.size else float("nan"),
        "ari_min": float(ari_arr.min()) if ari_arr.size else float("nan"),
        "ari_max": float(ari_arr.max()) if ari_arr.size else float("nan"),
        "ari_std": float(ari_arr.std(ddof=1)) if ari_arr.size > 1 else 0.0,
        "nmi_mean": float(nmi_arr.mean()) if nmi_arr.size else float("nan"),
        "nmi_median": float(np.median(nmi_arr)) if nmi_arr.size else float("nan"),
        "_ari_matrix": ari_mat,
    }


def _consensus_partition(partitions: list[np.ndarray], k: int) -> np.ndarray:
    """Co-association consensus: average co-clustering frequency, then cluster."""
    n = partitions[0].shape[0]
    coassoc = np.zeros((n, n), dtype=float)
    for labels in partitions:
        same = labels[:, None] == labels[None, :]
        coassoc += same.astype(float)
    coassoc /= len(partitions)
    distance = 1.0 - coassoc
    np.fill_diagonal(distance, 0.0)
    model = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="average"
    )
    return _canonicalize_labels(model.fit_predict(distance))


def restart_stability(
    W_pos: np.ndarray,
    W_neg: np.ndarray,
    k: int,
    *,
    backend: str,
    n_restarts: int,
    n_iter: int,
    n_init: int,
    base_seed: int = 0,
) -> dict[str, Any]:
    partitions: list[np.ndarray] = []
    gammas: list[float] = []
    n_effective: list[int] = []
    for r in range(n_restarts):
        seed = base_seed + r
        if backend == "hard":
            labels, H = approx_wl_signed(
                W_pos, W_neg, k, n_iter=n_iter, seed=seed, n_init=n_init
            )
        elif backend == "soft":
            labels, _ = soft_wl_signed(
                W_pos, W_neg, k, n_iter=n_iter, seed=seed, n_init=n_init
            )
            H = labels_to_H(labels, k)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        partitions.append(labels)
        gammas.append(float(gamma_signed(W_pos, W_neg, H)))
        n_effective.append(int(len(np.unique(labels))))

    agreement = _pairwise_agreement(partitions)
    ari_mat = agreement.pop("_ari_matrix")
    consensus = _consensus_partition(partitions, k)
    ari_to_consensus = [
        float(adjusted_rand_score(consensus, labels)) for labels in partitions
    ]
    # medoid restart = highest mean ARI to the other restarts
    mean_ari_to_others = (ari_mat.sum(axis=1) - 1.0) / max(len(partitions) - 1, 1)
    medoid_idx = int(np.argmax(mean_ari_to_others))

    gammas_arr = np.asarray(gammas, dtype=float)
    return {
        "backend": backend,
        "k": int(k),
        "n_restarts": int(n_restarts),
        **agreement,
        "ari_to_consensus_mean": float(np.mean(ari_to_consensus)),
        "ari_to_consensus_min": float(np.min(ari_to_consensus)),
        "gamma_mean": float(gammas_arr.mean()),
        "gamma_std": float(gammas_arr.std(ddof=1)) if gammas_arr.size > 1 else 0.0,
        "gamma_min": float(gammas_arr.min()),
        "gamma_cv": float(gammas_arr.std(ddof=1) / gammas_arr.mean())
        if gammas_arr.size > 1 and gammas_arr.mean() != 0
        else 0.0,
        "n_effective_clusters_mean": float(np.mean(n_effective)),
        "n_effective_clusters_min": int(np.min(n_effective)),
        "_partitions": partitions,
        "_consensus": consensus,
        "_medoid_idx": medoid_idx,
    }


# --------------------------------------------------------------------------- #
# Cell-size / degree artefact
# --------------------------------------------------------------------------- #
def cell_size_stats(labels: np.ndarray, k: int, strength: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(labels, minlength=k).astype(float)
    nonempty = counts[counts > 0]
    n = int(counts.sum())
    frac = nonempty / n
    entropy = float(-(frac * np.log(frac)).sum())
    entropy_norm = float(entropy / np.log(len(nonempty))) if len(nonempty) > 1 else 0.0

    sorted_counts = np.sort(nonempty)
    cum = np.cumsum(sorted_counts)
    gini = float(
        (2.0 * np.sum((np.arange(1, len(sorted_counts) + 1)) * sorted_counts))
        / (len(sorted_counts) * cum[-1])
        - (len(sorted_counts) + 1) / len(sorted_counts)
    )

    is_singleton_cell = counts == 1
    singleton_unit_mask = is_singleton_cell[labels]
    n_singletons = int(is_singleton_cell.sum())
    frac_units_in_singletons = float(singleton_unit_mask.mean())

    # degree artefact: do singleton units carry more weight strength?
    singleton_strength_ratio = float("nan")
    strength_spearman = float("nan")
    strength_spearman_p = float("nan")
    if 0 < n_singletons < len(labels):
        mean_singleton = float(strength[singleton_unit_mask].mean())
        mean_other = float(strength[~singleton_unit_mask].mean())
        if mean_other > 0:
            singleton_strength_ratio = mean_singleton / mean_other
        rho, p = spearmanr(strength, singleton_unit_mask.astype(float))
        strength_spearman = float(rho)
        strength_spearman_p = float(p)

    return {
        "n_nonempty_cells": int(len(nonempty)),
        "max_cell_fraction": float(nonempty.max() / n),
        "min_cell_size": int(nonempty.min()),
        "median_cell_size": float(np.median(nonempty)),
        "n_singleton_cells": n_singletons,
        "frac_units_in_singletons": frac_units_in_singletons,
        "size_entropy_norm": entropy_norm,
        "size_gini": gini,
        "singleton_strength_ratio": singleton_strength_ratio,
        "singleton_strength_spearman": strength_spearman,
        "singleton_strength_spearman_p": strength_spearman_p,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def analyze_model(
    model_dir: Path,
    activity_clusters: Path,
    *,
    k_list: list[int],
    matrix_variant: str,
    backends: list[str],
    n_restarts: int,
    n_iter: int,
    n_init: int,
    soft_restarts: int,
    soft_n_init: int,
    output_dir: Path,
) -> dict[str, Any]:
    activity = np.load(activity_clusters, allow_pickle=True)
    ind_active = np.asarray(activity["ind_active"], dtype=np.int64)

    W_rec_full = apply_matrix_variant(load_w_rec(model_dir), matrix_variant)
    W_rec = W_rec_full[np.ix_(ind_active, ind_active)]
    W_pos = np.maximum(W_rec, 0.0)
    W_neg = np.maximum(-W_rec, 0.0)
    # node strength: total incoming + outgoing magnitude on the active submatrix
    strength = np.abs(W_rec).sum(axis=1) + np.abs(W_rec).sum(axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    cross_backend: dict[int, dict[str, np.ndarray]] = {}

    for k in k_list:
        cross_backend.setdefault(k, {})
        for backend in backends:
            n_r = soft_restarts if backend == "soft" else n_restarts
            n_i = soft_n_init if backend == "soft" else n_init
            stab = restart_stability(
                W_pos,
                W_neg,
                k,
                backend=backend,
                n_restarts=n_r,
                n_iter=n_iter,
                n_init=n_i,
            )
            consensus = stab.pop("_consensus")
            partitions = stab.pop("_partitions")
            medoid_idx = stab.pop("_medoid_idx")
            cross_backend[k][backend] = consensus

            sizes = cell_size_stats(consensus, k, strength)
            row = {
                "model": model_dir.name,
                "matrix_variant": matrix_variant,
                **stab,
                **sizes,
            }
            rows.append(row)

    # hard-vs-soft agreement per k (do the back-ends find the same partition?)
    for k in k_list:
        if "hard" in cross_backend[k] and "soft" in cross_backend[k]:
            ari = float(
                adjusted_rand_score(cross_backend[k]["hard"], cross_backend[k]["soft"])
            )
            nmi = float(
                normalized_mutual_info_score(
                    cross_backend[k]["hard"], cross_backend[k]["soft"]
                )
            )
            for row in rows:
                if row["k"] == k:
                    row["hard_vs_soft_ari"] = ari
                    row["hard_vs_soft_nmi"] = nmi

    fieldnames = sorted({key for row in rows for key in row})
    csv_path = output_dir / "partition_stability.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary = {
        "model_dir": str(model_dir),
        "model": model_dir.name,
        "activity_clusters": str(activity_clusters),
        "matrix_variant": matrix_variant,
        "n_units_active": int(len(ind_active)),
        "k_list": k_list,
        "backends": backends,
        "n_restarts_hard": n_restarts,
        "n_restarts_soft": soft_restarts,
        "csv_path": str(csv_path),
        "rows": rows,
    }
    with (output_dir / "partition_stability_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def _resolve_k_list(args: argparse.Namespace, activity_clusters: Path) -> list[int]:
    if args.k:
        return sorted(set(args.k))
    activity = np.load(activity_clusters, allow_pickle=True)
    best_k = int(activity["best_k"])
    return [best_k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("--activity-clusters", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--k", type=int, action="append", help="Operating k (repeatable)."
    )
    parser.add_argument("--matrix-variant", default="transpose")
    parser.add_argument(
        "--backend",
        action="append",
        choices=["hard", "soft"],
        help="Refinement back-end(s) (repeatable). Default: hard + soft.",
    )
    parser.add_argument("--n-restarts", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--soft-restarts", type=int, default=10)
    parser.add_argument("--soft-n-init", type=int, default=5)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    activity_clusters = Path(args.activity_clusters)
    backends = args.backend or ["hard", "soft"]
    k_list = _resolve_k_list(args, activity_clusters)

    summary = analyze_model(
        model_dir,
        activity_clusters,
        k_list=k_list,
        matrix_variant=args.matrix_variant,
        backends=backends,
        n_restarts=args.n_restarts,
        n_iter=args.n_iter,
        n_init=args.n_init,
        soft_restarts=args.soft_restarts,
        soft_n_init=args.soft_n_init,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps({"model": summary["model"], "k_list": summary["k_list"]}))
    for row in summary["rows"]:
        print(
            f"  k={row['k']:>3} {row['backend']:>4} "
            f"ari_mean={row['ari_mean']:.3f} ari_min={row['ari_min']:.3f} "
            f"max_cell={row['max_cell_fraction']:.2f} "
            f"singletons={row['n_singleton_cells']} "
            f"str_ratio={row['singleton_strength_ratio']:.2f}"
        )


if __name__ == "__main__":
    main()
