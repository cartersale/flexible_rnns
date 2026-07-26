"""Cluster recurrent units by variance across task epochs.

This script consumes the ``variance_epoch.pkl`` files produced by
``analysis/variance.py`` and saves a compact clustering artifact. If requested,
it can compute ``variance_epoch.pkl`` first.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score




def _load_variance_epoch(model_dir: Path, variance_path: str | os.PathLike[str] | None = None) -> dict:
    variance_path = Path(variance_path) if variance_path is not None else model_dir / "variance_epoch.pkl"
    if not variance_path.exists():
        raise FileNotFoundError(
            f"Missing {variance_path}. Re-run with --compute-variance or call "
            "analysis.variance.compute_variance(model_dir) first."
        )
    with variance_path.open("rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def _normalize(unit_features: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return unit_features

    if method == "sum":
        denom = unit_features.sum(axis=1)
    elif method == "max":
        denom = unit_features.max(axis=1)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    denom = np.where(denom == 0, 1.0, denom)
    return (unit_features.T / denom).T


def cluster_epoch_variance(
    model_dir: str | os.PathLike[str],
    *,
    k_min: int = 3,
    k_max: int = 40,
    active_threshold: float = 1e-3,
    normalization: str = "sum",
    linkage_method: str = "ward",
    output_name: str = "epoch_variance_unit_clusters.npz",
    output_dir: str | os.PathLike[str] | None = None,
    variance_path: str | os.PathLike[str] | None = None,
) -> dict:
    """Cluster units using their task-epoch variance profiles.

    The variance file stores ``h_var_all`` as units x task_epochs. This function
    also saves ``V`` as task_epochs x units to match the notation in analysis
    notes.
    """

    model_dir = Path(model_dir)
    result = _load_variance_epoch(model_dir, variance_path)

    h_var_all = np.asarray(result["h_var_all"])
    keys = np.asarray(result["keys"], dtype=object)
    V = h_var_all.T

    unit_features = V.T
    if not np.isfinite(unit_features).all():
        n_bad = int(np.size(unit_features) - np.isfinite(unit_features).sum())
        raise ValueError(f"Found {n_bad} non-finite values in variance features.")

    ind_active = np.where(unit_features.sum(axis=1) > active_threshold)[0]
    unit_features_active = unit_features[ind_active]
    unit_features_norm = _normalize(unit_features_active, normalization)
    if not np.isfinite(unit_features_norm).all():
        n_bad = int(np.size(unit_features_norm) - np.isfinite(unit_features_norm).sum())
        raise ValueError(f"Found {n_bad} non-finite values after normalization.")

    if unit_features_norm.shape[0] < k_min:
        raise ValueError(
            f"Only {unit_features_norm.shape[0]} active units; cannot scan "
            f"k_min={k_min}."
        )

    Z = linkage(unit_features_norm, method=linkage_method)

    scores = {}
    best_k = None
    best_score = -np.inf
    best_labels = None
    max_valid_k = min(k_max, unit_features_norm.shape[0] - 1)

    for k in range(k_min, max_valid_k + 1):
        labels = fcluster(Z, k, criterion="maxclust")
        n_labels = len(np.unique(labels))
        if n_labels < 2 or n_labels >= len(labels):
            continue

        score = silhouette_score(unit_features_norm, labels)
        scores[k] = float(score)
        if score > best_score:
            best_score = float(score)
            best_k = int(k)
            best_labels = labels

    if best_labels is None:
        raise RuntimeError("Could not find a valid clustering in the requested k range.")

    labels_all_units = np.zeros(unit_features.shape[0], dtype=np.int64)
    labels_all_units[ind_active] = best_labels

    output_dir = Path(output_dir) if output_dir is not None else model_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    np.savez(
        output_path,
        V=V,
        unit_features=unit_features,
        unit_features_active=unit_features_active,
        unit_features_norm=unit_features_norm,
        ind_active=ind_active,
        Z=Z,
        labels_active=best_labels,
        labels_all_units=labels_all_units,
        keys=keys,
        best_k=best_k,
        best_score=best_score,
        silhouette_scores=np.asarray(sorted(scores.items()), dtype=float),
        active_threshold=active_threshold,
        normalization=normalization,
        linkage_method=linkage_method,
    )

    summary = {
        "model_dir": str(model_dir),
        "output_path": str(output_path),
        "variance_epoch_path": str(Path(variance_path) if variance_path is not None else model_dir / "variance_epoch.pkl"),
        "n_task_epochs": int(V.shape[0]),
        "n_units": int(V.shape[1]),
        "n_active_units": int(len(ind_active)),
        "active_threshold": active_threshold,
        "normalization": normalization,
        "linkage_method": linkage_method,
        "k_min": k_min,
        "k_max": k_max,
        "best_k": best_k,
        "best_score": best_score,
        "epoch_keys": [list(key) for key in result["keys"]],
        "silhouette_scores": scores,
    }

    with (output_dir / output_name.replace(".npz", ".json")).open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster units by variance profiles across task epochs."
    )
    parser.add_argument("model_dir", help="Model directory containing hp.json/checkpoint.")
    parser.add_argument(
        "--compute-variance",
        action="store_true",
        help="Compute variance_epoch.pkl before clustering.",
    )
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=40)
    parser.add_argument("--active-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--normalization",
        choices=("sum", "max", "none"),
        default="sum",
        help="Per-unit normalization applied before clustering.",
    )
    parser.add_argument("--linkage-method", default="ward")
    parser.add_argument("--output-name", default="epoch_variance_unit_clusters.npz")
    parser.add_argument(
        "--variance-path",
        default=None,
        help="Optional path to variance_epoch.pkl. Defaults to model_dir/variance_epoch.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for clustering outputs. Defaults to model_dir.",
    )
    args = parser.parse_args()

    if args.compute_variance:
        from flexsym.core import variance

        variance.compute_variance(args.model_dir)

    summary = cluster_epoch_variance(
        args.model_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        active_threshold=args.active_threshold,
        normalization=args.normalization,
        linkage_method=args.linkage_method,
        output_name=args.output_name,
        output_dir=args.output_dir,
        variance_path=args.variance_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
