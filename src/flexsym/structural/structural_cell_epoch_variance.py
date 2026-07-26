"""Average Driscoll-style epoch variance within structural cells."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STEP_DIR = Path(__file__).resolve().parent
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))


def _labels_to_cell_matrix(labels: np.ndarray, k: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    H = np.zeros((labels.size, k), dtype=float)
    H[np.arange(labels.size), labels] = 1.0
    return H


def _cell_means(features: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((k, features.shape[1]), dtype=float)
    sizes = np.zeros(k, dtype=np.int64)
    for cell in range(k):
        mask = labels == cell
        sizes[cell] = int(mask.sum())
        if sizes[cell] > 0:
            matrix[cell] = features[mask].mean(axis=0)
    return matrix, sizes


def _load_struct_labels(sweep_path: Path, k: int) -> np.ndarray:
    sweep = np.load(sweep_path, allow_pickle=True)
    rows = np.asarray(sweep["rows"], dtype=object)
    row_columns = [str(x) for x in np.asarray(sweep["row_columns"], dtype=object)]
    k_idx = row_columns.index("k")
    k_min = int(rows[0, k_idx])
    k_max = int(rows[-1, k_idx])
    if k < k_min or k > k_max:
        raise ValueError(f"k={k} is outside structural sweep range [{k_min}, {k_max}]")
    return np.asarray(sweep["labels_struct_by_k"][k - k_min], dtype=np.int64)


def _key_strings(keys: np.ndarray) -> list[str]:
    out = []
    for key in keys:
        if np.ndim(key) == 0:
            out.append(str(key))
        else:
            out.append("/".join(str(x) for x in key))
    return out


def _plot_matrix(matrix: np.ndarray, keys: list[str], output_dir: Path, k: int) -> None:
    fig_width = max(9.0, 0.22 * len(keys))
    fig_height = max(5.0, 0.16 * matrix.shape[0])
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("task period")
    ax.set_ylabel("structural cell")
    ax.set_title(f"Structural-cell normalized epoch variance, k={k}")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=90, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="mean normalized variance")
    fig.tight_layout()
    fig.savefig(output_dir / f"structural_cell_epoch_variance_k_{k}.png", dpi=200)
    fig.savefig(output_dir / f"structural_cell_epoch_variance_k_{k}.pdf")
    plt.close(fig)


def compute_structural_cell_epoch_variance(
    activity_clusters: str | os.PathLike[str],
    structural_sweep: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    k: int | None = None,
) -> dict:
    activity_path = Path(activity_clusters)
    sweep_path = Path(structural_sweep)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    activity = np.load(activity_path, allow_pickle=True)
    sweep = np.load(sweep_path, allow_pickle=True)

    if k is None:
        k = int(np.asarray(sweep["chosen_k"]).item())

    unit_features_norm = np.asarray(activity["unit_features_norm"], dtype=float)
    keys = np.asarray(activity["keys"], dtype=object)
    key_names = _key_strings(keys)
    labels = _load_struct_labels(sweep_path, k)

    if labels.size != unit_features_norm.shape[0]:
        raise ValueError(
            "Structural labels and active-unit features disagree: "
            f"{labels.size} labels vs {unit_features_norm.shape[0]} feature rows."
        )

    cell_epoch_variance, cell_sizes = _cell_means(unit_features_norm, labels, k)

    npz_path = output_dir / f"structural_cell_epoch_variance_k_{k}.npz"
    np.savez(
        npz_path,
        cell_epoch_variance=cell_epoch_variance,
        cell_sizes=cell_sizes.astype(np.int64),
        labels_struct=labels,
        keys=keys,
        key_names=np.asarray(key_names, dtype=object),
        k=k,
        activity_clusters_path=str(activity_path),
        structural_sweep_path=str(sweep_path),
        matrix_variant=str(np.asarray(sweep["matrix_variant"]).item()),
    )

    csv_path = output_dir / f"structural_cell_epoch_variance_k_{k}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["struct_cell", "cell_size", *key_names])
        for cell, row in enumerate(cell_epoch_variance):
            writer.writerow([cell, int(cell_sizes[cell]), *row.tolist()])

    keys_csv_path = output_dir / f"structural_cell_epoch_variance_keys_k_{k}.csv"
    with keys_csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["period_index", "task", "epoch", "column"])
        for i, key in enumerate(keys):
            if np.ndim(key) == 0:
                task, epoch = "", str(key)
            else:
                task = str(key[0]) if len(key) > 0 else ""
                epoch = str(key[1]) if len(key) > 1 else ""
            writer.writerow([i, task, epoch, key_names[i]])

    _plot_matrix(cell_epoch_variance, key_names, output_dir, k)

    summary = {
        "activity_clusters": str(activity_path),
        "structural_sweep": str(sweep_path),
        "output_dir": str(output_dir),
        "k": k,
        "matrix_variant": str(np.asarray(sweep["matrix_variant"]).item()),
        "n_cells": int(k),
        "n_task_periods": int(cell_epoch_variance.shape[1]),
        "n_active_units": int(unit_features_norm.shape[0]),
        "cell_size_min": int(cell_sizes.min()),
        "cell_size_max": int(cell_sizes.max()),
        "cell_size_mean": float(cell_sizes.mean()),
        "npz_path": str(npz_path),
        "csv_path": str(csv_path),
        "keys_csv_path": str(keys_csv_path),
    }
    with (output_dir / f"structural_cell_epoch_variance_summary_k_{k}.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity-clusters", required=True)
    parser.add_argument("--structural-sweep", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()
    summary = compute_structural_cell_epoch_variance(
        args.activity_clusters,
        args.structural_sweep,
        args.output_dir,
        k=args.k,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
