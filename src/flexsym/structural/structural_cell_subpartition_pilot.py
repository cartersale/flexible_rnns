"""Pilot functional sub-partitions inside one structural cell."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


STEP_DIR = Path(__file__).resolve().parent
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from signed_input_wl_structural_roles import load_w_rec  # noqa: E402
from structural_cell_activity_coherence import compute_hidden_by_epoch, load_structural_labels  # noqa: E402


PERIODS = [
    ("MemoryPro memory", "delaygo", "delay1"),
    ("MemoryAnti memory", "delayanti", "delay1"),
    ("DelayDM1 stimulus", "delaydm1", "stim1"),
]


def canonical(labels: np.ndarray) -> np.ndarray:
    mapping = {}
    out = np.empty_like(labels, dtype=np.int64)
    nxt = 0
    for i, label in enumerate(labels):
        label = int(label)
        if label not in mapping:
            mapping[label] = nxt
            nxt += 1
        out[i] = mapping[label]
    return out


def zscore_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return X / np.maximum(sd, 1e-12)


def choose_functional_partition(X: np.ndarray, max_k: int = 6, min_silhouette: float = 0.0) -> dict:
    """Cluster unit trajectories; include k=1 as the no-split baseline."""
    Xz = zscore_rows(X)
    n = Xz.shape[0]
    if n < 3:
        return {
            "labels": np.zeros(n, dtype=np.int64),
            "chosen_k": 1,
            "best_silhouette": float("nan"),
            "silhouette_scores": {},
        }
    k_max = min(max_k, n - 1)
    scores = {}
    labels_by_k = {}
    for k in range(2, k_max + 1):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xz)
        labels = canonical(labels)
        if len(np.unique(labels)) < 2:
            continue
        score = float(silhouette_score(Xz, labels))
        scores[k] = score
        labels_by_k[k] = labels
    if not scores:
        chosen_k = 1
    else:
        chosen_k = max(scores, key=scores.get)
        if scores[chosen_k] <= min_silhouette:
            chosen_k = 1
    labels = np.zeros(n, dtype=np.int64) if chosen_k == 1 else labels_by_k[chosen_k]
    return {
        "labels": labels,
        "chosen_k": int(chosen_k),
        "best_silhouette": float(scores.get(chosen_k, 0.0 if scores else float("nan"))),
        "silhouette_scores": {str(k): float(v) for k, v in sorted(scores.items())},
    }


def internal_structural_partition(W_sub: np.ndarray, max_k: int = 6) -> dict:
    """Approximate internal signed equitable subpartition of one induced cell.

    Units are represented by signed input profiles from candidate internal
    subcells. k is chosen by the same silhouette rule used for dynamics.
    """
    n = W_sub.shape[0]
    if n < 3:
        return {
            "labels": np.zeros(n, dtype=np.int64),
            "chosen_k": 1,
            "best_silhouette": float("nan"),
            "silhouette_scores": {},
        }
    W_pos = np.maximum(W_sub, 0.0)
    W_neg = np.maximum(-W_sub, 0.0)
    k_max = min(max_k, n - 1)
    scores = {}
    labels_by_k = {}
    rng = np.random.default_rng(0)
    for k in range(2, k_max + 1):
        labels = rng.integers(0, k, size=n)
        for _ in range(30):
            H = np.zeros((n, k), dtype=float)
            H[np.arange(n), labels] = 1.0
            X = np.concatenate([W_pos @ H, W_neg @ H], axis=1)
            labels_new = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
            labels_new = canonical(labels_new)
            if np.array_equal(labels_new, labels):
                break
            labels = labels_new
        H = np.zeros((n, k), dtype=float)
        H[np.arange(n), labels] = 1.0
        X = np.concatenate([W_pos @ H, W_neg @ H], axis=1)
        score = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else float("nan")
        if np.isfinite(score):
            scores[k] = score
            labels_by_k[k] = labels
    if not scores:
        chosen_k = 1
    else:
        chosen_k = max(scores, key=scores.get)
        if scores[chosen_k] <= 0.0:
            chosen_k = 1
    labels = np.zeros(n, dtype=np.int64) if chosen_k == 1 else labels_by_k[chosen_k]
    return {
        "labels": labels,
        "chosen_k": int(chosen_k),
        "best_silhouette": float(scores.get(chosen_k, 0.0 if scores else float("nan"))),
        "silhouette_scores": {str(k): float(v) for k, v in sorted(scores.items())},
    }


def plot_partitions(unit_ids, functional, structural, output_dir: Path, stem: str) -> tuple[Path, Path]:
    rows = [functional[name]["labels"] for name in functional]
    row_names = list(functional.keys()) + ["internal wiring"]
    matrix = np.vstack(rows + [structural["labels"]])

    fig, ax = plt.subplots(figsize=(9, 2.8), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="tab20")
    ax.set_yticks(np.arange(len(row_names)))
    ax.set_yticklabels(row_names)
    ax.set_xticks(np.arange(len(unit_ids)))
    ax.set_xticklabels(unit_ids, rotation=90, fontsize=8)
    ax.set_xlabel("unit id")
    ax.set_title("Functional sub-partitions vs internal structural subpartition")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="subcluster")
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--structural-sweep", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, default=33)
    parser.add_argument("--cell", type=int, default=None)
    parser.add_argument("--cell-variance", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-subclusters", type=int, default=6)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels, ind_active, variant = load_structural_labels(Path(args.structural_sweep), args.k)
    cell_var = np.load(args.cell_variance, allow_pickle=True)
    cell_sizes = np.asarray(cell_var["cell_sizes"], dtype=np.int64)
    key_names = [str(x) for x in np.asarray(cell_var["key_names"], dtype=object)]

    if args.cell is None:
        mem_idx = [key_names.index("delaygo/delay1"), key_names.index("delayanti/delay1")]
        valid = np.where(cell_sizes >= 5)[0]
        scores = cell_var["cell_epoch_variance"][valid][:, mem_idx].mean(axis=1)
        cell = int(valid[np.argmax(scores)])
    else:
        cell = int(args.cell)

    unit_positions = np.where(labels == cell)[0]
    unit_ids = ind_active[unit_positions]
    if unit_positions.size < 3:
        raise ValueError(f"cell {cell} has only {unit_positions.size} units")

    keys = [(rule, epoch) for _, rule, epoch in PERIODS]
    hidden = compute_hidden_by_epoch(model_dir, keys, batch_size=args.batch_size)

    functional = {}
    trajectory_features = {}
    for label, rule, epoch in PERIODS:
        h = hidden[(rule, epoch)][:, :, ind_active][:, :, unit_positions]
        X = np.transpose(h, (2, 0, 1)).reshape(h.shape[2], -1)
        result = choose_functional_partition(X, max_k=args.max_subclusters)
        functional[label] = result
        trajectory_features[label] = X

    W_rec = load_w_rec(model_dir)
    W_sub = W_rec[np.ix_(unit_ids, unit_ids)]
    structural = internal_structural_partition(W_sub, max_k=args.max_subclusters)

    comparisons = []
    labels_mem_pro = functional["MemoryPro memory"]["labels"]
    for other in ["MemoryAnti memory", "DelayDM1 stimulus"]:
        other_labels = functional[other]["labels"]
        comparisons.append(
            {
                "comparison": f"MemoryPro memory vs {other}",
                "ari": float(adjusted_rand_score(labels_mem_pro, other_labels)),
                "nmi": float(normalized_mutual_info_score(labels_mem_pro, other_labels)),
            }
        )
    for name, result in functional.items():
        comparisons.append(
            {
                "comparison": f"{name} vs internal wiring",
                "ari": float(adjusted_rand_score(result["labels"], structural["labels"])),
                "nmi": float(normalized_mutual_info_score(result["labels"], structural["labels"])),
            }
        )

    stem = f"softplus_{variant}_k{args.k}_cell{cell}_subpartition_pilot"
    png, pdf = plot_partitions(unit_ids.astype(int).tolist(), functional, structural, output_dir, stem)

    csv_path = output_dir / f"{stem}_unit_labels.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["unit_id", *functional.keys(), "internal_wiring"])
        for i, unit_id in enumerate(unit_ids):
            writer.writerow(
                [
                    int(unit_id),
                    *[int(functional[name]["labels"][i]) for name in functional],
                    int(structural["labels"][i]),
                ]
            )

    comparison_csv = output_dir / f"{stem}_comparisons.csv"
    with comparison_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["comparison", "ari", "nmi"])
        writer.writeheader()
        writer.writerows(comparisons)

    summary = {
        "model_dir": str(model_dir),
        "variant": variant,
        "k": args.k,
        "cell": cell,
        "cell_size": int(unit_positions.size),
        "unit_ids": unit_ids.astype(int).tolist(),
        "functional": {
            name: {
                "chosen_k": result["chosen_k"],
                "best_silhouette": result["best_silhouette"],
                "silhouette_scores": result["silhouette_scores"],
                "labels": result["labels"].astype(int).tolist(),
            }
            for name, result in functional.items()
        },
        "internal_wiring": {
            "chosen_k": structural["chosen_k"],
            "best_silhouette": structural["best_silhouette"],
            "silhouette_scores": structural["silhouette_scores"],
            "labels": structural["labels"].astype(int).tolist(),
        },
        "comparisons": comparisons,
        "png": str(png),
        "pdf": str(pdf),
        "unit_labels_csv": str(csv_path),
        "comparison_csv": str(comparison_csv),
    }
    summary_path = output_dir / f"{stem}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
