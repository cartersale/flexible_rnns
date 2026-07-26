"""Full-context wiring alignment for memory functional subpartitions."""

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

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


STEP_DIR = Path(__file__).resolve().parent
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from signed_input_wl_structural_roles import load_w_rec  # noqa: E402
from softplus_memory_subpartition_sweep import unit_features_for_period  # noqa: E402
from structural_cell_activity_coherence import compute_hidden_by_epoch, load_structural_labels  # noqa: E402
from structural_cell_subpartition_pilot import choose_functional_partition, zscore_rows  # noqa: E402


CONTROL_TARGETS = ("random_projection", "internal_noise_padded")


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


def signed_input_features(
    W_rec: np.ndarray,
    unit_ids: np.ndarray,
    context_unit_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Signed recurrent input features from target units to a context set."""
    if context_unit_ids is None:
        context_unit_ids = np.arange(W_rec.shape[1], dtype=np.int64)
    W_context = W_rec[np.ix_(unit_ids, context_unit_ids)]
    W_pos = np.maximum(W_context, 0.0)
    W_neg = np.maximum(-W_context, 0.0)
    return np.concatenate([W_pos, W_neg], axis=1)


def feature_rank(X: np.ndarray) -> int:
    """Numerical rank after the same row normalization used for clustering."""
    if X.size == 0:
        return 0
    return int(np.linalg.matrix_rank(zscore_rows(X)))


def matched_control_dim(internal_features: np.ndarray, mode: str = "internal_width") -> int:
    if mode == "internal_rank":
        return max(1, feature_rank(internal_features))
    if mode == "internal_width":
        return max(1, int(internal_features.shape[1]))
    raise ValueError(f"unknown control dimension mode: {mode}")


def random_project_features(X: np.ndarray, n_components: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian random projection with a fixed matched feature budget."""
    n_components = max(1, min(int(n_components), int(X.shape[1])))
    projection = rng.normal(0.0, 1.0 / np.sqrt(n_components), size=(X.shape[1], n_components))
    return X @ projection


def pad_features_with_noise(X: np.ndarray, target_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Pad a feature matrix with matched-scale independent noise columns."""
    target_dim = int(target_dim)
    if target_dim <= X.shape[1]:
        return X.copy()
    sd = float(np.std(X))
    if not np.isfinite(sd) or sd <= 0.0:
        sd = 1.0
    noise = rng.normal(0.0, sd, size=(X.shape[0], target_dim - X.shape[1]))
    return np.concatenate([X, noise], axis=1)


def choose_partition_from_features(X: np.ndarray, max_k: int = 6) -> dict[str, object]:
    """Cluster units from an arbitrary feature matrix using the standard rule."""
    X = zscore_rows(X)
    n = int(X.shape[0])
    if n < 3:
        return {
            "labels": np.zeros(n, dtype=np.int64),
            "chosen_k": 1,
            "best_silhouette": float("nan"),
            "silhouette_scores": {},
        }
    k_max = min(max_k, n - 1)
    scores: dict[int, float] = {}
    labels_by_k: dict[int, np.ndarray] = {}
    for k in range(2, k_max + 1):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        labels = canonical(labels)
        if len(np.unique(labels)) < 2:
            continue
        scores[k] = float(silhouette_score(X, labels))
        labels_by_k[k] = labels

    if not scores:
        chosen_k = 1
    else:
        chosen_k = max(scores, key=scores.get)
    labels = np.zeros(n, dtype=np.int64) if chosen_k == 1 else labels_by_k[chosen_k]
    return {
        "labels": labels,
        "chosen_k": int(chosen_k),
        "best_silhouette": float(scores.get(chosen_k, float("nan"))),
        "silhouette_scores": {str(k): float(v) for k, v in sorted(scores.items())},
    }


def choose_full_context_partition(
    W_rec: np.ndarray,
    unit_ids: np.ndarray,
    max_k: int = 6,
) -> dict[str, object]:
    """Cluster units by signed recurrent input profiles from the whole network."""
    return choose_partition_from_features(signed_input_features(W_rec, unit_ids), max_k=max_k)


def draw_information_matched_controls(
    W_rec: np.ndarray,
    unit_ids: np.ndarray,
    *,
    n_control: int,
    max_k: int,
    rng: np.random.Generator,
    control_dim_mode: str = "internal_width",
) -> dict[str, object]:
    """Draw dimensionality-matched structural controls for one cell.

    The random-projection control keeps the full-context connectivity source but
    compresses it to the within-cell signed feature budget. The noise-padded
    control keeps the internal connectivity source and pads it to the
    full-context width with matched-scale independent noise.
    """
    full_features = signed_input_features(W_rec, unit_ids)
    internal_features = signed_input_features(W_rec, unit_ids, unit_ids)
    control_dim = matched_control_dim(internal_features, control_dim_mode)
    target_dim = int(full_features.shape[1])

    random_projection = []
    internal_noise_padded = []
    for _ in range(max(0, int(n_control))):
        random_projection.append(
            choose_partition_from_features(
                random_project_features(full_features, control_dim, rng),
                max_k=max_k,
            )
        )
        internal_noise_padded.append(
            choose_partition_from_features(
                pad_features_with_noise(internal_features, target_dim, rng),
                max_k=max_k,
            )
        )

    return {
        "random_projection": random_projection,
        "internal_noise_padded": internal_noise_padded,
        "full_feature_dim": int(full_features.shape[1]),
        "full_feature_rank": feature_rank(full_features),
        "internal_feature_dim": int(internal_features.shape[1]),
        "internal_feature_rank": feature_rank(internal_features),
        "matched_control_dim": int(control_dim),
        "noise_padded_dim": int(target_dim),
    }


def control_metric_summary(
    functional_labels: np.ndarray,
    control_partitions: list[dict[str, object]],
    *,
    reference_labels: np.ndarray | None = None,
) -> dict[str, float]:
    """Summarize functional alignment across repeated control partitions."""
    if reference_labels is None:
        reference_ari = float("nan")
        reference_nmi = float("nan")
    else:
        reference_ari = float(adjusted_rand_score(functional_labels, reference_labels))
        reference_nmi = float(normalized_mutual_info_score(functional_labels, reference_labels))

    if not control_partitions:
        return {
            "ari_mean": float("nan"),
            "ari_median": float("nan"),
            "ari_sd": float("nan"),
            "ari_max": float("nan"),
            "ari_p_ge_reference": float("nan"),
            "nmi_mean": float("nan"),
            "nmi_median": float("nan"),
            "nmi_sd": float("nan"),
            "nmi_max": float("nan"),
            "nmi_p_ge_reference": float("nan"),
            "k_mean": float("nan"),
            "k_median": float("nan"),
        }

    ari = np.asarray(
        [adjusted_rand_score(functional_labels, np.asarray(p["labels"], dtype=np.int64)) for p in control_partitions],
        dtype=float,
    )
    nmi = np.asarray(
        [
            normalized_mutual_info_score(functional_labels, np.asarray(p["labels"], dtype=np.int64))
            for p in control_partitions
        ],
        dtype=float,
    )
    k = np.asarray([int(p["chosen_k"]) for p in control_partitions], dtype=float)
    ddof = 1 if ari.size > 1 else 0
    return {
        "ari_mean": float(np.mean(ari)),
        "ari_median": float(np.median(ari)),
        "ari_sd": float(np.std(ari, ddof=ddof)),
        "ari_max": float(np.max(ari)),
        "ari_p_ge_reference": float(np.mean(ari >= reference_ari)) if np.isfinite(reference_ari) else float("nan"),
        "nmi_mean": float(np.mean(nmi)),
        "nmi_median": float(np.median(nmi)),
        "nmi_sd": float(np.std(nmi, ddof=ddof)),
        "nmi_max": float(np.max(nmi)),
        "nmi_p_ge_reference": float(np.mean(nmi >= reference_nmi)) if np.isfinite(reference_nmi) else float("nan"),
        "k_mean": float(np.mean(k)),
        "k_median": float(np.median(k)),
    }


def shuffled_same_sizes(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = np.asarray(labels, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    return shuffled


def summarize_period(rows: list[dict[str, object]], period: str | None = None) -> dict[str, object]:
    selected = [r for r in rows if period is None or r["period"] == period]
    ari_obs = np.asarray([r["ari"] for r in selected], dtype=float)
    nmi_obs = np.asarray([r["nmi"] for r in selected], dtype=float)
    ari_null = np.asarray([r["ari_null"] for r in selected], dtype=float)
    nmi_null = np.asarray([r["nmi_null"] for r in selected], dtype=float)
    ari_null_mean_by_draw = ari_null.mean(axis=0)
    nmi_null_mean_by_draw = nmi_null.mean(axis=0)
    ari_mean = float(ari_obs.mean())
    nmi_mean = float(nmi_obs.mean())
    summary = {
        "period": "combined" if period is None else period,
        "n_tests": len(selected),
        "ari_mean": ari_mean,
        "ari_median": float(np.median(ari_obs)),
        "ari_null_mean": float(ari_null_mean_by_draw.mean()),
        "ari_null_sd": float(ari_null_mean_by_draw.std(ddof=1)),
        "ari_z": float((ari_mean - ari_null_mean_by_draw.mean()) / ari_null_mean_by_draw.std(ddof=1)),
        "ari_p_upper": float(np.mean(ari_null_mean_by_draw >= ari_mean)),
        "nmi_mean": nmi_mean,
        "nmi_median": float(np.median(nmi_obs)),
        "nmi_null_mean": float(nmi_null_mean_by_draw.mean()),
        "nmi_null_sd": float(nmi_null_mean_by_draw.std(ddof=1)),
        "nmi_z": float((nmi_mean - nmi_null_mean_by_draw.mean()) / nmi_null_mean_by_draw.std(ddof=1)),
        "nmi_p_upper": float(np.mean(nmi_null_mean_by_draw >= nmi_mean)),
    }
    for control in CONTROL_TARGETS:
        summary.update(
            {
                f"{control}_ari_mean": float(np.nanmean([r[f"{control}_ari_mean"] for r in selected])),
                f"{control}_ari_median": float(np.nanmedian([r[f"{control}_ari_median"] for r in selected])),
                f"{control}_ari_p_ge_full_context_mean": float(
                    np.nanmean([r[f"{control}_ari_p_ge_reference"] for r in selected])
                ),
                f"{control}_nmi_mean": float(np.nanmean([r[f"{control}_nmi_mean"] for r in selected])),
                f"{control}_nmi_median": float(np.nanmedian([r[f"{control}_nmi_median"] for r in selected])),
                f"{control}_nmi_p_ge_full_context_mean": float(
                    np.nanmean([r[f"{control}_nmi_p_ge_reference"] for r in selected])
                ),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--structural-sweep", required=True)
    parser.add_argument("--cell-variance", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, default=33)
    parser.add_argument("--min-cell-size", type=int, default=5)
    parser.add_argument("--memory-percentile", type=float, default=60.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-subclusters", type=int, default=6)
    parser.add_argument("--n-null", type=int, default=5000)
    parser.add_argument("--n-control", type=int, default=100)
    parser.add_argument("--control-dim", choices=("internal_width", "internal_rank"), default="internal_width")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels, ind_active, variant = load_structural_labels(Path(args.structural_sweep), args.k)
    cell_var = np.load(args.cell_variance, allow_pickle=True)
    cell_sizes = np.asarray(cell_var["cell_sizes"], dtype=np.int64)
    key_names = [str(x) for x in np.asarray(cell_var["key_names"], dtype=object)]
    mem_idx = [key_names.index("delaygo/delay1"), key_names.index("delayanti/delay1")]
    valid = np.where(cell_sizes >= args.min_cell_size)[0]
    thresholds = np.percentile(cell_var["cell_epoch_variance"][valid][:, mem_idx], args.memory_percentile, axis=0)
    memory_cells = [
        int(c)
        for c in valid
        if np.all(cell_var["cell_epoch_variance"][c, mem_idx] >= thresholds)
    ]

    hidden = compute_hidden_by_epoch(
        model_dir,
        [("delaygo", "delay1"), ("delayanti", "delay1")],
        batch_size=args.batch_size,
    )
    W_rec = load_w_rec(model_dir)
    rng = np.random.default_rng(args.seed)
    control_rng = np.random.default_rng(args.seed + 1_000_003)

    rows: list[dict[str, object]] = []
    unit_rows: list[dict[str, object]] = []
    for cell in memory_cells:
        unit_positions = np.where(labels == cell)[0]
        unit_ids = ind_active[unit_positions]
        full_context = choose_full_context_partition(W_rec, unit_ids, max_k=args.max_subclusters)
        full_labels = np.asarray(full_context["labels"], dtype=np.int64)
        controls = draw_information_matched_controls(
            W_rec,
            unit_ids,
            n_control=args.n_control,
            max_k=args.max_subclusters,
            rng=control_rng,
            control_dim_mode=args.control_dim,
        )

        for unit_id, full_label in zip(unit_ids, full_labels):
            unit_rows.append(
                {
                    "cell": cell,
                    "unit_id": int(unit_id),
                    "full_context_wiring": int(full_label),
                }
            )

        for period, rule, epoch in [
            ("MemoryPro", "delaygo", "delay1"),
            ("MemoryAnti", "delayanti", "delay1"),
        ]:
            X = unit_features_for_period(hidden, rule, epoch, ind_active, unit_positions)
            functional = choose_functional_partition(X, max_k=args.max_subclusters)
            func_labels = np.asarray(functional["labels"], dtype=np.int64)
            ari = float(adjusted_rand_score(func_labels, full_labels))
            nmi = float(normalized_mutual_info_score(func_labels, full_labels))
            random_projection_metrics = control_metric_summary(
                func_labels,
                controls["random_projection"],
                reference_labels=full_labels,
            )
            noise_padded_metrics = control_metric_summary(
                func_labels,
                controls["internal_noise_padded"],
                reference_labels=full_labels,
            )
            ari_null = np.empty(args.n_null, dtype=float)
            nmi_null = np.empty(args.n_null, dtype=float)
            for b in range(args.n_null):
                null_labels = shuffled_same_sizes(func_labels, rng)
                ari_null[b] = adjusted_rand_score(null_labels, full_labels)
                nmi_null[b] = normalized_mutual_info_score(null_labels, full_labels)
            rows.append(
                {
                    "cell": cell,
                    "period": period,
                    "cell_size": int(unit_positions.size),
                    "functional_k": int(functional["chosen_k"]),
                    "functional_silhouette": float(functional["best_silhouette"]),
                    "full_context_k": int(full_context["chosen_k"]),
                    "full_context_silhouette": float(full_context["best_silhouette"]),
                    "full_context_feature_dim": int(controls["full_feature_dim"]),
                    "full_context_feature_rank": int(controls["full_feature_rank"]),
                    "internal_feature_dim": int(controls["internal_feature_dim"]),
                    "internal_feature_rank": int(controls["internal_feature_rank"]),
                    "matched_control_dim": int(controls["matched_control_dim"]),
                    "noise_padded_dim": int(controls["noise_padded_dim"]),
                    "n_control": int(args.n_control),
                    "ari": ari,
                    "nmi": nmi,
                    "ari_null": ari_null,
                    "nmi_null": nmi_null,
                    "ari_null_mean": float(ari_null.mean()),
                    "ari_null_sd": float(ari_null.std(ddof=1)),
                    "ari_p_upper": float(np.mean(ari_null >= ari)),
                    "nmi_null_mean": float(nmi_null.mean()),
                    "nmi_null_sd": float(nmi_null.std(ddof=1)),
                    "nmi_p_upper": float(np.mean(nmi_null >= nmi)),
                    "functional_labels": func_labels.astype(int).tolist(),
                    "full_context_labels": full_labels.astype(int).tolist(),
                    "unit_ids": unit_ids.astype(int).tolist(),
                    **{f"random_projection_{k}": v for k, v in random_projection_metrics.items()},
                    **{f"internal_noise_padded_{k}": v for k, v in noise_padded_metrics.items()},
                }
            )

    csv_path = output_dir / "softplus_memory_functional_full_context_wiring_alignment_null.csv"
    fieldnames = [
        "cell",
        "period",
        "cell_size",
        "functional_k",
        "functional_silhouette",
        "full_context_k",
        "full_context_silhouette",
        "full_context_feature_dim",
        "full_context_feature_rank",
        "internal_feature_dim",
        "internal_feature_rank",
        "matched_control_dim",
        "noise_padded_dim",
        "n_control",
        "ari",
        "nmi",
        "ari_null_mean",
        "ari_null_sd",
        "ari_p_upper",
        "nmi_null_mean",
        "nmi_null_sd",
        "nmi_p_upper",
        "functional_labels",
        "full_context_labels",
        "unit_ids",
    ]
    for control in CONTROL_TARGETS:
        fieldnames.extend(
            [
                f"{control}_ari_mean",
                f"{control}_ari_median",
                f"{control}_ari_sd",
                f"{control}_ari_max",
                f"{control}_ari_p_ge_reference",
                f"{control}_nmi_mean",
                f"{control}_nmi_median",
                f"{control}_nmi_sd",
                f"{control}_nmi_max",
                f"{control}_nmi_p_ge_reference",
                f"{control}_k_mean",
                f"{control}_k_median",
            ]
        )
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})

    unit_labels_path = output_dir / "softplus_memory_functional_full_context_wiring_unit_labels.csv"
    with unit_labels_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cell", "unit_id", "full_context_wiring"])
        writer.writeheader()
        writer.writerows(unit_rows)

    summary_rows = [
        summarize_period(rows, "MemoryPro"),
        summarize_period(rows, "MemoryAnti"),
        summarize_period(rows, None),
    ]
    summary_csv = output_dir / "softplus_memory_functional_full_context_wiring_alignment_null_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "model_dir": str(model_dir),
        "variant": variant,
        "k": args.k,
        "n_null": args.n_null,
        "n_control": args.n_control,
        "control_dim": args.control_dim,
        "memory_cells": memory_cells,
        "n_memory_cells": len(memory_cells),
        "csv_path": str(csv_path),
        "unit_labels_path": str(unit_labels_path),
        "summary_csv": str(summary_csv),
        "summary_rows": summary_rows,
    }
    summary_json = output_dir / "softplus_memory_functional_full_context_wiring_alignment_null_summary.json"
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
