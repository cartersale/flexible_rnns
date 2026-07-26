"""Random-partition null for within-cell activity covariance metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import numpy as np



from flexsym.structural.structural_cell_activity_coherence import (  # noqa: E402
    cell_activity_metrics,
    compute_hidden_by_epoch,
    key_name,
    load_structural_labels,
)


METRIC_NAMES = [
    "coherence",
    "coherence_abs",
    "coherence_r2",
    "top_eig_frac",
    "participation_rank",
    "entropy_rank",
]


COMPARE_PAIRS = [
    ("memory_pro_vs_anti_shared", "delaygo/delay1", "delayanti/delay1"),
    ("memorypro_vs_delaydm1_stim_different", "delaygo/delay1", "delaydm1/stim1"),
    ("memorypro_vs_dmcgo_stim_different", "delaygo/delay1", "dmcgo/stim1"),
    ("reaction_go_vs_anti_shared_response", "reactgo/go1", "reactanti/go1"),
    ("reactgo_vs_delaygo_response_different", "reactgo/go1", "delaygo/go1"),
]


def metric_matrix_for_labels(
    labels: np.ndarray,
    unit_obs_by_period: list[np.ndarray],
    *,
    k: int,
    min_cell_size: int,
) -> dict[str, np.ndarray]:
    cell_sizes = np.bincount(labels, minlength=k)
    valid_cells = np.where(cell_sizes >= min_cell_size)[0]
    matrices = {
        name: np.full((k, len(unit_obs_by_period)), np.nan, dtype=float)
        for name in METRIC_NAMES
    }
    for period_idx, unit_obs in enumerate(unit_obs_by_period):
        for cell in valid_cells:
            unit_idx = np.where(labels == cell)[0]
            metrics = cell_activity_metrics(unit_obs[unit_idx])
            for name in METRIC_NAMES:
                matrices[name][cell, period_idx] = metrics[name]
    return matrices


def summarize_struct_vs_null(
    structural: dict[str, np.ndarray],
    null_values: dict[str, np.ndarray],
    *,
    label: str,
) -> list[dict[str, object]]:
    rows = []
    for metric in METRIC_NAMES:
        struct_mean = float(np.nanmean(structural[metric]))
        vals = null_values[metric]
        null_mean = float(np.nanmean(vals))
        null_sd = float(np.nanstd(vals, ddof=1))
        rows.append(
            {
                "summary": label,
                "metric": metric,
                "struct_value": struct_mean,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "z": (struct_mean - null_mean) / null_sd if null_sd > 0 else np.nan,
                "p_lower": float(np.mean(vals <= struct_mean)),
                "p_upper": float(np.mean(vals >= struct_mean)),
            }
        )
    return rows


def comparison_deltas(
    matrices: dict[str, np.ndarray],
    key_names: list[str],
) -> dict[tuple[str, str], float]:
    out = {}
    for comparison, a, b in COMPARE_PAIRS:
        ia = key_names.index(a)
        ib = key_names.index(b)
        for metric in METRIC_NAMES:
            out[(comparison, metric)] = float(np.nanmean(np.abs(matrices[metric][:, ia] - matrices[metric][:, ib])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--activity-clusters", required=True)
    parser.add_argument("--structural-sweep", required=True)
    parser.add_argument("--structural-coherence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n-random", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-cell-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    activity = np.load(args.activity_clusters, allow_pickle=True)
    keys_raw = [tuple(x) for x in np.asarray(activity["keys"], dtype=object)]
    key_names = [key_name(x) for x in keys_raw]

    labels, ind_active, variant = load_structural_labels(Path(args.structural_sweep), args.k)
    rng = np.random.default_rng(args.seed)

    hidden_by_epoch = compute_hidden_by_epoch(model_dir, keys_raw, batch_size=args.batch_size)
    unit_obs_by_period = []
    for key in keys_raw:
        h = hidden_by_epoch[key][:, :, ind_active]
        unit_obs_by_period.append(np.transpose(h, (2, 0, 1)).reshape(h.shape[2], -1))

    # task.py monkey-patches np.load during hidden-state generation to force
    # allow_pickle=True, so avoid passing that argument explicitly here.
    structural_npz = np.load(args.structural_coherence)
    structural = {name: np.asarray(structural_npz[name], dtype=float) for name in METRIC_NAMES}
    structural_deltas = comparison_deltas(structural, key_names)

    null_global = {name: np.full(args.n_random, np.nan, dtype=float) for name in METRIC_NAMES}
    null_deltas = {
        (comparison, metric): np.full(args.n_random, np.nan, dtype=float)
        for comparison, _, _ in COMPARE_PAIRS
        for metric in METRIC_NAMES
    }

    for b in range(args.n_random):
        random_labels = labels.copy()
        rng.shuffle(random_labels)
        matrices = metric_matrix_for_labels(
            random_labels,
            unit_obs_by_period,
            k=args.k,
            min_cell_size=args.min_cell_size,
        )
        for metric in METRIC_NAMES:
            null_global[metric][b] = np.nanmean(matrices[metric])
        deltas = comparison_deltas(matrices, key_names)
        for key, value in deltas.items():
            null_deltas[key][b] = value

    global_rows = summarize_struct_vs_null(structural, null_global, label="mean_over_cells_periods")
    global_csv = output_dir / f"random_partition_covariance_null_global_{variant}_k{args.k}.csv"
    with global_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(global_rows[0].keys()))
        writer.writeheader()
        writer.writerows(global_rows)

    delta_rows = []
    for (comparison, metric), vals in sorted(null_deltas.items()):
        struct_value = structural_deltas[(comparison, metric)]
        null_mean = float(np.nanmean(vals))
        null_sd = float(np.nanstd(vals, ddof=1))
        delta_rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "struct_value": struct_value,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "z": (struct_value - null_mean) / null_sd if null_sd > 0 else np.nan,
                "p_lower": float(np.mean(vals <= struct_value)),
                "p_upper": float(np.mean(vals >= struct_value)),
            }
        )
    delta_csv = output_dir / f"random_partition_covariance_null_deltas_{variant}_k{args.k}.csv"
    with delta_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(delta_rows)

    npz_path = output_dir / f"random_partition_covariance_null_{variant}_k{args.k}.npz"
    np.savez(
        npz_path,
        **{f"global_{name}": values for name, values in null_global.items()},
        null_delta_keys=np.asarray(list(null_deltas.keys()), dtype=object),
        null_delta_values=np.asarray([null_deltas[key] for key in null_deltas], dtype=float),
        labels_struct=labels,
        ind_active=ind_active,
        key_names=np.asarray(key_names, dtype=object),
        n_random=args.n_random,
        seed=args.seed,
        k=args.k,
        variant=variant,
    )

    summary = {
        "variant": variant,
        "k": args.k,
        "n_random": args.n_random,
        "seed": args.seed,
        "global_csv": str(global_csv),
        "delta_csv": str(delta_csv),
        "npz_path": str(npz_path),
    }
    with (output_dir / f"random_partition_covariance_null_summary_{variant}_k{args.k}.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
