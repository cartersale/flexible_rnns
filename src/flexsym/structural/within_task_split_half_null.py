"""Within-task split-half null for within-cell functional fractionation (Null #2).

Cross-task fractionation (structural_cell_cross_task_fractionation.py) clusters the
units inside each structural cell separately per task and reports how unstable those
sub-partitions are across tasks (low ARI -> the cell fractionates differently per
task). The obvious worry: maybe the within-cell partition is just unstable, full
stop -- even re-clustering the *same* task on a different set of trials would give a
low ARI. If so, low cross-task ARI says nothing special.

This script measures that ceiling. For one task it splits the trials in half, clusters
the units within each cell on each half independently, and computes ARI between the two
half-partitions. Averaged over many random half-splits (and over tasks) this is the
within-task split-half null: the ARI you would expect from trial-sampling alone, with
the task held fixed. Cross-task ARI is computed on the same cells with the same
clustering and reported alongside it. If cross-task ARI sits well below the split-half
null, the cells genuinely reorganize across tasks rather than just being noisy.

Scope for now: 256-unit softplus-diagonal RNN, seed 0 (LeakyRNN_softplus_diag_256_all_0).
"""

from __future__ import annotations

import argparse
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
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


from flexsym.structural.structural_cell_activity_coherence import (  # noqa: E402
    compute_hidden_by_epoch,
    load_structural_labels,
)
from flexsym.structural.structural_cell_cross_task_fractionation import (  # noqa: E402
    compute_fractionation_matrix,
    filter_valid_task_epochs,
)
from flexsym.structural.structural_cell_subpartition_pilot import choose_functional_partition  # noqa: E402


def unit_features_for_batch_subset(
    hidden_by_epoch: dict,
    rule: str,
    epoch: str,
    ind_active: np.ndarray,
    unit_positions: np.ndarray,
    batch_idx: np.ndarray,
) -> np.ndarray:
    """Per-unit activity features (unit x time*trials) for a subset of trials.

    Mirrors softplus_memory_subpartition_sweep.unit_features_for_period but restricts
    the trial/batch axis to ``batch_idx`` so each split-half sees disjoint trials.
    """
    h = hidden_by_epoch[(rule, epoch)][:, :, ind_active][:, :, unit_positions]
    h = h[:, batch_idx, :]
    return np.transpose(h, (2, 0, 1)).reshape(h.shape[2], -1)


def compute_split_half_null(
    hidden_by_epoch: dict,
    labels: np.ndarray,
    ind_active: np.ndarray,
    task_epochs: list[tuple[str, str]],
    *,
    min_cell_size: int = 5,
    max_k: int = 6,
    n_splits: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Split-half within-task fractionation stability for each cell x task.

    For each structural cell and each task/epoch, repeatedly partition the trials into
    two disjoint halves, cluster the cell's units on each half, and measure ARI/NMI
    between the two half-partitions. High ARI => the within-cell partition is a stable
    property of that single task; this is the null/ceiling for cross-task ARI.
    """
    k = int(labels.max() + 1)
    results = []

    print("Computing within-task split-half null...")
    for cell_id in range(k):
        unit_positions = np.where(labels == cell_id)[0]
        if len(unit_positions) < min_cell_size:
            continue
        print(f"  Cell {cell_id} ({len(unit_positions)} units)")

        for task, epoch in task_epochs:
            n_trials = hidden_by_epoch[(task, epoch)].shape[1]
            if n_trials < 2:
                continue
            for split_idx in range(n_splits):
                rng = np.random.default_rng(seed + 1000 * cell_id + split_idx)
                perm = rng.permutation(n_trials)
                half = n_trials // 2
                idx_a = np.sort(perm[:half])
                idx_b = np.sort(perm[half : 2 * half])

                try:
                    feat_a = unit_features_for_batch_subset(
                        hidden_by_epoch, task, epoch, ind_active, unit_positions, idx_a
                    )
                    feat_b = unit_features_for_batch_subset(
                        hidden_by_epoch, task, epoch, ind_active, unit_positions, idx_b
                    )
                    part_a = choose_functional_partition(feat_a, max_k=max_k)
                    part_b = choose_functional_partition(feat_b, max_k=max_k)
                except Exception as exc:  # noqa: BLE001
                    print(f"    Warning: {task}/{epoch} split {split_idx} failed: {exc}")
                    continue

                results.append(
                    {
                        "cell_id": cell_id,
                        "cell_size": len(unit_positions),
                        "task": task,
                        "epoch": epoch,
                        "split_idx": split_idx,
                        "n_trials_half": int(half),
                        "k_a": part_a["chosen_k"],
                        "k_b": part_b["chosen_k"],
                        "sil_a": part_a.get("best_silhouette", float("nan")),
                        "sil_b": part_b.get("best_silhouette", float("nan")),
                        "ari": float(adjusted_rand_score(part_a["labels"], part_b["labels"])),
                        "nmi": float(
                            normalized_mutual_info_score(part_a["labels"], part_b["labels"])
                        ),
                    }
                )

    return pd.DataFrame(results)


def compare_null_to_cross_task(
    split_half_df: pd.DataFrame,
    frac_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-cell comparison: within-task split-half ARI vs cross-task ARI."""
    # Cross-task: exclude self-comparisons (task1==task2 & epoch1==epoch2).
    cross = frac_df[
        (frac_df["task1"] != frac_df["task2"]) | (frac_df["epoch1"] != frac_df["epoch2"])
    ]

    rows = []
    cell_ids = sorted(set(split_half_df["cell_id"]).union(cross["cell_id"]))
    for cell_id in cell_ids:
        sh = split_half_df[split_half_df["cell_id"] == cell_id]
        ct = cross[cross["cell_id"] == cell_id]
        if len(sh) == 0 or len(ct) == 0:
            continue
        null_ari = sh["ari"].mean()
        cross_ari = ct["ari"].mean()
        rows.append(
            {
                "cell_id": cell_id,
                "cell_size": int(sh["cell_size"].iloc[0]),
                "split_half_ari_mean": float(null_ari),
                "split_half_ari_std": float(sh["ari"].std()),
                "split_half_nmi_mean": float(sh["nmi"].mean()),
                "cross_task_ari_mean": float(cross_ari),
                "cross_task_ari_std": float(ct["ari"].std()),
                "cross_task_nmi_mean": float(ct["nmi"].mean()),
                # gap > 0 => cross-task partition is less stable than the within-task null,
                # i.e. genuine cross-task reorganization beyond trial-sampling noise.
                "ari_gap": float(null_ari - cross_ari),
                "n_splits": int(len(sh)),
                "n_cross_pairs": int(len(ct)),
            }
        )

    return pd.DataFrame(rows)


def plot_null_vs_cross_task(
    comparison_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300,
) -> None:
    """Scatter + paired plot of within-task split-half ARI vs cross-task ARI."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Scatter: split-half null (x) vs cross-task (y), with identity line.
    ax = axes[0]
    ax.scatter(
        comparison_df["split_half_ari_mean"],
        comparison_df["cross_task_ari_mean"],
        s=np.clip(comparison_df["cell_size"] * 6, 30, 400),
        alpha=0.7,
        edgecolor="black",
    )
    lim = [0, 1]
    ax.plot(lim, lim, "k--", linewidth=1, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Within-task split-half ARI (null / ceiling)")
    ax.set_ylabel("Cross-task ARI")
    ax.set_title("Per-cell: cross-task vs within-task stability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Paired distribution.
    ax = axes[1]
    data = [
        comparison_df["split_half_ari_mean"].values,
        comparison_df["cross_task_ari_mean"].values,
    ]
    ax.boxplot(data, labels=["split-half\nnull", "cross-task"], showmeans=True)
    for _, row in comparison_df.iterrows():
        ax.plot(
            [1, 2],
            [row["split_half_ari_mean"], row["cross_task_ari_mean"]],
            color="gray",
            alpha=0.4,
            linewidth=0.8,
        )
    ax.set_ylabel("ARI")
    ax.set_title("Paired per-cell ARI")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Within-task split-half null vs cross-task fractionation", y=1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Within-task split-half null (#2) vs cross-task fractionation"
    )
    parser.add_argument("model_dir", help="Path to trained model checkpoint directory")
    parser.add_argument(
        "--structural-labels", required=True, help="Path to structural sweep .npz file"
    )
    parser.add_argument("--k", type=int, required=True, help="Number of structural cells")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--min-cell-size", type=int, default=5)
    parser.add_argument("--max-subclusters", type=int, default=6)
    parser.add_argument("--n-splits", type=int, default=20, help="Random half-splits per cell x task")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--task-epochs",
        nargs="+",
        default=[
            "delaygo/delay1",
            "delayanti/delay1",
            "fdgo/stim1",
            "fdgo/delay1",
            "fdanti/stim1",
            "fdanti/delay1",
        ],
        help="Task epochs to use (format: task/epoch); matches the cross-task default",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels, ind_active, variant = load_structural_labels(Path(args.structural_labels), k=args.k)
    print(
        f"Loaded structural labels: {len(np.unique(labels))} cells, "
        f"{len(ind_active)} active units (variant={variant})"
    )

    task_epochs = []
    for epoch_str in args.task_epochs:
        parts = epoch_str.split("/")
        if len(parts) == 2:
            task_epochs.append((parts[0], parts[1]))
        else:
            print(f"Warning: skipping malformed epoch '{epoch_str}'")
    task_epochs = filter_valid_task_epochs(model_dir, task_epochs, args.batch_size)
    if not task_epochs:
        raise ValueError("No valid task epochs remain after validation")
    print(f"Using {len(task_epochs)} task epochs: {task_epochs}")

    print("\nComputing hidden activity...")
    hidden_by_epoch = compute_hidden_by_epoch(model_dir, task_epochs, batch_size=args.batch_size)

    print("\nComputing within-task split-half null...")
    split_half_df = compute_split_half_null(
        hidden_by_epoch,
        labels,
        ind_active,
        task_epochs,
        min_cell_size=args.min_cell_size,
        max_k=args.max_subclusters,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    print("\nComputing cross-task fractionation (same cells / clustering)...")
    frac_df, _ = compute_fractionation_matrix(
        hidden_by_epoch,
        labels,
        ind_active,
        task_epochs,
        min_cell_size=args.min_cell_size,
        max_k=args.max_subclusters,
    )

    comparison_df = compare_null_to_cross_task(split_half_df, frac_df)

    # Save tables.
    split_half_df.to_csv(output_dir / "split_half_null.csv", index=False)
    frac_df.to_csv(output_dir / "cross_task_fractionation.csv", index=False)
    comparison_df.to_csv(output_dir / "null_vs_cross_task_comparison.csv", index=False)

    if len(comparison_df) > 0:
        plot_null_vs_cross_task(
            comparison_df, output_dir / "null_vs_cross_task.png", dpi=args.dpi
        )

    # Aggregate stats (paired Wilcoxon over cells, if available).
    wilcoxon_p = float("nan")
    if len(comparison_df) >= 1:
        try:
            from scipy.stats import wilcoxon

            if len(comparison_df) >= 2:
                _, wilcoxon_p = wilcoxon(
                    comparison_df["split_half_ari_mean"],
                    comparison_df["cross_task_ari_mean"],
                )
                wilcoxon_p = float(wilcoxon_p)
        except Exception as exc:  # noqa: BLE001
            print(f"Wilcoxon test skipped: {exc}")

    summary = {
        "model_dir": str(model_dir),
        "variant": variant,
        "k": int(args.k),
        "n_cells_compared": int(len(comparison_df)),
        "n_task_epochs": len(task_epochs),
        "task_epochs": [f"{r}/{e}" for r, e in task_epochs],
        "n_splits": int(args.n_splits),
        "min_cell_size": int(args.min_cell_size),
        "max_subclusters": int(args.max_subclusters),
        "mean_split_half_ari": float(comparison_df["split_half_ari_mean"].mean())
        if len(comparison_df)
        else float("nan"),
        "mean_cross_task_ari": float(comparison_df["cross_task_ari_mean"].mean())
        if len(comparison_df)
        else float("nan"),
        "mean_ari_gap": float(comparison_df["ari_gap"].mean())
        if len(comparison_df)
        else float("nan"),
        "n_cells_cross_below_null": int((comparison_df["ari_gap"] > 0).sum())
        if len(comparison_df)
        else 0,
        "wilcoxon_p_split_half_vs_cross": wilcoxon_p,
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Within-task split-half null vs cross-task ===")
    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
