"""Phase 1.2: Within-Cell Functional Fractionation

Test if structural cells fractionate into different functional sub-populations per task.
Compare sub-partition structure across tasks to identify stable vs dynamic fractionation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


from flexsym.structural.softplus_memory_subpartition_sweep import unit_features_for_period  # noqa: E402
from flexsym.structural.structural_cell_activity_coherence import compute_hidden_by_epoch, load_structural_labels  # noqa: E402
from flexsym.structural.structural_cell_subpartition_pilot import choose_functional_partition, zscore_rows  # noqa: E402


def filter_valid_task_epochs(model_dir: Path, task_epochs: list[tuple[str, str]], batch_size: int) -> list[tuple[str, str]]:
    """Keep only task/epoch pairs exposed by this repo's task generator."""
    from flexsym.core.task import generate_trials

    with (model_dir / "hp.json").open() as f:
        hp = json.load(f)
    hp.setdefault("rng", np.random.RandomState(0))

    valid: list[tuple[str, str]] = []
    by_rule: dict[str, list[str]] = {}
    for rule, epoch in task_epochs:
        by_rule.setdefault(rule, []).append(epoch)

    for rule, epochs in by_rule.items():
        try:
            trial = generate_trials(rule, hp, "test", noise_on=False, batch_size=batch_size)
        except Exception as exc:
            print(f"Warning: skipping rule '{rule}' ({exc})")
            continue
        available = set(trial.epochs)
        for epoch in epochs:
            if epoch in available:
                valid.append((rule, epoch))
            else:
                print(f"Warning: skipping unavailable epoch {rule}/{epoch}; available={sorted(available)}")
    return valid


def canonical(labels: np.ndarray) -> np.ndarray:
    """Relabel to 0, 1, 2, ... in order of first appearance."""
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


def cluster_units_for_task(
    hidden_by_epoch: dict,
    task: str,
    epoch: str,
    ind_active: np.ndarray,
    unit_positions: np.ndarray,
    max_k: int = 6,
) -> dict:
    """Cluster units within a cell by activity during specific task epoch.

    Returns dict with:
        - labels: Functional sub-partition labels
        - chosen_k: Number of clusters
        - silhouette: Silhouette score
    """
    n_units = len(unit_positions)

    if n_units < 3:
        return {
            "labels": np.zeros(n_units, dtype=np.int64),
            "chosen_k": 1,
            "silhouette": float("nan"),
        }

    # Get activity features for this period
    X = unit_features_for_period(
        hidden_by_epoch,
        task,
        epoch,
        ind_active,
        unit_positions,
    )

    # Use existing partition selection method
    result = choose_functional_partition(X, max_k=max_k)
    result["silhouette"] = result.get("best_silhouette", float("nan"))

    return result


def compute_fractionation_matrix(
    hidden_by_epoch: dict,
    labels: np.ndarray,
    ind_active: np.ndarray,
    task_epochs: list[tuple[str, str]],
    min_cell_size: int = 5,
    max_k: int = 6,
) -> tuple[pd.DataFrame, dict]:
    """Compute pairwise fractionation comparison matrix.

    For each cell, cluster units separately for each task, then compute
    ARI/NMI between all pairs of task-specific sub-partitions.

    Returns:
        - fractionation_df: DataFrame with all pairwise comparisons
        - partitions_by_cell: Dict mapping (cell_id, task, epoch) -> labels
    """
    k = int(labels.max() + 1)
    partitions_by_cell = {}
    results = []

    # First pass: compute sub-partitions for each cell × task
    print("Computing sub-partitions for each cell × task...")
    for cell_id in range(k):
        unit_positions = np.where(labels == cell_id)[0]

        if len(unit_positions) < min_cell_size:
            continue

        print(f"  Cell {cell_id} ({len(unit_positions)} units)")

        for task, epoch in task_epochs:
            try:
                result = cluster_units_for_task(
                    hidden_by_epoch,
                    task,
                    epoch,
                    ind_active,
                    unit_positions,
                    max_k=max_k,
                )
                partitions_by_cell[(cell_id, task, epoch)] = result
            except Exception as e:
                print(f"    Warning: {task}/{epoch} failed: {e}")
                continue

    # Second pass: compare all pairs
    print("\nComputing pairwise fractionation comparisons...")
    for cell_id in range(k):
        unit_positions = np.where(labels == cell_id)[0]

        if len(unit_positions) < min_cell_size:
            continue

        # Get all tasks for this cell
        cell_tasks = [(t, e) for (c, t, e) in partitions_by_cell.keys() if c == cell_id]

        # Compare all pairs
        for i, (task1, epoch1) in enumerate(cell_tasks):
            part1 = partitions_by_cell[(cell_id, task1, epoch1)]
            labels1 = part1["labels"]

            for task2, epoch2 in cell_tasks[i:]:  # Include self-comparison
                part2 = partitions_by_cell[(cell_id, task2, epoch2)]
                labels2 = part2["labels"]

                # Compute similarity
                ari = adjusted_rand_score(labels1, labels2)
                nmi = normalized_mutual_info_score(labels1, labels2)

                results.append({
                    "cell_id": cell_id,
                    "cell_size": len(unit_positions),
                    "task1": task1,
                    "epoch1": epoch1,
                    "task2": task2,
                    "epoch2": epoch2,
                    "k1": part1["chosen_k"],
                    "k2": part2["chosen_k"],
                    "sil1": part1["silhouette"],
                    "sil2": part2["silhouette"],
                    "ari": ari,
                    "nmi": nmi,
                })

    return pd.DataFrame(results), partitions_by_cell


def classify_fractionation_stability(
    frac_df: pd.DataFrame,
) -> pd.DataFrame:
    """Classify cells by fractionation stability.

    Categories:
    - Stable: High ARI across all task pairs (mean ARI > 0.7)
    - Dynamic: Low ARI across task pairs (mean ARI < 0.3)
    - Intermediate: Between stable and dynamic
    """
    cell_stats = []

    for cell_id in frac_df['cell_id'].unique():
        cell_data = frac_df[frac_df['cell_id'] == cell_id]

        # Exclude self-comparisons
        cross_task = cell_data[
            (cell_data['task1'] != cell_data['task2']) |
            (cell_data['epoch1'] != cell_data['epoch2'])
        ]

        if len(cross_task) == 0:
            continue

        mean_ari = cross_task['ari'].mean()
        mean_nmi = cross_task['nmi'].mean()
        std_ari = cross_task['ari'].std()

        # Classify
        if mean_ari > 0.7:
            stability = 'stable'
        elif mean_ari < 0.3:
            stability = 'dynamic'
        else:
            stability = 'intermediate'

        # Get mean k across tasks
        all_k = pd.concat([cross_task['k1'], cross_task['k2']]).values
        mean_k = all_k.mean()
        max_k = all_k.max()

        cell_stats.append({
            'cell_id': cell_id,
            'cell_size': cell_data['cell_size'].iloc[0],
            'stability': stability,
            'mean_ari': mean_ari,
            'std_ari': std_ari,
            'mean_nmi': mean_nmi,
            'mean_k': mean_k,
            'max_k': max_k,
            'n_comparisons': len(cross_task),
        })

    return pd.DataFrame(cell_stats)


def plot_fractionation_heatmap(
    frac_df: pd.DataFrame,
    output_path: Path,
    metric: str = 'ari',
    dpi: int = 300,
) -> None:
    """Plot heatmap of cross-task fractionation for each cell."""
    cells = sorted(frac_df['cell_id'].unique())

    # Determine grid size
    n_cells = len(cells)
    n_cols = min(4, n_cells)
    n_rows = int(np.ceil(n_cells / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_cells == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, cell_id in enumerate(cells):
        ax = axes[idx]
        cell_data = frac_df[frac_df['cell_id'] == cell_id].copy()

        # Create task labels
        cell_data['task_epoch1'] = cell_data['task1'] + '/' + cell_data['epoch1']
        cell_data['task_epoch2'] = cell_data['task2'] + '/' + cell_data['epoch2']

        # Pivot to matrix
        pivot = cell_data.pivot(index='task_epoch1', columns='task_epoch2', values=metric)

        # Make symmetric
        pivot = pivot.fillna(pivot.T)

        # Plot
        sns.heatmap(
            pivot,
            ax=ax,
            cmap='RdYlGn',
            vmin=0 if metric == 'ari' else 0,
            vmax=1,
            square=True,
            cbar_kws={'label': metric.upper()},
            xticklabels=True,
            yticklabels=True,
        )

        ax.set_title(f"Cell {cell_id} (n={cell_data['cell_size'].iloc[0]})")
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)

    # Hide unused subplots
    for idx in range(n_cells, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"Cross-Task Fractionation ({metric.upper()})", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Saved fractionation heatmap to {output_path}")


def plot_stability_summary(
    stability_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 300,
) -> None:
    """Plot summary of fractionation stability across cells."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Stability category counts
    ax = axes[0, 0]
    stability_counts = stability_df['stability'].value_counts()
    stability_counts.plot(kind='bar', ax=ax, color=['green', 'orange', 'red'])
    ax.set_title("Fractionation Stability Distribution")
    ax.set_xlabel("Stability Category")
    ax.set_ylabel("Number of Cells")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # 2. Mean ARI distribution
    ax = axes[0, 1]
    ax.hist(stability_df['mean_ari'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(0.3, color='red', linestyle='--', label='Dynamic threshold')
    ax.axvline(0.7, color='green', linestyle='--', label='Stable threshold')
    ax.set_title("Distribution of Mean Cross-Task ARI")
    ax.set_xlabel("Mean ARI")
    ax.set_ylabel("Number of Cells")
    ax.legend()

    # 3. Mean k vs stability
    ax = axes[1, 0]
    for stability in ['stable', 'intermediate', 'dynamic']:
        subset = stability_df[stability_df['stability'] == stability]
        ax.scatter(subset['mean_k'], subset['mean_ari'],
                  label=stability, alpha=0.6, s=50)
    ax.set_xlabel("Mean Number of Sub-partitions (k)")
    ax.set_ylabel("Mean ARI")
    ax.set_title("Sub-partition Complexity vs Stability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Cell size vs stability
    ax = axes[1, 1]
    for stability in ['stable', 'intermediate', 'dynamic']:
        subset = stability_df[stability_df['stability'] == stability]
        ax.scatter(subset['cell_size'], subset['mean_ari'],
                  label=stability, alpha=0.6, s=50)
    ax.set_xlabel("Cell Size (n units)")
    ax.set_ylabel("Mean ARI")
    ax.set_title("Cell Size vs Stability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Fractionation Stability Summary", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"Saved stability summary to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1.2: Within-cell functional fractionation across tasks"
    )
    parser.add_argument("model_dir", help="Path to trained model directory")
    parser.add_argument("--structural-labels", required=True,
                       help="Path to structural labels .npz file")
    parser.add_argument("--k", type=int, required=True,
                       help="Number of structural cells to load from the sweep")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--min-cell-size", type=int, default=5,
                       help="Minimum cell size to include")
    parser.add_argument("--max-subclusters", type=int, default=6,
                       help="Maximum number of sub-clusters per cell")
    parser.add_argument("--batch-size", type=int, default=1024,
                       help="Batch size for computing activity")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")

    # Task epochs to compare
    parser.add_argument("--task-epochs", nargs="+",
                       default=[
                           "delaygo/delay1",
                           "delayanti/delay1",
                           "fdgo/stim1",
                           "fdgo/delay1",
                           "fdanti/stim1",
                           "fdanti/delay1",
                       ],
                       help="Task epochs to compare (format: task/epoch)")

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load structural labels
    structural_path = Path(args.structural_labels)
    labels, ind_active, variant = load_structural_labels(structural_path, k=args.k)

    print(f"Loaded structural labels: {len(np.unique(labels))} cells, {len(ind_active)} active units")

    # Parse task epochs
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

    print(f"Comparing fractionation across {len(task_epochs)} task epochs")

    # Compute hidden activity
    print("\nComputing hidden activity...")
    hidden_by_epoch = compute_hidden_by_epoch(
        model_dir,
        task_epochs,
        batch_size=args.batch_size,
    )

    # Compute fractionation matrix
    print("\nComputing fractionation matrix...")
    frac_df, partitions = compute_fractionation_matrix(
        hidden_by_epoch,
        labels,
        ind_active,
        task_epochs,
        min_cell_size=args.min_cell_size,
        max_k=args.max_subclusters,
    )

    print(f"\nComputed {len(frac_df)} pairwise comparisons for {frac_df['cell_id'].nunique()} cells")

    # Classify stability
    print("\nClassifying fractionation stability...")
    stability_df = classify_fractionation_stability(frac_df)

    # Plot results
    print("\nGenerating visualizations...")
    plot_fractionation_heatmap(
        frac_df,
        output_dir / "fractionation_heatmap_ari.png",
        metric='ari',
        dpi=args.dpi,
    )

    plot_fractionation_heatmap(
        frac_df,
        output_dir / "fractionation_heatmap_nmi.png",
        metric='nmi',
        dpi=args.dpi,
    )

    plot_stability_summary(
        stability_df,
        output_dir / "stability_summary.png",
        dpi=args.dpi,
    )

    # Save results
    frac_df.to_csv(output_dir / "fractionation_matrix.csv", index=False)
    stability_df.to_csv(output_dir / "stability_classification.csv", index=False)

    # Summary statistics
    print("\n=== Fractionation Stability Summary ===")
    print(stability_df['stability'].value_counts())
    print(f"\nMean cross-task ARI: {stability_df['mean_ari'].mean():.3f}")
    print(f"Mean cross-task NMI: {stability_df['mean_nmi'].mean():.3f}")
    print(f"\nMean sub-partitions per cell: {stability_df['mean_k'].mean():.2f}")

    # Save summary
    summary = {
        "n_cells_analyzed": int(stability_df.shape[0]),
        "n_task_epochs": len(task_epochs),
        "n_comparisons": int(len(frac_df)),
        "stability_counts": stability_df['stability'].value_counts().to_dict(),
        "mean_cross_task_ari": float(stability_df['mean_ari'].mean()),
        "mean_cross_task_nmi": float(stability_df['mean_nmi'].mean()),
        "mean_k": float(stability_df['mean_k'].mean()),
    }

    with open(output_dir / "fractionation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
