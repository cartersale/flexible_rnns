"""Memory subpartition checks across softplus structural cells."""

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
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score



from flexsym.structural.signed_input_wl_structural_roles import load_w_rec  # noqa: E402
from flexsym.structural.structural_cell_activity_coherence import compute_hidden_by_epoch, load_structural_labels  # noqa: E402
from flexsym.structural.structural_cell_subpartition_pilot import (  # noqa: E402
    choose_functional_partition,
    internal_structural_partition,
)


def refinement_metrics(fine: np.ndarray, coarse: np.ndarray) -> dict[str, object]:
    fine = np.asarray(fine, dtype=np.int64)
    coarse = np.asarray(coarse, dtype=np.int64)
    fine_labels = sorted(np.unique(fine).tolist())
    table = np.zeros((len(fine_labels), int(coarse.max()) + 1), dtype=int)
    mapping = {}
    violations = []
    for i, label in enumerate(fine_labels):
        mask = fine == label
        vals, counts = np.unique(coarse[mask], return_counts=True)
        for val, count in zip(vals, counts):
            table[i, int(val)] = int(count)
        majority = int(vals[np.argmax(counts)])
        mapping[int(label)] = majority
        if len(vals) > 1:
            violations.append(int(label))
    return {
        "is_refinement": len(violations) == 0,
        "violating_fine_clusters": violations,
        "fine_to_coarse_majority": mapping,
        "purity": float(table.max(axis=1).sum() / max(table.sum(), 1)),
        "contingency": table.tolist(),
    }


def unit_features_for_period(hidden_by_epoch, rule: str, epoch: str, ind_active, unit_positions) -> np.ndarray:
    h = hidden_by_epoch[(rule, epoch)][:, :, ind_active][:, :, unit_positions]
    return np.transpose(h, (2, 0, 1)).reshape(h.shape[2], -1)


def plot_memory_geometry(
    X: np.ndarray,
    labels: np.ndarray,
    unit_ids: np.ndarray,
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    Xz = X - X.mean(axis=1, keepdims=True)
    Xz = Xz / np.maximum(Xz.std(axis=1, keepdims=True), 1e-12)
    coords = PCA(n_components=3, random_state=0).fit_transform(Xz)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    ax = axes[0]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=90, edgecolor="black")
    for unit_id, (x, y) in zip(unit_ids, coords[:, :2]):
        ax.text(x, y, str(int(unit_id)), fontsize=8, ha="center", va="center")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("MemoryPro unit trajectories")

    ax = axes[1]
    center = coords[:, :2].mean(axis=0)
    centered = coords[:, :2] - center
    radii = np.linalg.norm(centered, axis=1)
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    order = np.argsort(angles)
    ax.scatter(np.cos(angles), np.sin(angles), c=labels, cmap="tab10", s=90, edgecolor="black")
    for unit_id, angle in zip(unit_ids, angles):
        ax.text(np.cos(angle), np.sin(angle), str(int(unit_id)), fontsize=8, ha="center", va="center")
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="0.7", linewidth=1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Angular view, CV radius={np.std(radii)/np.mean(radii):.2f}")

    # A crude ring diagnostic: uniform angular coverage and roughly equal radii.
    gaps = np.diff(np.r_[angles[order], angles[order][0] + 2 * np.pi])
    ring_summary = {
        "radius_mean": float(np.mean(radii)),
        "radius_cv": float(np.std(radii) / max(np.mean(radii), 1e-12)),
        "angle_gap_cv": float(np.std(gaps) / max(np.mean(gaps), 1e-12)),
    }

    png = output_dir / f"{stem}_memory_geometry.png"
    pdf = output_dir / f"{stem}_memory_geometry.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf, ring_summary


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
    parser.add_argument("--pilot-cell", type=int, default=3)
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
        [("delaygo", "delay1"), ("delayanti", "delay1"), ("delaydm1", "stim1")],
        batch_size=args.batch_size,
    )
    W_rec = load_w_rec(model_dir)

    rows = []
    pilot_summary = None
    for cell in memory_cells:
        unit_positions = np.where(labels == cell)[0]
        unit_ids = ind_active[unit_positions]
        X_pro = unit_features_for_period(hidden, "delaygo", "delay1", ind_active, unit_positions)
        X_anti = unit_features_for_period(hidden, "delayanti", "delay1", ind_active, unit_positions)
        X_stim = unit_features_for_period(hidden, "delaydm1", "stim1", ind_active, unit_positions)
        part_pro = choose_functional_partition(X_pro, max_k=args.max_subclusters)
        part_anti = choose_functional_partition(X_anti, max_k=args.max_subclusters)
        part_stim = choose_functional_partition(X_stim, max_k=args.max_subclusters)
        W_sub = W_rec[np.ix_(unit_ids, unit_ids)]
        structural = internal_structural_partition(W_sub, max_k=args.max_subclusters)
        ref = refinement_metrics(part_pro["labels"], structural["labels"])

        row = {
            "cell": cell,
            "cell_size": int(unit_positions.size),
            "memory_variance_mean": float(cell_var["cell_epoch_variance"][cell, mem_idx].mean()),
            "memorypro_k": part_pro["chosen_k"],
            "memoryanti_k": part_anti["chosen_k"],
            "delaydm1_stim_k": part_stim["chosen_k"],
            "internal_k": structural["chosen_k"],
            "memorypro_silhouette": part_pro["best_silhouette"],
            "memoryanti_silhouette": part_anti["best_silhouette"],
            "delaydm1_stim_silhouette": part_stim["best_silhouette"],
            "internal_silhouette": structural["best_silhouette"],
            "pro_vs_anti_ari": float(adjusted_rand_score(part_pro["labels"], part_anti["labels"])),
            "pro_vs_anti_nmi": float(normalized_mutual_info_score(part_pro["labels"], part_anti["labels"])),
            "pro_vs_stim_ari": float(adjusted_rand_score(part_pro["labels"], part_stim["labels"])),
            "pro_vs_stim_nmi": float(normalized_mutual_info_score(part_pro["labels"], part_stim["labels"])),
            "pro_vs_internal_ari": float(adjusted_rand_score(part_pro["labels"], structural["labels"])),
            "pro_vs_internal_nmi": float(normalized_mutual_info_score(part_pro["labels"], structural["labels"])),
            "memory_refines_internal": ref["is_refinement"],
            "memory_to_internal_purity": ref["purity"],
            "memory_refinement_violations": json.dumps(ref["violating_fine_clusters"]),
        }
        rows.append(row)

        if cell == args.pilot_cell:
            stem = f"softplus_{variant}_k{args.k}_cell{cell}"
            geom_png, geom_pdf, ring_summary = plot_memory_geometry(
                X_pro,
                part_pro["labels"],
                unit_ids,
                output_dir,
                stem,
            )
            pilot_summary = {
                "cell": cell,
                "unit_ids": unit_ids.astype(int).tolist(),
                "memory_labels": part_pro["labels"].astype(int).tolist(),
                "internal_labels": structural["labels"].astype(int).tolist(),
                "refinement": ref,
                "geometry_png": str(geom_png),
                "geometry_pdf": str(geom_pdf),
                "ring_summary": ring_summary,
            }

    csv_path = output_dir / "softplus_memory_subpartition_sweep.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model_dir": str(model_dir),
        "variant": variant,
        "k": args.k,
        "min_cell_size": args.min_cell_size,
        "memory_percentile": args.memory_percentile,
        "memory_cells": memory_cells,
        "n_memory_cells": len(memory_cells),
        "mean_pro_vs_anti_ari": float(np.mean([r["pro_vs_anti_ari"] for r in rows])),
        "median_pro_vs_anti_ari": float(np.median([r["pro_vs_anti_ari"] for r in rows])),
        "mean_pro_vs_stim_ari": float(np.mean([r["pro_vs_stim_ari"] for r in rows])),
        "median_pro_vs_stim_ari": float(np.median([r["pro_vs_stim_ari"] for r in rows])),
        "n_exact_pro_anti_matches": int(sum(r["pro_vs_anti_ari"] == 1.0 for r in rows)),
        "n_memory_refines_internal": int(sum(bool(r["memory_refines_internal"]) for r in rows)),
        "csv_path": str(csv_path),
        "pilot": pilot_summary,
    }
    summary_path = output_dir / "softplus_memory_subpartition_sweep_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
