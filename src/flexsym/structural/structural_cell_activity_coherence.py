"""Within-structural-cell activity coherence across task periods.

For each structural cell C and task period T, coherence is the mean pairwise
correlation among unit activity vectors within C during T. Activity vectors are
hidden-state trajectories flattened over time x trial/condition.
"""

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




def load_structural_labels(sweep_path: Path, k: int) -> tuple[np.ndarray, np.ndarray, str]:
    sweep = np.load(sweep_path, allow_pickle=True)
    rows = np.asarray(sweep["rows"], dtype=object)
    row_columns = [str(x) for x in np.asarray(sweep["row_columns"], dtype=object)]
    k_idx = row_columns.index("k")
    k_min = int(rows[0, k_idx])
    k_max = int(rows[-1, k_idx])
    if k < k_min or k > k_max:
        raise ValueError(f"k={k} outside sweep range [{k_min}, {k_max}]")
    labels = np.asarray(sweep["labels_struct_by_k"][k - k_min], dtype=np.int64)
    ind_active = np.asarray(sweep["ind_active"], dtype=np.int64)
    variant = str(np.asarray(sweep["matrix_variant"]).item())
    return labels, ind_active, variant


def key_name(key: object) -> str:
    arr = np.asarray(key, dtype=object)
    if arr.ndim == 0:
        return str(arr.item())
    return "/".join(str(x) for x in arr)


def cell_activity_metrics(unit_by_obs: np.ndarray) -> dict[str, float]:
    """Coherence metrics among unit rows over flattened observations."""
    if unit_by_obs.shape[0] < 2:
        return {
            "coherence": float("nan"),
            "coherence_abs": float("nan"),
            "coherence_r2": float("nan"),
            "top_eig_frac": float("nan"),
            "participation_rank": float("nan"),
            "entropy_rank": float("nan"),
        }
    X = np.asarray(unit_by_obs, dtype=float)
    X = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X, axis=1)
    valid = norms > 1e-12
    if valid.sum() < 2:
        return {
            "coherence": float("nan"),
            "coherence_abs": float("nan"),
            "coherence_r2": float("nan"),
            "top_eig_frac": float("nan"),
            "participation_rank": float("nan"),
            "entropy_rank": float("nan"),
        }

    X_valid = X[valid]
    X_corr = X_valid / norms[valid, None]
    corr = X_corr @ X_corr.T
    iu = np.triu_indices(corr.shape[0], k=1)
    pair_corr = corr[iu]

    cov = (X_valid @ X_valid.T) / max(X_valid.shape[1] - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    eig_sum = float(eigvals.sum())
    if eig_sum <= 1e-12:
        top_eig_frac = float("nan")
        participation_rank = float("nan")
        entropy_rank = float("nan")
    else:
        p = eigvals / eig_sum
        p_nonzero = p[p > 0]
        top_eig_frac = float(eigvals[-1] / eig_sum)
        participation_rank = float((eig_sum * eig_sum) / np.sum(eigvals * eigvals))
        entropy_rank = float(np.exp(-np.sum(p_nonzero * np.log(p_nonzero))))

    return {
        "coherence": float(pair_corr.mean()),
        "coherence_abs": float(np.abs(pair_corr).mean()),
        "coherence_r2": float(np.square(pair_corr).mean()),
        "top_eig_frac": top_eig_frac,
        "participation_rank": participation_rank,
        "entropy_rank": entropy_rank,
    }


def compute_hidden_by_epoch(
    model_dir: Path,
    keys: list[tuple[str, str]],
    *,
    batch_size: int,
) -> dict[tuple[str, str], np.ndarray]:
    """Return {(rule, epoch): hidden[time, batch, unit]} for requested keys."""
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    from flexsym.core.network import Model
    from flexsym.core.task import generate_trials
    from flexsym.core import tools

    model = Model(str(model_dir), sigma_rec=0)
    by_rule: dict[str, list[str]] = {}
    for rule, epoch in keys:
        by_rule.setdefault(rule, []).append(epoch)

    out: dict[tuple[str, str], np.ndarray] = {}
    with tf.Session() as sess:
        model.restore()
        hp = model.hp
        for rule, epochs in by_rule.items():
            trial = generate_trials(
                rule,
                hp,
                "test",
                noise_on=False,
                batch_size=batch_size,
            )
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            h = sess.run(model.h, feed_dict=feed_dict)
            for epoch in epochs:
                start, stop = trial.epochs[epoch]
                start = 0 if start is None else start
                stop = h.shape[0] if stop is None else stop
                out[(rule, epoch)] = np.asarray(h[start:stop, :, :], dtype=float)
    return out


def plot_coherence(matrix: np.ndarray, cell_sizes: np.ndarray, key_names: list[str], output_dir: Path, variant: str, k: int) -> tuple[Path, Path]:
    order = np.argsort(np.nanmean(matrix, axis=1))[::-1]
    fig, ax = plt.subplots(figsize=(14, max(5.0, 0.2 * matrix.shape[0])), constrained_layout=True)
    finite = matrix[np.isfinite(matrix)]
    vmin = float(np.nanmin(finite)) if finite.size else 0.0
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if vmin < 0:
        cmap = "coolwarm"
        vmin, vmax = -max(abs(vmin), abs(vmax)), max(abs(vmin), abs(vmax))
    else:
        cmap = "viridis"
        vmin = 0.0
    im = ax.imshow(matrix[order], aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"Within-cell activity metric, {variant}, k={k}")
    ax.set_xlabel("task period")
    ax.set_ylabel("structural cell")
    ax.set_xticks(np.arange(len(key_names)))
    ax.set_xticklabels(key_names, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(order, fontsize=7)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(np.arange(matrix.shape[0]))
    ax2.set_yticklabels(cell_sizes[order], fontsize=7)
    ax2.set_ylabel("cell size", rotation=270, labelpad=16)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    png = output_dir / f"structural_cell_activity_coherence_{variant}_k{k}.png"
    pdf = output_dir / f"structural_cell_activity_coherence_{variant}_k{k}.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_focused(matrix: np.ndarray, cell_sizes: np.ndarray, key_names: list[str], engaged_cells: list[int], output_dir: Path, variant: str, k: int, metric_name: str) -> tuple[Path, Path]:
    selected_periods = [
        "delaygo/delay1",
        "delayanti/delay1",
        "delaydm1/stim1",
        "delaydm2/stim1",
        "dmcgo/stim1",
        "dmcnogo/stim1",
        "reactgo/go1",
        "reactanti/go1",
        "delaygo/go1",
        "delayanti/go1",
    ]
    idx = [key_names.index(name) for name in selected_periods if name in key_names]
    rows = np.asarray(engaged_cells, dtype=np.int64)
    sub = matrix[np.ix_(rows, idx)]
    fig, ax = plt.subplots(figsize=(9.5, max(3.2, 0.3 * len(rows))), constrained_layout=True)
    finite = sub[np.isfinite(sub)]
    vmin = float(np.nanmin(finite)) if finite.size else 0.0
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if vmin < 0:
        cmap = "coolwarm"
        vmin, vmax = -max(abs(vmin), abs(vmax)), max(abs(vmin), abs(vmax))
    else:
        cmap = "viridis"
        vmin = 0.0
    im = ax.imshow(sub, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{metric_name} in engaged multi-unit cells, {variant}, k={k}")
    ax.set_xlabel("task period")
    ax.set_ylabel("structural cell")
    ax.set_xticks(np.arange(len(idx)))
    ax.set_xticklabels([key_names[i] for i in idx], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([f"{c} (n={cell_sizes[c]})" for c in rows], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=metric_name)
    png = output_dir / f"structural_cell_activity_{metric_name}_focused_{variant}_k{k}.png"
    pdf = output_dir / f"structural_cell_activity_{metric_name}_focused_{variant}_k{k}.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--activity-clusters", required=True)
    parser.add_argument("--structural-sweep", required=True)
    parser.add_argument("--cell-variance", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, default=33)
    parser.add_argument("--min-cell-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    activity = np.load(args.activity_clusters, allow_pickle=True)
    keys_raw = [tuple(x) for x in np.asarray(activity["keys"], dtype=object)]
    key_names = [key_name(x) for x in keys_raw]

    labels, ind_active, variant = load_structural_labels(Path(args.structural_sweep), args.k)
    if labels.size != ind_active.size:
        raise ValueError("labels and active index lengths differ")
    k = args.k
    cell_sizes = np.bincount(labels, minlength=k)
    valid_cells = np.where(cell_sizes >= args.min_cell_size)[0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_by_epoch = compute_hidden_by_epoch(model_dir, keys_raw, batch_size=args.batch_size)
    metric_names = [
        "coherence",
        "coherence_abs",
        "coherence_r2",
        "top_eig_frac",
        "participation_rank",
        "entropy_rank",
    ]
    metric_matrices = {
        name: np.full((k, len(keys_raw)), np.nan, dtype=float) for name in metric_names
    }
    for period_idx, key in enumerate(keys_raw):
        h = hidden_by_epoch[key][:, :, ind_active]
        # active units x flattened(time, batch)
        unit_obs = np.transpose(h, (2, 0, 1)).reshape(h.shape[2], -1)
        for cell in valid_cells:
            unit_idx = np.where(labels == cell)[0]
            metrics = cell_activity_metrics(unit_obs[unit_idx])
            for name in metric_names:
                metric_matrices[name][cell, period_idx] = metrics[name]

    coherence = metric_matrices["coherence"]

    npz_path = output_dir / f"structural_cell_activity_coherence_{variant}_k{k}.npz"
    np.savez(
        npz_path,
        **metric_matrices,
        cell_sizes=cell_sizes,
        valid_cells=valid_cells,
        labels_struct=labels,
        ind_active=ind_active,
        keys=np.asarray(keys_raw, dtype=object),
        key_names=np.asarray(key_names, dtype=object),
        k=k,
        min_cell_size=args.min_cell_size,
        batch_size=args.batch_size,
        variant=variant,
    )

    csv_path = output_dir / f"structural_cell_activity_coherence_{variant}_k{k}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "struct_cell", "cell_size", *key_names])
        for metric_name in metric_names:
            matrix = metric_matrices[metric_name]
            for cell in range(k):
                writer.writerow([metric_name, cell, int(cell_sizes[cell]), *matrix[cell].tolist()])

    metric_csv_paths = {}
    for metric_name in metric_names:
        metric_csv_path = output_dir / f"structural_cell_activity_{metric_name}_{variant}_k{k}.csv"
        metric_csv_paths[metric_name] = str(metric_csv_path)
        with metric_csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["struct_cell", "cell_size", *key_names])
            for cell in range(k):
                writer.writerow([cell, int(cell_sizes[cell]), *metric_matrices[metric_name][cell].tolist()])

    plot_png, plot_pdf = plot_coherence(coherence, cell_sizes, key_names, output_dir, variant, k)
    abs_plot_png, abs_plot_pdf = plot_coherence(
        metric_matrices["coherence_abs"], cell_sizes, key_names, output_dir, f"{variant}_abs", k
    )
    eig_plot_png, eig_plot_pdf = plot_coherence(
        metric_matrices["top_eig_frac"], cell_sizes, key_names, output_dir, f"{variant}_top_eig_frac", k
    )

    # Focus on multi-unit cells active in both MemoryPro/MemoryAnti memory periods.
    # task.py monkey-patches np.load after hidden-state generation, forcing
    # allow_pickle=True. Avoid passing it explicitly from here onward.
    cell_var = np.load(args.cell_variance)["cell_epoch_variance"]
    mem_idx = [key_names.index("delaygo/delay1"), key_names.index("delayanti/delay1")]
    engaged = [
        int(c)
        for c in valid_cells
        if np.all(cell_var[c, mem_idx] >= np.percentile(cell_var[valid_cells][:, mem_idx], 60, axis=0))
    ]
    if not engaged:
        engaged = [int(c) for c in valid_cells[np.argsort(cell_var[valid_cells][:, mem_idx].mean(axis=1))[::-1][:8]]]
    focused_png, focused_pdf = plot_focused(coherence, cell_sizes, key_names, engaged, output_dir, variant, k, "coherence")
    focused_abs_png, focused_abs_pdf = plot_focused(
        metric_matrices["coherence_abs"], cell_sizes, key_names, engaged, output_dir, variant, k, "coherence_abs"
    )
    focused_eig_png, focused_eig_pdf = plot_focused(
        metric_matrices["top_eig_frac"], cell_sizes, key_names, engaged, output_dir, variant, k, "top_eig_frac"
    )

    comparisons = []
    compare_pairs = [
        ("memory_pro_vs_anti_shared", "delaygo/delay1", "delayanti/delay1"),
        ("memorypro_vs_delaydm1_stim_different", "delaygo/delay1", "delaydm1/stim1"),
        ("memorypro_vs_dmcgo_stim_different", "delaygo/delay1", "dmcgo/stim1"),
        ("reaction_go_vs_anti_shared_response", "reactgo/go1", "reactanti/go1"),
        ("reactgo_vs_delaygo_response_different", "reactgo/go1", "delaygo/go1"),
    ]
    for cell in valid_cells:
        for label, a, b in compare_pairs:
            ia = key_names.index(a)
            ib = key_names.index(b)
            comparisons.append(
                {
                    "struct_cell": int(cell),
                    "cell_size": int(cell_sizes[cell]),
                    "comparison": label,
                    "period_a": a,
                    "period_b": b,
                    "variance_a": float(cell_var[cell, ia]),
                    "variance_b": float(cell_var[cell, ib]),
                    **{
                        f"{metric_name}_a": float(metric_matrices[metric_name][cell, ia])
                        for metric_name in metric_names
                    },
                    **{
                        f"{metric_name}_b": float(metric_matrices[metric_name][cell, ib])
                        for metric_name in metric_names
                    },
                    **{
                        f"{metric_name}_abs_delta": float(
                            abs(metric_matrices[metric_name][cell, ia] - metric_matrices[metric_name][cell, ib])
                        )
                        for metric_name in metric_names
                    },
                }
            )
    comparison_csv = output_dir / f"structural_cell_activity_coherence_comparisons_{variant}_k{k}.csv"
    with comparison_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0].keys()))
        writer.writeheader()
        writer.writerows(comparisons)

    summary = {
        "model_dir": str(model_dir),
        "variant": variant,
        "k": k,
        "min_cell_size": args.min_cell_size,
        "batch_size": args.batch_size,
        "n_valid_cells": int(len(valid_cells)),
        "valid_cells": valid_cells.astype(int).tolist(),
        "memory_engaged_cells": engaged,
        "npz_path": str(npz_path),
        "csv_path": str(csv_path),
        "metric_csv_paths": metric_csv_paths,
        "comparison_csv": str(comparison_csv),
        "heatmap_png": str(plot_png),
        "heatmap_pdf": str(plot_pdf),
        "abs_heatmap_png": str(abs_plot_png),
        "abs_heatmap_pdf": str(abs_plot_pdf),
        "top_eig_frac_heatmap_png": str(eig_plot_png),
        "top_eig_frac_heatmap_pdf": str(eig_plot_pdf),
        "focused_png": str(focused_png),
        "focused_pdf": str(focused_pdf),
        "focused_abs_png": str(focused_abs_png),
        "focused_abs_pdf": str(focused_abs_pdf),
        "focused_top_eig_frac_png": str(focused_eig_png),
        "focused_top_eig_frac_pdf": str(focused_eig_pdf),
    }
    with (output_dir / f"structural_cell_activity_coherence_summary_{variant}_k{k}.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
