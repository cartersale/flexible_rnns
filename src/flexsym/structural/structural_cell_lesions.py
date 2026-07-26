"""Behavioral lesion test for structural cells (signed-WL equitable partition).

This is the structural-cell analogue of the activity-cluster lesion test in
``stepnet/analysis/clustering.py`` (``Analysis.lesions``). For a trained model it
ablates each structural cell from the signed Weisfeiler-Lehman partition of
``W_rec`` (via ``Model.lesion_units``) and measures the per-task performance and
cost change relative to the intact network.

To judge whether structural cells are *more task-specific / critical than chance*
it runs two baselines on the same model:

* **activity clusters** at the same ``k`` (already stored in the sweep ``.npz``)
* **size-matched random partitions** (a null whose cell-size multiset matches the
  structural partition).

Inputs are a trained ``model_dir`` (requires the full TF checkpoint, not just
``model_params.npz``) and a signed-WL sweep ``.npz`` (e.g. the materialized
``transpose_selected_k/signed_input_wl_sweep.npz``), mirroring how
``shapley_cell_importance.py`` takes both.
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
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


from flexsym.core import tools  # noqa: E402
from flexsym.core.network import Model, get_perf  # noqa: E402
from flexsym.core.task import generate_trials  # noqa: E402

PARTITION_TYPES = ("structural", "activity", "random")


def load_partitions(
    sweep_path: Path, k: int | None
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load structural + activity labels at a chosen ``k`` from a signed-WL sweep.

    Mirrors ``structural_cell_activity_coherence.load_structural_labels`` but also
    returns the activity labels for the same ``k``. ``labels`` are over
    ``ind_active``; values are ``0..k-1``.
    """
    # NB: importing ``task`` monkeypatches ``np.load`` to force allow_pickle=True,
    # so we must not pass allow_pickle here (it would collide).
    sweep = np.load(sweep_path)
    rows = np.asarray(sweep["rows"], dtype=object)
    row_columns = [str(x) for x in np.asarray(sweep["row_columns"], dtype=object)]
    k_idx = row_columns.index("k")
    k_min = int(rows[0, k_idx])
    k_max = int(rows[-1, k_idx])
    if k is None:
        k = int(sweep["chosen_k"])
    if k < k_min or k > k_max:
        raise ValueError(f"k={k} outside sweep range [{k_min}, {k_max}]")
    labels_struct = np.asarray(sweep["labels_struct_by_k"][k - k_min], dtype=np.int64)
    labels_act = np.asarray(sweep["labels_act_by_k"][k - k_min], dtype=np.int64)
    ind_active = np.asarray(sweep["ind_active"], dtype=np.int64)
    variant = str(np.asarray(sweep["matrix_variant"]).item())
    return k, labels_struct, labels_act, ind_active, variant


def build_groups(
    labels_struct: np.ndarray,
    labels_act: np.ndarray,
    ind_active: np.ndarray,
    n_random: int,
    seed: int,
) -> list[dict]:
    """Build the list of lesion groups (structural, activity, size-matched random).

    Each group is ``{type, group_id, source_partition, units}`` where ``units`` are
    **original n_rnn indices** ready to pass to ``Model.lesion_units``.
    """
    groups: list[dict] = []

    for c in sorted(np.unique(labels_struct)):
        groups.append(
            {
                "type": "structural",
                "group_id": int(c),
                "source_partition": 0,
                "units": ind_active[labels_struct == c],
            }
        )

    for c in sorted(np.unique(labels_act)):
        groups.append(
            {
                "type": "activity",
                "group_id": int(c),
                "source_partition": 0,
                "units": ind_active[labels_act == c],
            }
        )

    struct_sizes = [int((labels_struct == c).sum()) for c in sorted(np.unique(labels_struct))]
    rng = np.random.default_rng(seed)
    for r in range(n_random):
        perm = rng.permutation(ind_active)
        start = 0
        for gi, sz in enumerate(struct_sizes):
            grp = perm[start : start + sz]
            start += sz
            groups.append(
                {
                    "type": "random",
                    "group_id": gi,
                    "source_partition": r,
                    "units": grp,
                }
            )

    return groups


def evaluate_condition(
    model: Model,
    sess: tf.Session,
    hp: dict,
    rules: list[str],
    units,
    n_rep: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Restore intact weights, lesion ``units``, then measure per-rule perf and cost.

    ``units=None`` evaluates the intact network. Restoring from the checkpoint each
    call resets the previous lesion, so one graph/session is reused for all
    conditions (faster than rebuilding the graph as ``clustering.lesions`` does).
    """
    model.restore()
    model.lesion_units(sess, units)

    n_rep = max(1, int(n_rep))
    batch_size_rep = max(1, int(batch_size) // n_rep)
    perfs = np.zeros(len(rules))
    costs = np.zeros(len(rules))
    for ri, rule in enumerate(rules):
        perf_tmp = []
        cost_tmp = []
        for _ in range(n_rep):
            trial = generate_trials(rule, hp, "random", batch_size=batch_size_rep)
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            y_hat, c_lsq = sess.run([model.y_hat, model.cost_lsq], feed_dict=feed_dict)
            perf_tmp.append(float(np.mean(get_perf(y_hat, trial.y_loc))))
            cost_tmp.append(float(c_lsq))
        perfs[ri] = np.mean(perf_tmp)
        costs[ri] = np.mean(cost_tmp)
    return perfs, costs


def _stack(records: list[dict], key: str) -> np.ndarray:
    if not records:
        return np.zeros((0, 0))
    return np.asarray([rec[key] for rec in records], dtype=float)


def _save_heatmap(perf_changes: np.ndarray, rules: list[str], title: str, path_stem: Path) -> None:
    if perf_changes.shape[0] == 0:
        return
    fig, ax = plt.subplots(figsize=(max(3.0, 0.16 * perf_changes.shape[0] + 1.5), 3.5))
    im = ax.imshow(
        perf_changes.T,
        cmap="coolwarm",
        aspect="auto",
        interpolation="nearest",
        vmin=-0.5,
        vmax=0.5,
    )
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels(rules, fontsize=6)
    ax.set_xlabel(f"{title} (group index)")
    ax.set_title(f"Performance change after lesioning\n{title}", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85, ticks=[-0.5, 0.0, 0.5], label="perf change")
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"), dpi=200)
    fig.savefig(path_stem.with_suffix(".pdf"))
    plt.close(fig)


def _save_specificity(
    max_drop_by_type: dict[str, np.ndarray], path_stem: Path
) -> None:
    types = [t for t in PARTITION_TYPES if max_drop_by_type.get(t, np.zeros(0)).size]
    if not types:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    data = [max_drop_by_type[t] for t in types]
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(types) + 1))
    ax.set_xticklabels(types)
    ax.set_ylabel("max single-task performance drop per group")
    ax.set_title("Lesion specificity: structural vs activity vs random")
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"), dpi=200)
    fig.savefig(path_stem.with_suffix(".pdf"))
    plt.close(fig)


def run_structural_cell_lesions(
    model_dir: str | os.PathLike[str],
    structural_sweep: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str] | None = None,
    k: int | None = None,
    n_rep: int = 16,
    batch_size: int = 256,
    n_random: int = 10,
    seed: int = 0,
) -> dict:
    model_dir = Path(model_dir)
    sweep_path = Path(structural_sweep)
    if not (model_dir / "model.ckpt.index").exists():
        raise FileNotFoundError(
            f"No TF checkpoint (model.ckpt.*) in {model_dir}. The lesion test needs the "
            "full trained model, not just model_params.npz. Is the data drive mounted?"
        )

    k, labels_struct, labels_act, ind_active, variant = load_partitions(sweep_path, k)

    output_dir = Path(output_dir) if output_dir is not None else sweep_path.parent / "structural_cell_lesions"
    output_dir.mkdir(parents=True, exist_ok=True)

    hp = tools.load_hp(str(model_dir))
    rules = list(hp.get("rule_trains") or hp["rules"])

    groups = build_groups(labels_struct, labels_act, ind_active, n_random, seed)
    print(
        f"k={k} variant={variant}: "
        f"{len(np.unique(labels_struct))} structural cells, "
        f"{len(np.unique(labels_act))} activity clusters, "
        f"{n_random} random partitions -> {len(groups)} lesion conditions"
    )

    model = Model(str(model_dir))
    records: list[dict] = []
    with tf.Session() as sess:
        perfs_intact, costs_intact = evaluate_condition(
            model, sess, hp, rules, None, n_rep, batch_size
        )
        print("intact mean perf:", float(perfs_intact.mean()))
        for i, group in enumerate(groups):
            perfs, costs = evaluate_condition(
                model, sess, hp, rules, group["units"], n_rep, batch_size
            )
            records.append(
                {
                    "type": group["type"],
                    "group_id": group["group_id"],
                    "source_partition": group["source_partition"],
                    "group_size": int(len(group["units"])),
                    "perf_change": perfs - perfs_intact,
                    "cost_change": costs - costs_intact,
                }
            )
            if (i + 1) % 25 == 0:
                print(f"  evaluated {i + 1}/{len(groups)} conditions")

    by_type = {t: [rec for rec in records if rec["type"] == t] for t in PARTITION_TYPES}

    # ---- CSV (long form) -------------------------------------------------
    csv_path = output_dir / "structural_cell_lesions.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["partition_type", "group_id", "source_partition", "group_size", "rule", "perf_change", "cost_change"]
        )
        for rec in records:
            for ri, rule in enumerate(rules):
                writer.writerow(
                    [
                        rec["type"],
                        rec["group_id"],
                        rec["source_partition"],
                        rec["group_size"],
                        rule,
                        f"{rec['perf_change'][ri]:.6f}",
                        f"{rec['cost_change'][ri]:.6f}",
                    ]
                )

    # ---- NPZ -------------------------------------------------------------
    npz_payload = {
        "perfs_intact": perfs_intact,
        "costs_intact": costs_intact,
        "rules": np.asarray(rules, dtype=object),
        "k": k,
        "matrix_variant": variant,
        "ind_active": ind_active,
        "labels_struct": labels_struct,
        "labels_act": labels_act,
    }
    for t in PARTITION_TYPES:
        npz_payload[f"{t}_perf_changes"] = _stack(by_type[t], "perf_change")
        npz_payload[f"{t}_cost_changes"] = _stack(by_type[t], "cost_change")
        npz_payload[f"{t}_group_sizes"] = np.asarray([rec["group_size"] for rec in by_type[t]], dtype=np.int64)
        npz_payload[f"{t}_group_ids"] = np.asarray([rec["group_id"] for rec in by_type[t]], dtype=np.int64)
        npz_payload[f"{t}_source_partition"] = np.asarray(
            [rec["source_partition"] for rec in by_type[t]], dtype=np.int64
        )
    npz_path = output_dir / "structural_cell_lesions.npz"
    np.savez(npz_path, **npz_payload)

    # ---- Summary metrics -------------------------------------------------
    # drop = -perf_change (positive = performance lost by lesioning)
    drop_by_type = {t: -_stack(by_type[t], "perf_change") for t in PARTITION_TYPES}
    max_drop_by_type = {
        t: (drop_by_type[t].max(axis=1) if drop_by_type[t].size else np.zeros(0))
        for t in PARTITION_TYPES
    }
    total_drop_by_type = {
        t: (drop_by_type[t].sum(axis=1) if drop_by_type[t].size else np.zeros(0))
        for t in PARTITION_TYPES
    }

    struct_drop = drop_by_type["structural"]
    rand_drop = drop_by_type["random"]
    per_rule_contrast = []
    for ri, rule in enumerate(rules):
        struct_mean = float(struct_drop[:, ri].mean()) if struct_drop.size else float("nan")
        if rand_drop.size:
            null_mean = float(rand_drop[:, ri].mean())
            null_sd = float(rand_drop[:, ri].std(ddof=1)) if rand_drop.shape[0] > 1 else 0.0
        else:
            null_mean = null_sd = float("nan")
        z = (struct_mean - null_mean) / null_sd if null_sd and np.isfinite(null_sd) and null_sd > 0 else float("nan")
        per_rule_contrast.append(
            {
                "rule": rule,
                "structural_mean_drop": struct_mean,
                "random_null_mean_drop": null_mean,
                "random_null_sd_drop": null_sd,
                "z_structural_vs_random": z,
            }
        )

    struct_records = by_type["structural"]
    top_total = sorted(
        range(len(struct_records)),
        key=lambda i: total_drop_by_type["structural"][i],
        reverse=True,
    )[:5]
    top_max = sorted(
        range(len(struct_records)),
        key=lambda i: max_drop_by_type["structural"][i],
        reverse=True,
    )[:5]

    def _cell_summary(i: int) -> dict:
        rec = struct_records[i]
        drop = drop_by_type["structural"][i]
        worst = int(np.argmax(drop))
        return {
            "cell_id": rec["group_id"],
            "cell_size": rec["group_size"],
            "total_perf_drop": float(total_drop_by_type["structural"][i]),
            "max_perf_drop": float(max_drop_by_type["structural"][i]),
            "worst_rule": rules[worst],
        }

    summary = {
        "model_dir": str(model_dir),
        "structural_sweep": str(sweep_path),
        "output_dir": str(output_dir),
        "csv_path": str(csv_path),
        "npz_path": str(npz_path),
        "k": k,
        "matrix_variant": variant,
        "n_rules": len(rules),
        "n_rep": n_rep,
        "batch_size": batch_size,
        "n_random": n_random,
        "seed": seed,
        "n_groups": {t: len(by_type[t]) for t in PARTITION_TYPES},
        "intact_mean_perf": float(perfs_intact.mean()),
        "intact_perf_by_rule": {rule: float(perfs_intact[ri]) for ri, rule in enumerate(rules)},
        "mean_max_drop": {
            t: (float(max_drop_by_type[t].mean()) if max_drop_by_type[t].size else float("nan"))
            for t in PARTITION_TYPES
        },
        "median_max_drop": {
            t: (float(np.median(max_drop_by_type[t])) if max_drop_by_type[t].size else float("nan"))
            for t in PARTITION_TYPES
        },
        "top_structural_cells_by_total_drop": [_cell_summary(i) for i in top_total],
        "top_structural_cells_by_max_drop": [_cell_summary(i) for i in top_max],
        "per_rule_structural_vs_random": per_rule_contrast,
    }
    summary_path = output_dir / "structural_cell_lesions_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # ---- Figures ---------------------------------------------------------
    _save_heatmap(
        npz_payload["structural_perf_changes"], rules, "structural cells", output_dir / "lesion_heatmap_structural"
    )
    _save_heatmap(
        npz_payload["activity_perf_changes"], rules, "activity clusters", output_dir / "lesion_heatmap_activity"
    )
    _save_specificity(max_drop_by_type, output_dir / "lesion_specificity_struct_vs_act_vs_random")

    print(json.dumps({k: summary[k] for k in ("k", "intact_mean_perf", "mean_max_drop", "n_groups")}, indent=2))
    print(f"Wrote: {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavioral lesion test for structural cells vs activity clusters and a size-matched random null."
    )
    parser.add_argument("model_dir", help="Trained model directory (must contain model.ckpt.*).")
    parser.add_argument("--structural-sweep", required=True, help="Path to signed_input_wl_sweep.npz.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--k", type=int, default=None, help="Number of cells (default: chosen_k from the sweep).")
    parser.add_argument("--n-rep", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--n-random",
        type=int,
        default=10,
        help="Random size-matched partitions for the null. Dominates runtime (~n_random*k extra evals).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_structural_cell_lesions(
        args.model_dir,
        args.structural_sweep,
        output_dir=args.output_dir,
        k=args.k,
        n_rep=args.n_rep,
        batch_size=args.batch_size,
        n_random=args.n_random,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
