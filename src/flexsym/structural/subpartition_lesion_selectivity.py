"""Lesion functional subpartitions inside signed-WL structural cells.

This tests a sharper causal question than whole structural-cell lesions:
given a structural role cell, do task-epoch functional subpartitions inside
that cell produce more task-selective behavioral deficits than same-size
random subsets drawn from the same role cell?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

STEP_DIR = Path(__file__).resolve().parent
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

import tools  # noqa: E402
from network import Model  # noqa: E402
from structural_cell_activity_coherence import compute_hidden_by_epoch  # noqa: E402
from structural_cell_lesions import evaluate_condition  # noqa: E402
from structural_cell_subpartition_pilot import choose_functional_partition  # noqa: E402


DEFAULT_PERIODS = [
    ("delaygo", "delay1"),
    ("delayanti", "delay1"),
    ("delaydm1", "stim1"),
    ("delaydm2", "stim1"),
    ("contextdelaydm1", "stim1"),
    ("contextdelaydm2", "stim1"),
    ("multidelaydm", "stim1"),
    ("dmcgo", "stim1"),
    ("dmcnogo", "stim1"),
    ("dmsgo", "stim1"),
    ("dmsnogo", "stim1"),
]

TASK_FAMILIES = {
    "go": {"fdgo", "reactgo", "delaygo"},
    "anti": {"fdanti", "reactanti", "delayanti"},
    "decision_context": {
        "delaydm1",
        "delaydm2",
        "contextdelaydm1",
        "contextdelaydm2",
        "multidelaydm",
    },
    "dms": {"dmsgo", "dmsnogo"},
    "dmc": {"dmcgo", "dmcnogo"},
}


def load_sweep_labels(sweep_path: Path, k: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Load structural labels without tripping the repo's np.load monkeypatch."""
    sweep = np.load(sweep_path)
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


def parse_period(value: str) -> tuple[str, str]:
    if "/" in value:
        rule, epoch = value.split("/", 1)
    elif ":" in value:
        rule, epoch = value.split(":", 1)
    else:
        raise argparse.ArgumentTypeError("period must be formatted as rule/epoch")
    return rule, epoch


def infer_family(rule: str) -> str:
    for family, rules in TASK_FAMILIES.items():
        if rule in rules:
            return family
    return "other"


def task_selectivity(drops_by_rule: dict[str, float], rules: list[str]) -> dict[str, object]:
    drops = np.asarray([max(0.0, float(drops_by_rule.get(rule, 0.0))) for rule in rules], dtype=float)
    total = float(drops.sum())
    max_drop = float(drops.max()) if drops.size else 0.0
    mean_drop = float(drops.mean()) if drops.size else 0.0
    worst_rule = rules[int(np.argmax(drops))] if drops.size else ""
    if total > 0:
        p = drops / total
        p_nonzero = p[p > 0]
        entropy_norm = float(-(p_nonzero * np.log(p_nonzero)).sum() / math.log(len(drops)))
        top1_share = float(drops.max() / total)
        top3_share = float(np.sort(drops)[-3:].sum() / total)
        participation_ratio = float(total * total / np.square(drops).sum())
    else:
        entropy_norm = 0.0
        top1_share = 0.0
        top3_share = 0.0
        participation_ratio = 0.0

    family_drops = {
        family: float(sum(drops_by_rule.get(rule, 0.0) for rule in family_rules))
        for family, family_rules in TASK_FAMILIES.items()
    }
    best_family = max(family_drops, key=family_drops.get)
    best_family_drop = family_drops[best_family]
    best_family_share = best_family_drop / total if total > 0 else 0.0
    expected_family = infer_family(worst_rule)
    expected_family_drop = family_drops.get(expected_family, 0.0)

    return {
        "total_drop": total,
        "max_drop": max_drop,
        "mean_drop": mean_drop,
        "max_over_mean": max_drop / mean_drop if mean_drop > 0 else 0.0,
        "worst_rule": worst_rule,
        "top1_share": top1_share,
        "top3_share": top3_share,
        "entropy_selectivity": 1.0 - entropy_norm,
        "participation_ratio_tasks": participation_ratio,
        "best_family": best_family,
        "best_family_drop": best_family_drop,
        "best_family_share": best_family_share,
        "worst_rule_family": expected_family,
        "worst_rule_family_drop": expected_family_drop,
        "worst_rule_family_share": expected_family_drop / total if total > 0 else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {"n_conditions": len(rows), "by_condition_type": {}}
    for condition_type in sorted({str(r["condition_type"]) for r in rows}):
        sub = [r for r in rows if r["condition_type"] == condition_type]
        out["by_condition_type"][condition_type] = {}
        for metric in (
            "total_drop",
            "max_drop",
            "top1_share",
            "top3_share",
            "entropy_selectivity",
            "best_family_share",
            "worst_rule_family_share",
            "participation_ratio_tasks",
        ):
            vals = np.asarray([float(r[metric]) for r in sub], dtype=float)
            out["by_condition_type"][condition_type][metric] = {
                "mean": float(vals.mean()) if vals.size else float("nan"),
                "median": float(np.median(vals)) if vals.size else float("nan"),
                "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            }

    random_rows = [r for r in rows if r["condition_type"] == "within_cell_random"]
    for metric in ("top1_share", "entropy_selectivity", "best_family_share", "worst_rule_family_share"):
        rand = np.asarray([float(r[metric]) for r in random_rows], dtype=float)
        if rand.size > 1 and rand.std(ddof=1) > 0:
            mu = float(rand.mean())
            sd = float(rand.std(ddof=1))
            for row in rows:
                if row["condition_type"] == "functional_subpartition":
                    row[f"z_vs_within_cell_random_{metric}"] = (float(row[metric]) - mu) / sd
        else:
            for row in rows:
                if row["condition_type"] == "functional_subpartition":
                    row[f"z_vs_within_cell_random_{metric}"] = float("nan")

    top = sorted(
        [r for r in rows if r["condition_type"] == "functional_subpartition"],
        key=lambda r: (float(r["entropy_selectivity"]), float(r["total_drop"])),
        reverse=True,
    )[:10]
    out["top_functional_subpartitions_by_entropy_selectivity"] = top
    return out


def run_subpartition_lesion_selectivity(
    model_dir: str | os.PathLike[str],
    structural_sweep: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str],
    k: int,
    cells: list[int],
    periods: list[tuple[str, str]],
    n_rep: int,
    batch_size: int,
    hidden_batch_size: int,
    max_subclusters: int,
    n_random: int,
    seed: int,
) -> dict[str, object]:
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels, ind_active, variant = load_sweep_labels(Path(structural_sweep), k)
    hp = tools.load_hp(str(model_dir))
    if hp is None:
        raise FileNotFoundError(f"No hp.json or hparams.json found in {model_dir}")
    rules = list(hp.get("rule_trains") or hp["rules"])
    rng = np.random.default_rng(seed)

    hidden = compute_hidden_by_epoch(model_dir, periods, batch_size=hidden_batch_size)
    rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []

    model = Model(str(model_dir))
    with tf.Session() as sess:
        perfs_intact, costs_intact = evaluate_condition(model, sess, hp, rules, None, n_rep, batch_size)
        condition_index = 0
        for cell in cells:
            unit_positions = np.where(labels == int(cell))[0]
            unit_ids = ind_active[unit_positions]
            if unit_ids.size < 3:
                print(f"Skipping cell {cell}: only {unit_ids.size} units")
                continue
            for rule_for_partition, epoch_for_partition in periods:
                h = hidden[(rule_for_partition, epoch_for_partition)][:, :, ind_active][:, :, unit_positions]
                X = np.transpose(h, (2, 0, 1)).reshape(h.shape[2], -1)
                partition = choose_functional_partition(X, max_k=max_subclusters)
                sub_labels = np.asarray(partition["labels"], dtype=np.int64)
                if int(partition["chosen_k"]) < 2:
                    print(f"Skipping cell {cell} {rule_for_partition}/{epoch_for_partition}: no functional split")
                    continue

                subgroups: list[tuple[str, int, np.ndarray]] = []
                for sub_id in sorted(np.unique(sub_labels)):
                    sub_units = unit_ids[sub_labels == sub_id]
                    if sub_units.size:
                        subgroups.append(("functional_subpartition", int(sub_id), sub_units))
                        for random_i in range(n_random):
                            random_units = rng.choice(unit_ids, size=sub_units.size, replace=False)
                            subgroups.append(("within_cell_random", random_i, random_units))

                for condition_type, subgroup_id, units in subgroups:
                    perfs, costs = evaluate_condition(model, sess, hp, rules, units, n_rep, batch_size)
                    drops_by_rule = {
                        rule: max(0.0, float(perfs_intact[ri] - perfs[ri]))
                        for ri, rule in enumerate(rules)
                    }
                    metrics = task_selectivity(drops_by_rule, rules)
                    row = {
                        "condition_index": condition_index,
                        "condition_type": condition_type,
                        "cell": int(cell),
                        "cell_size": int(unit_ids.size),
                        "partition_rule": rule_for_partition,
                        "partition_epoch": epoch_for_partition,
                        "partition_family": infer_family(rule_for_partition),
                        "functional_k": int(partition["chosen_k"]),
                        "functional_silhouette": float(partition["best_silhouette"]),
                        "subgroup_id": subgroup_id,
                        "subgroup_size": int(units.size),
                        "units": " ".join(str(int(u)) for u in np.sort(units)),
                        **metrics,
                    }
                    rows.append(row)
                    for ri, rule in enumerate(rules):
                        long_rows.append(
                            {
                                "condition_index": condition_index,
                                "condition_type": condition_type,
                                "cell": int(cell),
                                "partition_rule": rule_for_partition,
                                "partition_epoch": epoch_for_partition,
                                "subgroup_id": subgroup_id,
                                "subgroup_size": int(units.size),
                                "rule": rule,
                                "perf_intact": float(perfs_intact[ri]),
                                "perf_lesioned": float(perfs[ri]),
                                "perf_drop": float(perfs_intact[ri] - perfs[ri]),
                                "cost_change": float(costs[ri] - costs_intact[ri]),
                            }
                        )
                    condition_index += 1
                    if condition_index % 25 == 0:
                        print(f"  evaluated {condition_index} subpartition/random conditions")

    summary = summarize(rows)
    summary.update(
        {
            "model_dir": str(model_dir),
            "structural_sweep": str(structural_sweep),
            "output_dir": str(output_dir),
            "k": k,
            "matrix_variant": variant,
            "cells": cells,
            "periods": [f"{rule}/{epoch}" for rule, epoch in periods],
            "n_rep": n_rep,
            "batch_size": batch_size,
            "hidden_batch_size": hidden_batch_size,
            "n_random": n_random,
            "seed": seed,
            "intact_mean_perf": float(np.mean(perfs_intact)),
        }
    )

    metrics_csv = output_dir / "subpartition_lesion_selectivity_metrics.csv"
    long_csv = output_dir / "subpartition_lesion_selectivity_long.csv"
    summary_json = output_dir / "subpartition_lesion_selectivity_summary.json"
    write_csv(metrics_csv, rows)
    write_csv(long_csv, long_rows)
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"Wrote: {summary_json}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--structural-sweep", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, default=33)
    parser.add_argument("--cells", type=int, nargs="+", default=[14, 17, 29, 30])
    parser.add_argument("--period", type=parse_period, action="append", dest="periods")
    parser.add_argument("--n-rep", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-batch-size", type=int, default=512)
    parser.add_argument("--max-subclusters", type=int, default=6)
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_subpartition_lesion_selectivity(
        args.model_dir,
        args.structural_sweep,
        output_dir=args.output_dir,
        k=args.k,
        cells=args.cells,
        periods=args.periods or DEFAULT_PERIODS,
        n_rep=args.n_rep,
        batch_size=args.batch_size,
        hidden_batch_size=args.hidden_batch_size,
        max_subclusters=args.max_subclusters,
        n_random=args.n_random,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
