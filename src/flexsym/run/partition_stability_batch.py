"""Batch driver for signed-WL partition stability (reviewer concern #5).

Runs ``structural_partition_stability.analyze_model`` for every LeakyRNN softplus
diag network that has activity clusters, at each model's operating k plus probe
k's (0.5x, 2x), across both hard and soft back-ends. Aggregates one row per
(model, k, backend) into a single CSV plus a per-size summary so we can state
whether the partition is reproducible before trusting anything downstream.

Operating k is read from the batch k-selection summaries (active-fraction
``selected_k``). Model directories are resolved from the original processed
store (seeds 0-2) and the staged new-run symlinks (seeds 3-9).
"""

from __future__ import annotations

from flexsym import paths

import argparse
import csv
import json
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np

from flexsym.structural.structural_partition_stability import analyze_model

REPO_DIR = paths.REPO_ROOT
ACTIVITY_ROOT = REPO_DIR / "results" / "activity_clusters_paper"
ACTIVITY_NAME = "epoch_variance_unit_clusters_paper_max.npz"

ORIGINAL_STORE = paths.ORIGINAL_STORE
STAGED_STORE = paths.STAGED_STORE

K_SUMMARIES = [
    REPO_DIR
    / "results/signed_input_wl_paper/batch_rnn_softplus_diag_k20pct/k_selection_summary.csv",
    REPO_DIR
    / "results/signed_input_wl_paper/batch_rnn_softplus_diag_newseeds/k_selection_summary.csv",
]


def load_operating_k() -> dict[str, dict[str, int]]:
    info: dict[str, dict[str, int]] = {}
    for summary in K_SUMMARIES:
        if not summary.exists():
            continue
        with summary.open() as f:
            for row in csv.DictReader(f):
                n_active = int(row["n_active"])
                # Older summaries (k20pct) predate the explicit selected_k column;
                # reconstruct it the way the runner does: min(activity_k, structural_k).
                if row.get("selected_k"):
                    selected_k = int(row["selected_k"])
                else:
                    selected_k = min(int(row["activity_k"]), int(row["structural_k"]))
                    selected_k = max(int(row["k_min"]), min(selected_k, n_active - 1))
                info[row["model"]] = {
                    "selected_k": selected_k,
                    "n_active": n_active,
                }
    return info


def resolve_model_dir(model: str) -> Path | None:
    for store in (ORIGINAL_STORE, STAGED_STORE):
        candidate = store / model
        if candidate.exists():
            return candidate
    return None


def probe_k_list(selected_k: int, n_active: int) -> list[int]:
    candidates = {
        max(3, int(round(0.5 * selected_k))),
        int(selected_k),
        min(int(round(2.0 * selected_k)), n_active - 1),
    }
    return sorted(k for k in candidates if 3 <= k <= n_active - 1)


def size_of(model: str) -> int:
    return int(model.split("_")[3])


def seed_of(model: str) -> int:
    return int(model.split("_")[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_DIR / "results/signed_input_wl_paper/partition_stability"),
    )
    parser.add_argument("--matrix-variant", default="transpose")
    parser.add_argument("--n-restarts", type=int, default=20)
    parser.add_argument("--soft-restarts", type=int, default=10)
    parser.add_argument("--models", nargs="*", default=None, help="Optional filter.")
    parser.add_argument("--no-soft", action="store_true", help="Hard back-end only.")
    parser.add_argument(
        "--operating-k-only",
        action="store_true",
        help="Only evaluate each model's operating k (skip the 0.5x/2x probes).",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    backends = ["hard"] if args.no_soft else ["hard", "soft"]

    op_k = load_operating_k()
    models = sorted(op_k, key=lambda m: (size_of(m), seed_of(m)))
    if args.models:
        wanted = set(args.models)
        models = [m for m in models if m in wanted]

    all_rows: list[dict[str, Any]] = []
    for model in models:
        activity = ACTIVITY_ROOT / model / ACTIVITY_NAME
        model_dir = resolve_model_dir(model)
        if not activity.exists() or model_dir is None:
            print(f"SKIP {model}: activity={activity.exists()} dir={model_dir}")
            continue

        if args.operating_k_only:
            k_list = [op_k[model]["selected_k"]]
        else:
            k_list = probe_k_list(op_k[model]["selected_k"], op_k[model]["n_active"])
        model_out = out_root / model
        done_marker = model_out / "partition_stability_summary.json"
        if done_marker.exists() and not args.force:
            summary = json.loads(done_marker.read_text())
            print(f"skip (done) {model}")
        else:
            print(f"RUN {model}  k_list={k_list}  n_active={op_k[model]['n_active']}")
            summary = analyze_model(
                model_dir,
                activity,
                k_list=k_list,
                matrix_variant=args.matrix_variant,
                backends=backends,
                n_restarts=args.n_restarts,
                n_iter=20,
                n_init=20,
                soft_restarts=args.soft_restarts,
                soft_n_init=5,
                output_dir=model_out,
            )
        for row in summary["rows"]:
            row = dict(row)
            row["size"] = size_of(model)
            row["seed"] = seed_of(model)
            row["selected_k"] = op_k[model]["selected_k"]
            row["is_operating_k"] = int(row["k"] == op_k[model]["selected_k"])
            all_rows.append(row)

    if not all_rows:
        raise SystemExit("No models analysed.")

    fieldnames = sorted({key for row in all_rows for key in row})
    agg_csv = out_root / "partition_stability_all.csv"
    with agg_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    # per-size summary at operating k, hard back-end
    per_size: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in all_rows:
        if row["is_operating_k"]:
            per_size.setdefault((row["size"], row["backend"]), []).append(row)
    size_rows = []
    for (size, backend), rows in sorted(per_size.items()):
        ari_means = np.array([r["ari_mean"] for r in rows], dtype=float)
        ari_mins = np.array([r["ari_min"] for r in rows], dtype=float)
        max_cell = np.array([r["max_cell_fraction"] for r in rows], dtype=float)
        frac_singleton = np.array([r["frac_units_in_singletons"] for r in rows], dtype=float)
        str_ratio = np.array(
            [r["singleton_strength_ratio"] for r in rows], dtype=float
        )
        size_rows.append(
            {
                "size": size,
                "backend": backend,
                "n_networks": len(rows),
                "restart_ari_mean": float(np.nanmean(ari_means)),
                "restart_ari_mean_sd": float(np.nanstd(ari_means, ddof=1)) if len(rows) > 1 else 0.0,
                "restart_ari_min_worst": float(np.nanmin(ari_mins)),
                "max_cell_fraction_mean": float(np.nanmean(max_cell)),
                "frac_units_in_singletons_mean": float(np.nanmean(frac_singleton)),
                "singleton_strength_ratio_mean": float(np.nanmean(str_ratio)),
            }
        )
    size_csv = out_root / "partition_stability_by_size.csv"
    with size_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(size_rows[0].keys()))
        writer.writeheader()
        writer.writerows(size_rows)

    print(f"\nWrote {agg_csv}")
    print(f"Wrote {size_csv}\n")
    print("Per-size restart stability at operating k:")
    for row in size_rows:
        print(
            f"  size={row['size']:>4} {row['backend']:>4}  "
            f"n={row['n_networks']}  "
            f"ARI_mean={row['restart_ari_mean']:.3f}"
            f"(worst_min={row['restart_ari_min_worst']:.3f})  "
            f"max_cell={row['max_cell_fraction_mean']:.2f}  "
            f"singleton_mass={row['frac_units_in_singletons_mean']:.2f}  "
            f"str_ratio={row['singleton_strength_ratio_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
