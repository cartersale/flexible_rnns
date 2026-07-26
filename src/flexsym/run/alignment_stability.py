"""Conclusion-stability of the full-context alignment result across restart partitions.

Reviewer concern #5, productive reframe: the signed-WL partition is not unique
(restart ARI ~0.4), but the question that matters is whether the *headline
downstream conclusion* depends on which of the near-degenerate partitions you
land on. Here we recompute the structure->function full-context alignment
(``softplus_memory_full_context_alignment``) across N restart partitions plus the
co-association consensus, regenerating BOTH the memory-cell selection and the
alignment for each partition, and report the spread of the headline statistic
(``ari_z`` / ``nmi_z``: mean functional<->full-context ARI vs shuffle null).

If ari_z is consistently strong and positive across restarts, partition
non-uniqueness is harmless for the claim and we say so. If it swings sign or
significance, the claim is partition-dependent.

The model activity is loaded once; partitions are looped in-process.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

STEP_DIR = Path(__file__).resolve().parent
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from signed_input_wl_structural_roles import (  # noqa: E402
    apply_matrix_variant,
    approx_wl_signed,
    load_w_rec,
)
from structural_partition_stability import _consensus_partition, soft_wl_signed  # noqa: E402
from structural_cell_epoch_variance import compute_structural_cell_epoch_variance  # noqa: E402
from structural_cell_activity_coherence import compute_hidden_by_epoch  # noqa: E402
from structural_cell_subpartition_pilot import choose_functional_partition  # noqa: E402
from softplus_memory_subpartition_sweep import unit_features_for_period  # noqa: E402
from softplus_memory_full_context_alignment import (  # noqa: E402
    CONTROL_TARGETS,
    choose_full_context_partition,
    control_metric_summary,
    draw_information_matched_controls,
    shuffled_same_sizes,
    summarize_period,
)

# task.py force-patches np.load to always pass allow_pickle=True, which then
# collides with downstream callers that ALSO pass allow_pickle=True explicitly
# ("multiple values for keyword argument"). Install a de-duping loader that
# keeps the intent (force allow_pickle=True) but tolerates explicit kwargs.
_patched_load = np.load
_orig_load = (
    _patched_load.__closure__[0].cell_contents
    if getattr(_patched_load, "__closure__", None)
    else _patched_load
)


def _dedup_load(*a: Any, **k: Any):
    k["allow_pickle"] = True
    return _orig_load(*a, **k)


np.load = _dedup_load

ACTIVITY_ROOT = STEP_DIR.parent / "results" / "activity_clusters_paper"
ACTIVITY_NAME = "epoch_variance_unit_clusters_paper_max.npz"
ORIGINAL_STORE = Path("/Volumes/X9_Pro/flexible_multitask_data/multitask_processed/suppfig4")
STAGED_STORE = Path("/Volumes/X9_Pro/flexible_multitask_data/new_runs/staged")

MEM_PERIODS = [
    ("MemoryPro", "delaygo", "delay1"),
    ("MemoryAnti", "delayanti", "delay1"),
]


def resolve_model_dir(model: str) -> Path:
    for store in (ORIGINAL_STORE, STAGED_STORE):
        if (store / model).exists():
            return store / model
    raise FileNotFoundError(model)


def write_min_sweep(path: Path, labels: np.ndarray, ind_active: np.ndarray, k: int, variant: str) -> None:
    np.savez(
        path,
        labels_struct_by_k=np.asarray([labels.astype(np.int64)], dtype=object),
        rows=np.asarray([[float(k)]], dtype=object),
        row_columns=np.asarray(["k"], dtype=object),
        ind_active=ind_active.astype(np.int64),
        matrix_variant=variant,
        chosen_k=k,
    )


def alignment_for_partition(
    labels: np.ndarray,
    *,
    activity_path: Path,
    ind_active: np.ndarray,
    W_rec: np.ndarray,
    hidden: Any,
    k: int,
    variant: str,
    min_cell_size: int,
    memory_percentile: float,
    max_subclusters: int,
    n_null: int,
    n_control: int,
    control_dim: str,
    seed: int,
    workdir: Path,
) -> dict[str, Any]:
    sweep_path = workdir / "sweep.npz"
    write_min_sweep(sweep_path, labels, ind_active, k, variant)
    cell_var_info = compute_structural_cell_epoch_variance(
        activity_path, sweep_path, workdir, k=k
    )
    cell_var = np.load(cell_var_info["npz_path"], allow_pickle=True)
    cell_sizes = np.asarray(cell_var["cell_sizes"], dtype=np.int64)
    key_names = [str(x) for x in np.asarray(cell_var["key_names"], dtype=object)]
    mem_idx = [key_names.index("delaygo/delay1"), key_names.index("delayanti/delay1")]
    valid = np.where(cell_sizes >= min_cell_size)[0]
    if valid.size == 0:
        return {"n_memory_cells": 0, "combined": None}
    thresholds = np.percentile(
        cell_var["cell_epoch_variance"][valid][:, mem_idx], memory_percentile, axis=0
    )
    memory_cells = [
        int(c)
        for c in valid
        if np.all(cell_var["cell_epoch_variance"][c, mem_idx] >= thresholds)
    ]

    rng = np.random.default_rng(seed)
    control_rng = np.random.default_rng(seed + 1_000_003)
    rows: list[dict[str, Any]] = []
    for cell in memory_cells:
        unit_positions = np.where(labels == cell)[0]
        unit_ids = ind_active[unit_positions]
        full_context = choose_full_context_partition(W_rec, unit_ids, max_k=max_subclusters)
        full_labels = np.asarray(full_context["labels"], dtype=np.int64)
        controls = draw_information_matched_controls(
            W_rec,
            unit_ids,
            n_control=n_control,
            max_k=max_subclusters,
            rng=control_rng,
            control_dim_mode=control_dim,
        )
        for period, rule, epoch in MEM_PERIODS:
            X = unit_features_for_period(hidden, rule, epoch, ind_active, unit_positions)
            functional = choose_functional_partition(X, max_k=max_subclusters)
            func_labels = np.asarray(functional["labels"], dtype=np.int64)
            ari = float(adjusted_rand_score(func_labels, full_labels))
            nmi = float(normalized_mutual_info_score(func_labels, full_labels))
            rp = control_metric_summary(
                func_labels, controls["random_projection"], reference_labels=full_labels
            )
            npm = control_metric_summary(
                func_labels, controls["internal_noise_padded"], reference_labels=full_labels
            )
            ari_null = np.empty(n_null, dtype=float)
            nmi_null = np.empty(n_null, dtype=float)
            for b in range(n_null):
                nl = shuffled_same_sizes(func_labels, rng)
                ari_null[b] = adjusted_rand_score(nl, full_labels)
                nmi_null[b] = normalized_mutual_info_score(nl, full_labels)
            rows.append(
                {
                    "cell": cell,
                    "period": period,
                    "ari": ari,
                    "nmi": nmi,
                    "ari_null": ari_null,
                    "nmi_null": nmi_null,
                    **{f"random_projection_{kk}": v for kk, v in rp.items()},
                    **{f"internal_noise_padded_{kk}": v for kk, v in npm.items()},
                }
            )
    if not rows:
        return {"n_memory_cells": len(memory_cells), "combined": None}
    combined = summarize_period(rows, None)
    return {"n_memory_cells": len(memory_cells), "combined": combined}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--n-restarts", type=int, default=20)
    p.add_argument("--backend", choices=("hard", "soft"), default="hard")
    p.add_argument("--matrix-variant", default="transpose")
    p.add_argument("--min-cell-size", type=int, default=5)
    p.add_argument("--memory-percentile", type=float, default=60.0)
    p.add_argument("--max-subclusters", type=int, default=6)
    p.add_argument("--n-null", type=int, default=2000)
    p.add_argument("--n-control", type=int, default=50)
    p.add_argument("--control-dim", choices=("internal_width", "internal_rank"), default="internal_width")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = resolve_model_dir(args.model)
    activity_path = ACTIVITY_ROOT / args.model / ACTIVITY_NAME

    activity = np.load(activity_path, allow_pickle=True)
    ind_active = np.asarray(activity["ind_active"], dtype=np.int64)
    W_rec = load_w_rec(model_dir)
    W_sub = apply_matrix_variant(W_rec, args.matrix_variant)[np.ix_(ind_active, ind_active)]
    W_pos = np.maximum(W_sub, 0.0)
    W_neg = np.maximum(-W_sub, 0.0)

    print(f"[{args.model}] loading hidden activity (delaygo/delayanti delay1) ...", flush=True)
    hidden = compute_hidden_by_epoch(
        model_dir,
        [("delaygo", "delay1"), ("delayanti", "delay1")],
        batch_size=args.batch_size,
    )
    # compute_hidden_by_epoch imports task.py, which re-wraps np.load on top of
    # ours; restore the de-duping loader so downstream allow_pickle=True is safe.
    np.load = _dedup_load

    # restart partitions + co-association consensus (hard or soft back-end)
    if args.backend == "soft":
        partitions = [
            soft_wl_signed(W_pos, W_neg, args.k, n_iter=20, seed=s, n_init=5)[0]
            for s in range(args.n_restarts)
        ]
    else:
        partitions = [
            approx_wl_signed(W_pos, W_neg, args.k, n_iter=20, seed=s, n_init=20)[0]
            for s in range(args.n_restarts)
        ]
    consensus = _consensus_partition(partitions, args.k)
    labelled = [("restart", s, partitions[s]) for s in range(args.n_restarts)]
    labelled.append(("consensus", -1, consensus))

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        for kind, idx, labels in labelled:
            workdir = Path(td) / f"{kind}_{idx}"
            workdir.mkdir(parents=True, exist_ok=True)
            res = alignment_for_partition(
                labels,
                activity_path=activity_path,
                ind_active=ind_active,
                W_rec=W_rec,
                hidden=hidden,
                k=args.k,
                variant=args.matrix_variant,
                min_cell_size=args.min_cell_size,
                memory_percentile=args.memory_percentile,
                max_subclusters=args.max_subclusters,
                n_null=args.n_null,
                n_control=args.n_control,
                control_dim=args.control_dim,
                seed=idx if idx >= 0 else 999,
                workdir=workdir,
            )
            c = res["combined"]
            rec = {
                "kind": kind,
                "restart_seed": idx,
                "n_memory_cells": res["n_memory_cells"],
                "ari_mean": c["ari_mean"] if c else float("nan"),
                "ari_z": c["ari_z"] if c else float("nan"),
                "ari_p_upper": c["ari_p_upper"] if c else float("nan"),
                "nmi_mean": c["nmi_mean"] if c else float("nan"),
                "nmi_z": c["nmi_z"] if c else float("nan"),
                "nmi_p_upper": c["nmi_p_upper"] if c else float("nan"),
                "n_tests": c["n_tests"] if c else 0,
            }
            for control in CONTROL_TARGETS:
                rec[f"{control}_ari_p_ge_full_context_mean"] = (
                    c[f"{control}_ari_p_ge_full_context_mean"] if c else float("nan")
                )
            records.append(rec)
            print(
                f"  {kind:9} seed={idx:>3}  mem_cells={rec['n_memory_cells']:>2}  "
                f"ari_z={rec['ari_z']:.2f}  nmi_z={rec['nmi_z']:.2f}  "
                f"ari={rec['ari_mean']:.3f}",
                flush=True,
            )

    csv_path = out_dir / f"alignment_stability_{args.model}_k{args.k}_{args.backend}.csv"
    fieldnames = list(records[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)

    restarts = [r for r in records if r["kind"] == "restart"]
    az = np.array([r["ari_z"] for r in restarts], dtype=float)
    nz = np.array([r["nmi_z"] for r in restarts], dtype=float)
    am = np.array([r["ari_mean"] for r in restarts], dtype=float)
    mc = np.array([r["n_memory_cells"] for r in restarts], dtype=float)
    consensus_rec = next(r for r in records if r["kind"] == "consensus")
    summary = {
        "model": args.model,
        "backend": args.backend,
        "k": args.k,
        "n_restarts": args.n_restarts,
        "n_null": args.n_null,
        "ari_z_mean": float(np.nanmean(az)),
        "ari_z_sd": float(np.nanstd(az, ddof=1)),
        "ari_z_min": float(np.nanmin(az)),
        "ari_z_max": float(np.nanmax(az)),
        "frac_restarts_ari_significant_p05": float(np.nanmean([r["ari_p_upper"] < 0.05 for r in restarts])),
        "frac_restarts_ari_positive_z": float(np.nanmean(az > 0)),
        "nmi_z_mean": float(np.nanmean(nz)),
        "nmi_z_min": float(np.nanmin(nz)),
        "ari_mean_mean": float(np.nanmean(am)),
        "ari_mean_sd": float(np.nanstd(am, ddof=1)),
        "n_memory_cells_mean": float(np.nanmean(mc)),
        "n_memory_cells_min": int(np.nanmin(mc)),
        "n_memory_cells_max": int(np.nanmax(mc)),
        "consensus_ari_z": consensus_rec["ari_z"],
        "consensus_nmi_z": consensus_rec["nmi_z"],
        "consensus_n_memory_cells": consensus_rec["n_memory_cells"],
        "csv_path": str(csv_path),
    }
    with (out_dir / f"alignment_stability_{args.model}_k{args.k}_{args.backend}_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
