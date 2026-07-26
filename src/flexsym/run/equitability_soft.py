"""Equitability z above chance under HARD vs SOFT assignment, across all seeds.

Reviewer concern #5: is the signed-WL equitability (and specifically the 1024
reversal, where gamma_z goes positive = worse than chance) a hard-k-means
artifact? The gamma null depends only on the activity partition (size-matched
shuffles), not on the assignment method, so the clean test is to hold the null
fixed and swap the structural partition's cost:

    z = (gamma_struct - gamma_null_mean) / gamma_null_sd        (negative = more
                                                                 equitable than chance)

computed with the hard k-means WL partition and with the soft (GMM) WL partition
at each network's operating k. Pure connectivity -- no TF model load.

Soft is run from several restarts; we report the best (min-gamma = most
equitable) and the mean, since the partition itself is not unique.
"""

from __future__ import annotations

from flexsym import paths

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
from scipy.cluster.hierarchy import fcluster


from flexsym.structural.signed_input_wl_structural_roles import (  # noqa: E402
    apply_matrix_variant,
    approx_wl_signed,
    gamma_signed,
    labels_to_H,
    load_w_rec,
    random_labels_same_sizes,
)
from flexsym.structural.structural_partition_stability import soft_wl_signed  # noqa: E402

REPO = paths.REPO_ROOT
ACT_ROOT = REPO / "results" / "activity_clusters_paper"
ACT_NAME = "epoch_variance_unit_clusters_paper_max.npz"
ORIGINAL = paths.ORIGINAL_STORE
STAGED = paths.STAGED_STORE
K_SUMMARIES = [
    REPO / "results/signed_input_wl_paper/batch_rnn_softplus_diag_k20pct/k_selection_summary.csv",
    REPO / "results/signed_input_wl_paper/batch_rnn_softplus_diag_newseeds/k_selection_summary.csv",
]


def load_operating_k() -> dict[str, int]:
    out: dict[str, int] = {}
    for s in K_SUMMARIES:
        if not s.exists():
            continue
        for r in csv.DictReader(s.open()):
            na = int(r["n_active"])
            if r.get("selected_k"):
                sk = int(r["selected_k"])
            else:
                sk = min(int(r["activity_k"]), int(r["structural_k"]))
                sk = max(int(r["k_min"]), min(sk, na - 1))
            out[r["model"]] = sk
    return out


def resolve(model: str) -> Path | None:
    for store in (ORIGINAL, STAGED):
        if (store / model).exists():
            return store / model
    return None


def size_of(m: str) -> int:
    return int(m.split("_")[3])


def seed_of(m: str) -> int:
    return int(m.split("_")[-1])


def analyse(model: str, k: int, *, n_null: int, soft_restarts: int, variant: str) -> dict[str, Any] | None:
    model_dir = resolve(model)
    act_path = ACT_ROOT / model / ACT_NAME
    if model_dir is None or not act_path.exists():
        return None
    act = np.load(act_path, allow_pickle=True)
    Z = np.asarray(act["Z"], dtype=float)
    ind = np.asarray(act["ind_active"], dtype=np.int64)
    labels_act = fcluster(Z, k, criterion="maxclust").astype(np.int64) - 1

    W = apply_matrix_variant(load_w_rec(model_dir), variant)[np.ix_(ind, ind)]
    W_pos = np.maximum(W, 0.0)
    W_neg = np.maximum(-W, 0.0)

    # fixed size-matched null (identical for hard & soft)
    null = np.empty(n_null, dtype=float)
    for b in range(n_null):
        H_null = labels_to_H(random_labels_same_sizes(labels_act, seed=b), k)
        null[b] = gamma_signed(W_pos, W_neg, H_null)
    null_mean = float(null.mean())
    null_sd = float(null.std(ddof=1))

    labels_hard, H_hard = approx_wl_signed(W_pos, W_neg, k, n_iter=20, seed=0, n_init=20)
    gamma_hard = float(gamma_signed(W_pos, W_neg, H_hard))

    soft_gammas = []
    for s in range(soft_restarts):
        labels_soft, _ = soft_wl_signed(W_pos, W_neg, k, n_iter=20, seed=s, n_init=5)
        soft_gammas.append(float(gamma_signed(W_pos, W_neg, labels_to_H(labels_soft, k))))
    soft_gammas = np.asarray(soft_gammas, dtype=float)

    z = lambda g: (g - null_mean) / null_sd if null_sd > 0 else float("nan")
    return {
        "model": model,
        "size": size_of(model),
        "seed": seed_of(model),
        "operating_k": k,
        "gamma_null_mean": null_mean,
        "gamma_null_sd": null_sd,
        "gamma_hard": gamma_hard,
        "z_hard": z(gamma_hard),
        "gamma_soft_min": float(soft_gammas.min()),
        "gamma_soft_mean": float(soft_gammas.mean()),
        "z_soft_best": z(float(soft_gammas.min())),
        "z_soft_mean": z(float(soft_gammas.mean())),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-null", type=int, default=1000)
    p.add_argument("--soft-restarts", type=int, default=5)
    p.add_argument("--matrix-variant", default="transpose")
    p.add_argument("--output-dir", default=str(REPO / "results/signed_input_wl_paper/equitability_hard_vs_soft"))
    p.add_argument("--models", nargs="*", default=None)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opk = load_operating_k()
    models = sorted(opk, key=lambda m: (size_of(m), seed_of(m)))
    if args.models:
        models = [m for m in models if m in set(args.models)]

    rows: list[dict[str, Any]] = []
    for m in models:
        res = analyse(m, opk[m], n_null=args.n_null, soft_restarts=args.soft_restarts, variant=args.matrix_variant)
        if res is None:
            print(f"SKIP {m}")
            continue
        rows.append(res)
        print(
            f"{m:36} k={res['operating_k']:>3}  "
            f"z_hard={res['z_hard']:+.2f}  z_soft_best={res['z_soft_best']:+.2f}  "
            f"z_soft_mean={res['z_soft_mean']:+.2f}",
            flush=True,
        )

    csv_path = out_dir / "equitability_hard_vs_soft.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== per-size: equitability z above chance (negative = more equitable) ===")
    print(f"{'size':>5} {'n':>3} | {'z_hard_mean':>11} {'hard_n<-3':>9} | {'z_soft_best_mean':>16} {'soft_n<-3':>9}")
    by = {}
    for r in rows:
        by.setdefault(r["size"], []).append(r)
    agg_rows = []
    for size in sorted(by):
        rs = by[size]
        zh = np.array([r["z_hard"] for r in rs])
        zsb = np.array([r["z_soft_best"] for r in rs])
        zsm = np.array([r["z_soft_mean"] for r in rs])
        agg_rows.append({
            "size": size, "n": len(rs),
            "z_hard_mean": float(zh.mean()), "z_hard_sd": float(zh.std(ddof=1)) if len(rs) > 1 else 0.0,
            "hard_n_lt_-3": int((zh < -3).sum()),
            "z_soft_best_mean": float(zsb.mean()), "z_soft_best_sd": float(zsb.std(ddof=1)) if len(rs) > 1 else 0.0,
            "soft_best_n_lt_-3": int((zsb < -3).sum()),
            "z_soft_mean_mean": float(zsm.mean()),
        })
        print(f"{size:>5} {len(rs):>3} | {zh.mean():>11.2f} {int((zh<-3).sum()):>9} | {zsb.mean():>16.2f} {int((zsb<-3).sum()):>9}")
    agg_path = out_dir / "equitability_hard_vs_soft_by_size.csv"
    with agg_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)
    print(f"\nWrote {csv_path}\nWrote {agg_path}")


if __name__ == "__main__":
    main()
