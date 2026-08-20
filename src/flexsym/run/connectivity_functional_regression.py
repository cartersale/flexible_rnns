"""Similarity / partial-regression test of connectivity -> function (P3, RSA form).

A more powerful, cluster-free version of the Part-3 test. Instead of clustering
each predictor and comparing partitions by ARI (lossy, noisy on 5-8 points, prone
to the k=1 degeneracy), we work with the *pairwise relationship structure* among
units inside a cell:

  For a cell's units, build pairwise similarity matrices
    F     : functional similarity   (activity-trajectory correlation)
    Sfull : full-context connectivity similarity (whole-network input profile corr)
    Sint  : internal connectivity similarity (within-cell wiring corr)
    Swin  : input-weight (W_in) similarity
    Adir  : direct recurrent coupling |W_ij|+|W_ji|
    Srand : rank-matched random projection of full-context (control)

  and regress F on the connectivity similarities (partial Mantel). The headline
  quantity is the PARTIAL correlation of full-context with function CONTROLLING
  FOR internal wiring:  r(F, Sfull | Sint)  -- does whole-network connectivity
  predict functional relationship beyond local wiring? Also the simple
  correlations and r(F, Sfull | Sint, Swin, Srand).

Supports a k-sweep (--k-list) on a fixed network to test whether the effect is
k-selective (coarse large-cell scale vs fine small-cell scale). Structural
partitions at each k are recomputed with the production aWL refinement.

Aggregation: per-cell partial correlations, permutation p (partial Mantel),
aggregated per seed then across seeds (seed as unit of replication, reviewer #4).
"""

from __future__ import annotations

from flexsym import paths

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


_patched_load = np.load
_orig_load = _patched_load.__closure__[0].cell_contents if getattr(_patched_load, "__closure__", None) else _patched_load


def _dedup_load(*a: Any, **k: Any):
    k["allow_pickle"] = True
    return _orig_load(*a, **k)


np.load = _dedup_load

from flexsym.structural.signed_input_wl_structural_roles import apply_matrix_variant, approx_wl_signed, load_w_rec  # noqa: E402
from flexsym.structural.structural_cell_activity_coherence import compute_hidden_by_epoch  # noqa: E402
from flexsym.structural.structural_cell_cross_task_fractionation import filter_valid_task_epochs  # noqa: E402
from flexsym.structural.softplus_memory_subpartition_sweep import unit_features_for_period  # noqa: E402
from flexsym.structural.softplus_memory_full_context_alignment import random_project_features, signed_input_features, feature_rank  # noqa: E402

ORIGINAL = paths.ORIGINAL_STORE
STAGED = paths.STAGED_STORE
ACT_ROOT = paths.RESULTS_ROOT / "activity_clusters_paper"
ACT_NAME = "epoch_variance_unit_clusters_paper_max.npz"
MEM_PERIODS = [("delaygo", "delay1"), ("delayanti", "delay1")]
# Broad battery spanning stimulus / delay(memory) / response epoch types across
# go / anti / decision-making task families. Filtered per-network to valid epochs.
FULL_BATTERY = [
    ("fdgo", "stim1"), ("fdanti", "stim1"), ("delaydm1", "stim1"), ("dmsgo", "stim1"),
    ("delaygo", "delay1"), ("delayanti", "delay1"), ("delaydm1", "delay1"),
    ("contextdelaydm1", "delay1"), ("multidelaydm", "delay1"), ("dmsgo", "delay1"),
    ("reactgo", "go1"), ("fdgo", "go1"), ("fdanti", "go1"),
]
EPOCH_TYPE = {"stim1": "stimulus", "delay1": "delay/memory", "go1": "response"}


def resolve(model: str) -> Path | None:
    for store in (ORIGINAL, STAGED):
        if (store / model).exists():
            return store / model
    return None


def corr_sim(features: np.ndarray) -> np.ndarray:
    """Pairwise Pearson correlation between rows (units) -> m x m similarity."""
    X = features - features.mean(axis=1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    Xn = X / n
    return Xn @ Xn.T


def upper(v: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(v.shape[0], 1)
    return v[iu]


def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def partial_corr(y: np.ndarray, x: np.ndarray, covars: list[np.ndarray]) -> float:
    """Partial correlation of y and x controlling for covars (all 1-D vectors)."""
    if covars:
        C = np.column_stack([np.ones_like(y)] + [zscore(c) for c in covars])
    else:
        C = np.ones((y.size, 1))
    # residualize
    beta_y, *_ = np.linalg.lstsq(C, y, rcond=None)
    beta_x, *_ = np.linalg.lstsq(C, x, rcond=None)
    ry = y - C @ beta_y
    rx = x - C @ beta_x
    if ry.std() == 0 or rx.std() == 0:
        return float("nan")
    return float(np.corrcoef(ry, rx)[0, 1])


def partial_mantel_p(F: np.ndarray, S: np.ndarray, covars_mats: list[np.ndarray], n_perm: int, rng) -> tuple[float, float]:
    """Partial correlation r(F,S|covars) on upper-tri, with unit-permutation p."""
    m = F.shape[0]
    y = zscore(upper(F))
    x = zscore(upper(S))
    cov = [zscore(upper(c)) for c in covars_mats]
    obs = partial_corr(y, x, cov)
    if not np.isfinite(obs):
        return obs, float("nan")
    ge = 1
    for _ in range(n_perm):
        p = rng.permutation(m)
        Sp = S[np.ix_(p, p)]
        r = partial_corr(y, zscore(upper(Sp)), cov)
        if np.isfinite(r) and abs(r) >= abs(obs):
            ge += 1
    return obs, ge / (n_perm + 1)


def struct_labels_at_k(W_pos, W_neg, k, seed=0):
    labels, _ = approx_wl_signed(W_pos, W_neg, k, n_iter=20, seed=seed, n_init=20)
    return labels


def analyse(model: str, k_list: list[int], *, task_epochs, min_cell_size, memory_percentile, n_perm, batch_size, variant, seed, output_dir):
    md = resolve(model)
    act_path = ACT_ROOT / model / ACT_NAME
    if md is None or not act_path.exists():
        return None
    task_epochs = filter_valid_task_epochs(md, task_epochs, batch_size)
    np.load = _dedup_load  # filter_valid_task_epochs imports task.py which re-wraps np.load
    ind = np.asarray(np.load(act_path, allow_pickle=True)["ind_active"], dtype=np.int64)
    import json as _json
    hp = _json.loads((md / "hp.json").read_text())
    n_rec = int(hp["n_rnn"])
    params = np.load(md / "model_params.npz")
    W_rec_full = np.asarray(params["w_in"][-n_rec:, :], dtype=float)
    W_in = np.asarray(params["w_in"][:-n_rec, :], dtype=float)
    W_var = apply_matrix_variant(W_rec_full, variant)[np.ix_(ind, ind)]
    W_pos = np.maximum(W_var, 0.0)
    W_neg = np.maximum(-W_var, 0.0)

    hidden = compute_hidden_by_epoch(md, task_epochs, batch_size=batch_size)
    np.load = _dedup_load
    all_units = np.arange(W_rec_full.shape[1], dtype=np.int64)
    rng = np.random.default_rng(seed)

    rows = []
    for k in k_list:
        labels = struct_labels_at_k(W_pos, W_neg, k, seed=seed)
        cells = [c for c in np.unique(labels) if int((labels == c).sum()) >= min_cell_size]
        for cell in cells:
            pos = np.where(labels == cell)[0]
            uids = ind[pos]
            m = uids.size
            # connectivity similarity matrices
            Sfull = corr_sim(signed_input_features(W_rec_full, uids, all_units))
            Sint = corr_sim(signed_input_features(W_rec_full, uids, uids))
            Swin = corr_sim(np.concatenate([np.maximum(W_in[:, uids].T, 0), np.maximum(-W_in[:, uids].T, 0)], axis=1))
            r = max(1, feature_rank(signed_input_features(W_rec_full, uids, uids)))
            Srand = corr_sim(random_project_features(signed_input_features(W_rec_full, uids, all_units), r, rng))
            Wsub = W_rec_full[np.ix_(uids, uids)]
            Adir = np.abs(Wsub) + np.abs(Wsub.T)
            for rule, epoch in task_epochs:
                Xact = unit_features_for_period(hidden, rule, epoch, ind, pos)
                F = corr_sim(Xact)
                if m < 4:
                    continue
                r_full = partial_corr(zscore(upper(F)), zscore(upper(Sfull)), [])
                r_int = partial_corr(zscore(upper(F)), zscore(upper(Sint)), [])
                r_full_g_int, p_full_g_int = partial_mantel_p(F, Sfull, [Sint], n_perm, rng)
                r_full_g_all, _ = partial_mantel_p(F, Sfull, [Sint, Swin, Srand], max(1, n_perm // 2), rng)
                r_dir = partial_corr(zscore(upper(F)), zscore(upper(Adir)), [])
                rows.append({
                    "model": model, "k": k, "cell": int(cell), "cell_size": int(m),
                    "period": f"{rule}/{epoch}",
                    "r_full": r_full, "r_internal": r_int, "r_direct": r_dir,
                    "r_full_given_internal": r_full_g_int,
                    "p_full_given_internal": p_full_g_int,
                    "r_full_given_int_win_rand": r_full_g_all,
                })
    if not rows:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"conn_func_regression_{model}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return {"model": model, "rows": rows, "csv_path": str(csv_path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--k-list", type=int, nargs="+", required=True)
    p.add_argument("--min-cell-size", type=int, default=6)
    p.add_argument("--memory-percentile", type=float, default=60.0)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--variant", default="transpose")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=str(paths.RESULTS_ROOT / "signed_input_wl_paper/conn_func_regression"))
    p.add_argument("--task-epochs", nargs="+", default=[f"{r}/{e}" for r, e in FULL_BATTERY],
                   help="task/epoch list; default = full battery (stim/delay/go across families)")
    args = p.parse_args()
    task_epochs = [tuple(te.split("/")) for te in args.task_epochs if "/" in te]
    res = analyse(args.model, args.k_list, task_epochs=task_epochs, min_cell_size=args.min_cell_size,
                  memory_percentile=args.memory_percentile, n_perm=args.n_perm, batch_size=args.batch_size,
                  variant=args.variant, seed=args.seed, output_dir=Path(args.output_dir))
    if res is None:
        print("no data"); return
    import numpy as _np
    rows = res["rows"]
    print(f"\n=== {args.model}: partial r(F, full-context | internal) by EPOCH TYPE ===")
    print(f"{'epoch_type':>14} {'ncell':>6} {'r_full':>8} {'r_internal':>10} {'r_full|int':>11} {'frac_p<.05':>11} {'r_full|int+win+rand':>18}")
    def etype(period): return EPOCH_TYPE.get(period.split("/")[1], period.split("/")[1])
    for typ in ["stimulus", "delay/memory", "response"]:
        rk = [r for r in rows if etype(r["period"]) == typ]
        if not rk: continue
        rf = _np.array([r["r_full"] for r in rk]); ri = _np.array([r["r_internal"] for r in rk])
        rfi = _np.array([r["r_full_given_internal"] for r in rk])
        pfi = _np.array([r["p_full_given_internal"] for r in rk])
        rall = _np.array([r["r_full_given_int_win_rand"] for r in rk])
        print(f"{typ:>14} {len(rk):>6} {_np.nanmean(rf):>8.3f} {_np.nanmean(ri):>10.3f} {_np.nanmean(rfi):>11.3f} {_np.nanmean(pfi<0.05):>11.2f} {_np.nanmean(rall):>18.3f}")


if __name__ == "__main__":
    main()
