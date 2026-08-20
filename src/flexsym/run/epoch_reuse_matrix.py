"""Pooled task-epoch x task-epoch sub-grouping reuse matrix (Fig 3).

For every structural cell (>=4 units), pooled over all 10 seeds, compute the
within-cell functional sub-partition in each task epoch, then the ARI between
sub-partitions for every pair of task epochs. Average ARI across all cells gives
one number per epoch pair -> a task-epoch x task-epoch reuse matrix. Ordered and
colour-coded by computation type (stimulus / memory / response), compositional
reuse appears as bright within-type blocks.
"""
from __future__ import annotations

from flexsym import paths
import sys, json
from pathlib import Path
import numpy as np
_p = np.load
_o = _p.__closure__[0].cell_contents if getattr(_p, "__closure__", None) else _p
def _dl(*a, **k):
    k["allow_pickle"] = True
    return _o(*a, **k)
np.load = _dl
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from flexsym.structural.structural_cell_activity_coherence import compute_hidden_by_epoch, load_structural_labels  # noqa: E402
from flexsym.structural.structural_cell_cross_task_fractionation import filter_valid_task_epochs  # noqa: E402
from flexsym.structural.structural_cell_subpartition_pilot import choose_functional_partition  # noqa: E402
from flexsym.structural.softplus_memory_subpartition_sweep import unit_features_for_period  # noqa: E402

REPO = paths.REPO_ROOT
ORIG = paths.ORIGINAL_STORE
STAGED = paths.STAGED_STORE
from flexsym.structural.operating_k import OPK
# curated task-epoch battery, grouped by computation type (order preserved)
BATTERY = [
    ("fdgo", "stim1"), ("fdanti", "stim1"), ("delaygo", "stim1"), ("delaydm1", "stim1"),
    ("delaygo", "delay1"), ("delayanti", "delay1"), ("delaydm1", "delay1"),
    ("delaydm2", "delay1"), ("contextdelaydm1", "delay1"), ("multidelaydm", "delay1"),
    ("reactgo", "go1"), ("reactanti", "go1"), ("fdgo", "go1"), ("delaygo", "go1"),
]
TYPE = {"stim1": "stimulus", "delay1": "memory", "go1": "response"}


def resolve(m):
    for s in (ORIG, STAGED):
        if (s / m).exists() and (s / m / "model.ckpt.index").exists():
            return s / m
    return None


def main():
    np.load = _dl
    n = len(BATTERY)
    ssum = np.zeros((n, n)); scnt = np.zeros((n, n)); n_cells = 0
    idx = {e: i for i, e in enumerate(BATTERY)}
    for seed, k in OPK.items():
        model = f"LeakyRNN_softplus_diag_256_all_{seed}"
        md = resolve(model)
        sweep = REPO / f"results/signed_input_wl_paper/{model}/transpose_selected_k_{k}/signed_input_wl_sweep.npz"
        if md is None or not sweep.exists():
            print(f"skip {model}"); continue
        labels, ind, _ = load_structural_labels(sweep, k)
        te = filter_valid_task_epochs(md, BATTERY, 512)
        np.load = _dl
        hidden = compute_hidden_by_epoch(md, te, batch_size=512)
        np.load = _dl
        for c in np.unique(labels):
            pos = np.where(labels == c)[0]
            if pos.size < 4:
                continue
            n_cells += 1
            lab = {}
            for e in te:
                X = unit_features_for_period(hidden, e[0], e[1], ind, pos)
                lab[e] = np.asarray(choose_functional_partition(X, max_k=6)["labels"], dtype=np.int64)
            for a in te:
                for b in te:
                    ia, ib = idx[a], idx[b]
                    ssum[ia, ib] += adjusted_rand_score(lab[a], lab[b]); scnt[ia, ib] += 1
        import tensorflow.compat.v1 as tf; tf.reset_default_graph()

    M = np.divide(ssum, scnt, out=np.full((n, n), np.nan), where=scnt > 0)
    out = {
        "n_cells": int(n_cells),
        "epochs": [f"{t}/{e}" for t, e in BATTERY],
        "types": [TYPE[e] for _, e in BATTERY],
        "matrix": M.tolist(),
    }
    Path(paths.RESULTS_ROOT / "signed_input_wl_paper/epoch_reuse_matrix.json").write_text(json.dumps(out, indent=2))
    print(f"n_cells={n_cells}")
    # summarize within-type vs between-type mean ARI
    typ = np.array([TYPE[e] for _, e in BATTERY])
    off = ~np.eye(n, dtype=bool)
    for t in ["stimulus", "memory", "response"]:
        rows = typ == t
        within = M[np.ix_(rows, rows)][off[np.ix_(rows, rows)]]
        betw = M[np.ix_(rows, ~rows)]
        print(f"  {t:>9}: within-type ARI {np.nanmean(within):.3f}  between-type {np.nanmean(betw):.3f}")


if __name__ == "__main__":
    main()
