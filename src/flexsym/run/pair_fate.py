"""Pooled pair-fate transitions for the Fig 3 reorganization Sankey.

For every within-cell unit pair, pooled over all structural cells and all 10
seeds (256-unit), we ask: given the pair is in the SAME functional sub-group
during a reference memory computation, what is its fate during
  (a) a DIFFERENT task's memory computation  (shared component), and
  (b) a response computation                 (different component)?
Reports P(together|together) and P(apart|together) for each, as pooled pair
counts -> the widths of the Sankey flows. No cherry-picked cell.
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

from flexsym.structural.structural_cell_activity_coherence import compute_hidden_by_epoch, load_structural_labels  # noqa: E402
from flexsym.structural.structural_cell_cross_task_fractionation import filter_valid_task_epochs  # noqa: E402
from flexsym.structural.structural_cell_subpartition_pilot import choose_functional_partition  # noqa: E402
from flexsym.structural.softplus_memory_subpartition_sweep import unit_features_for_period  # noqa: E402

REPO = paths.REPO_ROOT
ORIG = paths.ORIGINAL_STORE
STAGED = paths.STAGED_STORE
OPK = {0: 41, 1: 51, 2: 38, 3: 14, 4: 14, 5: 14, 6: 14, 7: 14, 8: 14, 9: 15}
TYPES = ["stimulus", "memory", "response"]
STIM = [("fdgo", "stim1"), ("fdanti", "stim1"), ("delaygo", "stim1"), ("delaydm1", "stim1"), ("delayanti", "stim1")]
MEM = [("delaygo", "delay1"), ("delayanti", "delay1"), ("delaydm1", "delay1"),
       ("delaydm2", "delay1"), ("contextdelaydm1", "delay1"), ("multidelaydm", "delay1")]
RESP = [("reactgo", "go1"), ("reactanti", "go1"), ("fdgo", "go1"), ("delaygo", "go1")]
BATTERY = STIM + MEM + RESP
EPTYPE = {**{e: "stimulus" for e in STIM}, **{e: "memory" for e in MEM}, **{e: "response" for e in RESP}}


def resolve(m):
    for s in (ORIG, STAGED):
        if (s / m).exists() and (s / m / "model.ckpt.index").exists():
            return s / m
    return None


def comembership(hidden, te, ind, pos):
    """m x m {0,1} co-membership for a cell's units in one epoch (None if partition collapses)."""
    X = unit_features_for_period(hidden, te[0], te[1], ind, pos)
    r = choose_functional_partition(X, max_k=6)
    lab = np.asarray(r["labels"], dtype=np.int64)
    if int(r["chosen_k"]) < 2:
        return None
    return lab[:, None] == lab[None, :]


def main():
    np.load = _dl
    # tog[ref_type][dest_type] = [together, apart] over pairs together in a ref epoch
    tog = {a: {b: np.zeros(2) for b in TYPES} for a in TYPES}
    base = {t: np.zeros(2) for t in TYPES}   # [together, total] unconditional, per dest type
    n_cells = 0
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
            co = {}
            for e in te:
                C = comembership(hidden, e, ind, pos)
                if C is not None:
                    co[e] = C[np.triu_indices(pos.size, 1)]
            eps = list(co)
            for e in eps:
                base[EPTYPE[e]][0] += co[e].sum(); base[EPTYPE[e]][1] += co[e].size
            for a in eps:
                for b in eps:
                    if a == b:
                        continue
                    ta = EPTYPE[a]; tb = EPTYPE[b]
                    mask = co[a] == 1
                    tog[ta][tb][0] += np.sum(co[b][mask] == 1)
                    tog[ta][tb][1] += np.sum(co[b][mask] == 0)
        import tensorflow.compat.v1 as tf; tf.reset_default_graph()

    P = {a: {b: float(tog[a][b][0]/tog[a][b].sum()) if tog[a][b].sum() > 0 else float("nan") for b in TYPES} for a in TYPES}
    base_rate = {t: float(base[t][0]/base[t][1]) for t in TYPES}
    excess = {a: {b: P[a][b] - base_rate[b] for b in TYPES} for a in TYPES}
    counts = {a: {b: tog[a][b].tolist() for b in TYPES} for a in TYPES}
    out = {"n_cells": int(n_cells), "types": TYPES, "P_stay_together": P,
           "base_rate": base_rate, "excess_reuse": excess, "counts": counts}
    Path(REPO / "results/signed_input_wl_paper/pair_fate_3x3.json").write_text(json.dumps(out, indent=2))
    print(f"n_cells={n_cells}  base rates: " + ", ".join(f"{t} {base_rate[t]:.2f}" for t in TYPES))
    print("EXCESS reuse (P_stay - chance), rows=reference, cols=destination:")
    print(f"{'ref/dest':>10} " + " ".join(f"{t:>9}" for t in TYPES))
    for a in TYPES:
        print(f"{a:>10} " + " ".join(f"{excess[a][b]:>+9.3f}" for b in TYPES))


if __name__ == "__main__":
    main()
