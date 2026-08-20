"""Memory-motif lesion test, improved design.

Prediction (pre-committed): cells that participate in the slow (marginal) subspace
of the delay-period Jacobian, when lesioned, selectively impair memory-maintenance
behaviour, controlling for overall lesion impact.

Changes from v1:
 (1) Matched behavioural contrast that isolates memory maintenance:
       Dmem(cell) = deficit(delaygo, delayanti) - deficit(reactgo, reactanti)
     (delay vs reaction versions of the same go/anti tasks), re-estimated at high reps.
 (2) Memory-mode predictor = participation in the slow SUBSPACE (all Jacobian modes
     with |lambda_J| > 0.9), not a single eigenvector.
 (3) Memory-SPECIFICITY: partial correlation of participation with Dmem controlling
     for the cell's total deficit magnitude.
 (4) Power: pool cells across networks, z-scored within network.
 (5) Robustness: an activity-based memory predictor (delay-vs-react activation).

Decision rule: if the partial correlation is null under this design, report null.
"""
from __future__ import annotations

from flexsym import paths
import sys, csv
from pathlib import Path
import numpy as np

_p = np.load
_o = _p.__closure__[0].cell_contents if getattr(_p, "__closure__", None) else _p
def _dl(*a, **k):
    k["allow_pickle"] = True
    return _o(*a, **k)
np.load = _dl

import tensorflow.compat.v1 as tf  # noqa: E402
tf.disable_v2_behavior()
from flexsym.core import tools  # noqa: E402
from flexsym.core.network import Model  # noqa: E402
from flexsym.structural.signed_input_wl_structural_roles import apply_matrix_variant, load_w_rec  # noqa: E402
from flexsym.structural.structural_cell_activity_coherence import compute_hidden_by_epoch  # noqa: E402
from flexsym.structural.structural_cell_lesions import evaluate_condition  # noqa: E402

REPO = paths.REPO_ROOT
DET = paths.RESULTS_ROOT / "signed_input_wl_paper/lesion_causal"
ORIG = paths.ORIGINAL_STORE
STAGED = paths.STAGED_STORE
ALPHA = 0.2
CONTRAST = ["delaygo", "delayanti", "reactgo", "reactanti"]  # delay(mem) vs react(no-mem)


def resolve(m):
    for s in (ORIG, STAGED):
        if (s / m).exists() and (s / m / "model.ckpt.index").exists():
            return s / m
    return None


def slow_subspace_participation(md, ind, labels, cells):
    WT = apply_matrix_variant(load_w_rec(md), "transpose")[np.ix_(ind, ind)]
    hid = compute_hidden_by_epoch(md, [("delaygo", "delay1")], batch_size=512)
    np.load = _dl
    hstar = hid[("delaygo", "delay1")][:, :, ind][-5:].mean(0).mean(0)
    g = 1 - np.exp(-np.maximum(hstar, 1e-6))
    M = g[:, None] * WT
    w, V = np.linalg.eig(M)
    Jw = np.abs((1 - ALPHA) + ALPHA * w)
    slow = np.where(Jw > 0.9)[0]
    if slow.size < 2:
        slow = np.argsort(-Jw)[:5]
    elif slow.size > 10:
        slow = slow[np.argsort(-Jw[slow])[:10]]
    B = np.concatenate([np.real(V[:, slow]), np.imag(V[:, slow])], axis=1)
    Q, _ = np.linalg.qr(B)
    part_unit = (Q ** 2).sum(axis=1)  # per-unit participation in slow subspace
    return np.array([part_unit[labels == c].mean() for c in cells]), Jw[slow]


def partial_corr(y, x, z):
    def resid(a, c):
        C = np.column_stack([np.ones_like(a), c])
        b, *_ = np.linalg.lstsq(C, a, rcond=None)
        return a - C @ b
    ry, rx = resid(y, z), resid(x, z)
    if ry.std() == 0 or rx.std() == 0:
        return np.nan
    return float(np.corrcoef(ry, rx)[0, 1])


def main():
    np.load = _dl  # task.py re-patched np.load at import time; restore de-duping loader
    n_rep, batch = 16, 512
    per_seed = []
    pooled = {"part": [], "dmem": [], "total": [], "actmem": [], "seed": []}
    for seed in range(10):
        model = f"LeakyRNN_softplus_diag_256_all_{seed}"
        det = DET / f"lesion_detail_{model}.npz"; md = resolve(model)
        if not det.exists() or md is None:
            print(f"skip {model}"); continue
        d = np.load(det, allow_pickle=True)
        labels = np.asarray(d["labels"], dtype=np.int64)
        ind = np.asarray(d["ind_active"], dtype=np.int64)
        rules_all = [str(x) for x in d["rules"]]
        A = np.asarray(d["A_unit_task"], dtype=float)   # [n_rnn, 15]
        total_saved = np.abs(np.asarray(d["struct_deficit"], dtype=float)).sum(1)  # per-cell importance
        cells = sorted(np.unique(labels))
        ai = {r: rules_all.index(r) for r in CONTRAST}
        # activity-based memory predictor per cell
        actmem_unit = ((A[:, ai["delaygo"]] + A[:, ai["delayanti"]]) -
                       (A[:, ai["reactgo"]] + A[:, ai["reactanti"]])) / 2
        actmem = np.array([actmem_unit[ind[labels == c]].mean() for c in cells])

        part, sloweig = slow_subspace_participation(md, ind, labels, cells)
        np.load = _dl

        # re-estimate the 4 contrast-task deficits per structural cell at high reps
        model_obj = Model(str(md)); hp = tools.load_hp(str(md))
        with tf.Session() as sess:
            perf_intact, _ = evaluate_condition(model_obj, sess, hp, CONTRAST, None, n_rep, batch)
            dmem = np.zeros(len(cells)); tot = np.zeros(len(cells))
            for j, c in enumerate(cells):
                units = ind[labels == c]
                perf, _ = evaluate_condition(model_obj, sess, hp, CONTRAST, units, n_rep, batch)
                dfc = perf_intact - perf  # [4]
                delay = (dfc[0] + dfc[1]) / 2; react = (dfc[2] + dfc[3]) / 2
                dmem[j] = delay - react
                tot[j] = np.abs(dfc).sum()
        tf.reset_default_graph()

        # per-network stats
        n = len(cells)
        pc = partial_corr(dmem, part, total_saved[:n])   # participation->Dmem | total importance
        r_raw = np.corrcoef(part, dmem)[0, 1] if np.std(part) > 0 and np.std(dmem) > 0 else np.nan
        r_act = np.corrcoef(actmem, dmem)[0, 1] if np.std(actmem) > 0 and np.std(dmem) > 0 else np.nan
        per_seed.append(dict(seed=seed, n=n, partial_r=pc, raw_r=r_raw, act_r=r_act))
        # pool (z-score within network)
        def z(v):
            v = np.asarray(v, float); return (v - v.mean()) / v.std() if v.std() > 0 else v * 0
        pooled["part"] += list(z(part)); pooled["dmem"] += list(z(dmem))
        pooled["total"] += list(z(total_saved[:n])); pooled["actmem"] += list(z(actmem))
        pooled["seed"] += [seed] * n
        print(f"seed {seed}: n={n} slowmodes={len(sloweig)}  partial_r={pc:+.3f}  raw_r={r_raw:+.3f}  act_r={r_act:+.3f}", flush=True)

    from scipy.stats import binomtest, wilcoxon
    pr = np.array([x["partial_r"] for x in per_seed])
    print("\n=== across seeds (seed = replication unit) ===")
    print(f"partial r(slow-subspace participation, Dmem | total): mean={np.nanmean(pr):+.3f}, "
          f"{int((pr>0).sum())}/{len(pr)}>0, sign P={binomtest(int((pr>0).sum()),len(pr),0.5).pvalue:.4f}")
    print(f"raw r: mean={np.nanmean([x['raw_r'] for x in per_seed]):+.3f}   "
          f"activity-memory r: mean={np.nanmean([x['act_r'] for x in per_seed]):+.3f}")
    # pooled partial correlation
    P = {k: np.asarray(v, float) for k, v in pooled.items()}
    pooled_pr = partial_corr(P["dmem"], P["part"], P["total"])
    pooled_act = partial_corr(P["dmem"], P["actmem"], P["total"])
    print(f"POOLED ({len(P['part'])} cells) partial r(participation,Dmem|total)={pooled_pr:+.3f}; "
          f"activity-memory partial r={pooled_act:+.3f}")
    with (DET / "memory_motif_v2.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed[0].keys())); w.writeheader(); w.writerows(per_seed)
    print(f"wrote {DET/'memory_motif_v2.csv'}")


if __name__ == "__main__":
    main()
