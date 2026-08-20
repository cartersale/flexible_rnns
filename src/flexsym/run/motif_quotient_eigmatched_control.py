"""Control: does the quotient predict the MOTIF direction, or any eigendirection?

The prediction result (motif_quotient_prediction.py) shows the coarse-grained
quotient predicts the motif direction v of each task epoch. A sceptic's objection:
maybe the quotient simply aligns with these Jacobians' eigenbasis in general, so it
would predict ANY eigenvector of the same Jacobian just as well -- in which case
"predicts the motif" is not special.

This control answers that directly. For each condition we keep the same mean
Jacobian Jb and the same Q_pred basis B, and compare:

  motif      : A_cond of the true motif direction v (the headline statistic)
  eig-matched: A_cond of a NON-motif eigenvector q of the SAME Jb whose eigenvalue
               modulus |lambda_q| is closest to the motif's |lambda_J|, and whose
               alignment with v is below ALIGN_MAX (so it is genuinely a different
               direction, not a second mode of the same motif -- e.g. the second
               ring-plane mode of a line attractor).

If the quotient captured the Jacobian spectrum generically, motif and eig-matched
would score alike. If the quotient captures the computation specifically, the motif
direction scores higher. Everything else -- partition, gains, r, containment
conditioning -- is identical between the two, so the only thing that varies is
which direction of the same operator is being predicted.

  conda run -n myenv python -m flexsym.run.motif_quotient_eigmatched_control --seeds 0-2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_reuse_control import MOTIF_TYPES, _row_for_k
from flexsym.run.motif_reuse_diagnostics import canonical_sign
from flexsym.run.motif_quotient_prediction import (
    condition_jacobians,
    containment_norm,
    indicator,
    lifted_basis,
    load_conditions,
    quotient,
    recover_alpha_gain,
    score,
)

FP_ROOT = paths.RESULTS_ROOT / "motif_fixed_points"
K_GRID = [8, 12, 16, 20, 25, 30, 40, 50]
M = 3                    # r: the saturating mode count from the main result
ALIGN_MAX = 0.30         # an eig-matched control vector must be this unaligned with v


def matched_nonmotif_direction(Jb, v, lam_J):
    """Real eigenvector of Jb with |lambda| nearest |lam_J|, not aligned with v.

    Returns (q, lam_q, align_q) or (None, nan, nan) if no eligible eigenvector.
    """
    lam, EV = np.linalg.eig(Jb)
    EVr = np.real(EV)
    norms = np.linalg.norm(EVr, axis=0)
    align = np.abs(EVr.T @ v) / np.maximum(norms * np.linalg.norm(v), 1e-12)
    eligible = np.flatnonzero((align < ALIGN_MAX) & (norms > 1e-9))
    if not eligible.size:
        return None, np.nan, np.nan
    j = eligible[np.argmin(np.abs(np.abs(lam[eligible]) - abs(lam_J)))]
    q = canonical_sign(EVr[:, j] / norms[j])
    return q, float(np.real(lam[j])), float(align[j])


def run_net(net, k_grid, n_perm, rng):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], dtype=float)
    ind_active = np.asarray(wl["ind_active"], dtype=int)
    jbars = condition_jacobians(net, ind_active, W)
    conds = load_conditions(net, ind_active, W, jbars)
    if len(conds) < 4:
        return None

    # attach the eig-matched control direction to every condition
    kept = []
    for c in conds:
        q, lam_q, align_q = matched_nonmotif_direction(c["Jb"], c["v"], c["lam_J"])
        if q is None:
            continue
        c["q"] = q
        c["lam_q"] = lam_q
        c["align_q"] = align_q
        kept.append(c)
    if len(kept) < 4:
        return None

    ls = wl["labels_struct_by_k"]
    per_k = {}
    for k in k_grid:
        r = _row_for_k(ls, k)
        role = np.asarray(ls[r], dtype=int)
        if np.unique(role).size < 2:
            continue

        def eval_dir(labels, key):
            """A_cond for direction `key` ('v' or 'q') under Q_pred, per condition."""
            H, sizes = indicator(labels)
            Qs = quotient(W, H, sizes)
            vals = []
            for c in kept:
                d = c[key]
                PHd = containment_norm(d, H, sizes)
                gbar = (H.T @ c["g"]) / sizes
                Qp = (1 - c["alpha"]) * np.eye(len(sizes)) + c["alpha"] * (gbar[:, None] * Qs)
                B, _ = lifted_basis(Qp, H, sizes, M)
                _, a_cond = score(d, B, PHd)
                vals.append(a_cond)
            return np.array(vals, dtype=float)

        motif = eval_dir(role, "v")
        ctrl = eval_dir(role, "q")
        types = np.array([c["motif"] for c in kept])

        # per-condition paired difference, then a label-permutation null on the gap
        gap = np.nanmean(motif - ctrl)
        lab = role.copy()
        null_gap = []
        for _ in range(n_perm):
            rng.shuffle(lab)
            null_gap.append(np.nanmean(eval_dir(lab, "v") - eval_dir(lab, "q")))
        null_gap = np.array(null_gap, dtype=float)
        sd = np.nanstd(null_gap)
        per_k[k] = {
            "k": int(np.unique(role).size),
            "n_cond": len(kept),
            "motif": float(np.nanmean(motif)),
            "eig_matched": float(np.nanmean(ctrl)),
            "gap": float(gap),
            "gap_z": float((gap - np.nanmean(null_gap)) / sd) if sd > 0 else 0.0,
            "n_motif_gt_ctrl": int(np.nansum(motif > ctrl)),
            "lam_J_med": float(np.nanmedian([c["lam_J"] for c in kept])),
            "lam_q_med": float(np.nanmedian([c["lam_q"] for c in kept])),
            "align_q_med": float(np.nanmedian([c["align_q"] for c in kept])),
            "by_type": {
                mt: {"n": int(np.sum(types == mt)),
                     "motif": float(np.nanmean(motif[types == mt])),
                     "eig_matched": float(np.nanmean(ctrl[types == mt])),
                     "gap": float(np.nanmean(motif[types == mt] - ctrl[types == mt])),
                     "n_motif_gt_ctrl": int(np.nansum(motif[types == mt] > ctrl[types == mt]))}
                for mt in MOTIF_TYPES if np.any(types == mt)
            },
        }
    return {"per_k": per_k, "n_cond": len(kept),
            "by_motif": {mt: sum(c["motif"] == mt for c in kept) for mt in MOTIF_TYPES}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", default="quotient_eigmatched_control.json")
    args = ap.parse_args()

    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    rng = np.random.default_rng(0)

    per_net = {}
    for s in seeds:
        net = f"LeakyRNN_softplus_diag_{args.size}_all_{s}"
        if not (FP_ROOT / net / "participation_cache_v2.npz").exists():
            print(f"[skip] {net}")
            continue
        r = run_net(net, K_GRID, args.n_perm, rng)
        if r:
            per_net[net] = r
            print(f"[done] {net}: {r['n_cond']} conditions, {r['by_motif']}")

    print(f"\n=== A_cond (Q_pred, r={M}): motif direction vs eig-matched "
          f"non-motif direction of the SAME Jacobian ===")
    print(f"  {'k':>4} {'motif':>7} {'eigmat':>7} {'gap':>7} {'gap_z':>7} "
          f"{'pos':>7} {'|lamJ|':>7} {'|lamq|':>7} {'align_q':>8}")
    for k in K_GRID:
        rows = [v["per_k"][k] for v in per_net.values() if k in v["per_k"]]
        if not rows:
            continue
        npos = sum(r["motif"] > r["eig_matched"] for r in rows)
        print(f"  {k:>4} {np.median([r['motif'] for r in rows]):>7.3f} "
              f"{np.median([r['eig_matched'] for r in rows]):>7.3f} "
              f"{np.median([r['gap'] for r in rows]):>7.3f} "
              f"{np.median([r['gap_z'] for r in rows]):>7.2f} "
              f"{npos:>4}/{len(rows)} "
              f"{np.median([abs(r['lam_J_med']) for r in rows]):>7.3f} "
              f"{np.median([abs(r['lam_q_med']) for r in rows]):>7.3f} "
              f"{np.median([r['align_q_med'] for r in rows]):>8.3f}")

    print(f"\n=== by motif type, at k=20 (gap = motif - eig-matched, per type) ===")
    print(f"  {'type':>16} {'n':>4} {'motif':>7} {'eigmat':>7} {'gap':>7} {'pos':>7}")
    for mt in MOTIF_TYPES:
        rows = [v["per_k"][20]["by_type"][mt] for v in per_net.values()
                if 20 in v["per_k"] and mt in v["per_k"][20]["by_type"]]
        if not rows:
            continue
        ntot = sum(r["n"] for r in rows)
        pos = sum(r["n_motif_gt_ctrl"] for r in rows)
        print(f"  {mt:>16} {ntot:>4} {np.mean([r['motif'] for r in rows]):>7.3f} "
              f"{np.mean([r['eig_matched'] for r in rows]):>7.3f} "
              f"{np.mean([r['gap'] for r in rows]):>7.3f} {pos:>4}/{ntot}")

    out = FP_ROOT / args.out
    json.dump(per_net, open(out, "w"), indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
