"""Causal test of the eigenvalue susceptibility: impose the gain change S_ev is about.

The leverage analyses are observational -- S_ev is a structural property, and the question is
whether the network's own gain changes are allocated with respect to it. But S_ev[i,j] =
d|lambda_j|/dg_i asserts what WOULD happen to collective mode j if role i's gain changed, so it
can be tested by performing the intervention.

Per condition:
  j*  leading mode of the quotient Jacobian (1-alpha)I + alpha diag(gbar) Q_struct
  i*  argmax_i |S_ev[i,j*]| -- the role predicted to have most leverage over it
  add delta to the gain of EVERY active unit in role i*, then measure the induced change in
  |lambda_j*| in the FULL N-unit Jacobian, matching modes across the perturbation by
  eigenvector overlap.

Because gbar is a role mean, a uniform increment on a role shifts gbar_i* by exactly delta, so
the prediction S_ev[i*,j*]*delta carries no fitted parameters.

NULL -- role permutation. Permuting role identities preserves the distribution of
susceptibilities exactly and destroys only their assignment. Since a permutation merely relabels
which role carries which susceptibility, the null over ALL permutations is the distribution of
measured shifts across the k roles, so we obtain it exactly by perturbing every role in turn
rather than by sampling. Selection is then the rank of role i* among all k; magnitude is
pred[i*]/actual[i] for a permuted role.

Reported with the trained seed as the unit of replication, per the manuscript convention.

    conda run -n myenv python -m flexsym.run.gain_targeted_modulation --seeds 3-12
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.gain_leverage_geometry import qmat, rotation_sensitivity
from flexsym.run.motif_quotient_prediction import (
    condition_jacobians,
    indicator,
    load_conditions,
    quotient,
)
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.structural.operating_k import OPK

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

FP = paths.RESULTS_ROOT / "motif_fixed_points"
TOL = 1e-12
OV_MIN = 0.80           # mode-identity floor for the across-role correlation
FRACS = (0.25, 1.0)     # delta, as a multiple of the condition's mean per-unit gain


def full_jacobian(W, g, alpha):
    return (1 - alpha) * np.eye(len(g)) + alpha * (g[:, None] * W)


def unit_cols(V):
    return V / np.maximum(np.linalg.norm(V, axis=0), TOL)


def track(Vb, Vp):
    """Match perturbed modes to base modes by |eigenvector overlap|."""
    r, c = linear_sum_assignment(-np.abs(Vb.conj().T @ Vp))
    return c[np.argsort(r)]


def run_condition(cd, W, role, H, sizes, frac):
    """One condition at one delta. Returns a row dict, or None if (alpha, g) is degenerate."""
    g, alpha = cd["g"], cd["alpha"]
    if not (np.all(np.isfinite(g)) and np.isfinite(alpha)):
        return None
    kk = H.shape[1]
    Qs = quotient(W, H, sizes)
    gbar = (H.T @ g) / sizes
    Sev = rotation_sensitivity(Qs, gbar, alpha)[1]
    lam_q, Vq = np.linalg.eig(qmat(Qs, gbar, alpha))
    Vq = unit_cols(Vq)
    j = int(np.argmax(np.abs(lam_q)))
    sev = Sev[:, j]
    istar = int(np.argmax(np.abs(sev)))

    Jf = full_jacobian(W, g, alpha)
    lam_f, Vf = np.linalg.eig(Jf)
    Vf = unit_cols(Vf)
    fm = int(np.argmax(np.abs(Vf.conj().T @ unit_cols(H @ Vq[:, [j]]).ravel())))

    delta = frac * float(np.mean(g))
    act = np.full(kk, np.nan)
    ovl = np.full(kk, np.nan)
    for c in range(kk):                       # every role -> the exact permutation null
        gp = g.copy()
        gp[role == c] += delta
        lam_p, Vp = np.linalg.eig(full_jacobian(W, gp, alpha))
        Vp = unit_cols(Vp)
        m = track(Vf, Vp)
        act[c] = abs(lam_p[m[fm]]) - abs(lam_f[fm])
        ovl[c] = float(np.abs(np.vdot(Vf[:, fm], Vp[:, m[fm]])))

    pred = sev * delta
    aa = np.abs(act)
    ok = ovl >= OV_MIN
    rank = int((aa > aa[istar]).sum())
    return {
        "key": cd["key"], "motif": cd["motif"], "frac": frac, "k": kk,
        "istar": istar, "size": int(sizes[istar]), "lam": float(abs(lam_f[fm])),
        "ov_star": float(ovl[istar]),
        "pred_star": float(pred[istar]), "act_star": float(act[istar]),
        "ratio": float(pred[istar] / act[istar]) if abs(act[istar]) > TOL else np.nan,
        "rank": rank, "pct": float(1.0 - rank / max(kk - 1, 1)),
        "fold": float(aa[istar] / max(np.median(aa), TOL)),
        "r_col": (float(np.corrcoef(pred[ok], act[ok])[0, 1]) if ok.sum() > 3 else np.nan),
    }


def run_net(seed, fracs=FRACS, verbose=True):
    net = f"LeakyRNN_softplus_diag_256_all_{seed}"
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], float)
    ind = np.asarray(wl["ind_active"], int)
    conds = load_conditions(net, ind, W, condition_jacobians(net, ind, W))
    k = int(OPK[seed])
    role = np.asarray(wl["labels_struct_by_k"][_row_for_k(wl["labels_struct_by_k"], k)], int)
    H, sizes = indicator(role)
    if verbose:
        print(f"=== {net}  k={H.shape[1]}  n_cond={len(conds)} ===", flush=True)
    rows = []
    for cd in conds:
        for frac in fracs:
            r = run_condition(cd, W, role, H, sizes, frac)
            if r is not None:
                rows.append(dict(net=net, seed=seed, **r))
    return rows


def summarise(rows, frac):
    """Per-network medians, then across networks with the seed as replication unit."""
    R = [x for x in rows if x["frac"] == frac]
    seeds = sorted({x["seed"] for x in R})
    if not seeds:
        return {}, [], None
    per = {}
    for s in seeds:
        q = [x for x in R if x["seed"] == s]
        k = q[0]["k"]
        rk = np.array([x["rank"] for x in q])
        ra = np.abs([x["ratio"] for x in q])
        per[s] = {
            "n": len(q), "k": k, "rank": float(np.median(rk)),
            "hit": float((rk == 0).mean()), "top3": float((rk < 3).mean()),
            "chance": 1.0 / k, "fold": float(np.median([x["fold"] for x in q])),
            "ratio": float(np.median(ra)), "within2": float(((ra > .5) & (ra < 2)).mean()),
            "rcol": float(np.nanmedian([x["r_col"] for x in q])),
            "pct": float(np.median([x["pct"] for x in q])),
        }
    g = lambda key: np.array([per[s][key] for s in seeds])
    null_rank = np.array([(per[s]["k"] - 1) / 2 for s in seeds])
    w = lambda a, b: float(wilcoxon(a - b).pvalue)
    return per, seeds, {
        "n_cond": len(R), "n_net": len(seeds),
        "rank": float(np.median(g("rank"))), "rank_null": float(np.median(null_rank)),
        "p_rank": w(g("rank"), null_rank),
        "hit": float(np.median(g("hit"))), "chance": float(np.median(g("chance"))),
        "p_hit": w(g("hit"), g("chance")),
        "top3": float(np.median(g("top3"))), "p_top3": w(g("top3"), 3 * g("chance")),
        "fold": float(np.median(g("fold"))),
        "fold_lo": float(g("fold").min()), "fold_hi": float(g("fold").max()),
        "ratio": float(np.median(g("ratio"))), "within2": float(np.median(g("within2"))),
        "rcol": float(np.median(g("rcol"))),
        "rcol_lo": float(g("rcol").min()), "rcol_hi": float(g("rcol").max()),
        "n_better": int((g("rank") < null_rank).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="3-12")
    ap.add_argument("--fracs", default="0.25,1.0",
                    help="delta as a multiple of the condition's mean per-unit gain")
    ap.add_argument("--out", default="gain_targeted_modulation.json")
    args = ap.parse_args()
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    fracs = tuple(float(x) for x in args.fracs.split(","))

    rows = []
    for s in seeds:
        try:
            rows += run_net(s, fracs)
        except (FileNotFoundError, KeyError) as e:
            print(f"[skip] seed {s}: {type(e).__name__}: {e}", flush=True)

    if not rows:
        print("no conditions ran -- check that results/main/motif_fixed_points is reachable")
        return
    for frac in fracs:
        per, ss, S = summarise(rows, frac)
        if S is None:
            continue
        print(f"\n########## delta = {frac} x mean gain ##########")
        for s in ss:
            p = per[s]
            print(f"  seed{s:<3} n={p['n']:3d} k={p['k']:3d} | medrank {p['rank']:5.1f} "
                  f"(null {(p['k']-1)/2:4.1f}) | hit {100*p['hit']:3.0f}% "
                  f"(chance {100*p['chance']:.1f}%) | top3 {100*p['top3']:3.0f}% "
                  f"| fold {p['fold']:5.1f}x | |ratio| {p['ratio']:.2f} | r {p['rcol']:+.2f}")
        print(f"\n  ACROSS {S['n_net']} NETWORKS ({S['n_cond']} conditions; seed = replication unit)")
        print(f"   median rank    {S['rank']:.1f} vs null {S['rank_null']:.1f}   p={S['p_rank']:.4f}")
        print(f"   exact hit      {100*S['hit']:.0f}% vs chance {100*S['chance']:.1f}%  p={S['p_hit']:.4f}")
        print(f"   top-3          {100*S['top3']:.0f}%  p={S['p_top3']:.4f}")
        print(f"   fold           {S['fold']:.1f}x (range {S['fold_lo']:.1f}-{S['fold_hi']:.1f})")
        print(f"   |pred/actual|  {S['ratio']:.2f}, within 2x {100*S['within2']:.0f}%")
        print(f"   across-role r  {S['rcol']:+.3f} (range {S['rcol_lo']:+.2f}-{S['rcol_hi']:+.2f})")
        print(f"   better than null in {S['n_better']}/{S['n_net']} networks")

    out = FP / args.out
    json.dump({"rows": rows,
               "summary": {str(f): summarise(rows, f)[2] for f in fracs
                           if summarise(rows, f)[2] is not None}},
              open(out, "w"), indent=1, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
