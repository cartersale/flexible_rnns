"""Are the reuse deficit (\S reuse) and the mode concentration (\S concentration) the same thing?

Hypothesis (to test, not assume): if motif energy concentrates on few connectivity-quotient
modes, each motif's participation vector over groups is supported on few groups; low-support
vectors have compressed between-motif correlation variance, which drags the reuse statistic
R = mean(same-corr) - mean(diff-corr) DOWN relative to a size-matched random partition whose
participation is spread over more groups. If so, R_conn < R_random is a consequence of the
concentration, not an independent negative, and the two sections are not independent.

Checks, per seed at k=20:
  1. effective support (participation ratio) of the per-motif participation vector, connectivity
     vs size-matched random -- is connectivity lower (more concentrated)?
  2. across seeds, does the connectivity-minus-random support gap correlate with the
     connectivity-minus-random reuse gap (Delta R)? A positive correlation supports the linkage.
Also: a diagnostic of the row-shuffle equitability (\S surrogate) -- gamma_obs and its own null
for trained vs row-shuffle, to characterise why the shuffle scores higher.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from scipy.stats import spearmanr, pearsonr

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_reuse_control import _row_for_k, MOTIF_TYPES
from flexsym.run.motif_quotient_prediction import condition_jacobians, load_conditions, indicator
from flexsym.structural.signed_input_wl_structural_roles import (
    approx_wl_signed, gamma_signed, labels_to_H, random_labels_same_sizes,
)

FP = paths.REPO_ROOT / "results/motif_fixed_points"
NET = lambda s: f"LeakyRNN_softplus_diag_256_all_{s}"


def participation(P, labels):
    """Per-motif participation vector over groups: summed squared loading within each group."""
    H, _ = indicator(labels)                       # (n, k)
    G = (P @ H) ** 2                                # (motifs, k) NO -- need per-unit loading
    # participation_c = sum_{i in c} p_i^2 ; P rows are unit vectors over active units
    part = (P ** 2) @ H                             # (motifs, k)
    return part / part.sum(1, keepdims=True)


def eff_support(part):
    """Participation ratio of each participation vector (effective number of groups)."""
    return (part.sum(1) ** 2) / (part ** 2).sum(1)


def project(P, labels):
    """Group-constant projection of every row of P (m, N): unit -> its group mean."""
    cats, inv = np.unique(labels, return_inverse=True)
    G = np.zeros((len(labels), cats.size)); G[np.arange(len(labels)), inv] = 1.0
    sizes = G.sum(0)
    return (P @ G / sizes) @ G.T


def reuse_R(P, labels, motifs, rules):
    """Reuse-control statistic: correlate GROUP-PROJECTED per-unit profiles over CROSS-TASK
    pairs; same-motif-type minus different-motif-type mean correlation."""
    Pp = project(P, labels)
    C = np.corrcoef(Pp)
    m = np.array(motifs); r = np.array(rules)
    n = len(motifs); same_v, diff_v = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if r[i] == r[j]:
                continue                       # cross-task pairs only
            (same_v if m[i] == m[j] else diff_v).append(C[i, j])
    if not same_v or not diff_v:
        return np.nan
    return float(np.mean(same_v) - np.mean(diff_v))


def run_net(net, k, n_rand, rng):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], float)
    ind = np.asarray(wl["ind_active"], int)
    jbars = condition_jacobians(net, ind, W)
    conds = load_conditions(net, ind, W, jbars)
    P = np.array([c["v"] for c in conds])
    motifs = [c["motif"] for c in conds]; rules = [c["rule"] for c in conds]
    ls = wl["labels_struct_by_k"]
    role = np.asarray(ls[_row_for_k(ls, k)], int)
    part_c = participation(P, role)
    supp_c = float(np.median(eff_support(part_c))); R_c = reuse_R(P, role, motifs, rules)
    _, sizes = np.unique(role, return_counts=True)
    supp_r, R_r = [], []
    for _ in range(n_rand):
        perm = np.repeat(np.arange(len(sizes)), sizes); rng.shuffle(perm)
        pr = participation(P, perm)
        supp_r.append(np.median(eff_support(pr))); R_r.append(reuse_R(P, perm, motifs, rules))
    return dict(supp_c=supp_c, supp_r=float(np.mean(supp_r)), R_c=R_c, R_r=float(np.nanmean(R_r)))


def gamma_diag(net, k):
    """Trained vs row-shuffle: WL-optimised gamma and its OWN size-matched null."""
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    Wt = np.asarray(wl["W_rec"], float)   # already restricted to active units in this npz
    rng = np.random.default_rng(0)
    row = np.empty_like(Wt)
    for i in range(len(Wt)):
        row[i] = Wt[i, rng.permutation(len(Wt))]
    out = {}
    for name, M in (("trained", Wt), ("row_shuffle", row)):
        Wp, Wn = np.maximum(M, 0), np.maximum(-M, 0)
        labels, H = approx_wl_signed(Wp, Wn, k, n_iter=20, seed=0, n_init=20)
        g = gamma_signed(Wp, Wn, H)
        null = np.array([gamma_signed(Wp, Wn, labels_to_H(random_labels_same_sizes(labels, seed=b), k)) for b in range(200)])
        out[name] = dict(gamma=float(g), null_mean=float(null.mean()), null_sd=float(null.std()),
                         z=float((null.mean()-g)/null.std()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9"); ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n-rand", type=int, default=50)
    args = ap.parse_args()
    seeds = list(range(10)) if args.seeds == "0-9" else [int(x) for x in args.seeds.split(",")]
    rng = np.random.default_rng(0)

    per = {}
    for s in seeds:
        if not (FP / NET(s) / "participation_cache_v2.npz").exists(): continue
        per[s] = run_net(NET(s), args.k, args.n_rand, rng)
        r = per[s]
        print(f" seed {s}: support conn {r['supp_c']:.2f} rand {r['supp_r']:.2f} | R conn {r['R_c']:+.3f} rand {r['R_r']:+.3f} | dSupp {r['supp_c']-r['supp_r']:+.2f} dR {r['R_c']-r['R_r']:+.3f}", flush=True)

    dsupp = np.array([per[s]['supp_c']-per[s]['supp_r'] for s in per])
    dR = np.array([per[s]['R_c']-per[s]['R_r'] for s in per])
    rr, pp = spearmanr(dsupp, dR); rp, ppp = pearsonr(dsupp, dR)
    print(f"\n LINKAGE (k={args.k}): connectivity support is {'LOWER' if dsupp.mean()<0 else 'higher'} than random "
          f"(mean dSupp {dsupp.mean():+.2f}); dR mean {dR.mean():+.3f}")
    print(f" corr(dSupp, dR): Spearman {rr:+.2f} p={pp:.3f}; Pearson {rp:+.2f} p={ppp:.3f}")
    print(f"  (positive corr => more concentration [lower support] goes with more negative reuse gap)")

    print("\n ROW-SHUFFLE gamma diagnostic (seed 0, k=20):")
    gd = gamma_diag(NET(0), args.k)
    for name, v in gd.items():
        print(f"   {name:11s}: gamma {v['gamma']:.3f}  own-null mean {v['null_mean']:.3f} sd {v['null_sd']:.3f}  z {v['z']:+.2f}")

    (FP / "reuse_concentration_linkage.json").write_text(json.dumps(
        {"per_seed": {str(s): per[s] for s in per},
         "corr_dSupp_dR": {"spearman": float(rr), "spearman_p": float(pp), "pearson": float(rp)},
         "gamma_diag_seed0": gd}, indent=2, default=float))
    print(f"\nwrote {FP/'reuse_concentration_linkage.json'}")


if __name__ == "__main__":
    main()
