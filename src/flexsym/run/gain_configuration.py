"""Structure x state -> function: does gain configure a fixed structural substrate?

The composition tests failed because they asked the architecture to carry
configuration identity. The decomposition says it should not:

    Q_c = (1-a) I + a diag(gbar_c) Q_struct

Q_struct is task-free and fixed. Everything condition-specific is the k-vector
gbar_c. Under that reading the invariant is not a mode but the substrate whose
modes gain reshapes -- eig(D_A Q) != eig(D_B Q) in general even for fixed Q, so
"which quotient mode is memory" has no reason to be conserved.

Three tests.

  1. COUNTERFACTUAL GAIN SWAP. Hold the structure fixed, swap the gain state.
     S[i,j] = how well the quotient built from condition j's gains captures
     condition i's motif direction. If gains carry configuration, S is diagonally
     dominant. Scored by the within-row percentile of the matched entry (scale
     free, so representations of different dimension compare fairly; chance 0.5).

     Representations, which are the controls operator-transport lacked:
       role      group-mean gains over the connectivity partition
       activity  group-mean gains over the activity partition at matched k
       unit      full per-unit gains, no partition   (does coarse-graining help?)
       shuffled  role-mean gains permuted across cells (distribution preserved)
       random    size-matched random partitions, quotient recomputed

  2. STRUCTURAL SUSCEPTIBILITY. For Q(g) = (1-a)I + a D(g) Q_struct,

         d lambda_j / d g_i = a * u_j[i] * (Q_struct v_j)[i] / (u_j' v_j)

     with u_j, v_j the left/right eigenvectors. This is a purely STRUCTURAL map:
     which roles can move which collective modes. Reported as leverage on |lambda|
     (what sets contracting / neutral / unstable), with its concentration over
     roles compared against random and activity partitions.

  3. DOES THE NETWORK USE THE LEVERAGE? Two forms.
     3a  First-order validity: across modes, does s_j . dgbar predict the OBSERVED
         d|lambda_j| between the two conditions of a contrast (modes matched by
         eigenvector overlap)? This tests the linearisation and the susceptibility
         object at once.
     3b  Targeting: is dgbar aligned with the leverage of the mode that actually
         carries the motif, more than with the leverage of other modes?

    conda run -n myenv python -m flexsym.run.gain_configuration --seeds 0-9
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_operator_composition import FAMILIES, key, med
from flexsym.run.motif_quotient_prediction import (
    condition_jacobians,
    indicator,
    lifted_basis,
    load_conditions,
    quotient,
)
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.structural.operating_k import OPK, model_name

FP = paths.REPO_ROOT / "results" / "motif_fixed_points"
TOP_M = 3
N_PERM = 50
TOL = 1e-12


# --------------------------------------------------------------------------- #
# bases
# --------------------------------------------------------------------------- #
def unit_basis(W, g, alpha, m):
    """Top-m eigenvector basis of the full-resolution predicted Jacobian."""
    J = (1 - alpha) * np.eye(len(W)) + alpha * (g[:, None] * W)
    lam, V = np.linalg.eig(J)
    order = np.argsort(-np.abs(lam))[:m]
    U = np.real(V[:, order])
    if np.any(np.abs(np.imag(V[:, order])) > 1e-12):
        U = np.hstack([U, np.imag(V[:, order])])
    B, R = np.linalg.qr(U)
    return B[:, np.abs(np.diag(R)) > 1e-10]


def containment(v, H, sizes):
    gs = H.T @ v
    return float(np.sqrt(np.sum(gs ** 2 / sizes)))


def swap_matrix(conds, labels, W, Qs_cache=None, shuffle_rng=None):
    """S[i,j] = ||B(gbar_j)' v_i|| / ||P_H v_i|| for a given partition."""
    H, sizes = indicator(labels)
    Qs = quotient(W, H, sizes)
    n = len(conds)
    gbars = []
    for c in conds:
        gb = (H.T @ c["g"]) / sizes
        if shuffle_rng is not None:
            gb = gb[shuffle_rng.permutation(len(gb))]
        gbars.append(gb)
    bases = [lifted_basis((1 - c["alpha"]) * np.eye(len(sizes))
                          + c["alpha"] * (gb[:, None] * Qs), H, sizes, TOP_M)[0]
             for c, gb in zip(conds, gbars)]
    denom = [containment(c["v"], H, sizes) for c in conds]
    S = np.full((n, n), np.nan)
    for i, c in enumerate(conds):
        if denom[i] <= TOL:
            continue
        for j in range(n):
            if bases[j].shape[1] == 0:
                continue
            S[i, j] = np.linalg.norm(bases[j].T @ c["v"]) / denom[i]
    return S


def swap_matrix_unit(conds, W):
    n = len(conds)
    bases = [unit_basis(W, c["g"], c["alpha"], TOP_M) for c in conds]
    S = np.full((n, n), np.nan)
    for i, c in enumerate(conds):
        for j in range(n):
            S[i, j] = np.linalg.norm(bases[j].T @ c["v"]) / np.linalg.norm(c["v"])
    return S


def diag_percentile(S):
    """Mean within-row percentile of the matched (diagonal) entry. Chance 0.5."""
    pct = []
    for i in range(len(S)):
        row = S[i].copy()
        d = row[i]
        row[i] = np.nan
        oth = row[np.isfinite(row)]
        if not np.isfinite(d) or not len(oth):
            continue
        pct.append(float((oth < d).mean() + 0.5 * (oth == d).mean()))
    return float(np.mean(pct)) if pct else np.nan


def diag_margin(S):
    d = np.diag(S)
    off = S[~np.eye(len(S), dtype=bool)]
    return float(np.nanmean(d) - np.nanmean(off))


# --------------------------------------------------------------------------- #
# susceptibility
# --------------------------------------------------------------------------- #
def susceptibility(Qs, gbar, alpha):
    """d|lambda_j| / d g_i at the operating point gbar.

    Returns (Ssens (k roles x k modes, real), lam (k,), Vr (right eigvecs)).
    """
    k = len(gbar)
    Q = (1 - alpha) * np.eye(k) + alpha * (gbar[:, None] * Qs)
    lam, Vr = np.linalg.eig(Q)
    Vl = np.linalg.inv(Vr).conj().T          # left eigenvectors, u_j' v_j = 1
    QsV = Qs @ Vr
    S = np.zeros((k, k))
    for j in range(k):
        dl = alpha * np.conj(Vl[:, j]) * QsV[:, j]          # d lambda_j / d g_i
        if abs(lam[j]) > TOL:
            S[:, j] = np.real(np.conj(lam[j]) / abs(lam[j]) * dl)   # d|lambda|
        else:
            S[:, j] = np.real(dl)
    return S, lam, Vr


def participation_ratio(x):
    p = np.abs(x)
    s = p.sum()
    if s <= TOL:
        return np.nan
    p = p / s
    return float(1.0 / (p ** 2).sum())


def match_modes(V1, V2):
    """Greedy-optimal mode correspondence by |right-eigenvector overlap|."""
    A = np.abs(np.real(V1).T @ np.real(V2))
    A /= np.maximum(np.linalg.norm(np.real(V1), axis=0)[:, None]
                    * np.linalg.norm(np.real(V2), axis=0)[None, :], TOL)
    r, c = linear_sum_assignment(-A)
    return r, c, A[r, c]


# --------------------------------------------------------------------------- #
def run_net(net, k, rng, n_perm=N_PERM):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], float)
    ind = np.asarray(wl["ind_active"], int)
    jbars = condition_jacobians(net, ind, W)
    conds = load_conditions(net, ind, W, jbars)
    if len(conds) < 8:
        return None
    ls, la = wl["labels_struct_by_k"], wl["labels_act_by_k"]
    r = _row_for_k(ls, k)
    role = np.asarray(ls[r], int)
    actl = np.asarray(la[r], int)
    H, sizes = indicator(role)
    Qs = quotient(W, H, sizes)
    alpha = float(np.median([c["alpha"] for c in conds]))
    byk = {c["key"]: c for c in conds}
    present = set(byk)

    # ---- 1. gain swap ----------------------------------------------------- #
    S_role = swap_matrix(conds, role, W)
    S_act = swap_matrix(conds, actl, W)
    S_unit = swap_matrix_unit(conds, W)
    S_shuf = swap_matrix(conds, role, W, shuffle_rng=rng)
    _, szs = np.unique(role, return_counts=True)
    rand_pct, rand_mrg = [], []
    for _ in range(n_perm):
        perm = np.repeat(np.arange(len(szs)), szs)
        rng.shuffle(perm)
        Sr = swap_matrix(conds, perm, W)
        rand_pct.append(diag_percentile(Sr))
        rand_mrg.append(diag_margin(Sr))

    swap = {
        "role": {"pct": diag_percentile(S_role), "margin": diag_margin(S_role)},
        "activity": {"pct": diag_percentile(S_act), "margin": diag_margin(S_act)},
        "unit": {"pct": diag_percentile(S_unit), "margin": diag_margin(S_unit)},
        "shuffled_gains": {"pct": diag_percentile(S_shuf),
                           "margin": diag_margin(S_shuf)},
        "random_partition": {"pct": float(np.nanmean(rand_pct)),
                             "margin": float(np.nanmean(rand_mrg)),
                             "pct_sd": float(np.nanstd(rand_pct))},
    }

    # context-specific swap: does the context gain vector alone flip the prediction?
    ctx = []
    for prim, epoch, pairs in FAMILIES:
        if prim != "context":
            continue
        for a, b in pairs:
            ka, kb = key(a, epoch), key(b, epoch)
            if ka in present and kb in present:
                ia = [i for i, c in enumerate(conds) if c["key"] == ka][0]
                ib = [i for i, c in enumerate(conds) if c["key"] == kb][0]
                ctx.append({"pair": f"{a}->{b}@{epoch}",
                            "own_a": S_role[ia, ia], "swapped_a": S_role[ia, ib],
                            "own_b": S_role[ib, ib], "swapped_b": S_role[ib, ia]})

    # ---- 2. susceptibility ------------------------------------------------ #
    gref = np.median(np.array([(H.T @ c["g"]) / sizes for c in conds]), axis=0)
    Ssens, lam_ref, Vref = susceptibility(Qs, gref, alpha)
    order = np.argsort(-np.abs(lam_ref))
    pr_role = [participation_ratio(Ssens[:, j]) for j in order[:10]]

    Ha, sa = indicator(actl)
    Qa = quotient(W, Ha, sa)
    ga = np.median(np.array([(Ha.T @ c["g"]) / sa for c in conds]), axis=0)
    Sa, lam_a, _ = susceptibility(Qa, ga, alpha)
    pr_act = [participation_ratio(Sa[:, j])
              for j in np.argsort(-np.abs(lam_a))[:10]]

    pr_rand = []
    for _ in range(n_perm):
        perm = np.repeat(np.arange(len(szs)), szs)
        rng.shuffle(perm)
        Hr, sr = indicator(perm)
        Qr = quotient(W, Hr, sr)
        gr = np.median(np.array([(Hr.T @ c["g"]) / sr for c in conds]), axis=0)
        Sr_, lam_r, _ = susceptibility(Qr, gr, alpha)
        pr_rand.append(np.median([participation_ratio(Sr_[:, j])
                                  for j in np.argsort(-np.abs(lam_r))[:10]]))

    susc = {"pr_role": float(np.nanmedian(pr_role)),
            "pr_activity": float(np.nanmedian(pr_act)),
            "pr_random": float(np.nanmedian(pr_rand)),
            "k": int(len(sizes))}

    # ---- 3. does the network use the leverage? ---------------------------- #
    fam_rows = []
    for prim, epoch, pairs in FAMILIES:
        for a, b in pairs:
            ka, kb = key(a, epoch), key(b, epoch)
            if ka not in present or kb not in present:
                continue
            ga_ = (H.T @ byk[ka]["g"]) / sizes
            gb_ = (H.T @ byk[kb]["g"]) / sizes
            dg = gb_ - ga_
            if np.linalg.norm(dg) <= TOL:
                continue
            # 3a: first-order prediction of the observed |lambda| change
            Sa_, lam_a_, Va_ = susceptibility(Qs, ga_, alpha)
            _, lam_b_, Vb_ = susceptibility(Qs, gb_, alpha)
            ri, ci, ov = match_modes(Va_, Vb_)
            keep = ov > 0.5
            if keep.sum() >= 5:
                pred = Sa_[:, ri[keep]].T @ dg
                obs = np.abs(lam_b_[ci[keep]]) - np.abs(lam_a_[ri[keep]])
                rr = float(np.corrcoef(pred, obs)[0, 1]) if np.std(pred) > TOL else np.nan
                # null: dg permuted across roles
                nulls = []
                for _ in range(50):
                    dp = dg[rng.permutation(len(dg))]
                    pn = Sa_[:, ri[keep]].T @ dp
                    if np.std(pn) > TOL:
                        nulls.append(float(np.corrcoef(pn, obs)[0, 1]))
                nr = float(np.nanmean(nulls)) if nulls else np.nan
            else:
                rr, nr, keep = np.nan, np.nan, keep
            # 3b: is dg aligned with the leverage of the motif-carrying mode?
            # j* must be indexed in the BASE eigenbasis Va_, because Sa_ columns
            # are base modes. Using Vb_ here mismatched the two orderings.
            S_lift = H / np.sqrt(sizes)[None, :]
            M = S_lift @ np.real(Va_)
            M /= np.maximum(np.linalg.norm(M, axis=0), TOL)
            energy = (M.T @ byk[kb]["v"]) ** 2
            jstar = int(np.argmax(energy))
            sj = Sa_[:, jstar]
            def acos(x, y):
                nx, ny = np.linalg.norm(x), np.linalg.norm(y)
                return abs(float(x @ y / (nx * ny))) if nx > TOL and ny > TOL else np.nan
            a_star = acos(dg, sj)
            a_other = float(np.nanmedian([acos(dg, Sa_[:, j])
                                          for j in range(len(sizes)) if j != jstar]))
            # null: dgbar permuted across roles, leverage vector held fixed --
            # destroys which roles the gain change lands on, keeps its distribution
            a_perm = float(np.nanmean([acos(dg[rng.permutation(len(dg))], sj)
                                       for _ in range(200)]))
            fam_rows.append({"family": f"{prim}@{epoch}", "pair": f"{a}->{b}",
                             "n_modes": int(keep.sum()),
                             "r_pred_obs": rr, "r_null": nr,
                             "align_star": a_star, "align_other": a_other,
                             "align_perm_null": a_perm})

    return {"k": int(np.unique(role).size), "n_cond": len(conds), "alpha": alpha,
            "swap": swap, "context_swap": ctx, "susceptibility": susc,
            "leverage": fam_rows}


# --------------------------------------------------------------------------- #
def wilc(a, b):
    p = [(x, y) for x, y in zip(a, b)
         if np.isfinite(x) and np.isfinite(y) and x != y]
    if len(p) < 5:
        return np.nan
    return float(wilcoxon([q[0] for q in p], [q[1] for q in p]).pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--out", default="gain_configuration.json")
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
        net = model_name(s)
        if not (FP / net / "participation_cache_v2.npz").exists():
            print(f"[skip] {net}")
            continue
        r = run_net(net, OPK[s], rng, args.n_perm)
        if r:
            per_net[net] = r
            print(f"[done] seed {s}  k={r['k']}  n_cond={r['n_cond']}  "
                  f"swap pct role={r['swap']['role']['pct']:.3f} "
                  f"unit={r['swap']['unit']['pct']:.3f}", flush=True)
    if not per_net:
        return
    V = list(per_net.values())

    print("\n" + "=" * 80)
    print("1. COUNTERFACTUAL GAIN SWAP")
    print("   S[i,j] = motif i scored against the quotient built from gains j.")
    print("   pct = within-row percentile of the matched entry (chance 0.50)")
    print("=" * 80)
    reps = ["role", "activity", "unit", "shuffled_gains", "random_partition"]
    print(f"  {'representation':20s} {'pct':>7} {'margin':>8} {'pct>0.5':>8}")
    for rep in reps:
        p = [v["swap"][rep]["pct"] for v in V]
        m = [v["swap"][rep]["margin"] for v in V]
        print(f"  {rep:20s} {med(p):>7.3f} {med(m):>+8.3f} "
              f"{sum(x > 0.5 for x in p):>5}/{len(V)}")
    rl = [v["swap"]["role"]["pct"] for v in V]
    for rep in ["unit", "activity", "shuffled_gains", "random_partition"]:
        o = [v["swap"][rep]["pct"] for v in V]
        print(f"  role - {rep:18s} {med([a - b for a, b in zip(rl, o)]):+7.3f}"
              f"  p={wilc(rl, o):.4f}  positive in "
              f"{sum(a > b for a, b in zip(rl, o))}/{len(V)}")

    print("\n  --- context pairs: own gains vs the other condition's gains ---")
    pairs = sorted({c["pair"] for v in V for c in v["context_swap"]})
    print(f"  {'pair':34s} {'nets':>5} {'own':>7} {'swapped':>8} {'diff':>7}")
    for pr in pairs:
        rows = [c for v in V for c in v["context_swap"] if c["pair"] == pr]
        own = [x["own_a"] for x in rows] + [x["own_b"] for x in rows]
        swp = [x["swapped_a"] for x in rows] + [x["swapped_b"] for x in rows]
        print(f"  {pr:34s} {len(rows):>5} {med(own):>7.3f} {med(swp):>8.3f} "
              f"{med([a - b for a, b in zip(own, swp)]):>+7.3f}")

    print("\n" + "=" * 80)
    print("2. STRUCTURAL SUSCEPTIBILITY  d|lambda_j|/dg_i")
    print("   participation ratio over roles (low = few high-leverage roles)")
    print("=" * 80)
    pr_r = [v["susceptibility"]["pr_role"] for v in V]
    pr_a = [v["susceptibility"]["pr_activity"] for v in V]
    pr_n = [v["susceptibility"]["pr_random"] for v in V]
    kk = [v["susceptibility"]["k"] for v in V]
    print(f"  median k = {med(kk):.0f}   (PR = k would mean leverage spread evenly)")
    print(f"  {'partition':20s} {'PR':>7} {'PR/k':>7}")
    for nm, arr in (("role", pr_r), ("activity", pr_a), ("random", pr_n)):
        print(f"  {nm:20s} {med(arr):>7.2f} "
              f"{med([a / b for a, b in zip(arr, kk)]):>7.3f}")
    print(f"  role - random  {med([a - b for a, b in zip(pr_r, pr_n)]):+7.2f}"
          f"  p={wilc(pr_r, pr_n):.4f}  lower in "
          f"{sum(a < b for a, b in zip(pr_r, pr_n))}/{len(V)}")
    print(f"  role - activity{med([a - b for a, b in zip(pr_r, pr_a)]):+7.2f}"
          f"  p={wilc(pr_r, pr_a):.4f}  lower in "
          f"{sum(a < b for a, b in zip(pr_r, pr_a))}/{len(V)}")

    print("\n" + "=" * 80)
    print("3a. FIRST-ORDER VALIDITY: does s . dgbar predict the observed d|lambda|?")
    print("=" * 80)
    fams = sorted({f["family"] for v in V for f in v["leverage"]})
    print(f"  {'family':18s} {'n':>4} {'r(pred,obs)':>12} {'r null':>8} {'modes':>6}")
    for f in fams:
        rows = [x for v in V for x in v["leverage"] if x["family"] == f]
        print(f"  {f:18s} {len(rows):>4} {med([x['r_pred_obs'] for x in rows]):>12.3f} "
              f"{med([x['r_null'] for x in rows]):>8.3f} "
              f"{med([x['n_modes'] for x in rows]):>6.0f}")
    per_seed = [med([x["r_pred_obs"] for x in v["leverage"]]) for v in V]
    per_seed_n = [med([x["r_null"] for x in v["leverage"]]) for v in V]
    print(f"\n  pooled (per-net median): r = {med(per_seed):.3f}   null "
          f"{med(per_seed_n):.3f}   diff {med([a - b for a, b in zip(per_seed, per_seed_n)]):+.3f}"
          f"  p={wilc(per_seed, per_seed_n):.4f}  positive in "
          f"{sum(a > b for a, b in zip(per_seed, per_seed_n))}/{len(V)}")

    print("\n" + "=" * 80)
    print("3b. TARGETING: is dgbar aligned with the leverage of the motif's mode")
    print("    more than with the leverage of other modes?  (|cos|)")
    print("=" * 80)
    kmed = med([v["susceptibility"]["k"] for v in V])
    print(f"  isotropic reference: median |cos| of random vectors at k={kmed:.0f} "
          f"is ~{0.6745 / np.sqrt(kmed):.3f}")
    print(f"  {'family':18s} {'n':>4} {'align j*':>9} {'other modes':>12} "
          f"{'perm null':>10} {'j*-perm':>8}")
    for f in fams:
        rows = [x for v in V for x in v["leverage"] if x["family"] == f]
        a1 = [x["align_star"] for x in rows]
        a2 = [x["align_other"] for x in rows]
        a3 = [x["align_perm_null"] for x in rows]
        print(f"  {f:18s} {len(rows):>4} {med(a1):>9.3f} {med(a2):>12.3f} "
              f"{med(a3):>10.3f} {med([a - b for a, b in zip(a1, a3)]):>+8.3f}")
    s1 = [med([x["align_star"] for x in v["leverage"]]) for v in V]
    s2 = [med([x["align_other"] for x in v["leverage"]]) for v in V]
    s3 = [med([x["align_perm_null"] for x in v["leverage"]]) for v in V]
    print(f"\n  pooled (per-net median): j* {med(s1):.3f}  other modes {med(s2):.3f}"
          f"  perm null {med(s3):.3f}")
    print(f"    j* - other modes {med([a - b for a, b in zip(s1, s2)]):+.3f}"
          f"  p={wilc(s1, s2):.4f}  positive in "
          f"{sum(a > b for a, b in zip(s1, s2))}/{len(V)}")
    print(f"    j* - perm null   {med([a - b for a, b in zip(s1, s3)]):+.3f}"
          f"  p={wilc(s1, s3):.4f}  positive in "
          f"{sum(a > b for a, b in zip(s1, s3))}/{len(V)}")

    out = FP / args.out
    out.write_text(json.dumps(per_net, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
