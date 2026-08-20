"""Is containment / A_cond / mode-energy just a restatement of eigenvalue magnitude (timescale)?

Reviewer control. The motif directions are slow Jacobian modes (|lambda| near 1). If containment in
the connectivity subspace, the quotient prediction A_cond, and the top-quotient-mode energy are all
simply increasing functions of |lambda| (i.e. slow modes are trivially captured), then the
connectivity story would be a restatement of "slow modes are special". We therefore take ALL
eigenvectors of every per-condition mean Jacobian J_bar -- not just the motif direction -- and:
  1. regress each eigenvector's containment / A_cond / top-3 quotient-mode energy on |lambda| and on
     the implied timescale tau = -1/log|lambda| (stable modes);
  2. locate the classified MOTIF eigenvectors on the |lambda|->containment relationship and test
     whether they are contained MORE than their |lambda| predicts (positive residual), i.e. whether
     the motifs are special beyond their timescale.

Per condition c we have J_bar, alpha, gain g (from recover_alpha_gain, stored in load_conditions),
the connectivity partition (H, sizes), Qp = (1-alpha) I + alpha diag(gbar) Qs and its top-3 lifted
modes B (the A_cond basis, r=3), and the orthonormal quotient modes M for mode energy. For each
Jacobian eigenvector v (real part, unit-normalised) we record |lambda|, tau, containment, A_cond,
top-3 mode energy, and whether v is the condition's motif direction.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.run.motif_quotient_prediction import (
    condition_jacobians, load_conditions, indicator, quotient, lifted_basis,
    score, containment_norm,
)
from flexsym.run.motif_quotient_mode_concentration import quotient_modes

FP = paths.RESULTS_ROOT / "motif_fixed_points"


def net_name(activation, size, s):
    return f"LeakyRNN_{activation}_diag_{size}_all_{s}"


def run_net(net, k, m=3):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], float)
    ind = np.asarray(wl["ind_active"], int)
    jbars = condition_jacobians(net, ind, W)
    conds = load_conditions(net, ind, W, jbars)
    if len(conds) < 3:
        return None
    ls = wl["labels_struct_by_k"]
    role = np.asarray(ls[_row_for_k(ls, k)], int)
    H, sizes = indicator(role)
    S = H / np.sqrt(sizes)[None, :]
    Qs = quotient(W, H, sizes)
    M, _ = quotient_modes(W, role)                       # orthonormal top quotient modes (|lambda| order)
    M3 = M[:, :3]

    rows = []
    for c in conds:
        Jb = c["Jb"]; alpha = c["alpha"]; g = c["g"]
        gbar = (H.T @ g) / sizes
        Qp = (1 - alpha) * np.eye(len(sizes)) + alpha * (gbar[:, None] * Qs)
        B, _ = lifted_basis(Qp, H, sizes, m)
        lam, V = np.linalg.eig(Jb)
        mv = c["v"]                                      # motif direction (unit)
        for i in range(len(lam)):
            v = np.real(V[:, i]); nv = np.linalg.norm(v)
            if nv < 1e-9:
                continue
            v = v / nv
            al = float(np.abs(lam[i]))
            PHv = containment_norm(v, H, sizes)
            cont = PHv / 1.0                             # ||P_H v||/||v||, ||v||=1
            _, acond = score(v, B, PHv)
            e_in = np.linalg.norm(S.T @ v) ** 2          # in-subspace energy
            top3 = (np.linalg.norm(M3.T @ v) ** 2 / e_in) if e_in > 1e-12 else np.nan
            is_motif = float(np.abs(mv @ v))             # |alignment| with the motif direction
            rows.append((al, cont, acond, top3, is_motif))
    a = np.array(rows)                                   # (n_vec, 5): |lam|, cont, acond, top3, motif_align
    return {"net": net, "arr": a}


def regress(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5 or np.std(x) == 0:
        return {"r": np.nan, "r2": np.nan, "slope": np.nan, "n": int(len(x))}
    r, _ = pearsonr(x, y); rho, _ = spearmanr(x, y)
    slope = np.polyfit(x, y, 1)[0]
    return {"r": float(r), "rho": float(rho), "r2": float(r ** 2), "slope": float(slope), "n": int(len(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation", default="softplus"); ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seeds", default="0-9"); ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--motif-thresh", type=float, default=0.9)
    args = ap.parse_args()
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            lo, hi = part.split("-"); seeds += list(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))

    per = {}
    for s in seeds:
        net = net_name(args.activation, args.size, s)
        if not (FP / net / "participation_cache_v2.npz").exists():
            print(f"[skip] {net}"); continue
        r = run_net(net, args.k)
        if r: per[net] = r
        print(f"[done] {net}: {len(r['arr'])} eigenvectors", flush=True)

    # pool all eigenvectors
    A = np.vstack([per[n]["arr"] for n in per])
    lam, cont, acond, top3, malign = A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4]
    stable = lam < 1.0
    tau = np.full_like(lam, np.nan)
    tau[stable] = -1.0 / np.log(np.clip(lam[stable], 1e-6, 0.999999))
    logtau = np.log(np.clip(tau, 1e-6, None))

    print(f"\n=== regress on |lambda| (all {len(lam)} Jacobian eigenvectors, {args.activation}, k={args.k}) ===")
    for name, y in (("containment", cont), ("A_cond", acond), ("top3 mode energy", top3)):
        rl = regress(lam, y); rt = regress(logtau[stable], y[stable])
        print(f"  {name:18s}: vs |lambda|  r={rl['r']:+.3f} r2={rl['r2']:.3f} slope={rl['slope']:+.3f}"
              f"  | vs log tau (stable) r={rt['r']:+.3f} r2={rt['r2']:.3f}")

    # binned shape of containment vs |lambda|
    print(f"\n=== mean containment / A_cond by |lambda| bin ===")
    edges = [0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0, 1.05, 10]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (lam >= lo) & (lam < hi)
        if sel.sum() < 5: continue
        print(f"  |lam| [{lo:.2f},{hi:.2f}): n={int(sel.sum()):>6}  cont {cont[sel].mean():.3f}  "
              f"A_cond {np.nanmean(acond[sel]):.3f}  top3 {np.nanmean(top3[sel]):.3f}")

    # are the MOTIF eigenvectors contained beyond what |lambda| predicts?
    print(f"\n=== motif eigenvectors vs the |lambda|->containment fit (motif = |align|>{args.motif_thresh}) ===")
    per_seed_resid = []
    coef = np.polyfit(lam[np.isfinite(cont)], cont[np.isfinite(cont)], 1)
    for n in per:
        a = per[n]["arr"]; mm = a[:, 4] > args.motif_thresh
        if mm.sum() == 0: continue
        pred = np.polyval(coef, a[mm, 0])
        resid = a[mm, 1] - pred
        per_seed_resid.append(float(np.nanmean(resid)))
    per_seed_resid = np.array(per_seed_resid)
    n_motif = int((malign > args.motif_thresh).sum())
    print(f"  {n_motif} motif eigenvectors; mean containment {cont[malign>args.motif_thresh].mean():.3f} "
          f"vs |lambda|-predicted {np.polyval(coef, lam[malign>args.motif_thresh]).mean():.3f}")
    if len(per_seed_resid) > 1:
        p = wilcoxon(per_seed_resid).pvalue
        print(f"  per-seed mean residual (obs - |lambda|-fit): {per_seed_resid.mean():+.3f} "
              f"({int((per_seed_resid>0).sum())}/{len(per_seed_resid)} seeds >0, signed-rank p={p:.4f})")

    # MATCHED-|lambda| control: motif vs non-motif eigenvectors at the SAME |lambda| (within a
    # window), per net. This is the fair test -- it removes the slow-mode (timescale) effect.
    print(f"\n=== matched-|lambda| control: motif containment minus non-motif at same |lambda| (window +-{0.02}) ===")
    win = 0.02
    per_seed_matched = []
    for n in per:
        a = per[n]["arr"]; lamn = a[:, 0]; contn = a[:, 1]; mm = a[:, 4] > args.motif_thresh
        diffs = []
        for idx in np.where(mm)[0]:
            near = (np.abs(lamn - lamn[idx]) < win) & (a[:, 4] <= args.motif_thresh)
            if near.sum() >= 5:
                diffs.append(contn[idx] - contn[near].mean())
        if diffs:
            per_seed_matched.append(float(np.mean(diffs)))
    per_seed_matched = np.array(per_seed_matched)
    if len(per_seed_matched) > 1:
        pm = wilcoxon(per_seed_matched).pvalue
        print(f"  motif - matched-|lambda| non-motif containment: {per_seed_matched.mean():+.3f} "
              f"({int((per_seed_matched>0).sum())}/{len(per_seed_matched)} seeds >0, signed-rank p={pm:.4f})")

    # matched-|lambda| for A_cond and top3 too
    for col, label in ((2, "A_cond"), (3, "top3 mode energy")):
        ps = []
        for n in per:
            a = per[n]["arr"]; lamn = a[:, 0]; yv = a[:, col]; mm = a[:, 4] > args.motif_thresh
            diffs = []
            for idx in np.where(mm)[0]:
                near = (np.abs(lamn - lamn[idx]) < win) & (a[:, 4] <= args.motif_thresh)
                if near.sum() >= 5 and np.isfinite(yv[idx]):
                    diffs.append(yv[idx] - np.nanmean(yv[near]))
            if diffs:
                ps.append(float(np.nanmean(diffs)))
        ps = np.array(ps)
        if len(ps) > 1:
            print(f"  {label:16s}: motif - matched non-motif {ps.mean():+.3f} "
                  f"({int((ps>0).sum())}/{len(ps)} >0, p={wilcoxon(ps).pvalue:.4f})")

    out = {"activation": args.activation, "k": args.k, "n_eigvec": int(len(lam)),
           "regress_lambda": {name: regress(lam, y) for name, y in
                              (("containment", cont), ("A_cond", acond), ("top3", top3))},
           "regress_logtau": {name: regress(logtau[stable], y[stable]) for name, y in
                              (("containment", cont), ("A_cond", acond), ("top3", top3))},
           "motif_resid_per_seed": per_seed_resid.tolist(), "lambda_fit": coef.tolist()}
    (FP / f"eigvec_lambda_regression_{args.activation}.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote eigvec_lambda_regression_{args.activation}.json")


if __name__ == "__main__":
    main()
