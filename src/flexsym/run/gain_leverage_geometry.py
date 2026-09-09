"""Is the gain change eigenvalue-conservative, and is rotation the better geometry?

`gain_configuration` §3b found that dgbar is LESS aligned with eigenvalue-magnitude
leverage than a role-permuted dgbar (0.071 vs 0.108, 0/10 nets). Two readings, and
this run separates them.

  (i)  STABILITY. The network deliberately avoids high-leverage roles because
       moving eigenvalues is costly -- an eigenvalue-conservative solution. If so,
       the real dgbar should produce a SMALLER eigenvalue response than
       norm-matched and distribution-matched surrogates, and the deficit should be
       largest for near-critical modes (|lambda| ~ 1), where movement is most
       consequential.

  (ii) WRONG GEOMETRY. Function may depend on eigenvector ROTATION rather than
       eigenvalue movement. d|lambda_j|/dg_i is then simply not the structural
       quantity that matters, and the anti-alignment says nothing about strategy.
       The corresponding object is ||d v_j / d g_i|| -- how much gain on role i
       rotates mode j -- computed here by finite difference with mode matching and
       phase alignment, which is robust where the analytic 1/(lam_j - lam_m)
       expression is not.

Two sensitivity matrices (k roles x k modes), both evaluated at the BASE condition
of each contrast, both column-normalised so every mode contributes equally and the
question is purely about DIRECTION in gain space:

    Sev[i,j] = d|lambda_j| / dg_i          analytic, validated in §3a
    Srot[i,j] = ||d v_j / d g_i||          finite difference

Three mode classes, all indexed in the base eigenbasis:

    star      the mode carrying the plus condition's motif direction
    critical  the 5 modes with |lambda| closest to 1 (near-neutral / slow)
    other     the rest

Two statistics per (sensitivity, class), against the real dgbar and two surrogates
-- `perm` (dgbar permuted across roles: matches norm AND distribution exactly) and
`gauss` (isotropic, norm-matched only):

    response    ||S[:,C]' dghat|| / sqrt(|C|)      how much the class actually moves
    projection  ||P_C dghat||, P_C = top-3 left singular vectors of S[:,C]
                                                   how much dgbar lives in the
                                                   high-leverage subspace

Prediction (i) is response_real / response_perm < 1, most so for `critical`.
Prediction (ii) is that the rotation matrix reverses or removes the deficit.

    conda run -n myenv python -m flexsym.run.gain_leverage_geometry --seeds 0-9
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
from flexsym.run.gain_configuration import susceptibility
from flexsym.run.motif_operator_composition import FAMILIES, key, med
from flexsym.run.motif_quotient_prediction import (
    condition_jacobians,
    indicator,
    load_conditions,
    quotient,
)
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.structural.operating_k import OPK, model_name

FP = paths.RESULTS_ROOT / "motif_fixed_points"
EPS = 1e-5
N_SURR = 200
N_CRIT = 5          # modes counted as near-critical
R_SUB = 3           # dimension of the high-leverage subspace
TOL = 1e-12
CLASSES = ["star", "critical", "other"]
SENS = ["ev", "rot"]


def qmat(Qs, g, alpha):
    return (1 - alpha) * np.eye(len(g)) + alpha * (g[:, None] * Qs)


def unit_cols(V):
    return V / np.maximum(np.linalg.norm(V, axis=0), TOL)


def rotation_sensitivity(Qs, g, alpha, eps=EPS):
    """Finite-difference ||dv_j/dg_i|| and d|lambda_j|/dg_i, mode-matched.

    Returns (Rot (k,k), dLam_fd (k,k), lam, V).
    """
    k = len(g)
    lam, V = np.linalg.eig(qmat(Qs, g, alpha))
    V = unit_cols(V)
    Rot = np.zeros((k, k))
    dLam = np.zeros((k, k))
    for i in range(k):
        gp = g.copy()
        gp[i] += eps
        lp, Vp = np.linalg.eig(qmat(Qs, gp, alpha))
        Vp = unit_cols(Vp)
        A = np.abs(V.conj().T @ Vp)
        r, c = linear_sum_assignment(-A)
        for j, cc in zip(r, c):
            v0, v1 = V[:, j], Vp[:, cc]
            ph = np.vdot(v0, v1)
            if abs(ph) > TOL:
                v1 = v1 * np.conj(ph) / abs(ph)      # phase/sign alignment
            Rot[i, j] = np.linalg.norm(v1 - v0) / eps
            dLam[i, j] = (abs(lp[cc]) - abs(lam[j])) / eps
    return Rot, dLam, lam, V


def colnorm(S):
    return S / np.maximum(np.linalg.norm(S, axis=0, keepdims=True), TOL)


def stats_for(Sc, dghat, surrogates):
    """response and projection for the real gain change and its surrogates."""
    if Sc.shape[1] == 0:
        return None
    nc = np.sqrt(Sc.shape[1])
    U = np.linalg.svd(Sc, full_matrices=False)[0][:, :min(R_SUB, Sc.shape[1])]

    def resp(x):
        return float(np.linalg.norm(Sc.T @ x) / nc)

    def proj(x):
        return float(np.linalg.norm(U.T @ x))

    out = {"response": resp(dghat), "projection": proj(dghat)}
    for name, arr in surrogates.items():
        out[f"response_{name}"] = float(np.mean([resp(x) for x in arr]))
        out[f"projection_{name}"] = float(np.mean([proj(x) for x in arr]))
    return out


def run_net(net, k, rng, n_surr=N_SURR):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], float)
    ind = np.asarray(wl["ind_active"], int)
    jbars = condition_jacobians(net, ind, W)
    conds = load_conditions(net, ind, W, jbars)
    if len(conds) < 8:
        return None
    ls = wl["labels_struct_by_k"]
    role = np.asarray(ls[_row_for_k(ls, k)], int)
    H, sizes = indicator(role)
    Qs = quotient(W, H, sizes)
    alpha = float(np.median([c["alpha"] for c in conds]))
    byk = {c["key"]: c for c in conds}
    present = set(byk)
    kk = len(sizes)

    rows = []
    fd_checks = []
    for prim, epoch, pairs in FAMILIES:
        for a, b in pairs:
            ka, kb = key(a, epoch), key(b, epoch)
            if ka not in present or kb not in present:
                continue
            ga = (H.T @ byk[ka]["g"]) / sizes
            gb = (H.T @ byk[kb]["g"]) / sizes
            dg = gb - ga
            nd = np.linalg.norm(dg)
            if nd <= TOL:
                continue
            dghat = dg / nd

            Sev_raw, lam, Vb = susceptibility(Qs, ga, alpha)
            Rot, dLam_fd, lam2, Va = rotation_sensitivity(Qs, ga, alpha)
            # validation: finite-difference vs analytic eigenvalue sensitivity
            m1, m2 = Sev_raw.ravel(), dLam_fd.ravel()
            ok = np.isfinite(m1) & np.isfinite(m2)
            if ok.sum() > 10 and np.std(m1[ok]) > TOL and np.std(m2[ok]) > TOL:
                fd_checks.append(float(np.corrcoef(m1[ok], m2[ok])[0, 1]))

            # mode classes, all in the base eigenbasis Va
            M = H @ np.real(Va)         # eigen-directions of Q lift through H, not H D^-1/2
            M /= np.maximum(np.linalg.norm(M, axis=0), TOL)
            energy = (M.T @ byk[kb]["v"]) ** 2
            jstar = int(np.argmax(energy))
            crit = list(np.argsort(np.abs(np.abs(lam2) - 1.0))[:N_CRIT])
            idx = {"star": [jstar],
                   "critical": [j for j in crit if j != jstar],
                   "other": [j for j in range(kk)
                             if j != jstar and j not in crit]}

            # size-stratified permutation: dgbar_i is a mean over cell i, so its
            # magnitude falls with cell size. If leverage also tracks size, a plain
            # permutation would manufacture suppression. This surrogate shuffles
            # only WITHIN size quartiles, holding that relationship fixed.
            bins = np.digitize(sizes, np.quantile(sizes, [0.25, 0.5, 0.75]))
            blocks = [np.where(bins == bnum)[0] for bnum in np.unique(bins)]
            perm_size = []
            for _ in range(n_surr):
                x = dghat.copy()
                for blk in blocks:
                    x[blk] = x[rng.permutation(blk)]
                perm_size.append(x)

            surr = {
                "perm": [dghat[rng.permutation(kk)] for _ in range(n_surr)],
                "permsize": perm_size,
                "gauss": [(lambda z: z / np.linalg.norm(z))(rng.standard_normal(kk))
                          for _ in range(n_surr)],
            }
            Smats = {"ev": colnorm(Sev_raw), "rot": colnorm(Rot)}

            rec = {"family": f"{prim}@{epoch}", "pair": f"{a}->{b}",
                   "n_star": 1, "n_crit": len(idx["critical"]),
                   "n_other": len(idx["other"]),
                   "r_absdg_size": float(np.corrcoef(np.abs(dghat), sizes)[0, 1]),
                   "r_evlev_size": float(np.corrcoef(
                       np.linalg.norm(colnorm(Sev_raw), axis=1), sizes)[0, 1])}
            for tname, Sm in Smats.items():
                for cname in CLASSES:
                    cols = idx[cname]
                    if not cols:
                        continue
                    st = stats_for(Sm[:, cols], dghat, surr)
                    if st:
                        rec[f"{tname}_{cname}"] = st
            rows.append(rec)

    return {"k": kk, "n_cond": len(conds), "alpha": alpha,
            "fd_vs_analytic_r": float(np.nanmedian(fd_checks)) if fd_checks else np.nan,
            "contrasts": rows}


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
    ap.add_argument("--n-surr", type=int, default=N_SURR)
    ap.add_argument("--out", default="gain_leverage_geometry.json")
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
        r = run_net(net, OPK[s], rng, args.n_surr)
        if r:
            per_net[net] = r
            print(f"[done] seed {s}  k={r['k']}  contrasts={len(r['contrasts'])}  "
                  f"FD-vs-analytic r={r['fd_vs_analytic_r']:.4f}", flush=True)
    if not per_net:
        return
    V = list(per_net.values())

    print(f"\nfinite-difference vs analytic d|lambda|/dg, median r = "
          f"{med([v['fd_vs_analytic_r'] for v in V]):.4f}  (validates the FD step)")

    def per_seed(tname, cname, field):
        out = []
        for v in V:
            vals = [c[f"{tname}_{cname}"][field] for c in v["contrasts"]
                    if f"{tname}_{cname}" in c]
            out.append(med(vals) if vals else np.nan)
        return out

    for stat in ("response", "projection"):
        print("\n" + "=" * 84)
        lab = ("how much the class MOVES" if stat == "response"
               else "how much dgbar lives in the top-3 leverage subspace")
        print(f"{stat.upper()} -- {lab}")
        print("=" * 84)
        print(f"  {'sensitivity':12s} {'class':10s} {'real':>7} {'perm':>7} "
              f"{'r/perm':>7} {'p':>7} | {'permsize':>9} {'r/psize':>8} "
              f"{'<psize':>7} {'p':>7}")
        for tname in SENS:
            for cname in CLASSES:
                real = per_seed(tname, cname, stat)
                perm = per_seed(tname, cname, f"{stat}_perm")
                psz = per_seed(tname, cname, f"{stat}_permsize")
                if not np.any(np.isfinite(real)):
                    continue

                def ratio(den):
                    return med([a / b for a, b in zip(real, den)
                                if np.isfinite(a) and np.isfinite(b) and b > 0])

                nlow = sum(a < b for a, b in zip(real, psz)
                           if np.isfinite(a) and np.isfinite(b))
                lname = {"ev": "eigenvalue", "rot": "rotation"}[tname]
                print(f"  {lname:12s} {cname:10s} {med(real):>7.3f} {med(perm):>7.3f} "
                      f"{ratio(perm):>7.3f} {wilc(real, perm):>7.4f} | "
                      f"{med(psz):>9.3f} {ratio(psz):>8.3f} "
                      f"{nlow:>4}/{len(V)} {wilc(real, psz):>7.4f}")

    print("\n" + "=" * 84)
    print("EIGENVALUE vs ROTATION, same class -- is the suppression geometry-specific?")
    print("  ratio = real/permsize; positive diff = rotation spared relative to spectrum")
    print("=" * 84)
    print(f"  {'class':10s} {'ev ratio':>9} {'rot ratio':>10} {'diff':>7} "
          f"{'rot>ev':>8} {'p':>8}")
    for cname in CLASSES:
        rev, rrot = [], []
        for v in V:
            for tn, acc in (("ev", rev), ("rot", rrot)):
                vals = [c[f"{tn}_{cname}"]["response"]
                        / c[f"{tn}_{cname}"]["response_permsize"]
                        for c in v["contrasts"]
                        if f"{tn}_{cname}" in c
                        and c[f"{tn}_{cname}"]["response_permsize"] > 0]
                acc.append(med(vals) if vals else np.nan)
        print(f"  {cname:10s} {med(rev):>9.3f} {med(rrot):>10.3f} "
              f"{med([a - b for a, b in zip(rrot, rev)]):>+7.3f} "
              f"{sum(a > b for a, b in zip(rrot, rev)):>5}/{len(V)} "
              f"{wilc(rrot, rev):>8.4f}")

    print("\n" + "=" * 84)
    print("RELATIVE ALLOCATION -- leverage on a class divided by leverage on the")
    print("  bulk ('other'). Absolute suppression is size-driven; the ratio is not.")
    print("=" * 84)
    print(f"  {'sensitivity':12s} {'class':10s} {'real':>7} {'permsize':>9} "
          f"{'ratio':>7} {'real>ps':>8} {'p':>8}")
    for tname in SENS:
        for cname in ("star", "critical"):
            R, P = [], []
            for v in V:
                rr, pp = [], []
                for c in v["contrasts"]:
                    a, o = c.get(f"{tname}_{cname}"), c.get(f"{tname}_other")
                    if not a or not o:
                        continue
                    if o["response"] > 0:
                        rr.append(a["response"] / o["response"])
                    if o["response_permsize"] > 0:
                        pp.append(a["response_permsize"] / o["response_permsize"])
                R.append(med(rr) if rr else np.nan)
                P.append(med(pp) if pp else np.nan)
            lname = {"ev": "eigenvalue", "rot": "rotation"}[tname]
            print(f"  {lname:12s} {cname:10s} {med(R):>7.3f} {med(P):>9.3f} "
                  f"{med(R) / med(P):>7.3f} "
                  f"{sum(a > b for a, b in zip(R, P) if np.isfinite(a) and np.isfinite(b)):>5}"
                  f"/{len(V)} {wilc(R, P):>8.4f}")

    rs = [med([c["r_absdg_size"] for c in v["contrasts"]]) for v in V]
    rl = [med([c["r_evlev_size"] for c in v["contrasts"]]) for v in V]
    print(f"\n  size diagnostics: corr(|dgbar_i|, cell size) = {med(rs):+.3f}; "
          f"corr(eigenvalue leverage, cell size) = {med(rl):+.3f}")

    print("\n" + "=" * 84)
    print("PER-FAMILY response ratio real/perm  (<1 = conservative)")
    print("=" * 84)
    fams = sorted({c["family"] for v in V for c in v["contrasts"]})
    print(f"  {'family':18s} {'ev star':>9} {'ev crit':>9} {'ev other':>9} "
          f"{'rot star':>9} {'rot crit':>9} {'rot other':>9}")
    for f in fams:
        cs = [c for v in V for c in v["contrasts"] if c["family"] == f]
        cells = []
        for tname in SENS:
            for cname in CLASSES:
                rr = [c[f"{tname}_{cname}"]["response"]
                      / c[f"{tname}_{cname}"]["response_perm"]
                      for c in cs if f"{tname}_{cname}" in c
                      and c[f"{tname}_{cname}"]["response_perm"] > 0]
                cells.append(med(rr))
        print(f"  {f:18s} " + " ".join(f"{x:>9.3f}" for x in cells))

    out = FP / args.out
    out.write_text(json.dumps(per_net, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
