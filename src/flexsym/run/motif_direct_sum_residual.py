"""Residualised direct sum: is C's motif direction in span(A) + span(B) once the
subspace shared by ALL conditions is removed?

`motif_operator_composition` test C was ceiling-limited. Its null sat at 0.77-0.92
because motif directions concentrate globally: a 6-12 dimensional span built from
ANY two conditions already captures ~80% of any motif direction inside col(H).
That shared concentration swamps the contrast the test is trying to measure, so
its negative could not be trusted.

Fix. Build a global subspace G from the motif directions of the OTHER conditions
-- leave-triple-out, so C, A and B never contribute to the thing being projected
out -- and run the identical test in the orthogonal complement:

    v_C^p   = (I - P_G) v_C
    Ba^p    = orth (I - P_G) Ba          (same for Bb)
    S^p     = orth (I - P_G) S           S = H/sqrt(sizes), basis of col(H)
    rho     = ||[Ba^p Bb^p]' v_C^p|| / ||S^p' v_C^p||

Both the observed span and every null span are truncated to the SAME dimension d,
so the comparison stays dimension-matched after residualisation (a residualised
span can lose rank if it lay inside G).

r = 0 reproduces the original test. Sweeping r is the point: if compositional
structure exists but is hidden under shared concentration, the observed-minus-null
margin should GROW with r while the null falls away from its ceiling.

Reported per (net, triple): rho, the two single-operand rhos, the null mean, an
empirical one-sided p from the mismatched-pair null, and the gain over the best
single operand. Seed remains the unit of replication.

    conda run -n myenv python -m flexsym.run.motif_direct_sum_residual --seeds 0-9
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import wilcoxon

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_operator_composition import TRIPLES, med, top_span
from flexsym.run.motif_quotient_prediction import (
    condition_jacobians,
    indicator,
    load_conditions,
    quotient,
)
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.structural.operating_k import OPK, model_name

FP = paths.RESULTS_ROOT / "motif_fixed_points"
R_GRID = [0, 1, 2, 3, 5, 8]      # dimensions of the shared subspace removed
TOP_M = 3
N_NULL = 200
TOL = 1e-9


def orth(M, tol=TOL):
    """Orthonormal basis for col(M), dropping rank-deficient directions."""
    if M.size == 0 or M.shape[1] == 0:
        return M.reshape(M.shape[0], 0)
    Q, R = np.linalg.qr(M)
    keep = np.abs(np.diag(R)) > tol * max(1.0, np.abs(np.diag(R)).max())
    return Q[:, keep]


def global_subspace(V, r):
    """Top-r principal directions of the motif directions V (rows), unit-norm."""
    if r == 0 or len(V) == 0:
        return np.zeros((V.shape[1], 0))
    U, s, _ = np.linalg.svd(np.asarray(V).T, full_matrices=False)
    return U[:, :min(r, (s > TOL).sum())]


def residualise(X, G):
    """(I - P_G) X, then orthonormalise."""
    if G.shape[1] == 0:
        return orth(X) if X.ndim == 2 else X
    if X.ndim == 1:
        return X - G @ (G.T @ X)
    return orth(X - G @ (G.T @ X))


def proj_norm(B, v, d=None):
    """||B'v|| using at most d columns of B."""
    if B.shape[1] == 0:
        return 0.0
    if d is not None:
        B = B[:, :d]
    return float(np.linalg.norm(B.T @ v))


def run_net(net, k, rng, n_null=N_NULL):
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
    S = H / np.sqrt(sizes)[None, :]

    byk = {c["key"]: c for c in conds}
    spans = {c["key"]: top_span(quotient(c["Jb"], H, sizes), H, sizes, TOP_M)
             for c in conds}

    out = {}
    for kc, ka, kb in TRIPLES:
        if not all(x in byk for x in (kc, ka, kb)):
            continue
        v = byk[kc]["v"]
        pool = [x for x in spans if x not in (kc, ka, kb)]
        if len(pool) < 4:
            continue
        # leave-triple-out global subspace: the triple never defines what is removed
        Vother = np.array([byk[x]["v"] for x in pool])
        draws = [rng.choice(len(pool), 2, replace=False) for _ in range(n_null)]

        per_r = {}
        for r in R_GRID:
            G = global_subspace(Vother, r)
            vr = residualise(v, G)
            nv = np.linalg.norm(vr)
            Sr = residualise(S, G)
            denom = proj_norm(Sr, vr)
            if nv < TOL or denom < TOL:
                continue
            Ba, Bb = residualise(spans[ka], G), residualise(spans[kb], G)
            Bab = orth(np.hstack([Ba, Bb])) if Ba.shape[1] + Bb.shape[1] else Ba
            d = Bab.shape[1]
            if d == 0:
                continue
            rho = proj_norm(Bab, vr) / denom
            ra = proj_norm(Ba, vr) / denom
            rb = proj_norm(Bb, vr) / denom
            nulls = []
            for i, j in draws:
                Bn = orth(np.hstack([residualise(spans[pool[i]], G),
                                     residualise(spans[pool[j]], G)]))
                if Bn.shape[1] < d:
                    continue
                nulls.append(proj_norm(Bn, vr, d) / denom)
            if len(nulls) < 20:
                continue
            nulls = np.array(nulls)
            mu, sd = float(nulls.mean()), float(nulls.std())
            per_r[r] = {
                "dim": int(d),
                "rho_AB": rho, "rho_A": ra, "rho_B": rb,
                "null": mu,
                "z": float((rho - mu) / sd) if sd > 0 else np.nan,
                "p": float((1 + int((nulls >= rho).sum())) / (1 + len(nulls))),
                "gain_over_best_single": float(rho - max(ra, rb)),
                "n_null": len(nulls),
            }
        if per_r:
            out[kc] = {"A": ka, "B": kb, "by_r": per_r}
    return {"k": int(np.unique(role).size), "n_cond": len(conds), "triples": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--n-null", type=int, default=N_NULL)
    ap.add_argument("--out", default="direct_sum_residual.json")
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
        r = run_net(net, OPK[s], rng, args.n_null)
        if r:
            per_net[net] = r
            print(f"[done] seed {s}  k={r['k']}  triples={len(r['triples'])}",
                  flush=True)
    if not per_net:
        return
    V = list(per_net.values())

    def rows(t, r):
        return [v["triples"][t]["by_r"][r] for v in V
                if t in v["triples"] and r in v["triples"][t]["by_r"]]

    triples = sorted({t for v in V for t in v["triples"]})

    print("\n" + "=" * 86)
    print("RESIDUALISED DIRECT SUM -- r = dimensions of the shared subspace removed")
    print("  rho = ||P_[A,B] v_C|| / ||P_colH v_C||, all in the residual space")
    print("=" * 86)
    for t in triples:
        print(f"\n  {t}   (A = {[v['triples'][t]['A'] for v in V if t in v['triples']][0]}"
              f", B = {[v['triples'][t]['B'] for v in V if t in v['triples']][0]})")
        print(f"    {'r':>3} {'nets':>5} {'dim':>4} {'rho_AB':>7} {'rho_A':>7} "
              f"{'rho_B':>7} {'null':>7} {'margin':>7} {'z':>6} {'p<.05':>6}")
        for r in R_GRID:
            rr = rows(t, r)
            if not rr:
                continue
            marg = [x["rho_AB"] - x["null"] for x in rr]
            print(f"    {r:>3} {len(rr):>5} {med([x['dim'] for x in rr]):>4.0f} "
                  f"{med([x['rho_AB'] for x in rr]):>7.3f} "
                  f"{med([x['rho_A'] for x in rr]):>7.3f} "
                  f"{med([x['rho_B'] for x in rr]):>7.3f} "
                  f"{med([x['null'] for x in rr]):>7.3f} "
                  f"{med(marg):>+7.3f} {med([x['z'] for x in rr]):>6.2f} "
                  f"{sum(x['p'] < 0.05 for x in rr):>3}/{len(rr)}")

    print("\n" + "=" * 86)
    print("POOLED over triples (per-net median across its triples, seed = unit)")
    print("=" * 86)
    print(f"  {'r':>3} {'nets':>5} {'rho_AB':>7} {'null':>7} {'margin':>7} "
          f"{'>null':>7} {'p_wilc':>8} {'gain/best1':>11} {'z':>6}")
    for r in R_GRID:
        obs, nul, gain, zz = [], [], [], []
        for v in V:
            o = [v["triples"][t]["by_r"][r] for t in v["triples"]
                 if r in v["triples"][t]["by_r"]]
            if not o:
                continue
            obs.append(med([x["rho_AB"] for x in o]))
            nul.append(med([x["null"] for x in o]))
            gain.append(med([x["gain_over_best_single"] for x in o]))
            zz.append(med([x["z"] for x in o]))
        if not obs:
            continue
        pairs = [(a, b) for a, b in zip(obs, nul) if np.isfinite(a) and np.isfinite(b)]
        p = (wilcoxon([x[0] for x in pairs], [x[1] for x in pairs]).pvalue
             if len(pairs) >= 5 else np.nan)
        print(f"  {r:>3} {len(obs):>5} {med(obs):>7.3f} {med(nul):>7.3f} "
              f"{med([a - b for a, b in zip(obs, nul)]):>+7.3f} "
              f"{sum(a > b for a, b in zip(obs, nul)):>4}/{len(obs)} "
              f"{p:>8.4f} {med(gain):>+11.3f} {med(zz):>6.2f}")

    out = FP / args.out
    out.write_text(json.dumps(per_net, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
