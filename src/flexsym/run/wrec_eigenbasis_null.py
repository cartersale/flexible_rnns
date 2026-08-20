"""W_rec-eigenbasis null for containment and top-3 motif-energy concentration.

Random and activity partitions are weak nulls. The eigenbasis of W_rec itself is the strong
null a reviewer names first: does the connectivity quotient beat "just use the top eigenvectors
of W_rec" at (a) containing the motif directions and (b) concentrating motif energy on a few
modes? The compression table already hinted this is close (A_raw 0.479 vs 0.466 at k=20, r=3).

Per seed and per k we compare, matched on dimension:
  containment  ||P p||/||p|| onto the k-dim group-constant subspace col(H) vs onto the span of
               the top-k eigenvectors of W_rec (real parts, orthonormalised).
  top-3 energy fraction of aggregate motif energy on the top-3 modes: quotient eigenmodes
               (ordered by |lambda_Q|) vs top-3 eigenvectors of W_rec (ordered by |lambda|).
Signed-rank across the ten seeds. Writes wrec_eigenbasis_null.json.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from scipy.stats import wilcoxon

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.run.motif_quotient_prediction import condition_jacobians, load_conditions, indicator
from flexsym.run.motif_quotient_mode_concentration import quotient_modes

FP = paths.RESULTS_ROOT / "motif_fixed_points"
NET = lambda s: f"LeakyRNN_softplus_diag_256_all_{s}"
K_GRID = [8, 12, 16, 20, 25, 30, 40, 50]


def wrec_eigvecs(W, k):
    """Top-k eigenvectors of W by |lambda|, real parts, orthonormalised (dim <= k)."""
    lam, V = np.linalg.eig(W)
    order = np.argsort(-np.abs(lam))[:k]
    B, _ = np.linalg.qr(np.real(V[:, order]))
    return B, np.abs(lam)[order]


def contained(P_cols, p):
    """||projection of p onto span(orthonormal columns P_cols)|| / ||p||."""
    return float(np.linalg.norm(P_cols.T @ p) / (np.linalg.norm(p) + 1e-12))


def top3_energy(modes, P):
    """Fraction of aggregate motif energy (rows of P) on the top-3 columns of `modes`."""
    E = ((P @ modes) ** 2).sum(0)
    tot = E.sum()
    return float(np.sort(E)[::-1][:3].sum() / tot) if tot > 0 else np.nan


def run_net(net, kg):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], float)
    ind = np.asarray(wl["ind_active"], int)
    jbars = condition_jacobians(net, ind, W)
    conds = load_conditions(net, ind, W, jbars)
    P = np.array([c["v"] for c in conds])                 # motif directions (unit)
    ls = wl["labels_struct_by_k"]
    out = {}
    for k in kg:
        role = np.asarray(ls[_row_for_k(ls, k)], int)
        kk = int(role.max() + 1)
        if kk < 2:
            continue
        # connectivity: group-constant subspace (orthonormal) + quotient eigenmodes
        H, sizes = indicator(role)
        S = H / np.sqrt(sizes)[None, :]                   # orthonormal columns spanning col(H)
        Mq, _ = quotient_modes(W, role)                   # lifted quotient eigenmodes (k)
        # W_rec eigenbasis matched to dimension k
        E, _ = wrec_eigvecs(W, kk)
        cont_conn = np.array([contained(S, p) for p in P])
        cont_eig = np.array([contained(E, p) for p in P])
        out[k] = {
            "k": kk, "n_motifs": len(conds),
            "contain_conn": float(np.median(cont_conn)),
            "contain_wrec_eig": float(np.median(cont_eig)),
            "contain_conn_gt_eig_frac": float(np.mean(cont_conn > cont_eig)),
            "top3_conn": top3_energy(Mq, P),
            "top3_wrec_eig": top3_energy(E, P),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--k-grid", default=None)
    args = ap.parse_args()
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-"); seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    kg = [int(x) for x in args.k_grid.split(",")] if args.k_grid else K_GRID

    per = {}
    for s in seeds:
        net = NET(s)
        if not (FP / net / "participation_cache_v2.npz").exists():
            print(f"[skip] {net}"); continue
        per[net] = run_net(net, kg)
        print(f"[done] {net}", flush=True)

    print(f"\n=== W_rec-eigenbasis null vs connectivity quotient (median/10) ===")
    print(f"  {'k':>4} | {'contain conn':>12} {'contain eig':>12} {'conn>eig(motif%)':>16} "
          f"{'p':>7} | {'top3 conn':>10} {'top3 eig':>10} {'p':>7}")
    for k in kg:
        rows = [per[n][k] for n in per if k in per[n]]
        if not rows:
            continue
        cc = [r["contain_conn"] for r in rows]; ce = [r["contain_wrec_eig"] for r in rows]
        tc = [r["top3_conn"] for r in rows]; te = [r["top3_wrec_eig"] for r in rows]
        wf = np.median([r["contain_conn_gt_eig_frac"] for r in rows])
        pc = wilcoxon(cc, ce).pvalue if len(cc) > 1 and any(a != b for a, b in zip(cc, ce)) else np.nan
        pt = wilcoxon(tc, te).pvalue if len(tc) > 1 else np.nan
        print(f"  {k:>4} | {np.median(cc):>12.3f} {np.median(ce):>12.3f} {wf:>16.2f} "
              f"{pc:>7.3f} | {np.median(tc):>10.3f} {np.median(te):>10.3f} {pt:>7.3f}")

    (FP / "wrec_eigenbasis_null.json").write_text(json.dumps(per, indent=2, default=float))
    print(f"\nwrote {FP/'wrec_eigenbasis_null.json'}")


if __name__ == "__main__":
    main()
