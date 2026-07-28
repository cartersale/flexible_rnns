"""Alternative k-dimensional subspaces for motif-direction containment, plus group-size/singleton
reporting for the connectivity partitions.

Adds to the containment picture (\S containment / \S wrec-null) an INPUT-WEIGHT-SPAN subspace: the
span of the top-k right-singular directions of W_in over the active units (a k-matched subspace
defined purely by the task-input map, not by recurrence). We report, per seed and k, the median
motif-direction containment onto
  * the connectivity group-constant subspace col(H),
  * the top-k eigenvectors of W_rec (strong recurrent null; from wrec_eigenbasis_null),
  * the top-k input-weight-span directions.
Also reports the connectivity partition's group-size distribution and singleton count at each k
(for the methods: how many groups are singletons, size quantiles).
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
from flexsym.run.wrec_eigenbasis_null import wrec_eigvecs, contained
from flexsym.run.motif_rotation_readout_control import _model_params

FP = paths.REPO_ROOT / "results/motif_fixed_points"
NET = lambda s: f"LeakyRNN_softplus_diag_256_all_{s}"
K_GRID = [8, 12, 16, 20, 25, 30, 40, 50]


def input_span(net, ind, k):
    """Top-k right-singular directions of W_in over the active units (orthonormal, dim <= k)."""
    win = np.asarray(_model_params(net)["w_in"], float)[:, ind]   # (n_input, n_active)
    _, _, Vt = np.linalg.svd(win, full_matrices=False)            # rows = unit-space directions
    B, _ = np.linalg.qr(Vt[:k].T)
    return B


def run_net(net, kg):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], float)
    ind = np.asarray(wl["ind_active"], int)
    jbars = condition_jacobians(net, ind, W)
    conds = load_conditions(net, ind, W, jbars)
    P = np.array([c["v"] for c in conds])
    ls = wl["labels_struct_by_k"]
    out = {}
    for k in kg:
        role = np.asarray(ls[_row_for_k(ls, k)], int)
        kk = int(role.max() + 1)
        if kk < 2:
            continue
        H, sizes = indicator(role)
        S = H / np.sqrt(sizes)[None, :]
        E, _ = wrec_eigvecs(W, kk)
        Bin = input_span(net, ind, kk)
        cc = np.array([contained(S, p) for p in P])
        ce = np.array([contained(E, p) for p in P])
        ci = np.array([contained(Bin, p) for p in P])
        sizes_i = sizes.astype(int).tolist()
        out[k] = {
            "k": kk,
            "contain_conn": float(np.median(cc)),
            "contain_wrec_eig": float(np.median(ce)),
            "contain_input_span": float(np.median(ci)),
            "conn_gt_input_frac": float(np.mean(cc > ci)),
            "n_singletons": int(sum(s == 1 for s in sizes_i)),
            "size_min": int(min(sizes_i)), "size_med": float(np.median(sizes_i)),
            "size_max": int(max(sizes_i)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    args = ap.parse_args()
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-"); seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))

    per = {}
    for s in seeds:
        net = NET(s)
        if not (FP / net / "participation_cache_v2.npz").exists():
            print(f"[skip] {net}"); continue
        per[net] = run_net(net, K_GRID)
        print(f"[done] {net}", flush=True)

    print("\n=== containment onto k-matched subspaces (median/10) ===")
    print(f"  {'k':>4} | {'conn':>7} {'wrec-eig':>9} {'input-span':>10} {'conn>input%':>11} {'p(conn v input)':>15}")
    for k in K_GRID:
        rows = [per[n][k] for n in per if k in per[n]]
        if not rows: continue
        cc = [r["contain_conn"] for r in rows]; ci = [r["contain_input_span"] for r in rows]
        ce = [r["contain_wrec_eig"] for r in rows]
        wf = np.median([r["conn_gt_input_frac"] for r in rows])
        p = wilcoxon(cc, ci).pvalue if len(cc) > 1 else np.nan
        print(f"  {k:>4} | {np.median(cc):>7.3f} {np.median(ce):>9.3f} {np.median(ci):>10.3f} {wf:>11.2f} {p:>15.3f}")

    print("\n=== connectivity group sizes / singletons (median over seeds) ===")
    print(f"  {'k':>4} | {'#groups':>7} {'#singletons':>12} {'size min/med/max':>18}")
    for k in K_GRID:
        rows = [per[n][k] for n in per if k in per[n]]
        if not rows: continue
        kk = int(np.median([r["k"] for r in rows]))
        ns = int(np.median([r["n_singletons"] for r in rows]))
        smin = int(np.median([r["size_min"] for r in rows]))
        smed = float(np.median([r["size_med"] for r in rows]))
        smax = int(np.median([r["size_max"] for r in rows]))
        print(f"  {k:>4} | {kk:>7} {ns:>12} {f'{smin}/{smed:.0f}/{smax}':>18}")

    (FP / "containment_alt_subspaces.json").write_text(json.dumps(per, indent=2, default=float))
    print(f"\nwrote {FP/'containment_alt_subspaces.json'}")


if __name__ == "__main__":
    main()
