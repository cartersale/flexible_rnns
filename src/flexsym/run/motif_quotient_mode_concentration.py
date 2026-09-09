"""Do different tasks' motifs concentrate on a SHARED few connectivity-quotient modes?

Not "are the motif directions similar in full space" -- they are (all slow modes), and a
random partition preserves that (the eig-matched control). The sharper question: express
each task-epoch motif in the coordinates of the connectivity quotient's eigenmodes, and ask
whether the COLLECTION of motifs lands on a shared low-dimensional set of those modes that a
random quotient would NOT concentrate them onto.

Setup, per network:
  - quotient Q = (H'H)^-1 H' W H on the connectivity partition; eigenvectors of Q, lifted to
    units and orthonormalised, give k quotient modes M (ordered by |lambda|, i.e. by the
    coarse-grained dynamics).
  - each motif direction p_m (unit vector) projects to mode coordinates c_mj = <M_j, p_m>.
  - aggregate energy on mode j: E_j = sum_m c_mj^2, normalised by total contained energy.
  - CONCENTRATION = participation ratio PR = (sum E_j)^2 / sum E_j^2 (effective number of
    modes the motifs collectively occupy). Low PR = a shared low-dimensional set.
    Also: fraction of motif energy on the top-3 quotient modes.

Because a shared low-dim set requires different tasks to load the SAME modes, aggregate PR is
the right single statistic: if tasks used disjoint modes the aggregate would spread out.

Null: size-matched RANDOM partitions, each with its OWN quotient + eigenmodes; project the
same motifs and recompute PR / top-3. The connectivity quotient concentrates the motifs
specifically iff PR_conn < PR_rand (fewer shared modes) and top-3 energy is higher.

  conda run -n myenv python -m flexsym.run.motif_quotient_mode_concentration --seeds 0-2
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_reuse_control import MOTIF_TYPES, _row_for_k
from flexsym.run.motif_quotient_prediction import (
    condition_jacobians, indicator, load_conditions, quotient,
)

FP_ROOT = paths.RESULTS_ROOT / "motif_fixed_points"
K_GRID = [12, 20, 30]
N_PERM = 300
TOP_R = 3


def quotient_modes(W, labels):
    """Eigenvectors of the quotient, lifted to units and orthonormalised, sorted by |lambda|."""
    H, sizes = indicator(labels)
    Q = quotient(W, H, sizes)
    lam, V = np.linalg.eig(Q)
    order = np.argsort(-np.abs(lam))
    M = H @ np.real(V[:, order])                          # W H = H Q: eigen-directions lift via H
    M, _ = np.linalg.qr(M)                                # orthonormalise, keeps |lambda| order
    return M, np.abs(lam)[order]


def concentration(P, M):
    """PR and top-R energy fraction of the motif set P (rows unit vectors) in modes M."""
    C = P @ M                                             # (n_motifs, k) mode coordinates
    E = (C ** 2).sum(0)                                   # aggregate energy per mode
    total = E.sum()
    if total <= 0:
        return np.nan, np.nan
    pr = float(total ** 2 / np.sum(E ** 2))              # participation ratio (few modes = low)
    top = float(np.sort(E)[::-1][:TOP_R].sum() / total)  # energy on the top-R modes
    return pr, top


def run_net(net, k_grid, n_perm, rng):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], float)
    ind = np.asarray(wl["ind_active"], int)
    jbars = condition_jacobians(net, ind, W)
    conds = load_conditions(net, ind, W, jbars)
    if len(conds) < 6:
        return None
    P = np.array([c["v"] for c in conds])                 # (n_motifs, n_active) unit vectors
    ls = wl["labels_struct_by_k"]
    out = {}
    for k in k_grid:
        role = np.asarray(ls[_row_for_k(ls, k)], int)
        if np.unique(role).size < 2:
            continue
        M, _ = quotient_modes(W, role)
        pr_c, top_c = concentration(P, M)
        # size-matched random partitions, each with its own quotient modes
        _, sizes = np.unique(role, return_counts=True)
        pr_r, top_r = [], []
        for _ in range(n_perm):
            perm = np.repeat(np.arange(len(sizes)), sizes)
            rng.shuffle(perm)
            Mr, _ = quotient_modes(W, perm)
            a, b = concentration(P, Mr)
            pr_r.append(a)
            top_r.append(b)
        pr_r, top_r = np.array(pr_r), np.array(top_r)
        out[k] = {
            "k": int(np.unique(role).size), "n_motifs": len(conds),
            "pr_conn": pr_c, "pr_rand_mean": float(np.nanmean(pr_r)),
            "pr_z": float((pr_c - np.nanmean(pr_r)) / (np.nanstd(pr_r) + 1e-12)),
            "pr_p_lower": float((np.sum(pr_r <= pr_c) + 1) / (len(pr_r) + 1)),
            "top_conn": top_c, "top_rand_mean": float(np.nanmean(top_r)),
            "top_z": float((top_c - np.nanmean(top_r)) / (np.nanstd(top_r) + 1e-12)),
        }
        # per motif type: do same-type motifs (e.g. all memory rings) share modes?
        out[k]["by_type"] = {}
        for mt in MOTIF_TYPES:
            sub = P[[i for i, c in enumerate(conds) if c["motif"] == mt]]
            if len(sub) >= 3:
                pc, tc = concentration(sub, M)
                pr_rt = np.array([concentration(sub, quotient_modes(W, _shuf(sizes, rng))[0])[0]
                                  for _ in range(max(50, n_perm // 4))])
                out[k]["by_type"][mt] = {
                    "n": len(sub), "pr_conn": pc, "pr_rand_mean": float(np.nanmean(pr_rt)),
                    "pr_z": float((pc - np.nanmean(pr_rt)) / (np.nanstd(pr_rt) + 1e-12))}
    return out


def _shuf(sizes, rng):
    perm = np.repeat(np.arange(len(sizes)), sizes)
    rng.shuffle(perm)
    return perm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--activation", default="softplus")
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--out", default="quotient_mode_concentration.json")
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
        net = f"LeakyRNN_{args.activation}_diag_{args.size}_all_{s}"
        if not (FP_ROOT / net / "participation_cache_v2.npz").exists():
            print(f"[skip] {net}")
            continue
        r = run_net(net, K_GRID, args.n_perm, rng)
        if r:
            per_net[net] = r
            print(f"[done] {net}")

    print(f"\n=== motif energy concentration on quotient eigenmodes (lower PR = shared few "
          f"modes) ===")
    print(f"  {'k':>4} {'n_mot':>6} | {'PR_conn':>8} {'PR_rand':>8} {'PR_z':>6} {'p':>6} | "
          f"{'top3_conn':>9} {'top3_rand':>9} {'top3_z':>7}")
    for k in K_GRID:
        rows = [v[k] for v in per_net.values() if k in v]
        if not rows:
            continue
        print(f"  {k:>4} {int(np.median([r['n_motifs'] for r in rows])):>6} | "
              f"{np.median([r['pr_conn'] for r in rows]):>8.2f} "
              f"{np.median([r['pr_rand_mean'] for r in rows]):>8.2f} "
              f"{np.median([r['pr_z'] for r in rows]):>6.2f} "
              f"{np.median([r['pr_p_lower'] for r in rows]):>6.3f} | "
              f"{np.median([r['top_conn'] for r in rows]):>9.3f} "
              f"{np.median([r['top_rand_mean'] for r in rows]):>9.3f} "
              f"{np.median([r['top_z'] for r in rows]):>7.2f}")

    print(f"\n  by motif type at k=20 (PR_conn vs PR_rand, lower_conn = shared modes):")
    for mt in MOTIF_TYPES:
        rows = [v[20]["by_type"][mt] for v in per_net.values()
                if 20 in v and mt in v[20]["by_type"]]
        if rows:
            print(f"    {mt:>16}: n={int(np.median([r['n'] for r in rows]))} "
                  f"PR_conn {np.median([r['pr_conn'] for r in rows]):.2f} vs "
                  f"PR_rand {np.median([r['pr_rand_mean'] for r in rows]):.2f} "
                  f"(z {np.median([r['pr_z'] for r in rows]):+.2f})")

    out = FP_ROOT / args.out
    json.dump(per_net, open(out, "w"), indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
