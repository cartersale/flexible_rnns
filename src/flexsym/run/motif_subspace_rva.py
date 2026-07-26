"""Decisive test: role vs activity at the SUBSPACE level, matched k + matched
group-size distributions, across the 10-seed ensemble.

The paper's positioning hinges on connectivity role groups capturing something
the activity partition does not. The pilot's "role >> activity in 100% of motifs"
compared role at k=33 vs activity at k=45 -- not k-matched -- so it does not
settle it. Here we make the comparison fair:

  * MATCHED k: role (signed-WL) and activity partitions are taken at the SAME k
    (both cached per-k in the WL sweep: labels_struct_by_k / labels_act_by_k),
    swept over a grid so the verdict is not hostage to one k.
  * MATCHED group-size distributions: raw containment f = ||P v|| / ||v|| depends
    on the size distribution, and role/activity size profiles differ even at
    matched k. We therefore compare EXCESS containment
        Delta = f_obs - mean_over( random partitions of the SAME sizes )
    for each partition. Delta subtracts each partition's own size-baseline, so
    Delta_role - Delta_act isolates the grouping STRUCTURE (connectivity vs
    activity), independent of k and size distribution.

Decisive per-net statistic (at each matched k): median over motifs of
(Delta_role - Delta_act). Positive => connectivity adds subspace containment
beyond activity, matched on k and size. Signed-rank across the 10 seeds.

  conda run -n myenv python stepnet/run_motif_subspace_rva.py
"""

from __future__ import annotations

from flexsym import paths

import argparse
import json
from pathlib import Path

import numpy as np

import sys

from flexsym.motifs.motif_classify import npload  # noqa: E402
from flexsym.motifs.motif_participation import WL_ROOT, ACT_ROOT, ACT_NAME  # noqa: E402

FP_ROOT = paths.REPO_ROOT / "results" / "motif_fixed_points"
K_GRID = [8, 12, 16, 20, 25, 30, 40, 50]
N_PERM = 200
MAX_VECS = 40


def frac_vec(V, labels):
    """||P v|| / ||v|| for every row of V (m, N) under a partition. Vectorised.

    P projects onto the group-constant subspace; ||P v||^2 = sum_g (sum_{i in g} v_i)^2 / n_g.
    """
    cats, inv = np.unique(labels, return_inverse=True)
    G = np.zeros((len(labels), cats.size))
    G[np.arange(len(labels)), inv] = 1.0
    sizes = G.sum(0)                       # (k,)
    gs = V @ G                             # (m, k) group sums
    pnorm2 = ((gs ** 2) / sizes).sum(1)    # (m,)
    vnorm2 = (V ** 2).sum(1)
    return np.sqrt(pnorm2 / np.maximum(vnorm2, 1e-12))


def excess_and_z(V, labels, n_perm, rng):
    """Return (Delta, z) where Delta = mean_f - mean_random_f (same sizes),
    z = (mean_f - mu_rand)/sd_rand, aggregated over the vec set V."""
    obs = float(frac_vec(V, labels).mean())
    lab = np.asarray(labels).copy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(lab)                   # random partition, SAME size distribution
        null[i] = frac_vec(V, lab).mean()
    mu, sd = null.mean(), null.std()
    return obs - mu, ((obs - mu) / sd if sd > 0 else 0.0), obs


def _row_for_k(labels_by_k, k):
    counts = np.array([np.unique(labels_by_k[r]).size for r in range(labels_by_k.shape[0])])
    exact = np.where(counts == k)[0]
    return int(exact[0]) if exact.size else int(np.argmin(np.abs(counts - k)))


def collect_vecs(net_dir, ind_active):
    """signed motif eigvecs restricted to active units, per motif record."""
    cache = Path(net_dir) / "participation_cache_v2.npz"
    d = npload(cache)
    recs = list(d["records"])
    out = []
    for m in recs:
        v = m.get("vecs_act")
        if v is None or not len(v):
            continue
        V = np.asarray(v, dtype=float)
        V = V[np.all(np.isfinite(V), axis=1)]      # drop any non-finite eigvecs
        if not len(V):
            continue
        out.append({"motif": m["motif"], "rule": m["rule"], "V": V})
    return out


def run_net_rva(net, k_grid, n_perm, rng):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    act = npload(ACT_ROOT / net / ACT_NAME)
    ind_active = np.asarray(act["ind_active"], dtype=int)
    ls = wl["labels_struct_by_k"]
    la = wl["labels_act_by_k"]
    motifs = collect_vecs(FP_ROOT / net, ind_active)
    per_k = {}
    for k in k_grid:
        r = _row_for_k(ls, k)
        role = np.asarray(ls[r], dtype=int)
        actl = np.asarray(la[r], dtype=int)
        if np.unique(role).size < 2 or np.unique(actl).size < 2:
            continue
        d_diff, z_diff = [], []
        dr_l, da_l = [], []
        for m in motifs:
            V = m["V"]
            if V.shape[0] > MAX_VECS:
                V = V[np.linspace(0, V.shape[0] - 1, MAX_VECS).astype(int)]
            dr, zr, _ = excess_and_z(V, role, n_perm, rng)
            da, za, _ = excess_and_z(V, actl, n_perm, rng)
            d_diff.append(dr - da); z_diff.append(zr - za)
            dr_l.append(dr); da_l.append(da)
        per_k[k] = {
            "median_delta_diff": float(np.median(d_diff)),
            "median_z_diff": float(np.median(z_diff)),
            "frac_role_gt_act": float(np.mean(np.array(d_diff) > 0)),
            "median_delta_role": float(np.median(dr_l)),
            "median_delta_activity": float(np.median(da_l)),
            "n_motifs": len(d_diff),
        }
    return per_k


def signed_rank(vals):
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size < 3 or np.allclose(v, 0):
        return float("nan"), v.size, float(np.median(v)) if v.size else float("nan")
    from scipy.stats import wilcoxon
    _, p = wilcoxon(v)
    return float(p), v.size, float(np.median(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-"); seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    rng = np.random.default_rng(0)

    per_net = {}
    for s in seeds:
        net = f"LeakyRNN_softplus_diag_{args.size}_all_{s}"
        if not (FP_ROOT / net / "participation_cache_v2.npz").exists():
            print(f"[skip] {net}: no cache")
            continue
        per_net[net] = run_net_rva(net, K_GRID, args.n_perm, rng)
        print(f"[done] {net}")

    print(f"\n=== role vs activity, SUBSPACE, matched k + matched sizes "
          f"({len(per_net)} seeds) ===")
    print(f"{'k':>4} {'med(dRole-dAct)':>15} {'role>act seeds':>14} {'signed-rank p':>13} "
          f"{'med dRole':>10} {'med dAct':>10}")
    agg = {}
    for k in K_GRID:
        diffs = [pn[k]["median_delta_diff"] for pn in per_net.values() if k in pn]
        p, n, med = signed_rank(diffs)
        fr = float(np.mean(np.array(diffs) > 0)) if diffs else float("nan")
        dr = float(np.median([pn[k]["median_delta_role"] for pn in per_net.values() if k in pn]))
        da = float(np.median([pn[k]["median_delta_activity"] for pn in per_net.values() if k in pn]))
        agg[k] = {"median_diff": med, "frac_pos": fr, "p": p, "n": n,
                  "median_delta_role": dr, "median_delta_activity": da}
        print(f"{k:>4} {med:>15.4f} {fr*100:>13.0f}% {p:>13.4g} {dr:>10.4f} {da:>10.4f}")

    out = FP_ROOT / "subspace_role_vs_activity.json"
    json.dump({"per_net": per_net, "aggregate": agg}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
