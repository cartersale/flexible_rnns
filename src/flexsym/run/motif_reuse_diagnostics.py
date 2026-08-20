"""Diagnostics for the motif-reuse / partition-specificity question.

Three analyses, all on cross-task motif pairs (same motif type vs different
motif type), run on the valid networks (seeds 0-2):

  D1  FULL-SPACE similarity. Compare motif directions directly, before any
      partition. If same-computation directions are already far more similar in
      the full 204-d space, the reuse result belongs to the motifs, not to the
      connectivity coordinates. Reported as mean same / mean diff, R_full,
      Cohen's d and AUC, for three motif descriptors.

  D2  INCREMENTAL PRESERVATION. How much of the full-space same-vs-different
      difference survives coarse-graining onto each partition:
          rho_H = R_H / R_full        (normalised preservation ratio)
      for connectivity, size-matched random, and activity at matched k. The
      question is not whether R_H > 0 (it is, for every partition) but whether
      connectivity preserves full-space reuse BETTER than random.

  D3  PARTITION-SENSITIVE FEATURES. The |loading| profile summarised by a group
      mean may be too insensitive to distinguish partitions. Five descriptors
      that use the partition more sharply:
        signed    group-mean of the SIGNED dominant motif direction (incoherent
                  groups cancel; |loading| cannot show this)
        rank      Spearman correlation of the k group means (ordering only)
        top       Jaccard overlap of the top-k/4 highest-loading groups (identity
                  of the groups carrying the motif)
        angle     mean cosine principal angle between the projected top-3 motif
                  subspaces (uses the motif subspace, not one vector)
        conc      effective number of groups carrying the motif (concentration;
                  a per-motif scalar, not a pair statistic)
      Each is compared against size-matched random partitions and against the
      activity partition at matched k.

  conda run -n myenv python stepnet/run_motif_reuse_diagnostics.py --seeds 0-2
"""

from __future__ import annotations

from flexsym import paths

import argparse
import json
import sys
from pathlib import Path

import numpy as np


from flexsym.motifs.motif_classify import npload  # noqa: E402
from flexsym.motifs.motif_participation import WL_ROOT  # noqa: E402
from flexsym.run.motif_reuse_control import (  # noqa: E402
    MOTIF_TYPES, _row_for_k, pair_masks, project, reuse_R,
)

FP_ROOT = paths.RESULTS_ROOT / "motif_fixed_points"
K_GRID = [8, 12, 20, 30, 40, 50]
N_PERM = 100
SUB_RANK = 3          # dimensionality of the per-motif subspace for 'angle'


# --------------------------------------------------------------------------- #
# motif descriptors
# --------------------------------------------------------------------------- #
def canonical_sign(v):
    """Fix the arbitrary global sign of an eigenvector: largest |component| > 0."""
    return v if v[np.argmax(np.abs(v))] >= 0 else -v


def load_motifs(net_dir):
    """Per condition: |loading| profile p, signed dominant direction s, subspace B.

    The signed direction is the leading right singular vector of the condition's
    signed eigenvector set (sign-canonicalised). For a ring/line attractor the
    per-point eigenvector rotates, so a plain average cancels; the leading
    singular vector is the stable summary of that rotating family.
    """
    d = npload(Path(net_dir) / "participation_cache_v2.npz")
    out = []
    for m in list(d["records"]):
        if m["motif"] not in MOTIF_TYPES:
            continue
        p = np.asarray(m["p_act"], dtype=float)
        if not np.all(np.isfinite(p)) or np.linalg.norm(p) == 0:
            continue
        rec = {"motif": m["motif"], "rule": m["rule"], "epoch": m["epoch"],
               "p": p, "s": None, "B": None}
        V = m.get("vecs_act")
        if V is not None and len(V):
            V = np.asarray(V, dtype=float)
            V = V[np.all(np.isfinite(V), axis=1)]
            if len(V):
                _, _, Vt = np.linalg.svd(V, full_matrices=False)
                rec["s"] = canonical_sign(Vt[0])
                rec["B"] = Vt[:min(SUB_RANK, Vt.shape[0])]
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# pair statistics
# --------------------------------------------------------------------------- #
def _R_from_C(C, same, diff):
    if not np.any(same) or not np.any(diff):
        return float("nan")
    return float(np.nanmean(C[same]) - np.nanmean(C[diff]))


def stat_corr(P, same, diff):
    return _R_from_C(np.corrcoef(P), same, diff)


def stat_rank(P, labels, same, diff):
    """Spearman correlation of the k group means (ordering of groups only)."""
    from scipy.stats import rankdata
    cats, inv = np.unique(labels, return_inverse=True)
    G = np.zeros((len(labels), cats.size))
    G[np.arange(len(labels)), inv] = 1.0
    M = (P @ G) / G.sum(0)                       # (m, k) group means
    return _R_from_C(np.corrcoef(rankdata(M, axis=1)), same, diff)


def stat_top(P, labels, same, diff, frac=0.25):
    """Jaccard overlap of the top-frac highest-loading groups."""
    cats, inv = np.unique(labels, return_inverse=True)
    G = np.zeros((len(labels), cats.size))
    G[np.arange(len(labels)), inv] = 1.0
    M = (P @ G) / G.sum(0)
    k = M.shape[1]
    t = max(1, int(round(frac * k)))
    top = np.zeros_like(M, dtype=bool)
    idx = np.argsort(-M, axis=1)[:, :t]
    np.put_along_axis(top, idx, True, axis=1)
    inter = top.astype(float) @ top.astype(float).T
    J = inter / (2 * t - inter)
    return _R_from_C(J, same, diff)


def stat_angle(Bs, labels, same, diff):
    """Mean cosine principal angle between projected top-r motif subspaces."""
    Q = []
    for B in Bs:
        PB = project(B, labels)
        q, r = np.linalg.qr(PB.T)                # (N, r)
        keep = np.abs(np.diag(r)) > 1e-10
        Q.append(q[:, keep] if keep.any() else None)
    n = len(Q)
    C = np.full((n, n), np.nan)
    for i in range(n):
        if Q[i] is None:
            continue
        for j in range(i + 1, n):
            if Q[j] is None:
                continue
            sv = np.linalg.svd(Q[i].T @ Q[j], compute_uv=False)
            C[i, j] = C[j, i] = float(np.mean(np.clip(sv, 0, 1)))
    return _R_from_C(C, same, diff)


def concentration(P, labels):
    """Effective number of groups carrying each motif (lower = concentrated)."""
    cats, inv = np.unique(labels, return_inverse=True)
    G = np.zeros((len(labels), cats.size))
    G[np.arange(len(labels)), inv] = 1.0
    sizes = G.sum(0)
    M = (P @ G) / sizes
    E = sizes * M ** 2                            # energy per group
    W = E / np.maximum(E.sum(1, keepdims=True), 1e-12)
    return float(np.mean(1.0 / np.sum(W ** 2, axis=1)))


# --------------------------------------------------------------------------- #
# D1: full space
# --------------------------------------------------------------------------- #
def _summarize(C, s_mask, d_mask, tag):
    s, d = C[s_mask], C[d_mask]
    s, d = s[np.isfinite(s)], d[np.isfinite(d)]
    if len(s) < 2 or len(d) < 2:
        return None
    pooled = np.sqrt((s.var(ddof=1) + d.var(ddof=1)) / 2)
    auc = float(np.mean(s[:, None] > d[None, :])
                + 0.5 * np.mean(s[:, None] == d[None, :]))
    return {"feature": tag, "mean_same": float(s.mean()),
            "mean_diff": float(d.mean()), "R_full": float(s.mean() - d.mean()),
            "cohen_d": float((s.mean() - d.mean()) / pooled) if pooled > 0 else np.nan,
            "auc": auc, "n_same": int(len(s)), "n_diff": int(len(d))}


def subspace_cos_matrix(Bs):
    """Mean cosine principal angle between every pair of (orthonormalised) Bs."""
    Q = [np.linalg.qr(B.T)[0] for B in Bs]
    n = len(Q)
    C = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            sv = np.linalg.svd(Q[i].T @ Q[j], compute_uv=False)
            C[i, j] = C[j, i] = float(np.mean(np.clip(sv, 0, 1)))
    return C


def full_space(motifs, same, diff, types, have_s, sub_masks):
    out = []
    P = np.vstack([m["p"] for m in motifs])
    out.append(_summarize(np.corrcoef(P), same, diff, "abs_loading"))

    if have_s and sub_masks[0] is not None:
        s_m, d_m = sub_masks
        S = np.vstack([motifs[i]["s"] for i in have_s])
        out.append(_summarize(np.abs(np.corrcoef(S)), s_m, d_m, "signed_dominant"))
        C = subspace_cos_matrix([motifs[i]["B"] for i in have_s])
        out.append(_summarize(C, s_m, d_m, "subspace_angle"))

    per_type = {}
    C = np.corrcoef(P)
    base = np.nanmean(C[diff])
    for mt in MOTIF_TYPES:
        sel = same & (types[:, None] == mt) & (types[None, :] == mt)
        if np.any(sel):
            per_type[mt] = {"R": float(np.nanmean(C[sel]) - base),
                            "mean_same": float(np.nanmean(C[sel])),
                            "n_pairs": int(sel.sum())}
    return [o for o in out if o], per_type


# --------------------------------------------------------------------------- #
# D2 + D3: per-partition evaluation
# --------------------------------------------------------------------------- #
def eval_partition(labels, P, S, Bs, sub_masks, same, diff, R_full):
    """All partition-dependent statistics for one label vector."""
    out = {}
    out["R_abs"] = stat_corr(project(P, labels), same, diff)
    out["rho_abs"] = out["R_abs"] / R_full if R_full else np.nan
    out["rank"] = stat_rank(P, labels, same, diff)
    out["top"] = stat_top(P, labels, same, diff)
    out["conc"] = concentration(P, labels)
    if S is not None:
        s_m, d_m = sub_masks
        out["signed"] = _R_from_C(np.abs(np.corrcoef(project(S, labels))), s_m, d_m)
        out["angle"] = stat_angle(Bs, labels, s_m, d_m)
    return out


FEATURES = ["R_abs", "rho_abs", "rank", "top", "signed", "angle", "conc"]


def run_net(net, k_grid, n_perm, rng):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    motifs = load_motifs(FP_ROOT / net)
    if len(motifs) < 4:
        return None
    P = np.vstack([m["p"] for m in motifs])
    same, diff, types = pair_masks(motifs)
    if not np.any(same) or not np.any(diff):
        return None

    have_s = [i for i, m in enumerate(motifs) if m["s"] is not None]
    S = np.vstack([motifs[i]["s"] for i in have_s]) if len(have_s) > 3 else None
    Bs = [motifs[i]["B"] for i in have_s] if S is not None else None
    sub_masks = (same[np.ix_(have_s, have_s)], diff[np.ix_(have_s, have_s)]) \
        if S is not None else (None, None)

    d1, per_type = full_space(motifs, same, diff, types, have_s, sub_masks)
    R_full = reuse_R(P, same, diff)

    ls, la = wl["labels_struct_by_k"], wl["labels_act_by_k"]
    per_k = {}
    for k in k_grid:
        r = _row_for_k(ls, k)
        role = np.asarray(ls[r], dtype=int)
        actl = np.asarray(la[r], dtype=int)
        if np.unique(role).size < 2 or np.unique(actl).size < 2:
            continue
        obs = eval_partition(role, P, S, Bs, sub_masks, same, diff, R_full)
        act = eval_partition(actl, P, S, Bs, sub_masks, same, diff, R_full)
        lab = role.copy()
        null = {f: [] for f in FEATURES if f in obs}
        for _ in range(n_perm):
            rng.shuffle(lab)
            nv = eval_partition(lab, P, S, Bs, sub_masks, same, diff, R_full)
            for f in null:
                null[f].append(nv[f])
        rec = {"k_role": int(np.unique(role).size),
               "k_act": int(np.unique(actl).size)}
        for f in null:
            arr = np.array(null[f], dtype=float)
            mu, sd = np.nanmean(arr), np.nanstd(arr)
            rec[f] = {"obs": obs[f], "rand": float(mu), "act": act[f],
                      "delta": obs[f] - float(mu),
                      "z": float((obs[f] - mu) / sd) if sd > 0 else 0.0,
                      "delta_act": act[f] - float(mu)}
        per_k[k] = rec
    return {"n_motifs": len(motifs), "n_signed": len(have_s), "R_full": R_full,
            "full_space": d1, "full_space_by_type": per_type, "per_k": per_k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--k-grid", default=None)
    ap.add_argument("--out", default="reuse_diagnostics.json")
    args = ap.parse_args()
    k_grid = [int(x) for x in args.k_grid.split(",")] if args.k_grid else K_GRID

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
        net = f"LeakyRNN_softplus_diag_{args.size}_all_{s}"
        if not (FP_ROOT / net / "participation_cache_v2.npz").exists():
            print(f"[skip] {net}")
            continue
        r = run_net(net, k_grid, args.n_perm, rng)
        if r:
            per_net[net] = r
            print(f"[done] {net}: {r['n_motifs']} motifs "
                  f"({r['n_signed']} with signed dirs), R_full={r['R_full']:.3f}")

    # ---- D1 ----
    print("\n=== D1  FULL-SPACE similarity (no partition) ===")
    print(f"{'feature':16} {'seed':>5} {'same':>8} {'diff':>8} {'R_full':>8} "
          f"{'d':>7} {'AUC':>6}")
    for net, v in per_net.items():
        for f in v["full_space"]:
            print(f"{f['feature']:16} {net[-1]:>5} {f['mean_same']:>8.3f} "
                  f"{f['mean_diff']:>8.3f} {f['R_full']:>8.3f} "
                  f"{f['cohen_d']:>7.2f} {f['auc']:>6.2f}")
    print("\n  by motif type (abs_loading, full space):")
    for net, v in per_net.items():
        for mt, d in v["full_space_by_type"].items():
            print(f"    seed {net[-1]}  {mt:18} R={d['R'].__format__('+.3f')} "
                  f"(same r={d['mean_same']:.3f}, n={d['n_pairs']})")

    # ---- D2 ----
    print("\n=== D2  PRESERVATION of full-space reuse:  rho = R_H / R_full ===")
    print(f"{'k':>4} {'rho_conn':>9} {'rho_rand':>9} {'rho_act':>9} "
          f"{'d(conn-rand)':>13} {'pos':>5}")
    for k in k_grid:
        rows = [v["per_k"][k] for v in per_net.values() if k in v["per_k"]]
        if not rows:
            continue
        rc = [x["rho_abs"]["obs"] for x in rows]
        rr = [x["rho_abs"]["rand"] for x in rows]
        ra = [x["rho_abs"]["act"] for x in rows]
        dd = [x["rho_abs"]["delta"] for x in rows]
        print(f"{k:>4} {np.median(rc):>9.3f} {np.median(rr):>9.3f} "
              f"{np.median(ra):>9.3f} {np.median(dd):>13.3f} "
              f"{sum(x > 0 for x in dd):>3}/{len(dd)}")

    # ---- D3 ----
    print("\n=== D3  PARTITION-SENSITIVE features: obs vs size-matched random ===")
    for f in ["R_abs", "rank", "top", "signed", "angle", "conc"]:
        rows_any = [(k, [v["per_k"][k][f] for v in per_net.values()
                         if k in v["per_k"] and f in v["per_k"][k]])
                    for k in k_grid]
        rows_any = [(k, r) for k, r in rows_any if r]
        if not rows_any:
            continue
        print(f"\n  -- {f} --")
        print(f"  {'k':>4} {'obs':>8} {'rand':>8} {'act':>8} {'delta':>8} "
              f"{'z':>7} {'pos':>5}")
        for k, rows in rows_any:
            o = [x["obs"] for x in rows]
            rd = [x["rand"] for x in rows]
            a = [x["act"] for x in rows]
            d = [x["delta"] for x in rows]
            z = [x["z"] for x in rows]
            print(f"  {k:>4} {np.median(o):>8.3f} {np.median(rd):>8.3f} "
                  f"{np.median(a):>8.3f} {np.median(d):>8.3f} "
                  f"{np.median(z):>7.2f} {sum(x > 0 for x in d):>3}/{len(d)}")

    out = FP_ROOT / args.out
    json.dump(per_net, open(out, "w"), indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
