"""Does the QUOTIENT predict the motif direction?

Containment (run_motif_subspace_rva.py) shows the motif direction lies in the
group-constant subspace col(H). That is a statement about the subspace only. The
stronger, falsifiable claim is that the coarse-grained dynamics -- the quotient
matrix on the role partition -- predicts WHICH direction inside that subspace the
motif occupies, and with which eigenvalue.

Setup. The trained cell is h+ = (1-a) h + a * f(W h + W_in x + b), so the
Jacobian at a fixed point is J = (1-a) I + a diag(g) W with g = f'(u). Both a and
the per-unit gains g are recovered exactly from a stored J and W (the row ratio
J[i,j]/W[i,j] is constant along row i), so the coarse-grained predictor below
uses no fitted quantities.

Three quotients, in increasing order of task information:
  Q_struct  = (H'H)^-1 H' W H                 connectivity only, task-free
  Q_pred    = (1-a) I + a diag(g_bar) Q_struct  + GROUP-MEAN gains (the honest
              coarse-grained prediction: partition + connectivity + group gains)
  Q_task    = (H'H)^-1 H' Jbar H              quotient of the condition's mean
              Jacobian (upper bound; exact if the partition were equitable for J)

Test. Take the top-m eigenvectors of Q by |lambda|, lift them to N units
(piecewise constant on groups), orthonormalise -> B. Score the motif direction v:

  A_raw  = ||B' v|| / ||v||           how much of v the quotient predicts
  A_cond = ||B' v|| / ||P_H v||       CONDITIONAL on v already being in col(H)

A_cond is the statistic that matters: it factors out containment, so it asks only
"which direction within the subspace", not "is it in the subspace". Its chance
level is ~sqrt(m/k) (m arbitrary directions out of a k-dim subspace).

Nulls: size-matched random partitions, with the quotient RECOMPUTED on each
random partition (so the null has its own coarse-grained dynamics), plus the
activity partition at matched k.

Also reported: eigenvalue agreement |lambda_Q - lambda_J| for the best-aligned
quotient eigenvector, where lambda_J is the Jacobian eigenvalue of the motif.

  conda run -n myenv python stepnet/run_motif_quotient_prediction.py --seeds 0-2
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
from flexsym.run.motif_reuse_control import MOTIF_TYPES, _row_for_k  # noqa: E402
from flexsym.run.motif_reuse_diagnostics import canonical_sign  # noqa: E402

FP_ROOT = paths.RESULTS_ROOT / "motif_fixed_points"
K_GRID = [8, 12, 16, 20, 25, 30, 40, 50]
M_GRID = [1, 3, 5]
N_PERM = 200
MAX_FP = 150            # fixed points averaged into Jbar per condition
W_THRESH = 1e-3         # |W| above which the row-ratio gain recovery is used


# --------------------------------------------------------------------------- #
# per-condition mean Jacobian + exact (alpha, gains) recovery
# --------------------------------------------------------------------------- #
def condition_jacobians(net, ind_active, W, cache_name="jbar_cache.npz"):
    """Mean Jacobian per condition (active units), cached to disk.

    J_xstar is (n_fp, 256, 256) float32 -- ~250 MB per condition -- so the mean
    is taken once and cached.
    """
    net_dir = FP_ROOT / net
    cache = net_dir / cache_name
    if cache.exists():
        d = npload(cache)
        if np.array_equal(np.asarray(d["ind_active"]), ind_active):
            return {str(k): np.asarray(v, dtype=float)
                    for k, v in zip(d["keys"], d["jbars"])}
    keys, jbars = [], []
    for f in sorted(net_dir.glob("*.npz")):
        if "cache" in f.name:
            continue
        try:
            d = npload(f)
            J = np.asarray(d["J_xstar"], dtype=np.float32)
        except Exception:
            continue
        if J.ndim != 3 or not len(J):
            continue
        if len(J) > MAX_FP:
            J = J[np.linspace(0, len(J) - 1, MAX_FP).astype(int)]
        Jb = J.mean(0)[np.ix_(ind_active, ind_active)].astype(float)
        keys.append(f.stem)
        jbars.append(Jb)
        del J
    np.savez_compressed(cache, keys=np.array(keys), jbars=np.array(jbars),
                        ind_active=ind_active)
    return dict(zip(keys, jbars))


def recover_alpha_gain(Jb, W):
    """Exact (alpha, g) from J = (1-alpha) I + alpha diag(g) W.

    r_i = alpha * g_i is the constant row ratio J[i,j]/W[i,j] over off-diagonal
    entries with |W| above threshold; alpha then follows from the diagonal.
    """
    n = W.shape[0]
    off = ~np.eye(n, dtype=bool)
    use = off & (np.abs(W) > W_THRESH)
    with np.errstate(all="ignore"):
        num = np.where(use, Jb * W, 0.0).sum(1)
        den = np.where(use, W * W, 0.0).sum(1)
    r = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)     # least squares
    one_minus_a = np.median(np.diag(Jb) - r * np.diag(W))
    alpha = float(np.clip(1.0 - one_minus_a, 1e-6, 1.0))
    return alpha, r / alpha


# --------------------------------------------------------------------------- #
# quotients
# --------------------------------------------------------------------------- #
def indicator(labels):
    cats, inv = np.unique(labels, return_inverse=True)
    H = np.zeros((len(labels), cats.size))
    H[np.arange(len(labels)), inv] = 1.0
    return H, H.sum(0)


def quotient(M, H, sizes):
    """(H'H)^-1 H' M H : row-averaged block sums.

    This is the quotient in the RAW role coordinates in which W H = H Q holds
    exactly for an equitable partition, so an eigenvector z of Q lifts to units
    as H z (NOT as H D^-1/2 z -- see quotient_ortho).
    """
    return (H.T @ M @ H) / sizes[:, None]


def quotient_ortho(M, H, sizes):
    """D^-1/2 H' M H D^-1/2 : the same quotient in ORTHONORMAL role coordinates.

    With S = H D^-1/2 (S'S = I), an equitable partition gives M S = S Qtilde,
    where Qtilde = D^1/2 Q D^-1/2 is similar to Q and so has the same spectrum.
    Use this operator whenever the lift to unit space is performed through S;
    lifting eigenvectors of the raw Q through S mixes the two coordinate systems.
    """
    r = np.sqrt(sizes)
    return (H.T @ M @ H) / r[:, None] / r[None, :]


def lifted_basis(Q, H, sizes, m):
    """Top-m eigenvectors of Q by |lambda|, lifted to units and orthonormalised.

    Returns (B, lambdas) with B (N, r), r <= m.
    """
    lam, V = np.linalg.eig(Q)
    order = np.argsort(-np.abs(lam))[:m]
    lam = lam[order]
    U = H @ np.real(V[:, order])           # lift: piecewise constant on groups
    # complex pairs contribute their real and imaginary parts
    if np.any(np.abs(np.imag(V[:, order])) > 1e-12):
        U = np.hstack([U, H @ np.imag(V[:, order])])
    Bq, R = np.linalg.qr(U)
    keep = np.abs(np.diag(R)) > 1e-10
    return Bq[:, keep], lam


def score(v, B, PHv_norm):
    """A_raw and A_cond for a motif direction v against basis B."""
    vn = np.linalg.norm(v)
    if vn == 0 or B.shape[1] == 0:
        return np.nan, np.nan
    proj = np.linalg.norm(B.T @ v)
    return float(proj / vn), (float(proj / PHv_norm) if PHv_norm > 0 else np.nan)


def containment_norm(v, H, sizes):
    """||P_H v||."""
    gs = H.T @ v
    return float(np.sqrt(np.sum(gs ** 2 / sizes)))


# --------------------------------------------------------------------------- #
# per-partition evaluation
# --------------------------------------------------------------------------- #
def eval_labels(labels, W, conds, m_grid):
    """A_cond / A_raw per quotient variant and m, averaged over conditions."""
    H, sizes = indicator(labels)
    Qs = quotient(W, H, sizes)
    out = {f"{q}_m{m}": [] for q in ("struct", "pred", "task") for m in m_grid}
    lam_err = []
    for c in conds:
        v = c["v"]
        PHv = containment_norm(v, H, sizes)
        gbar = (H.T @ c["g"]) / sizes
        Qp = (1 - c["alpha"]) * np.eye(len(sizes)) + c["alpha"] * (gbar[:, None] * Qs)
        Qt = quotient(c["Jb"], H, sizes)
        for tag, Q in (("struct", Qs), ("pred", Qp), ("task", Qt)):
            for m in m_grid:
                B, lam = lifted_basis(Q, H, sizes, m)
                a_raw, a_cond = score(v, B, PHv)
                out[f"{tag}_m{m}"].append(a_cond)
                if tag == "pred" and m == m_grid[-1] and np.isfinite(c["lam_J"]):
                    lam_err.append(float(np.min(np.abs(lam - c["lam_J"]))))
    res = {k: float(np.nanmean(v)) for k, v in out.items()}
    res["lam_err"] = float(np.nanmean(lam_err)) if lam_err else np.nan
    return res


def load_conditions(net, ind_active, W, jbars):
    """Motif directions + the Jacobian machinery, per condition."""
    d = npload(FP_ROOT / net / "participation_cache_v2.npz")
    conds = []
    for m in list(d["records"]):
        if m["motif"] not in MOTIF_TYPES:
            continue
        key = m.get("key") or f"{m['rule']}__{m['epoch']}"
        Jb = jbars.get(key)
        V = m.get("vecs_act")
        if Jb is None or V is None or not len(V):
            continue
        V = np.asarray(V, dtype=float)
        V = V[np.all(np.isfinite(V), axis=1)]
        if not len(V):
            continue
        _, _, Vt = np.linalg.svd(V, full_matrices=False)
        v = canonical_sign(Vt[0])
        alpha, g = recover_alpha_gain(Jb, W)
        # eigenvalue of Jbar whose eigenvector best matches the motif direction
        lam, EV = np.linalg.eig(Jb)
        align = np.abs(np.real(EV).T @ v) / np.maximum(
            np.linalg.norm(np.real(EV), axis=0) * np.linalg.norm(v), 1e-12)
        lam_J = float(np.real(lam[int(np.argmax(align))]))
        conds.append({"key": key, "motif": m["motif"], "rule": m["rule"],
                      "v": v, "Jb": Jb, "g": g, "alpha": alpha, "lam_J": lam_J})
    return conds


def run_net(net, k_grid, m_grid, n_perm, rng):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], dtype=float)
    ind_active = np.asarray(wl["ind_active"], dtype=int)
    jbars = condition_jacobians(net, ind_active, W)
    conds = load_conditions(net, ind_active, W, jbars)
    if len(conds) < 4:
        return None
    ls, la = wl["labels_struct_by_k"], wl["labels_act_by_k"]
    per_k = {}
    for k in k_grid:
        r = _row_for_k(ls, k)
        role = np.asarray(ls[r], dtype=int)
        actl = np.asarray(la[r], dtype=int)
        if np.unique(role).size < 2:
            continue
        obs = eval_labels(role, W, conds, m_grid)
        act = eval_labels(actl, W, conds, m_grid)
        lab = role.copy()
        null = {kk: [] for kk in obs}
        for _ in range(n_perm):
            rng.shuffle(lab)
            nv = eval_labels(lab, W, conds, m_grid)
            for kk in null:
                null[kk].append(nv[kk])
        rec = {"k": int(np.unique(role).size), "n_cond": len(conds)}
        for kk in obs:
            arr = np.array(null[kk], dtype=float)
            mu, sd = np.nanmean(arr), np.nanstd(arr)
            rec[kk] = {"obs": obs[kk], "rand": float(mu), "act": act[kk],
                       "delta": obs[kk] - float(mu),
                       "z": float((obs[kk] - mu) / sd) if sd > 0 else 0.0}
        per_k[k] = rec
    by_motif = {}
    for mt in MOTIF_TYPES:
        n = sum(c["motif"] == mt for c in conds)
        if n:
            by_motif[mt] = n
    return {"per_k": per_k, "n_cond": len(conds), "by_motif": by_motif,
            "alpha": float(np.median([c["alpha"] for c in conds])),
            "lam_J_by_motif": {mt: float(np.median([c["lam_J"] for c in conds
                                                    if c["motif"] == mt]))
                               for mt in by_motif}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--k-grid", default=None)
    ap.add_argument("--out", default="quotient_prediction.json")
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
        r = run_net(net, k_grid, M_GRID, args.n_perm, rng)
        if r:
            per_net[net] = r
            print(f"[done] {net}: {r['n_cond']} conditions, alpha={r['alpha']:.3f}, "
                  f"{r['by_motif']}")

    print("\nmotif Jacobian eigenvalue (median) by type:")
    for net, v in per_net.items():
        print(f"  seed {net[-1]}: " + "  ".join(
            f"{mt}={lv:.3f}" for mt, lv in v["lam_J_by_motif"].items()))

    for tag, label in (("struct", "Q_struct  (connectivity only)"),
                       ("pred", "Q_pred    (connectivity + group-mean gains)"),
                       ("task", "Q_task    (quotient of the mean Jacobian)")):
        print(f"\n=== {label} :  A_cond = ||B'v|| / ||P_H v|| ===")
        for m in M_GRID:
            key = f"{tag}_m{m}"
            print(f"\n  m={m}")
            print(f"  {'k':>4} {'chance':>7} {'conn':>7} {'rand':>7} {'act':>7} "
                  f"{'delta':>7} {'z':>7} {'pos':>5}")
            for k in k_grid:
                rows = [v["per_k"][k] for v in per_net.values() if k in v["per_k"]]
                if not rows:
                    continue
                o = [x[key]["obs"] for x in rows]
                rd = [x[key]["rand"] for x in rows]
                a = [x[key]["act"] for x in rows]
                dl = [x[key]["delta"] for x in rows]
                z = [x[key]["z"] for x in rows]
                print(f"  {k:>4} {np.sqrt(m / k):>7.3f} {np.median(o):>7.3f} "
                      f"{np.median(rd):>7.3f} {np.median(a):>7.3f} "
                      f"{np.median(dl):>7.3f} {np.median(z):>7.2f} "
                      f"{sum(x > 0 for x in dl):>3}/{len(dl)}")

    print("\n=== eigenvalue agreement |lambda_Qpred - lambda_J| (mean over conditions) ===")
    print(f"  {'k':>4} {'conn':>8} {'rand':>8} {'act':>8} {'z':>7}")
    for k in k_grid:
        rows = [v["per_k"][k] for v in per_net.values() if k in v["per_k"]]
        if not rows:
            continue
        print(f"  {k:>4} {np.median([x['lam_err']['obs'] for x in rows]):>8.4f} "
              f"{np.median([x['lam_err']['rand'] for x in rows]):>8.4f} "
              f"{np.median([x['lam_err']['act'] for x in rows]):>8.4f} "
              f"{np.median([x['lam_err']['z'] for x in rows]):>7.2f}")

    out = FP_ROOT / args.out
    json.dump(per_net, open(out, "w"), indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
