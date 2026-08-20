"""Does the quotient predict the ring's MOVEMENT as task input is interpolated?

The rotation-readout result is a two-point statement: the memory ring is output-null at the
delay input and output-potent at the response input. This asks the continuous question: as
the input is swept u(beta) = (1-beta) u_delay + beta u_response, can the coarse-grained
quotient predict the whole branch -- where the ring goes, how its tangent plane rotates, how
its eigenvalues move, and when it becomes output-potent?

Two branches, computed by numerical continuation (Newton on h = softplus(W_rec h + W_in u + b);
the leak term cancels at a fixed point):

  full     the true N-unit fixed-point branch: seed from the delay-ring cloud and continue
           every point through the beta grid.
  quotient the k-unit reduced system h_bar = softplus(Q_rec h_bar + Q_in u + b_bar) with
           Q_rec, Q_in, b_bar the group-averages on the connectivity partition; lift by H.

Compared along beta:
  loc       fixed-point centroid in quotient coordinates (cosine, relative error)
  tangent   principal angles between the true ring plane and the predicted plane
  eig       slowest local-Jacobian eigenvalues, true vs reduced
  potency   readout potency of each plane (the rotation curve)
  rotation  principal angle of the ring plane at beta relative to beta=0

  conda run -n myenv python -m flexsym.run.motif_quotient_branch --net ...0 --task delaygo
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_reuse_control import _row_for_k

np.seterr(all="ignore")
ALPHA = 0.2
RING_DIM = 2
Q_SLOW = 1e-6           # slow-point cut for reading the stored delay/go1 clouds
Q_KEEP = 1e-4           # slow cut for our leaky re-finding (settling plateaus ~1e-4,
                        # not machine precision, on a marginal manifold)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid(x):
    return 0.5 * (np.tanh(0.5 * x) + 1.0)


def load_net(net, k):
    hit = (glob.glob(str(paths.ORIGINAL_STORE / net / "model_params.npz"))
           or glob.glob(str(paths.STAGED_STORE / net / "model_params.npz")))
    p = np.load(hit[0])
    w_in = np.asarray(p["w_in"], float)
    n_rec = len(p["b_in"])
    n_in = w_in.shape[0] - n_rec
    W_rec = w_in[n_in:, :].T                      # (N, N) receives orientation
    W_in = w_in[:n_in, :].T                       # (N, n_input)
    b = np.asarray(p["b_in"], float)
    W_out_resp = np.asarray(p["w_out"], float)[:, 1:]   # (N, 2) response channels
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    ind = np.asarray(wl["ind_active"], int)
    labels = np.asarray(wl["labels_struct_by_k"][_row_for_k(wl["labels_struct_by_k"], k)], int)
    return dict(W_rec=W_rec, W_in=W_in, b=b, W_out=W_out_resp, ind=ind, labels=labels)


def newton_fp(h, u, W_rec, W_in, b, iters=80, tol=1e-12):
    I = np.eye(len(h))
    for _ in range(iters):
        x = W_rec @ h + W_in @ u + b
        F = h - softplus(x)
        Jf = I - sigmoid(x)[:, None] * W_rec
        try:
            step = np.linalg.solve(Jf, F)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(Jf, F, rcond=None)[0]
        h = h - step
        if np.linalg.norm(step) < tol:
            break
    return h


def jac(h, u, W_rec, W_in, b):
    x = W_rec @ h + W_in @ u + b
    return (1 - ALPHA) * np.eye(len(h)) + ALPHA * sigmoid(x)[:, None] * W_rec


def plane(cloud, ndim=RING_DIM):
    X = cloud - cloud.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:ndim].T


def principal_angles(U, V):
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def potency(U, R):
    s = np.linalg.svd(U.T @ R, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1) ** 2))


def quotient_matrices(net_mats, ind):
    """Group-averaged reduced operators on active units."""
    W = net_mats["W_rec"][np.ix_(ind, ind)]
    Win = net_mats["W_in"][ind]
    b = net_mats["b"][ind]
    labels = net_mats["labels"]
    cats, inv = np.unique(labels, return_inverse=True)
    H = np.zeros((len(labels), cats.size))
    H[np.arange(len(labels)), inv] = 1.0
    sizes = H.sum(0)
    Q_rec = (H.T @ W @ H) / sizes[:, None]
    Q_in = (H.T @ Win) / sizes[:, None]
    b_bar = (H.T @ b) / sizes
    return H, sizes, Q_rec, Q_in, b_bar


def settle(cloud, u, W, Win, b, n=4000):
    """Relax a cloud onto the slow set with the true leaky dynamics (stable/marginal set)."""
    h = cloud.copy()
    for _ in range(n):
        h = (1 - ALPHA) * h + ALPHA * softplus(h @ W.T + u @ Win.T + b)
    return h


def velocity(cloud, u, W, Win, b):
    """Per-point speed ||-h + softplus(W h + W_in u + b)|| (the FPF velocity)."""
    return np.linalg.norm(-cloud + softplus(cloud @ W.T + u @ Win.T + b), axis=1)


def refind(pool, u, W, Win, b, q_slow):
    """Re-find the slow set at input u: settle the broad pool, keep slow points, dedup.

    Independent of any other beta -- this is fixed-point finding, not continuation, so a
    persisting (rotated) ring is recovered and a genuine collapse shows up as an empty or
    pointlike slow set.
    """
    settled = settle(pool, u, W, Win, b)
    q = velocity(settled, u, W, Win, b)
    slow = settled[q < q_slow]
    if len(slow) < 3:
        return slow
    # dedup by rounding to a coarse grid (keep one representative per cell)
    keys = np.round(slow / 0.05).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return slow[np.sort(idx)]


def eff_dim(cloud):
    """Participation ratio of the cloud covariance: ~2 for a ring, ~0-1 for a point."""
    if len(cloud) < 3:
        return 0.0
    lam = np.linalg.eigvalsh(np.cov(cloud.T))
    lam = lam[lam > 0]
    return float(lam.sum() ** 2 / np.sum(lam ** 2)) if lam.size else 0.0


def top_lambda(cloud, u, W, Win, b):
    """Median top-|lambda| of the local Jacobian over slow points (fixed points)."""
    if len(cloud) == 0:
        return np.nan
    sub = cloud if len(cloud) <= 40 else cloud[np.linspace(0, len(cloud) - 1, 40).astype(int)]
    return float(np.median([np.abs(np.linalg.eigvals(jac(h, u, W, Win, b))).max()
                            for h in sub]))


def build_pool(Xd, qd, Xr, qr, rng, n=600):
    """Broad seed pool spanning both ends: delay ring, response ring, interpolations, noise."""
    rd = Xd[qd < Q_SLOW]
    rr = Xr[qr < Q_SLOW]
    parts = [rd, rr]
    if len(rd) and len(rr):
        a = rd[rng.integers(0, len(rd), n // 2)]
        b_ = rr[rng.integers(0, len(rr), n // 2)]
        t = rng.random((n // 2, 1))
        parts.append(a * (1 - t) + b_ * t)                       # convex interpolations
    pool = np.vstack(parts)
    if len(pool) > n:
        pool = pool[rng.choice(len(pool), n, replace=False)]
    pool = pool + rng.normal(0, 0.05, pool.shape)                # jitter off the manifold
    return pool


def slow_eigplane(h, u, W, Win, b, ndim=RING_DIM):
    """Top-ndim eigenvectors of the local Jacobian at fixed point h (the slow subspace)."""
    lam, V = np.linalg.eig(jac(h, u, W, Win, b))
    order = np.argsort(-np.abs(lam))[:ndim]
    U, _ = np.linalg.qr(np.real(V[:, order]))
    return float(np.abs(lam).max()), U


def refind_metrics(pool, u, W, Win, b, Ra, q_slow, lift=None):
    """Re-find the slow set at input u; report its existence (cloud geometry) plus the
    marginal eigenvalue and readout potency read from the local Jacobian's slow subspace
    at the fixed point (well-defined whether the set is a manifold or a single point).

    lift: for the reduced system, the S = H/sqrt(sizes) map from quotient coords to unit
    space, so readout potency is computed after lifting the slow subspace to the units."""
    slow = refind(pool, u, W, Win, b, q_slow)
    n = len(slow)
    if n == 0:
        return dict(n_slow=0, radius=0.0, eff_dim=0.0, lam=np.nan,
                    cen=None, eplane=None, potency=np.nan)
    cen = slow.mean(0)
    lam, U = slow_eigplane(cen, u, W, Win, b)     # marginal eig + slow subspace at centroid
    U_read = U if lift is None else np.linalg.qr(lift @ U)[0]
    return dict(n_slow=n, eff_dim=eff_dim(slow),
                radius=float(np.median(np.linalg.norm(slow - cen, axis=1))) if n > 2 else 0.0,
                lam=lam, cen=cen, eplane=U, potency=potency(U_read, Ra))


def run(net, task, k, n_beta, seed_n):
    m = load_net(net, k)
    ind = m["ind"]
    W, Win, b = m["W_rec"], m["W_in"], m["b"]
    R_full, _ = np.linalg.qr(m["W_out"])
    Ra = R_full[ind]
    rng = np.random.default_rng(0)

    d = npload(paths.RESULTS_ROOT / "motif_fixed_points" / net / f"{task}__delay1.npz")
    u_d = np.asarray(d["inputs"][0], float)
    rr = npload(paths.RESULTS_ROOT / "motif_fixed_points_response" / net / f"{task}__go1.npz")
    u_r = np.asarray(rr["inputs"][0], float)
    Xd, qd = np.asarray(d["xstar"], float), np.asarray(d["qstar"], float)
    Xr, qr = np.asarray(rr["xstar"], float), np.asarray(rr["qstar"], float)

    # broad pool in FULL space, and its projection into quotient coords
    pool = build_pool(Xd, qd, Xr, qr, rng, n=600)
    H, sizes, Q_rec, Q_in, b_bar = quotient_matrices(m, ind)
    qpool = (pool[:, ind] @ H) / sizes
    S = H / np.sqrt(sizes)[None, :]                # orthonormal-column lift S'S = I

    betas = np.linspace(0, 1, n_beta)
    # reference planes at beta=0 for the rotation measure
    ref_t = refind_metrics(pool[:, ind], u_d, W[np.ix_(ind, ind)], Win[ind], b[ind], Ra, Q_KEEP)
    ref_q = refind_metrics(qpool, u_d, Q_rec, Q_in, b_bar, Ra, Q_KEEP, lift=S)
    Wa, Wina, ba = W[np.ix_(ind, ind)], Win[ind], b[ind]     # active-restricted true system

    rows = []
    for beta in betas:
        u = (1 - beta) * u_d + beta * u_r
        T = refind_metrics(pool[:, ind], u, Wa, Wina, ba, Ra, Q_KEEP)
        Q = refind_metrics(qpool, u, Q_rec, Q_in, b_bar, Ra, Q_KEEP, lift=S)
        row = {"beta": float(beta),
               # (1) existence / disappearance of the slow manifold
               "n_slow_true": T["n_slow"], "n_slow_pred": Q["n_slow"],
               "eff_dim_true": T["eff_dim"], "eff_dim_pred": Q["eff_dim"],
               "radius_true": T["radius"], "radius_pred": Q["radius"],
               # (2) marginal eigenvalue
               "lam_true": T["lam"], "lam_pred": Q["lam"],
               # (3) coarse location (centroid in quotient coords), (4) tangent, (5) potency
               "potency_true": T["potency"], "potency_pred": Q["potency"]}
        if T["cen"] is not None and Q["cen"] is not None:
            cen_q = (H.T @ T["cen"]) / sizes
            row["loc_cos"] = float(cen_q @ Q["cen"] /
                                   (np.linalg.norm(cen_q) * np.linalg.norm(Q["cen"]) + 1e-12))
        else:
            row["loc_cos"] = np.nan
        if T["eplane"] is not None and Q["eplane"] is not None:
            Pq_lift, _ = np.linalg.qr(S @ Q["eplane"])
            row["tangent_true_vs_pred"] = float(principal_angles(T["eplane"], Pq_lift).mean())
            row["rotation_true"] = (float(principal_angles(T["eplane"], ref_t["eplane"]).mean())
                                    if ref_t["eplane"] is not None else np.nan)
            row["rotation_pred"] = (float(principal_angles(Q["eplane"], ref_q["eplane"]).mean())
                                    if ref_q["eplane"] is not None else np.nan)
        else:
            row["tangent_true_vs_pred"] = row["rotation_true"] = row["rotation_pred"] = np.nan
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="LeakyRNN_softplus_diag_256_all_0")
    ap.add_argument("--task", default="delaygo")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n-beta", type=int, default=21)
    ap.add_argument("--seed-n", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = run(args.net, args.task, args.k, args.n_beta, args.seed_n)
    def g(r, key):
        v = r.get(key, np.nan)
        return v if v is not None else np.nan

    print(f"\n=== {args.net}  {args.task}  (k={args.k}) — re-found manifold per beta ===")
    print(f"  {'beta':>4} | {'nT':>3} {'nQ':>3} {'edT':>4} {'edQ':>4} | {'lamT':>6} {'lamQ':>6} | "
          f"{'loc':>6} {'tan°':>5} | {'potT':>7} {'potQ':>7} | {'rotT':>5} {'rotQ':>5}")
    for r in rows:
        print(f"  {r['beta']:>4.2f} | {r['n_slow_true']:>3} {r['n_slow_pred']:>3} "
              f"{g(r,'eff_dim_true'):>4.1f} {g(r,'eff_dim_pred'):>4.1f} | "
              f"{g(r,'lam_true'):>6.3f} {g(r,'lam_pred'):>6.3f} | "
              f"{g(r,'loc_cos'):>6.2f} {g(r,'tangent_true_vs_pred'):>5.0f} | "
              f"{g(r,'potency_true'):>7.4f} {g(r,'potency_pred'):>7.4f} | "
              f"{g(r,'rotation_true'):>5.0f} {g(r,'rotation_pred'):>5.0f}")

    out = paths.RESULTS_ROOT / "motif_fixed_points" / (
        args.out or f"quotient_branch_{args.net[-1]}_{args.task}.json")
    json.dump(rows, open(out, "w"), indent=2, default=float)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
