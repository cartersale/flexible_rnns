"""Per-condition (lambda_Q, lambda_J) pairs, for the eigenvalue-prediction panel.

`motif_quotient_prediction` reports only the mean |lambda_Q - lambda_J| per network,
which is enough for the text but not for a scatter. This dumps the pairs themselves.

For each condition, lambda_J is the eigenvalue of the condition's mean Jacobian whose
eigenvector best matches the motif direction (as `load_conditions` computes it), and
lambda_Q is the eigenvalue of the quotient mode best ALIGNED with that direction --
the manuscript's wording -- among the top-M modes by |lambda|. The nearest-eigenvalue
variant is recorded too, since the two can differ.

The same is done for the activity partition at matched k and for size-matched random
partitions (median over draws, with the quotient recomputed on each), giving the two
comparison distributions.

    conda run -n myenv python -m flexsym.run.quotient_lambda_pairs --seeds 3-12
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_quotient_prediction import (
    condition_jacobians,
    indicator,
    load_conditions,
    quotient,
)
from flexsym.run.motif_reuse_control import _row_for_k

FP_ROOT = paths.RESULTS_ROOT / "motif_fixed_points"
K = 20
M = 5                   # modes considered, matching the reported analysis
N_PERM = 20


def best_aligned(labels, W, c, v, m=M):
    """(lambda of the best-aligned quotient mode, lambda of the nearest one)."""
    H, sizes = indicator(labels)
    Qs = quotient(W, H, sizes)
    gbar = (H.T @ c["g"]) / sizes
    Qp = (1 - c["alpha"]) * np.eye(len(sizes)) + c["alpha"] * (gbar[:, None] * Qs)
    lam, V = np.linalg.eig(Qp)
    order = np.argsort(-np.abs(lam))[:m]
    lam, V = lam[order], V[:, order]
    U = H @ np.real(V)                                   # lift to units
    nrm = np.linalg.norm(U, axis=0)
    ok = nrm > 1e-12
    if not ok.any():
        return np.nan, np.nan
    cos = np.abs(U[:, ok].T @ v) / (nrm[ok] * np.linalg.norm(v))
    lam_ok = lam[ok]
    return (float(np.real(lam_ok[int(np.argmax(cos))])),
            float(np.real(lam_ok[int(np.argmin(np.abs(np.real(lam_ok) - c["lam_J"])))])))


def run_net(net, rng):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], dtype=float)
    ind_active = np.asarray(wl["ind_active"], dtype=int)
    jbars = condition_jacobians(net, ind_active, W)
    conds = load_conditions(net, ind_active, W, jbars)
    r = _row_for_k(wl["labels_struct_by_k"], K)
    role = np.asarray(wl["labels_struct_by_k"][r], dtype=int)
    act = np.asarray(wl["labels_act_by_k"][r], dtype=int)

    rows = []
    for c in conds:
        v = c["v"]
        lam_q, lam_near = best_aligned(role, W, c, v)
        lam_a, _ = best_aligned(act, W, c, v)
        perm = role.copy()
        drand = []
        for _ in range(N_PERM):
            rng.shuffle(perm)
            lr, _ = best_aligned(perm, W, c, v)
            if np.isfinite(lr):
                drand.append(abs(lr - c["lam_J"]))
        rows.append(dict(
            net=net, seed=int(net.rsplit("_", 1)[1]), k=int(np.unique(role).size),
            cond=c["key"], rule=c["rule"], motif=c["motif"],
            lam_J=c["lam_J"], lam_Q=lam_q, lam_Q_nearest=lam_near, lam_Q_act=lam_a,
            dlam=abs(lam_q - c["lam_J"]), dlam_act=abs(lam_a - c["lam_J"]),
            dlam_rand=float(np.median(drand)) if drand else np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="3-12")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="quotient_lambda_pairs.csv")
    args = ap.parse_args()
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))

    rng = np.random.default_rng(0)
    rows = []
    for s in seeds:
        net = f"LeakyRNN_softplus_diag_{args.size}_all_{s}"
        if not (FP_ROOT / net / "participation_cache_v2.npz").exists():
            print(f"[skip] {net}")
            continue
        r = run_net(net, rng)
        rows += r
        print(f"[done] {net}: {len(r)} conditions")

    out = FP_ROOT / args.out
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    d = np.array([r["dlam"] for r in rows]); a = np.array([r["dlam_act"] for r in rows])
    q = np.array([r["dlam_rand"] for r in rows])
    print(f"\n|dlam| median: role {np.nanmedian(d):.3f}  "
          f"random {np.nanmedian(q):.3f}  activity {np.nanmedian(a):.3f}")
    print(f"wrote {out}  ({len(rows)} conditions)")


if __name__ == "__main__":
    main()
