"""Do the connectivity cells scaffold the NONLINEAR memory motif? (Driscoll bridge)

At a memory-delay fixed point h*, the leaky-RNN state-update Jacobian is
    J = (1-alpha) I + alpha * diag(f'(x*)) W^T ,   f'(x*) = 1 - exp(-h*)  (softplus)
so J's eigenvectors are those of  M = diag(gain) W^T  (the delay-period Jacobian).
We test whether M's eigenvectors are piecewise-constant on the structural cells
(step-R^2) vs random same-size partitions (z), and whether the SLOW mode
(J-eigenvalue near 1 = the line-attractor / memory direction) is cell-organized.
Compares the nonlinear delay Jacobian to the linear recurrent matrix W^T.
"""

from __future__ import annotations

from flexsym import paths

import csv
import sys
from pathlib import Path

import numpy as np


_p = np.load
_o = _p.__closure__[0].cell_contents if getattr(_p, "__closure__", None) else _p


def _dl(*a, **k):
    k["allow_pickle"] = True
    return _o(*a, **k)


np.load = _dl

from flexsym.structural.signed_input_wl_structural_roles import (  # noqa: E402
    apply_matrix_variant, approx_wl_signed, load_w_rec, random_labels_same_sizes,
)
from flexsym.structural.structural_cell_activity_coherence import compute_hidden_by_epoch  # noqa: E402

REPO = paths.REPO_ROOT
ACT = REPO / "results/activity_clusters_paper"
ORIG = paths.ORIGINAL_STORE
STAGED = paths.STAGED_STORE
from flexsym.structural.operating_k import OPK
ALPHA = 0.2


def resolve(m):
    for s in (ORIG, STAGED):
        if (s / m).exists():
            return s / m
    return None


def step_r2(v, labels):
    pred = np.zeros_like(v)
    for c in np.unique(labels):
        pred[labels == c] = v[labels == c].mean()
    return 1 - ((v - pred) ** 2).sum() / ((v - v.mean()) ** 2).sum()


def best_r2(Mat, labels, k, topn=12):
    w, V = np.linalg.eig(Mat)
    order = np.argsort(-np.abs(w))[:topn]
    best = -1
    for i in order:
        v = np.real(V[:, i]); nv = np.linalg.norm(v)
        if nv:
            best = max(best, step_r2(v / nv, labels))
    return best


def main():
    rows = []
    for seed, k in OPK.items():
        model = f"LeakyRNN_softplus_diag_256_all_{seed}"
        md = resolve(model); act = ACT / model / "epoch_variance_unit_clusters_paper_max.npz"
        if md is None or not act.exists():
            continue
        ind = np.asarray(np.load(act, allow_pickle=True)["ind_active"], dtype=np.int64)
        WT = apply_matrix_variant(load_w_rec(md), "transpose")[np.ix_(ind, ind)]
        labels, _ = approx_wl_signed(np.maximum(WT, 0), np.maximum(-WT, 0), k, n_iter=20, seed=0, n_init=20)
        np.load = _dl
        hid = compute_hidden_by_epoch(md, [("delaygo", "delay1")], batch_size=512)
        np.load = _dl
        h = hid[("delaygo", "delay1")][:, :, ind]
        hstar = h[-5:].mean(0).mean(0)
        g = 1 - np.exp(-np.maximum(hstar, 1e-6))
        M = g[:, None] * WT

        r2_lin = best_r2(WT, labels, k)
        r2_jac = best_r2(M, labels, k)
        null = np.array([best_r2(M, random_labels_same_sizes(labels, seed=b), k) for b in range(60)])
        z = (r2_jac - null.mean()) / null.std()
        r2_rand = null.mean()
        # slow (memory) mode
        w, V = np.linalg.eig(M); Jw = (1 - ALPHA) + ALPHA * w
        islow = np.argmin(np.abs(Jw - 1)); vs = np.real(V[:, islow]); vs /= np.linalg.norm(vs)
        rows.append({
            "seed": seed, "k": k,
            "gain_within_cell_cv": float(np.mean([g[labels == c].std() / max(g[labels == c].mean(), 1e-9)
                                                  for c in np.unique(labels) if (labels == c).sum() > 1])),
            "r2_linear_WT": float(r2_lin), "r2_delay_jacobian": float(r2_jac),
            "r2_random_partition": float(r2_rand), "jacobian_z": float(z),
            "slow_J_eig_abs": float(np.abs(Jw[islow])), "slow_mode_step_r2": float(step_r2(vs, labels)),
        })
        print(f"seed {seed}: lin R²={r2_lin:.2f}  jac R²={r2_jac:.2f}  rand={r2_rand:.2f}  z={z:+.1f}  "
              f"slowJeig={np.abs(Jw[islow]):.3f} slowR²={step_r2(vs, labels):.2f}", flush=True)

    out = REPO / "results/signed_input_wl_paper/motif_jacobian_lumpability.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    import numpy as _np
    def col(c): return _np.array([r[c] for r in rows])
    print("\n=== across 10 seeds (256) ===")
    print(f"  linear W^T   step-R²: {col('r2_linear_WT').mean():.3f}")
    print(f"  delay Jacobian step-R²: {col('r2_delay_jacobian').mean():.3f}  (random {col('r2_random_partition').mean():.3f})")
    print(f"  Jacobian z vs random: {col('jacobian_z').mean():+.1f}  ({int((col('jacobian_z')>2).sum())}/{len(rows)} > 2)")
    print(f"  slow-mode J-eig: {col('slow_J_eig_abs').mean():.3f}  slow-mode step-R²: {col('slow_mode_step_r2').mean():.3f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
