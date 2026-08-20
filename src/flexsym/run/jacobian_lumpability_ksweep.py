"""Do the connectivity cells also lump the NONLINEAR memory-delay operator?

Everything in Act 1 so far is a statement about the linear recurrent matrix. The claim the
paper actually needs is about the dynamics the network runs. During a memory delay the
leaky-RNN state update linearises around the sustained state h* as

    J = (1 - alpha) I + alpha * diag(f'(x*)) W^T ,     f'(x*) = 1 - exp(-h*)  for softplus

so J and M = diag(gain) W^T share eigenvectors, and J's eigenvalues are an affine map of
M's. If the structure found in W^T were a property of the weights that the nonlinearity
destroys, M's eigenvectors would not be piecewise-constant on the same cells.

The test is deliberately a *transfer* test: the partition is optimised on W^T only, then
applied unchanged to M. Nothing is refitted to the Jacobian, so a high score cannot come
from the partition adapting to it.

Reported per (network, k):
  r2_topk / r2_best on M, with a size-matched random-partition null and z  (the test)
  the same on W^T                                                          (reference)
  gain_cv     within-cell coefficient of variation of the gains -- if gains were constant
              within cells the transfer would be trivial, so a LARGE cv makes the result
              stronger, not weaker
  slow_*      the mode whose J-eigenvalue is closest to 1 (the line-attractor / memory
              direction): its |eigenvalue| and its step-R^2 on the cells

Requires a TF forward pass to get h*, so it covers the networks whose checkpoints are on
the data volume. Writes
results/signed_input_wl_paper/lumpability/jacobian_lumpability_ksweep.csv.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

# flexsym.core.task monkeypatches np.load; take a handle to the original first so the
# reload below is idempotent and passing allow_pickle stays legal here.
_orig_load = np.load
_inner = (_orig_load.__closure__[0].cell_contents
          if getattr(_orig_load, "__closure__", None) else _orig_load)


def _load_pickled(*a, **k):
    k["allow_pickle"] = True
    return _inner(*a, **k)


np.load = _load_pickled

from flexsym import paths  # noqa: E402
from flexsym.run.lumpability_r2_ksweep import GRID, leading_eigvecs, scores  # noqa: E402
from flexsym.structural.signed_input_wl_structural_roles import (  # noqa: E402
    apply_matrix_variant, approx_wl_signed, load_w_rec, random_labels_same_sizes,
)
from flexsym.structural.structural_cell_activity_coherence import compute_hidden_by_epoch  # noqa: E402

ACT_ROOT = paths.RESULTS_ROOT / "activity_clusters_paper"
ACT_NAME = "epoch_variance_unit_clusters_paper_max.npz"
OUT = paths.RESULTS_ROOT / "signed_input_wl_paper/lumpability/jacobian_lumpability_ksweep.csv"

K_GRID = [10, 20, 30, 40]
ALPHA = 0.2
N_NULL = 60
TOP_N = 12
EPOCH = ("delaygo", "delay1")


def resolve(model: str) -> Path | None:
    for store in (paths.ORIGINAL_STORE, paths.STAGED_STORE):
        if (store / model).exists():
            return store / model
    return None


def score(M: np.ndarray, labels: np.ndarray, n_null: int, top_n: int) -> dict:
    kk = int(np.unique(labels).size)
    V = leading_eigvecs(M, M.shape[0])
    Vb, Vk = V[:top_n], V[:kk]
    best, topk = scores(Vb, Vk, labels)
    nb = np.empty(n_null)
    nk = np.empty(n_null)
    for b in range(n_null):
        rl = random_labels_same_sizes(labels, seed=b)
        nb[b], nk[b] = scores(Vb, Vk, rl)
    sb, sk = nb.std(ddof=1), nk.std(ddof=1)
    return {"r2_best": best, "r2_best_null_mean": float(nb.mean()),
            "z_best": float((best - nb.mean()) / sb) if sb > 0 else np.nan,
            "r2_topk": topk, "r2_topk_null_mean": float(nk.mean()),
            "z_topk": float((topk - nk.mean()) / sk) if sk > 0 else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-grid", default=",".join(str(k) for k in K_GRID))
    ap.add_argument("--n-null", type=int, default=N_NULL)
    ap.add_argument("--rule", default=EPOCH[0])
    ap.add_argument("--epoch", default=EPOCH[1])
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()
    kg = [int(x) for x in args.k_grid.split(",")]
    key = (args.rule, args.epoch)

    rows, skipped = [], []
    t0 = time.time()
    for size, seed in GRID:
        model = f"LeakyRNN_softplus_diag_{size}_all_{seed}"
        md = resolve(model)
        act = ACT_ROOT / model / ACT_NAME
        if md is None or not act.exists():
            skipped.append(f"{model}({'no weights' if md is None else 'no activity'})")
            continue
        ind = np.asarray(np.load(act)["ind_active"], dtype=np.int64)
        WT = apply_matrix_variant(load_w_rec(md), "transpose")[np.ix_(ind, ind)]

        np.load = _load_pickled
        hid = compute_hidden_by_epoch(md, [key], batch_size=args.batch_size)
        np.load = _load_pickled
        h = hid[key][:, :, ind]
        hstar = h[-5:].mean(0).mean(0)             # sustained delay state, averaged
        gain = 1.0 - np.exp(-np.maximum(hstar, 1e-6))   # softplus f'
        M = gain[:, None] * WT

        for k in kg:
            if k >= WT.shape[0]:
                continue
            # partition optimised on the LINEAR matrix only, then transferred to M
            labels, _ = approx_wl_signed(np.maximum(WT, 0.0), np.maximum(-WT, 0.0),
                                         k, n_iter=20, seed=0, n_init=20)
            if np.unique(labels).size < 2:
                continue
            jac = score(M, labels, args.n_null, TOP_N)
            lin = score(WT, labels, args.n_null, TOP_N)

            w, V = np.linalg.eig(M)
            Jw = (1 - ALPHA) + ALPHA * w
            islow = int(np.argmin(np.abs(Jw - 1.0)))
            vs = np.real(V[:, islow])
            vs = vs / max(np.linalg.norm(vs), 1e-12)
            slow_r2 = scores(vs[None, :], vs[None, :], labels)[0]

            cvs = [gain[labels == c].std() / max(gain[labels == c].mean(), 1e-12)
                   for c in np.unique(labels) if (labels == c).sum() > 1]
            rows.append({
                "model": model, "size": size, "seed": seed, "k": int(np.unique(labels).size),
                "k_req": k, "n_active": int(ind.size), "rule": args.rule, "epoch": args.epoch,
                "gain_cv_within_cell": float(np.mean(cvs)) if cvs else np.nan,
                **{f"jac_{a}": b for a, b in jac.items()},
                **{f"lin_{a}": b for a, b in lin.items()},
                "slow_J_eig_abs": float(np.abs(Jw[islow])),
                "slow_mode_step_r2": float(slow_r2),
            })
            r = rows[-1]
            print(f"[{time.time()-t0:5.0f}s] {model} k={r['k']}: "
                  f"jac R2_topk {r['jac_r2_topk']:.3f} (null {r['jac_r2_topk_null_mean']:.3f}, "
                  f"z {r['jac_z_topk']:+.1f})  lin {r['lin_r2_topk']:.3f}  "
                  f"gainCV {r['gain_cv_within_cell']:.2f}  slowR2 {r['slow_mode_step_r2']:.2f}",
                  flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows)} rows)")
    if skipped:
        print("skipped: " + ", ".join(skipped))


if __name__ == "__main__":
    main()
