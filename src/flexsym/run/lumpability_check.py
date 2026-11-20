"""Lumpability / invariant-subspace check for structural partitions.

For a partition with indicator H (n x k), P = H (H^T H)^{-1} H^T projects onto
cell-constant vectors. For a matrix M:

  1. Invariant subspace:   ||(I-P) M H|| / ||M H||   (normalised gamma defect).
  2. Eigenvector leakage:  leakage(v) = ||v - P v|| / ||v|| for each leading eigenvector v.

Both are computed for the structural partition and for random same-size partitions
(z-score; negative z = less leakage than chance), on the optimised matrix variant
(default transpose) and on raw W. No TF model load.

Networks are those with a cached transpose WL sweep; k is each network's operating
k (`flexsym.structural.operating_k`), and the labels are that row of the sweep.
"""

from __future__ import annotations

from flexsym import paths

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np


from flexsym.structural.signed_input_wl_structural_roles import (  # noqa: E402
    apply_matrix_variant,
    load_w_rec,
)
from flexsym.motifs.motif_classify import npload  # noqa: E402
from flexsym.run.motif_reuse_control import _row_for_k  # noqa: E402
from flexsym.structural.operating_k import operating_k  # noqa: E402
from flexsym.structural.partition import pw_constant_projector, random_labels_same_sizes  # noqa: E402

WL_ROOT = paths.RESULTS_ROOT / "signed_input_wl_paper"
ACT_ROOT = paths.RESULTS_ROOT / "activity_clusters_paper"
ACT_NAME = "epoch_variance_unit_clusters_paper_max.npz"
ORIGINAL = paths.ORIGINAL_STORE
STAGED = paths.STAGED_STORE


def resolve(model: str) -> Path | None:
    for store in (ORIGINAL, STAGED):
        if (store / model).exists():
            return store / model
    return None


def eig_leakage(M: np.ndarray, P: np.ndarray, m: int) -> float:
    """Mean leakage ||v - P v|| / ||v|| over the m eigenvectors of largest |lambda|."""
    w, V = np.linalg.eig(M)
    order = np.argsort(-np.abs(w))[:m]
    leaks = []
    for i in order:
        v = V[:, i]
        nv = np.linalg.norm(v)
        if nv == 0:
            continue
        leaks.append(np.linalg.norm(v - P @ v) / nv)
    return float(np.mean(leaks)) if leaks else float("nan")


def subspace_invariance(M: np.ndarray, P: np.ndarray) -> float:
    """||(I - P) M P|| / ||M P||  (0 = col(P) exactly M-invariant)."""
    MP = M @ P
    denom = np.linalg.norm(MP)
    if denom == 0:
        return float("nan")
    return float(np.linalg.norm(MP - P @ MP) / denom)


def analyse(model: str, k: int, *, variant: str, n_null: int, m_eig: int) -> dict[str, Any] | None:
    md = resolve(model)
    act_path = ACT_ROOT / model / ACT_NAME
    if md is None or not act_path.exists():
        return None
    ind = np.asarray(np.load(act_path, allow_pickle=True)["ind_active"], dtype=np.int64)
    W_raw = load_w_rec(md)[np.ix_(ind, ind)]
    matrices = {"transpose": W_raw.T, "W": W_raw}

    # structural labels at operating k, from the cached transpose WL sweep
    sweep = WL_ROOT / model / "transpose" / "signed_input_wl_sweep.npz"
    if not sweep.exists():
        return None
    ls = npload(sweep)["labels_struct_by_k"]
    labels = np.asarray(ls[_row_for_k(ls, k)], dtype=np.int64)
    actual_k = int(np.unique(labels).size)
    kk = int(labels.max() + 1)
    P = pw_constant_projector(labels, kk)

    m = min(m_eig, kk) if m_eig > 0 else kk
    out: dict[str, Any] = {"model": model, "size": int(model.split("_")[3]), "seed": int(model.split("_")[-1]), "k": actual_k, "m_eig": m}
    for name, M in matrices.items():
        leak = eig_leakage(M, P, m)
        inv = subspace_invariance(M, P)
        null_leak = np.empty(n_null)
        null_inv = np.empty(n_null)
        for b in range(n_null):
            nl = random_labels_same_sizes(labels, seed=b)
            Pn = pw_constant_projector(nl, kk)
            null_leak[b] = eig_leakage(M, Pn, m)
            null_inv[b] = subspace_invariance(M, Pn)
        lz = (leak - null_leak.mean()) / null_leak.std(ddof=1) if null_leak.std(ddof=1) > 0 else float("nan")
        iz = (inv - null_inv.mean()) / null_inv.std(ddof=1) if null_inv.std(ddof=1) > 0 else float("nan")
        out[f"{name}_eig_leakage"] = leak
        out[f"{name}_eig_leakage_null"] = float(null_leak.mean())
        out[f"{name}_eig_leakage_z"] = lz
        out[f"{name}_invariance"] = inv
        out[f"{name}_invariance_null"] = float(null_inv.mean())
        out[f"{name}_invariance_z"] = iz
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-null", type=int, default=200)
    p.add_argument("--m-eig", type=int, default=0, help="Top-m eigenvectors by |lambda| (0 = k).")
    p.add_argument("--variant", default="transpose")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--output-dir", default=str(paths.RESULTS_ROOT / "signed_input_wl_paper/lumpability"))
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opk = {d.name: operating_k(d.name) for d in WL_ROOT.glob("LeakyRNN_*")}
    opk = {m: k for m, k in opk.items() if k is not None}
    models = sorted(opk, key=lambda m: (int(m.split("_")[3]), int(m.split("_")[-1])))
    if args.models:
        models = [m for m in models if m in set(args.models)]

    rows = []
    for m in models:
        r = analyse(m, opk[m], variant=args.variant, n_null=args.n_null, m_eig=args.m_eig)
        if r is None:
            print(f"SKIP {m}")
            continue
        rows.append(r)
        print(f"{m:36} k={r['k']:>3}  T:leak_z={r['transpose_eig_leakage_z']:+.2f} inv_z={r['transpose_invariance_z']:+.2f} | W:leak_z={r['W_eig_leakage_z']:+.2f} inv_z={r['W_invariance_z']:+.2f}", flush=True)

    if not rows:
        raise SystemExit("no models")
    csv_path = out_dir / "lumpability.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {csv_path}")

    import numpy as _np
    by = {}
    for r in rows:
        by.setdefault(r["size"], []).append(r)
    print("\n=== per-size eigenvector-leakage z (negative = more piecewise-constant than chance) ===")
    for size in sorted(by):
        rs = by[size]
        tz = _np.array([r["transpose_eig_leakage_z"] for r in rs])
        wz = _np.array([r["W_eig_leakage_z"] for r in rs])
        print(f"  size={size:>4} n={len(rs)}  transpose leak_z={_np.nanmean(tz):+.2f}  W leak_z={_np.nanmean(wz):+.2f}")


if __name__ == "__main__":
    main()
