"""Lumpability / invariant-subspace check for structural partitions (P1.4).

Reviewer #6 (highest-value theory): equitable partitions are not just "role
equivalence" -- they are lumpable, invariant subspaces of the linear dynamics,
and the eigenvectors are piecewise-constant on cells. That is the theorem-backed
bridge from connectivity to Driscoll's dynamical motifs. Here we test the
empirical prediction directly.

Given a partition with indicator H (n x k), let P = H (H^T H)^{-1} H^T be the
orthogonal projector onto functions that are constant within each cell. Two
consequences of (approximate) equitability of a matrix M:

  1. Invariant subspace:   col(H) is (nearly) M-invariant  <=>
     ||(I-P) M H|| / ||M H||  is small   (this is the normalized gamma defect).
  2. Piecewise-constant eigenvectors: each leading eigenvector v of M is well
     approximated by its cell-averaged version P v, i.e.
     leakage(v) = ||v - P v|| / ||v||  is small.

We compute both for the structural partition and compare to random same-size
partitions (z-score). Negative z = structural partition yields MORE
piecewise-constant eigenvectors / a more invariant subspace than chance.

We report for the matrix variant the partition was optimized on (default
transpose) and, for the dynamics interpretation, the raw recurrent matrix W.
Pure connectivity -- no TF model load.
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
    random_labels_same_sizes,
)

REPO = paths.REPO_ROOT
ACT_ROOT = paths.RESULTS_ROOT / "activity_clusters_paper"
ACT_NAME = "epoch_variance_unit_clusters_paper_max.npz"
ORIGINAL = paths.ORIGINAL_STORE
STAGED = paths.STAGED_STORE


def resolve(model: str) -> Path | None:
    for store in (ORIGINAL, STAGED):
        if (store / model).exists():
            return store / model
    return None


def pw_constant_projector(labels: np.ndarray, k: int) -> np.ndarray:
    H = np.zeros((labels.size, k), dtype=float)
    H[np.arange(labels.size), labels] = 1.0
    sizes = H.sum(axis=0)
    keep = sizes > 0
    H = H[:, keep]
    # P = H (H^T H)^{-1} H^T ; H^T H is diagonal(cell sizes)
    return H @ np.diag(1.0 / H.sum(axis=0)) @ H.T


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

    # structural labels at operating k (from the materialized sweep)
    loaded = _load_labels(model, k)
    if loaded is None:
        return None
    labels, actual_k = loaded
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


def _load_labels(model: str, k: int) -> tuple[np.ndarray, int] | None:
    """Return (labels, actual_k). Falls back to any materialized transpose_selected_k_<n>."""
    import glob as _glob
    from flexsym.structural.structural_cell_activity_coherence import load_structural_labels

    candidates = [
        (REPO / f"results/signed_input_wl_paper/{model}/transpose_selected_k_{k}/signed_input_wl_sweep.npz", k),
    ]
    # fall back to whatever single-k sweeps were materialized for this model
    for d in sorted(_glob.glob(str(REPO / f"results/signed_input_wl_paper/{model}/transpose_selected_k_*"))):
        try:
            kk = int(Path(d).name.rsplit("_", 1)[1])
        except ValueError:
            continue
        candidates.append((Path(d) / "signed_input_wl_sweep.npz", kk))
    for p, kk in candidates:
        if p.exists():
            try:
                labels, _, _ = load_structural_labels(p, kk)
                return labels, kk
            except Exception:
                continue
    return None


def load_operating_k() -> dict[str, int]:
    out: dict[str, int] = {}
    for s in [
        paths.RESULTS_ROOT / "signed_input_wl_paper/batch_rnn_softplus_diag_k20pct/k_selection_summary.csv",
        paths.RESULTS_ROOT / "signed_input_wl_paper/batch_rnn_softplus_diag_newseeds/k_selection_summary.csv",
    ]:
        if not s.exists():
            continue
        for r in csv.DictReader(s.open()):
            na = int(r["n_active"])
            sk = int(r["selected_k"]) if r.get("selected_k") else max(int(r["k_min"]), min(min(int(r["activity_k"]), int(r["structural_k"])), na - 1))
            out[r["model"]] = sk
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
    opk = load_operating_k()
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
