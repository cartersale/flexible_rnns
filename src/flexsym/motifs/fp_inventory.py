"""Classify the fixed-point inventory of a trained net from motif_fixed_points npz.

For each (rule, epoch) we recovered ~10^3 approximate fixed/slow points with
their Jacobian eigenvalues. We reduce this to an interpretable per-network
inventory:

  MARGINAL / attractor manifold: an epoch hosts a continuous attractor if its
    points are (a) numerically fixed (low q) and (b) marginally stable -- median
    max|eig(J)| ~ 1 (leaky discrete-time map). We count these.
  LINE vs RING geometry: PCA the fixed-point cloud. Intrinsic dimension ~1 (one PC
    dominates) => LINE attractor; ~2 AND circular (points fill an annulus of
    roughly constant radius spanning ~2pi) => RING. A filled 2-D blob is neither.

Per-network summary:
  n_attractor_epochs, n_line_epochs, n_ring_epochs,
  n_memory_attractor_epochs (restricted to memory/delay epochs),
  max_ring_score, mean_attractor_dim.
"""

from __future__ import annotations

from flexsym import paths

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

REPO = paths.REPO_ROOT
FP_ROOT = paths.RESULTS_ROOT / "motif_fixed_points"
OUT = paths.RESULTS_ROOT / "signed_input_wl_paper" / "subspace_dynamics"

MEMORY_EPOCHS = {"delay1", "delay2"}


def ring_score(x: np.ndarray) -> tuple[float, float, int]:
    """Return (ring_score, intrinsic_dim, n_pts) for a fixed-point cloud.

    ring_score in [0,1]: high when top-2 PCs dominate AND the 2-D projection is
    an annulus (radius CV low, angles cover the full circle). intrinsic_dim =
    participation ratio of the PCA spectrum.
    """
    n = x.shape[0]
    if n < 10:
        return 0.0, 0.0, n
    xc = x - x.mean(0)
    s = np.linalg.svd(xc, compute_uv=False)
    s2 = s**2
    pr = float((s2.sum() ** 2) / (s2**2).sum())  # participation ratio
    var2 = float(s2[:2].sum() / s2.sum())
    # project onto top-2 PCs
    U = np.linalg.svd(xc, full_matrices=False)[0][:, :2] * s[:2]
    r = np.linalg.norm(U, axis=1)
    ang = np.arctan2(U[:, 1], U[:, 0])
    radius_cv = float(r.std() / (r.mean() + 1e-12))
    # angular coverage: fraction of 24 angular bins occupied
    occ = float(np.mean(np.bincount(((ang + np.pi) / (2 * np.pi) * 24).astype(int) % 24,
                                     minlength=24) > 0))
    annulus = max(0.0, 1.0 - radius_cv)          # ~1 if constant radius
    rscore = var2 * annulus * occ                # all three must hold
    return float(rscore), pr, n


def classify_net(net_dir: Path, *, q_pct: float = 75.0, marg_thresh: float = 0.99,
                 ring_min: float = 0.35, line_dim_max: float = 1.4) -> dict:
    files = sorted(glob.glob(str(net_dir / "*__*.npz")))
    n_attr = n_line = n_ring = n_mem_attr = 0
    ring_scores, attr_dims = [], []
    per_epoch = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        x = np.asarray(z["xstar"], dtype=float)
        ev = np.asarray(z["eigvals"])
        q = np.asarray(z["qstar"], dtype=float)
        epoch = str(z["epoch"]) if "epoch" in z.files else Path(f).stem.split("__")[1]
        if x.shape[0] < 10:
            continue
        absmax = np.abs(ev).max(axis=1)
        marginal = float(np.median(absmax))
        is_attractor = marginal > marg_thresh          # continuous attractor manifold
        rscore, pr, n = ring_score(x)
        if is_attractor:
            n_attr += 1
            attr_dims.append(pr)
            ring_scores.append(rscore)
            if rscore >= ring_min and pr > line_dim_max:
                n_ring += 1
            elif pr <= line_dim_max:
                n_line += 1
            if epoch in MEMORY_EPOCHS:
                n_mem_attr += 1
        per_epoch.append({"file": Path(f).stem, "epoch": epoch, "n_pts": int(n),
                          "median_maxeig": marginal, "is_attractor": bool(is_attractor),
                          "pca_dim": float(pr), "ring_score": float(rscore)})
    return {
        "net": net_dir.name,
        "n_epochs_analyzed": len(per_epoch),
        "n_attractor_epochs": n_attr,
        "n_line_epochs": n_line,
        "n_ring_epochs": n_ring,
        "n_memory_attractor_epochs": n_mem_attr,
        "max_ring_score": float(max(ring_scores)) if ring_scores else 0.0,
        "mean_ring_score": float(np.mean(ring_scores)) if ring_scores else 0.0,
        "mean_attractor_dim": float(np.mean(attr_dims)) if attr_dims else 0.0,
        "_per_epoch": per_epoch,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nets", nargs="*", default=None, help="net dir names; default=all present")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    net_dirs = ([FP_ROOT / n for n in args.nets] if args.nets
                else sorted(d for d in FP_ROOT.iterdir() if d.is_dir()))
    rows = []
    for nd in net_dirs:
        if not nd.exists() or not list(nd.glob("*__*.npz")):
            print(f"SKIP {nd.name} (no FP npz)")
            continue
        r = classify_net(nd)
        per = r.pop("_per_epoch")
        rows.append(r)
        print(f"{r['net']}: attractor={r['n_attractor_epochs']} "
              f"line={r['n_line_epochs']} ring={r['n_ring_epochs']} "
              f"mem_attr={r['n_memory_attractor_epochs']} max_ring={r['max_ring_score']:.2f} "
              f"mean_dim={r['mean_attractor_dim']:.2f}")
        if args.verbose:
            for e in per:
                if e["is_attractor"]:
                    print(f"    {e['file']:28} dim={e['pca_dim']:.2f} ring={e['ring_score']:.2f} n={e['n_pts']}")
    if rows:
        with (OUT / "fp_inventory_256.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {OUT/'fp_inventory_256.csv'} ({len(rows)} nets)")


if __name__ == "__main__":
    main()
