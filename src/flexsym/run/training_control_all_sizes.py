"""Training control across ALL network sizes (128/256/512/1024), not just 256.

Runs the identical pipeline as run_training_control.py -- WL-optimize a partition at the
network's operating k on each matrix, then score equitability z and eigenvector step-R^2
against size-matched random partitions -- for every network in lumpability.csv, whose `k`
column is the operating scale used throughout the paper.

  trained       : the real recurrent matrix (transpose variant, active units)
  row-shuffle   : permute each row -> preserves each unit's input strength, kills the pattern
  full-shuffle  : permute all entries -> preserves the value distribution only
  gaussian      : i.i.d. normal matched in Frobenius norm (an untrained network)

Writes results/signed_input_wl_paper/training_control_all_sizes.csv.
Pure connectivity -- no TF model load.
"""

from __future__ import annotations

from flexsym import paths

import csv
import sys
from pathlib import Path

import numpy as np


from flexsym.run.training_control import TYPES, analyse_matrix, make_matrix, resolve  # noqa: E402
from flexsym.structural.signed_input_wl_structural_roles import apply_matrix_variant, load_w_rec  # noqa: E402

REPO = paths.REPO_ROOT
ACT = paths.RESULTS_ROOT / "activity_clusters_paper"
LUMP = paths.RESULTS_ROOT / "signed_input_wl_paper/lumpability/lumpability.csv"
OUT = paths.RESULTS_ROOT / "signed_input_wl_paper/training_control_all_sizes.csv"


def main():
    todo = list(csv.DictReader(open(LUMP)))
    rows = []
    for i, r in enumerate(todo, 1):
        model, size, seed, k = r["model"], int(r["size"]), int(r["seed"]), int(r["k"])
        md = resolve(model)
        act = ACT / model / "epoch_variance_unit_clusters_paper_max.npz"
        if md is None or not act.exists():
            print(f"[{i}/{len(todo)}] skip {model} (missing weights or activity)", flush=True)
            continue
        ind = np.asarray(np.load(act, allow_pickle=True)["ind_active"], dtype=np.int64)
        W = apply_matrix_variant(load_w_rec(md), "transpose")[np.ix_(ind, ind)]
        rng = np.random.default_rng(seed)
        rec = {"model": model, "size": size, "seed": seed, "k": k, "n_active": len(ind)}
        for kind in TYPES:
            M = make_matrix(W, kind, rng)
            z_eq, r2, z_lump = analyse_matrix(M, k, rng)
            rec[f"{kind}_equit_z"] = z_eq
            rec[f"{kind}_lump_r2"] = r2
            rec[f"{kind}_lump_z"] = z_lump
        rows.append(rec)
        print(f"[{i}/{len(todo)}] {size}u seed {seed} (k={k}, n={len(ind)}): "
              + " ".join(f"{t[:4]}: eq={rec[f'{t}_equit_z']:+.1f} R2={rec[f'{t}_lump_r2']:.2f}"
                         for t in TYPES), flush=True)

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== mean over seeds, by size (equitability score = -z; step R^2) ===")
    for size in (128, 256, 512, 1024):
        sub = [r for r in rows if r["size"] == size]
        if not sub:
            continue
        print(f"  {size:>4}u (n={len(sub)}): " + "  ".join(
            f"{t}: score {np.mean([-r[f'{t}_equit_z'] for r in sub]):+5.2f} / "
            f"R2 {np.mean([r[f'{t}_lump_r2'] for r in sub]):.3f}" for t in TYPES))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
