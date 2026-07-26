"""Run the TF-free metrics (Tiers A+B) across the restart ensemble.

For every 256-unit network and every persisted partition -- 20 WL restarts, the
co-association consensus, and 20 size-matched random partitions -- compute the
structural metrics (Gamma_signed, equitability z, leakage, invariance, linear &
Jacobian step-R^2) and the two activity-based downstream effects (reorganization
gap, connectivity->function partial r). Hidden activity is computed once per
network via ``ActivityCache`` and reused across all 41 partitions.

One row per (network, partition_kind, partition_index) -> CSV consumed by the
stability statistics (sign grid, ICC, Gamma-vs-effect regression). The lesion
effect is added by the separate TF driver ``run_ensemble_lesion.py``.
"""

from __future__ import annotations

from flexsym import paths

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np


from flexsym.structural.ensemble_analyze import (  # noqa: E402
    ActivityCache,
    all_needed_epochs,
    analyze,
)
from flexsym.structural.signed_input_wl_structural_roles import apply_matrix_variant, load_w_rec  # noqa: E402

REPO = paths.REPO_ROOT
ENS = REPO / "results" / "signed_input_wl_paper" / "restart_ensemble"
OUT = REPO / "results" / "signed_input_wl_paper" / "restart_ensemble"
ORIG = paths.ORIGINAL_STORE
STAGED = paths.STAGED_STORE


def resolve(model: str) -> Path | None:
    for store in (ORIG, STAGED):
        if (store / model).exists():
            return store / model
    return None


def run_net(seed: int, *, n_null: int, batch_size: int, tiers: tuple[str, ...]) -> list[dict]:
    model = f"LeakyRNN_softplus_diag_256_all_{seed}"
    ens_path = ENS / f"{model}_ensemble.npz"
    md = resolve(model)
    if not ens_path.exists() or md is None:
        print(f"SKIP {model}")
        return []
    z = np.load(ens_path, allow_pickle=True)
    k = int(z["k"])
    ind_active = np.asarray(z["ind_active"], dtype=np.int64)
    restart_labels = np.asarray(z["restart_labels"], dtype=np.int64)
    consensus_labels = np.asarray(z["consensus_labels"], dtype=np.int64)
    random_labels = np.asarray(z["random_labels"], dtype=np.int64)
    restart_gamma = np.asarray(z["restart_gamma"], dtype=float)

    W_full = load_w_rec(md)
    W_sub = apply_matrix_variant(W_full, "transpose")[np.ix_(ind_active, ind_active)]

    hidden = None
    if "B" in tiers:
        cache = ActivityCache(model, md, batch_size=batch_size)
        t0 = time.time()
        hidden = cache.get(all_needed_epochs())
        print(f"[{model}] hidden: {len(hidden)} epochs in {time.time()-t0:.0f}s", flush=True)

    partitions = [("restart", r, restart_labels[r]) for r in range(len(restart_labels))]
    partitions.append(("consensus", -1, consensus_labels))
    partitions += [("random", r, random_labels[r]) for r in range(len(random_labels))]

    rows = []
    for kind, idx, labels in partitions:
        t0 = time.time()
        m = analyze(
            W_sub, labels, k,
            ind_active=ind_active, W_rec_full=W_full, hidden=hidden,
            n_null=n_null, seed=(idx if idx >= 0 else 999), tiers=tiers,
        )
        row = {"model": model, "seed": seed, "k": k, "kind": kind, "index": idx,
               "gamma_restart": float(restart_gamma[idx]) if kind == "restart" else float("nan")}
        row.update(m)
        rows.append(row)
        print(f"  {kind:9} {idx:>3}  gap={m.get('reorg_gap', float('nan')):+.3f} "
              f"pr={m.get('partial_r', float('nan')):+.3f} "
              f"gz={m.get('gamma_z', float('nan')):+.2f} "
              f"r2j={m.get('step_r2_jacobian', float('nan')):.3f} "
              f"({time.time()-t0:.1f}s)", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    p.add_argument("--n-null", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--tiers", default="AB", help="subset of A/B to run")
    p.add_argument("--out", default=str(OUT / "ensemble_metrics_nolesion.csv"))
    args = p.parse_args()
    tiers = tuple(t for t in ("A", "B") if t in args.tiers.upper())

    all_rows: list[dict] = []
    for seed in args.seeds:
        all_rows += run_net(seed, n_null=args.n_null, batch_size=args.batch_size, tiers=tiers)
    if not all_rows:
        raise SystemExit("no rows")

    fieldnames = sorted({kk for r in all_rows for kk in r})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({kk: r.get(kk, "") for kk in fieldnames})
    print(f"\nwrote {out_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
