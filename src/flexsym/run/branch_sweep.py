"""Sweep the input-interpolation branch negative over k, task and seed.

The branch analysis (motif_quotient_branch) asks whether the reduced quotient system reproduces
the FULL network's re-found slow manifold as the input ramps delay->response. It is a NEGATIVE:
the reduced system's slow set collapses (effective dimension 0) while the full one persists, and
the reduced slow plane is far from the full one. This driver confirms that the negative is not
specific to one task/seed/k by running a small grid and summarising, per cell, the beta-averaged
mismatch: effective dimension (full vs reduced), the true-vs-predicted slow-plane principal angle,
and readout potency (full vs reduced).
"""
from __future__ import annotations
import argparse, json
import numpy as np

from flexsym import paths
from flexsym.run.motif_quotient_branch import run

FP = paths.RESULTS_ROOT / "motif_fixed_points"
NET = lambda s: f"LeakyRNN_softplus_diag_256_all_{s}"


def summarise(rows):
    a = lambda key: np.array([r[key] for r in rows], float)
    return {
        "eff_dim_true": float(np.nanmean(a("eff_dim_true"))),
        "eff_dim_pred": float(np.nanmean(a("eff_dim_pred"))),
        "n_slow_true": float(np.nanmean(a("n_slow_true"))),
        "n_slow_pred": float(np.nanmean(a("n_slow_pred"))),
        "tangent_angle_deg": float(np.nanmean(a("tangent_true_vs_pred"))),
        "potency_true": float(np.nanmean(a("potency_true"))),
        "potency_pred": float(np.nanmean(a("potency_pred"))),
        "loc_cos": float(np.nanmean(a("loc_cos"))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,3")
    ap.add_argument("--tasks", default="delaygo,delayanti")
    ap.add_argument("--ks", default="12,20,30")
    ap.add_argument("--n-beta", type=int, default=11)
    ap.add_argument("--seed-n", type=int, default=40)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    tasks = args.tasks.split(","); ks = [int(x) for x in args.ks.split(",")]

    out = {}
    for s in seeds:
        for task in tasks:
            for k in ks:
                cell = f"seed{s}/{task}/k{k}"
                try:
                    rows = run(NET(s), task, k, args.n_beta, args.seed_n)
                except FileNotFoundError as e:
                    print(f"[skip] {cell}: {e}", flush=True); continue
                out[cell] = summarise(rows)
                r = out[cell]
                print(f"[done] {cell}: effdim full {r['eff_dim_true']:.2f} / red {r['eff_dim_pred']:.2f} | "
                      f"tangent {r['tangent_angle_deg']:.0f}deg | potency full {r['potency_true']:.3f} / red {r['potency_pred']:.3f}",
                      flush=True)

    print("\n=== branch negative across the sweep ===")
    ed_t = np.mean([v["eff_dim_true"] for v in out.values()])
    ed_p = np.mean([v["eff_dim_pred"] for v in out.values()])
    ang = np.mean([v["tangent_angle_deg"] for v in out.values()])
    pt = np.mean([v["potency_true"] for v in out.values()]); pp = np.mean([v["potency_pred"] for v in out.values()])
    n = len(out)
    print(f" cells: {n}; mean eff-dim full {ed_t:.2f} vs reduced {ed_p:.2f}; "
          f"mean true-vs-reduced slow-plane angle {ang:.0f}deg; mean potency full {pt:.3f} vs reduced {pp:.3f}")
    print(f" reduced eff-dim < full in {sum(v['eff_dim_pred'] < v['eff_dim_true'] - 0.25 for v in out.values())}/{n} cells")

    (FP / "branch_sweep.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {FP/'branch_sweep.json'}")


if __name__ == "__main__":
    main()
