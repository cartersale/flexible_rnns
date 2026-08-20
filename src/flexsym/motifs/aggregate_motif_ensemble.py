"""Aggregate the connectivity->motif participation tests across networks (C1).

Seed is the unit of replication (design plan C1): compute one summary per
network, then test across networks with a Wilcoxon signed-rank test. Reads every
results/motif_fixed_points/<net>/participation_stats.json and reports, overall
and per motif type:

  * median eta^2-z (role predicts participation)          -> Test 1
  * unique-to-role CV R^2, signed-rank over networks vs 0 -> Test 2 (headline)
  * unique-to-activity CV R^2 (reference)
  * line-attractor cross-task reuse minus baseline, signed-rank vs 0 -> Test 3

Usage:
  conda run -n myenv python stepnet/aggregate_motif_ensemble.py
  conda run -n myenv python stepnet/aggregate_motif_ensemble.py --glob '*_256_*'
"""

from __future__ import annotations

from flexsym import paths

import argparse
import collections
import json
from pathlib import Path

import numpy as np

STEP_DIR = Path(__file__).resolve().parent
FP_ROOT = paths.RESULTS_ROOT / "motif_fixed_points"

MOTIF_TYPES = ("line_attractor", "saddle_decision", "point_attractor", "rotation")


def _signed_rank(values):
    """Wilcoxon signed-rank vs 0; returns (statistic, p, n, median)."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    n = v.size
    med = float(np.median(v)) if n else float("nan")
    if n < 3 or np.allclose(v, 0):
        return None, float("nan"), n, med
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(v)
        return float(stat), float(p), n, med
    except Exception:
        return None, float("nan"), n, med


def per_net_summaries(paths):
    """One record per network with overall + per-motif-type medians."""
    nets = []
    for p in paths:
        d = json.load(open(p))
        pm = d.get("per_motif", [])
        if not pm:
            continue
        by = collections.defaultdict(list)
        for m in pm:
            by[m["motif"]].append(m)
        sub = [m for m in pm if "sub_z_role" in m]
        rec = {
            "net": d["net"], "k": d.get("k"),
            "median_z_role": float(np.median([m["z_role"] for m in pm])),
            "median_unique_role": float(np.median([m["cv_unique_role"] for m in pm])),
            "median_unique_activity":
                float(np.median([m["cv_unique_activity"] for m in pm])),
            "median_subspace_z_role":
                float(np.median([m["sub_z_role"] for m in sub])) if sub else float("nan"),
            "median_subspace_frac":
                float(np.median([m["sub_frac"] for m in sub])) if sub else float("nan"),
            "line_reuse_minus_baseline":
                d.get("reuse", {}).get("line_reuse_minus_baseline", float("nan")),
            "by_type": {},
        }
        for mt, ms in by.items():
            rec["by_type"][mt] = {
                "n": len(ms),
                "median_z_role": float(np.median([m["z_role"] for m in ms])),
                "median_unique_role":
                    float(np.median([m["cv_unique_role"] for m in ms])),
                "median_unique_activity":
                    float(np.median([m["cv_unique_activity"] for m in ms])),
            }
        nets.append(rec)
    return nets


def aggregate(nets):
    out = {"n_networks": len(nets), "overall": {}, "by_type": {}}

    def report(values, label):
        stat, p, n, med = _signed_rank(values)
        return {"median": med, "n_nets": n, "wilcoxon_p": p,
                "frac_pos": float(np.mean(np.asarray(values) > 0)) if values else float("nan")}

    out["overall"] = {
        "median_z_role": float(np.median([r["median_z_role"] for r in nets])),
        "unique_role": report([r["median_unique_role"] for r in nets], "unique_role"),
        "unique_activity":
            report([r["median_unique_activity"] for r in nets], "unique_activity"),
        "subspace_z_role":
            report([r["median_subspace_z_role"] for r in nets], "subspace_z"),
        "median_subspace_frac":
            float(np.median([r["median_subspace_frac"] for r in nets])),
        "line_reuse_minus_baseline":
            report([r["line_reuse_minus_baseline"] for r in nets], "reuse"),
    }
    for mt in MOTIF_TYPES:
        vals_ur = [r["by_type"][mt]["median_unique_role"]
                   for r in nets if mt in r["by_type"]]
        vals_z = [r["by_type"][mt]["median_z_role"]
                  for r in nets if mt in r["by_type"]]
        if not vals_ur:
            continue
        out["by_type"][mt] = {
            "median_z_role": float(np.median(vals_z)),
            "unique_role": report(vals_ur, f"unique_role[{mt}]"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*", help="net-name glob under results/motif_fixed_points")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = sorted(FP_ROOT.glob(f"{args.glob}/participation_stats.json"))
    if not paths:
        print(f"no participation_stats.json under {FP_ROOT}/{args.glob}")
        return
    nets = per_net_summaries(paths)
    agg = aggregate(nets)

    print(f"=== ensemble aggregation ({agg['n_networks']} networks) ===")
    o = agg["overall"]
    print(f"median eta^2-z (role):            {o['median_z_role']:.2f}")
    ur = o["unique_role"]
    print(f"unique-to-role (headline):        median {ur['median']:+.3f}  "
          f"frac_pos {ur['frac_pos']:.2f}  signed-rank p={ur['wilcoxon_p']:.4g}  "
          f"(n={ur['n_nets']})")
    ua = o["unique_activity"]
    print(f"unique-to-activity (reference):   median {ua['median']:+.3f}  "
          f"p={ua['wilcoxon_p']:.4g}")
    sz = o["subspace_z_role"]
    print(f"subspace lumpability z (role):    median {sz['median']:+.2f}  "
          f"frac_pos {sz['frac_pos']:.2f}  signed-rank p={sz['wilcoxon_p']:.4g}  "
          f"(subspace_frac median {o['median_subspace_frac']:.3f})")
    lr = o["line_reuse_minus_baseline"]
    print(f"line-attractor reuse - baseline:  median {lr['median']:+.3f}  "
          f"frac_pos {lr['frac_pos']:.2f}  p={lr['wilcoxon_p']:.4g}")
    print("\nby motif type (unique-to-role):")
    for mt, v in agg["by_type"].items():
        u = v["unique_role"]
        print(f"  {mt:16s} median {u['median']:+.3f}  frac_pos {u['frac_pos']:.2f}  "
              f"p={u['wilcoxon_p']:.4g}  z_role {v['median_z_role']:.2f}  (n={u['n_nets']})")

    out = Path(args.out) if args.out else FP_ROOT / "ensemble_aggregate.json"
    json.dump({"summary": agg, "per_net": nets}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
