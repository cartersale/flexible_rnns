"""Aggregate the per-network motif-participation tests across the ten 256-unit networks.

run/motif_participation_stats.py writes one participation_stats.json per network and
leaves aggregation "to be done separately once the ensemble has been processed" -- this
is that step. The trained network is the unit of replication: every per-network value is
the median over that network's conditions, and cross-network claims are tested with a
Wilcoxon signed-rank over the ten networks.

Reports, per motif type and pooled:

  eta2_role / z_role      does role membership predict per-unit motif participation
                          (vs size-matched label permutations)
  cv_unique_role          variance in participation explained by roles and NOT by the
  cv_unique_activity      activity clusters, and vice versa, from the nested ridge-CV
  cv_shared               commonality decomposition (M0..M3)
  sub_step_r2 / sub_z_*   whether the motif's own signed directions lie in the
                          group-constant subspace, scored against role and activity

Writes results/signed_input_wl_paper/motif_participation/aggregate_by_motif.csv,
aggregate_by_net.csv and aggregate_summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from scipy.stats import wilcoxon

from flexsym import paths

FP_ROOT = paths.RESULTS_ROOT / "motif_fixed_points"
OUT_DIR = paths.RESULTS_ROOT / "signed_input_wl_paper" / "motif_participation"
MOTIFS = ("line_attractor", "saddle_decision", "point_attractor", "rotation")
FIELDS = ("eta2_role", "z_role", "eta2_activity",
          "cv_M1_activity", "cv_M3_role", "cv_M2_both",
          "cv_unique_role", "cv_unique_activity", "cv_shared",
          "sub_step_r2", "sub_null_r2", "sub_z_role",
          "sub_step_r2_activity", "sub_z_activity")


def signed_rank(vals, alt="greater"):
    """Wilcoxon signed-rank of per-network values against zero."""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size < 3 or np.allclose(v, 0):
        return {"n": int(v.size), "median": float(np.median(v)) if v.size else np.nan,
                "p": np.nan}
    try:
        _, p = wilcoxon(v, alternative=alt)
    except ValueError:
        p = np.nan
    return {"n": int(v.size), "median": float(np.median(v)), "p": float(p)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seeds", default="0-9")
    args = ap.parse_args()
    a, b = args.seeds.split("-")
    seeds = list(range(int(a), int(b) + 1))

    nets, missing = [], []
    for s in seeds:
        net = f"LeakyRNN_softplus_diag_{args.size}_all_{s}"
        p = FP_ROOT / net / "participation_stats.json"
        if not p.exists():
            missing.append(net)
            continue
        nets.append(json.loads(p.read_text()))
    if missing:
        print("missing participation_stats.json: " + ", ".join(missing))

    # ---- per-network medians, overall and by motif type ---------------------
    by_net_rows = []
    for d in nets:
        pm = d["per_motif"]
        for label, subset in [("all", pm)] + [(m, [r for r in pm if r["motif"] == m])
                                              for m in MOTIFS]:
            if not subset:
                continue
            row = {"net": d["net"], "k": d["k"], "motif": label, "n_cond": len(subset)}
            for f in FIELDS:
                vals = [r[f] for r in subset if f in r and np.isfinite(r[f])]
                row[f] = float(np.median(vals)) if vals else np.nan
            row["frac_role_sig"] = float(np.mean([r["p_perm_role"] < 0.05 for r in subset]))
            by_net_rows.append(row)

    # ---- cross-network signed-rank tests ------------------------------------
    by_motif_rows, summary = [], {}
    for label in ("all",) + MOTIFS:
        sub = [r for r in by_net_rows if r["motif"] == label]
        if len(sub) < 3:
            continue
        rec = {"motif": label, "n_nets": len(sub),
               "n_cond_total": int(sum(r["n_cond"] for r in sub))}
        for f in FIELDS + ("frac_role_sig",):
            v = [r[f] for r in sub]
            rec[f"{f}_median"] = float(np.nanmedian(v))
            rec[f"{f}_iqr_lo"] = float(np.nanpercentile(v, 25))
            rec[f"{f}_iqr_hi"] = float(np.nanpercentile(v, 75))
        for f in ("cv_unique_role", "cv_unique_activity", "z_role",
                  "sub_z_role", "sub_z_activity"):
            t = signed_rank([r[f] for r in sub])
            rec[f"{f}_p"] = t["p"]
        # is role uniqueness larger than activity uniqueness?
        d_ru = [r["cv_unique_role"] - r["cv_unique_activity"] for r in sub]
        rec["role_minus_activity_median"] = float(np.nanmedian(d_ru))
        rec["role_minus_activity_p"] = signed_rank(d_ru, alt="two-sided")["p"]
        by_motif_rows.append(rec)
        summary[label] = rec

    # ---- reuse across tasks --------------------------------------------------
    reuse = {}
    diff = [d["reuse"]["different_type_mean"] for d in nets]
    reuse["different_type_median"] = float(np.median(diff))
    for m in MOTIFS:
        vals = [d["reuse"]["same_type"][m]["mean"] for d in nets
                if m in d["reuse"]["same_type"] and np.isfinite(d["reuse"]["same_type"][m]["mean"])]
        if vals:
            reuse[f"same_{m}_median"] = float(np.median(vals))
    excess = [d["reuse"]["line_reuse_minus_baseline"] for d in nets]
    reuse["line_reuse_minus_baseline"] = signed_rank(excess)
    summary["reuse"] = reuse
    summary["k_by_net"] = {d["net"]: d["k"] for d in nets}
    summary["n_conditions_by_net"] = {d["net"]: d["n_motifs"] for d in nets}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("aggregate_by_net.csv", by_net_rows),
                       ("aggregate_by_motif.csv", by_motif_rows)):
        with (OUT_DIR / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (OUT_DIR / "aggregate_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---- report --------------------------------------------------------------
    print(f"\n{len(nets)} networks, k = {sorted(set(d['k'] for d in nets))}")
    print(f"\n{'motif':17} {'nets':>4} {'cond':>5} {'eta2_role':>10} {'z_role':>7} "
          f"{'uniq_role':>10} {'uniq_act':>9} {'shared':>8} {'p(role>0)':>10}")
    for r in by_motif_rows:
        print(f"{r['motif']:17} {r['n_nets']:>4} {r['n_cond_total']:>5} "
              f"{r['eta2_role_median']:>10.3f} {r['z_role_median']:>7.1f} "
              f"{r['cv_unique_role_median']:>10.3f} {r['cv_unique_activity_median']:>9.3f} "
              f"{r['cv_shared_median']:>8.3f} {r['cv_unique_role_p']:>10.2g}")
    print(f"\n{'motif':17} {'sub_r2':>8} {'sub_null':>9} {'sub_zRole':>10} {'sub_zAct':>9}")
    for r in by_motif_rows:
        print(f"{r['motif']:17} {r['sub_step_r2_median']:>8.3f} {r['sub_null_r2_median']:>9.3f} "
              f"{r['sub_z_role_median']:>10.1f} {r['sub_z_activity_median']:>9.1f}")
    print(f"\nreuse: different-type r {reuse['different_type_median']:.3f}; "
          + "; ".join(f"{m.split('_')[0]} {reuse[f'same_{m}_median']:.3f}"
                      for m in MOTIFS if f"same_{m}_median" in reuse))
    lr = reuse["line_reuse_minus_baseline"]
    print(f"line-attractor reuse minus baseline: median {lr['median']:.3f} "
          f"(n={lr['n']}, p={lr['p']:.2g})")
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
