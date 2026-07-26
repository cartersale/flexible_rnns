"""Run the connectivity->motif participation tests for one network.

Loads the recovered motifs (results/motif_fixed_points/<net>/) + cached
partitions, then runs:
  Test 1  role-group predicts participation (eta^2 + permutation z), per motif.
  Test 2  nested M0..M3 CV R^2 + commonality (unique-to-role headline), per motif.
  Test 3  motif reuse across tasks (same-type vs different-type p_m correlation).

Writes results/motif_fixed_points/<net>/participation_stats.json and a per-motif
CSV. Aggregation across networks (signed-rank over seeds) is done separately
once the ensemble has been processed.

  conda run -n myenv python stepnet/run_motif_participation_stats.py \
      --net LeakyRNN_softplus_diag_256_all_0
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

import sys
STEP_DIR = Path(__file__).resolve().parent
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from motif_classify import classify_net, npload  # noqa: E402
from motif_participation import (  # noqa: E402
    load_partitions, perm_z, nested_commonality, eta_squared, subspace_lumpability,
)

REPO_DIR = STEP_DIR.parent
FP_ROOT = REPO_DIR / "results" / "motif_fixed_points"

# motif types with a well-defined participation vector
MOTIF_TYPES = ("line_attractor", "saddle_decision", "rotation", "point_attractor")


def collect_motifs(net_dir, ind_active, use_cache=True):
    """One record per condition with a participation vector, restricted to active.

    Classification (the eig-heavy step) is cached to participation_cache.npz so
    re-running the statistics (e.g. iterating on the regression) is fast.
    """
    cache = Path(net_dir) / "participation_cache_v2.npz"
    if use_cache and cache.exists():
        d = npload(cache)
        recs = list(d["records"])
        if np.array_equal(np.asarray(d["ind_active"]), ind_active):
            return recs
    res = classify_net(net_dir)
    out = []
    for key, cm in res.items():
        if cm.participation is None or cm.condition_motif not in MOTIF_TYPES:
            continue
        p_full = np.asarray(cm.participation, dtype=float)
        p_act = p_full[ind_active]
        if np.linalg.norm(p_act) == 0:
            continue
        vecs_act = None
        if cm.signed_eigvecs is not None and len(cm.signed_eigvecs):
            v = np.asarray(cm.signed_eigvecs, dtype=float)[:, ind_active]
            if v.shape[0] > 100:                 # cap cache size
                v = v[np.linspace(0, v.shape[0] - 1, 100).astype(int)]
            vecs_act = v
        out.append({
            "key": key, "rule": cm.rule, "epoch": cm.epoch,
            "motif": cm.condition_motif, "epoch_type": cm.epoch_type,
            "p_act": p_act, "vecs_act": vecs_act,
            "n_fp": cm.extra["n_fp"], "n_saddle": cm.extra["n_saddle"],
        })
    np.savez(cache, records=np.array(out, dtype=object), ind_active=ind_active)
    return out


def run_net(net, k=None, n_perm=1000, folds=5, seed=0):
    net_dir = FP_ROOT / net
    part = load_partitions(net, k=k)
    ind_active = part["ind_active"]
    role, act = part["role_labels"], part["act_labels"]
    rng = np.random.default_rng(seed)

    motifs = collect_motifs(net_dir, ind_active)
    per_motif = []
    for m in motifs:
        y = m["p_act"]
        t1 = perm_z(y, role, n_perm=n_perm, rng=rng)
        t2 = nested_commonality(y, role, act, folds=folds, rng=rng)
        # activity-cluster eta^2 for reference
        eta_act = eta_squared(y, act)
        rec = {
            "key": m["key"], "rule": m["rule"], "epoch": m["epoch"],
            "motif": m["motif"], "epoch_type": m["epoch_type"],
            "eta2_role": t1["eta2"], "z_role": t1["z"], "p_perm_role": t1["p_perm"],
            "eta2_activity": eta_act,
            **{f"cv_{kk}": vv for kk, vv in t2.items()},
        }
        # Test 4: subspace / lumpability on the signed motif directions
        if m.get("vecs_act") is not None and len(m["vecs_act"]):
            t4 = subspace_lumpability(m["vecs_act"], role, rng=rng)
            # same measurement against the ACTIVITY partition, for reference
            t4a = subspace_lumpability(m["vecs_act"], act, rng=rng)
            rec.update({
                "sub_step_r2": t4["step_r2"], "sub_frac": t4["subspace_frac"],
                "sub_z_role": t4["z"], "sub_null_r2": t4["null_r2_mean"],
                "sub_z_activity": t4a["z"], "sub_step_r2_activity": t4a["step_r2"],
            })
        per_motif.append(rec)

    # ---- Test 3: motif reuse across tasks ----
    reuse = reuse_test(motifs)

    summary = summarize(per_motif)
    result = {"net": net, "k": part["k"], "n_motifs": len(motifs),
              "per_motif": per_motif, "reuse": reuse, "summary": summary}

    # write outputs
    with open(net_dir / "participation_stats.json", "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    if per_motif:
        with open(net_dir / "participation_per_motif.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_motif[0].keys()))
            w.writeheader()
            w.writerows(per_motif)
    return result


def reuse_test(motifs):
    """Same-type vs different-type participation correlation across tasks.

    Correlations are between motif instances in DIFFERENT tasks (rules), so the
    signal is 'the same computation reuses the same role groups across tasks',
    not within-task epoch similarity.
    """
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    same, diff = {mt: [] for mt in MOTIF_TYPES}, []
    for a, b in combinations(motifs, 2):
        if a["rule"] == b["rule"]:
            continue                      # cross-task only
        c = corr(a["p_act"], b["p_act"])
        if a["motif"] == b["motif"]:
            same[a["motif"]].append(c)
        else:
            diff.append(c)
    out = {"different_type_mean": float(np.mean(diff)) if diff else float("nan"),
           "different_type_n": len(diff), "same_type": {}}
    for mt, cs in same.items():
        if cs:
            out["same_type"][mt] = {"mean": float(np.mean(cs)), "n": len(cs)}
    # headline: line-attractor same-type reuse minus different-type baseline
    la = out["same_type"].get("line_attractor", {}).get("mean", float("nan"))
    out["line_reuse_minus_baseline"] = (
        la - out["different_type_mean"]
        if not np.isnan(la) and not np.isnan(out["different_type_mean"])
        else float("nan")
    )
    return out


def summarize(per_motif):
    if not per_motif:
        return {}
    z = np.array([m["z_role"] for m in per_motif])
    ur = np.array([m["cv_unique_role"] for m in per_motif])
    ua = np.array([m["cv_unique_activity"] for m in per_motif])
    out = {
        "n_motifs": len(per_motif),
        "median_z_role": float(np.median(z)),
        "frac_z_gt2": float(np.mean(z > 2)),
        "median_unique_role": float(np.median(ur)),
        "frac_unique_role_pos": float(np.mean(ur > 0)),
        "median_unique_activity": float(np.median(ua)),
    }
    sub = [m for m in per_motif if "sub_z_role" in m]
    if sub:
        sz = np.array([m["sub_z_role"] for m in sub])
        sf = np.array([m["sub_frac"] for m in sub])
        out.update({
            "n_motifs_subspace": len(sub),
            "median_subspace_frac": float(np.median(sf)),
            "median_subspace_z_role": float(np.median(sz)),
            "frac_subspace_z_gt2": float(np.mean(sz > 2)),
            "median_subspace_z_activity":
                float(np.median([m["sub_z_activity"] for m in sub])),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="LeakyRNN_softplus_diag_256_all_0")
    ap.add_argument("--k", type=int, default=None, help="override role-group k")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    res = run_net(args.net, k=args.k, n_perm=args.n_perm, folds=args.folds,
                  seed=args.seed)
    s = res["summary"]
    print(f"\n=== {args.net}  (role-group k={res['k']}, {res['n_motifs']} motifs) ===")
    print(f"{'condition':30s} {'motif':16s} {'z_role':>7s} "
          f"{'uniq_role':>9s} {'uniq_act':>9s} {'sub_frac':>8s} {'sub_zR':>7s} "
          f"{'sub_zA':>7s}")
    for m in res["per_motif"]:
        print(f"{m['key']:30s} {m['motif']:16s} "
              f"{m['z_role']:7.2f} {m['cv_unique_role']:9.3f} "
              f"{m['cv_unique_activity']:9.3f} "
              f"{m.get('sub_frac', float('nan')):8.3f} "
              f"{m.get('sub_z_role', float('nan')):7.2f} "
              f"{m.get('sub_z_activity', float('nan')):7.2f}")
    print("\n-- summary --")
    for kk, vv in s.items():
        print(f"  {kk:26s} {vv}")
    print("\n-- reuse (cross-task) --")
    r = res["reuse"]
    print(f"  different-type mean r: {r['different_type_mean']:.3f} "
          f"(n={r['different_type_n']})")
    for mt, d in r["same_type"].items():
        print(f"  same-type {mt:16s} mean r: {d['mean']:.3f} (n={d['n']})")
    print(f"  line-attractor reuse minus baseline: "
          f"{r['line_reuse_minus_baseline']:.3f}")


if __name__ == "__main__":
    main()
