"""Random-partition control for the cross-task motif-reuse result.

The reuse statistic

    R = mean corr( p_a, p_b | same motif type, different tasks )
      - mean corr( p_a, p_b | different motif type, different tasks )

was previously computed on the RAW per-unit participation vectors, so it says
nothing about the connectivity partition: it is a statement about units, not
about role groups. This script makes reuse partition-conditioned and asks
whether it is SPECIFIC to connectivity.

Method. Coarse-grain each participation profile onto a partition before
correlating: p -> P p, where P is the group-constant projector (each unit gets
its group's mean). R computed on the projected profiles measures how much of the
same-computation-reuses-the-same-units signal survives being described only at
the level of that partition's groups. Then compare

    R_connectivity   (signed-WL role groups)
    R_random         (partitions with the SAME group-size distribution)
    R_activity       (activity clusters, at the SAME k)

Two matched controls, as in run_motif_subspace_rva.py:
  * matched k: role and activity labels are taken from the same row of the WL
    sweep (labels_struct_by_k / labels_act_by_k), swept over a k grid.
  * matched group sizes: each partition is compared to its OWN size-matched
    random ensemble, giving excess reuse
        Delta_R = R_obs - mean(R_random_same_sizes)
    so Delta_R_conn - Delta_R_act isolates grouping structure from size profile.

Per-net statistic at each k: Delta_R_conn (is reuse specific to connectivity at
all?) and Delta_R_conn - Delta_R_act (is it more specific than activity?).
Signed-rank across seeds.

  conda run -n myenv python stepnet/run_motif_reuse_control.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

STEP_DIR = Path(__file__).resolve().parent
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from motif_classify import npload  # noqa: E402
from motif_participation import WL_ROOT, ACT_ROOT, ACT_NAME  # noqa: E402

FP_ROOT = STEP_DIR.parent / "results" / "motif_fixed_points"
K_GRID = [8, 12, 16, 20, 25, 30, 40, 50]
N_PERM = 200
MOTIF_TYPES = ("line_attractor", "saddle_decision", "rotation", "point_attractor")


# --------------------------------------------------------------------------- #
# reuse statistic
# --------------------------------------------------------------------------- #
def project(P, labels):
    """Group-constant projection of every row of P (m, N): unit -> group mean."""
    cats, inv = np.unique(labels, return_inverse=True)
    G = np.zeros((len(labels), cats.size))
    G[np.arange(len(labels)), inv] = 1.0
    sizes = G.sum(0)                       # every group is non-empty by construction
    with np.errstate(all="ignore"):        # numpy>=2 BLAS raises spurious FP flags
        return (P @ G / sizes) @ G.T


def pair_masks(motifs):
    """Boolean (m, m) upper-triangular masks: same-type and different-type,
    both restricted to pairs from DIFFERENT tasks (rules)."""
    types = np.array([m["motif"] for m in motifs])
    rules = np.array([m["rule"] for m in motifs])
    cross = rules[:, None] != rules[None, :]
    same_t = types[:, None] == types[None, :]
    iu = np.triu(np.ones((len(motifs), len(motifs)), dtype=bool), k=1)
    return (cross & same_t & iu), (cross & ~same_t & iu), types


def reuse_R(P, same_mask, diff_mask):
    """R = mean same-type corr - mean different-type corr, over cross-task pairs."""
    C = np.corrcoef(P)
    if not np.any(same_mask) or not np.any(diff_mask):
        return float("nan")
    return float(np.nanmean(C[same_mask]) - np.nanmean(C[diff_mask]))


def reuse_by_type(P, motifs, same_mask, diff_mask, types):
    """Per-motif-type R, using the shared different-type baseline."""
    C = np.corrcoef(P)
    base = np.nanmean(C[diff_mask]) if np.any(diff_mask) else np.nan
    out = {}
    for mt in MOTIF_TYPES:
        sel = same_mask & (types[:, None] == mt) & (types[None, :] == mt)
        if np.any(sel):
            out[mt] = {"R": float(np.nanmean(C[sel]) - base),
                       "n_pairs": int(sel.sum())}
    return out


def excess_reuse(P, labels, same_mask, diff_mask, n_perm, rng):
    """R on the partition-projected profiles, vs size-matched random partitions.

    Returns (R_obs, R_rand_mean, Delta, z, p_perm).
    """
    obs = reuse_R(project(P, labels), same_mask, diff_mask)
    lab = np.asarray(labels).copy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(lab)               # random partition, SAME size distribution
        null[i] = reuse_R(project(P, lab), same_mask, diff_mask)
    mu, sd = float(np.nanmean(null)), float(np.nanstd(null))
    z = (obs - mu) / sd if sd > 0 else 0.0
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return obs, mu, obs - mu, float(z), p


# --------------------------------------------------------------------------- #
# per-net driver
# --------------------------------------------------------------------------- #
def _row_for_k(labels_by_k, k):
    counts = np.array([np.unique(labels_by_k[r]).size
                       for r in range(labels_by_k.shape[0])])
    exact = np.where(counts == k)[0]
    return int(exact[0]) if exact.size else int(np.argmin(np.abs(counts - k)))


def collect_participation(net_dir):
    d = npload(Path(net_dir) / "participation_cache_v2.npz")
    out = []
    for m in list(d["records"]):
        p = np.asarray(m["p_act"], dtype=float)
        if not np.all(np.isfinite(p)) or np.linalg.norm(p) == 0:
            continue
        if m["motif"] not in MOTIF_TYPES:
            continue
        out.append({"motif": m["motif"], "rule": m["rule"],
                    "epoch": m["epoch"], "p": p})
    return out


def run_net(net, k_grid, n_perm, rng):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    motifs = collect_participation(FP_ROOT / net)
    if len(motifs) < 4:
        return None
    P = np.vstack([m["p"] for m in motifs])
    same_mask, diff_mask, types = pair_masks(motifs)
    if not np.any(same_mask) or not np.any(diff_mask):
        return None

    ls, la = wl["labels_struct_by_k"], wl["labels_act_by_k"]
    res = {"n_motifs": len(motifs),
           "n_same_pairs": int(same_mask.sum()),
           "n_diff_pairs": int(diff_mask.sum()),
           "R_raw": reuse_R(P, same_mask, diff_mask),
           "R_raw_by_type": reuse_by_type(P, motifs, same_mask, diff_mask, types),
           "per_k": {}}

    for k in k_grid:
        r = _row_for_k(ls, k)
        role = np.asarray(ls[r], dtype=int)
        actl = np.asarray(la[r], dtype=int)
        if np.unique(role).size < 2 or np.unique(actl).size < 2:
            continue
        Rc, Rc_rand, dRc, zc, pc = excess_reuse(P, role, same_mask, diff_mask,
                                                n_perm, rng)
        Ra, Ra_rand, dRa, za, pa = excess_reuse(P, actl, same_mask, diff_mask,
                                                n_perm, rng)
        res["per_k"][k] = {
            "k_role": int(np.unique(role).size), "k_act": int(np.unique(actl).size),
            "R_connectivity": Rc, "R_random_conn_sizes": Rc_rand,
            "delta_conn": dRc, "z_conn": zc, "p_conn": pc,
            "R_activity": Ra, "R_random_act_sizes": Ra_rand,
            "delta_act": dRa, "z_act": za, "p_act": pa,
            "delta_diff": dRc - dRa,
            "by_type_conn": reuse_by_type(project(P, role), motifs,
                                          same_mask, diff_mask, types),
        }
    return res


def signed_rank(vals):
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size < 3 or np.allclose(v, 0):
        return float("nan"), v.size, (float(np.median(v)) if v.size else float("nan"))
    from scipy.stats import wilcoxon
    _, p = wilcoxon(v)
    return float(p), v.size, float(np.median(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--k-grid", default=None,
                    help="comma-separated k values (default: the standard grid)")
    ap.add_argument("--out", default="reuse_random_partition_control.json")
    args = ap.parse_args()
    k_grid = ([int(x) for x in args.k_grid.split(",")] if args.k_grid else K_GRID)

    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    rng = np.random.default_rng(0)

    per_net = {}
    for s in seeds:
        net = f"LeakyRNN_softplus_diag_{args.size}_all_{s}"
        if not (FP_ROOT / net / "participation_cache_v2.npz").exists():
            print(f"[skip] {net}: no cache")
            continue
        r = run_net(net, k_grid, args.n_perm, rng)
        if r is None:
            print(f"[skip] {net}: too few motif pairs")
            continue
        per_net[net] = r
        print(f"[done] {net}  R_raw={r['R_raw']:.3f}  "
              f"({r['n_motifs']} motifs, {r['n_same_pairs']} same / "
              f"{r['n_diff_pairs']} diff cross-task pairs)")

    print(f"\n=== motif reuse: connectivity vs size-matched random vs activity "
          f"({len(per_net)} seeds) ===")
    print("R = same-type minus different-type cross-task correlation of "
          "partition-projected participation profiles\n")
    print(f"{'k':>4} {'R_conn':>8} {'R_rand':>8} {'R_act':>8} "
          f"{'d_conn':>8} {'d_act':>8} {'d_conn-d_act':>13} "
          f"{'conn>rand':>10} {'p(dconn)':>9} {'p(diff)':>9}")
    agg = {}
    for k in k_grid:
        rows = [pn["per_k"][k] for pn in per_net.values() if k in pn["per_k"]]
        if not rows:
            continue
        dc = [x["delta_conn"] for x in rows]
        da = [x["delta_act"] for x in rows]
        dd = [x["delta_diff"] for x in rows]
        p_dc, n, med_dc = signed_rank(dc)
        p_dd, _, med_dd = signed_rank(dd)
        med = lambda key: float(np.median([x[key] for x in rows]))  # noqa: E731
        frac = float(np.mean(np.array(dc) > 0))
        agg[k] = {"n_seeds": n,
                  "R_connectivity": med("R_connectivity"),
                  "R_random": med("R_random_conn_sizes"),
                  "R_activity": med("R_activity"),
                  "delta_conn": med_dc, "delta_act": float(np.median(da)),
                  "delta_diff": med_dd, "frac_conn_gt_rand": frac,
                  "p_delta_conn": p_dc, "p_delta_diff": p_dd,
                  "median_z_conn": med("z_conn"), "median_z_act": med("z_act")}
        a = agg[k]
        print(f"{k:>4} {a['R_connectivity']:>8.3f} {a['R_random']:>8.3f} "
              f"{a['R_activity']:>8.3f} {a['delta_conn']:>8.3f} "
              f"{a['delta_act']:>8.3f} {a['delta_diff']:>13.3f} "
              f"{frac*100:>9.0f}% {p_dc:>9.3g} {p_dd:>9.3g}")

    raws = [pn["R_raw"] for pn in per_net.values()]
    print(f"\nreference: R on RAW (unprojected) profiles, median = "
          f"{np.median(raws):.3f} over {len(raws)} seeds")

    out = FP_ROOT / args.out
    json.dump({"per_net": per_net, "aggregate": agg}, open(out, "w"),
              indent=2, default=float)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
