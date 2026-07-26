"""Held-out gain estimates: is the quotient prediction PROSPECTIVE?

In run_motif_quotient_prediction.py the group gains g_bar are read off the same
fixed points whose motif direction is being predicted. That is mathematically
valid but in-sample. Here the gains always come from data disjoint from the
motif direction being predicted, with the quotient Q_struct held fixed.

Four splits, increasingly demanding:

  interleaved  fixed points of a condition split odd/even; gains from one half,
               motif direction from the other. Controls for circularity only.
  arc          for ring/line-attractor conditions, fixed points are ordered by
               angle in the leading PC plane and split into CONTIGUOUS arcs:
               gains estimated on one arc, tangent direction predicted at
               held-out angles. This is extrapolation across the manifold, not
               interpolation.
  cross_task   gains from a condition of the SAME motif type in a DIFFERENT task
               (e.g. estimate on delaygo, predict delayanti). The motif-reuse
               claim, restated as a prediction rather than a correlation.
  cross_type   gains from a DIFFERENT motif type -- the negative control. If this
               works as well, the gains are not carrying task information.

Each is scored as A_cond = ||B'v_test|| / ||P_H v_test|| against (i) the
in-sample bound (gains from the test data itself) and (ii) size-matched random
partitions carrying the same held-out gains.

  conda run -n myenv python stepnet/run_motif_quotient_heldout.py --seeds 0-2
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
from motif_participation import WL_ROOT  # noqa: E402
from run_motif_reuse_control import MOTIF_TYPES, _row_for_k  # noqa: E402
from run_motif_quotient_prediction import (  # noqa: E402
    indicator, quotient, containment_norm, recover_alpha_gain,
)
from run_motif_quotient_controls import basis_from_Q  # noqa: E402

FP_ROOT = STEP_DIR.parent / "results" / "motif_fixed_points"
MAX_FP = 120
R_MODES = 3


# --------------------------------------------------------------------------- #
def per_point_cache(net, ind_active, W, motif_of, lamJ_of):
    """Per fixed point: gain vector and the motif eigenvector, per condition."""
    net_dir = FP_ROOT / net
    cache = net_dir / "perpoint_cache.npz"
    if cache.exists():
        d = npload(cache)
        if np.array_equal(np.asarray(d["ind_active"]), ind_active):
            return {k: {"g": g, "V": v, "ang": a} for k, g, v, a in
                    zip(d["keys"], d["gains"], d["vecs"], d["angles"])}
    keys, gains, vecs, angles = [], [], [], []
    for f in sorted(net_dir.glob("*.npz")):
        if "cache" in f.name or f.stem not in motif_of:
            continue
        d = npload(f)
        J = np.asarray(d["J_xstar"], dtype=np.float32)
        X = np.asarray(d["xstar"], dtype=float)[:, ind_active]
        if J.ndim != 3 or len(J) < 8:
            continue
        sel = np.linspace(0, len(J) - 1, min(MAX_FP, len(J))).astype(int)
        J = J[sel][:, ind_active][:, :, ind_active].astype(float)
        X = X[sel]
        lam_target = lamJ_of[f.stem]
        G, V = [], []
        for Ji in J:
            _, g = recover_alpha_gain(Ji, W)
            lam, EV = np.linalg.eig(Ji)
            j = int(np.argmin(np.abs(lam - lam_target)))
            u = np.real(EV[:, j])
            G.append(g)
            V.append(u / max(np.linalg.norm(u), 1e-12))
        # order along the manifold: angle in the leading PC plane of the states
        Xc = X - X.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        pc = Xc @ Vt[:2].T
        keys.append(f.stem)
        gains.append(np.array(G))
        vecs.append(np.array(V))
        angles.append(np.arctan2(pc[:, 1], pc[:, 0]))
        del J
    np.savez_compressed(cache, keys=np.array(keys),
                        gains=np.array(gains, dtype=object),
                        vecs=np.array(vecs, dtype=object),
                        angles=np.array(angles, dtype=object),
                        ind_active=ind_active)
    return {k: {"g": g, "V": v, "ang": a}
            for k, g, v, a in zip(keys, gains, vecs, angles)}


def dominant(V):
    """Sign-invariant dominant direction of a set of unit eigenvectors."""
    if len(V) == 0:
        return None
    _, _, Vt = np.linalg.svd(np.asarray(V, dtype=float), full_matrices=False)
    u = Vt[0]
    return u / np.linalg.norm(u)


def predict(v_test, g_train, W, labels, alpha):
    """A_cond of the top-R quotient modes built from held-out gains."""
    H, sizes = indicator(labels)
    Qs = quotient(W, H, sizes)
    gbar = (H.T @ g_train) / sizes
    Q = (1 - alpha) * np.eye(len(sizes)) + alpha * (gbar[:, None] * Qs)
    B, _ = basis_from_Q(Q, H, R_MODES)
    if B.shape[1] == 0:
        return np.nan
    PHv = containment_norm(v_test, H, sizes)
    return float(np.linalg.norm(B.T @ v_test) / PHv) if PHv > 0 else np.nan


# --------------------------------------------------------------------------- #
def run_net(net, k_list, n_perm, rng):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], dtype=float)
    ia = np.asarray(wl["ind_active"], dtype=int)

    cache = npload(FP_ROOT / net / "participation_cache_v2.npz")
    recs = [m for m in list(cache["records"]) if m["motif"] in MOTIF_TYPES]
    motif_of = {(m.get("key") or f"{m['rule']}__{m['epoch']}"): m["motif"]
                for m in recs}
    rule_of = {(m.get("key") or f"{m['rule']}__{m['epoch']}"): m["rule"]
               for m in recs}

    from run_motif_quotient_prediction import condition_jacobians, load_conditions
    jb = condition_jacobians(net, ia, W)
    conds = load_conditions(net, ia, W, jb)
    lamJ_of = {c["key"]: c["lam_J"] for c in conds}
    alpha = float(np.median([c["alpha"] for c in conds]))
    pp = per_point_cache(net, ia, W, motif_of, lamJ_of)

    ls = wl["labels_struct_by_k"]
    out = {}
    for k in k_list:
        role = np.asarray(ls[_row_for_k(ls, k)], dtype=int)
        rows = {s: [] for s in ("interleaved", "arc", "cross_task", "cross_type",
                                "in_sample")}
        rnd = {s: [] for s in rows}
        by_type = {}
        for key, d in pp.items():
            G, V, ang = d["g"], d["V"], d["ang"]
            n = len(V)
            if n < 8:
                continue
            # --- interleaved split ---
            a_idx, b_idx = np.arange(0, n, 2), np.arange(1, n, 2)
            v_test = dominant(V[b_idx])
            rows["in_sample"].append(predict(v_test, G[b_idx].mean(0), W, role, alpha))
            rows["interleaved"].append(
                predict(v_test, G[a_idx].mean(0), W, role, alpha))
            # --- contiguous arc split (extrapolation along the manifold) ---
            order = np.argsort(ang)
            half = n // 2
            tr, te = order[:half], order[half:]
            v_arc = dominant(V[te])
            rows["arc"].append(predict(v_arc, G[tr].mean(0), W, role, alpha))
            # --- cross-task / cross-type donors ---
            same = [q for q in pp if motif_of[q] == motif_of[key]
                    and rule_of[q] != rule_of[key]]
            other = [q for q in pp if motif_of[q] != motif_of[key]]
            v_full = dominant(V)
            g_same = g_other = None
            if same:
                g_same = pp[same[int(rng.integers(len(same)))]]["g"].mean(0)
                rows["cross_task"].append(predict(v_full, g_same, W, role, alpha))
            if other:
                g_other = pp[other[int(rng.integers(len(other)))]]["g"].mean(0)
                rows["cross_type"].append(predict(v_full, g_other, W, role, alpha))
            by_type.setdefault(motif_of[key], []).append(rows["arc"][-1])
            # --- size-matched random-partition null, same held-out gains ---
            lab = role.copy()
            acc = {s: [] for s in rows}
            for _ in range(n_perm):
                rng.shuffle(lab)
                acc["in_sample"].append(predict(v_test, G[b_idx].mean(0), W, lab, alpha))
                acc["interleaved"].append(predict(v_test, G[a_idx].mean(0), W, lab, alpha))
                acc["arc"].append(predict(v_arc, G[tr].mean(0), W, lab, alpha))
                if g_same is not None:
                    acc["cross_task"].append(predict(v_full, g_same, W, lab, alpha))
                if g_other is not None:
                    acc["cross_type"].append(predict(v_full, g_other, W, lab, alpha))
            for s in rows:
                if acc[s]:
                    rnd[s].append(float(np.nanmean(acc[s])))
        out[k] = {s: {"obs": float(np.nanmean(v)) if v else np.nan,
                      "rand": float(np.nanmean(rnd[s])) if rnd[s] else np.nan,
                      "n": len(v)}
                  for s, v in rows.items()}
        out[k]["arc_by_type"] = {mt: {"obs": float(np.nanmean(v)), "n": len(v)}
                                 for mt, v in by_type.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--k-list", default="12,20,30")
    ap.add_argument("--n-perm", type=int, default=30)
    ap.add_argument("--out", default="quotient_heldout.json")
    args = ap.parse_args()
    k_list = [int(x) for x in args.k_list.split(",")]
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
            continue
        per_net[net] = run_net(net, k_list, args.n_perm, rng)
        print(f"[done] {net}")

    print("\n=== C4  arc-split (extrapolation) by motif type ===")
    for k in k_list:
        for mt in ("line_attractor", "point_attractor", "saddle_decision"):
            v = [pn[k]["arc_by_type"][mt]["obs"] for pn in per_net.values()
                 if mt in pn[k]["arc_by_type"]]
            if v:
                print(f"  k={k:<3} {mt:18} A_cond={np.median(v):.3f} "
                      f"(n_seeds={len(v)})")

    print("\n=== C4  HELD-OUT GAINS: A_cond of top-3 quotient modes (r=3) ===")
    print(f"{'k':>4} {'split':13} {'conn':>7} {'rand':>7} {'delta':>7} "
          f"{'pos':>5} {'n cond':>7}")
    for k in k_list:
        for s in ("in_sample", "interleaved", "arc", "cross_task", "cross_type"):
            o = [pn[k][s]["obs"] for pn in per_net.values()
                 if np.isfinite(pn[k][s]["obs"])]
            r = [pn[k][s]["rand"] for pn in per_net.values()
                 if np.isfinite(pn[k][s]["rand"])]
            if not o:
                continue
            d = ([oo - rr for oo, rr in zip(o, r)] if r else [])
            print(f"{k:>4} {s:13} {np.median(o):>7.3f} "
                  f"{(np.median(r) if r else np.nan):>7.3f} "
                  f"{(np.median(d) if d else np.nan):>7.3f} "
                  f"{(str(sum(x > 0 for x in d)) + '/' + str(len(d)) if d else '-'):>5} "
                  f"{int(np.median([pn[k][s]['n'] for pn in per_net.values()])):>7}")
        print()

    json.dump(per_net, open(FP_ROOT / args.out, "w"), indent=2, default=float)
    print(f"wrote {FP_ROOT / args.out}")


if __name__ == "__main__":
    main()
