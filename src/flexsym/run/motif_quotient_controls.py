"""Controls for the quotient-predicts-motif-direction result.

C2  COMPRESSION COST. What does the quotient give up relative to predictors that
    use the full N-dimensional operator? Compares top-r modes of: the quotient
    (k x k), the full mean Jacobian, W_rec, the W_rec singular basis, and r
    random directions inside the role subspace. Reported with predictor
    dimension and free-parameter count, so the comparison is about compression.

C3  WHAT DOES THE WORK -- connectivity or gains? Ablations of
    Q_pred = (1-a) I + a diag(g_bar) Q_struct :
      no_gain     (1-a) I + a Q_struct                 connectivity only
      scalar_gain (1-a) I + a g_global Q_struct        one global gain
      gain_only   (1-a) I + a diag(g_bar)              gains, no quotient structure
      perm_gain   group gains permuted across nodes    same gain distribution
      cross_gain  gains from a DIFFERENT condition     mismatched task/epoch
    If Q_struct alone already fixes the modes, gains are decoration. If gain_only
    works, the quotient is decoration. If perm_gain/cross_gain break it, the
    specific matching of gains to structural roles is what carries the motif.

C5  MODE-NUMBER CURVES. A_raw(r) and A_cond(r) for r = 1..10, connectivity vs
    size-matched random partitions -- saturation rate, not a single r.

C6  MOTIF-APPROPRIATE MODE SELECTION. Selecting quotient modes by top |lambda|
    is generic. Dynamically motivated rules: line/point attractor -> modes
    closest to lambda = 1; saddle -> most unstable (largest Re lambda). Applied
    per motif class and compared to the generic rule.

  conda run -n myenv python stepnet/run_motif_quotient_controls.py --seeds 0-2
"""

from __future__ import annotations

from flexsym import paths

import argparse
import json
import sys
from pathlib import Path

import numpy as np


from flexsym.motifs.motif_classify import npload  # noqa: E402
from flexsym.motifs.motif_participation import WL_ROOT  # noqa: E402
from flexsym.run.motif_reuse_control import _row_for_k  # noqa: E402
from flexsym.run.motif_quotient_prediction import (  # noqa: E402
    condition_jacobians, load_conditions, indicator, quotient,
    containment_norm,
)

FP_ROOT = paths.RESULTS_ROOT / "motif_fixed_points"
R_MAX = 10
N_PERM = 60


# --------------------------------------------------------------------------- #
# mode selection + scoring
# --------------------------------------------------------------------------- #
def select_modes(lam, rule, r, lam_J=None):
    if rule == "top":
        key = -np.abs(lam)
    elif rule == "unit":
        key = np.abs(lam - 1.0)
    elif rule == "unstable":
        key = -np.real(lam)
    elif rule == "match":                       # closest to the motif's own lambda
        key = np.abs(lam - lam_J)
    else:
        raise ValueError(rule)
    return np.argsort(key)[:r]


def basis_from_Q(Q, H, r, rule="top", lam_J=None, eig=None):
    lam, V = np.linalg.eig(Q) if eig is None else eig
    idx = select_modes(lam, rule, r, lam_J)
    U = H @ np.real(V[:, idx])
    if np.any(np.abs(np.imag(V[:, idx])) > 1e-12):
        U = np.hstack([U, H @ np.imag(V[:, idx])])
    B, R = np.linalg.qr(U)
    return B[:, np.abs(np.diag(R)) > 1e-10], lam[idx]


def basis_from_full(M, r, rule="top", lam_J=None):
    """Top-r modes of a full N x N operator (no partition)."""
    lam, V = np.linalg.eig(M)
    idx = select_modes(lam, rule, r, lam_J)
    U = np.real(V[:, idx])
    if np.any(np.abs(np.imag(V[:, idx])) > 1e-12):
        U = np.hstack([U, np.imag(V[:, idx])])
    B, R = np.linalg.qr(U)
    return B[:, np.abs(np.diag(R)) > 1e-10]


def align(v, B):
    if B is None or B.shape[1] == 0:
        return np.nan
    return float(np.linalg.norm(B.T @ v) / np.linalg.norm(v))


# --------------------------------------------------------------------------- #
# predictor family
# --------------------------------------------------------------------------- #
def build_predictors(c, H, sizes, Qs, alpha, rng, donor_g=None):
    """dict name -> (k x k quotient) for the ablation family (C3)."""
    kk = len(sizes)
    gbar = (H.T @ c["g"]) / sizes
    I = np.eye(kk)
    P = {
        "Q_pred": (1 - alpha) * I + alpha * (gbar[:, None] * Qs),
        "no_gain": (1 - alpha) * I + alpha * Qs,
        "scalar_gain": (1 - alpha) * I + alpha * float(np.mean(c["g"])) * Qs,
        "gain_only": (1 - alpha) * I + alpha * np.diag(gbar),
        "perm_gain": (1 - alpha) * I + alpha * (rng.permutation(gbar)[:, None] * Qs),
    }
    if donor_g is not None:
        gd = (H.T @ donor_g) / sizes
        P["cross_gain"] = (1 - alpha) * I + alpha * (gd[:, None] * Qs)
    return P


def eval_all(labels, W, conds, rng, r_list, donors):
    """Everything partition-dependent, for one label vector."""
    H, sizes = indicator(labels)
    Qs = quotient(W, H, sizes)
    kk = len(sizes)
    out = {}
    acc = {}

    def push(name, a_raw, a_cond):
        acc.setdefault(name, ([], []))
        acc[name][0].append(a_raw)
        acc[name][1].append(a_cond)

    for ci, c in enumerate(conds):
        v, alpha = c["v"], c["alpha"]
        PHv = containment_norm(v, H, sizes)
        nv = np.linalg.norm(v)
        Ps = build_predictors(c, H, sizes, Qs, alpha, rng,
                              donor_g=donors[ci]["g"] if donors[ci] else None)
        eigs = {name: np.linalg.eig(Q) for name, Q in Ps.items()}
        for name, Q in Ps.items():
            for r in r_list:
                if r > kk:
                    continue
                B, _ = basis_from_Q(Q, H, r, eig=eigs[name])
                pr = np.linalg.norm(B.T @ v) if B.shape[1] else np.nan
                push(f"{name}|r{r}", pr / nv, pr / PHv if PHv > 0 else np.nan)
                if name == "Q_pred":       # generic rule, per motif class (C6)
                    push(f"Q_pred|{c['motif']}|r{r}", pr / nv,
                         pr / PHv if PHv > 0 else np.nan)
        # random directions inside the role subspace (C2 baseline)
        for r in r_list:
            if r > kk:
                continue
            Z = H @ rng.standard_normal((kk, r))
            B, _ = np.linalg.qr(Z)
            pr = np.linalg.norm(B.T @ v)
            push(f"rand_dir|r{r}", pr / nv, pr / PHv if PHv > 0 else np.nan)
        # motif-appropriate mode selection (C6)
        rule = {"line_attractor": "unit", "point_attractor": "unit",
                "saddle_decision": "unstable"}.get(c["motif"], "top")
        lamP = eigs["Q_pred"][0]
        for r in r_list:
            if r > kk:
                continue
            for tag, rl in (("sel_motif", rule), ("sel_match", "match")):
                B, _ = basis_from_Q(Ps["Q_pred"], H, r, rule=rl,
                                    lam_J=c["lam_J"], eig=eigs["Q_pred"])
                pr = np.linalg.norm(B.T @ v) if B.shape[1] else np.nan
                push(f"{tag}|r{r}", pr / nv, pr / PHv if PHv > 0 else np.nan)
                push(f"{tag}|{c['motif']}|r{r}", pr / nv,
                     pr / PHv if PHv > 0 else np.nan)
            # do the selection rules actually pick different modes?
            same = float(len(set(select_modes(lamP, "top", r)) &
                             set(select_modes(lamP, rule, r, c["lam_J"]))) / r)
            push(f"rule_overlap|{c['motif']}|r{r}", same, same)
        out.setdefault("containment", []).append(PHv / nv)

    res = {k: {"raw": float(np.nanmean(a)), "cond": float(np.nanmean(b))}
           for k, (a, b) in acc.items()}
    res["containment"] = {"raw": float(np.nanmean(out["containment"])),
                          "cond": np.nan}
    return res


def full_space_predictors(conds, W, r_list):
    """C2: predictors that use the whole N x N operator (partition-free)."""
    acc = {}
    _, _, Vt = np.linalg.svd(W)
    for c in conds:
        v = c["v"]
        for r in r_list:
            for name, B in (
                ("full_J", basis_from_full(c["Jb"], r)),
                ("full_W", basis_from_full(W, r)),
                ("W_svd", np.linalg.qr(Vt[:r].T)[0]),
                ("full_J_match", basis_from_full(c["Jb"], r, "match", c["lam_J"])),
            ):
                acc.setdefault(f"{name}|r{r}", []).append(align(v, B))
    return {k: float(np.nanmean(vv)) for k, vv in acc.items()}


# --------------------------------------------------------------------------- #
def pick_donors(conds, rng):
    """For each condition, a donor condition of the SAME motif type in a
    DIFFERENT task (cross-task held-out gains); None if unavailable."""
    donors = []
    for c in conds:
        cand = [d for d in conds
                if d["motif"] == c["motif"] and d["rule"] != c["rule"]]
        donors.append(cand[int(rng.integers(len(cand)))] if cand else None)
    return donors


def run_net(net, k_list, n_perm, rng):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], dtype=float)
    ia = np.asarray(wl["ind_active"], dtype=int)
    jb = condition_jacobians(net, ia, W)
    conds = load_conditions(net, ia, W, jb)
    if len(conds) < 4:
        return None
    donors = pick_donors(conds, rng)
    r_list = list(range(1, R_MAX + 1))
    ls = wl["labels_struct_by_k"]
    la = wl["labels_act_by_k"]
    out = {"n_cond": len(conds),
           "full_space": full_space_predictors(conds, W, r_list),
           "per_k": {}}
    for k in k_list:
        r = _row_for_k(ls, k)
        role = np.asarray(ls[r], dtype=int)
        actl = np.asarray(la[r], dtype=int)
        obs = eval_all(role, W, conds, rng, r_list, donors)
        act = eval_all(actl, W, conds, rng, r_list, donors)
        lab = role.copy()
        null = {}
        for _ in range(n_perm):
            rng.shuffle(lab)
            nv = eval_all(lab, W, conds, rng, r_list, donors)
            for kk, vv in nv.items():
                null.setdefault(kk, {"raw": [], "cond": []})
                null[kk]["raw"].append(vv["raw"])
                null[kk]["cond"].append(vv["cond"])
        rec = {}
        for kk in obs:
            e = {}
            for fld in ("raw", "cond"):
                arr = np.array(null[kk][fld], dtype=float)
                mu, sd = np.nanmean(arr), np.nanstd(arr)
                e[fld] = {"obs": obs[kk][fld], "rand": float(mu),
                          "act": act[kk][fld],
                          "delta": obs[kk][fld] - float(mu),
                          "z": float((obs[kk][fld] - mu) / sd) if sd > 0 else 0.0}
            rec[kk] = e
        out["per_k"][k] = rec
    return out


def med(per_net, k, key, fld, which):
    v = [pn["per_k"][k][key][fld][which] for pn in per_net.values()
         if k in pn["per_k"] and key in pn["per_k"][k]]
    return float(np.median(v)) if v else np.nan


def npos(per_net, k, key, fld):
    v = [pn["per_k"][k][key][fld]["delta"] for pn in per_net.values()
         if k in pn["per_k"] and key in pn["per_k"][k]]
    return f"{sum(x > 0 for x in v)}/{len(v)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--k-list", default="12,20,30")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--out", default="quotient_controls.json")
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
        r = run_net(net, k_list, args.n_perm, rng)
        if r:
            per_net[net] = r
            print(f"[done] {net} ({r['n_cond']} conditions)")

    K0 = k_list[len(k_list) // 2]

    # ---------------- C2 ----------------
    print(f"\n=== C2  COMPRESSION: alignment A_raw with the full motif direction "
          f"(k={K0}) ===")
    print(f"  {'predictor':22} {'dims':>5} {'free params':>12} "
          f"{'r=1':>7} {'r=3':>7} {'r=5':>7} {'r=10':>7}")
    N = 204
    rows = [("quotient Q_pred (role)", f"{K0}^2+{K0}", "Q_pred", K0 * K0 + K0),
            ("random dirs in col(H)", "r*N", "rand_dir", None),
            ("full mean Jacobian", "N^2", "full_J", N * N),
            ("full W_rec", "N^2", "full_W", N * N),
            ("W_rec SVD basis", "N^2", "W_svd", N * N)]
    for label, pstr, key, pnum in rows:
        vals = []
        for r in (1, 3, 5, 10):
            if key in ("full_J", "full_W", "W_svd"):
                v = np.median([pn["full_space"][f"{key}|r{r}"]
                               for pn in per_net.values()])
            else:
                v = med(per_net, K0, f"{key}|r{r}", "raw", "obs")
            vals.append(v)
        pp = f"{pnum}" if pnum else pstr
        print(f"  {label:22} {'r':>5} {pp:>12} " +
              " ".join(f"{x:>7.3f}" for x in vals))
    print(f"  {'(containment ceiling)':22} {'-':>5} {'-':>12} " +
          " ".join(f"{med(per_net, K0, 'containment', 'raw', 'obs'):>7.3f}"
                   for _ in range(4)))

    # ---------------- C3 ----------------
    print(f"\n=== C3  ABLATIONS of Q_pred  (A_cond, r=3) ===")
    print(f"  {'k':>4} {'variant':14} {'conn':>7} {'rand':>7} {'act':>7} "
          f"{'delta':>7} {'z':>6} {'pos':>5}")
    for k in k_list:
        for name in ("Q_pred", "no_gain", "scalar_gain", "gain_only",
                     "perm_gain", "cross_gain"):
            key = f"{name}|r3"
            if np.isnan(med(per_net, k, key, "cond", "obs")):
                continue
            print(f"  {k:>4} {name:14} "
                  f"{med(per_net, k, key, 'cond', 'obs'):>7.3f} "
                  f"{med(per_net, k, key, 'cond', 'rand'):>7.3f} "
                  f"{med(per_net, k, key, 'cond', 'act'):>7.3f} "
                  f"{med(per_net, k, key, 'cond', 'delta'):>7.3f} "
                  f"{med(per_net, k, key, 'cond', 'z'):>6.2f} "
                  f"{npos(per_net, k, key, 'cond'):>5}")
        print()

    # ---------------- C5 ----------------
    print(f"=== C5  MODE-NUMBER CURVES (k={K0}) ===")
    print(f"  {'r':>3} {'A_raw conn':>11} {'A_raw rand':>11} "
          f"{'A_cond conn':>12} {'A_cond rand':>12} {'A_cond act':>11} {'z':>6}")
    for r in range(1, R_MAX + 1):
        key = f"Q_pred|r{r}"
        print(f"  {r:>3} {med(per_net, K0, key, 'raw', 'obs'):>11.3f} "
              f"{med(per_net, K0, key, 'raw', 'rand'):>11.3f} "
              f"{med(per_net, K0, key, 'cond', 'obs'):>12.3f} "
              f"{med(per_net, K0, key, 'cond', 'rand'):>12.3f} "
              f"{med(per_net, K0, key, 'cond', 'act'):>11.3f} "
              f"{med(per_net, K0, key, 'cond', 'z'):>6.2f}")

    # ---------------- C6 ----------------
    print(f"\n=== C6  MODE SELECTION RULE, by motif class (A_cond, k={K0}) ===")
    print(f"  {'motif':18} {'r':>3} {'top|lam|':>9} {'dynamical':>10} "
          f"{'match lam_J':>12} {'rule overlap':>13}")
    for mt in ("line_attractor", "point_attractor", "saddle_decision"):
        for r in (1, 2, 3, 5):
            print(f"  {mt:18} {r:>3} "
                  f"{med(per_net, K0, f'Q_pred|{mt}|r{r}', 'cond', 'obs'):>9.3f} "
                  f"{med(per_net, K0, f'sel_motif|{mt}|r{r}', 'cond', 'obs'):>10.3f} "
                  f"{med(per_net, K0, f'sel_match|{mt}|r{r}', 'cond', 'obs'):>12.3f} "
                  f"{med(per_net, K0, f'rule_overlap|{mt}|r{r}', 'raw', 'obs'):>13.2f}")

    json.dump(per_net, open(FP_ROOT / args.out, "w"), indent=2, default=float)
    print(f"\nwrote {FP_ROOT / args.out}")


if __name__ == "__main__":
    main()
