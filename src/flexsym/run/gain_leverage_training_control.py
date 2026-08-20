"""Training control for the two surviving gain-leverage results.

`gain_configuration` §2 found susceptibility concentrated on connectivity roles
more than on activity or random partitions, and §4 found gain changes allocating
leverage to near-critical modes ~1.9x more than a size-matched permutation.
Neither is worth anything until it survives the strength-preserving surrogate --
the control that demoted equitability to a degree artefact in claim 1.

Surrogates are the repo standard (`flexsym.run.training_control.make_matrix`):

  trained       the real recurrent matrix
  row-shuffle   permute within each row -> preserves each unit's input strength,
                destroys the pattern. THE control. If it reproduces the effect,
                the effect is a strength consequence.
  full-shuffle  permute all entries -> preserves the value distribution only
  gaussian      i.i.d. normal matched in Frobenius norm (an untrained network)

The partition is REFIT on every surrogate at the same k with `approx_wl_signed`,
exactly as `run.training_control` does. Holding the trained partition fixed while
shuffling W would be a rigged comparison -- that grouping was derived from the
trained matrix and is arbitrary for a surrogate.

Gains stay real throughout. They are a property of the trained network's fixed
points and have no surrogate counterpart, so per-UNIT gains are projected onto
each surrogate's own partition, `gbar = (H' g)/sizes`. §4 then asks a sharp
question: does the real gain change target the leverage structure of a shuffled
network as well as it targets the real one?

    conda run -n myenv python -m flexsym.run.gain_leverage_training_control --seeds 0-9
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import wilcoxon

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.gain_configuration import susceptibility
from flexsym.run.gain_leverage_geometry import (
    N_CRIT,
    colnorm,
    rotation_sensitivity,
)
from flexsym.run.motif_operator_composition import FAMILIES, key, med
from flexsym.run.motif_quotient_prediction import (
    condition_jacobians,
    indicator,
    load_conditions,
    quotient,
)
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.run.training_control import TYPES, make_matrix
from flexsym.structural.operating_k import OPK, model_name
from flexsym.structural.signed_input_wl_structural_roles import approx_wl_signed

FP = paths.RESULTS_ROOT / "motif_fixed_points"
N_SURR = 200
TOL = 1e-12


def fit_partition(M, k):
    Wp, Wn = np.maximum(M, 0), np.maximum(-M, 0)
    labels, _ = approx_wl_signed(Wp, Wn, k, n_iter=20, seed=0, n_init=20)
    return np.asarray(labels, int)


def participation_ratio(x):
    p = np.abs(x)
    s = p.sum()
    if s <= TOL:
        return np.nan
    p = p / s
    return float(1.0 / (p ** 2).sum())


def allocation(Sm, dghat, idx, sizes, rng, n_surr):
    """Leverage on a class relative to the bulk, real vs size-stratified perm."""
    bins = np.digitize(sizes, np.quantile(sizes, [0.25, 0.5, 0.75]))
    blocks = [np.where(bins == b)[0] for b in np.unique(bins)]

    def resp(S, cols, x):
        if not cols:
            return np.nan
        return float(np.linalg.norm(S[:, cols].T @ x) / np.sqrt(len(cols)))

    surr = []
    for _ in range(n_surr):
        x = dghat.copy()
        for blk in blocks:
            x[blk] = x[rng.permutation(blk)]
        surr.append(x)

    out = {}
    base_real = resp(Sm, idx["other"], dghat)
    base_surr = np.array([resp(Sm, idx["other"], x) for x in surr])
    for cname in ("star", "critical"):
        cols = idx[cname]
        if not cols or not np.isfinite(base_real) or base_real <= 0:
            continue
        real = resp(Sm, cols, dghat) / base_real
        sv = np.array([resp(Sm, cols, x) for x in surr])
        ok = np.isfinite(sv) & np.isfinite(base_surr) & (base_surr > 0)
        out[cname] = {"real": real,
                      "permsize": float(np.mean(sv[ok] / base_surr[ok])) if ok.any()
                      else np.nan}
    return out


def run_net(net, k, rng, n_surr=N_SURR):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], float)
    ind = np.asarray(wl["ind_active"], int)
    jbars = condition_jacobians(net, ind, W)
    conds = load_conditions(net, ind, W, jbars)
    if len(conds) < 8:
        return None
    alpha = float(np.median([c["alpha"] for c in conds]))
    byk = {c["key"]: c for c in conds}
    present = set(byk)
    ls = wl["labels_struct_by_k"]
    stored = np.asarray(ls[_row_for_k(ls, k)], int)

    out = {"k": k, "n_cond": len(conds), "alpha": alpha, "by_type": {}}
    mrng = np.random.default_rng(int(net.split("_")[-1]))
    for kind in TYPES:
        M = make_matrix(W, kind, mrng)
        labels = fit_partition(M, k)
        H, sizes = indicator(labels)
        if len(sizes) < 4:
            continue
        Qs = quotient(M, H, sizes)
        gref = np.median(np.array([(H.T @ c["g"]) / sizes for c in conds]), axis=0)
        Ssens, lam, _ = susceptibility(Qs, gref, alpha)
        order = np.argsort(-np.abs(lam))[:10]
        pr = float(np.nanmedian([participation_ratio(Ssens[:, j]) for j in order]))

        rows = []
        for prim, epoch, pairs in FAMILIES:
            for a, b in pairs:
                ka, kb = key(a, epoch), key(b, epoch)
                if ka not in present or kb not in present:
                    continue
                ga = (H.T @ byk[ka]["g"]) / sizes
                gb = (H.T @ byk[kb]["g"]) / sizes
                dg = gb - ga
                nd = np.linalg.norm(dg)
                if nd <= TOL:
                    continue
                dghat = dg / nd
                Sev, _, _ = susceptibility(Qs, ga, alpha)
                Rot, _, lam2, Va = rotation_sensitivity(Qs, ga, alpha)
                Slift = H / np.sqrt(sizes)[None, :]
                Mv = Slift @ np.real(Va)
                Mv /= np.maximum(np.linalg.norm(Mv, axis=0), TOL)
                jstar = int(np.argmax((Mv.T @ byk[kb]["v"]) ** 2))
                crit = list(np.argsort(np.abs(np.abs(lam2) - 1.0))[:N_CRIT])
                idx = {"star": [jstar],
                       "critical": [j for j in crit if j != jstar],
                       "other": [j for j in range(len(sizes))
                                 if j != jstar and j not in crit]}
                rec = {"family": f"{prim}@{epoch}"}
                for tname, Sm in (("ev", colnorm(Sev)), ("rot", colnorm(Rot))):
                    al = allocation(Sm, dghat, idx, sizes, rng, n_surr)
                    for cname, vals in al.items():
                        rec[f"{tname}_{cname}"] = vals
                rows.append(rec)

        out["by_type"][kind] = {"pr": pr, "k_fit": int(len(sizes)),
                                "contrasts": rows}
        if kind == "trained":
            from sklearn.metrics import adjusted_rand_score
            out["ari_refit_vs_stored"] = float(
                adjusted_rand_score(labels, stored))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--n-surr", type=int, default=N_SURR)
    ap.add_argument("--out", default="gain_leverage_training_control.json")
    args = ap.parse_args()
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
        net = model_name(s)
        if not (FP / net / "participation_cache_v2.npz").exists():
            print(f"[skip] {net}")
            continue
        r = run_net(net, OPK[s], rng, args.n_surr)
        if r:
            per_net[net] = r
            print(f"[done] seed {s} k={r['k']} "
                  + " ".join(f"{t[:4]}PR={r['by_type'][t]['pr']:.1f}"
                             for t in TYPES if t in r["by_type"])
                  + f"  ARI(refit,stored)={r.get('ari_refit_vs_stored', float('nan')):.2f}",
                  flush=True)
    if not per_net:
        return
    V = list(per_net.values())

    def wilc(a, b):
        p = [(x, y) for x, y in zip(a, b)
             if np.isfinite(x) and np.isfinite(y) and x != y]
        if len(p) < 5:
            return np.nan
        return float(wilcoxon([q[0] for q in p], [q[1] for q in p]).pvalue)

    print(f"\n  refit vs stored partition, median ARI = "
          f"{med([v.get('ari_refit_vs_stored', np.nan) for v in V]):.3f}"
          f"  (restart degeneracy is known: ARI ~0.47 at 256)")

    print("\n" + "=" * 76)
    print("SECTION 2 CONTROL -- susceptibility concentration (PR over roles)")
    print("  lower PR = leverage carried by fewer roles")
    print("=" * 76)
    print(f"  {'matrix':14s} {'PR':>7} {'PR/k':>7} {'vs trained':>12} {'p':>8}")
    tr = [v["by_type"]["trained"]["pr"] for v in V if "trained" in v["by_type"]]
    for kind in TYPES:
        pr = [v["by_type"][kind]["pr"] for v in V if kind in v["by_type"]]
        kf = [v["by_type"][kind]["k_fit"] for v in V if kind in v["by_type"]]
        if kind == "trained":
            print(f"  {kind:14s} {med(pr):>7.2f} "
                  f"{med([a / b for a, b in zip(pr, kf)]):>7.3f} {'--':>12} {'--':>8}")
        else:
            print(f"  {kind:14s} {med(pr):>7.2f} "
                  f"{med([a / b for a, b in zip(pr, kf)]):>7.3f} "
                  f"{med([a - b for a, b in zip(pr, tr)]):>+12.2f} "
                  f"{wilc(tr, pr):>8.4f}   trained lower in "
                  f"{sum(a < b for a, b in zip(tr, pr))}/{len(tr)}")

    print("\n" + "=" * 76)
    print("SECTION 4 CONTROL -- relative allocation (class leverage / bulk leverage)")
    print("  real vs size-stratified permutation, on each matrix's own partition")
    print("=" * 76)
    for tname, lname in (("ev", "eigenvalue"), ("rot", "rotation")):
        for cname in ("star", "critical"):
            print(f"\n  {lname} / {cname}")
            print(f"    {'matrix':14s} {'real':>7} {'permsize':>9} {'ratio':>7} "
                  f"{'real>ps':>8} {'p':>8}")
            ratios = {}
            for kind in TYPES:
                R, P = [], []
                for v in V:
                    if kind not in v["by_type"]:
                        continue
                    rr = [c[f"{tname}_{cname}"]["real"]
                          for c in v["by_type"][kind]["contrasts"]
                          if f"{tname}_{cname}" in c]
                    pp = [c[f"{tname}_{cname}"]["permsize"]
                          for c in v["by_type"][kind]["contrasts"]
                          if f"{tname}_{cname}" in c]
                    R.append(med(rr) if rr else np.nan)
                    P.append(med(pp) if pp else np.nan)
                ratios[kind] = [a / b for a, b in zip(R, P)
                                if np.isfinite(a) and np.isfinite(b) and b > 0]
                n = sum(a > b for a, b in zip(R, P)
                        if np.isfinite(a) and np.isfinite(b))
                print(f"    {kind:14s} {med(R):>7.3f} {med(P):>9.3f} "
                      f"{med(R) / med(P):>7.3f} {n:>5}/{len(R)} {wilc(R, P):>8.4f}")
            if "trained" in ratios and "row-shuffle" in ratios:
                a, b = ratios["trained"], ratios["row-shuffle"]
                print(f"    trained vs row-shuffle ratio: "
                      f"{med(a):.3f} vs {med(b):.3f}   p={wilc(a, b):.4f}"
                      f"   trained higher in {sum(x > y for x, y in zip(a, b))}/{len(a)}")

    out = FP / args.out
    out.write_text(json.dumps(per_net, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
