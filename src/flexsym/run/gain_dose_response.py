"""A priori: rank roles by leverage over the leading modes of W_rec, THEN break the tasks.

Two things separate this from gain_behavioural_specificity:

1. The ranking uses W_rec ALONE. The quotient Q_struct = (H'H)^-1 H' W H is
   task-free and gain-free; leverage is d|lambda_j|/dg_i evaluated at g = 1,
   alpha = 1, i.e. of diag(g) Q_struct. No fixed points, no stored Jacobian, no
   condition. It is computed and WRITTEN TO DISK for every network before a
   single trial is run (--prereg-only / the *_prereg.json), so the ordering is
   frozen in advance rather than chosen after seeing the deficits.

2. The behavioural readout is a dose-response threshold, not a deficit at one
   gain. Sweeping gamma per role gives the critical gamma at which the network
   stops doing the tasks. A connectivity strength statistic can order roles;
   only a susceptibility says WHERE on the gamma axis each role collapses.

Baselines, all from the same W_rec, because "high-leverage roles matter" must
beat "well-connected roles matter":
    size, mean in-strength, mean out-strength, and leading-eigenvector
    participation |u1_i v1_i| (summed and per-unit). To first order
    d|lambda_1|/dg_i IS a left x right participation product, so this last one is
    the baseline that matters. Reported as raw rank correlations and as the
    PARTIAL correlation of leverage with the deficit after the baselines are
    regressed out on ranks.

Sign. The leverage is signed: raising the gain of a role with positive
d|lambda|/dg pushes the leading mode toward instability, lowering it pulls back.
Strength statistics are sign-blind by construction, so the asymmetry between
+gamma and -gamma at matched |gamma| is a prediction only the susceptibility
makes. Tested as spearman(signed leverage, asymmetry) and as a two-group
comparison.

    conda run -n myenv python -m flexsym.run.gain_dose_response --seeds 3-12
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon

from flexsym import paths
from flexsym.core.np_sim import NpNet, make_trials, perf_by_rule
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.gain_leverage_geometry import rotation_sensitivity
from flexsym.run.motif_quotient_prediction import indicator, quotient
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.structural.operating_k import OPK

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

OUT = paths.RESULTS_ROOT / "motif_fixed_points"
GAMMA_POS = (0.15, 0.3, 0.5, 0.75, 1.1, 1.5)
GAMMA_NEG = (-0.3, -0.5, -0.75)
THRESH = 0.75           # mean-over-tasks performance defining "stops doing the tasks"
BASELINES = ("size", "in_mean", "out_mean", "part_sum", "part_mean")


# --------------------------------------------------------------------------- #
# phase A -- the frozen ranking, from W_rec alone
# --------------------------------------------------------------------------- #
def leverage_from_wrec(W, role, m=3):
    """Leverage of each role over the leading modes of Q_struct. No gains, no task."""
    H, sizes = indicator(role)
    Qs = quotient(W, H, sizes)
    k = H.shape[1]
    dLam, lam = rotation_sensitivity(Qs, np.ones(k), 1.0)[1:3]
    order = np.argsort(-np.abs(lam))
    j = int(order[0])
    lev = np.abs(dLam[:, j])                        # leading structural mode
    lev_top = np.abs(dLam[:, order[:m]]).sum(1)     # top-m modes
    return {"lev": lev, "lev_signed": dLam[:, j], "lev_top": lev_top,
            "lam_lead": complex(lam[j]), "j": j, "sizes": sizes.astype(int),
            "Qs": Qs}


def baselines_from_wrec(W, role, k):
    """Connectivity statistics on the same matrix, for the same roles."""
    lam, V = np.linalg.eig(W)
    lamL, U = np.linalg.eig(W.T)
    i1 = int(np.argmax(np.abs(lam)))
    j1 = int(np.argmin(np.abs(lamL - lam[i1])))
    part = np.abs(np.real(U[:, j1] * V[:, i1]))     # left x right participation
    ins = np.abs(W).sum(1)                          # into unit i
    outs = np.abs(W).sum(0)                         # out of unit i
    B = {}
    for name, v, how in (("size", np.ones(len(role)), "sum"),
                         ("in_mean", ins, "mean"), ("out_mean", outs, "mean"),
                         ("part_sum", part, "sum"), ("part_mean", part, "mean")):
        f = np.sum if how == "sum" else np.mean
        B[name] = np.array([f(v[role == c]) for c in range(k)])
    return B


def prereg_net(seed, k=None):
    net = f"LeakyRNN_softplus_diag_256_all_{seed}"
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], float)
    ind = np.asarray(wl["ind_active"], int)
    k = int(OPK[seed]) if k is None else int(k)
    role = np.asarray(wl["labels_struct_by_k"][_row_for_k(wl["labels_struct_by_k"], k)],
                      int)
    kk = int(np.unique(role).size)
    L = leverage_from_wrec(W, role)
    B = baselines_from_wrec(W, role, kk)
    return {"seed": seed, "net": net, "k": kk, "role": role, "ind": ind,
            "lev": L["lev"], "lev_signed": L["lev_signed"], "lev_top": L["lev_top"],
            "sizes": L["sizes"], "lam_lead": abs(L["lam_lead"]), "base": B,
            "rank_by_lev": np.argsort(-L["lev"]).tolist()}


def prereg_json(P):
    return {"net": P["net"], "seed": P["seed"], "k": P["k"],
            "lam_lead": float(P["lam_lead"]),
            "leverage": P["lev"].tolist(), "leverage_signed": P["lev_signed"].tolist(),
            "leverage_top3": P["lev_top"].tolist(), "sizes": P["sizes"].tolist(),
            "ranking_high_to_low": P["rank_by_lev"],
            "top_role": int(P["rank_by_lev"][0]),
            "baselines": {b: P["base"][b].tolist() for b in BASELINES}}


# --------------------------------------------------------------------------- #
# phase B -- the dose-response sweep
# --------------------------------------------------------------------------- #
def sweep(net, trials, rules, units_by_role, gammas, noise_seed=0, verbose=True):
    """perf[g, r, t] over the gamma grid, plus the intact row."""
    k = len(units_by_role)
    p0 = perf_by_rule(net, trials, seed=noise_seed)
    perf0 = np.array([p0[t] for t in rules])
    P = np.zeros((len(gammas), k, len(rules)))
    t0 = time.time()
    for gi, gam in enumerate(gammas):
        for r, units in enumerate(units_by_role):
            p = perf_by_rule(net, trials, seed=noise_seed,
                             gain=net.gain_vector(units, gam))
            P[gi, r] = [p[t] for t in rules]
        if verbose:
            print(f"    gamma {gam:+.2f} done  ({time.time()-t0:.0f}s)", flush=True)
    return P, perf0


def critical_gamma(curve, gammas, thresh=THRESH):
    """Smallest gamma whose mean-over-tasks performance falls below `thresh`.

    Linear interpolation between grid points; censored (returned as 1.5 * max
    gamma, flagged) when the network never breaks over the grid.
    """
    gmax = max(gammas)
    below = np.where(curve < thresh)[0]
    if not len(below):
        return 1.5 * gmax, True
    i = int(below[0])
    if i == 0:
        return float(gammas[0]), False
    y0, y1 = curve[i - 1], curve[i]
    x0, x1 = gammas[i - 1], gammas[i]
    f = (y0 - thresh) / max(y0 - y1, 1e-12)
    return float(x0 + f * (x1 - x0)), False


def readouts(P, perf0, gammas, thresh=THRESH):
    """Per role: critical gamma, integrated deficit, +/- asymmetry."""
    pos = [i for i, g in enumerate(gammas) if g > 0]
    neg = [i for i, g in enumerate(gammas) if g < 0]
    k = P.shape[1]
    gc, cens, auc = np.zeros(k), np.zeros(k, bool), np.zeros(k)
    for r in range(k):
        curve = P[pos, r].mean(-1)
        gc[r], cens[r] = critical_gamma(curve, [gammas[i] for i in pos], thresh)
        auc[r] = float((perf0.mean() - curve).mean())
    asym = np.full(k, np.nan)
    if neg:
        matched = [(i, j) for i in pos for j in neg
                   if abs(gammas[i] + gammas[j]) < 1e-9]
        if matched:
            up = np.array([perf0.mean() - P[i, :, :].mean(-1) for i, _ in matched])
            dn = np.array([perf0.mean() - P[j, :, :].mean(-1) for _, j in matched])
            asym = (up - dn).mean(0)
    return {"gc": gc, "censored": cens, "auc": auc, "asym": asym}


# --------------------------------------------------------------------------- #
# phase C -- statistics
# --------------------------------------------------------------------------- #
def partial_spearman(x, y, Z):
    """Spearman of x with y after regressing both on the baselines Z, on ranks."""
    rx, ry = rankdata(x), rankdata(y)
    RZ = np.column_stack([rankdata(z) for z in Z] + [np.ones(len(x))])
    ex = rx - RZ @ np.linalg.lstsq(RZ, rx, rcond=None)[0]
    ey = ry - RZ @ np.linalg.lstsq(RZ, ry, rcond=None)[0]
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return np.nan
    return float(spearmanr(ex, ey)[0])


def net_stats(P, R, pre, gammas):
    lev, lev_s = pre["lev"], pre["lev_signed"]
    base = pre["base"]
    Z = [base[b] for b in BASELINES]
    out = {"k": pre["k"], "n_censored": int(R["censored"].sum()),
           "gc_median": float(np.median(R["gc"])), "auc_median": float(np.median(R["auc"]))}
    # graded predictions: high leverage -> more damage, lower critical gamma
    out["rho_auc"] = float(spearmanr(lev, R["auc"])[0])
    out["rho_gc"] = float(spearmanr(lev, -R["gc"])[0])
    out["rho_auc_top3"] = float(spearmanr(pre["lev_top"], R["auc"])[0])
    for b in BASELINES:
        out[f"rho_auc_{b}"] = float(spearmanr(base[b], R["auc"])[0])
        out[f"rho_gc_{b}"] = float(spearmanr(base[b], -R["gc"])[0])
    out["partial_auc"] = partial_spearman(lev, R["auc"], Z)
    out["partial_gc"] = partial_spearman(lev, -R["gc"], Z)
    # the frozen top-1 claim
    i = int(np.argmax(lev))
    out["top_rank_auc"] = int((R["auc"] > R["auc"][i]).sum())
    out["top_rank_gc"] = int((R["gc"] < R["gc"][i]).sum())
    out["top_is_max"] = bool(out["top_rank_auc"] == 0)
    # sign: only the susceptibility predicts the +/- asymmetry
    if np.all(np.isfinite(R["asym"])):
        out["rho_asym"] = float(spearmanr(lev_s, R["asym"])[0])
        p, n = lev_s > 0, lev_s < 0
        if p.sum() > 1 and n.sum() > 1:
            out["asym_pos"] = float(np.median(R["asym"][p]))
            out["asym_neg"] = float(np.median(R["asym"][n]))
            out["asym_gap"] = out["asym_pos"] - out["asym_neg"]
        out["frac_signed_pos"] = float(np.mean(p))
    return out


def across(stats):
    g = lambda key: np.array([s[key] for s in stats
                              if np.isfinite(s.get(key, np.nan))])
    w = lambda v: float(wilcoxon(v).pvalue) if len(v) > 5 else np.nan
    A = {"n_net": len(stats)}
    for key in ("rho_auc", "rho_gc", "rho_auc_top3", "partial_auc", "partial_gc",
                "rho_asym", "asym_gap", *[f"rho_auc_{b}" for b in BASELINES],
                *[f"rho_gc_{b}" for b in BASELINES]):
        v = g(key)
        if len(v):
            A[key] = float(np.median(v))
            A[key + "_lo"], A[key + "_hi"] = float(v.min()), float(v.max())
            A["p_" + key] = w(v)
            A["n_pos_" + key] = int((v > 0).sum())
    for key in ("top_rank_auc", "top_rank_gc", "n_censored", "k", "gc_median",
                "auc_median"):
        v = g(key)
        if len(v):
            A[key] = float(np.median(v))
    # leverage vs its strongest baseline, paired per network
    best, bestv = None, -np.inf
    for b in BASELINES:
        v = g(f"rho_auc_{b}")
        if len(v) and np.median(v) > bestv:
            best, bestv = b, float(np.median(v))
    if best:
        d = g("rho_auc") - g(f"rho_auc_{best}")
        A["best_baseline"] = best
        A["lev_minus_best"] = float(np.median(d))
        A["p_lev_minus_best"] = w(d)
        A["n_lev_beats_best"] = int((d > 0).sum())
    return A


def report(A):
    print(f"\n########## {A['n_net']} networks  "
          f"(k~{A['k']:.0f} roles, {A['n_censored']:.0f} never broke) ##########")
    print(f"  graded: rho(leverage, integrated deficit)   {A['rho_auc']:+.3f} "
          f"[{A['rho_auc_lo']:+.2f},{A['rho_auc_hi']:+.2f}] p={A['p_rho_auc']:.4f} "
          f"({A['n_pos_rho_auc']}/{A['n_net']} > 0)")
    print(f"          rho(leverage, -critical gamma)      {A['rho_gc']:+.3f} "
          f"[{A['rho_gc_lo']:+.2f},{A['rho_gc_hi']:+.2f}] p={A['p_rho_gc']:.4f} "
          f"({A['n_pos_rho_gc']}/{A['n_net']} > 0)")
    print(f"          top-3 modes instead of leading      {A['rho_auc_top3']:+.3f}")
    print("\n  baselines from the same W_rec (rho with integrated deficit):")
    for b in BASELINES:
        print(f"      {b:<10} {A[f'rho_auc_{b}']:+.3f} "
              f"[{A[f'rho_auc_{b}_lo']:+.2f},{A[f'rho_auc_{b}_hi']:+.2f}]")
    if "best_baseline" in A:
        print(f"      leverage - best ({A['best_baseline']}): {A['lev_minus_best']:+.3f} "
              f"({A['n_lev_beats_best']}/{A['n_net']} nets) p={A['p_lev_minus_best']:.4f}")
    print(f"\n  partial rho | baselines: deficit {A['partial_auc']:+.3f} "
          f"p={A['p_partial_auc']:.4f} ({A['n_pos_partial_auc']}/{A['n_net']} > 0)"
          f"   critical gamma {A['partial_gc']:+.3f} p={A['p_partial_gc']:.4f}")
    print(f"  frozen top-1 role: deficit rank {A['top_rank_auc']:.1f}, "
          f"critical-gamma rank {A['top_rank_gc']:.1f} (of ~{A['k']:.0f})")
    if "rho_asym" in A:
        print(f"  sign: rho(signed leverage, +/- asymmetry) {A['rho_asym']:+.3f} "
              f"p={A['p_rho_asym']:.4f} ({A['n_pos_rho_asym']}/{A['n_net']} > 0)"
              f"   gap {A.get('asym_gap', np.nan):+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="3-12")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--trial-seed", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=THRESH)
    ap.add_argument("--gammas", default=",".join(str(g) for g in GAMMA_POS + GAMMA_NEG))
    ap.add_argument("--prereg-only", action="store_true")
    ap.add_argument("--out", default="gain_dose_response.json")
    args = ap.parse_args()

    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    gammas = sorted(float(g) for g in args.gammas.split(","))

    # ---- phase A: freeze the ranking BEFORE any trial is generated ----------
    ks = {s: int(OPK[s]) for s in seeds}      # before task.py rebinds np.load
    pre = {}
    for s in seeds:
        try:
            pre[s] = prereg_net(s, ks[s])
        except (FileNotFoundError, KeyError) as e:
            print(f"[skip] seed {s}: {type(e).__name__}: {e}")
    if not pre:
        return
    paths.ensure_dir(OUT, "results dir")
    pfile = OUT / args.out.replace(".json", "_prereg.json")
    json.dump({"gammas": gammas, "thresh": args.thresh,
               "nets": [prereg_json(pre[s]) for s in sorted(pre)]},
              open(pfile, "w"), indent=1)
    print(f"froze rankings for {len(pre)} networks -> {pfile}")
    for s in sorted(pre):
        P = pre[s]
        print(f"  seed{s:<3} k={P['k']:3d} |lambda1|={P['lam_lead']:.3f} "
              f"top role {P['rank_by_lev'][0]:3d} (size {P['sizes'][P['rank_by_lev'][0]]:3d}) "
              f"leverage {P['lev'].max():.3f} vs median {np.median(P['lev']):.3f}")
    if args.prereg_only:
        return

    # ---- phase B/C ----------------------------------------------------------
    stats, records = [], []
    for s in sorted(pre):
        P = pre[s]
        net = NpNet(P["net"], dtype=np.float32)
        rules = list(net.rules)
        trials = make_trials(net, rules, batch_size=args.batch_size,
                             seed=args.trial_seed)
        units = [P["ind"][P["role"] == c] for c in range(P["k"])]
        print(f"=== seed {s}  k={P['k']}  {len(gammas)} gammas ===", flush=True)
        perf, perf0 = sweep(net, trials, rules, units, gammas)
        R = readouts(perf, perf0, gammas, args.thresh)
        st = net_stats(perf, R, P, gammas)
        stats.append(st)
        records.append({"seed": s, "net": P["net"], "k": P["k"], "rules": rules,
                        "gammas": gammas, "perf": perf.tolist(),
                        "perf_intact": perf0.tolist(),
                        "gc": R["gc"].tolist(), "censored": R["censored"].tolist(),
                        "auc": R["auc"].tolist(), "asym": R["asym"].tolist(),
                        "stats": st})
        print(f"  seed{s:<3} rho_auc {st['rho_auc']:+.3f} rho_gc {st['rho_gc']:+.3f} "
              f"partial {st['partial_auc']:+.3f} top-1 rank {st['top_rank_auc']:2d} "
              f"asym rho {st.get('rho_asym', np.nan):+.3f} "
              f"censored {st['n_censored']}/{st['k']}", flush=True)

    A = across(stats)
    report(A)
    json.dump({"summary": A, "nets": records, "gammas": gammas,
               "thresh": args.thresh}, open(OUT / args.out, "w"), indent=1,
              default=float)
    print(f"\nwrote {OUT / args.out}")


if __name__ == "__main__":
    main()
