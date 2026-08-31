"""Does the connectivity-only susceptibility map predict a BEHAVIOURAL deficit matrix?

gain_targeted_modulation.py perturbs a role's gain and measures an eigenvalue.
This does the same intervention and measures whether the network can still do
the tasks.

Intervention (numpy forward simulation, flexsym.core.np_sim):
  direct   -- multiply the activation of every active unit in role r by (1+gamma),
              so its Jacobian gain becomes (1+gamma) f'(u): exactly the dg the
              susceptibility is a derivative in.
  inject   -- add a constant current to those units, shifting their operating
              point. The induced gain change is state-dependent, so it is
              MEASURED (mean f'(u) per unit, per rule) rather than assumed. This
              is what an optogenetic / neuromodulatory perturbation actually is.

Observed object: the role x task deficit matrix D. Perturb every role in turn,
evaluate performance on all 15 tasks. Trials and the recurrent-noise stream are
fixed across conditions (common random numbers), so D is a matrix of paired
differences.

Predicted object: from S_ev[i,j] = d|lambda_j|/dg_i on the quotient of W_rec.
For each condition c (a rule x epoch fixed-point set) take j* = the leading
quotient mode and the column sev_c = S_ev[:,j*]. Then

  theory    P[r,t] = mean over conditions of rule t of |sev_c[r]| * gamma * gbar_c[r]
  measured  P[r,t] = mean over conditions of rule t of |sev_c . dgbar[r,t]|

The theory form has no fitted parameters and no simulation input at all: gamma
is the imposed knob, gbar and S_ev come from the stored Jacobian and W_rec. The
measured form uses the realised role-gain change (needed for `inject`, and a
consistency check for `direct`).

Statistics
  matrix    Pearson / Spearman of vec(P) vs vec(D), raw and DOUBLE-CENTERED.
            Double-centering removes role main effects (big roles hurt every
            task) and task main effects (fragile tasks are hurt by everything),
            leaving only the role x task interaction -- which is the whole claim.
  null      permute role identity in P (rows). This preserves the susceptibility
            distribution exactly and destroys only its assignment, matching the
            null in gain_targeted_modulation.
  selection for each condition, i* = argmax_i |sev_c[i]| is the role S_ev says
            has most leverage. Rank of D[i*, t(c)] among roles (does the predicted
            role hurt that task most?) and among tasks (does that role hurt its
            own task most?), plus the specificity index
            SI = D[i*,t] / mean_{t'!=t} D[i*,t'].

Seed is the unit of replication.

    conda run -n myenv python -m flexsym.run.gain_behavioural_specificity --seeds 3-12
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon

from flexsym import paths
from flexsym.core.np_sim import NpNet, make_trials, perf_by_rule
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.gain_leverage_geometry import qmat, rotation_sensitivity
from flexsym.run.motif_quotient_prediction import (
    condition_jacobians,
    indicator,
    load_conditions,
    quotient,
)
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.structural.operating_k import OPK

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

OUT = paths.RESULTS_ROOT / "motif_fixed_points"
TOL = 1e-12
N_PERM = 2000


# --------------------------------------------------------------------------- #
# prediction side: S_ev per condition
# --------------------------------------------------------------------------- #
def condition_susceptibility(conds, W, H, sizes):
    """sev = S_ev[:, j*] and gbar per condition, j* = leading quotient mode."""
    Qs = quotient(W, H, sizes)
    out = []
    for cd in conds:
        g, alpha = cd["g"], cd["alpha"]
        if not (np.all(np.isfinite(g)) and np.isfinite(alpha)):
            continue
        gbar = (H.T @ g) / sizes
        Sev = rotation_sensitivity(Qs, gbar, alpha)[1]
        lam_q = np.linalg.eigvals(qmat(Qs, gbar, alpha))
        j = int(np.argmax(np.abs(lam_q)))
        sev = Sev[:, j]
        out.append({"key": cd["key"], "motif": cd["motif"], "rule": cd["rule"],
                    "sev": sev, "gbar": gbar, "j": j,
                    "istar": int(np.argmax(np.abs(sev)))})
    return out


def predicted_matrices(csev, rules, k, gamma, dgbar=None, agg="mean"):
    """theory, measured, and the S_ev-free magnitude control.

    dgbar: (k roles perturbed, n_rules, k roles) measured role-gain change.

    `magnitude` is ||dgbar[r,t]|| -- how far the intervention moved the gains at
    all, with no susceptibility in it. It is the control the measured predictor
    needs: a perturbation that propagates widely both changes many gains AND
    breaks behaviour, so `measured` beating the role-permutation null does not by
    itself implicate S_ev. Only measured > magnitude does.
    """
    f = np.nanmean if agg == "mean" else np.nanmax
    Pt = np.full((k, len(rules)), np.nan)
    Pm = np.full((k, len(rules)), np.nan) if dgbar is not None else None
    Pg = np.full((k, len(rules)), np.nan) if dgbar is not None else None
    for ti, t in enumerate(rules):
        cs = [c for c in csev if c["rule"] == t]
        if not cs:
            continue
        Pt[:, ti] = f(np.array([np.abs(c["sev"]) * gamma * c["gbar"] for c in cs]), 0)
        if dgbar is not None:
            Pm[:, ti] = f(np.array([np.abs(dgbar[:, ti, :] @ c["sev"]) for c in cs]), 0)
            Pg[:, ti] = np.linalg.norm(dgbar[:, ti, :], axis=-1)
    return Pt, Pm, Pg


# --------------------------------------------------------------------------- #
# observation side: the behavioural deficit matrix
# --------------------------------------------------------------------------- #
def deficit_matrix(net, trials, rules, units_by_role, mode, amount, noise_seed=0,
                   verbose=True):
    """(D, dgbar, perf_intact). D[r,t] = intact perf - perf with role r modulated."""
    k = len(units_by_role)
    p0, g0 = perf_by_rule(net, trials, seed=noise_seed, track_gain=True)
    perf0 = np.array([p0[t] for t in rules])
    D = np.zeros((k, len(rules)))
    G = np.zeros((k, len(rules), net.n_rnn))
    t0 = time.time()
    for r, units in enumerate(units_by_role):
        kw = ({"gain": net.gain_vector(units, amount)} if mode == "direct"
              else {"bias": net.bias_vector(units, amount)})
        p, g = perf_by_rule(net, trials, seed=noise_seed, track_gain=True, **kw)
        D[r] = perf0 - np.array([p[t] for t in rules])
        G[r] = np.array([g[t] for t in rules])
        if verbose and (r + 1) % 10 == 0:
            print(f"    role {r+1}/{k}  ({time.time()-t0:.0f}s)", flush=True)
    G0 = np.array([g0[t] for t in rules])
    return D, G - G0[None], perf0


def role_means(dG_units, role, ind_active, k):
    """(k, n_rules, n_units) unit-wise gain change -> (k, n_rules, k) role means."""
    out = np.zeros((dG_units.shape[0], dG_units.shape[1], k))
    for c in range(k):
        out[:, :, c] = dG_units[:, :, ind_active[role == c]].mean(-1)
    return out


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def dcenter(M):
    return M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()


def matrix_stats(P, D, rng, n_perm=N_PERM):
    """Correlate predicted with observed; role-permutation null."""
    cols = np.where(np.all(np.isfinite(P), 0))[0]
    if len(cols) < 2:
        return {}
    p, d = P[:, cols], D[:, cols]
    res = {"n_task_col": int(len(cols)), "n_role": int(p.shape[0])}
    for tag, (a, b) in (("raw", (p, d)), ("dc", (dcenter(p), dcenter(d)))):
        vr, vo = a.ravel(), b.ravel()
        res[f"r_{tag}"] = float(pearsonr(vr, vo)[0])
        res[f"rho_{tag}"] = float(spearmanr(vr, vo)[0])
    null_r, null_rho = [], []
    pdc, ddc = dcenter(p), dcenter(d)
    for _ in range(n_perm):
        q = rng.permutation(p.shape[0])
        null_r.append(pearsonr(dcenter(p[q]).ravel(), ddc.ravel())[0])
        null_rho.append(spearmanr(p[q].ravel(), d.ravel())[0])
    null_r, null_rho = np.array(null_r), np.array(null_rho)
    res["p_r_dc"] = float((np.sum(null_r >= res["r_dc"]) + 1) / (n_perm + 1))
    res["p_rho_raw"] = float((np.sum(null_rho >= res["rho_raw"]) + 1) / (n_perm + 1))
    res["z_r_dc"] = float((res["r_dc"] - null_r.mean()) / max(null_r.std(), TOL))
    return res


def selection_stats(csev, D, rules):
    """Per condition: does the S_ev-selected role hurt its own task, specifically?"""
    k = D.shape[0]
    rows = []
    for c in csev:
        if c["rule"] not in rules:
            continue
        t = rules.index(c["rule"])
        i = c["istar"]
        col, row = D[:, t], D[i]
        other = np.delete(row, t)
        mo = float(np.mean(other))
        rows.append({
            "key": c["key"], "motif": c["motif"], "rule": c["rule"], "istar": i,
            "deficit": float(D[i, t]),
            "rank_role": int((col > D[i, t]).sum()),        # among roles, this task
            "rank_task": int((row > D[i, t]).sum()),        # among tasks, this role
            "k": k, "n_task": len(rules),
            "si": float(D[i, t] / mo) if abs(mo) > 1e-6 else np.nan,
            "si_diff": float(D[i, t] - mo),
            "col_max": float(col.max()), "row_max": float(row.max()),
        })
    return rows


def summarise_net(sel, mat, D, sizes, perf0):
    if not sel:
        return {}
    rr = np.array([s["rank_role"] for s in sel])
    rt = np.array([s["rank_task"] for s in sel])
    k, nt = sel[0]["k"], sel[0]["n_task"]
    return {
        "n_cond": len(sel),
        "rank_role": float(np.median(rr)), "rank_role_null": (k - 1) / 2,
        "hit_role": float((rr == 0).mean()), "chance_role": 1.0 / k,
        "rank_task": float(np.median(rt)), "rank_task_null": (nt - 1) / 2,
        "hit_task": float((rt == 0).mean()), "chance_task": 1.0 / nt,
        "si": float(np.nanmedian([s["si"] for s in sel])),
        "si_diff": float(np.median([s["si_diff"] for s in sel])),
        "deficit": float(np.median([s["deficit"] for s in sel])),
        "D_mean": float(D.mean()), "D_max": float(D.max()),
        "rho_size": float(spearmanr(sizes, D.mean(1))[0]),
        "perf_intact": float(perf0.mean()),
        **mat,
    }


# --------------------------------------------------------------------------- #
def run_net(seed, mode, amount, batch_size, n_perm, agg, trial_seed, k=None,
            verbose=True):
    net_name = f"LeakyRNN_softplus_diag_256_all_{seed}"
    wl = npload(WL_ROOT / net_name / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], float)
    ind = np.asarray(wl["ind_active"], int)
    conds = load_conditions(net_name, ind, W, condition_jacobians(net_name, ind, W))
    k = int(OPK[seed]) if k is None else int(k)
    role = np.asarray(wl["labels_struct_by_k"][_row_for_k(wl["labels_struct_by_k"], k)],
                      int)
    H, sizes = indicator(role)
    kk = H.shape[1]
    csev = condition_susceptibility(conds, W, H, sizes)

    net = NpNet(net_name)
    rules = list(net.rules)
    trials = make_trials(net, rules, batch_size=batch_size, seed=trial_seed)
    units_by_role = [ind[role == c] for c in range(kk)]
    if verbose:
        have = sorted({c["rule"] for c in csev})
        print(f"=== seed {seed}  k={kk}  {len(csev)} conditions over {len(have)} rules "
              f"| {mode} {amount:+g} ===", flush=True)

    D, dG, perf0 = deficit_matrix(net, trials, rules, units_by_role, mode, amount,
                                  verbose=verbose)
    dgbar = role_means(dG, role, ind, kk)
    Pt, Pm, Pg = predicted_matrices(csev, rules, kk, amount, dgbar, agg=agg)

    rng = np.random.default_rng(seed)
    stats = {}
    for tag, P in (("theory", Pt), ("measured", Pm), ("magnitude", Pg)):
        m = matrix_stats(P, D, rng, n_perm)
        stats[tag] = summarise_net(selection_stats(csev, D, rules), m, D,
                                   sizes.astype(int), perf0)
    sel = selection_stats(csev, D, rules)
    return {
        "seed": seed, "net": net_name, "k": kk, "mode": mode, "amount": amount,
        "rules": rules, "sizes": sizes.astype(int).tolist(),
        "perf_intact": perf0.tolist(),
        "D": D.tolist(), "P_theory": Pt.tolist(), "P_measured": Pm.tolist(),
        "P_magnitude": Pg.tolist(), "dgbar": dgbar.tolist(),
        "stats": stats, "selection": sel,
    }


def across(nets, tag):
    """Seed as the unit of replication."""
    S = [n["stats"][tag] for n in nets if n["stats"].get(tag)]
    if len(S) < 2:
        return {}
    g = lambda key: np.array([s[key] for s in S if np.isfinite(s.get(key, np.nan))])
    w = lambda a, b: float(wilcoxon(a - b).pvalue) if len(a) > 5 else np.nan
    out = {"n_net": len(S)}
    for key in ("r_raw", "r_dc", "rho_raw", "rho_dc"):
        v = g(key)
        if len(v):
            out[key] = float(np.median(v))
            out[key + "_lo"], out[key + "_hi"] = float(v.min()), float(v.max())
            out["p_" + key] = w(v, np.zeros(len(v)))
            out["n_pos_" + key] = int((v > 0).sum())
    for a, b in (("rank_role", "rank_role_null"), ("rank_task", "rank_task_null"),
                 ("hit_role", "chance_role"), ("hit_task", "chance_task")):
        out[a] = float(np.median(g(a)))
        out[b] = float(np.median(g(b)))
        out[f"p_{a}"] = w(g(a), g(b))
    for key in ("si", "si_diff", "deficit", "D_mean", "D_max", "rho_size",
                "perf_intact", "z_r_dc"):
        v = g(key)
        if len(v):
            out[key] = float(np.median(v))
    v = g("si")
    if len(v):
        out["p_si"] = w(v, np.ones(len(v)))
    return out


def report(nets, tag):
    A = across(nets, tag)
    if not A:
        print(f"  [{tag}] too few networks")
        return A
    print(f"\n  ---- prediction: {tag} ----")
    print(f"   matrix r (raw)        {A.get('r_raw', np.nan):+.3f} "
          f"[{A.get('r_raw_lo', np.nan):+.2f},{A.get('r_raw_hi', np.nan):+.2f}] "
          f"p={A.get('p_r_raw', np.nan):.4f}  ({A.get('n_pos_r_raw', 0)}/{A['n_net']} > 0)")
    print(f"   matrix r (2-centered) {A.get('r_dc', np.nan):+.3f} "
          f"[{A.get('r_dc_lo', np.nan):+.2f},{A.get('r_dc_hi', np.nan):+.2f}] "
          f"p={A.get('p_r_dc', np.nan):.4f}  ({A.get('n_pos_r_dc', 0)}/{A['n_net']} > 0)"
          f"   z vs role-perm null {A.get('z_r_dc', np.nan):+.2f}")
    print(f"   spearman rho (raw)    {A.get('rho_raw', np.nan):+.3f}   "
          f"(2-centered) {A.get('rho_dc', np.nan):+.3f}")
    print(f"   selected role rank    {A['rank_role']:.1f} vs null "
          f"{A['rank_role_null']:.1f}  p={A['p_rank_role']:.4f} "
          f"| hit {100*A['hit_role']:.0f}% vs {100*A['chance_role']:.1f}%")
    print(f"   its task rank         {A['rank_task']:.1f} vs null "
          f"{A['rank_task_null']:.1f}  p={A['p_rank_task']:.4f} "
          f"| hit {100*A['hit_task']:.0f}% vs {100*A['chance_task']:.1f}%")
    print(f"   specificity index     {A.get('si', np.nan):.2f} "
          f"(target/non-target) p={A.get('p_si', np.nan):.4f} "
          f"| target-nontarget deficit {A.get('si_diff', np.nan):+.3f}")
    print(f"   deficits: mean {A['D_mean']:+.3f}, max {A['D_max']:+.3f}, "
          f"intact perf {A['perf_intact']:.3f}, rho(role size, deficit) "
          f"{A['rho_size']:+.2f}")
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="3-12")
    ap.add_argument("--mode", default="direct", choices=("direct", "inject", "both"))
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="direct: activation-slope multiplier is 1+gamma")
    ap.add_argument("--current", type=float, default=0.5,
                    help="inject: constant current added to the role's units")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--trial-seed", type=int, default=0)
    ap.add_argument("--agg", default="mean", choices=("mean", "max"))
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--out", default="gain_behavioural_specificity.json")
    args = ap.parse_args()

    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    # OPK reads .npz files through the plain np.load; importing flexsym.core.task
    # (which make_trials does) monkeypatches np.load to force allow_pickle and
    # then those reads raise. Resolve every k BEFORE the first trial is made.
    ks = {s: int(OPK[s]) for s in seeds}

    modes = (("direct", args.gamma), ("inject", args.current))
    if args.mode != "both":
        modes = [m for m in modes if m[0] == args.mode]

    all_nets = {}
    for mode, amount in modes:
        nets = []
        for s in seeds:
            try:
                nets.append(run_net(s, mode, amount, args.batch_size, args.n_perm,
                                    args.agg, args.trial_seed, k=ks[s]))
            except (FileNotFoundError, KeyError) as e:
                print(f"[skip] seed {s}: {type(e).__name__}: {e}", flush=True)
                continue
            st = nets[-1]["stats"].get("theory", {})
            print(f"  seed{s:<3} r_dc {st.get('r_dc', np.nan):+.3f} "
                  f"rho_raw {st.get('rho_raw', np.nan):+.3f} "
                  f"rank_role {st.get('rank_role', np.nan):.1f}/"
                  f"{st.get('rank_role_null', np.nan):.1f} "
                  f"SI {st.get('si', np.nan):.2f} "
                  f"Dmean {st.get('D_mean', np.nan):+.3f}", flush=True)
        if not nets:
            continue
        print(f"\n########## {mode} {amount:+g}  ({len(nets)} networks) ##########")
        summary = {tag: report(nets, tag)
                   for tag in ("theory", "measured", "magnitude")}
        # does S_ev add anything to the raw size of the induced gain change?
        d = np.array([n["stats"]["measured"]["r_dc"] - n["stats"]["magnitude"]["r_dc"]
                      for n in nets if n["stats"].get("magnitude")])
        if len(d) > 5:
            print(f"\n  measured - magnitude (interaction r): {np.median(d):+.3f} "
                  f"({int((d > 0).sum())}/{len(d)} nets) "
                  f"p={wilcoxon(d).pvalue:.4f}")
            summary["measured_minus_magnitude"] = {
                "median": float(np.median(d)), "n_pos": int((d > 0).sum()),
                "n": int(len(d)), "p": float(wilcoxon(d).pvalue)}
        all_nets[f"{mode}_{amount:g}"] = {"nets": nets, "summary": summary}

    if all_nets:
        paths.ensure_dir(OUT, "results dir")
        p = OUT / args.out
        json.dump(all_nets, open(p, "w"), indent=1, default=float)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
