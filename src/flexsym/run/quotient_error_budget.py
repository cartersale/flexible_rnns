"""The three-factor error budget of a quotient reduction: B, D, C.

`transverse_contraction.py` asks whether the transverse block D contracts.
That is only the middle of the story. The quantity that decides how wrong a
reduced (quotient) model is has three factors and a round trip through them:

    y --B--> z --D--> z --C--> y

    B = Pi J P    how much collective (quotient) motion leaks OUT of the
                  partition-respecting subspace into within-role differences
    D = Pi J Pi   what the transverse dynamics then DO to that leaked error
    C = P J Pi    how much the resulting within-role differences feed BACK
                  into the collective variables

A partition can be mediocre in B and still give an excellent reduction if the
directions it leaks into are crushed; a partition with tiny B can give a bad
reduction if its leakage enters a strongly amplifying transverse direction.
Neither B alone (the usual "how nearly equitable is this partition?" score,
and what Gamma_signed is a proxy for) nor the spectrum of D alone is
sufficient. This script measures all three and the composites.

TRANSVERSE AMPLIFICATION. Eigenvalues of D are not enough: a non-normal D can
have every eigenvalue inside the unit circle yet amplify a perturbation by
orders of magnitude before it decays. The quantity that enters the bound is the
total lifetime amplification

    M = sum_{t>=0} ||D^t||_2                (discrete-time counterpart of
                                             M = int_0^inf ||e^{Ds}|| ds)

reported two ways:

  M_emp(T)   summed directly out to horizon T. Exact for the horizon that
             matters (trials are tens of steps), and defined even when D is not
             asymptotically contracting.
  M_cert     a Lyapunov certificate, i.e. a guaranteed bound over ALL time with
             no sampling. Theta solves D' Theta D - Theta = -I (so Theta >= I,
             kappa = lambda_max(Theta)); then ||D^t||_2 <= sqrt(kappa) rho_th^t
             with rho_th = sqrt(1 - 1/kappa_max), giving
                 M <= M_cert = sqrt(kappa) / (1 - rho_th),
             and, in the continuous-time reading with decay rate
             alpha = -log rho_th per step, M <= sqrt(kappa) / alpha.
             Exists only when rho(D) < 1.

COMPOSITES (the two numbers the framework says to look at):

    BM   = ||B||_2 * M            leak-out x amplification
    BMC  = ||B||_2 * M * ||C||_2  the full round trip; the coefficient in
                                  error <~ BMC * sup||y|| + M ||C|| ||z(0)||

OPERATOR VS FROBENIUS. Gamma_signed sums squared within-role input mismatches:
it is a total-energy (Frobenius-like) score. The bound wants the worst-case
gain ||B||_2. `b_conc` = ||B||_F / ||B||_2 separates the two regimes: near 1
means the leakage is concentrated in one direction (worst case ~ total energy);
large means it is spread thinly over many directions, so Gamma overstates the
danger. Both norms are recorded, along with Gamma_signed of the partition on
the matrix it was fitted to, so the proxy can be checked directly.

Also recorded per cell, to test the bound rather than assume it: the measured
quotient prediction error from group-constant initial conditions
(`theorem_a_probe` from transverse_contraction), and z0_typ = ||z(0)||/||x(0)||
for real task trajectories is NOT computed here (see the probe's z0 = 0 setup) --
this script covers the first term of the bound only.

Surrogates, partitions and gain modes follow transverse_contraction.py exactly
(fresh signed-WL partition fitted on each surrogate; gains paired with the
trained network). The `unit` gain mode is excluded by default: it puts rho(J)
around 11, where every matrix is wildly expansive and M is meaningless.

  conda run -n myenv python -m flexsym.run.quotient_error_budget --seeds 3-12

Writes results/main/signed_input_wl_paper/lumpability/quotient_error_budget.csv
and a companion .json (paired trained-vs-surrogate tests, the Gamma/||B||_2
correspondence, and how well each composite predicts the measured error).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings

import numpy as np
from scipy.linalg import solve_discrete_lyapunov, svdvals
from scipy.stats import spearmanr, wilcoxon

from flexsym import paths
from flexsym.run.control_ksweep import make_matrix
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.run.transverse_contraction import (
    WL_ROOT,
    gain_vectors,
    indicator,
    load_network,
    theorem_a_probe,
    transverse_basis,
)
from flexsym.structural.signed_input_wl_structural_roles import (
    approx_wl_signed,
    gamma_signed,
    random_labels_same_sizes,
)

OUT = WL_ROOT / "lumpability" / "quotient_error_budget.csv"

K_GRID = [8, 12, 16, 20, 25, 30, 40, 50]
TYPES = ["trained", "trained-cached", "random-part", "row-shuffle", "spectrum",
         "gaussian"]
GAIN_MODES = ["full", "cond"]
HORIZONS = [10, 20, 50, 200]     # M_emp is reported at each; the last is "lifetime"
N_RANDPART = 5
N_COND = 8
N_TRAJ = 8
M_CAP = 1e12                     # bail out of the sum if D is explosive


# --------------------------------------------------------------------------- #
# the three factors
# --------------------------------------------------------------------------- #
def _top_sv(P: np.ndarray, v: np.ndarray, iters: int) -> tuple[float, np.ndarray]:
    """(||P||_2, refreshed leading right singular vector) by warm-started power
    iteration on P'P. Full SVDs at every one of the T powers dominate the cost
    and buy nothing: v changes slowly along the sequence D^t, so a warm start
    converges in a handful of matvecs. Accuracy is checked against svdvals in
    the module test (`--check-sv`)."""
    for _ in range(iters):
        u = P @ v
        nu = np.linalg.norm(u)
        if nu <= 0 or not np.isfinite(nu):
            return (0.0 if nu == 0 else np.inf), v
        w = P.T @ (u / nu)
        nw = np.linalg.norm(w)
        if nw <= 0 or not np.isfinite(nw):
            return float(nu), v
        v = w / nw
    return float(np.linalg.norm(P @ v)), v


def m_empirical(D: np.ndarray, horizons: list[int],
                exact: bool = True) -> tuple[dict, float, int]:
    """(M_emp at each horizon, sup_t ||D^t||, argmax t), summing ||D^t||_2 directly.

    t = 0 contributes ||I|| = 1, so M_emp >= 1 always. Truncates (and returns NaN
    for the horizons beyond the truncation) if the running sum passes M_CAP.
    """
    T = max(horizons)
    P = np.eye(D.shape[0])
    v = np.ones(D.shape[0]) / np.sqrt(D.shape[0])
    total, out = 1.0, {}
    peak, t_peak = 1.0, 0
    truncated = T
    for t in range(1, T + 1):
        P = D @ P
        if exact:
            sv = float(svdvals(P)[0])
        else:
            sv, v = _top_sv(P, v, 12 if t == 1 else 4)
        if not np.isfinite(sv):
            truncated = t - 1
            break
        total += sv
        if sv > peak:
            peak, t_peak = sv, t
        if t in horizons:
            out[t] = total
        if total > M_CAP:
            truncated = t
            break
    for h in horizons:
        out.setdefault(h, total if h <= truncated else np.nan)
    return out, peak, t_peak


NAN_CERT = {"kappa": np.nan, "theta_lmax": np.nan, "rho_theta": np.nan,
            "alpha_dec": np.nan, "M_cert": np.nan, "M_cert_cont": np.nan}


def m_certificate(D: np.ndarray) -> dict:
    """Lyapunov certificate for sum_t ||D^t||, valid for all time.

    Theta = sum_t (D^t)' D^t solves D' Theta D - Theta = -I and satisfies
    Theta >= I, so lambda_min >= 1 and kappa = lambda_max/lambda_min. In the
    Theta-norm ||D||_Theta = sqrt(1 - 1/lambda_max) =: rho_th < 1, hence
    ||D^t||_2 <= sqrt(kappa) rho_th^t and M <= sqrt(kappa)/(1 - rho_th).
    Returns NaNs unless rho(D) < 1 (the certificate does not exist otherwise).
    """
    nan = dict(NAN_CERT)
    try:
        Th = solve_discrete_lyapunov(D.T, np.eye(D.shape[0]))
    except Exception:
        return nan
    ev = np.linalg.eigvalsh(0.5 * (Th + Th.T))
    if not np.all(np.isfinite(ev)) or ev.min() <= 0:
        return nan
    lmax, lmin = float(ev.max()), float(ev.min())
    kappa = lmax / lmin
    if lmax <= 1.0:
        return nan
    rho_th = float(np.sqrt(1.0 - 1.0 / lmax))
    alpha = -float(np.log(rho_th)) if rho_th > 0 else np.inf
    return {"kappa": kappa, "theta_lmax": lmax, "rho_theta": rho_th,
            "alpha_dec": alpha,
            "M_cert": float(np.sqrt(kappa) / (1.0 - rho_th)),
            "M_cert_cont": float(np.sqrt(kappa) / alpha) if alpha > 0 else np.nan}


def budget(M: np.ndarray, labels: np.ndarray, g: np.ndarray, alpha_leak: float,
           horizons: list[int], rng: np.random.Generator, n_traj: int) -> dict:
    n = M.shape[0]
    H = indicator(labels)
    V, Upar = transverse_basis(H)
    J = (1 - alpha_leak) * np.eye(n) + alpha_leak * (g[:, None] * M)

    D = V.T @ J @ V
    Bm = V.T @ J @ Upar          # quotient -> transverse   (leak out)
    Cm = Upar.T @ J @ V          # transverse -> quotient   (leak back)

    lam = np.linalg.eigvals(D)
    rho = float(np.max(np.abs(lam)))
    d2 = float(svdvals(D)[0])
    b2, c2 = float(svdvals(Bm)[0]), float(svdvals(Cm)[0])
    bf, cf = float(np.linalg.norm(Bm)), float(np.linalg.norm(Cm))

    memp, peak, t_peak = m_empirical(D, horizons)
    cert = m_certificate(D) if rho < 1 else dict(NAN_CERT)

    row = {
        "rho_D": rho, "norm_D": d2, "peak": peak, "t_peak": t_peak,
        "contracting": int(rho < 1),
        "transient": int(rho < 1 and peak > 1 + 1e-9),
        "B2": b2, "BF": bf, "b_conc": (bf / b2 if b2 > 0 else np.nan),
        "C2": c2, "CF": cf, "c_conc": (cf / c2 if c2 > 0 else np.nan),
        **cert,
    }
    for h in horizons:
        row[f"M_emp_t{h}"] = memp[h]
        row[f"BM_emp_t{h}"] = b2 * memp[h]
        row[f"BMC_emp_t{h}"] = b2 * memp[h] * c2
    row["BM_cert"] = b2 * cert["M_cert"]
    row["BMC_cert"] = b2 * cert["M_cert"] * c2
    # the naive Neumann version, for comparison: M <= 1/(1 - ||D||_2)
    row["M_naive"] = 1.0 / (1.0 - d2) if d2 < 1 else np.nan
    row["BMC_naive"] = b2 * row["M_naive"] * c2
    row.update(theorem_a_probe(J, H, V, Upar, max(horizons), n_traj, rng))
    return row


def summarise(cells: list[dict]) -> dict:
    if len(cells) == 1:
        return dict(cells[0], n_cells=1)
    out = {"n_cells": len(cells)}
    for key in cells[0]:
        vals = np.array([c[key] for c in cells], dtype=float)
        if not np.any(np.isfinite(vals)):
            out[key] = np.nan
        elif key in ("transient", "contracting"):
            out[key] = float(np.nanmean(vals))
        else:
            out[key] = float(np.nanmedian(vals))
    return out


# --------------------------------------------------------------------------- #
def run_network(net: str, seed: int, size: int, types: list[str], kg: list[int],
                gain_modes: list[str], horizons: list[int], args) -> list[dict]:
    W, ls_cached, alpha_leak, G, _ind = load_network(net)
    n = W.shape[0]
    rows = []
    if G is None:
        print(f"  [{net}] no usable jbar cache -- skipped (needs operating gains)",
              flush=True)
        return rows

    for kind in types:
        rng_m = np.random.default_rng(abs(hash((seed, size, kind))) % (2 ** 32))
        base = kind.split("::")[0]
        M = W if base in ("trained", "trained-cached", "random-part") \
            else make_matrix(W, base, rng_m)
        Mpos, Mneg = np.maximum(M, 0.0), np.maximum(-M, 0.0)
        fro_M = float(np.linalg.norm(M))

        for k in kg:
            if base == "trained-cached":
                label_sets = [(0, np.asarray(ls_cached[_row_for_k(ls_cached, k)], int))]
            elif base == "random-part":
                lab = np.asarray(ls_cached[_row_for_k(ls_cached, k)], int)
                label_sets = [(d, random_labels_same_sizes(lab, seed=d))
                              for d in range(args.n_randpart)]
            else:
                lab, _ = approx_wl_signed(Mpos, Mneg, k, n_iter=20, seed=0, n_init=20)
                label_sets = [(0, np.asarray(lab, dtype=int))]

            for draw, labels in label_sets:
                k_real = int(np.unique(labels).size)
                if k_real < 2 or k_real >= n:
                    continue
                H = indicator(labels)
                gam = gamma_signed(Mpos, Mneg, H)
                for mode in gain_modes:
                    gs = gain_vectors(mode, G, n, args.n_cond)
                    if not gs:
                        continue
                    rng = np.random.default_rng(
                        abs(hash((seed, kind, k, draw, mode))) % (2 ** 32))
                    cells = [budget(M, labels, g, alpha_leak, horizons, rng,
                                    args.n_traj) for g in gs]
                    rows.append({
                        "model": net, "size": size, "seed": seed, "type": kind,
                        "draw": draw, "k_req": k, "k": k_real, "n_active": n,
                        "gain_mode": mode, "alpha_leak": alpha_leak,
                        "gamma_signed": gam, "gamma_signed_norm": gam / fro_M,
                        **summarise(cells),
                    })
            done = [r for r in rows if r["type"] == kind and r["k_req"] == k
                    and r["gain_mode"] == gain_modes[0]]
            if done:
                r0 = done[0]
                print(f"    {kind:14s} k={r0['k']:3d} [{gain_modes[0]}] "
                      f"B={r0['B2']:.3g} C={r0['C2']:.3g} "
                      f"M50={r0['M_emp_t50']:.3g} Mcert={r0['M_cert']:.3g} "
                      f"BMC50={r0['BMC_emp_t50']:.3g}", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def metric_list(horizons: list[int]) -> list[str]:
    m = ["B2", "C2", "b_conc", "c_conc", "rho_D", "norm_D", "peak", "kappa",
         "alpha_dec", "M_cert", "BM_cert", "BMC_cert", "contracting",
         "gamma_signed", "err_par_t5", "err_par_t20", "ratio_max"]
    for h in horizons:
        m += [f"M_emp_t{h}", f"BM_emp_t{h}", f"BMC_emp_t{h}"]
    return m


def paired_stats(rows: list[dict], gain_modes: list[str], types: list[str],
                 horizons: list[int]) -> dict:
    out = {}
    mets = metric_list(horizons)
    for mode in gain_modes:
        for k in sorted({r["k_req"] for r in rows}):
            sub = [r for r in rows if r["gain_mode"] == mode and r["k_req"] == k]
            by_type: dict = {}
            for r in sub:
                by_type.setdefault(r["type"], {}).setdefault(r["seed"], []).append(r)
            if "trained" not in by_type:
                continue
            seeds = sorted(by_type["trained"])
            for other in types:
                if other == "trained" or other not in by_type:
                    continue
                common = [s for s in seeds if s in by_type[other]]
                if len(common) < 5:
                    continue
                rec = {"n": len(common)}
                for met in mets:
                    a = np.array([np.nanmean([r[met] for r in by_type["trained"][s]])
                                  for s in common], float)
                    b = np.array([np.nanmean([r[met] for r in by_type[other][s]])
                                  for s in common], float)
                    ok = np.isfinite(a) & np.isfinite(b)
                    if ok.sum() < 5 or np.allclose(a[ok], b[ok]):
                        rec[met] = {"trained": float(np.nanmedian(a)),
                                    "other": float(np.nanmedian(b)),
                                    "p": np.nan, "n_pairs": int(ok.sum())}
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _stat, p = wilcoxon(a[ok], b[ok])
                    rec[met] = {"trained": float(np.median(a[ok])),
                                "other": float(np.median(b[ok])),
                                "delta": float(np.median(a[ok] - b[ok])),
                                "p": float(p), "n_pairs": int(ok.sum())}
                out[f"{mode}|k{k}|trained_vs_{other}"] = rec
    return out


def predictor_stats(rows: list[dict], gain_modes: list[str],
                    horizons: list[int]) -> dict:
    """Which factor actually tracks the measured quotient error?

    Spearman of each candidate against err_par_t20 (and the ratio it bounds),
    pooled over the sweep and within k. If the framework is right, the round-trip
    composite should beat ||B||_2 alone and beat Gamma_signed.
    """
    cands = ["gamma_signed", "gamma_signed_norm", "B2", "BF", "C2", "M_cert",
             "BM_cert", "BMC_cert", "BMC_naive", "rho_D", "peak"] + \
            [f"M_emp_t{h}" for h in horizons] + \
            [f"BM_emp_t{h}" for h in horizons] + \
            [f"BMC_emp_t{h}" for h in horizons]
    out = {}
    for mode in gain_modes:
        for target in ("err_par_t20", "err_par", "ratio_max"):
            for scope in ("trained", "all_types"):
                sub = [r for r in rows if r["gain_mode"] == mode and
                       (scope == "all_types" or r["type"] == "trained")]
                if len(sub) < 8:
                    continue
                y = np.array([r[target] for r in sub], float)
                rec = {}
                for c in cands:
                    x = np.array([r[c] for r in sub], float)
                    ok = np.isfinite(x) & np.isfinite(y)
                    if ok.sum() < 8:
                        continue
                    rho_s, p = spearmanr(x[ok], y[ok])
                    rec[c] = {"rho": float(rho_s), "p": float(p), "n": int(ok.sum())}
                out[f"{mode}|{target}|{scope}|pooled"] = rec
            # within-k, trained only (removes the k trend driving everything)
            sub = [r for r in rows if r["gain_mode"] == mode and r["type"] == "trained"]
            per_k = {}
            for k in sorted({r["k_req"] for r in sub}):
                s_k = [r for r in sub if r["k_req"] == k]
                yy = np.array([r[target] for r in s_k], float)
                kk = {}
                for c in cands:
                    x = np.array([r[c] for r in s_k], float)
                    ok = np.isfinite(x) & np.isfinite(yy)
                    if ok.sum() >= 6:
                        rho_s, p = spearmanr(x[ok], yy[ok])
                        kk[c] = {"rho": float(rho_s), "p": float(p),
                                 "n": int(ok.sum())}
                if kk:
                    per_k[str(k)] = kk
            if per_k:
                out[f"{mode}|{target}|trained|within_k"] = per_k
    return out


def second_order_test(rows: list[dict], gain_modes: list[str]) -> dict:
    """Is the error second order in a joint defect, or only bilinear?

    Regress log(err_par_t20) on log(B2), log(C2) separately, and on
    log(B2*C2). Exponents near 1 on each of B and C is the bilinear
    (conservative) reading; a joint defect eps_joint = max(B2, C2) with
    error ~ eps_joint^2 is the stronger claim.
    """
    out = {}
    for mode in gain_modes:
        sub = [r for r in rows if r["gain_mode"] == mode and r["type"] == "trained"]
        y = np.array([r["err_par_t20"] for r in sub], float)
        b = np.array([r["B2"] for r in sub], float)
        c = np.array([r["C2"] for r in sub], float)
        ok = np.isfinite(y) & np.isfinite(b) & np.isfinite(c) & (y > 0) & (b > 0) & (c > 0)
        if ok.sum() < 12:
            continue
        X = np.column_stack([np.log(b[ok]), np.log(c[ok]), np.ones(ok.sum())])
        coef, *_ = np.linalg.lstsq(X, np.log(y[ok]), rcond=None)
        joint = np.log(np.maximum(b[ok], c[ok]))
        Xj = np.column_stack([joint, np.ones(ok.sum())])
        cj, *_ = np.linalg.lstsq(Xj, np.log(y[ok]), rcond=None)
        out[mode] = {"n": int(ok.sum()),
                     "exp_B": float(coef[0]), "exp_C": float(coef[1]),
                     "exp_joint_max": float(cj[0])}
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seeds", default="3-12")
    ap.add_argument("--k-grid", default=",".join(str(k) for k in K_GRID))
    ap.add_argument("--types", default=",".join(TYPES))
    ap.add_argument("--gain-modes", default=",".join(GAIN_MODES))
    ap.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    ap.add_argument("--n-randpart", type=int, default=N_RANDPART)
    ap.add_argument("--n-cond", type=int, default=N_COND)
    ap.add_argument("--n-traj", type=int, default=N_TRAJ)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds: list[int] = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    kg = [int(x) for x in args.k_grid.split(",")]
    types = args.types.split(",")
    gain_modes = args.gain_modes.split(",")
    horizons = sorted(int(x) for x in args.horizons.split(","))
    out_path = paths.REPO_ROOT / args.out if args.out else OUT

    rows, missing = [], []
    t0 = time.time()
    for s in seeds:
        net = f"LeakyRNN_softplus_diag_{args.size}_all_{s}"
        if not (WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz").exists():
            missing.append(net)
            continue
        print(f"[{time.time() - t0:6.0f}s] {net}", flush=True)
        rows += run_network(net, s, args.size, types, kg, gain_modes, horizons, args)

    if not rows:
        raise SystemExit("no rows produced")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path} ({len(rows)} rows)")

    summary = {"paired": paired_stats(rows, gain_modes, types, horizons),
               "predictors": predictor_stats(rows, gain_modes, horizons),
               "second_order": second_order_test(rows, gain_modes),
               "config": {"seeds": seeds, "k_grid": kg, "types": types,
                          "gain_modes": gain_modes, "horizons": horizons,
                          "n_randpart": args.n_randpart, "n_cond": args.n_cond,
                          "n_traj": args.n_traj}}
    jp = out_path.with_suffix(".json")
    jp.write_text(json.dumps(summary, indent=1))
    print(f"wrote {jp}")
    if missing:
        print("no cached sweep: " + ", ".join(missing))


if __name__ == "__main__":
    main()
