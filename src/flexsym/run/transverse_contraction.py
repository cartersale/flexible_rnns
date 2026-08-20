"""Is the transverse (quotient-complement) block of the Jacobian contracting?

The quotient story says the k-dimensional group-constant subspace carries the
computation. That is only half a claim: it is worth something only if what the
quotient throws away -- the (n-k)-dimensional transverse complement -- decays.
This script measures the transverse block

    P    = H (H'H)^-1 H'            projector onto group-constant directions
    Pi   = I - P                    transverse projector
    J    = (1-alpha) I + alpha diag(g) W
    Dt   = V' J V,   V (n x (n-k)) orthonormal, range(V) = range(Pi)

The restriction to V matters: building D = Pi J Pi as an n x n matrix gives it a
k-dimensional kernel whose zero eigenvalues are artifacts of the embedding, not
properties of the transverse dynamics. V comes from the full SVD of H (its last
n-k left singular vectors span range(Pi) exactly).

Four measurements per (network, k, surrogate type, gain mode):

  (i)   spectral radius        rho(Dt)                    -- asymptotic contraction
  (ii)  transient growth       peak = sup_t ||Dt^t||_2    -- computed with rescaled
                               powers so rho > 1 cases do not overflow
  (iii) departure from normality  Henrici  sqrt(||Dt||_F^2 - sum|lambda_i|^2),
                               reported raw and normalised by ||Dt||_F
  (iv)  quadratic certificate  Theta > 0 with Dt' Theta Dt - Theta < 0. Feasibility
                               is equivalent to rho < 1, so the informative number is
                               kappa(Theta): how far from normal the certificate has
                               to work. Two versions:
                                 kappa_gram  the observability Gramian
                                             (Dt' Theta Dt - Theta = -I), closed form,
                                             always computed; sup_t ||Dt^t||^2 <= kappa
                                 kappa_sdp   the kappa-MINIMISING Theta from an SDP
                                             (--sdp; minutes per cell, so it is opt-in
                                             and runs on a subgrid)

The interesting case is rho < 1 < peak: asymptotically contracting, transiently
amplifying.

GAIN MODES. The Jacobian needs gains. `unit` (g = 1) is the purely structural,
task-free version and is the cleanest comparison to the surrogates, but note that
these networks have ||W|| large and gains small (g ~ 0.03), so g = 1 puts rho(J)
around 11 and every matrix, trained or surrogate, is wildly expansive -- the
"is it below 1" question is vacuous there and only the paired trained-vs-surrogate
contrast means anything. The modes that make the absolute question meaningful:
  unit    g = 1                                        purely structural
  scalar  g = median recovered gain of this network    one number, still task-free
  full    g = per-unit gain averaged over conditions   task-averaged operating point
  cond    per-condition recovered gains, metrics summarised across conditions
Surrogate matrices are always paired with the SAME gain vector as their trained
network -- the surrogate manipulates connectivity, not gains.

SURROGATES AND PARTITIONS. Every surrogate gets a FRESH signed-WL partition fitted
on the surrogate itself at each k (via run/control_ksweep.make_matrix and
approx_wl_signed), never the trained labels: otherwise the comparison is
"trained D under a good partition" vs "shuffled D under a partition that does not
fit it", which is true by construction. `trained` is refitted the same way for a
matched procedure; `trained-cached` reuses the cached labels_struct_by_k for
continuity with the other analyses; `random-part` is the trained matrix with a
size-matched random partition, which separates "training makes transverse blocks
contracting" from "the connectivity partition specifically finds a contracting
complement".

ALSO RECORDED, per cell:
  eps    = ||Pi J P||_2  and  Bnorm = ||P J Pi||_2, giving the Theorem-A style bound
           eps * Bnorm / (1 - ||Dt||)  (finite only when ||Dt|| < 1), against a
           measured quotient prediction error from group-constant initial conditions
           -- an empirical check on how tight that bound is.
  gamma_signed of the partition on the matrix it was fitted to (raw and normalised
           by ||M||_F), so eps can be correlated against Gamma_signed across the sweep.

Writes results/main/signed_input_wl_paper/lumpability/transverse_contraction.csv and
a companion .json with the paired Wilcoxon summaries and the eps/Gamma correlation.

  conda run -n myenv python -m flexsym.run.transverse_contraction --seeds 3-12
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
from flexsym.motifs.motif_classify import npload
from flexsym.run.control_ksweep import make_matrix
from flexsym.run.motif_quotient_prediction import recover_alpha_gain
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.structural.signed_input_wl_structural_roles import (
    approx_wl_signed,
    gamma_signed,
    random_labels_same_sizes,
)

WL_ROOT = paths.RESULTS_ROOT / "signed_input_wl_paper"
FP_ROOT = paths.RESULTS_ROOT / "motif_fixed_points"
OUT = WL_ROOT / "lumpability" / "transverse_contraction.csv"

K_GRID = [8, 12, 16, 20, 25, 30, 40, 50]
#: `trained` and `random-part` are built here; the rest go through control_ksweep.
TYPES = ["trained", "trained-cached", "random-part", "row-shuffle", "full-shuffle",
         "gaussian", "lowrank-r5", "lowrank-r20", "spectrum"]
GAIN_MODES = ["unit", "scalar", "full", "cond"]
T_POWER = 200           # iterations for sup_t ||D^t||; alpha = 0.2 decays well inside this
N_RANDPART = 5          # size-matched random partitions of the trained matrix
N_COND = 8              # conditions subsampled for gain mode `cond`
N_TRAJ = 8              # group-constant initial conditions for the prediction error
ALPHA_FALLBACK = 0.2


# --------------------------------------------------------------------------- #
# partition geometry
# --------------------------------------------------------------------------- #
def indicator(labels: np.ndarray) -> np.ndarray:
    cats, inv = np.unique(labels, return_inverse=True)
    H = np.zeros((labels.size, cats.size))
    H[np.arange(labels.size), inv] = 1.0
    return H


def transverse_basis(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(V, U_par): orthonormal bases of range(Pi) and range(P) from the SVD of H.

    H has full column rank (every group is non-empty), so the first k left singular
    vectors span range(H) = range(P) and the remaining n-k span its complement.
    """
    U, s, _ = np.linalg.svd(H, full_matrices=True)
    k = H.shape[1]
    if s[k - 1] <= 1e-10 * s[0]:
        raise ValueError("indicator matrix is rank-deficient")
    return U[:, k:], U[:, :k]


# --------------------------------------------------------------------------- #
# the four measurements
# --------------------------------------------------------------------------- #
def power_envelope(D: np.ndarray, T: int) -> tuple[float, int]:
    """(sup_t ||D^t||_2 for t <= T, argmax t), computed in log space.

    The powers are rescaled to unit norm at every step and the scale carried in a
    log accumulator, so an expansive D (rho > 1, which g = 1 guarantees here) does
    not overflow.
    """
    M = np.eye(D.shape[0])
    log_scale = 0.0
    best, best_t = 0.0, 0          # t = 0 gives ||I|| = 1, i.e. log 0
    for t in range(1, T + 1):
        M = D @ M
        sv = svdvals(M)[0]
        if not np.isfinite(sv) or sv <= 0:
            break
        log_scale += float(np.log(sv))
        M /= sv
        if log_scale > best:
            best, best_t = log_scale, t
    return float(np.exp(min(best, 700.0))), best_t


def henrici(D: np.ndarray, lam: np.ndarray) -> tuple[float, float]:
    fro = float(np.linalg.norm(D, "fro"))
    dep = float(np.sqrt(max(fro ** 2 - float(np.sum(np.abs(lam) ** 2)), 0.0)))
    return dep, (dep / fro if fro > 0 else np.nan)


def gramian_certificate(D: np.ndarray) -> tuple[float, float]:
    """(kappa(Theta), lambda_max(Theta)) for the Gramian Theta = sum_t (D^t)' D^t.

    Solves D' Theta D - Theta = -I. Exists iff rho(D) < 1, and certifies
    sup_t ||D^t||^2 <= kappa(Theta). Not the kappa-minimising certificate (that is
    the SDP) but closed form and O(m^3).
    """
    try:
        Th = solve_discrete_lyapunov(D.T, np.eye(D.shape[0]))
    except Exception:
        return np.nan, np.nan
    ev = np.linalg.eigvalsh(0.5 * (Th + Th.T))
    if not np.all(np.isfinite(ev)) or ev.min() <= 0:
        return np.nan, np.nan
    return float(ev.max() / ev.min()), float(ev.max())


def sdp_kappa(D: np.ndarray, margin: float = 1e-4, solver: str = "SCS") -> float:
    """min kappa(Theta) over Theta with D' Theta D - Theta < 0, via
    minimise t s.t. I <= Theta <= t I, D' Theta D - Theta <= -margin I.

    Minutes per call at n-k ~ 200, hence opt-in.
    """
    try:
        import cvxpy as cp
    except ImportError:
        return np.nan
    m = D.shape[0]
    t = cp.Variable()
    Th = cp.Variable((m, m), symmetric=True)
    cons = [Th >> np.eye(m), Th << t * np.eye(m),
            D.T @ Th @ D - Th << -margin * np.eye(m)]
    try:
        prob = cp.Problem(cp.Minimize(t), cons)
        prob.solve(solver=solver, verbose=False)
    except Exception:
        return np.nan
    if Th.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return np.nan
    ev = np.linalg.eigvalsh(0.5 * (Th.value + Th.value.T))
    return float(ev.max() / ev.min()) if ev.min() > 0 else np.nan


# --------------------------------------------------------------------------- #
# Theorem-A ingredients + empirical quotient prediction error
# --------------------------------------------------------------------------- #
def theorem_a_probe(J: np.ndarray, H: np.ndarray, V: np.ndarray, Upar: np.ndarray,
                    T: int, n_traj: int, rng: np.random.Generator) -> dict:
    """Empirical counterpart of the eps ||B|| / (1 - ||D||) bound.

    In (parallel, transverse) coordinates y = Upar' x, z = V' x the dynamics are
    y+ = A y + B z and z+ = E y + D z with ||E|| = eps. Starting on the
    group-constant subspace (z_0 = 0), the theorem bounds the transverse-to-parallel
    ratio ||z_t|| / ||y_t|| by eps / (1 - ||D||) and hence the perturbation the
    transverse component feeds back into the parallel dynamics, ||B|| ||z_t|| / ||y_t||,
    by eps ||B|| / (1 - ||D||). Both measured quantities are returned so the bound can
    be compared to what actually happens.

    Also returned: the relative deviation between the true parallel component and the
    quotient's own trajectory, ||P x_t - H q_t|| / ||P x_t|| (what the quotient
    prediction is worth), and the same normalised by the full state ||x_t||.
    """
    sizes = H.sum(0)
    Q = (H.T @ J @ H) / sizes[:, None]
    marks = [1, 5, 20]
    ratio, par_err, tot_err = [], [], []
    at_r = {t: [] for t in marks}
    at_e = {t: [] for t in marks}
    for _ in range(n_traj):
        c = rng.normal(size=H.shape[1])
        x = H @ c
        nx0 = np.linalg.norm(x)
        if nx0 <= 0:
            continue
        x, c = x / nx0, c / nx0
        r_i, pe_i, te_i = 0.0, 0.0, 0.0
        for t in range(1, T + 1):
            x = J @ x
            c = Q @ c
            nx = np.linalg.norm(x)
            if not np.isfinite(nx) or nx <= 1e-300:
                break
            y = Upar.T @ x
            z = V.T @ x
            ny = np.linalg.norm(y)
            r_t = float(np.linalg.norm(z) / ny) if ny > 1e-300 else np.nan
            e_t = float(np.linalg.norm(Upar.T @ (x - H @ c)) / ny) if ny > 1e-300 else np.nan
            if np.isfinite(r_t):
                r_i = max(r_i, r_t)
                pe_i = max(pe_i, e_t)
            te_i = max(te_i, float(np.linalg.norm(x - H @ c) / nx))
            if t in at_r:
                at_r[t].append(r_t)
                at_e[t].append(e_t)
        ratio.append(r_i)
        par_err.append(pe_i)
        tot_err.append(te_i)
    if not ratio:
        keys = ["ratio_max", "err_par", "err_tot"] + [f"{p}{t}" for t in marks
                                                      for p in ("ratio_t", "err_par_t")]
        return {k: np.nan for k in keys}
    out = {"ratio_max": float(np.mean(ratio)), "err_par": float(np.mean(par_err)),
           "err_tot": float(np.mean(tot_err))}
    for t in marks:
        out[f"ratio_t{t}"] = float(np.nanmean(at_r[t])) if at_r[t] else np.nan
        out[f"err_par_t{t}"] = float(np.nanmean(at_e[t])) if at_e[t] else np.nan
    return out


def measure(M: np.ndarray, labels: np.ndarray, g: np.ndarray, alpha: float,
            T: int, rng: np.random.Generator, want_sdp: bool,
            n_traj: int) -> dict:
    n = M.shape[0]
    H = indicator(labels)
    V, Upar = transverse_basis(H)
    J = (1 - alpha) * np.eye(n) + alpha * (g[:, None] * M)
    D = V.T @ J @ V

    lam = np.linalg.eigvals(D)
    rho = float(np.max(np.abs(lam)))
    nrm = float(svdvals(D)[0])
    peak, t_peak = power_envelope(D, T)
    dep, dep_n = henrici(D, lam)
    kappa_g, gram_lmax = gramian_certificate(D) if rho < 1 else (np.nan, np.nan)
    kappa_s = sdp_kappa(D) if (want_sdp and rho < 1) else np.nan

    eps = float(svdvals(V.T @ J @ Upar)[0])
    bnorm = float(svdvals(Upar.T @ J @ V)[0])
    bound = eps * bnorm / (1 - nrm) if nrm < 1 else np.nan
    # certified version: in the Theta-norm of the Gramian, ||D||_Theta =
    # sqrt(1 - 1/lambda_max(Theta)) < 1 even when ||D||_2 > 1, and the uniform
    # bound picks up the advertised kappa(Theta)^(1/2).
    bound_theta = np.nan
    if np.isfinite(kappa_g):
        lmax = gram_lmax
        if np.isfinite(lmax) and lmax > 1:
            d_theta = float(np.sqrt(1.0 - 1.0 / lmax))
            bound_theta = np.sqrt(kappa_g) * eps * bnorm / (1 - d_theta)
    probe = theorem_a_probe(J, H, V, Upar, T, n_traj, rng)

    lam_J = np.linalg.eigvals(J)
    sizes = H.sum(0)
    lam_Q = np.linalg.eigvals((H.T @ J @ H) / sizes[:, None])
    return {
        "rho": rho, "norm2": nrm, "peak": peak, "t_peak": t_peak,
        "transient": int(rho < 1 and peak > 1 + 1e-9),
        "contracting": int(rho < 1),
        "henrici": dep, "henrici_norm": dep_n,
        "kappa_gram": kappa_g, "kappa_sdp": kappa_s,
        "eps": eps, "b_norm": bnorm, "bound": bound, "bound_theta": bound_theta,
        "ratio_bound": (eps / (1 - nrm)) if nrm < 1 else np.nan,
        **probe,
        "bound_over_meas": (bound / (bnorm * probe["ratio_max"]))
                           if (np.isfinite(bound) and probe["ratio_max"] > 0) else np.nan,
        "bound_theta_over_meas": (bound_theta / (bnorm * probe["ratio_max"]))
                                 if (np.isfinite(bound_theta) and probe["ratio_max"] > 0)
                                 else np.nan,
        "rho_J": float(np.max(np.abs(lam_J))),
        "rho_Q": float(np.max(np.abs(lam_Q))),
    }


# --------------------------------------------------------------------------- #
# per network
# --------------------------------------------------------------------------- #
def load_network(net: str) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """(W, labels_by_k, alpha, gain_matrix (n_cond x n), cached label table)."""
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], dtype=float)
    ind = np.asarray(wl["ind_active"], dtype=int)
    ls = wl["labels_struct_by_k"]

    cache = FP_ROOT / net / "jbar_cache.npz"
    alpha, G = ALPHA_FALLBACK, None
    if cache.exists():
        d = npload(cache)
        if np.array_equal(np.asarray(d["ind_active"], dtype=int), ind):
            alphas, gains = [], []
            for Jb in d["jbars"]:
                a, g = recover_alpha_gain(np.asarray(Jb, dtype=float), W)
                alphas.append(a)
                gains.append(g)
            if alphas:
                alpha = float(np.median(alphas))
                G = np.asarray(gains, dtype=float)
    return W, ls, alpha, G, ind


def gain_vectors(mode: str, G: np.ndarray | None, n: int, n_cond: int) -> list[np.ndarray]:
    if mode == "unit":
        return [np.ones(n)]
    if G is None:
        return []
    if mode == "scalar":
        return [np.full(n, float(np.median(G)))]
    if mode == "full":
        return [G.mean(0)]
    if mode == "cond":
        idx = np.linspace(0, len(G) - 1, min(n_cond, len(G))).astype(int)
        return [G[i] for i in np.unique(idx)]
    raise ValueError(mode)


def summarise(cells: list[dict]) -> dict:
    """Collapse the per-condition cells of gain mode `cond` into one row."""
    if len(cells) == 1:
        return dict(cells[0], n_cells=1)
    out = {"n_cells": len(cells)}
    for key in cells[0]:
        vals = np.array([c[key] for c in cells], dtype=float)
        if not np.any(np.isfinite(vals)):
            out[key] = np.nan
        elif key in ("transient", "contracting"):
            out[key] = float(np.nanmean(vals))       # fraction of conditions
        else:
            out[key] = float(np.nanmedian(vals))
    return out


def run_network(net: str, seed: int, size: int, types: list[str], kg: list[int],
                gain_modes: list[str], args) -> list[dict]:
    W, ls_cached, alpha, G, _ind = load_network(net)
    n = W.shape[0]
    rows = []
    if G is None and any(m != "unit" for m in gain_modes):
        print(f"  [{net}] no usable jbar cache -- only the `unit` gain mode", flush=True)

    for kind in types:
        # matrix + label source per surrogate type
        rng_m = np.random.default_rng(abs(hash((seed, size, kind))) % (2 ** 32))
        base = kind.split("::")[0]
        if base in ("trained", "trained-cached", "random-part"):
            M = W
        else:
            M = make_matrix(W, base, rng_m)
        Mpos, Mneg = np.maximum(M, 0.0), np.maximum(-M, 0.0)
        fro_M = float(np.linalg.norm(M))

        for k in kg:
            label_sets = []
            if base == "trained-cached":
                lab = np.asarray(ls_cached[_row_for_k(ls_cached, k)], dtype=int)
                label_sets = [(0, lab)]
            elif base == "random-part":
                lab = np.asarray(ls_cached[_row_for_k(ls_cached, k)], dtype=int)
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
                    rng = np.random.default_rng(abs(hash((seed, kind, k, draw, mode)))
                                                % (2 ** 32))
                    want_sdp = (args.sdp and base in args.sdp_types.split(",")
                                and k in [int(x) for x in args.sdp_k.split(",")]
                                and mode in args.sdp_modes.split(","))
                    cells = [measure(M, labels, g, alpha, args.T, rng, want_sdp,
                                     args.n_traj) for g in gs]
                    rows.append({
                        "model": net, "size": size, "seed": seed, "type": kind,
                        "draw": draw, "k_req": k, "k": k_real, "n_active": n,
                        "gain_mode": mode, "alpha": alpha,
                        "gamma_signed": gam, "gamma_signed_norm": gam / fro_M,
                        **summarise(cells),
                    })
            done = [r for r in rows if r["type"] == kind and r["k_req"] == k]
            if done:
                r0 = [r for r in done if r["gain_mode"] == gain_modes[0]]
                if r0:
                    r0 = r0[0]
                    print(f"    {kind:14s} k={r0['k']:3d} [{gain_modes[0]}] "
                          f"rho={r0['rho']:.3f} peak={r0['peak']:.3g} "
                          f"henr={r0['henrici_norm']:.3f} eps={r0['eps']:.3g}",
                          flush=True)
    return rows


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
METRICS = ["rho", "norm2", "peak", "henrici_norm", "kappa_gram", "eps", "b_norm",
           "contracting", "transient", "ratio_max", "err_par_t5", "err_par_t20"]


def paired_stats(rows: list[dict], gain_modes: list[str], types: list[str]) -> dict:
    """Trained vs every other type, paired within (network, k), seed as the unit."""
    out = {}
    ks = sorted({r["k_req"] for r in rows})
    for mode in gain_modes:
        for k in ks:
            sub = [r for r in rows if r["gain_mode"] == mode and r["k_req"] == k]
            if not sub:
                continue
            by_type = {}
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
                for met in METRICS:
                    a = np.array([np.nanmean([r[met] for r in by_type["trained"][s]])
                                  for s in common], dtype=float)
                    b = np.array([np.nanmean([r[met] for r in by_type[other][s]])
                                  for s in common], dtype=float)
                    ok = np.isfinite(a) & np.isfinite(b)
                    if ok.sum() < 5 or np.allclose(a[ok], b[ok]):
                        rec[met] = {"trained": float(np.nanmedian(a)),
                                    "other": float(np.nanmedian(b)),
                                    "p": np.nan, "n_pairs": int(ok.sum())}
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        stat, p = wilcoxon(a[ok], b[ok])
                    rec[met] = {"trained": float(np.median(a[ok])),
                                "other": float(np.median(b[ok])),
                                "delta": float(np.median(a[ok] - b[ok])),
                                "p": float(p), "n_pairs": int(ok.sum())}
                out[f"{mode}|k{k}|trained_vs_{other}"] = rec
    return out


def eps_gamma_corr(rows: list[dict], gain_modes: list[str]) -> dict:
    """Spearman eps vs Gamma_signed across the sweep, trained networks only."""
    out = {}
    for mode in gain_modes:
        sub = [r for r in rows if r["gain_mode"] == mode and r["type"] == "trained"]
        if len(sub) < 6:
            continue
        for gkey in ("gamma_signed", "gamma_signed_norm"):
            x = np.array([r[gkey] for r in sub], float)
            y = np.array([r["eps"] for r in sub], float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 6:
                continue
            rho_s, p = spearmanr(x[ok], y[ok])
            out[f"{mode}|{gkey}|pooled"] = {"rho": float(rho_s), "p": float(p),
                                            "n": int(ok.sum())}
        # within-k (removes the k trend that drives both)
        per_k = {}
        for k in sorted({r["k_req"] for r in sub}):
            s = [r for r in sub if r["k_req"] == k]
            x = np.array([r["gamma_signed"] for r in s], float)
            y = np.array([r["eps"] for r in s], float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 6:
                rho_s, p = spearmanr(x[ok], y[ok])
                per_k[str(k)] = {"rho": float(rho_s), "p": float(p), "n": int(ok.sum())}
        if per_k:
            out[f"{mode}|gamma_signed|within_k"] = per_k
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seeds", default="3-12")
    ap.add_argument("--k-grid", default=",".join(str(k) for k in K_GRID))
    ap.add_argument("--types", default=",".join(TYPES))
    ap.add_argument("--gain-modes", default=",".join(GAIN_MODES))
    ap.add_argument("--T", type=int, default=T_POWER)
    ap.add_argument("--n-randpart", type=int, default=N_RANDPART)
    ap.add_argument("--n-cond", type=int, default=N_COND)
    ap.add_argument("--n-traj", type=int, default=N_TRAJ)
    ap.add_argument("--sdp", action="store_true",
                    help="Also solve the kappa-minimising SDP (slow; subgrid only).")
    ap.add_argument("--sdp-types", default="trained,random-part,row-shuffle")
    ap.add_argument("--sdp-k", default="20")
    ap.add_argument("--sdp-modes", default="full")
    ap.add_argument("--out", default=None, help="Override the output CSV path.")
    args = ap.parse_args()

    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    kg = [int(x) for x in args.k_grid.split(",")]
    types = args.types.split(",")
    gain_modes = args.gain_modes.split(",")
    out_path = paths.REPO_ROOT / args.out if args.out else OUT

    rows, missing = [], []
    t0 = time.time()
    for s in seeds:
        net = f"LeakyRNN_softplus_diag_{args.size}_all_{s}"
        if not (WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz").exists():
            missing.append(net)
            continue
        print(f"[{time.time() - t0:6.0f}s] {net}", flush=True)
        rows += run_network(net, s, args.size, types, kg, gain_modes, args)

    if not rows:
        raise SystemExit("no rows produced")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path} ({len(rows)} rows)")

    summary = {"paired": paired_stats(rows, gain_modes, types),
               "eps_vs_gamma": eps_gamma_corr(rows, gain_modes),
               "config": {"seeds": seeds, "k_grid": kg, "types": types,
                          "gain_modes": gain_modes, "T": args.T,
                          "n_randpart": args.n_randpart, "n_cond": args.n_cond}}
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=1))
    print(f"wrote {json_path}")
    if missing:
        print("no cached sweep: " + ", ".join(missing))


if __name__ == "__main__":
    main()
