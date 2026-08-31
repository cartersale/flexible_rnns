"""Relative leakage: is the partition nearly invariant, or merely low-defect?

`quotient_error_budget.py` found that the structural partition has LARGER
absolute leakage ||Pi M P||_2 than a size-matched random partition of the same
matrix, while leaving a far more contracting transverse block. That ranking is
suspicious, and the mechanism that explains it also predicts the fix.

THE MECHANISM. Gamma_signed groups units by their aggregate input profiles, i.e.
it makes rows of M similar within cells. If rows i and j are similar then
(Mx)_i ~ (Mx)_j for EVERY state x, not only cell-constant ones. So the objective
shrinks Pi M as a whole operator, and Pi M contains both B = Pi M P and
D = Pi M Pi -- D was never targeted, it came along. The stronger version is
spectral: the leading eigenvectors of M are approximately piecewise-constant on
these roles (the step-R^2 result), so col(H) is aligned with the DOMINANT
eigenspace and the complement is left holding the tail of the spectrum. D
contracts because the partition put the small modes there.

THE PREDICTION TESTED HERE. If col(H) spans high-gain directions, then everything
downstream of it is large, leaked energy included. Comparing absolute ||Pi M P||
across partitions compares leakage from a high-gain subspace against leakage from
a typical-gain one. The scale-free measure is RELATIVE leakage

    rel = ||Pi M P|| / ||M P||        (operator and Frobenius versions)

i.e. what FRACTION of what the partition-respecting subspace produces lands
outside it -- subspace invariance rather than raw defect. The structural
partition should beat the random one here even where it loses on ||B||_2, and
the ranking inversion should largely vanish. If it does not, the spectral-
alignment story is wrong and the contraction has some other source.

TWO SUPPORTING QUANTITIES, because the ratio alone is ambiguous. A ratio can fall
either because the partition is genuinely more invariant or merely because its
denominator is inflated, and only the second is an artifact. So we also measure
the mechanism directly:

  gain_ratio  (||M P||_F / sqrt(k)) / (||M Pi||_F / sqrt(n-k)), the rms gain of
              the parallel subspace against the transverse one. The spectral
              story says this is > 1 for the structural partition and ~1 for a
              random one (a random subspace samples a typical mixture).
  align_topk  mean squared cosine of the principal angles between col(H) and the
              span of the top-k eigenvectors of M, against the k/n baseline a
              random subspace of the same dimension achieves. This is the
              alignment claim itself, stated as a subspace overlap.

All three should move together if the mechanism holds; rel improving while
gain_ratio and align_topk sit at their random-subspace baselines would mean the
normalisation fixed the ranking for reasons other than the stated one.

Computed for the structural matrix M (gain-free, the object Gamma is fitted to)
and for the Jacobian J = (1-a) I + a diag(g) M at the operating gains. Note the
identity part of J is partition-respecting, so it enters the denominator only.
Surrogates and partition-fitting follow transverse_contraction.py exactly.

Cheap: SVDs only, no matrix powers and no Lyapunov solves.

  conda run -n myenv python -m flexsym.run.relative_leakage --seeds 3-12

Writes results/main/signed_input_wl_paper/lumpability/relative_leakage.csv + .json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings

import numpy as np
from scipy.linalg import svdvals
from scipy.stats import wilcoxon

from flexsym import paths
from flexsym.run.control_ksweep import make_matrix
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.run.transverse_contraction import (
    WL_ROOT,
    gain_vectors,
    indicator,
    load_network,
    transverse_basis,
)
from flexsym.structural.signed_input_wl_structural_roles import (
    approx_wl_signed,
    gamma_signed,
    random_labels_same_sizes,
)

OUT = WL_ROOT / "lumpability" / "relative_leakage.csv"

K_GRID = [8, 12, 16, 20, 25, 30, 40, 50]
TYPES = ["trained", "trained-cached", "random-part", "row-shuffle", "spectrum",
         "gaussian"]
GAIN_MODES = ["structural", "full", "cond"]     # `structural` = M itself, no gains
N_RANDPART = 5
N_COND = 8


def top_eig_subspace(M: np.ndarray, m: int) -> np.ndarray:
    """Orthonormal real basis for the span of the m leading eigenvectors of M.

    Complex conjugate pairs contribute their real and imaginary parts, so the
    span is real and its dimension can exceed m by one when the cut falls inside
    a pair; the basis is orthonormalised and truncated to keep the comparison at
    dimension m.
    """
    lam, vec = np.linalg.eig(M)
    order = np.argsort(-np.abs(lam))
    cols = []
    for idx in order:
        v = vec[:, idx]
        if np.max(np.abs(v.imag)) < 1e-12:
            cols.append(v.real)
        else:
            cols.append(v.real)
            cols.append(v.imag)
        if len(cols) >= m:
            break
    A = np.column_stack(cols[:max(m, 1)])
    Q, _ = np.linalg.qr(A)
    return Q[:, :m]


def subspace_overlap(Upar: np.ndarray, E: np.ndarray) -> float:
    """Mean squared cosine of principal angles between col(Upar) and col(E).

    Both bases orthonormal. Equals ||Upar' E||_F^2 / dim(E); for a random
    subspace of dimension k in n dimensions the expectation is k/n.
    """
    if E.shape[1] == 0:
        return np.nan
    return float(np.linalg.norm(Upar.T @ E, "fro") ** 2 / E.shape[1])


def leakage(A: np.ndarray, V: np.ndarray, Upar: np.ndarray) -> dict:
    """Absolute and relative leakage of the operator A across the partition split.

    A P is represented as A @ Upar (n x k): what the partition-respecting subspace
    produces. Its transverse part is V' A Upar. The relative measures divide by the
    total, making them invariant to the gain of the subspace being left.
    """
    AP = A @ Upar                       # n x k
    APi = A @ V                         # n x (n-k)
    B = V.T @ AP                        # leak out
    C = Upar.T @ APi                    # leak back
    nAP2, nAPF = float(svdvals(AP)[0]), float(np.linalg.norm(AP, "fro"))
    nAPi2, nAPiF = float(svdvals(APi)[0]), float(np.linalg.norm(APi, "fro"))
    b2, bF = float(svdvals(B)[0]), float(np.linalg.norm(B, "fro"))
    c2, cF = float(svdvals(C)[0]), float(np.linalg.norm(C, "fro"))
    k, m = Upar.shape[1], V.shape[1]
    return {
        "B2": b2, "BF": bF, "C2": c2, "CF": cF,
        "gain_par_2": nAP2, "gain_par_F": nAPF, "gain_perp_2": nAPi2,
        "gain_perp_F": nAPiF,
        "rel_B2": b2 / nAP2 if nAP2 > 0 else np.nan,
        "rel_BF": bF / nAPF if nAPF > 0 else np.nan,
        "rel_C2": c2 / nAPi2 if nAPi2 > 0 else np.nan,
        "rel_CF": cF / nAPiF if nAPiF > 0 else np.nan,
        "gain_par_rms": nAPF / np.sqrt(k),
        "gain_perp_rms": nAPiF / np.sqrt(m),
        "gain_ratio": (nAPF / np.sqrt(k)) / (nAPiF / np.sqrt(m))
                      if nAPiF > 0 else np.nan,
    }


def summarise(cells: list[dict]) -> dict:
    if len(cells) == 1:
        return dict(cells[0], n_cells=1)
    out = {"n_cells": len(cells)}
    for key in cells[0]:
        vals = np.array([c[key] for c in cells], float)
        out[key] = float(np.nanmedian(vals)) if np.any(np.isfinite(vals)) else np.nan
    return out


def run_network(net: str, seed: int, size: int, types: list[str], kg: list[int],
                gain_modes: list[str], args) -> list[dict]:
    W, ls_cached, alpha_leak, G, _ind = load_network(net)
    n = W.shape[0]
    rows = []

    for kind in types:
        rng_m = np.random.default_rng(abs(hash((seed, size, kind))) % (2 ** 32))
        base = kind.split("::")[0]
        M = W if base in ("trained", "trained-cached", "random-part") \
            else make_matrix(W, base, rng_m)
        Mpos, Mneg = np.maximum(M, 0.0), np.maximum(-M, 0.0)
        fro_M = float(np.linalg.norm(M))
        eig_cache: dict[int, np.ndarray] = {}

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
                V, Upar = transverse_basis(H)
                gam = gamma_signed(Mpos, Mneg, H)
                if k_real not in eig_cache:
                    eig_cache[k_real] = top_eig_subspace(M, k_real)
                align = subspace_overlap(Upar, eig_cache[k_real])
                for mode in gain_modes:
                    if mode == "structural":
                        cells = [leakage(M, V, Upar)]
                    else:
                        gs = gain_vectors(mode, G, n, args.n_cond)
                        if not gs:
                            continue
                        cells = [leakage((1 - alpha_leak) * np.eye(n)
                                         + alpha_leak * (g[:, None] * M), V, Upar)
                                 for g in gs]
                    rows.append({
                        "model": net, "size": size, "seed": seed, "type": kind,
                        "draw": draw, "k_req": k, "k": k_real, "n_active": n,
                        "gain_mode": mode, "alpha_leak": alpha_leak,
                        "gamma_signed": gam, "gamma_signed_norm": gam / fro_M,
                        "align_topk": align, "align_baseline": k_real / n,
                        "align_excess": align - k_real / n,
                        **summarise(cells),
                    })
            done = [r for r in rows if r["type"] == kind and r["k_req"] == k
                    and r["gain_mode"] == gain_modes[0]]
            if done:
                r0 = done[0]
                print(f"    {kind:14s} k={r0['k']:3d} [{gain_modes[0]}] "
                      f"B2={r0['B2']:.3g} relB2={r0['rel_B2']:.3f} "
                      f"relBF={r0['rel_BF']:.3f} gain_ratio={r0['gain_ratio']:.3f} "
                      f"align={r0['align_topk']:.3f} (base {r0['align_baseline']:.3f})",
                      flush=True)
    return rows


METRICS = ["B2", "BF", "rel_B2", "rel_BF", "C2", "rel_C2", "rel_CF",
           "gain_par_rms", "gain_perp_rms", "gain_ratio", "gain_par_2",
           "align_topk", "align_excess", "gamma_signed"]


def paired_stats(rows: list[dict], gain_modes: list[str], types: list[str]) -> dict:
    out = {}
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
                for met in METRICS:
                    a = np.array([np.nanmean([r[met] for r in by_type["trained"][s]])
                                  for s in common], float)
                    b = np.array([np.nanmean([r[met] for r in by_type[other][s]])
                                  for s in common], float)
                    ok = np.isfinite(a) & np.isfinite(b)
                    if ok.sum() < 5 or np.allclose(a[ok], b[ok]):
                        rec[met] = {"trained": float(np.nanmedian(a)),
                                    "other": float(np.nanmedian(b)), "p": np.nan}
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _s, p = wilcoxon(a[ok], b[ok])
                    rec[met] = {"trained": float(np.median(a[ok])),
                                "other": float(np.median(b[ok])),
                                "delta": float(np.median(a[ok] - b[ok])),
                                "p": float(p), "n_pairs": int(ok.sum()),
                                "win": int(np.sum(a[ok] < b[ok]))}
                out[f"{mode}|k{k}|trained_vs_{other}"] = rec
    return out


def inversion_table(rows: list[dict], gain_modes: list[str]) -> dict:
    """The headline: does the trained-vs-random-partition ranking flip when the
    leakage is normalised? Reports, per (gain mode, k), the fraction of networks
    on which trained has the SMALLER value -- for the raw defect and the relative
    one. The prediction is a fraction well below 0.5 for B2 and well above for
    rel_B*.
    """
    out = {}
    for mode in gain_modes:
        for k in sorted({r["k_req"] for r in rows}):
            tr = {}
            rp = {}
            for r in rows:
                if r["gain_mode"] != mode or r["k_req"] != k:
                    continue
                if r["type"] == "trained":
                    tr[r["seed"]] = r
                elif r["type"] == "random-part":
                    rp.setdefault(r["seed"], []).append(r)
            common = sorted(set(tr) & set(rp))
            if len(common) < 5:
                continue
            rec = {"n": len(common)}
            for met in ("B2", "BF", "rel_B2", "rel_BF", "gain_ratio", "align_topk",
                        "gamma_signed"):
                a = np.array([tr[s][met] for s in common], float)
                b = np.array([np.nanmean([r[met] for r in rp[s]]) for s in common],
                             float)
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() < 5:
                    continue
                rec[met] = {"trained_med": float(np.median(a[ok])),
                            "randpart_med": float(np.median(b[ok])),
                            "frac_trained_lower": float(np.mean(a[ok] < b[ok]))}
            out[f"{mode}|k{k}"] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seeds", default="3-12")
    ap.add_argument("--k-grid", default=",".join(str(k) for k in K_GRID))
    ap.add_argument("--types", default=",".join(TYPES))
    ap.add_argument("--gain-modes", default=",".join(GAIN_MODES))
    ap.add_argument("--n-randpart", type=int, default=N_RANDPART)
    ap.add_argument("--n-cond", type=int, default=N_COND)
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
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path} ({len(rows)} rows)")

    summary = {"paired": paired_stats(rows, gain_modes, types),
               "inversion": inversion_table(rows, gain_modes),
               "config": {"seeds": seeds, "k_grid": kg, "types": types,
                          "gain_modes": gain_modes, "n_randpart": args.n_randpart,
                          "n_cond": args.n_cond}}
    jp = out_path.with_suffix(".json")
    jp.write_text(json.dumps(summary, indent=1))
    print(f"wrote {jp}")
    if missing:
        print("no cached sweep: " + ", ".join(missing))


if __name__ == "__main__":
    main()
