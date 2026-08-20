"""Are the modes the partition routes into the quotient the modes the motifs live in?

run/transverse_contraction.py established the routing half: the full per-condition
Jacobian is marginally expansive (rho ~ 1.05), and the signed-WL partition puts the
marginal modes in the quotient block and leaves a contracting transverse complement,
where a size-matched random partition of the same matrix leaves rho ~ rho(J) behind.
That says the partition sorts modes by timescale. It does not say the modes it keeps
are the ones the computation uses.

This script asks that, mode by mode. For each network, each k, each motif condition:

    lam_i, u_i      eigenmodes of the condition's mean Jacobian Jbar
    c_i             containment  ||P u_i|| / ||u_i||     -- how much of mode i the
                                                            quotient keeps (1 = fully
                                                            routed into the quotient,
                                                            0 = purely transverse)
    a_i             alignment    |u_i^H v| / (||u_i|| ||v||)  with the motif direction v
    w_i             modal weight |c_i| ||u_i|| / ||v|| from the left-eigenvector
                                 expansion v = sum_i c_i u_i -- the actual modal content
                                 of v, which the cosine overstates when modes are
                                 near-degenerate (these operators are non-normal)

and reports, across the n modes:

  rho(c, a)         do quotient-routed modes carry the motif direction?
  rho(c, |lam|)     does the partition route by timescale?  (the routing result again,
                    now mode-resolved rather than as a block spectral radius)
  rho(a, |lam|)     do motifs live in slow modes?
  rho(c, a | |lam|) partial, ranks, |lam| regressed out. THE number: if routing and motif
                    alignment are both just "slow modes", this goes to zero and the
                    partition is not doing anything beyond a timescale filter. If it
                    stays positive, the partition keeps the computational modes
                    specifically.

Plus a subspace version that does not depend on a per-mode statistic: take the k modes
with the highest containment (the quotient-routed set), orthonormalise, and measure the
motif direction's alignment with their span -- against the top-k by |lam| and against k
modes chosen at random.

NULLS. Two, both paired within (network, k, condition):
  rand-part   containment recomputed under a size-matched random partition, same modes,
              same v. Separates "the connectivity partition routes the motif modes" from
              "any k-dimensional group-constant subspace would".
  cross-cond  the motif direction of a DIFFERENT condition of the same network, same
              modes, same partition. Separates a real condition-specific correspondence
              from the fact that all motif directions look somewhat alike.

Writes results/main/signed_input_wl_paper/lumpability/quotient_mode_motif_alignment.csv
(one row per network x k x condition) and a .json with the paired Wilcoxon summaries.

  conda run -n myenv python -m flexsym.run.quotient_mode_motif_alignment --seeds 3-12
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_quotient_prediction import (
    condition_jacobians, indicator, load_conditions,
)
from flexsym.run.motif_reuse_control import MOTIF_TYPES, _row_for_k
from flexsym.structural.signed_input_wl_structural_roles import random_labels_same_sizes

OUT = paths.RESULTS_ROOT / "signed_input_wl_paper/lumpability/quotient_mode_motif_alignment.csv"
K_GRID = [8, 12, 16, 20, 25, 30, 40, 50]
TOP_R = 10          # "top modes by alignment" set size for the coarse comparison


# --------------------------------------------------------------------------- #
# per-mode quantities
# --------------------------------------------------------------------------- #
def mode_decomposition(J: np.ndarray) -> dict:
    """Right eigenmodes of J, their |lambda|, and the left-eigenvector dual basis.

    v = sum_i coeff_i u_i with coeff_i = (w_i^H v) / (w_i^H u_i), w_i the left
    eigenvectors (rows of the inverse eigenvector matrix), so the modal weight of a
    direction in mode i is |coeff_i| ||u_i||.
    """
    lam, U = np.linalg.eig(J)
    norms = np.linalg.norm(U, axis=0)
    try:
        Winv = np.linalg.inv(U)          # rows are the dual (left) vectors
    except np.linalg.LinAlgError:
        Winv = np.linalg.pinv(U)
    return {"lam": lam, "U": U, "norms": norms, "Winv": Winv}


def containment(md: dict, P_apply) -> np.ndarray:
    """||P u_i|| / ||u_i|| per mode, for a projector applied as a callable."""
    PU = P_apply(md["U"])
    return np.asarray(np.linalg.norm(PU, axis=0) / np.maximum(md["norms"], 1e-300),
                      dtype=float)


def alignment(md: dict, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(cosine alignment, modal weight) of v with each eigenmode."""
    nv = np.linalg.norm(v)
    cos = np.abs(md["U"].conj().T @ v) / np.maximum(md["norms"] * nv, 1e-300)
    coeff = md["Winv"] @ v
    weight = np.abs(coeff) * md["norms"] / max(nv, 1e-300)
    return np.asarray(cos, dtype=float), np.asarray(weight, dtype=float)


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman(x, y) with z partialled out, on ranks."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 6:
        return np.nan
    from scipy.stats import rankdata
    rx, ry, rz = (rankdata(a[ok]) for a in (x, y, z))
    Z = np.column_stack([np.ones_like(rz), rz])
    bx = np.linalg.lstsq(Z, rx, rcond=None)[0]
    by = np.linalg.lstsq(Z, ry, rcond=None)[0]
    ex, ey = rx - Z @ bx, ry - Z @ by
    if np.std(ex) == 0 or np.std(ey) == 0:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def span_alignment(md: dict, idx: np.ndarray, v: np.ndarray) -> float:
    """||B' v|| / ||v|| for B an orthonormal basis of the real span of modes `idx`."""
    U = md["U"][:, idx]
    B = np.hstack([np.real(U), np.imag(U)])
    keep = np.linalg.norm(B, axis=0) > 1e-12
    B = B[:, keep]
    if not B.shape[1]:
        return np.nan
    Q, R = np.linalg.qr(B)
    Q = Q[:, np.abs(np.diag(R)) > 1e-10]
    if not Q.shape[1]:
        return np.nan
    return float(np.linalg.norm(Q.T @ v) / np.linalg.norm(v))


# --------------------------------------------------------------------------- #
def projector_apply(labels: np.ndarray):
    """P = H (H'H)^-1 H' as a callable, without forming the n x n matrix."""
    H, sizes = indicator(labels)
    def apply(X):
        return H @ ((H.T @ X) / sizes[:, None])
    return apply


def one_cell(md: dict, v: np.ndarray, v_cross: np.ndarray | None,
             P_obs, P_rand, k: int, rng: np.random.Generator) -> dict:
    lam_abs = np.abs(md["lam"])
    c_obs = containment(md, P_obs)
    c_rnd = containment(md, P_rand)
    a_cos, a_w = alignment(md, v)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = {
            "rho_c_a": float(spearmanr(c_obs, a_cos).statistic),
            "rho_c_w": float(spearmanr(c_obs, a_w).statistic),
            "rho_c_lam": float(spearmanr(c_obs, lam_abs).statistic),
            "rho_a_lam": float(spearmanr(a_cos, lam_abs).statistic),
            "rho_c_a_null_part": float(spearmanr(c_rnd, a_cos).statistic),
        }
    rec["rho_c_a_partial_lam"] = partial_spearman(c_obs, a_cos, lam_abs)
    rec["rho_c_w_partial_lam"] = partial_spearman(c_obs, a_w, lam_abs)
    rec["rho_c_a_null_part_partial"] = partial_spearman(c_rnd, a_cos, lam_abs)
    if v_cross is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a_x, _ = alignment(md, v_cross)
            rec["rho_c_a_null_cross"] = float(spearmanr(c_obs, a_x).statistic)
        rec["rho_c_a_null_cross_partial"] = partial_spearman(c_obs, a_x, lam_abs)
    else:
        rec["rho_c_a_null_cross"] = np.nan
        rec["rho_c_a_null_cross_partial"] = np.nan

    # coarse version: containment of the most motif-aligned modes vs all modes
    top = np.argsort(-a_cos)[:TOP_R]
    rec["c_top_align"] = float(np.mean(c_obs[top]))
    rec["c_all"] = float(np.mean(c_obs))
    rec["c_top_align_null_part"] = float(np.mean(c_rnd[top]))
    rec["lam_top_align"] = float(np.mean(lam_abs[top]))
    rec["lam_all"] = float(np.mean(lam_abs))
    # the block-level routing result says the quotient holds the marginal modes; at the
    # mode level that is a claim about the extreme end of the spectrum, not a global
    # rank trend, so look at the slowest modes directly.
    slow = np.argsort(-lam_abs)[:TOP_R]
    rec["c_top_lam"] = float(np.mean(c_obs[slow]))
    rec["c_top_lam_null_part"] = float(np.mean(c_rnd[slow]))

    # subspace version: the k quotient-routed modes vs the k slowest vs k at random
    n = md["U"].shape[1]
    kk = min(k, n)
    idx_routed = np.argsort(-c_obs)[:kk]
    idx_slow = np.argsort(-lam_abs)[:kk]
    idx_rand = rng.choice(n, size=kk, replace=False)
    rec["span_routed"] = span_alignment(md, idx_routed, v)
    rec["span_slow"] = span_alignment(md, idx_slow, v)
    rec["span_rand"] = span_alignment(md, idx_rand, v)
    rec["span_routed_null_part"] = span_alignment(md, np.argsort(-c_rnd)[:kk], v)
    return rec


def run_net(net: str, seed: int, kg: list[int], rng: np.random.Generator) -> list[dict]:
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], dtype=float)
    ind = np.asarray(wl["ind_active"], dtype=int)
    ls = wl["labels_struct_by_k"]
    jbars = condition_jacobians(net, ind, W)
    conds = load_conditions(net, ind, W, jbars)
    if len(conds) < 4:
        return []
    # one eigendecomposition per condition, reused across every k
    mds = [mode_decomposition(c["Jb"]) for c in conds]
    rows = []
    for k in kg:
        labels = np.asarray(ls[_row_for_k(ls, k)], dtype=int)
        k_real = int(np.unique(labels).size)
        if k_real < 2:
            continue
        P_obs = projector_apply(labels)
        P_rand = projector_apply(random_labels_same_sizes(labels, seed=seed * 100 + k))
        for i, (c, md) in enumerate(zip(conds, mds)):
            v_cross = conds[(i + 1) % len(conds)]["v"] if len(conds) > 1 else None
            rec = one_cell(md, c["v"], v_cross, P_obs, P_rand, k_real, rng)
            rows.append({"model": net, "seed": seed, "k_req": k, "k": k_real,
                         "n_active": W.shape[0], "cond": c["key"], "motif": c["motif"],
                         "rule": c["rule"], "lam_J": c["lam_J"], **rec})
        print(f"    k={k_real:3d}  rho(c,a)={np.nanmedian([r['rho_c_a'] for r in rows if r['k']==k_real]):+.3f}"
              f"  partial={np.nanmedian([r['rho_c_a_partial_lam'] for r in rows if r['k']==k_real]):+.3f}"
              f"  span_routed={np.nanmedian([r['span_routed'] for r in rows if r['k']==k_real]):.3f}"
              f"  span_rand={np.nanmedian([r['span_rand'] for r in rows if r['k']==k_real]):.3f}",
              flush=True)
    return rows


# --------------------------------------------------------------------------- #
PAIRS = [("rho_c_a", "rho_c_a_null_part"), ("rho_c_a", "rho_c_a_null_cross"),
         ("rho_c_a_partial_lam", "rho_c_a_null_part_partial"),
         ("rho_c_a_partial_lam", "rho_c_a_null_cross_partial"),
         ("span_routed", "span_rand"), ("span_routed", "span_slow"),
         ("span_routed", "span_routed_null_part"),
         ("c_top_align", "c_all"), ("c_top_align", "c_top_align_null_part"),
         ("c_top_lam", "c_all"), ("c_top_lam", "c_top_lam_null_part")]


def summarise(rows: list[dict]) -> dict:
    out = {}
    ks = sorted({r["k_req"] for r in rows})
    for k in ks:
        sub = [r for r in rows if r["k_req"] == k]
        seeds = sorted({r["seed"] for r in sub})
        rec = {"n_seeds": len(seeds),
               "n_cond_median": float(np.median([sum(r["seed"] == s for r in sub)
                                                 for s in seeds]))}
        for obs, null in PAIRS:
            a = np.array([np.nanmedian([r[obs] for r in sub if r["seed"] == s])
                          for s in seeds], dtype=float)
            b = np.array([np.nanmedian([r[null] for r in sub if r["seed"] == s])
                          for s in seeds], dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 5 or np.allclose(a[ok], b[ok]):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _stat, p = wilcoxon(a[ok], b[ok])
            rec[f"{obs}_vs_{null}"] = {"obs": float(np.median(a[ok])),
                                       "null": float(np.median(b[ok])),
                                       "delta": float(np.median(a[ok] - b[ok])),
                                       "p": float(p), "n": int(ok.sum())}
        for met in ("rho_c_lam", "rho_a_lam", "rho_c_w", "rho_c_w_partial_lam"):
            vals = np.array([np.nanmedian([r[met] for r in sub if r["seed"] == s])
                             for s in seeds], dtype=float)
            ok = np.isfinite(vals)
            if ok.sum() >= 5:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _s, p = wilcoxon(vals[ok])
                rec[met] = {"median": float(np.median(vals[ok])), "p_vs_0": float(p),
                            "n": int(ok.sum())}
        out[f"k{k}"] = rec
    # per motif type at every k, on the headline statistic
    by_motif = {}
    for mt in MOTIF_TYPES:
        s = [r for r in rows if r["motif"] == mt]
        if len(s) < 10:
            continue
        by_motif[mt] = {"n": len(s),
                        "rho_c_a": float(np.nanmedian([r["rho_c_a"] for r in s])),
                        "partial": float(np.nanmedian([r["rho_c_a_partial_lam"]
                                                       for r in s])),
                        "span_routed": float(np.nanmedian([r["span_routed"] for r in s])),
                        "span_rand": float(np.nanmedian([r["span_rand"] for r in s]))}
    return {"per_k": out, "by_motif": by_motif}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seeds", default="3-12")
    ap.add_argument("--k-grid", default=",".join(str(k) for k in K_GRID))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    kg = [int(x) for x in args.k_grid.split(",")]
    out_path = paths.REPO_ROOT / args.out if args.out else OUT

    rows, missing = [], []
    t0 = time.time()
    for s in seeds:
        net = f"LeakyRNN_softplus_diag_{args.size}_all_{s}"
        if not (WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz").exists():
            missing.append(net)
            continue
        print(f"[{time.time() - t0:6.0f}s] {net}", flush=True)
        rows += run_net(net, s, kg, np.random.default_rng(s))

    if not rows:
        raise SystemExit("no rows produced")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_path} ({len(rows)} rows)")
    out_path.with_suffix(".json").write_text(json.dumps(summarise(rows), indent=1))
    print(f"wrote {out_path.with_suffix('.json')}")
    if missing:
        print("missing: " + ", ".join(missing))


if __name__ == "__main__":
    main()
