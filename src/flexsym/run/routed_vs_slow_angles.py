"""Principal angles between the quotient-routed subspace and the k slowest modes.

run/quotient_mode_motif_alignment.py found that the motif direction's alignment with
the routed subspace (the k Jacobian modes the partition keeps, ranked by containment
||P u_i|| / ||u_i||) is statistically indistinguishable from its alignment with the k
slowest modes (ranked by |lambda|) at the operating scale. That could mean either of
two very different things: the two subspaces are the same subspace, or they are
different subspaces that happen to contain the motif direction equally well.

Principal angles settle it. For each network, k, and motif condition:

    R   orthonormal basis of the real span of the top-k modes by containment
    S   orthonormal basis of the real span of the top-k modes by |lambda|
    cos(theta_i) = singular values of R' S

and we report the angle spectrum (how many directions are genuinely shared, where the
two subspaces part company), against two baselines built the same way:

    R_rand   routed under a size-matched RANDOM partition of the same matrix
    U_rand   k modes drawn at random

Because complex eigenvectors are taken as conjugate pairs, the real spans have
dimension ~k; the angle spectrum is truncated to min(dim R, dim S) and the realised
dimensions are recorded.

Finally, the question the angles raise: WHERE does the motif direction sit relative to
the disagreement? The motif's energy inside R is decomposed over the principal
directions of R, and the fraction landing on the shared ones (cos theta > 0.9) is
reported. If the motif lives on the shared directions the two subspaces are
interchangeable for this purpose; if it lives on the directions where they part, the
routed subspace is doing something the slow-mode filter is not.

Writes results/main/signed_input_wl_paper/lumpability/routed_vs_slow_angles.{csv,json}.

  conda run -n myenv python -m flexsym.run.routed_vs_slow_angles --seeds 3-12
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
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_quotient_prediction import condition_jacobians, load_conditions
from flexsym.run.motif_reuse_control import MOTIF_TYPES, _row_for_k
from flexsym.run.quotient_mode_motif_alignment import (
    containment, mode_decomposition, projector_apply,
)
from flexsym.structural.signed_input_wl_structural_roles import random_labels_same_sizes

OUT = paths.RESULTS_ROOT / "signed_input_wl_paper/lumpability/routed_vs_slow_angles.csv"
K_GRID = [8, 12, 16, 20, 25, 30, 40, 50]
SHARED_COS = 0.9        # cos theta above which a principal direction counts as shared


def real_basis(md: dict, idx: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the real span of the selected (possibly complex) modes."""
    U = md["U"][:, idx]
    B = np.hstack([np.real(U), np.imag(U)])
    B = B[:, np.linalg.norm(B, axis=0) > 1e-12]
    if not B.shape[1]:
        return np.zeros((md["U"].shape[0], 0))
    Q, R = np.linalg.qr(B)
    return Q[:, np.abs(np.diag(R)) > 1e-10 * max(1.0, abs(R[0, 0]))]


def angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between two orthonormal bases."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.array([])
    s = np.clip(svdvals(A.T @ B), 0.0, 1.0)
    d = min(A.shape[1], B.shape[1])
    return np.arccos(s[:d])


def motif_split(A: np.ndarray, B: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """(energy fraction of v|_A on shared principal directions, fraction of v in A).

    v is projected into A, expressed in A's principal-direction basis relative to B,
    and the energy on directions with cos theta > SHARED_COS is reported as a fraction
    of the energy v has inside A.
    """
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.nan, np.nan
    coords = A.T @ v                       # v's coordinates in A
    nv = np.linalg.norm(v)
    in_A = float(np.linalg.norm(coords) / nv) if nv > 0 else np.nan
    U, s, _ = np.linalg.svd(A.T @ B)
    pc = U.T @ coords                      # coordinates in A's principal directions
    tot = float(np.sum(pc ** 2))
    if tot <= 0:
        return np.nan, in_A
    d = min(len(s), len(pc))
    shared = np.zeros(len(pc), dtype=bool)
    shared[:d] = np.clip(s[:d], 0, 1) > SHARED_COS
    return float(np.sum(pc[shared] ** 2) / tot), in_A


def one_cell(md: dict, v: np.ndarray, c_obs: np.ndarray, c_rnd: np.ndarray,
             k: int, rng: np.random.Generator) -> dict:
    lam_abs = np.abs(md["lam"])
    n = md["U"].shape[1]
    kk = min(k, n)
    i_R = np.argsort(-c_obs)[:kk]
    i_S = np.argsort(-lam_abs)[:kk]
    i_Rr = np.argsort(-c_rnd)[:kk]
    i_U = rng.choice(n, size=kk, replace=False)

    R, S = real_basis(md, i_R), real_basis(md, i_S)
    Rr, Ur = real_basis(md, i_Rr), real_basis(md, i_U)
    th = angles(R, S)
    cos = np.cos(th)
    rec = {
        "dim_R": int(R.shape[1]), "dim_S": int(S.shape[1]), "n_theta": int(len(th)),
        "n_mode_overlap": int(len(np.intersect1d(i_R, i_S))),
        "frac_mode_overlap": float(len(np.intersect1d(i_R, i_S)) / kk),
        "theta_min_deg": float(np.degrees(th[0])) if len(th) else np.nan,
        "theta_med_deg": float(np.degrees(np.median(th))) if len(th) else np.nan,
        "theta_max_deg": float(np.degrees(th[-1])) if len(th) else np.nan,
        "cos2_mean": float(np.mean(cos ** 2)) if len(th) else np.nan,
        "n_shared": int(np.sum(cos > SHARED_COS)),
        "frac_shared": float(np.mean(cos > SHARED_COS)) if len(th) else np.nan,
        "frac_cos_gt_099": float(np.mean(cos > 0.99)) if len(th) else np.nan,
        "frac_orthogonal": float(np.mean(cos < 0.1)) if len(th) else np.nan,
    }
    for name, Bm in (("randpart", Rr), ("randmodes", Ur)):
        t2 = angles(R, Bm)
        c2 = np.cos(t2)
        rec[f"cos2_mean_{name}"] = float(np.mean(c2 ** 2)) if len(t2) else np.nan
        rec[f"frac_shared_{name}"] = float(np.mean(c2 > SHARED_COS)) if len(t2) else np.nan
    t3 = angles(S, Ur)
    rec["cos2_mean_slow_vs_randmodes"] = float(np.mean(np.cos(t3) ** 2)) if len(t3) else np.nan

    shared_energy, v_in_R = motif_split(R, S, v)
    rec["motif_energy_shared"] = shared_energy
    rec["motif_in_R"] = v_in_R
    rec["motif_energy_shared_S"] = motif_split(S, R, v)[0]
    rec["motif_in_S"] = motif_split(S, R, v)[1]
    return rec


def run_net(net: str, seed: int, kg: list[int], rng: np.random.Generator) -> list[dict]:
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], dtype=float)
    ind = np.asarray(wl["ind_active"], dtype=int)
    ls = wl["labels_struct_by_k"]
    conds = load_conditions(net, ind, W, condition_jacobians(net, ind, W))
    if len(conds) < 4:
        return []
    mds = [mode_decomposition(c["Jb"]) for c in conds]
    rows = []
    for k in kg:
        labels = np.asarray(ls[_row_for_k(ls, k)], dtype=int)
        k_real = int(np.unique(labels).size)
        if k_real < 2:
            continue
        P_obs = projector_apply(labels)
        P_rand = projector_apply(random_labels_same_sizes(labels, seed=seed * 100 + k))
        for c, md in zip(conds, mds):
            rec = one_cell(md, c["v"], containment(md, P_obs), containment(md, P_rand),
                           k_real, rng)
            rows.append({"model": net, "seed": seed, "k_req": k, "k": k_real,
                         "cond": c["key"], "motif": c["motif"], **rec})
        cur = [r for r in rows if r["k"] == k_real]
        print(f"    k={k_real:3d}  shared {np.median([r['n_shared'] for r in cur]):.0f}/"
              f"{np.median([r['n_theta'] for r in cur]):.0f} dirs  "
              f"theta med {np.median([r['theta_med_deg'] for r in cur]):5.1f}deg  "
              f"max {np.median([r['theta_max_deg'] for r in cur]):5.1f}deg  "
              f"modes shared {np.median([r['frac_mode_overlap'] for r in cur]):.2f}  "
              f"motif energy on shared dirs {np.nanmedian([r['motif_energy_shared'] for r in cur]):.2f}",
              flush=True)
    return rows


SINGLE = ["frac_shared", "frac_mode_overlap", "cos2_mean", "theta_med_deg",
          "theta_max_deg", "motif_energy_shared", "motif_in_R", "motif_in_S"]
PAIRS = [("cos2_mean", "cos2_mean_randpart"), ("cos2_mean", "cos2_mean_randmodes"),
         ("frac_shared", "frac_shared_randpart"), ("frac_shared", "frac_shared_randmodes"),
         ("motif_energy_shared", "motif_energy_shared_S")]


def summarise(rows: list[dict]) -> dict:
    out = {}
    for k in sorted({r["k_req"] for r in rows}):
        sub = [r for r in rows if r["k_req"] == k]
        seeds = sorted({r["seed"] for r in sub})
        rec = {}
        for met in SINGLE:
            vals = np.array([np.nanmedian([r[met] for r in sub if r["seed"] == s])
                             for s in seeds], dtype=float)
            rec[met] = float(np.nanmedian(vals))
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
                _s, p = wilcoxon(a[ok], b[ok])
            rec[f"{obs}_vs_{null}"] = {"obs": float(np.median(a[ok])),
                                       "null": float(np.median(b[ok])),
                                       "p": float(p), "n": int(ok.sum())}
        out[f"k{k}"] = rec
    by_motif = {}
    for mt in MOTIF_TYPES:
        s = [r for r in rows if r["motif"] == mt]
        if len(s) >= 10:
            by_motif[mt] = {m: float(np.nanmedian([r[m] for r in s])) for m in SINGLE}
            by_motif[mt]["n"] = len(s)
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
