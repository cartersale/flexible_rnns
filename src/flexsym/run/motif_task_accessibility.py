"""Is the motif mode the TASK-ACTIVE mode among dynamically-similar directions?

The eig-matched control (motif_quotient_eigmatched_control.py) showed the connectivity
quotient cannot geometrically separate a motif direction v from a matched-eigenvalue
non-motif eigenvector q of the same Jacobian: both are near-marginal slow modes and the
quotient predicts them about equally. That is a statement about connectivity geometry,
which is blind to how the task uses a mode.

This asks the complementary question: do v and q differ in TASK ACCESSIBILITY? For each
condition we hold the same operating point and compare v against its matched q on seven
measures of how the task drives, moves along, reads out, and controls each mode:

  input_proj    |cos(mode, B u)|            alignment with the epoch's input drive
  disp_proj     |cos(mode, h* - baseline)|  alignment with where the epoch sits in state
  traj_var      m'Sigma m / tr(Sigma)       fraction of the fixed-point cloud's spread
  readout_pot   ||R' m||^2                   fraction of the mode inside the readout plane
  persistence   mean_i m' J_i m             local eigenvalue along m across the cloud
                                             (ring tangent ~1 everywhere; dark mode < 1)
  controllab    ||B' w||                     modal controllability (w = left eigvector)
  observab      ||C_resp m||                 modal observability at the response output

B = alpha diag(g) W_in is the input Jacobian, C = W_out' the readout, both at the fixed
point. If the task selects the motif within the slow subspace, v beats q here even though
the quotient could not tell them apart.

  conda run -n myenv python -m flexsym.run.motif_task_accessibility --seeds 0-2
"""

from __future__ import annotations

import argparse
import csv
import glob
import json

import numpy as np

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_reuse_control import MOTIF_TYPES
from flexsym.run.motif_quotient_prediction import (
    condition_jacobians,
    load_conditions,
)
from flexsym.run.motif_quotient_eigmatched_control import matched_nonmotif_direction

FP_ROOT = paths.RESULTS_ROOT / "motif_fixed_points"
MAX_CLOUD = 60           # fixed points subsampled per condition for cloud/persistence
METRICS = ["input_proj", "disp_proj", "traj_var", "readout_pot",
           "persistence", "controllab", "observab"]


def model_matrices(net, ind):
    """Input Jacobian factor W_in and readout C at active units, from model params."""
    mp = glob.glob(str(paths.ORIGINAL_STORE / net / "model_params.npz"))
    if not mp:
        mp = glob.glob(str(paths.STAGED_STORE / net / "model_params.npz"))
    p = np.load(mp[0])
    n_rec = len(p["b_in"])
    w_in = np.asarray(p["w_in"], float)          # (n_input + n_rec, n_rec)
    n_in = w_in.shape[0] - n_rec
    W_in = w_in[:n_in, :][:, ind].T              # (n_active, n_input), receives orient.
    C = np.asarray(p["w_out"], float)[ind].T     # (n_output, n_active)
    return W_in, C


def unit(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def cloud_stats(net, key, ind):
    """Fixed-point cloud: covariance, centroid, and subsampled per-point Jacobians."""
    d = npload(FP_ROOT / net / f"{key}.npz")
    X = np.asarray(d["xstar"], float)[:, ind]
    J = np.asarray(d["J_xstar"], float)
    if len(X) > MAX_CLOUD:
        idx = np.linspace(0, len(X) - 1, MAX_CLOUD).astype(int)
        X, J = X[idx], J[idx]
    Jact = J[:, ind][:, :, ind]                  # (m, n_active, n_active)
    Sigma = np.cov(X.T) if len(X) > 1 else np.zeros((len(ind), len(ind)))
    return X.mean(0), Sigma, Jact


def modal_left_vector(Jb, m):
    """Unit left eigenvector of Jb for the eigenpair whose right vector best matches m."""
    lam, V = np.linalg.eig(Jb)
    with np.errstate(all="ignore"):
        Vinv = np.linalg.inv(V)
    align = np.abs(np.real(V).T @ m) / np.maximum(
        np.linalg.norm(np.real(V), axis=0) * np.linalg.norm(m), 1e-12)
    j = int(np.argmax(align))
    return unit(np.real(Vinv[j]))


def accessibility(mode, *, drive, disp, Sigma, R, Jcloud, Jb, B, Cresp):
    m = unit(mode)
    tr = np.trace(Sigma)
    return {
        "input_proj": float(abs(m @ unit(drive))) if np.linalg.norm(drive) > 0 else 0.0,
        "disp_proj": float(abs(m @ unit(disp))) if np.linalg.norm(disp) > 0 else 0.0,
        "traj_var": float((m @ Sigma @ m) / tr) if tr > 0 else 0.0,
        "readout_pot": float(np.sum((R.T @ m) ** 2)),
        "persistence": float(np.mean([m @ Ji @ m for Ji in Jcloud])),
        "controllab": float(np.linalg.norm(B.T @ modal_left_vector(Jb, m))),
        "observab": float(np.linalg.norm(Cresp @ m)),
    }


def run_net(net, baseline_cache):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], float)
    ind = np.asarray(wl["ind_active"], int)
    jbars = condition_jacobians(net, ind, W)
    conds = load_conditions(net, ind, W, jbars)
    W_in, C = model_matrices(net, ind)
    Cresp = C[1:]                                # two response channels
    # network baseline state: mean over all conditions' fixed-point centroids
    centroids = {}
    for c in conds:
        d = npload(FP_ROOT / net / f"{c['key']}.npz")
        centroids[c["key"]] = np.asarray(d["xstar"], float)[:, ind].mean(0)
    baseline = np.mean(list(centroids.values()), axis=0)

    rows = []
    for c in conds:
        q, lam_q, align_q = matched_nonmotif_direction(c["Jb"], c["v"], c["lam_J"])
        if q is None:
            continue
        Jb = c["Jb"]
        B = c["alpha"] * (c["g"][:, None] * W_in)
        u = npload(FP_ROOT / net / f"{c['key']}.npz")["inputs"][0]
        drive = B @ np.asarray(u, float)
        disp = centroids[c["key"]] - baseline
        cen, Sigma, Jcloud = cloud_stats(net, c["key"], ind)
        Rp, _ = np.linalg.qr(Cresp.T)            # orthonormal readout-plane basis
        kw = dict(drive=drive, disp=disp, Sigma=Sigma, R=Rp, Jcloud=Jcloud,
                  Jb=Jb, B=B, Cresp=Cresp)
        am = accessibility(c["v"], **kw)
        aq = accessibility(q, **kw)
        rows.append({"net": net, "key": c["key"], "motif": c["motif"],
                     "lam_J": c["lam_J"], "lam_q": lam_q, "align_q": align_q,
                     **{f"m_{k}": am[k] for k in METRICS},
                     **{f"q_{k}": aq[k] for k in METRICS}})
    return rows


def summarize(rows):
    def paired(sel, key):
        r = [x for x in rows if sel(x)]
        m = np.array([x[f"m_{key}"] for x in r])
        q = np.array([x[f"q_{key}"] for x in r])
        d = m - q
        n = len(r)
        return dict(n=n, motif=float(np.nanmean(m)), ctrl=float(np.nanmean(q)),
                    gap=float(np.nanmean(d)),
                    win=int(np.nansum(m > q)),
                    win_frac=float(np.nanmean(m > q)))

    print(f"\n=== task accessibility: motif mode v vs matched non-motif mode q "
          f"(all {len(rows)} conditions, seeds 0-2) ===")
    print(f"  {'metric':>13} {'motif':>7} {'ctrl':>7} {'gap':>8} {'win':>9}")
    for key in METRICS:
        s = paired(lambda x: True, key)
        print(f"  {key:>13} {s['motif']:>7.3f} {s['ctrl']:>7.3f} {s['gap']:>+8.3f} "
              f"{s['win']:>4}/{s['n']} ({s['win_frac']*100:.0f}%)")

    for mt in MOTIF_TYPES:
        r = [x for x in rows if x["motif"] == mt]
        if not r:
            continue
        print(f"\n  -- {mt} (n={len(r)}) --")
        for key in METRICS:
            s = paired(lambda x: x["motif"] == mt, key)
            print(f"     {key:>13} motif {s['motif']:>6.3f}  ctrl {s['ctrl']:>6.3f}  "
                  f"gap {s['gap']:>+7.3f}  win {s['win']}/{s['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--out", default="task_accessibility")
    args = ap.parse_args()
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))

    rows = []
    for s in seeds:
        net = f"LeakyRNN_softplus_diag_{args.size}_all_{s}"
        if not (FP_ROOT / net / "participation_cache_v2.npz").exists():
            print(f"[skip] {net}")
            continue
        r = run_net(net, {})
        rows += r
        print(f"[done] {net}: {len(r)} conditions")

    summarize(rows)

    csv_path = FP_ROOT / f"{args.out}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(FP_ROOT / f"{args.out}.json", "w"), indent=2, default=float)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
