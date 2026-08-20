"""Fair test of the rotation motif: ring plane vs matched non-ring plane, at readout.

The rotation result (motif_rotation_from_response.py) shows the memory ring is output-null
during the delay and rotates into output-potent space at the response epoch, beating a
RANDOM 2-D plane (~0.008) by ~10x. The eig-matched control asks the sharper question the
random-plane null does not: at the response epoch, is the ring plane more output-potent
than a NON-RING subspace of the SAME response Jacobian whose modes have matched eigenvalues?

For each delayed-response task (delay epoch + go1 response epoch):
  ring plane    top-RING_DIM PCs of the response fixed-point cloud (the rotated manifold)
  match plane   RING_DIM eigenvectors of the response mean Jacobian whose |lambda| are
                closest to the ring modes' |lambda| but which are NOT aligned with the
                ring plane (a dynamically-matched, non-motif subspace at the same point)

We report each plane's readout potency at the response epoch, and, as the null the
rotation paper used, a random 2-D plane. The rotation motif is task-specific (not just a
property of slow subspaces) only if the ring plane out-projects the matched plane.

  conda run -n myenv python -m flexsym.run.motif_rotation_readout_control --seeds 0-2
"""

from __future__ import annotations

import argparse
import csv
import glob

import numpy as np

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT

RESP = paths.RESULTS_ROOT / "motif_fixed_points_response"
MEM = paths.RESULTS_ROOT / "motif_fixed_points"
Q_SLOW = 1e-6
RING_DIM = 2
ALIGN_MAX = 0.40         # a match-plane eigenvector must be this unaligned with the ring
MAX_FP = 150             # fixed points averaged into the response mean Jacobian


def response_jacobian(npz_path, ind):
    """Mean Jacobian over an epoch's fixed-point cloud, at active units."""
    d = npload(npz_path)
    J = np.asarray(d["J_xstar"], np.float32)
    if J.ndim != 3 or not len(J):
        return None
    if len(J) > MAX_FP:
        J = J[np.linspace(0, len(J) - 1, MAX_FP).astype(int)]
    return J.mean(0)[np.ix_(ind, ind)].astype(float)


def ring_plane(xstar, q, ndim=RING_DIM):
    keep = xstar[q < Q_SLOW]
    if len(keep) < ndim + 1:
        return None
    X = keep - keep.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:ndim].T                            # (N, ndim) orthonormal


def potency(U, R):
    s = np.linalg.svd(U.T @ R, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1) ** 2))


def matched_nonring_plane(Jb, ring, ndim=RING_DIM):
    """ndim eigenvectors of Jb with |lambda| nearest the ring modes', unaligned with ring."""
    lam, V = np.linalg.eig(Jb)
    Vr = np.real(V)
    norms = np.linalg.norm(Vr, axis=0)
    # eigenvalues of the modes the ring occupies: the ndim most ring-aligned eigenvectors
    ring_align = np.linalg.norm(ring.T @ (Vr / np.maximum(norms, 1e-12)), axis=0)
    ring_lams = np.abs(lam[np.argsort(-ring_align)[:ndim]])
    target = float(np.mean(ring_lams))
    # candidate non-ring modes: low alignment with the ring plane
    cand = np.flatnonzero((ring_align < ALIGN_MAX) & (norms > 1e-9))
    if cand.size < ndim:
        return None
    pick = cand[np.argsort(np.abs(np.abs(lam[cand]) - target))[:ndim]]
    U, _ = np.linalg.qr(Vr[:, pick])
    return U[:, :ndim]


def random_plane(n, ndim, rng):
    U, _ = np.linalg.qr(rng.normal(size=(n, ndim)))
    return U[:, :ndim]


def _model_params(net):
    """Released nets live in ORIGINAL_STORE; our own training grid in STAGED_STORE."""
    for store in (paths.ORIGINAL_STORE, paths.STAGED_STORE):
        hit = glob.glob(str(store / net / "model_params.npz"))
        if hit:
            return np.load(hit[0])
    raise FileNotFoundError(f"model_params.npz for {net} not in ORIGINAL_STORE or STAGED_STORE")


def run_net(net, rng):
    params = _model_params(net)
    R_full, _ = np.linalg.qr(np.asarray(params["w_out"], float)[:, 1:])   # readout plane

    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    ind = np.asarray(wl["ind_active"], int)
    W = np.asarray(wl["W_rec"], float)
    R = R_full[ind]                                # readout at active units

    rows = []
    for resp in sorted((RESP / net).glob("*__go1.npz")):
        if resp.name.startswith("._"):
            continue
        rule = resp.stem.split("__")[0]
        mem = next((MEM / net / f"{rule}__{e}.npz" for e in ("delay2", "delay1")
                    if (MEM / net / f"{rule}__{e}.npz").exists()), None)
        if mem is None:
            continue
        dr = npload(resp)
        Xr = np.asarray(dr["xstar"], float)[:, ind]
        qr = np.asarray(dr["qstar"], float)
        dm = npload(mem)
        Xm = np.asarray(dm["xstar"], float)[:, ind]
        qm = np.asarray(dm["qstar"], float)

        ring_resp = ring_plane(Xr, qr)
        ring_mem = ring_plane(Xm, qm)
        if ring_resp is None or ring_mem is None:
            continue
        Jb = response_jacobian(resp, ind)
        if Jb is None:
            continue
        match = matched_nonring_plane(Jb, ring_resp)
        if match is None:
            continue
        rnd = np.mean([potency(random_plane(len(ind), RING_DIM, rng), R)
                       for _ in range(50)])
        rows.append({
            "net": net, "rule": rule,
            "ring_delay_potency": potency(ring_mem, R),      # output-null before rotation
            "ring_resp_potency": potency(ring_resp, R),      # after rotation
            "match_resp_potency": potency(match, R),         # matched non-ring, same point
            "random_potency": float(rnd),
            "ring_minus_match": potency(ring_resp, R) - potency(match, R),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--out", default="rotation_readout_control.csv")
    args = ap.parse_args()
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    rng = np.random.default_rng(0)

    rows = []
    for s in seeds:
        net = f"LeakyRNN_softplus_diag_{args.size}_all_{s}"
        if not (RESP / net).exists():
            print(f"[skip] {net}: no response epochs")
            continue
        r = run_net(net, rng)
        rows += r
        print(f"[done] {net}: {len(r)} delayed-response tasks")

    if not rows:
        print("no comparable tasks")
        return
    rd = np.array([r["ring_delay_potency"] for r in rows])
    rr = np.array([r["ring_resp_potency"] for r in rows])
    mp = np.array([r["match_resp_potency"] for r in rows])
    rp = np.array([r["random_potency"] for r in rows])
    gap = rr - mp

    print(f"\n=== readout potency at the response epoch ({len(rows)} tasks, seeds "
          f"{args.seeds}) ===")
    print(f"  ring plane, delay epoch (pre-rotation)  {np.median(rd):.4f}")
    print(f"  ring plane, response epoch (rotated)    {np.median(rr):.4f}")
    print(f"  matched non-ring plane, response epoch  {np.median(mp):.4f}")
    print(f"  random plane, response epoch            {np.median(rp):.4f}")
    print(f"\n  ring vs matched:  gap {np.median(gap):+.4f},  "
          f"ring>matched in {int((gap > 0).sum())}/{len(rows)} tasks")
    print(f"  ring vs random:   ring>random in "
          f"{int((rr > rp).sum())}/{len(rows)} tasks")

    out = MEM / args.out
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
