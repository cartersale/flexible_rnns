"""Reproduce the Driscoll rotation motif: the ring rotating into output-potent space.

Driscoll et al. (2024) report that "the ring attractor rotated from output null space
into output potent space when the fixation input changed to zero". That is a
reorientation of the attractor manifold ACROSS an input transition, not a complex
Jacobian eigenvalue at fixed input, so the epoch-wise motif classifier cannot see it --
it needs the memory epoch and the response epoch compared to each other.

For each task with both a memory delay and a response epoch we:
  1. take the slow-point cloud of each epoch (q < Q_SLOW), which is the ring,
  2. fit a plane to each (top-2 PCs of the centred cloud),
  3. measure the principal angles between the delay plane and the response plane
     -- the rotation itself,
  4. measure each plane's overlap with the readout row space of W_out (the two
     response-direction channels) -- output-null vs output-potent.

The rotation motif is reproduced if the planes are rotated relative to each other AND
the response plane is markedly more output-potent than the delay plane.

Writes results/<...>/rotation_motif.csv (repo-side, small).
"""

from __future__ import annotations

from flexsym import paths

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = paths.REPO_ROOT
DRIVE = paths.MODEL_ROOT
MODELS = DRIVE / "multitask_processed/suppfig4"
OUT = paths.RESULTS_ROOT / "motif_rotation"

Q_SLOW = 1e-6      # slow-point threshold: the line/ring-attractor cloud
RING_DIM = 2       # a ring is locally 1-D but spans a plane


def resolve_root(name: str) -> Path:
    """Heavy fixed-point trees live on the drive; fall back to the repo copy."""
    for base in (DRIVE / "results" / name, paths.RESULTS_ROOT / name):
        if base.exists():
            return base
    raise FileNotFoundError(f"{name} not found on drive or in repo (is X9_Pro mounted?)")


def plane(xstar, q, ndim=RING_DIM):
    """Top-`ndim` principal subspace of the slow-point cloud, plus its size."""
    keep = xstar[q < Q_SLOW]
    if len(keep) < ndim + 1:
        return None, len(keep)
    X = keep - keep.mean(0, keepdims=True)
    _, sv, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:ndim].T, len(keep)          # (N, ndim) orthonormal columns


def principal_angles(U, V):
    """Principal angles (degrees) between two orthonormal-column subspaces."""
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def potency(U, R):
    """Mean squared cosine between subspace U and the readout row space R (0-1)."""
    s = np.linalg.svd(U.T @ R, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="LeakyRNN_softplus_diag_256_all_0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    mem_root = resolve_root("motif_fixed_points") / args.net
    resp_root = resolve_root("motif_fixed_points_response") / args.net

    params = np.load(MODELS / args.net / "model_params.npz")
    # readout row space: the two response-direction channels (channel 0 is fixation)
    W_out = np.asarray(params["w_out"], dtype=float)[:, 1:]
    R, _ = np.linalg.qr(W_out)

    # what a randomly oriented plane would overlap with the readout, for reference
    rng = np.random.default_rng(0)
    base = []
    for _ in range(2000):
        U, _ = np.linalg.qr(rng.normal(size=(R.shape[0], RING_DIM)))
        base.append(potency(U, R))
    base = np.asarray(base)
    lo, hi = np.percentile(base, 5), np.percentile(base, 95)
    print(f"random-plane potency: mean {base.mean():.4f} (5th {lo:.5f}, 95th {hi:.4f})\n")

    rows = []
    # skip macOS AppleDouble sidecars ("._name"), which the drive filesystem creates
    for resp in sorted(p for p in resp_root.glob("*__go1.npz")
                       if not p.name.startswith("._")):
        rule = resp.stem.split("__")[0]
        # the memory epoch this task ends with: prefer the last delay
        cands = [mem_root / f"{rule}__delay2.npz", mem_root / f"{rule}__delay1.npz"]
        mem = next((c for c in cands if c.exists()), None)
        if mem is None:
            print(f"  {rule}: no delay epoch (no memory to rotate) — skipped")
            continue

        dm = np.load(mem, allow_pickle=True)
        dr = np.load(resp, allow_pickle=True)
        Ud, nd = plane(np.asarray(dm["xstar"], float), np.asarray(dm["qstar"], float))
        Ur, nr = plane(np.asarray(dr["xstar"], float), np.asarray(dr["qstar"], float))
        if Ud is None or Ur is None:
            print(f"  {rule}: too few slow points (delay {nd}, response {nr}) — skipped")
            continue

        ang = principal_angles(Ud, Ur)
        pot_d, pot_r = potency(Ud, R), potency(Ur, R)
        rows.append({
            "net": args.net, "rule": rule, "delay_epoch": mem.stem.split("__")[1],
            "n_slow_delay": nd, "n_slow_response": nr,
            "angle_min_deg": float(ang.min()), "angle_max_deg": float(ang.max()),
            "angle_mean_deg": float(ang.mean()),
            "potency_delay": pot_d, "potency_response": pot_r,
            "potency_gain": pot_r - pot_d,
        })
        print(f"  {rule:18s} rotation {ang.mean():5.1f}deg   "
              f"potency {pot_d:.3f} -> {pot_r:.3f}  ({pot_r - pot_d:+.3f})", flush=True)

    if not rows:
        print("no comparable epochs found")
        return
    dest = OUT / f"rotation_motif_{args.net}.csv"
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    gain = np.array([r["potency_gain"] for r in rows])
    pd_ = np.array([r["potency_delay"] for r in rows])
    pr_ = np.array([r["potency_response"] for r in rows])
    ang = np.array([r["angle_mean_deg"] for r in rows])
    print(f"\n=== {args.net}: {len(rows)} tasks with a delay + response ===")
    print(f"  mean rotation      {ang.mean():.1f} deg  (range {ang.min():.1f}-{ang.max():.1f})")
    print(f"  potency delay      {np.mean([r['potency_delay'] for r in rows]):.3f}")
    print(f"  potency response   {np.mean([r['potency_response'] for r in rows]):.3f}")
    print(f"  more potent after the fixation drop in {int((gain > 0).sum())}/{len(rows)} tasks")
    print(f"  delay below 5th pct of random:      {int((pd_ < lo).sum())}/{len(rows)}")
    print(f"  response above 95th pct of random:  {int((pr_ > hi).sum())}/{len(rows)}")
    print(f"wrote {dest}")
    (OUT / f"rotation_motif_{args.net}.json").write_text(json.dumps({
        "net": args.net, "n_tasks": len(rows),
        "mean_angle_deg": float(ang.mean()),
        "mean_potency_delay": float(np.mean([r["potency_delay"] for r in rows])),
        "mean_potency_response": float(np.mean([r["potency_response"] for r in rows])),
        "n_potency_gain_positive": int((gain > 0).sum()),
        "random_plane_potency_mean": float(base.mean()),
        "n_delay_below_random_p5": int((pd_ < lo).sum()),
        "n_response_above_random_p95": int((pr_ > hi).sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
