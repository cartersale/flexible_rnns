"""Lumpability of the signed connectivity partition on the released tanh 256-unit networks.

The full pipeline was only ever run for the softplus nets. This script reproduces, self-contained,
the two prerequisites for the tanh nets and tests Result 1 (lumpability) on them:
  1. active units  ind_active = { i : sum_epochs Var_trials h_i > 1e-3 } (the softplus definition,
     flexsym.core.variance), from a forward pass over the 15 tasks;
  2. the signed Weisfeiler-Leman partition of the transpose-variant active W_rec across a k-sweep.
It then measures eigenvector leakage and subspace invariance of W_rec onto the group-constant
subspace against size-matched random partitions (the Result-1 statistic), and writes the WL sweep
npz in the format the motif/concentration code expects, so concentration can be run once the tanh
fixed points are found.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import glob
import numpy as np

from flexsym import paths
from flexsym.structural.signed_input_wl_structural_roles import (
    approx_wl_signed, load_w_rec, labels_to_H, random_labels_same_sizes,
)
from flexsym.run.lumpability_check import eig_leakage, subspace_invariance, pw_constant_projector
from flexsym.motifs.motif_participation import WL_ROOT

# The full k = 3..60 grid, matching lumpability_r2_ksweep, so that a tanh row's
# position in labels_struct_by_k is its k - 3 and the architectures are compared
# on one grid. The earlier coarse grid left k_req mislabelled for these nets.
K_GRID = list(range(3, 61))


def model_dir(net):
    for store in (paths.ORIGINAL_STORE, paths.STAGED_STORE):
        hit = glob.glob(str(store / net))
        if hit:
            return Path(hit[0])
    raise FileNotFoundError(net)


def active_units(md):
    from flexsym.training.compute_variance_epoch_to_file import compute_variance_epoch
    import tensorflow.compat.v1 as tf
    res = compute_variance_epoch(str(md), batch_size=256)
    tf.reset_default_graph()
    h_var_all = np.asarray(res["h_var_all"], float)          # (n_hidden, n_epochs)
    return np.where(h_var_all.sum(axis=1) > 1e-3)[0]


def run_net(net, n_null):
    md = model_dir(net)
    ind = active_units(md)
    W_full = load_w_rec(md)                                   # (n_rnn, n_rnn), original
    Wt = W_full.T[np.ix_(ind, ind)]                          # transpose-variant active submatrix
    Wp, Wn = np.maximum(Wt, 0.0), np.maximum(-Wt, 0.0)
    n_active = len(ind)
    op_k = int(round(0.20 * n_active))
    ks = sorted(set(K_GRID + [op_k]))

    labels_by_k = []
    for k in ks:
        lab, _ = approx_wl_signed(Wp, Wn, k, n_iter=20, seed=0, n_init=20)
        labels_by_k.append(lab)
    labels_arr = np.array(labels_by_k, dtype=object)

    # save WL sweep npz in the expected location/format for downstream motif/concentration code
    out_dir = WL_ROOT / net / "transpose"; out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "signed_input_wl_sweep.npz",
             W_rec=Wt, W_pos=Wp, W_neg=Wn, ind_active=ind,
             labels_struct_by_k=labels_arr, matrix_variant="transpose")

    # lumpability at operating k, both matrix orientations, vs size-matched random null
    lab_op, _ = approx_wl_signed(Wp, Wn, op_k, n_iter=20, seed=0, n_init=20)
    kk = int(lab_op.max() + 1)
    P = pw_constant_projector(lab_op, kk)
    res = {"net": net, "n_active": int(n_active), "op_k": op_k, "k": kk}
    for name, M in (("transpose", Wt), ("W", W_full[np.ix_(ind, ind)])):
        leak = eig_leakage(M, P, kk); inv = subspace_invariance(M, P)
        nl = np.empty(n_null); ni = np.empty(n_null)
        for b in range(n_null):
            r = random_labels_same_sizes(lab_op, seed=b); Pn = pw_constant_projector(r, kk)
            nl[b] = eig_leakage(M, Pn, kk); ni[b] = subspace_invariance(M, Pn)
        res[f"{name}_leak"] = float(leak); res[f"{name}_leak_null"] = float(nl.mean())
        res[f"{name}_leak_z"] = float((leak - nl.mean()) / nl.std(ddof=1))
        res[f"{name}_inv"] = float(inv); res[f"{name}_inv_z"] = float((inv - ni.mean()) / ni.std(ddof=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-null", type=int, default=200)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    rows = []
    for s in seeds:
        net = f"LeakyRNN_tanh_diag_256_all_{s}"
        r = run_net(net, args.n_null)
        rows.append(r)
        print(f"[{net}] n_active={r['n_active']} op_k={r['op_k']}(k={r['k']}) | "
              f"transpose leak_z={r['transpose_leak_z']:+.2f} inv_z={r['transpose_inv_z']:+.2f} | "
              f"W leak_z={r['W_leak_z']:+.2f} inv_z={r['W_inv_z']:+.2f}", flush=True)

    lz = np.array([r["transpose_leak_z"] for r in rows])
    print(f"\n=== tanh_diag_256 lumpability (leakage z; negative = more piecewise-constant than chance) ===")
    print(f" transpose leak_z: {', '.join(f'{z:+.1f}' for z in lz)}  (all negative: {bool(np.all(lz<0))})")
    print(f" mean transpose leak_z {lz.mean():+.2f}; mean W leak_z {np.mean([r['W_leak_z'] for r in rows]):+.2f}")

    (paths.RESULTS_ROOT / "motif_fixed_points/tanh_lumpability.json").write_text(
        json.dumps(rows, indent=2, default=float))
    print("wrote results/motif_fixed_points/tanh_lumpability.json")


if __name__ == "__main__":
    main()
