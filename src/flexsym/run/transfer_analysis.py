"""Transfer analysis: does the connectivity coordinate system stay fixed while motifs rearrange
inside it, as the base net is fine-tuned on the held-out tasks?

Reads the pre-transfer base net and the fine-tuning snapshots (transfer_train.py) and tracks, per
snapshot:
  A. Coordinate-system stability. WL partition of W_rec on a FIXED unit set (base active units), and
     the group-constant subspace S_t; principal angles of S_t against the pre-transfer S_0, plus
     label ARI. The reference scale is this base net's OWN restart-noise floor (WL re-run from many
     random inits on the base matrix -- the non-identifiability floor of section 4). Coordinates are
     "fixed" if S_t stays within the restart floor.
  B. Motif movement. Numpy fixed-point finding of the held-out tasks' epochs from each snapshot's
     weights (h* = softplus(W_rec h* + W_in u + b)); motif directions are expressed in the FIXED
     base coordinates -- containment in S_0 and drift (angle vs the final snapshot's motif) across
     fine-tuning.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
from scipy.linalg import subspace_angles
from sklearn.metrics import adjusted_rand_score

from flexsym.structural.signed_input_wl_structural_roles import (
    approx_wl_signed, labels_to_H, random_labels_same_sizes,
)

N_INPUT = 20


def load_params(npz):
    d = np.load(npz, allow_pickle=True)
    w_in = np.asarray(d["w_in"], float)
    return {"W_rec": w_in[N_INPUT:, :], "W_in": w_in[:N_INPUT, :],
            "b_rec": np.asarray(d["b_rec"], float) if "b_rec" in d else np.zeros(w_in.shape[1]),
            "w_out": np.asarray(d["w_out"], float), "b_out": np.asarray(d["b_out"], float),
            "perf": json.loads(str(d["perf"])) if "perf" in d else {}}


def base_active_units(base_dir):
    """Active units of the base net (Sum_epoch Var > 1e-3), via a TF forward pass."""
    from flexsym.training.compute_variance_epoch_to_file import compute_variance_epoch
    import tensorflow.compat.v1 as tf
    res = compute_variance_epoch(str(base_dir), batch_size=200)
    tf.reset_default_graph()
    hv = np.asarray(res["h_var_all"], float)
    return np.where(hv.sum(1) > 1e-3)[0]


def wl_subspace(W_rec, ind, k, seed=0):
    """Transpose-variant signed WL at k on the active submatrix; return (labels, S) with S the
    orthonormal group-constant lift."""
    Wt = W_rec.T[np.ix_(ind, ind)]
    Wp, Wn = np.maximum(Wt, 0.0), np.maximum(-Wt, 0.0)
    lab, _ = approx_wl_signed(Wp, Wn, k, n_iter=20, seed=seed, n_init=20)
    H = labels_to_H(lab, int(lab.max() + 1)).astype(float)
    S = H / np.sqrt(H.sum(0))[None, :]
    return lab, S


def princ(S0, S1):
    ang = subspace_angles(S0, S1)
    return float(np.degrees(ang).mean()), float(np.mean(np.cos(ang) ** 2))


def restart_floor(W_rec, ind, k, n_restart=12):
    """Non-identifiability floor: WL from many inits on the SAME base matrix; pairwise principal
    angles among the resulting subspaces."""
    subs, labs = [], []
    for r in range(n_restart):
        lab, S = wl_subspace(W_rec, ind, k, seed=1000 + r)
        subs.append(S); labs.append(lab)
    angs, mscas, aris = [], [], []
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            a, m = princ(subs[i], subs[j]); angs.append(a); mscas.append(m)
            aris.append(adjusted_rand_score(labs[i], labs[j]))
    return {"angle_mean": float(np.mean(angs)), "angle_sd": float(np.std(angs)),
            "msca_mean": float(np.mean(mscas)), "ari_mean": float(np.mean(aris))}


# --------------------------------------------------------------------------- #
# Part B: numpy fixed-point finding of held-out task epochs
# --------------------------------------------------------------------------- #
def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def epoch_inputs(rule, hp, epochs=("stim1", "delay1", "go1")):
    from flexsym.core.task import generate_trials
    trial = generate_trials(rule, hp, "test", noise_on=False, batch_size=64)
    out = {}
    for e in epochs:
        if e not in trial.epochs:
            continue
        a, b = trial.epochs[e]
        a = 0 if a is None else a; b = trial.x.shape[0] if b is None else b
        t = (a + b) // 2
        out[e] = (np.asarray(trial.x[t], float), np.asarray(trial.x[:, :, :], float), (a, b))
    return out, trial


def find_fixed_points(W, Win, b, u, seeds, alpha=0.2, iters=2000, tol=1e-8):
    """h* = softplus(W h* + Win u + b) via leaky relaxation from seed states."""
    fps, qs = [], []
    for h0 in seeds:
        h = h0.copy()
        for _ in range(iters):
            f = softplus(W @ h + Win @ u + b)
            hn = (1 - alpha) * h + alpha * f
            if np.max(np.abs(hn - h)) < tol:
                h = hn; break
            h = hn
        q = 0.5 * np.sum((softplus(W @ h + Win @ u + b) - h) ** 2)
        fps.append(h); qs.append(float(q))
    return np.array(fps), np.array(qs)


def classify_and_dir(fps, qs, W, Win, u, b, alpha=0.2, q_thr=1e-6):
    """Return (motif_type, direction) from the recovered slow points."""
    good = fps[qs < q_thr]
    if len(good) < 2:
        if len(good) == 0:
            return None, None
    # deduplicate
    uniq = [good[0]]
    for h in good[1:]:
        if min(np.linalg.norm(h - g) for g in uniq) > 1e-2:
            uniq.append(h)
    uniq = np.array(uniq)
    # line attractor: >=3 near-marginal points spanning ~1D
    if len(uniq) >= 3:
        X = uniq - uniq.mean(0)
        s = np.linalg.svd(X, compute_uv=False)
        if s[0] > 1e-3 and (len(s) < 2 or s[1] < 0.3 * s[0]):
            return "line_attractor", X[np.argmax(np.linalg.norm(X, axis=1))] / (np.linalg.norm(X[np.argmax(np.linalg.norm(X, axis=1))]) + 1e-12)
    # else classify the lowest-q point by its Jacobian
    h = uniq[np.argmin([0.5 * np.sum((softplus(W @ h + Win @ u + b) - h) ** 2) for h in uniq])]
    D = 1.0 / (1.0 + np.exp(-(W @ h + Win @ u + b)))          # softplus'(x)=sigmoid(x)
    J = (1 - alpha) * np.eye(len(h)) + alpha * (D[:, None] * W)
    lam, V = np.linalg.eig(J)
    i = int(np.argmax(np.abs(lam)))
    v = np.real(V[:, i]); v = v / (np.linalg.norm(v) + 1e-12)
    return ("saddle_decision" if np.abs(lam[i]) > 1.0 else "point_attractor"), v


def motif_dirs(params, rule, hp, ind):
    """Motif directions (in active-unit coords) for a held-out rule at one snapshot."""
    W, Win, b = params["W_rec"], params["W_in"].T, params["b_rec"]   # Win: (n_rnn, n_input)
    einfo, trial = epoch_inputs(rule, hp)
    rng = np.random.default_rng(0)
    out = {}
    for e, (u, xfull, (a, bb)) in einfo.items():
        u = u.mean(0) if u.ndim > 1 else u                    # (n_input,)
        # seeds from trajectory states in this epoch (numpy rollout not needed: use zeros + noise +
        # random small states); a light pool suffices for slow points
        seeds = [rng.standard_normal(W.shape[0]) * s for s in (0.1, 0.5, 1.0) for _ in range(40)]
        seeds += [np.zeros(W.shape[0])]
        fps, qs = find_fixed_points(W, Win, b, u, seeds)
        mt, v = classify_and_dir(fps, qs, W, Win, u, b)
        if v is not None:
            out[e] = {"motif": mt, "v_act": v[ind]}           # restrict to base active units
    return out


def main():
    from flexsym.core.train import get_default_hp
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--hold-out", default="delayanti,dmcgo")
    ap.add_argument("--k", type=int, default=51)
    ap.add_argument("--motifs", action="store_true", help="also run Part B (numpy FP motif movement)")
    ap.add_argument("--motif-every", type=int, default=3)
    args = ap.parse_args()
    root = Path(args.root)
    held = args.hold_out.split(",")

    base = load_params(root / "base" / "model_params.npz")
    ind = base_active_units(root / "base")
    print(f"base active units: {len(ind)}", flush=True)
    lab0, S0 = wl_subspace(base["W_rec"], ind, args.k)
    floor = restart_floor(base["W_rec"], ind, args.k)
    print(f"restart-noise floor: angle {floor['angle_mean']:.1f}+-{floor['angle_sd']:.1f} deg, "
          f"msca {floor['msca_mean']:.3f}, ARI {floor['ari_mean']:.3f}", flush=True)

    snaps = sorted(glob.glob(str(root / "finetune" / "snap_*.npz")))
    hp = get_default_hp("all"); hp["rng"] = np.random.RandomState(0)
    rows = []
    for i, sp in enumerate(snaps):
        p = load_params(sp)
        step = int(Path(sp).stem.split("_")[1])
        lab_t, S_t = wl_subspace(p["W_rec"], ind, args.k)
        ang, msca = princ(S0, S_t)
        ari = adjusted_rand_score(lab0, lab_t)
        perf = p["perf"]
        row = {"step": step, "angle_vs_S0": ang, "msca_vs_S0": msca, "ari_vs_lab0": ari,
               "held_perf": {r: perf.get(r) for r in held}}
        rows.append(row)
        print(f"snap {step:6d}: S_t vs S0 angle {ang:5.1f} deg (floor {floor['angle_mean']:.1f}) "
              f"msca {msca:.3f} ARI {ari:.3f} | held perf {[round(perf.get(r,0),2) for r in held]}",
              flush=True)

    # Part B: motif movement in fixed base coordinates
    motif_track = {}
    if args.motifs:
        idx = list(range(0, len(snaps), args.motif_every))
        if idx and idx[-1] != len(snaps) - 1:
            idx.append(len(snaps) - 1)
        M0 = None
        for i in idx:
            p = load_params(snaps[i]); step = int(Path(snaps[i]).stem.split("_")[1])
            for rule in held:
                md = motif_dirs(p, rule, hp, ind)
                for e, m in md.items():
                    key = f"{rule}/{e}"
                    cont = float(np.linalg.norm(S0.T @ m["v_act"]) / (np.linalg.norm(m["v_act"]) + 1e-12))
                    motif_track.setdefault(key, []).append(
                        {"step": step, "motif": m["motif"], "containment_S0": cont, "v": m["v_act"].tolist()})
                    print(f"  motif {key} @step {step}: {m['motif']} containment_S0 {cont:.3f}", flush=True)
        # drift: angle of each motif vs its final direction
        for key, seq in motif_track.items():
            vf = np.array(seq[-1]["v"])
            for s in seq:
                v = np.array(s["v"])
                s["angle_vs_final"] = float(np.degrees(np.arccos(np.clip(np.abs(v @ vf) /
                                       (np.linalg.norm(v) * np.linalg.norm(vf) + 1e-12), 0, 1))))
                s.pop("v")

    out = {"held_out": held, "base_active": int(len(ind)), "restart_floor": floor,
           "stability": rows, "motif_track": motif_track}
    (root / "transfer_analysis.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {root/'transfer_analysis.json'}")


if __name__ == "__main__":
    main()
