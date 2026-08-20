"""Do independently trained seeds converge on similar quotient-mode decompositions?

Modes have no cross-seed correspondence (mode j of seed 0 is not mode j of seed 1) and live in
different unit spaces, so we cannot compare them by index or by lifted direction. But the fifteen
TASKS are shared across all seeds, so we fingerprint each quotient mode by a seed-invariant
TASK PROFILE: the vector over the 15 tasks of how much each task's motifs load that mode. A mode's
eigenvalue |lambda| is also seed-invariant. Convergence = high-energy modes of different seeds have
matching task profiles (and similar |lambda|) beyond a task-label-shuffled null.

Per seed at fixed k:
  * orthonormal eigen-ordered quotient modes M (complex-aware, from motif_type_mode_profiles);
  * per mode j: aggregate motif energy E_j = sum_m (p_m . M_j)^2 and task profile
    T[j, r] = sum_{m in rule r} (p_m . M_j)^2 (a point in shared 15-task space); keep the top-m
    modes by E_j and L1-normalise their profiles.
Cross-seed: for every seed pair, Hungarian-match their top-m modes by task-profile cosine; record
the mean matched cosine and matched |lambda| gap. NULL: apply a random permutation of the 15 task
axis to one seed's profiles (breaks task identity, keeps profile shape), rematch. Convergence iff
observed matched cosine > null. A global view pools all top modes and reports each mode's dominant
task, to see which tasks recur across seeds.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon

from flexsym import paths
from flexsym.core.task import rules_dict
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.run.motif_quotient_prediction import condition_jacobians, load_conditions, indicator, quotient
from flexsym.run.motif_type_mode_profiles import ortho_eigbasis, lifted

FP = paths.RESULTS_ROOT / "motif_fixed_points"
NET = lambda s: f"LeakyRNN_softplus_diag_256_all_{s}"
RULES = rules_dict["all"]


def mode_fingerprints(net, k, top_m):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], float)
    ind = np.asarray(wl["ind_active"], int)
    jbars = condition_jacobians(net, ind, W)
    conds = load_conditions(net, ind, W, jbars)
    P = np.array([c["v"] for c in conds]); rules = [c["rule"] for c in conds]
    ls = wl["labels_struct_by_k"]
    role = np.asarray(ls[_row_for_k(ls, k)], int)
    H, sizes = indicator(role); S = H / np.sqrt(sizes)[None, :]
    Q = quotient(W, H, sizes)
    lam = np.abs(np.linalg.eigvals(Q)); lam.sort(); lam = lam[::-1]     # |lambda| desc (mode order)
    M = lifted(ortho_eigbasis(Q), S)                                    # (n_active, k), |lambda| order
    C = P @ M                                                           # (motifs, k)
    E = (C ** 2)                                                        # per-motif per-mode energy
    Emode = E.sum(0)                                                    # aggregate energy per mode
    # task profiles per mode
    T = np.zeros((M.shape[1], len(RULES)))
    for m, r in enumerate(rules):
        T[:, RULES.index(r)] += E[m]
    top = np.argsort(-Emode)[:top_m]
    prof = T[top]; prof = prof / (prof.sum(1, keepdims=True) + 1e-12)   # L1-normalised task profiles
    lam_top = (lam[top] if len(lam) >= M.shape[1] else lam[:len(top)])
    return {"prof": prof, "lam": np.asarray(lam_top),
            "prof_list": prof.tolist(), "lam_list": np.asarray(lam_top).tolist(),
            "dominant_task": [RULES[i] for i in prof.argmax(1)],
            "energy_frac": (Emode[top] / Emode.sum()).tolist()}


def cos_matrix(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return An @ Bn.T


def matched_cosine(A, B):
    Cm = cos_matrix(A, B)
    ri, ci = linear_sum_assignment(-Cm)
    return float(Cm[ri, ci].mean()), (ri, ci)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9"); ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--top-m", type=int, default=6); ap.add_argument("--n-perm", type=int, default=500)
    args = ap.parse_args()
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-"); seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    rng = np.random.default_rng(0)

    fp = {}
    for s in seeds:
        net = NET(s)
        if not (FP / net / "participation_cache_v2.npz").exists():
            print(f"[skip] {net}"); continue
        fp[s] = mode_fingerprints(net, args.k, args.top_m)
        print(f"[seed {s}] top-{args.top_m} modes dominant tasks: {fp[s]['dominant_task']}", flush=True)

    ss = list(fp)
    obs_pairs, null_pairs, lamgap = [], [], []
    for i in range(len(ss)):
        for j in range(i + 1, len(ss)):
            A, B = fp[ss[i]]["prof"], fp[ss[j]]["prof"]
            mc, (ri, ci) = matched_cosine(A, B)
            obs_pairs.append(mc)
            la, lb = fp[ss[i]]["lam"], fp[ss[j]]["lam"]
            lamgap.append(float(np.mean(np.abs(np.asarray(la)[ri] - np.asarray(lb)[ci]))))
            ndraw = []
            for _ in range(args.n_perm):
                perm = rng.permutation(len(RULES))
                ndraw.append(matched_cosine(A, B[:, perm])[0])
            null_pairs.append(float(np.mean(ndraw)))

    obs_pairs = np.array(obs_pairs); null_pairs = np.array(null_pairs)
    p = wilcoxon(obs_pairs, null_pairs).pvalue
    print(f"\n=== cross-seed quotient-mode convergence (task-profile matching, k={args.k}, top-{args.top_m}) ===")
    print(f" seed pairs: {len(obs_pairs)}")
    print(f" mean matched task-profile cosine: observed {obs_pairs.mean():.3f} vs task-shuffled null {null_pairs.mean():.3f}")
    print(f"   observed > null in {int((obs_pairs>null_pairs).sum())}/{len(obs_pairs)} pairs; signed-rank p={p:.4f}")
    print(f" mean matched |lambda| gap: {np.mean(lamgap):.3f}")

    # which tasks dominate across seeds
    from collections import Counter
    doms = Counter(t for s in fp for t in fp[s]["dominant_task"])
    print(f" dominant-task tally across all seeds' top modes: {dict(doms.most_common())}")

    clean = {str(s): {kk: vv for kk, vv in fp[s].items() if kk not in ("prof", "lam")} for s in fp}
    (FP / "motif_mode_cross_seed.json").write_text(json.dumps(
        {"per_seed": clean,
         "obs_matched_cosine": obs_pairs.tolist(), "null_matched_cosine": null_pairs.tolist(),
         "lam_gap": lamgap, "wilcoxon_p": float(p)}, indent=2, default=float))
    print(f"\nwrote {FP/'motif_mode_cross_seed.json'}")


if __name__ == "__main__":
    main()
