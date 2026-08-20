"""Do the backbone results depend on the SIGNED-WL partition, or would any sensible graph
partition at matched k give the same coordinates?

We build, per network and per k, four partitions of the active recurrent block:
  * WL        signed Weisfeiler-Leman role groups (the partition used throughout the report);
  * spectral  normalised spectral clustering on the symmetrised absolute weight affinity
              |W|+|W|^T, k clusters;
  * modularity greedy modularity communities on the same weighted graph, forced to k communities;
  * random    size-matched random partitions (mean over a small ensemble), as the floor.

For each partition we evaluate the three metrics that define Results 1/3/4, all as properties of a
label vector (so they apply to any partition):
  leakage     mean fraction of the top-m W_rec eigenvectors' energy OUTSIDE the group-constant
              subspace (lower = more lumpable); Result 1.
  containment median motif-direction containment ||P p||/||p|| (higher = motifs live in the
              subspace); Result 3.
  top3        fraction of aggregate motif energy on the top-3 quotient eigenmodes (higher = more
              concentrated); Result 4.
Signed-rank across the ten networks compares WL to spectral and to modularity at each k.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from scipy.stats import wilcoxon
from sklearn.cluster import SpectralClustering
import networkx as nx

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.run.motif_quotient_prediction import condition_jacobians, load_conditions, indicator
from flexsym.run.motif_quotient_mode_concentration import quotient_modes
from flexsym.run.wrec_eigenbasis_null import contained, top3_energy

FP = paths.RESULTS_ROOT / "motif_fixed_points"
NET = lambda s: f"LeakyRNN_softplus_diag_256_all_{s}"
K_GRID = [12, 20, 30]


def spectral_labels(W, k, seed=0):
    A = np.abs(W) + np.abs(W.T)
    np.fill_diagonal(A, 0.0)
    sc = SpectralClustering(n_clusters=k, affinity="precomputed", assign_labels="discretize",
                            random_state=seed)
    return sc.fit_predict(A)


def modularity_labels(W, k):
    A = np.abs(W) + np.abs(W.T); np.fill_diagonal(A, 0.0)
    G = nx.from_numpy_array(A)
    comms = nx.community.greedy_modularity_communities(G, weight="weight", best_n=k, cutoff=k)
    lab = np.empty(len(W), int)
    for c, nodes in enumerate(comms):
        for n in nodes:
            lab[n] = c
    return lab


def leakage(W, labels, m=None):
    """Mean fraction of top-m |lambda| eigenvector energy outside col(H)."""
    H, sizes = indicator(labels)
    S = H / np.sqrt(sizes)[None, :]
    lam, V = np.linalg.eig(W)
    order = np.argsort(-np.abs(lam))
    m = m or (labels.max() + 1)
    frac = []
    for j in order[:m]:
        v = np.real(V[:, j]); v = v / (np.linalg.norm(v) + 1e-12)
        frac.append(1.0 - np.linalg.norm(S.T @ v) ** 2)
    return float(np.mean(frac))


def metrics(W, labels, P):
    H, sizes = indicator(labels)
    S = H / np.sqrt(sizes)[None, :]
    Mq, _ = quotient_modes(W, labels)
    return {
        "leakage": leakage(W, labels),
        "containment": float(np.median([contained(S, p) for p in P])),
        "top3": top3_energy(Mq, P),
    }


def run_net(net, kg, n_rand, rng):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], float)
    ind = np.asarray(wl["ind_active"], int)
    jbars = condition_jacobians(net, ind, W)
    conds = load_conditions(net, ind, W, jbars)
    P = np.array([c["v"] for c in conds])
    ls = wl["labels_struct_by_k"]
    out = {}
    for k in kg:
        role = np.asarray(ls[_row_for_k(ls, k)], int)
        kk = int(role.max() + 1)
        _, sizes = np.unique(role, return_counts=True)
        parts = {"wl": role, "spectral": spectral_labels(W, kk), "modularity": modularity_labels(W, kk)}
        cell = {name: metrics(W, lab, P) for name, lab in parts.items()}
        # random floor (size-matched to WL), mean over ensemble
        rnd = []
        for _ in range(n_rand):
            perm = np.repeat(np.arange(len(sizes)), sizes); rng.shuffle(perm)
            rnd.append(metrics(W, perm, P))
        cell["random"] = {m: float(np.mean([r[m] for r in rnd])) for m in ("leakage", "containment", "top3")}
        cell["k"] = kk
        out[k] = cell
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9"); ap.add_argument("--n-rand", type=int, default=20)
    args = ap.parse_args()
    seeds = []
    for part in args.seeds.split(","):
        if "-" in part:
            a, b = part.split("-"); seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    rng = np.random.default_rng(0)

    per = {}
    for s in seeds:
        net = NET(s)
        if not (FP / net / "participation_cache_v2.npz").exists():
            print(f"[skip] {net}"); continue
        per[net] = run_net(net, K_GRID, args.n_rand, rng)
        print(f"[done] {net}", flush=True)

    for metric, better in (("leakage", "lower"), ("containment", "higher"), ("top3", "higher")):
        print(f"\n=== {metric} ({better} = more structure): median over seeds ===")
        print(f"  {'k':>4} | {'WL':>8} {'spectral':>9} {'modular':>9} {'random':>8} | {'WL vs spec p':>12} {'WL vs mod p':>11}")
        for k in K_GRID:
            rows = [per[n][k] for n in per if k in per[n]]
            if not rows: continue
            wl_ = [r["wl"][metric] for r in rows]; sp = [r["spectral"][metric] for r in rows]
            mo = [r["modularity"][metric] for r in rows]; rn = [r["random"][metric] for r in rows]
            psp = wilcoxon(wl_, sp).pvalue if len(wl_) > 1 else np.nan
            pmo = wilcoxon(wl_, mo).pvalue if len(wl_) > 1 else np.nan
            print(f"  {k:>4} | {np.median(wl_):>8.3f} {np.median(sp):>9.3f} {np.median(mo):>9.3f} "
                  f"{np.median(rn):>8.3f} | {psp:>12.3f} {pmo:>11.3f}")

    (FP / "partition_family_comparison.json").write_text(json.dumps(per, indent=2, default=float))
    print(f"\nwrote {FP/'partition_family_comparison.json'}")


if __name__ == "__main__":
    main()
