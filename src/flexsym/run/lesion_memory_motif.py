"""Memory-motif targeted lesion test: do slow-mode cells break memory tasks?

Ties R5 (lesions) to R2 (the slow memory mode). For each network we identify the
slow (line-attractor / memory) mode of the delay-period Jacobian and its loading
on each structural cell, then ask whether cells that carry the memory mode, when
lesioned, selectively impair delay-dependent (memory) tasks over non-memory
(reaction / stimulus-present) tasks. Uses the per-cell lesion deficit matrices
already saved by run_lesion_causal_batch.
"""
from __future__ import annotations

from flexsym import paths
import sys
from pathlib import Path
import numpy as np

_p = np.load
_o = _p.__closure__[0].cell_contents if getattr(_p, "__closure__", None) else _p
def _dl(*a, **k):
    k["allow_pickle"] = True
    return _o(*a, **k)
np.load = _dl

from flexsym.structural.signed_input_wl_structural_roles import apply_matrix_variant, load_w_rec  # noqa: E402
from flexsym.structural.structural_cell_activity_coherence import compute_hidden_by_epoch  # noqa: E402

REPO = paths.REPO_ROOT
DET = REPO / "results/signed_input_wl_paper/lesion_causal"
ORIG = paths.ORIGINAL_STORE
STAGED = paths.STAGED_STORE
ALPHA = 0.2
MEMORY = {"delaygo", "delayanti", "delaydm1", "delaydm2", "contextdelaydm1",
          "contextdelaydm2", "multidelaydm", "dmsgo", "dmsnogo", "dmcgo", "dmcnogo"}
NONMEM = {"fdgo", "reactgo", "fdanti", "reactanti"}


def resolve(m):
    for s in (ORIG, STAGED):
        if (s / m).exists() and (s / m / "model.ckpt.index").exists():
            return s / m
    return None


def slow_mode_loadings(md, ind, labels):
    WT = apply_matrix_variant(load_w_rec(md), "transpose")[np.ix_(ind, ind)]
    hid = compute_hidden_by_epoch(md, [("delaygo", "delay1")], batch_size=512)
    np.load = _dl
    hstar = hid[("delaygo", "delay1")][:, :, ind][-5:].mean(0).mean(0)
    g = 1 - np.exp(-np.maximum(hstar, 1e-6))
    M = g[:, None] * WT
    w, V = np.linalg.eig(M)
    Jw = (1 - ALPHA) + ALPHA * w
    v = np.real(V[:, np.argmin(np.abs(Jw - 1))])
    v = np.abs(v) / (np.abs(v).max() + 1e-12)
    cells = sorted(np.unique(labels))
    return np.array([v[labels == c].mean() for c in cells]), cells


def main():
    from scipy.stats import wilcoxon, binomtest
    rows = []
    for seed in range(10):
        model = f"LeakyRNN_softplus_diag_256_all_{seed}"
        det = DET / f"lesion_detail_{model}.npz"
        md = resolve(model)
        if not det.exists() or md is None:
            print(f"skip {model}"); continue
        d = np.load(det, allow_pickle=True)
        rules = [str(x) for x in d["rules"]]
        deficit = np.asarray(d["struct_deficit"], dtype=float)  # [n_cells, 15]
        labels = np.asarray(d["labels"], dtype=np.int64)
        ind = np.asarray(d["ind_active"], dtype=np.int64)
        mem_idx = [i for i, r in enumerate(rules) if r in MEMORY]
        non_idx = [i for i, r in enumerate(rules) if r in NONMEM]
        mem_sel = deficit[:, mem_idx].mean(1) - deficit[:, non_idx].mean(1)

        loading, cells = slow_mode_loadings(md, ind, labels)
        np.load = _dl
        n = min(len(loading), len(mem_sel))
        loading, mem_sel = loading[:n], mem_sel[:n]
        # correlation across cells
        r = np.corrcoef(loading, mem_sel)[0, 1] if np.std(loading) > 0 and np.std(mem_sel) > 0 else np.nan
        # memory-mode cells (top third by loading) vs rest
        thr = np.quantile(loading, 2/3)
        top = mem_sel[loading >= thr]; rest = mem_sel[loading < thr]
        rows.append(dict(seed=seed, r=r, top_memsel=float(np.mean(top)),
                         rest_memsel=float(np.mean(rest)), n=n))
        print(f"seed {seed}: corr(loading,mem-selectivity)={r:+.3f}  "
              f"top-mode cells mem-sel={np.mean(top):+.3f} vs rest={np.mean(rest):+.3f}", flush=True)
        import tensorflow.compat.v1 as tf; tf.reset_default_graph()

    R = np.array([x["r"] for x in rows])
    top = np.array([x["top_memsel"] for x in rows]); rest = np.array([x["rest_memsel"] for x in rows])
    print("\n=== across seeds ===")
    print(f"corr(slow-mode loading, memory-selectivity of deficit): mean r={np.nanmean(R):+.3f}, "
          f"{int((R>0).sum())}/{len(R)} >0, sign P={binomtest(int((R>0).sum()),len(R),0.5).pvalue:.4f}")
    diff = top - rest
    print(f"memory-mode cells vs rest (deficit memory-selectivity): {np.mean(top):+.3f} vs {np.mean(rest):+.3f}, "
          f"{int((diff>0).sum())}/{len(diff)} top>rest, Wilcoxon P={wilcoxon(diff).pvalue:.4f}")
    import csv
    with (DET / "memory_motif_test.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {DET/'memory_motif_test.csv'}")


if __name__ == "__main__":
    main()
