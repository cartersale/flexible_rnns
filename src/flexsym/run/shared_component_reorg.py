"""Compositional-reuse test: do cells re-form the same subgrouping for shared components?

Within each structural cell we cluster the units functionally in each of a broad set
of task-epochs, then compare all pairs of subpartitions. If reorganization is
compositional, the subgrouping should be more similar for epoch pairs that demand
the SAME computation (both memory-maintenance delays, both stimulus, both response)
than for pairs demanding DIFFERENT computations. Component = epoch type (delay1 /
stim1 / go1); shared = same epoch type across tasks, non-shared = different.
Aggregated per seed, tested across seeds.
"""
from __future__ import annotations
import sys, csv, glob
from pathlib import Path
import numpy as np

STEP_DIR = Path(__file__).resolve().parent
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))
_p = np.load
_o = _p.__closure__[0].cell_contents if getattr(_p, "__closure__", None) else _p
def _dl(*a, **k):
    k["allow_pickle"] = True
    return _o(*a, **k)
np.load = _dl

from sklearn.metrics import adjusted_rand_score  # noqa: E402
from structural_cell_activity_coherence import compute_hidden_by_epoch, load_structural_labels  # noqa: E402
from structural_cell_cross_task_fractionation import filter_valid_task_epochs  # noqa: E402
from structural_cell_subpartition_pilot import choose_functional_partition  # noqa: E402
from softplus_memory_subpartition_sweep import unit_features_for_period  # noqa: E402

REPO = STEP_DIR.parent
ORIG = Path("/Volumes/X9_Pro/flexible_multitask_data/multitask_processed/suppfig4")
STAGED = Path("/Volumes/X9_Pro/flexible_multitask_data/new_runs/staged")
OPK = {0: 41, 1: 51, 2: 38, 3: 14, 4: 14, 5: 14, 6: 14, 7: 14, 8: 14, 9: 15}
OUT = REPO / "results/signed_input_wl_paper/shared_component_reorg"
# broad battery: memory(delay1), stimulus(stim1), response(go1) across many tasks
BATTERY = [
    ("delaygo", "delay1"), ("delayanti", "delay1"), ("delaydm1", "delay1"), ("delaydm2", "delay1"),
    ("contextdelaydm1", "delay1"), ("multidelaydm", "delay1"),
    ("fdgo", "stim1"), ("fdanti", "stim1"), ("delaygo", "stim1"), ("delaydm1", "stim1"),
    ("reactgo", "go1"), ("reactanti", "go1"), ("fdgo", "go1"), ("delaygo", "go1"),
]


def resolve(m):
    for s in (ORIG, STAGED):
        if (s / m).exists() and (s / m / "model.ckpt.index").exists():
            return s / m
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed = []
    for seed, k in OPK.items():
        model = f"LeakyRNN_softplus_diag_256_all_{seed}"
        md = resolve(model)
        sweep = REPO / f"results/signed_input_wl_paper/{model}/transpose_selected_k_{k}/signed_input_wl_sweep.npz"
        if md is None or not sweep.exists():
            print(f"skip {model}"); continue
        labels, ind, variant = load_structural_labels(sweep, k)
        te = filter_valid_task_epochs(md, BATTERY, 512)
        np.load = _dl
        hidden = compute_hidden_by_epoch(md, te, batch_size=512)
        np.load = _dl

        cells = [c for c in np.unique(labels) if int((labels == c).sum()) >= 5]
        shared_pc, nonshared_pc = [], []
        for c in cells:
            pos = np.where(labels == c)[0]
            parts = {}
            for rule, ep in te:
                X = unit_features_for_period(hidden, rule, ep, ind, pos)
                r = choose_functional_partition(X, max_k=6)
                parts[(rule, ep)] = (np.asarray(r["labels"], dtype=np.int64), int(r["chosen_k"]))
            sh, no = [], []
            keys = list(parts)
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    (a, ka), (b, kb) = parts[keys[i]], parts[keys[j]]
                    if ka < 2 or kb < 2:
                        continue  # informative pairs only
                    ari = adjusted_rand_score(a, b)
                    (sh if keys[i][1] == keys[j][1] else no).append(ari)
            if sh and no:
                shared_pc.append(np.mean(sh)); nonshared_pc.append(np.mean(no))
        if not shared_pc:
            print(f"seed {seed}: no cells"); continue
        sh_m, no_m = np.mean(shared_pc), np.mean(nonshared_pc)
        per_seed.append(dict(seed=seed, k=k, ncell=len(shared_pc), n_epochs=len(te),
                             shared=sh_m, nonshared=no_m, diff=sh_m - no_m,
                             frac_pos=float(np.mean(np.array(shared_pc) > np.array(nonshared_pc)))))
        print(f"seed {seed}: n_ep={len(te)} ncell={len(shared_pc):>2}  shared={sh_m:.3f}  non-shared={no_m:.3f}  diff={sh_m-no_m:+.3f}  cells+={np.mean(np.array(shared_pc)>np.array(nonshared_pc)):.2f}", flush=True)

    from scipy.stats import wilcoxon, binomtest
    sh = np.array([r["shared"] for r in per_seed]); no = np.array([r["nonshared"] for r in per_seed]); diff = sh - no
    print("\n=== across seeds ===")
    print(f"shared-component ARI={sh.mean():.3f}  non-shared ARI={no.mean():.3f}  diff={diff.mean():+.3f}")
    print(f"seeds shared>non-shared: {int((diff>0).sum())}/{len(diff)}  sign P={binomtest(int((diff>0).sum()),len(diff),0.5).pvalue:.4f}  Wilcoxon P={wilcoxon(diff).pvalue:.4f}")
    with (OUT / "shared_component_reorg.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed[0].keys())); w.writeheader(); w.writerows(per_seed)
    print(f"wrote {OUT/'shared_component_reorg.csv'}")


if __name__ == "__main__":
    main()
