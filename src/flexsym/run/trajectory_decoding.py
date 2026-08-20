"""Trajectory-level decoding from the k group-averaged quotient coordinates.

Question: do the k connectivity-quotient coordinates (group means of the state) carry the
computation, or does coarse-graining to k groups discard the task-relevant state? We decode two
trajectory variables from the instantaneous hidden state and compare feature sets at matched
dimension k:
  full   all active units (dimension n_active ~ 210)   -- ceiling
  wl     WL role-group means  z = (h_active . H)/sizes  (dimension k)
  rand-part   size-matched RANDOM partition group means (dimension k) -- partition floor
  rand-proj   random Gaussian linear projection to k dims             -- generic-compression floor
Decoded variables: (a) task identity (15-way) and (b) epoch/phase (fixation/stim/delay/response),
pooled across tasks. Balanced multinomial logistic regression, 5-fold stratified CV, accuracy.

If wl approaches full and beats rand-part / rand-proj at matched k, the coarse coordinates retain
the task-relevant state; if wl ~ rand-proj, the group-averaging is just a generic compression.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from flexsym import paths
from flexsym.core.task import rules_dict
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.structural.structural_cell_activity_coherence import compute_hidden_by_epoch
from flexsym.structural.structural_cell_cross_task_fractionation import filter_valid_task_epochs

FP = paths.RESULTS_ROOT / "motif_fixed_points"
NET = lambda s: f"LeakyRNN_softplus_diag_256_all_{s}"
N_MAX = 15000
EPOCHS = ["fix1", "stim1", "delay1", "stim2", "delay2", "go1"]
PHASE = {"fix1": "fixation", "stim1": "stimulus", "stim2": "stimulus",
         "delay1": "delay", "delay2": "delay", "go1": "response"}


def model_dir(net):
    import glob
    for store in (paths.ORIGINAL_STORE, paths.STAGED_STORE):
        hit = glob.glob(str(store / net))
        if hit:
            return hit[0]
    raise FileNotFoundError(net)


def group_means(Ha, sizes, X):
    """X: (samples, n_active) -> (samples, k) group means."""
    return (X @ Ha) / sizes


def indicator_np(labels):
    cats = np.unique(labels); H = np.zeros((len(labels), len(cats)))
    for j, c in enumerate(cats):
        H[labels == c, j] = 1.0
    return H, H.sum(0)


def decode(X, y, seed=0):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=400, C=1.0, multi_class="multinomial"))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean())


def run_net(net, k, batch, per_epoch, rng):
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    ind = np.asarray(wl["ind_active"], int)
    ls = wl["labels_struct_by_k"]
    role = np.asarray(ls[_row_for_k(ls, k)], int)
    kk = int(role.max() + 1)

    rules = rules_dict["all"]
    from pathlib import Path
    mdir = Path(model_dir(net))
    keys = filter_valid_task_epochs(mdir, [(r, e) for r in rules for e in EPOCHS], batch)
    hid = compute_hidden_by_epoch(mdir, keys, batch_size=batch)

    X, task_y, phase_y = [], [], []
    for (r, e), h in hid.items():
        if h.shape[0] == 0:
            continue
        T = h.shape[0]
        tsel = np.linspace(0, T - 1, min(per_epoch, T)).astype(int)
        block = h[tsel][:, :, ind].reshape(-1, len(ind))     # (samples, n_active)
        X.append(block)
        task_y += [rules.index(r)] * block.shape[0]
        phase_y += [PHASE[e]] * block.shape[0]
    X = np.concatenate(X, 0); task_y = np.array(task_y); phase_y = np.array(phase_y)
    if X.shape[0] > N_MAX:                                    # cap for tractable CV
        idx = rng.choice(X.shape[0], N_MAX, replace=False)
        X, task_y, phase_y = X[idx], task_y[idx], phase_y[idx]

    Hwl, swl = indicator_np(role)
    _, sizes = np.unique(role, return_counts=True)
    # size-matched random partition
    perm = np.repeat(np.arange(len(sizes)), sizes); rng.shuffle(perm)
    Hr, sr = indicator_np(perm)
    Gp = rng.standard_normal((len(ind), kk))                 # random projection

    feats = {
        "full": X,
        "wl": group_means(Hwl, swl, X),
        "rand_part": group_means(Hr, sr, X),
        "rand_proj": X @ Gp,
    }
    out = {"k": kk, "n_samples": int(X.shape[0])}
    for target, y in (("task", task_y), ("phase", phase_y)):
        out[target] = {name: decode(F, y) for name, F in feats.items()}
        out[target]["chance"] = float(np.unique(y, return_counts=True)[1].max() / len(y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,4,9"); ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--batch", type=int, default=40); ap.add_argument("--per-epoch", type=int, default=4)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    rng = np.random.default_rng(0)

    per = {}
    for s in seeds:
        net = NET(s)
        per[net] = run_net(net, args.k, args.batch, args.per_epoch, rng)
        r = per[net]
        print(f"[{net}] k={r['k']} n={r['n_samples']}", flush=True)
        for tgt in ("task", "phase"):
            d = r[tgt]
            print(f"   {tgt:5s}: full {d['full']:.3f} | wl {d['wl']:.3f} | rand-part {d['rand_part']:.3f} "
                  f"| rand-proj {d['rand_proj']:.3f} | chance {d['chance']:.3f}", flush=True)

    from scipy.stats import wilcoxon
    print("\n=== mean over seeds (k=%d) ===" % args.k)
    for tgt in ("task", "phase"):
        agg = {name: np.mean([per[n][tgt][name] for n in per]) for name in ("full", "wl", "rand_part", "rand_proj", "chance")}
        wl_ = [per[n][tgt]["wl"] for n in per]; rp = [per[n][tgt]["rand_proj"] for n in per]
        rpa = [per[n][tgt]["rand_part"] for n in per]
        p_proj = wilcoxon(wl_, rp).pvalue if len(per) > 1 else np.nan
        p_part = wilcoxon(wl_, rpa).pvalue if len(per) > 1 else np.nan
        nbeat = sum(a > b for a, b in zip(wl_, rp))
        print(f" {tgt:5s}: full {agg['full']:.3f} | wl {agg['wl']:.3f} | rand-part {agg['rand_part']:.3f} "
              f"| rand-proj {agg['rand_proj']:.3f} | chance {agg['chance']:.3f} "
              f"|| wl>rand-proj {nbeat}/{len(per)} p={p_proj:.3f}; wl vs rand-part p={p_part:.3f}")

    (FP / "trajectory_decoding.json").write_text(json.dumps(per, indent=2, default=float))
    print(f"\nwrote {FP/'trajectory_decoding.json'}")


if __name__ == "__main__":
    main()
