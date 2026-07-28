"""Causal test: does preserving the connectivity quotient Q_struct preserve the computation?

We resample the trained recurrent weight matrix under two surrogates that share the SAME global
weight multiset but differ in whether the block (group x group) structure is preserved, inject each
back into the network (all other weights fixed, no retraining), and measure task performance on the
fifteen tasks:

  Q-preserving   permute residuals WITHIN each (g_i, g_j) role block -> preserves every block mean,
                 hence Q_struct = D^-1 H^T W_a H exactly, while destroying within-block wiring.
  partition-agnostic   permute ALL active-active entries globally -> same weight multiset, but
                 block means (and Q_struct) destroyed.

Both are matched to the trained matrix's global weight distribution; they differ only in whether
the coarse quotient is retained. If Q-preserving retains more performance than partition-agnostic,
the coarse block structure carries part of the computation (a causal counterpart to the
correlational quotient results). The original (unperturbed) network is the reference ceiling.
Only the active-active submatrix is resampled; inactive rows/cols and the input/output maps are
left as trained.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_reuse_control import _row_for_k

FP = paths.REPO_ROOT / "results/motif_fixed_points"
NET = lambda s: f"LeakyRNN_softplus_diag_256_all_{s}"
N_INPUT = 20


def model_dir(net):
    for store in (paths.ORIGINAL_STORE, paths.STAGED_STORE):
        hit = glob.glob(str(store / net))
        if hit:
            return hit[0]
    raise FileNotFoundError(net)


def block_permute(Wa, labels, rng):
    """Permute residuals within each (g_i,g_j) block; preserves all block means (Q_struct)."""
    out = Wa.copy()
    cats = np.unique(labels)
    for gi in cats:
        ri = np.where(labels == gi)[0]
        for gj in cats:
            rj = np.where(labels == gj)[0]
            block = Wa[np.ix_(ri, rj)]
            flat = block.ravel().copy()
            rng.shuffle(flat)
            out[np.ix_(ri, rj)] = flat.reshape(block.shape)
    return out


def global_permute(Wa, rng):
    """Permute ALL entries globally; same multiset, destroys block means."""
    flat = Wa.ravel().copy(); rng.shuffle(flat)
    return flat.reshape(Wa.shape)


def eval_perf(sess, model, tools, generate_trials, get_perf, hp, rules, batch):
    accs = []
    for r in rules:
        trial = generate_trials(r, hp, "test", noise_on=False, batch_size=batch)
        y_hat = sess.run(model.y_hat, feed_dict=tools.gen_feed_dict(model, trial, hp))
        accs.append(float(np.mean(get_perf(y_hat, trial.y_loc))))
    return float(np.mean(accs)), accs


def run_net(net, k, n_draw, batch):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from flexsym.core.network import Model, get_perf
    from flexsym.core.task import generate_trials, rules_dict
    from flexsym.core import tools

    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    ind = np.asarray(wl["ind_active"], int)
    ls = wl["labels_struct_by_k"]
    role = np.asarray(ls[_row_for_k(ls, k)], int)
    rules = rules_dict["all"]

    mdir = model_dir(net)
    model = Model(mdir, sigma_rec=0)
    kvar = [v for v in tf.global_variables() if "kernel" in v.name][0]
    ph = tf.placeholder(tf.float32, kvar.shape)
    assign = tf.assign(kvar, ph)

    rng = np.random.default_rng(0)
    with tf.Session() as sess:
        model.restore()
        K0 = sess.run(kvar).copy()                       # (n_input+n_rnn, n_rnn)
        Wr0 = K0[N_INPUT:, :].copy()                     # (n_rnn, n_rnn)
        Wa0 = Wr0[np.ix_(ind, ind)].copy()

        def set_and_eval(Wr):
            K = K0.copy(); K[N_INPUT:, :] = Wr
            sess.run(assign, feed_dict={ph: K})
            return eval_perf(sess, model, tools, generate_trials, get_perf, model.hp, rules, batch)

        orig_mean, _ = set_and_eval(Wr0)

        qp, pa, qp_eig, pa_eig = [], [], [], []
        lam0 = np.sort(np.abs(np.linalg.eigvals(Wa0)))[::-1][:10]   # top-10 |lambda| of trained
        for _ in range(n_draw):
            Waq = block_permute(Wa0, role, rng)
            Wr = Wr0.copy(); Wr[np.ix_(ind, ind)] = Waq
            qp.append(set_and_eval(Wr)[0])
            qp_eig.append(float(np.mean(np.abs(np.sort(np.abs(np.linalg.eigvals(Waq)))[::-1][:10] - lam0))))
            Wag = global_permute(Wa0, rng)
            Wr = Wr0.copy(); Wr[np.ix_(ind, ind)] = Wag
            pa.append(set_and_eval(Wr)[0])
            pa_eig.append(float(np.mean(np.abs(np.sort(np.abs(np.linalg.eigvals(Wag)))[::-1][:10] - lam0))))

    tf.reset_default_graph()
    return {"orig": orig_mean, "qp_mean": float(np.mean(qp)), "qp_sd": float(np.std(qp)),
            "pa_mean": float(np.mean(pa)), "pa_sd": float(np.std(pa)),
            "qp_eig_err": float(np.mean(qp_eig)), "pa_eig_err": float(np.mean(pa_eig)),
            "qp": qp, "pa": pa, "k": int(role.max() + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,4,9"); ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n-draw", type=int, default=5); ap.add_argument("--batch", type=int, default=40)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    per = {}
    for s in seeds:
        net = NET(s)
        per[net] = run_net(net, args.k, args.n_draw, args.batch)
        r = per[net]
        print(f"[{net}] k={r['k']}  orig {r['orig']:.3f} | Q-preserving {r['qp_mean']:.3f}"
              f"+-{r['qp_sd']:.3f} | partition-agnostic {r['pa_mean']:.3f}+-{r['pa_sd']:.3f}"
              f" || top-10|lam| err: QP {r['qp_eig_err']:.3f} vs PA {r['pa_eig_err']:.3f}", flush=True)

    from scipy.stats import wilcoxon
    print("\n=== causal block-model resample (mean over seeds, k=%d) ===" % args.k)
    o = np.mean([per[n]["orig"] for n in per])
    qp = np.array([per[n]["qp_mean"] for n in per]); pa = np.array([per[n]["pa_mean"] for n in per])
    p = wilcoxon(qp, pa).pvalue if len(per) > 1 else np.nan
    print(f" orig {o:.3f} | Q-preserving {qp.mean():.3f} | partition-agnostic {pa.mean():.3f} "
          f"| Q>agnostic {int((qp>pa).sum())}/{len(per)} p={p:.3f}")
    qe = np.array([per[n]["qp_eig_err"] for n in per]); pe = np.array([per[n]["pa_eig_err"] for n in per])
    pei = wilcoxon(qe, pe).pvalue if len(per) > 1 else np.nan
    print(f" top-10 |lambda| error vs trained: Q-preserving {qe.mean():.3f} | partition-agnostic {pe.mean():.3f} "
          f"| QP<PA {int((qe<pe).sum())}/{len(per)} p={pei:.3f}  (lower = slow spectrum better preserved)")

    (FP / "causal_block_resample.json").write_text(json.dumps(per, indent=2, default=float))
    print(f"\nwrote {FP/'causal_block_resample.json'}")


if __name__ == "__main__":
    main()
