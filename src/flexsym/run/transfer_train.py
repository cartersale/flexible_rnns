"""Transfer-learning driver: train a base net on all-but-held-out tasks, then fine-tune on the
held-out tasks while saving DENSE weight snapshots for the coordinate-stability / motif-movement
analysis.

Two phases:
  base      train from scratch on the 15 tasks MINUS the held-out set (--hold-out), to target
            performance or a wall-clock / step cap. Weights -> <root>/base/.
  finetune  load the base weights and continue training on ONLY the held-out tasks, dumping a
            model_params snapshot every --snap-steps steps to <root>/finetune/snap_<step>.npz.
            The base weights are snapshot 0 (pre-transfer reference).

Snapshots are plain model_params.npz (w_in incl. W_rec = w_in[-n_rnn:], w_out, biases) plus the
current per-task performance, which is all the downstream analysis needs (WL partition + FP motifs
are recomputed offline from W_rec / the restored graph).
"""
from __future__ import annotations
import argparse, glob, json, os, shutil
from pathlib import Path
import numpy as np


def _hp(size, rules, seed):
    from flexsym.core.train import get_default_hp
    hp = get_default_hp("all")
    hp.update({"activation": "softplus", "w_rec_init": "diag", "n_rnn": size,
               "l2_h": 1e-6, "l2_weight": 1e-6, "sigma_rec": 0.05, "sigma_x": 0.1,
               "rnn_type": "LeakyRNN", "rule_trains": rules, "rules": rules,
               "target_perf": 0.97})
    hp["rule_probs"] = list(np.ones(len(rules)) / len(rules))
    return hp


def dump_snapshot(sess, model, out_npz, rules, hp):
    import tensorflow.compat.v1 as tf
    from flexsym.core.network import get_perf
    from flexsym.core.task import generate_trials
    from flexsym.core import tools
    params = {v.name.split(":")[0].replace("/", "_"): sess.run(v) for v in tf.trainable_variables()}
    # canonical keys expected by load_w_rec-style code
    kernel = [v for v in tf.trainable_variables() if "kernel" in v.name][0]
    b_rec = [v for v in tf.trainable_variables() if "rnn" in v.name and "bias" in v.name][0]
    w_in = sess.run(kernel)                                   # (n_input+n_rnn, n_rnn)
    w_out = [v for v in tf.trainable_variables() if "output" in v.name and "weight" in v.name][0]
    b_out = [v for v in tf.trainable_variables() if "output" in v.name and "bias" in v.name][0]
    perf = {}
    for r in rules:
        trial = generate_trials(r, hp, "test", noise_on=False, batch_size=100)
        yhat = sess.run(model.y_hat, feed_dict=tools.gen_feed_dict(model, trial, hp))
        perf[r] = float(np.mean(get_perf(yhat, trial.y_loc)))
    np.savez(out_npz, w_in=w_in, b_rec=sess.run(b_rec), w_out=sess.run(w_out),
             b_out=sess.run(b_out), perf=json.dumps(perf))
    return perf


def train_phase(model_dir, rules, size, seed, max_steps, max_hours, load_dir=None,
                snap_dir=None, snap_steps=None):
    import time
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from flexsym.core.network import Model
    from flexsym.core import tools
    from flexsym.core.train import do_eval
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    hp = _hp(size, rules, seed); hp["rng"] = np.random.RandomState(seed); hp["seed"] = seed
    tools.save_hp(hp, model_dir)
    tf.reset_default_graph()
    m = Model(model_dir, hp=hp)
    t0 = time.time(); snaps = []
    ckpt = os.path.join(model_dir, "model.ckpt")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # resume this dir's own checkpoint if present (robust to kills), else warm-start from load_dir
        if os.path.exists(ckpt + ".meta"):
            m.saver.restore(sess, ckpt); print(f"  resumed from {ckpt}", flush=True)
        elif load_dir is not None:
            m.saver.restore(sess, os.path.join(load_dir, "model.ckpt"))
        step = 0
        log = {"perf_min": [], "perf_avg": [], "step": []}
        if snap_dir is not None:
            Path(snap_dir).mkdir(parents=True, exist_ok=True)
            existing = sorted(glob.glob(os.path.join(snap_dir, "snap_*.npz")))
            step = int(Path(existing[-1]).stem.split("_")[1]) if existing else 0   # resume snap count
            if not existing:
                p = dump_snapshot(sess, m, os.path.join(snap_dir, f"snap_{step:06d}.npz"), rules, hp)
                snaps.append((step, min(p.values()), float(np.mean(list(p.values())))))
                print(f"  snap {step}: perf_min {min(p.values()):.3f}", flush=True)
        while step * hp["batch_size_train"] <= max_steps:
            if step % 500 == 0:
                pmin = _quick_perf(sess, m, rules, hp)
                log["step"].append(step); log["perf_min"].append(pmin)
                el = (time.time() - t0) / 3600
                print(f"  step {step} perf_min {pmin:.3f} ({el:.2f}h)", flush=True)
                if snap_dir is None:                     # base phase: persist progress periodically
                    m.saver.save(sess, ckpt)
                    if step % 3000 == 0:
                        dump_snapshot(sess, m, os.path.join(model_dir, "model_params.npz"), rules, hp)
                if pmin > hp["target_perf"]:
                    print("  reached target"); break
                if max_hours and el > max_hours:
                    print("  hit max_hours"); break
            sess.run(m.train_step); step += 1
            if snap_dir is not None and snap_steps and step % snap_steps == 0:
                p = dump_snapshot(sess, m, os.path.join(snap_dir, f"snap_{step:06d}.npz"), rules, hp)
                snaps.append((step, min(p.values()), float(np.mean(list(p.values())))))
                m.saver.save(sess, ckpt)
                print(f"  snap {step}: perf_min {min(p.values()):.3f}", flush=True)
        m.saver.save(sess, ckpt)
        dump_snapshot(sess, m, os.path.join(model_dir, "model_params.npz"), rules, hp)
    return snaps


def _quick_perf(sess, model, rules, hp):
    from flexsym.core.network import get_perf
    from flexsym.core.task import generate_trials
    from flexsym.core import tools
    ps = []
    for r in rules:
        trial = generate_trials(r, hp, "test", noise_on=False, batch_size=100)
        yhat = sess.run(model.y_hat, feed_dict=tools.gen_feed_dict(model, trial, hp))
        ps.append(float(np.mean(get_perf(yhat, trial.y_loc))))
    return float(np.min(ps))


def main():
    from flexsym.core.task import rules_dict
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--size", type=int, default=256); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hold-out", default="delayanti,dmcgo")
    ap.add_argument("--phase", choices=["base", "finetune", "both"], default="both")
    ap.add_argument("--base-max-steps", type=float, default=4e5)
    ap.add_argument("--base-max-hours", type=float, default=6.0)
    ap.add_argument("--ft-max-steps", type=float, default=8e4)
    ap.add_argument("--ft-max-hours", type=float, default=2.0)
    ap.add_argument("--snap-steps", type=int, default=1500)
    args = ap.parse_args()

    all_rules = rules_dict["all"]
    held = args.hold_out.split(",")
    base_rules = [r for r in all_rules if r not in held]
    root = Path(args.root)
    print(f"held out: {held}\nbase rules ({len(base_rules)}): {base_rules}", flush=True)

    if args.phase in ("base", "both"):
        print("=== BASE PHASE ===", flush=True)
        train_phase(root / "base", base_rules, args.size, args.seed,
                    args.base_max_steps, args.base_max_hours)
    if args.phase in ("finetune", "both"):
        print("=== FINETUNE PHASE (held-out only) ===", flush=True)
        snaps = train_phase(root / "finetune_model", held, args.size, args.seed,
                            args.ft_max_steps, args.ft_max_hours, load_dir=str(root / "base"),
                            snap_dir=str(root / "finetune"), snap_steps=args.snap_steps)
        (root / "finetune_snaps.json").write_text(json.dumps(snaps, default=float, indent=2))
    print("TRANSFER_DONE", flush=True)


if __name__ == "__main__":
    main()
