"""Train one softplus LeakyRNN from scratch, dumping front-loaded weight snapshots.

The nets trained by ``train_softplus_diag.py`` keep only the final checkpoint, so
nothing there can say *when* during training the connectivity coordinate system
settles relative to the motifs that live in it. This module trains the same
architecture on the same hyperparameters but persists the weights on a
log-spaced grid of steps, which is what the trajectory analyses consume:

    flexsym.run.trajectory_subspace   WL partition at EVERY snapshot (cheap)
    flexsym.run.trajectory_motifs     fixed points at ~5 snapshots (expensive)

Two things differ deliberately from ``train_softplus_diag.py``:

* ``--init`` is exposed (``diag`` or ``randortho``) so the diagonal-init nets can
  be compared against an unstructured-init control. Everything else -- activation,
  regularisation exponents, noise, learning rate -- is imported from that module
  rather than restated, so the two grids stay hyperparameter-identical.
* There is no early stop. Every net runs the full ``--max-trials`` budget, so the
  snapshot grid is identical across nets and the curves share an x-axis. A net
  that converges early simply keeps training; the step at which ``perf_min``
  first crosses ``--converged-perf`` is recorded in the manifest for annotation.

Snapshots are plain ``.npz`` holding the full trainable state (``w_in`` and its
``W_rec`` tail, ``b_rec``, ``w_out``, ``b_out``) plus step, trials, elapsed wall
clock, and per-rule performance and loss at that step -- everything the offline
analyses need, since both the WL partition and the numpy fixed-point finder work
from ``W_rec`` directly.

Run one net:

    python -m flexsym.training.train_trajectory --init diag --seed 0

or let ``run_trajectory_grid.py`` fan the 5 x 2 grid across cores.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from flexsym import paths
from flexsym.training.train_softplus_diag import (
    ACTIVATION, L1H, L1W, L2H, L2W, LR, RNN_TYPE, RULESET,
    SIGMA_REC, SIGMA_X, W_REC_COEFF, _coeff,
)

INIT_CHOICES = ("diag", "randortho")

#: Trials, not steps -- `train()` and this loop both budget in trials
#: (trials = step * batch_size_train). 1e8 matches the existing 256 grid.
DEFAULT_MAX_TRIALS = 1e8

#: Mantissas of the front-loaded ramp: 1, 2, 5 per decade, i.e.
#: 1, 2, 5, 10, 20, 50, 100, ...
RAMP_MANTISSAS = (1, 2, 5)

#: Points in the linear tail that follows the ramp. See `snapshot_steps`.
DEFAULT_N_TAIL = 10


def snapshot_steps(max_trials: float, batch_size: int,
                   n_tail: int = DEFAULT_N_TAIL,
                   tail_start: int | None = None) -> np.ndarray:
    """Front-loaded snapshot grid, in optimiser steps: 0, a 1-2-5 decade ramp,
    then a linear tail.

    Heavy front-loading is the whole point: on the existing 256 nets `perf_avg`
    goes 0.10 -> 0.95 inside the first 13M of 100M trials and is flat after ~25M,
    so a linear grid would spend two thirds of its snapshots on a plateau.

    The ramp starts at step 1, so the first handful of snapshots sit where W_rec
    has barely moved off its init. That is deliberate here: the question this
    grid is built for is when during training the connectivity coordinate system
    departs from its initialisation, which needs the departure itself sampled,
    not just the settled regime afterwards. Note that the downstream WL partition
    is not very meaningful on the earliest snapshots -- W_rec is still close to
    `diag`/`randortho` there -- so those points inform the drift curves rather
    than the partition ones.

    The linear tail keeps the plateau honestly sampled: a pure geometric grid
    puts its last few points at 0.3, 0.5, 1.0 of training, which cannot
    distinguish "settled" from "still drifting slowly". `tail_start` defaults to
    a tenth of the budget, which is comfortably past convergence on the existing
    grid.

    Step 0 is always included.
    """
    max_step = int(max_trials // batch_size)
    if tail_start is None:
        tail_start = max(1, max_step // 10)
    if not 1 <= tail_start <= max_step:
        raise ValueError(
            f"tail_start={tail_start} must be in [1, {max_step}]; "
            f"raise --max-trials or lower --tail-start."
        )

    ramp = [m * 10 ** k
            for k in range(len(str(int(tail_start))))
            for m in RAMP_MANTISSAS
            if m * 10 ** k < tail_start]
    tail = np.round(np.linspace(tail_start, max_step, n_tail)).astype(np.int64)
    return np.unique(np.concatenate([[0], np.array(ramp, dtype=np.int64), tail]))


def build_run_dir(root, init: str, seed: int) -> Path:
    """Trajectory runs live in their own tree, not the (size, seed) grid's.

    They are a different kind of artefact -- many snapshots of one net rather
    than one checkpoint per net -- and mixing them into the grid layout would
    make every `model_globs` pattern in configs/ pick them up by accident.
    """
    return Path(root) / f"softplus_{init}_256" / f"seed{seed}"


def build_hp(n_rnn: int, init: str, seed: int) -> dict:
    from flexsym.core.train import get_default_hp
    from flexsym.core.task import rules_dict

    rules = rules_dict[RULESET]
    hp = get_default_hp(RULESET)
    hp.update({
        'activation': ACTIVATION,
        'w_rec_init': init,
        'n_rnn': n_rnn,
        'l1_h': _coeff(L1H),
        'l2_h': _coeff(L2H),
        'l1_weight': _coeff(L1W),
        'l2_weight': _coeff(L2W),
        'l2_weight_init': 0,
        'sigma_rec': SIGMA_REC,
        'sigma_x': SIGMA_X,
        'rnn_type': RNN_TYPE,
        'use_separate_input': False,
        'learning_rate': 10 ** (LR / 2),
        'w_rec_coeff': W_REC_COEFF,
        'rule_trains': rules,
        'rules': rules,
        # Never early-stop: the training loop in this module ignores target_perf,
        # but keep it at the codebase default so a stray comparison can't fire.
        'target_perf': 1.0,
    })
    hp['rule_probs'] = list(np.ones(len(rules)) / len(rules))
    hp['seed'] = seed
    return hp


def eval_perf(sess, model, rules, hp, batch_size=200):
    """Per-rule performance and loss, noise off.

    ~15 forward passes; negligible next to the training steps between snapshots.
    Loss is fetched in the same `sess.run` as `y_hat` so it is measured on
    exactly the trials the performance number is measured on. `cost_lsq` is the
    task loss the optimiser minimises; `cost_reg` is the activity/weight
    regularisation added to it (`network.py`: `cost = cost_lsq + cost_reg`), kept
    separate so a drop in total loss cannot be misread as a task improvement when
    it is really the regulariser relaxing.

    Returns (perf, loss) keyed by rule; loss values are (cost_lsq, cost_reg).
    """
    from flexsym.core import tools
    from flexsym.core.network import get_perf
    from flexsym.core.task import generate_trials

    perf, loss = {}, {}
    for r in rules:
        trial = generate_trials(r, hp, 'test', noise_on=False, batch_size=batch_size)
        yhat, c_lsq, c_reg = sess.run(
            [model.y_hat, model.cost_lsq, model.cost_reg],
            feed_dict=tools.gen_feed_dict(model, trial, hp))
        perf[r] = float(np.mean(get_perf(yhat, trial.y_loc)))
        loss[r] = (float(c_lsq), float(c_reg))
    return perf, loss


def dump_snapshot(sess, model, out_npz, step, rules, hp, wall_s):
    """Persist the trainable weights + performance + loss at `step`.

    `n_input` is stored explicitly rather than assumed: W_rec is the tail of the
    input kernel, and hardcoding the split point (as the transfer analysis does,
    with N_INPUT=20) silently produces a wrong W_rec on any ruleset with a
    different input width.

    `wall_s` is seconds of training elapsed when this snapshot was taken, and is
    stored per snapshot rather than only as a run total so the cost of reaching
    any point on the grid can be read off directly.
    """
    import tensorflow.compat.v1 as tf

    tv = tf.trainable_variables()
    kernel = [v for v in tv if 'kernel' in v.name][0]
    b_rec = [v for v in tv if 'rnn' in v.name and 'bias' in v.name][0]
    w_out = [v for v in tv if 'output' in v.name and 'weight' in v.name][0]
    b_out = [v for v in tv if 'output' in v.name and 'bias' in v.name][0]

    w_in = sess.run(kernel)                       # (n_input + n_rnn, n_rnn)
    n_rnn = int(hp['n_rnn'])
    perf, loss = eval_perf(sess, model, rules, hp)
    np.savez(
        out_npz,
        w_in=w_in,
        W_rec=w_in[w_in.shape[0] - n_rnn:, :],
        n_input=w_in.shape[0] - n_rnn,
        b_rec=sess.run(b_rec), w_out=sess.run(w_out), b_out=sess.run(b_out),
        step=step, trials=step * hp['batch_size_train'], wall_s=wall_s,
        perf=json.dumps(perf), loss=json.dumps(loss),
    )
    return perf, loss


def _progress_path(run_dir: Path) -> Path:
    return run_dir / 'progress.json'


def train_trajectory(root, init: str, seed: int, n_rnn: int = 256,
                     max_trials: float = DEFAULT_MAX_TRIALS,
                     n_tail: int = DEFAULT_N_TAIL,
                     tail_start: int | None = None,
                     converged_perf: float = 0.97,
                     log_every: int = 500, resume: bool = True) -> Path:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from flexsym.core import tools
    from flexsym.core.network import Model
    from flexsym.core.task import rules_dict

    if init not in INIT_CHOICES:
        raise ValueError(f"init must be one of {INIT_CHOICES}, got {init!r}")

    run_dir = build_run_dir(root, init, seed)
    snap_dir = run_dir / 'snapshots'
    paths.ensure_dir(snap_dir, f'snapshot dir for {init} seed{seed}')

    rules = rules_dict[RULESET]
    hp = build_hp(n_rnn, init, seed)
    hp['rng'] = np.random.RandomState(seed)
    tools.save_hp(hp, str(run_dir))

    steps = snapshot_steps(max_trials, hp['batch_size_train'], n_tail, tail_start)
    max_step = int(steps[-1])
    print(f"[{init} seed{seed}] {len(steps)} snapshots over {max_step} steps "
          f"({max_step * hp['batch_size_train']:.3g} trials): "
          f"{', '.join(str(s) for s in steps[:6])} ... {steps[-1]}", flush=True)

    # Snapshot grid and completion state are written up front so a killed run can
    # be resumed, and so an analysis run can see the intended grid even mid-train.
    manifest = {
        'init': init, 'seed': seed, 'n_rnn': n_rnn,
        'max_trials': float(max_trials), 'batch_size_train': hp['batch_size_train'],
        'snapshot_steps': [int(s) for s in steps],
        'n_tail': n_tail, 'tail_start': int(steps[-n_tail]),
        'converged_perf': converged_perf, 'converged_step': None,
        'ruleset': RULESET, 'rules': rules,
    }

    ckpt = str(run_dir / 'model.ckpt')
    tf.reset_default_graph()
    model = Model(str(run_dir), hp=hp)
    t0 = time.time()
    perf_log = []

    # Same CPU thread-pool capping as `core.train.train`. This module runs its
    # own session rather than going through train(), so without this it silently
    # ignores the TF_INTRA/INTER_THREADS that `worker_env` sets and TF sizes its
    # pools to the whole machine -- 10 such workers on one box thrash badly
    # (~135 threads each) and run several times slower than the same 10 capped.
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=int(os.environ.get('TF_INTRA_THREADS', 0)),
        inter_op_parallelism_threads=int(os.environ.get('TF_INTER_THREADS', 0)),
    )

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        if resume and os.path.exists(ckpt + '.meta') and _progress_path(run_dir).exists():
            prog = json.loads(_progress_path(run_dir).read_text())
            model.saver.restore(sess, ckpt)
            step = int(prog['step'])
            perf_log = prog.get('perf_log', [])
            manifest['converged_step'] = prog.get('converged_step')
            t0 = time.time() - float(prog.get('elapsed_s', 0.0))
            print(f"  resumed at step {step}", flush=True)

        pending = [int(s) for s in steps
                   if not (snap_dir / f'snap_{int(s):09d}.npz').exists()]
        if step > 0 and pending and pending[0] < step:
            # Snapshots strictly before the resume point can never be taken now.
            # Say so rather than silently returning a gappy grid.
            missed = [s for s in pending if s < step]
            print(f"  WARNING: {len(missed)} snapshot step(s) already passed and "
                  f"were never written: {missed}. Re-run with --no-resume for a "
                  f"complete grid.", flush=True)

        wanted = set(int(s) for s in steps)
        while step <= max_step:
            if step in wanted and not (snap_dir / f'snap_{step:09d}.npz').exists():
                wall_s = time.time() - t0
                perf, loss = dump_snapshot(
                    sess, model, snap_dir / f'snap_{step:09d}.npz',
                    step, rules, hp, wall_s)
                pmin, pavg = min(perf.values()), float(np.mean(list(perf.values())))
                lsq = float(np.mean([v[0] for v in loss.values()]))
                reg = float(np.mean([v[1] for v in loss.values()]))
                perf_log.append({'step': step, 'trials': step * hp['batch_size_train'],
                                 'wall_s': wall_s,
                                 'perf_min': pmin, 'perf_avg': pavg,
                                 'cost_lsq': lsq, 'cost_reg': reg})
                if manifest['converged_step'] is None and pmin > converged_perf:
                    manifest['converged_step'] = step
                model.saver.save(sess, ckpt)
                _progress_path(run_dir).write_text(json.dumps(
                    {'step': step, 'perf_log': perf_log,
                     'converged_step': manifest['converged_step'],
                     'elapsed_s': time.time() - t0}, indent=2))
                (run_dir / 'trajectory_manifest.json').write_text(
                    json.dumps(manifest, indent=2))
                print(f"  snap step {step:>9d} ({step * hp['batch_size_train']:>12.4g} "
                      f"trials) perf_min {pmin:.3f} perf_avg {pavg:.3f} "
                      f"lsq {lsq:.4g} reg {reg:.4g} "
                      f"[{wall_s / 3600:.2f}h]", flush=True)

            if step == max_step:
                break
            sess.run(model.train_step)
            step += 1

            if step % log_every == 0:
                model.saver.save(sess, ckpt)
                _progress_path(run_dir).write_text(json.dumps(
                    {'step': step, 'perf_log': perf_log,
                     'converged_step': manifest['converged_step'],
                     'elapsed_s': time.time() - t0}, indent=2))

        model.saver.save(sess, ckpt)

    manifest['elapsed_hours'] = (time.time() - t0) / 3600
    manifest['perf_log'] = perf_log
    (run_dir / 'trajectory_manifest.json').write_text(json.dumps(manifest, indent=2))
    # Completion sentinel, matching the grid launcher's convention: written only
    # after the loop returns normally, so periodic mid-run saves of model.ckpt
    # cannot be mistaken for a finished run.
    (run_dir / 'TRAJECTORY_COMPLETE').write_text(json.dumps(
        {'steps_done': max_step, 'snapshots': len(steps)}, indent=2))
    print(f"[{init} seed{seed}] done -> {run_dir}", flush=True)
    return run_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default=str(paths.MODEL_ROOT / 'trajectories'),
                   help='root for trajectory runs (default MODEL_ROOT/trajectories)')
    p.add_argument('--init', choices=INIT_CHOICES, required=True,
                   help="w_rec init: 'diag' or the unstructured 'randortho' control")
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--n-rnn', type=int, default=256)
    p.add_argument('--max-trials', type=float, default=DEFAULT_MAX_TRIALS,
                   help='fixed trial budget; no early stop (default 1e8, ~46h/net '
                        'on one core, matching the existing 256 grid)')
    p.add_argument('--n-tail', type=int, default=DEFAULT_N_TAIL,
                   help='points in the linear tail after the 1-2-5 ramp (default 10)')
    p.add_argument('--tail-start', type=int, default=None,
                   help='step where the ramp hands over to the linear tail '
                        '(default: a tenth of the budget)')
    p.add_argument('--converged-perf', type=float, default=0.97,
                   help='perf_min threshold recorded (not acted on) as the '
                        'convergence step for annotating the curves')
    p.add_argument('--no-resume', dest='resume', action='store_false')
    p.set_defaults(resume=True)
    return p.parse_args()


if __name__ == '__main__':
    a = parse_args()
    train_trajectory(a.root, a.init, a.seed, n_rnn=a.n_rnn, max_trials=a.max_trials,
                     n_tail=a.n_tail, tail_start=a.tail_start,
                     converged_perf=a.converged_perf, resume=a.resume)
