"""Train a single softplus / diagonal-init LeakyRNN on the 'all' ruleset.

Single-(size, seed) entry point used by ``run_seeds_parallel.py``. Mirrors the
hyperparameters and directory-naming convention of ``general_model_train.py`` but
fixes the activation/init to softplus-diag and exposes the network size and random
seed on the command line, so a grid of (size, seed) networks can be launched as
independent processes.

Run directly for one network:

    python train_softplus_diag.py --n-rnn 256 --seed 3

or let ``run_seeds_parallel.py`` fan out a (sizes x seeds) grid across CPU cores.
"""
from __future__ import division

import os
import json
import argparse

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from flexsym.core import train
from flexsym.core.task import rules_dict


# --- fixed network configuration (softplus-diag) -----------------------------
RNN_TYPE = 'LeakyRNN'
ACTIVATION = 'softplus'
INIT = 'diag'
RULESET = 'all'

# the network sizes used in the paper for this architecture
DEFAULT_SIZES = [128, 256, 512, 1024]

# regularization / noise / learning exponents, matching general_model_train.py.
# Each constant is an exponent fed through _coeff(): 0 means "off" (coefficient 0),
# a negative value like -6 means 10**-6. A bare 10**0 would be 1.0 -- a catastrophic
# L1 penalty -- which is exactly the bug that produced the first seeds-3-9 batch.
L2W = -6           # l2 weight regularization
L2H = -6           # l2 activity regularization
L1W = 0            # l1 weight regularization (off)
L1H = 0            # l1 activity regularization (off)
LR = -6            # learning rate -> 10**(LR/2)
SIGMA_REC = 1 / 20  # recurrent unit noise
SIGMA_X = 2 / 20    # input noise
W_REC_COEFF = 1     # coefficient on weight-matrix init


def build_filedir(n_rnn, seed, data_root='data'):
    """Reproduce the on-disk layout used by general_model_train.py."""
    rule_trains = rules_dict[RULESET]
    rule_trains_str = '_'.join(rule_trains)
    net_name = ('lr' + str(-LR) + 'l2_w' + str(-L2W) + '_h' + str(-L2H) +
                '_sig_rec' + str(SIGMA_REC) + '_sig_x' + str(SIGMA_X) +
                '_w_rec_coeff' + str(W_REC_COEFF) + '_' + rule_trains_str)
    return os.path.join(
        data_root, RULESET, RNN_TYPE, ACTIVATION, INIT,
        str(len(rule_trains)) + '_tasks', str(n_rnn) + '_n_rnn',
        net_name, str(seed))


def _coeff(exponent):
    """Regularization exponent -> coefficient. 0 means off, matching hp_sweep.py's guard
    (np.min([-e, 10**e])): exponent 0 -> min(0, 1) = 0; exponent -6 -> min(6, 1e-6) = 1e-6."""
    import numpy as np
    return float(np.min([-exponent, 10 ** exponent]))


def train_network(n_rnn, seed, max_steps=1e8, display_step=10000,
                  data_root='data', target_perf=None, resume=True,
                  max_hours=None):
    rule_trains = rules_dict[RULESET]
    filedir = build_filedir(n_rnn, seed, data_root=data_root)
    hp = {
        'activation': ACTIVATION,
        'w_rec_init': INIT,
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
    }
    # The training loop breaks when perf_min > target_perf. The codebase default
    # is 1.0, which is never exceeded, so by default a net runs to the max_steps
    # ceiling. Setting target_perf < 1 lets a converged net stop early and free
    # its core -- essential for the large (512/1024) nets on CPU.
    if target_perf is not None:
        hp['target_perf'] = target_perf
    status = train.train(
        filedir,
        seed=seed,
        max_steps=max_steps,
        ruleset=RULESET,
        rule_trains=rule_trains,
        hp=hp,
        display_step=display_step,
        rich_output=False,
        resume=resume,
        max_hours=max_hours,
    )
    status = status or {}
    # Completion sentinel: written only after train() returns normally, so the
    # launcher can distinguish a finished net from one that was killed mid-run
    # (whose model.ckpt.* files already exist from periodic saves).
    with open(os.path.join(filedir, 'TRAINING_COMPLETE'), 'w') as f:
        json.dump(status, f)
    # If a wall-clock deadline cut off a net before it converged, flag it loudly
    # and on disk so it is not silently treated as a good network.
    not_converged = os.path.join(filedir, 'NOT_CONVERGED')
    if status.get('timed_out'):
        with open(not_converged, 'w') as f:
            json.dump(status, f)
        print('WARNING: n_rnn={} seed={} did NOT converge (perf_min={}, '
              'target={}) -- hit the max-hours deadline. Marked NOT_CONVERGED.'
              .format(n_rnn, seed, status.get('final_perf_min'),
                      status.get('target_perf')))
    elif os.path.exists(not_converged):
        # A previous timed-out run later finished cleanly on resume; clear stale flag.
        os.remove(not_converged)
    return filedir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-rnn', type=int, required=True,
                   help='network size (e.g. 128, 256, 512, 1024)')
    p.add_argument('--seed', type=int, required=True,
                   help='random seed for this network')
    p.add_argument('--max-steps', type=float, default=1e8,
                   help='maximum number of training steps (default 1e8)')
    p.add_argument('--display-step', type=int, default=10000,
                   help='steps between performance logging (default 10000)')
    p.add_argument('--data-root', default='data',
                   help='root directory for saved checkpoints (default ./data)')
    p.add_argument('--target-perf', type=float, default=None,
                   help='stop early once the worst-task performance exceeds this '
                        '(e.g. 0.97). Default (None) keeps the codebase default of '
                        '1.0, i.e. train to the max-steps ceiling.')
    p.add_argument('--max-hours', type=float, default=None,
                   help='wall-clock safety deadline (hours). If the net has not '
                        'converged by then, stop and flag it NOT_CONVERGED.')
    p.add_argument('--no-resume', dest='resume', action='store_false',
                   help='ignore any existing checkpoint and train from scratch '
                        '(default: resume from the latest checkpoint if present).')
    p.set_defaults(resume=True)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = train_network(args.n_rnn, args.seed, max_steps=args.max_steps,
                        display_step=args.display_step, data_root=args.data_root,
                        target_perf=args.target_perf, resume=args.resume,
                        max_hours=args.max_hours)
    print('Finished n_rnn={} seed={} -> {}'.format(args.n_rnn, args.seed, out))
