"""Find dynamical-motif fixed/slow points for a trained multitask RNN.

For each task and each analysable epoch we hold the input constant at the
task-relevant values (rule cue + fixation on, stimulus zero for delay/memory
epochs; rule cue + stimulus for stimulus epochs), sample initial states from
actual trial trajectories in that epoch (plus Gaussian noise to escape the
trajectory manifold), and minimise q(h) = 1/2 ||F(h) - h||^2 with Matt Golub's
FixedPointFinder (Adam then joint optimisation, exact autograd on the trained
graph). Jacobians J = dF/dh at each recovered point are saved for downstream
motif classification and per-unit participation analysis.

Outputs (per model) go to results/motif_fixed_points/<net>/:
  <rule>__<epoch>.npz  with keys:
    xstar (n, N), qstar (n,), J_xstar (n, N, N), inputs (n, n_input),
    eigvals (n, N) complex, epoch_type (str), rule (str), n_inits (int)
  summary.json  with per-(rule, epoch) counts and q statistics.

Run under conda env `myenv` (TF 2.19, tf.compat.v1). FixedPointFinder and
recurrent-whisperer are loaded from repo-local external/ via sys.path.

Example:
  conda run -n myenv python stepnet/run_motif_fixed_points.py \
      --net LeakyRNN_softplus_diag_256_all_0 --n-inits 1000
"""

from __future__ import annotations

from flexsym import paths

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_DIR = paths.REPO_ROOT
paths.add_external_to_syspath()  # FixedPointFinder + recurrent-whisperer

import numpy as np
import tensorflow.compat.v1 as tf  # noqa: E402

tf.disable_v2_behavior()

from flexsym.core import tools  # noqa: E402
from flexsym.core.network import FixedPoint_Model  # noqa: E402
from flexsym.core.task import generate_trials, rules_dict  # noqa: E402
from FixedPointFinderTF import FixedPointFinderTF as FixedPointFinder  # noqa: E402

ORIGINAL_STORE = paths.ORIGINAL_STORE
STAGED_STORE = paths.STAGED_STORE
OUTPUT_ROOT = REPO_DIR / "results" / "motif_fixed_points"

# Epochs we analyse. go1 (pure response) is skipped by default: the motifs of
# interest (memory line attractors, decision saddles) live in the delay and
# stimulus epochs. Order is only used for stable iteration/reporting.
# Pass --include-response to analyse go1 as well: Driscoll et al. report the
# rotation motif at the memory->response transition (the ring attractor rotating
# from output-null into output-potent space as the fixation input drops to zero),
# which cannot be seen without the response epoch.
MEMORY_EPOCHS = ("fix1", "delay1", "delay2")
STIM_EPOCHS = ("stim1", "stim2")
SKIP_EPOCHS = ("go1",)

# Index range of the stimulus channels in trial.x (two 2-unit sin/cos rings).
STIM_CHANNELS = slice(1, 5)
STIM_PRESENT_THRESH = 1e-3  # ||stim channels|| above this => stimulus present


def resolve_model_dir(net: str) -> Path | None:
    for store in (ORIGINAL_STORE, STAGED_STORE):
        candidate = store / net
        if candidate.exists():
            return candidate
    return None


def epoch_bounds(trial, epoch: str, tdim: int) -> tuple[int, int]:
    """Resolve (start, end) for an epoch, mapping None -> trial edges."""
    e0, e1 = trial.epochs[epoch]
    e0 = 0 if e0 is None else int(e0)
    e1 = tdim if e1 is None else int(e1)
    return e0, e1


def classify_epoch_type(trial, epoch: str, e0: int, e1: int) -> str:
    """Data-driven: 'memory' if stim channels ~0 across the epoch, else 'stim'."""
    if epoch in ("go1",):
        return "response"
    stim = trial.x[e0:e1, :, STIM_CHANNELS]
    mean_stim_norm = float(np.mean(np.linalg.norm(stim, axis=2)))
    return "stim" if mean_stim_norm > STIM_PRESENT_THRESH else "memory"


def gather_states_inputs(h, trial, e0: int, e1: int):
    """Pool (state, matching input) pairs over an epoch window.

    h: (tdim, batch, N). Returns states (M, N), inputs (M, n_input) where each
    input is the actual trial input at the state's (t, trial) source, so the
    held-constant input is exactly the task-relevant value for that point and
    the batch's direction grid is covered automatically.
    """
    states = h[e0:e1]                       # (Te, B, N)
    inputs = trial.x[e0:e1]                 # (Te, B, n_input)
    Te, B, N = states.shape
    states = states.reshape(Te * B, N)
    inputs = inputs.reshape(Te * B, inputs.shape[-1])
    return states.astype(np.float32), inputs.astype(np.float32)


def subsample(states, inputs, n_inits, noise_scale, rng):
    """Pick n_inits (state, input) pairs and jitter states off the manifold."""
    M = states.shape[0]
    if M <= n_inits:
        idx = np.arange(M)
    else:
        idx = rng.choice(M, size=n_inits, replace=False)
    s = states[idx].copy()
    u = inputs[idx].copy()
    s = s + rng.normal(0.0, noise_scale, size=s.shape).astype(np.float32)
    return s, u


def find_for_model(
    net: str,
    rules,
    n_inits: int,
    noise_scale: float,
    tol_q: float,
    max_iters: int,
    batch_size: int,
    seed: int,
    out_dir: Path,
    skip_epochs=SKIP_EPOCHS,
):
    model_dir = resolve_model_dir(net)
    if model_dir is None:
        raise FileNotFoundError(
            f"model dir for {net} not found in {ORIGINAL_STORE} or {STAGED_STORE} "
            "(is the X9_Pro drive mounted?)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    summary = {"net": net, "model_dir": str(model_dir), "epochs": {}}

    tf.reset_default_graph()
    model = FixedPoint_Model(str(model_dir))
    with tf.Session() as sess:
        model.restore()
        hp = model.hp
        if rules is None:
            rules = list(hp["rule_trains"])
        fpf = FixedPointFinder(
            model.cell,
            sess,
            method="joint",
            tol_q=tol_q,
            max_iters=max_iters,
            tol_unique=1e-3,
            do_compute_jacobians=True,
            do_decompose_jacobians=False,
            verbose=False,
        )

        for rule in rules:
            trial = generate_trials(
                rule, hp, mode="test", noise_on=False, batch_size=batch_size
            )
            feed_dict = tools.gen_feed_dict(model, trial, hp)
            h = sess.run(model.h, feed_dict=feed_dict)  # (tdim, B, N)
            tdim = h.shape[0]

            for epoch in trial.epochs:
                if epoch in skip_epochs:
                    continue
                e0, e1 = epoch_bounds(trial, epoch, tdim)
                if e1 - e0 < 2:
                    continue
                etype = classify_epoch_type(trial, epoch, e0, e1)
                states, inputs = gather_states_inputs(h, trial, e0, e1)
                init_states, init_inputs = subsample(
                    states, inputs, n_inits, noise_scale, rng
                )

                unique_fps, all_fps = fpf.find_fixed_points(init_states, init_inputs)
                n = int(unique_fps.n) if unique_fps.n is not None else 0
                key = f"{rule}__{epoch}"
                q_all = np.asarray(all_fps.qstar, dtype=float)

                rec = {
                    "epoch_type": etype,
                    "n_inits": int(init_states.shape[0]),
                    "n_unique": n,
                    "n_all": int(all_fps.n) if all_fps.n is not None else 0,
                    "q_all_pct": {
                        p: float(np.percentile(q_all, p))
                        for p in (1, 10, 25, 50, 75, 90, 99)
                    }
                    if q_all.size
                    else {},
                }
                summary["epochs"][key] = rec
                print(
                    f"  {key:32s} type={etype:8s} "
                    f"unique={n:4d}/{rec['n_all']:4d}  "
                    f"q50={rec['q_all_pct'].get(50, float('nan')):.2e}"
                )

                if n == 0:
                    continue
                J = np.asarray(unique_fps.J_xstar, dtype=np.float32)
                eigvals = np.linalg.eigvals(J.astype(np.float64)).astype(np.complex64)
                np.savez_compressed(
                    out_dir / f"{key}.npz",
                    xstar=np.asarray(unique_fps.xstar, dtype=np.float32),
                    qstar=np.asarray(unique_fps.qstar, dtype=np.float32),
                    J_xstar=J,
                    inputs=np.asarray(unique_fps.inputs, dtype=np.float32),
                    eigvals=eigvals,
                    epoch_type=etype,
                    rule=rule,
                    epoch=epoch,
                    n_inits=int(init_states.shape[0]),
                )

    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote summary -> {out_dir / 'summary.json'}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--net", default="LeakyRNN_softplus_diag_256_all_0")
    ap.add_argument(
        "--rules",
        default=None,
        help="comma-separated subset of rules (default: all rule_trains)",
    )
    ap.add_argument("--n-inits", type=int, default=1000)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--tol-q", type=float, default=1e-9)
    ap.add_argument("--max-iters", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output dir (default results/motif_fixed_points/<net>)")
    ap.add_argument("--include-response", action="store_true",
                    help="also analyse the go1 response epoch (needed for the rotation motif)")
    ap.add_argument("--only-response", action="store_true",
                    help="analyse ONLY the go1 response epoch")
    args = ap.parse_args()

    rules = args.rules.split(",") if args.rules else None
    out_dir = Path(args.out) if args.out else OUTPUT_ROOT / args.net
    if args.only_response:
        skip = tuple(e for e in MEMORY_EPOCHS + STIM_EPOCHS)
    elif args.include_response:
        skip = ()
    else:
        skip = SKIP_EPOCHS
    print(f"=== motif fixed points: {args.net} ===")
    find_for_model(
        net=args.net,
        rules=rules,
        n_inits=args.n_inits,
        noise_scale=args.noise,
        tol_q=args.tol_q,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
        seed=args.seed,
        out_dir=out_dir,
        skip_epochs=skip,
    )


if __name__ == "__main__":
    main()
