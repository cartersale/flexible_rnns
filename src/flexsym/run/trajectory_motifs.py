"""Trajectory analysis, expensive tier: fixed points at ~5 snapshots.

Fixed-point finding is the costly step, and dense sampling buys nothing here --
the question is only whether the motifs settle *after* the connectivity
coordinate system does, which a handful of well-spread snapshots answers. So this
runs on ~5 snapshots per net while `trajectory_subspace.py` runs on all ~23.

Two measures per (rule, epoch), giving the fourth curve:

  * `n_fixed_points` -- how many distinct slow points the relaxation recovers.
    Early in training the landscape is unstructured and the count is unstable;
    it settling is one face of motif stabilisation.
  * `angle_vs_final_deg` -- movement of the motif direction, the angle between
    this snapshot's motif direction and the SAME (rule, epoch) motif direction at
    the final snapshot. Signs of eigenvectors and attractor axes are arbitrary,
    so |cos| is used; the angle is in [0, 90].

Also emitted is `containment_S_final`, the fraction of each motif direction lying
inside the final WL group-constant subspace. That is what ties the two tiers
together: if containment is already high at a snapshot where the direction is
still moving, the motif is rearranging *inside* a coordinate system that has
already fixed -- which is the claim the four curves are meant to test.

Snapshot choice is index-linear over the snapshot list. Because the grid is
already log-spaced in steps, evenly-spaced indices are log-spaced steps; the
final snapshot is always included, since it is the reference.

    python -m flexsym.run.trajectory_motifs                      # whole grid
    python -m flexsym.run.trajectory_motifs --init diag --seeds 0 --n-checkpoints 5

Runs one net at a time (~1 min per snapshot for 15 rules x 3 epochs). To use more
cores, launch several processes scoped with --init / --seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flexsym import paths
from flexsym.run.trajectory_subspace import (
    load_snapshot, reference_set, snapshot_paths,
)
from flexsym.run.transfer_analysis import wl_subspace
from flexsym.training.train_trajectory import INIT_CHOICES, build_run_dir

DEFAULT_EPOCHS = ('stim1', 'delay1', 'go1')
DEFAULT_N_CHECKPOINTS = 5

#: q below which a relaxed state counts as a slow point, and the distance below
#: which two slow points are the same one. Both match transfer_analysis.py so the
#: counts here are comparable to the transfer experiment's.
Q_THRESHOLD = 1e-6
DEDUP_DISTANCE = 1e-2


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def find_fixed_points(W, Win, b, u, seeds, alpha=0.2, iters=2000, tol=1e-8):
    """h* = softplus(W h* + Win u + b) by leaky relaxation, all seeds at once.

    Vectorised over seeds -- the per-seed loop in transfer_analysis.py is fine for
    a couple of held-out rules but not for 15 rules x 3 epochs x 5 snapshots x 10
    nets, where it dominates the whole analysis.

    Returns (H, q) with H (n_seeds, n) the relaxed states and q their speeds.
    """
    H = np.asarray(seeds, float)
    drive = Win @ u + b                                    # (n,)
    for _ in range(iters):
        F = softplus(H @ W.T + drive)
        H_new = (1 - alpha) * H + alpha * F
        delta = np.max(np.abs(H_new - H))
        H = H_new
        if delta < tol:
            break
    resid = softplus(H @ W.T + drive) - H
    return H, 0.5 * np.sum(resid ** 2, axis=1)


def dedup(H, q, q_thr=Q_THRESHOLD, dist=DEDUP_DISTANCE):
    """Distinct slow points, lowest-q first."""
    good = H[q < q_thr]
    good_q = q[q < q_thr]
    if len(good) == 0:
        return np.empty((0, H.shape[1])), np.empty(0)
    order = np.argsort(good_q)
    good, good_q = good[order], good_q[order]
    keep = [0]
    for i in range(1, len(good)):
        if np.min(np.linalg.norm(good[keep] - good[i], axis=1)) > dist:
            keep.append(i)
    return good[keep], good_q[keep]


def classify_and_dir(uniq, W, Win, u, b, alpha=0.2):
    """(motif_type, direction) from the distinct slow points.

    Same taxonomy as transfer_analysis.py: a set of >=3 near-marginal points
    spanning ~1D is a line attractor and its direction is that span; otherwise the
    lowest-q point is classified by the leading eigenvalue of its Jacobian and the
    direction is the corresponding eigenvector.
    """
    if len(uniq) == 0:
        return None, None
    if len(uniq) >= 3:
        X = uniq - uniq.mean(0)
        s = np.linalg.svd(X, compute_uv=False)
        if s[0] > 1e-3 and (len(s) < 2 or s[1] < 0.3 * s[0]):
            j = int(np.argmax(np.linalg.norm(X, axis=1)))
            v = X[j] / (np.linalg.norm(X[j]) + 1e-12)
            return 'line_attractor', v
    h = uniq[0]                                             # lowest q
    D = 1.0 / (1.0 + np.exp(-(W @ h + Win @ u + b)))        # softplus' = sigmoid
    J = (1 - alpha) * np.eye(len(h)) + alpha * (D[:, None] * W)
    lam, V = np.linalg.eig(J)
    i = int(np.argmax(np.abs(lam)))
    v = np.real(V[:, i])
    v = v / (np.linalg.norm(v) + 1e-12)
    kind = 'saddle_decision' if np.abs(lam[i]) > 1.0 else 'point_attractor'
    return kind, v


def epoch_drives(rule, hp, epochs, batch_size=64):
    """Mean input vector over each requested epoch of `rule`."""
    from flexsym.core.task import generate_trials

    trial = generate_trials(rule, hp, 'test', noise_on=False, batch_size=batch_size)
    out = {}
    for e in epochs:
        if e not in trial.epochs:
            continue
        a, b = trial.epochs[e]
        a = 0 if a is None else a
        b = trial.x.shape[0] if b is None else b
        t = (a + b) // 2
        out[e] = np.asarray(trial.x[t], float).mean(0)      # (n_input,)
    return out


def seed_states(n, rng, n_per_scale=40):
    seeds = [rng.standard_normal(n) * s
             for s in (0.1, 0.5, 1.0) for _ in range(n_per_scale)]
    seeds.append(np.zeros(n))
    return np.array(seeds)


def pick_checkpoints(snaps, n_checkpoints):
    """~n log-spaced snapshots, final always included.

    Index-linear because the snapshot grid is already log-spaced in steps.
    """
    if len(snaps) <= n_checkpoints:
        return list(range(len(snaps)))
    idx = np.unique(np.round(
        np.linspace(0, len(snaps) - 1, n_checkpoints)).astype(int))
    return list(idx)


def analyse_run(run_dir: Path, rules=None, epochs=DEFAULT_EPOCHS,
                n_checkpoints=DEFAULT_N_CHECKPOINTS, wl_seed=0, rng_seed=0) -> dict:
    from flexsym.core import tools

    snaps = snapshot_paths(run_dir)
    if len(snaps) < 2:
        raise FileNotFoundError(f'{run_dir} has {len(snaps)} snapshot(s); train it first.')
    manifest = json.loads((run_dir / 'trajectory_manifest.json').read_text())
    if not (run_dir / 'TRAJECTORY_COMPLETE').exists():
        print(f'  WARNING: {run_dir.name} is not finished; the reference motifs '
              f'are those of the last snapshot on disk, not the final net.', flush=True)

    hp = tools.load_hp(str(run_dir))
    hp['rng'] = np.random.RandomState(rng_seed)
    rules = list(rules) if rules else list(manifest['rules'])

    ind, k = reference_set(run_dir)
    final = load_snapshot(snaps[-1])
    _, S_f = wl_subspace(final['W_rec'], ind, k, seed=wl_seed)

    picked = pick_checkpoints(snaps, n_checkpoints)
    print(f"  {run_dir.parent.name}/{run_dir.name}: {len(picked)} of {len(snaps)} "
          f"snapshots -> steps "
          f"{[load_snapshot(snaps[i])['step'] for i in picked]}", flush=True)

    # Drives depend only on the task, not the weights -- build once per net.
    drives = {r: epoch_drives(r, hp, epochs) for r in rules}
    n_input = int(np.load(snaps[-1], allow_pickle=True)['n_input'])
    rng = np.random.default_rng(rng_seed)

    rows = []
    for i in picked:
        s = load_snapshot(snaps[i])
        d = np.load(s['path'], allow_pickle=True)
        W = s['W_rec']
        Win = np.asarray(d['w_in'], float)[:n_input, :].T    # (n_rnn, n_input)
        b = np.asarray(d['b_rec'], float)
        seeds = seed_states(W.shape[0], rng)
        for rule in rules:
            for e, u in drives[rule].items():
                H, q = find_fixed_points(W, Win, b, u, seeds)
                uniq, _ = dedup(H, q)
                kind, v = classify_and_dir(uniq, W, Win, u, b)
                row = {'step': s['step'], 'trials': s['trials'], 'rule': rule,
                       'epoch': e, 'n_fixed_points': int(len(uniq)),
                       'motif_type': kind, 'containment_S_final': None,
                       'v': None}
                if v is not None:
                    v_act = v[ind]
                    nv = np.linalg.norm(v_act) + 1e-12
                    row['containment_S_final'] = float(np.linalg.norm(S_f.T @ v_act) / nv)
                    row['v'] = v_act.tolist()
                rows.append(row)
        print(f"    step {s['step']:>9d}: "
              f"{sum(r['n_fixed_points'] for r in rows if r['step'] == s['step'])} "
              f"slow points across {len(drives)} rules", flush=True)

    # Motif movement: angle against the same (rule, epoch) at the last analysed
    # snapshot. Directions are sign-arbitrary, hence |cos|.
    final_step = rows[-1]['step']
    finals = {(r['rule'], r['epoch']): r['v'] for r in rows
              if r['step'] == final_step and r['v'] is not None}
    for r in rows:
        vf = finals.get((r['rule'], r['epoch']))
        if r['v'] is None or vf is None:
            r['angle_vs_final_deg'] = None
        else:
            v, vf = np.array(r['v']), np.array(vf)
            c = np.abs(v @ vf) / (np.linalg.norm(v) * np.linalg.norm(vf) + 1e-12)
            r['angle_vs_final_deg'] = float(np.degrees(np.arccos(np.clip(c, 0, 1))))
        r.pop('v')

    return {'run_dir': str(run_dir), 'init': manifest['init'], 'seed': manifest['seed'],
            'n_active': int(len(ind)), 'k': int(k),
            'converged_step': manifest.get('converged_step'),
            'reference_step': final_step, 'epochs': list(epochs), 'rows': rows}


def write_csv(results, out_csv: Path) -> None:
    import csv

    fields = ['init', 'seed', 'converged_step', 'step', 'trials', 'rule', 'epoch',
              'n_fixed_points', 'motif_type', 'angle_vs_final_deg',
              'containment_S_final']
    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            for row in r['rows']:
                w.writerow({'init': r['init'], 'seed': r['seed'],
                            'converged_step': r['converged_step'],
                            **{k: row.get(k) for k in fields[3:]}})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=str(paths.MODEL_ROOT / 'trajectories'))
    ap.add_argument('--init', default=','.join(INIT_CHOICES))
    ap.add_argument('--seeds', default='0-4')
    ap.add_argument('--rules', default=None,
                    help='comma-separated rules (default: all 15 the net was trained on)')
    ap.add_argument('--epochs', default=','.join(DEFAULT_EPOCHS))
    ap.add_argument('--n-checkpoints', type=int, default=DEFAULT_N_CHECKPOINTS,
                    help='snapshots to run fixed points on (default 5)')
    ap.add_argument('--wl-seed', type=int, default=0)
    ap.add_argument('--out-dir', default=str(paths.RESULTS_ROOT / 'trajectory'))
    args = ap.parse_args()

    from flexsym.training.run_seeds_parallel import parse_int_list

    inits = [s.strip() for s in args.init.split(',') if s.strip()]
    seeds = parse_int_list(args.seeds)
    rules = [s.strip() for s in args.rules.split(',')] if args.rules else None
    epochs = tuple(s.strip() for s in args.epochs.split(',') if s.strip())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results, missing = [], []
    for init in inits:
        for seed in seeds:
            run_dir = build_run_dir(args.root, init, seed)
            if not run_dir.exists():
                missing.append(f'{init}/seed{seed}')
                continue
            results.append(analyse_run(run_dir, rules=rules, epochs=epochs,
                                       n_checkpoints=args.n_checkpoints,
                                       wl_seed=args.wl_seed))

    if missing:
        print(f"\nSkipped {len(missing)} untrained run(s): {', '.join(missing)}")
    if not results:
        raise SystemExit('No trained runs found; train the grid first.')

    (out_dir / 'trajectory_motifs.json').write_text(
        json.dumps(results, indent=2, default=float))
    write_csv(results, out_dir / 'trajectory_motifs.csv')
    print(f"\nwrote {out_dir / 'trajectory_motifs.json'}"
          f"\n      {out_dir / 'trajectory_motifs.csv'}")


if __name__ == '__main__':
    main()
