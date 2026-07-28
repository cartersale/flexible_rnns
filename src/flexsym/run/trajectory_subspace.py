"""Trajectory analysis, cheap tier: WL partition at EVERY snapshot.

Signed WL refinement of a 256x256 matrix is seconds, so this runs on the full
snapshot grid -- no subsampling. It produces three of the four curves:

  1. Subspace agreement with the final network. Mean squared principal-angle
     cosine between the group-constant subspace S_t at step t and S_final.
     Judged against this net's OWN restart-noise floor, recomputed here rather
     than assumed: the report's ~0.63 is the floor for the diag-init nets, and a
     randortho net has no reason to share it. The 0.63 reference is carried
     through to the figure as well, but the per-net floor is the honest test --
     "S_t has converged" means "S_t is within this net's floor of S_final".
  2. Incremental drift between adjacent snapshots, msca(S_t, S_{t+1}). Because
     the grid is log-spaced the step gap grows along the curve, so `step_gap` and
     `log_step_gap` are emitted alongside; read raw drift for "has it stopped
     moving", drift-per-log-step for "is it moving faster or slower per unit of
     training time".
  3. Task performance, read straight from each snapshot (perf_min / perf_avg).

The unit set and k are held FIXED across a net's snapshots, both taken from its
FINAL checkpoint: the active units and the operating k of an early net differ
from the converged one, and letting them float would confound "the subspace
moved" with "we partitioned a different set of units at a different granularity".
k comes from `operating_k.k_from_clustering`, the same active_fraction policy the
batch configs use.

    python -m flexsym.run.trajectory_subspace                # whole grid
    python -m flexsym.run.trajectory_subspace --init diag --seeds 0
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from flexsym import paths
from flexsym.run.transfer_analysis import princ, restart_floor, wl_subspace
from flexsym.structural.operating_k import k_from_clustering
from flexsym.training.train_trajectory import INIT_CHOICES, build_run_dir

#: The restart-noise floor reported for the diag-init 256 nets
#: (docs/methods_results_report.tex sec. "restart pairs"). Carried as a reference
#: line only -- every net's own floor is recomputed below.
REPORT_FLOOR_MSCA = 0.63

ACTIVITY_NAME = 'epoch_variance_unit_clusters_traj.npz'


def npz_load(path):
    """np.load that survives `flexsym.core.task`.

    Importing that module globally rebinds `np.load` to a lambda that injects
    `allow_pickle=True`, so any later call passing the flag explicitly dies with
    "got multiple values for keyword argument". Which of the two np.loads is live
    depends on whether TF-touching code has been imported yet, so both paths have
    to work.
    """
    try:
        return np.load(path, allow_pickle=True)
    except TypeError:
        return np.load(path)


def snapshot_paths(run_dir: Path) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(str(run_dir / 'snapshots' / 'snap_*.npz')))]


def load_snapshot(path: Path) -> dict:
    """Load one snapshot. W_rec is stored directly, but fall back to slicing the
    kernel by the stored n_input so snapshots written by an older run still load."""
    d = npz_load(path)
    if 'W_rec' in d:
        W_rec = np.asarray(d['W_rec'], float)
    else:
        w_in = np.asarray(d['w_in'], float)
        W_rec = w_in[int(d['n_input']):, :]
    perf = json.loads(str(d['perf'])) if 'perf' in d else {}
    return {'W_rec': W_rec, 'step': int(d['step']), 'trials': int(d['trials']),
            'perf': perf, 'path': path}


def reference_set(run_dir: Path, k_max: int = 60, batch_size: int = 1024,
                  recompute: bool = False) -> tuple[np.ndarray, int]:
    """Active units and operating k of the FINAL checkpoint.

    Runs the standard variance -> activity-clustering pipeline on the run's own
    model.ckpt (which the trainer leaves at the final step), then applies the
    active_fraction k policy. Cached beside the snapshots.
    """
    import pickle

    cache = run_dir / ACTIVITY_NAME
    if cache.exists() and not recompute:
        z = npz_load(cache)
        ind = np.asarray(z['ind_active'], dtype=np.int64)
        return ind, k_from_clustering(int(z['best_k']), ind.size)

    from flexsym.training.cluster_epoch_variance import cluster_epoch_variance
    from flexsym.training.compute_variance_epoch_to_file import compute_variance_epoch

    var_path = run_dir / 'variance_epoch.pkl'
    if not var_path.exists() or recompute:
        import tensorflow.compat.v1 as tf
        res = compute_variance_epoch(str(run_dir), batch_size=batch_size)
        tf.reset_default_graph()
        with var_path.open('wb') as f:
            pickle.dump(res, f)

    cluster_epoch_variance(run_dir, k_min=3, k_max=k_max, normalization='max',
                           linkage_method='ward', active_threshold=1e-3,
                           output_name=ACTIVITY_NAME, variance_path=var_path)
    z = npz_load(cache)
    ind = np.asarray(z['ind_active'], dtype=np.int64)
    return ind, k_from_clustering(int(z['best_k']), ind.size)


def analyse_run(run_dir: Path, wl_seed: int = 0, n_restart: int = 12,
                recompute_reference: bool = False) -> dict:
    snaps = snapshot_paths(run_dir)
    if len(snaps) < 2:
        raise FileNotFoundError(
            f"{run_dir} has {len(snaps)} snapshot(s); train it first with "
            f"python -m flexsym.training.train_trajectory")

    manifest = json.loads((run_dir / 'trajectory_manifest.json').read_text())
    complete = (run_dir / 'TRAJECTORY_COMPLETE').exists()
    if not complete:
        # The reference is the LAST SNAPSHOT ON DISK, which for a live run is not
        # the final network. Say so loudly -- the curves are still readable, but
        # "agreement with the final network" means something different.
        print(f"  WARNING: {run_dir.name} has no TRAJECTORY_COMPLETE sentinel; "
              f"using snapshot at step {load_snapshot(snaps[-1])['step']} as the "
              f"reference. Curves are provisional.", flush=True)

    ind, k = reference_set(run_dir, recompute=recompute_reference)
    print(f"  {run_dir.parent.name}/{run_dir.name}: {len(snaps)} snapshots, "
          f"{len(ind)} active units, k={k}", flush=True)

    final = load_snapshot(snaps[-1])
    lab_f, S_f = wl_subspace(final['W_rec'], ind, k, seed=wl_seed)
    floor = restart_floor(final['W_rec'], ind, k, n_restart=n_restart)
    print(f"    restart floor: msca {floor['msca_mean']:.3f}, "
          f"angle {floor['angle_mean']:.1f}+-{floor['angle_sd']:.1f} deg, "
          f"ARI {floor['ari_mean']:.3f} (report reference {REPORT_FLOOR_MSCA})",
          flush=True)

    rows, prev = [], None
    for sp in snaps:
        s = load_snapshot(sp)
        lab_t, S_t = wl_subspace(s['W_rec'], ind, k, seed=wl_seed)
        angle_f, msca_f = princ(S_f, S_t)
        row = {
            'step': s['step'], 'trials': s['trials'],
            'msca_vs_final': msca_f, 'angle_vs_final_deg': angle_f,
            'ari_vs_final': float(adjusted_rand_score(lab_f, lab_t)),
            'perf_min': (min(s['perf'].values()) if s['perf'] else None),
            'perf_avg': (float(np.mean(list(s['perf'].values()))) if s['perf'] else None),
            'msca_incremental': None, 'angle_incremental_deg': None,
            'ari_incremental': None, 'step_gap': None, 'log_step_gap': None,
        }
        if prev is not None:
            angle_i, msca_i = princ(prev['S'], S_t)
            row['msca_incremental'] = msca_i
            row['angle_incremental_deg'] = angle_i
            row['ari_incremental'] = float(adjusted_rand_score(prev['lab'], lab_t))
            row['step_gap'] = s['step'] - prev['step']
            # log gap is undefined out of step 0; leave it None there rather than
            # inventing a value, so drift-per-log-step starts at the second gap.
            row['log_step_gap'] = (float(np.log10(s['step'] / prev['step']))
                                   if prev['step'] > 0 and s['step'] > 0 else None)
        rows.append(row)
        prev = {'S': S_t, 'lab': lab_t, 'step': s['step']}

    return {
        'run_dir': str(run_dir), 'init': manifest['init'], 'seed': manifest['seed'],
        'n_rnn': manifest['n_rnn'], 'complete': complete,
        'converged_step': manifest.get('converged_step'),
        'n_active': int(len(ind)), 'k': int(k), 'wl_seed': wl_seed,
        'restart_floor': floor, 'report_floor_msca': REPORT_FLOOR_MSCA,
        'reference_step': final['step'], 'rows': rows,
    }


def write_csv(results: list[dict], out_csv: Path) -> None:
    import csv

    fields = ['init', 'seed', 'k', 'n_active', 'floor_msca', 'converged_step',
              'step', 'trials', 'msca_vs_final', 'angle_vs_final_deg',
              'ari_vs_final', 'msca_incremental', 'angle_incremental_deg',
              'ari_incremental', 'step_gap', 'log_step_gap', 'perf_min', 'perf_avg']
    with out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            for row in r['rows']:
                w.writerow({
                    'init': r['init'], 'seed': r['seed'], 'k': r['k'],
                    'n_active': r['n_active'],
                    'floor_msca': r['restart_floor']['msca_mean'],
                    'converged_step': r['converged_step'],
                    **{k: row.get(k) for k in fields[6:]},
                })


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=str(paths.MODEL_ROOT / 'trajectories'))
    ap.add_argument('--init', default=','.join(INIT_CHOICES))
    ap.add_argument('--seeds', default='0-4')
    ap.add_argument('--wl-seed', type=int, default=0,
                    help='WL init seed, held fixed across snapshots so the curve '
                         'is not dominated by WL restart noise (default 0)')
    ap.add_argument('--n-restart', type=int, default=12,
                    help='restarts used to estimate each net\'s own noise floor')
    ap.add_argument('--recompute-reference', action='store_true',
                    help='redo the variance + clustering that fix the unit set and k')
    ap.add_argument('--out-dir', default=str(paths.RESULTS_ROOT / 'trajectory'))
    args = ap.parse_args()

    from flexsym.training.run_seeds_parallel import parse_int_list

    inits = [s.strip() for s in args.init.split(',') if s.strip()]
    seeds = parse_int_list(args.seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results, missing = [], []
    for init in inits:
        for seed in seeds:
            run_dir = build_run_dir(args.root, init, seed)
            if not run_dir.exists():
                missing.append(f'{init}/seed{seed}')
                continue
            results.append(analyse_run(run_dir, wl_seed=args.wl_seed,
                                       n_restart=args.n_restart,
                                       recompute_reference=args.recompute_reference))

    if missing:
        print(f"\nSkipped {len(missing)} untrained run(s): {', '.join(missing)}")
    if not results:
        raise SystemExit('No trained runs found; train the grid first.')

    (out_dir / 'trajectory_subspace.json').write_text(
        json.dumps(results, indent=2, default=float))
    write_csv(results, out_dir / 'trajectory_subspace.csv')
    print(f"\nwrote {out_dir / 'trajectory_subspace.json'}"
          f"\n      {out_dir / 'trajectory_subspace.csv'}")


if __name__ == '__main__':
    main()
