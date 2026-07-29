"""Fan out the trajectory grid -- 5 diag-init and 5 randortho-init softplus nets.

Same launcher shape as ``run_seeds_parallel.py`` (one subprocess per net, capped
thread pools, resumable, completion sentinel), but over an (init x seed) grid
instead of (size x seed), and pointed at ``train_trajectory.py`` so each net
dumps its log-spaced snapshot grid.

    # the default 5 + 5 grid, 10 workers x 1 thread
    python -m flexsym.training.run_trajectory_grid

    # short pilot: check the snapshot grid and the analyses end-to-end first
    python -m flexsym.training.run_trajectory_grid --seeds 0 --max-trials 2e6

Budget: at the default 1e8 trials a 256-unit net takes ~46 h on one core (from
the existing grid's logs), so the full 10-net run is ~460 core-hours -- about two
wall-clock days on 10 cores. Run the pilot first.

Logs go to <log-dir>/<init>_seed<n>.log. Nets with a TRAJECTORY_COMPLETE
sentinel are skipped unless --force, so re-running is safe.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from flexsym import paths
from flexsym.training.run_seeds_parallel import parse_int_list, worker_env
from flexsym.training.train_trajectory import INIT_CHOICES, build_run_dir

ENTRY_MODULE = 'flexsym.training.train_trajectory'


def run_done(root, init, seed) -> bool:
    return (build_run_dir(root, init, seed) / 'TRAJECTORY_COMPLETE').exists()


def launch(job, args):
    init, seed = job
    log_path = os.path.join(args.log_dir, f'{init}_seed{seed}.log')
    log_f = open(log_path, 'w')
    cmd = [
        sys.executable, '-m', ENTRY_MODULE,
        '--root', args.root,
        '--init', init,
        '--seed', str(seed),
        '--n-rnn', str(args.n_rnn),
        '--max-trials', repr(args.max_trials),
        '--n-tail', str(args.n_tail),
    ]
    if args.tail_start is not None:
        cmd += ['--tail-start', str(args.tail_start)]
    if not args.resume:
        cmd += ['--no-resume']
    log_f.write('# ' + ' '.join(cmd) + '\n')
    log_f.flush()
    proc = subprocess.Popen(cmd, cwd=str(paths.REPO_ROOT),
                            env=worker_env(args.threads_per_worker),
                            stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_f, log_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', default=str(paths.MODEL_ROOT / 'trajectories'))
    p.add_argument('--inits', default=','.join(INIT_CHOICES),
                   help=f"inits to train (default '{','.join(INIT_CHOICES)}')")
    p.add_argument('--seeds', default='0-4', help='seeds per init (default 0-4)')
    p.add_argument('--n-rnn', type=int, default=256)
    p.add_argument('--max-trials', type=float, default=1e8)
    p.add_argument('--n-tail', type=int, default=10)
    p.add_argument('--tail-start', type=int, default=None)
    p.add_argument('--workers', type=int, default=10,
                   help='max nets training concurrently (default 10 = whole grid)')
    p.add_argument('--threads-per-worker', type=int, default=1)
    p.add_argument('--log-dir', default='logs/trajectory_grid')
    p.add_argument('--force', action='store_true')
    p.add_argument('--no-resume', dest='resume', action='store_false')
    p.set_defaults(resume=True)
    args = p.parse_args()

    inits = [s.strip() for s in args.inits.split(',') if s.strip()]
    bad = [i for i in inits if i not in INIT_CHOICES]
    if bad:
        p.error(f"unknown init(s) {bad}; choose from {list(INIT_CHOICES)}")
    seeds = parse_int_list(args.seeds)
    os.makedirs(args.log_dir, exist_ok=True)

    # Interleave inits so a partial run still has both arms represented -- an
    # init-major order would finish all 5 diag nets before starting any control.
    grid = [(init, s) for s in seeds for init in inits]

    pending = []
    for job in grid:
        if not args.force and run_done(args.root, *job):
            print(f'[skip] {job[0]} seed{job[1]} already complete')
        else:
            pending.append(job)

    if not pending:
        print('Nothing to do; every requested net is already trained.')
        return

    print(f'Training {len(pending)} nets | inits={inits} seeds={seeds} | '
          f'{args.workers} workers x {args.threads_per_worker} threads | '
          f'{args.max_trials:.3g} trials each | logs -> {args.log_dir}')

    queue, running, failures = list(pending), {}, []
    while queue or running:
        while queue and len(running) < args.workers:
            job = queue.pop(0)
            proc, log_f, log_path = launch(job, args)
            running[proc.pid] = (job, proc, log_f, log_path)
            print(f'[start] {job[0]} seed{job[1]} (pid {proc.pid}) -> {log_path}')

        done = []
        for pid, (job, proc, log_f, log_path) in running.items():
            ret = proc.poll()
            if ret is not None:
                log_f.close()
                done.append(pid)
                if ret == 0:
                    print(f'[done]  {job[0]} seed{job[1]} (pid {pid})')
                else:
                    failures.append(job)
                    print(f'[FAIL]  {job[0]} seed{job[1]} exit {ret} -- see {log_path}')
        for pid in done:
            del running[pid]
        if queue or running:
            time.sleep(2)

    print('\nAll nets processed.')
    if failures:
        print(f'Failed: {sorted(failures)}')
        sys.exit(1)


if __name__ == '__main__':
    main()
