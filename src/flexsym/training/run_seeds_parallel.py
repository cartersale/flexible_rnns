"""Fan out training of a (sizes x seeds) grid of softplus-diag networks across CPU cores.

Designed for a headless CPU node (default tuned for a 28-core budget). Each
(size, seed) network is trained in its own subprocess via ``train_softplus_diag.py``;
a bounded number run concurrently and each is capped to a fixed number of BLAS/TF
threads so the workers do not oversubscribe the CPU.

Core budgeting
--------------
    total threads used ~= --workers * --threads-per-worker
Keep this <= the cores you want to use. The default is 28 workers * 1 thread = 28,
which exactly covers a 4-size x 7-seed grid (28 jobs) one-per-core. Each network is
small in RAM (~1-4 GB even at n_rnn=1024), so concurrency is core-bound, not RAM-bound.

Examples
--------
    # 7 new seeds (3..9) for every default size (128,256,512,1024) -> 28 jobs
    python run_seeds_parallel.py --sizes 128,256,512,1024 --seeds 3-9

    # just the 256 net, seeds 3..9
    python run_seeds_parallel.py --sizes 256 --seeds 3-9

Logs for each job go to <log-dir>/n<size>_seed<n>.log. Jobs whose checkpoint already
exists are skipped unless --force is given, so the launcher is safe to re-run.
"""
from __future__ import division

import os
import sys
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY = os.path.join(HERE, 'train_softplus_diag.py')


def parse_int_list(spec):
    """Parse '3-9' or '128,256' (or a mix like '0-4,7,9') into a sorted list."""
    out = set()
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            lo, hi = part.split('-')
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def checkpoint_done(n_rnn, seed, data_root):
    """True if training for this (size, seed) ran to completion.

    Checks the TRAINING_COMPLETE sentinel rather than model.ckpt.meta, because
    periodic mid-run saves create model.ckpt.* before a net has finished.
    """
    sys.path.insert(0, HERE)
    from train_softplus_diag import build_filedir
    filedir = build_filedir(n_rnn, seed, data_root=data_root)
    complete = os.path.exists(os.path.join(filedir, 'TRAINING_COMPLETE'))
    # A timed-out net is "complete" but did not converge -- leave it re-queueable
    # so a re-launch (with a longer --max-hours) resumes it from its checkpoint.
    not_converged = os.path.exists(os.path.join(filedir, 'NOT_CONVERGED'))
    return complete and not not_converged


def worker_env(threads):
    env = os.environ.copy()
    # Cap every thread pool a worker might touch.
    env['TF_INTRA_THREADS'] = str(threads)
    env['TF_INTER_THREADS'] = str(threads)
    env['OMP_NUM_THREADS'] = str(threads)
    env['OPENBLAS_NUM_THREADS'] = str(threads)
    env['MKL_NUM_THREADS'] = str(threads)
    env['NUMEXPR_NUM_THREADS'] = str(threads)
    # Hide any GPUs so workers stay on CPU and never contend for one device.
    env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def launch(job, args):
    n_rnn, seed = job
    log_path = os.path.join(args.log_dir, 'n{}_seed{}.log'.format(n_rnn, seed))
    log_f = open(log_path, 'w')
    cmd = [
        sys.executable, ENTRY,
        '--n-rnn', str(n_rnn),
        '--seed', str(seed),
        '--max-steps', repr(args.max_steps),
        '--display-step', str(args.display_step),
        '--data-root', args.data_root,
    ]
    # Apply early-stopping only to the requested sizes (default: all sizes).
    if args.target_perf is not None and (
            args.target_perf_sizes is None or n_rnn in args.target_perf_sizes):
        cmd += ['--target-perf', str(args.target_perf)]
    # Apply the wall-clock deadline only to the requested sizes (default: all).
    if args.max_hours is not None and (
            args.max_hours_sizes is None or n_rnn in args.max_hours_sizes):
        cmd += ['--max-hours', str(args.max_hours)]
    if not args.resume:
        cmd += ['--no-resume']
    log_f.write('# ' + ' '.join(cmd) + '\n')
    log_f.flush()
    proc = subprocess.Popen(
        cmd, cwd=HERE, env=worker_env(args.threads_per_worker),
        stdout=log_f, stderr=subprocess.STDOUT,
    )
    return proc, log_f, log_path


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sizes', default='128,256,512,1024',
                   help="network sizes, e.g. '128,256,512,1024' (default all four)")
    p.add_argument('--seeds', default='3-9',
                   help="seeds to train, e.g. '3-9' or '3,4,5' (default 3-9)")
    p.add_argument('--workers', type=int, default=28,
                   help='max jobs training concurrently (default 28)')
    p.add_argument('--threads-per-worker', type=int, default=1,
                   help='BLAS/TF threads per worker (default 1)')
    p.add_argument('--max-steps', type=float, default=1e8,
                   help='max training steps per network (default 1e8)')
    p.add_argument('--target-perf', type=float, default=None,
                   help='early-stop a net once worst-task perf exceeds this '
                        '(e.g. 0.97). Strongly recommended for the 512/1024 nets, '
                        'whose ceiling-length CPU runs take days/weeks otherwise.')
    p.add_argument('--target-perf-sizes', default=None,
                   help='restrict --target-perf to these sizes, e.g. "1024" or '
                        '"512,1024". Sizes not listed train to the max-steps '
                        'ceiling. Default: apply --target-perf to all sizes.')
    p.add_argument('--max-hours', type=float, default=None,
                   help='wall-clock safety deadline per net (hours). A net that '
                        'has not converged by then stops and is flagged '
                        'NOT_CONVERGED. Use with --max-hours-sizes to scope it.')
    p.add_argument('--max-hours-sizes', default=None,
                   help='restrict --max-hours to these sizes, e.g. "1024". '
                        'Default: apply to all sizes. Scope it to early-stopped '
                        'sizes so it does not cut off train-to-ceiling nets.')
    p.add_argument('--no-resume', dest='resume', action='store_false',
                   help='train every net from scratch, ignoring existing '
                        'checkpoints (default: resume killed/incomplete nets).')
    p.set_defaults(resume=True)
    p.add_argument('--display-step', type=int, default=10000,
                   help='steps between perf logging (default 10000)')
    p.add_argument('--data-root', default='data',
                   help='root dir for checkpoints (default ./data)')
    p.add_argument('--log-dir', default='logs/train_softplus_diag_grid',
                   help='dir for per-job logs')
    p.add_argument('--force', action='store_true',
                   help='retrain even if a checkpoint already exists')
    p.add_argument('--biggest-first', action='store_true', default=True,
                   help='schedule larger networks first (default on) so the '
                        'slow 1024-unit jobs are not left running alone at the end')
    args = p.parse_args()

    sizes = parse_int_list(args.sizes)
    seeds = parse_int_list(args.seeds)
    # Normalize to a set of ints (or None = all sizes) for the per-size filter.
    args.target_perf_sizes = (set(parse_int_list(args.target_perf_sizes))
                              if args.target_perf_sizes is not None else None)
    args.max_hours_sizes = (set(parse_int_list(args.max_hours_sizes))
                            if args.max_hours_sizes is not None else None)
    os.makedirs(args.log_dir, exist_ok=True)

    # Build the full (size, seed) grid. Largest sizes first so the longest jobs
    # start at t=0 and the tail of the run isn't a single 1024 net hogging a core.
    grid = [(n, s) for n in sizes for s in seeds]
    if args.biggest_first:
        grid.sort(key=lambda job: (-job[0], job[1]))

    pending = []
    for job in grid:
        n_rnn, seed = job
        if not args.force and checkpoint_done(n_rnn, seed, args.data_root):
            print('[skip] n_rnn={} seed={} already has a checkpoint'.format(n_rnn, seed))
        else:
            pending.append(job)

    if not pending:
        print('Nothing to do; all requested (size, seed) networks are already trained.')
        return

    total_threads = args.workers * args.threads_per_worker
    print('Training {} networks | sizes={} seeds={} | {} workers x {} threads = {} threads | logs -> {}'
          .format(len(pending), sizes, seeds, args.workers,
                  args.threads_per_worker, total_threads, args.log_dir))

    queue = list(pending)
    running = {}   # pid -> (job, proc, log_f, log_path)
    failures = []

    while queue or running:
        # Fill open worker slots.
        while queue and len(running) < args.workers:
            job = queue.pop(0)
            proc, log_f, log_path = launch(job, args)
            running[proc.pid] = (job, proc, log_f, log_path)
            print('[start] n_rnn={} seed={} (pid {}) -> {}'
                  .format(job[0], job[1], proc.pid, log_path))

        # Reap finished workers.
        done_pids = []
        for pid, (job, proc, log_f, log_path) in running.items():
            ret = proc.poll()
            if ret is not None:
                log_f.close()
                done_pids.append(pid)
                if ret == 0:
                    print('[done]  n_rnn={} seed={} (pid {})'.format(job[0], job[1], pid))
                else:
                    failures.append(job)
                    print('[FAIL]  n_rnn={} seed={} (pid {}) exit {} -- see {}'
                          .format(job[0], job[1], pid, ret, log_path))
        for pid in done_pids:
            del running[pid]

        if queue or running:
            time.sleep(2)

    print('\nAll networks processed.')
    if failures:
        print('Failed: {}'.format(sorted(failures)))
        sys.exit(1)
    print('All requested networks completed successfully.')


if __name__ == '__main__':
    main()
