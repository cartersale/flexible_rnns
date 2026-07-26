"""Run the full motif->connectivity pipeline across an ensemble of networks.

For each network: (1) find fixed/slow points (lighter settings: 500 inits,
3000 iters -- the pilot showed convergence is clean and manifolds well sampled
at this level), (2) classify + run the participation tests (Tests 1-4), then
(3) aggregate across networks with a signed-rank over seeds (C1 replication).

  # 256-unit ensemble, seeds 0-9
  conda run -n myenv python stepnet/run_motif_ensemble.py --size 256 --seeds 0-9

  # explicit net list
  conda run -n myenv python stepnet/run_motif_ensemble.py \
      --nets LeakyRNN_softplus_diag_256_all_0,LeakyRNN_softplus_diag_256_all_1

Skips the FP-finding step for a net if its summary.json already exists (use
--refind to force). Long-running: back this with nohup / a background run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

STEP_DIR = Path(__file__).resolve().parent
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from run_motif_fixed_points import find_for_model, OUTPUT_ROOT, resolve_model_dir
from run_motif_participation_stats import run_net

# lighter finder settings for the ensemble (see module docstring)
N_INITS = 500
MAX_ITERS = 3000
TOL_Q = 1e-9


def parse_seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default=None, help="comma-separated net names")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--refind", action="store_true", help="force re-run FP finding")
    ap.add_argument("--n-inits", type=int, default=N_INITS)
    ap.add_argument("--max-iters", type=int, default=MAX_ITERS)
    args = ap.parse_args()

    if args.nets:
        nets = args.nets.split(",")
    else:
        nets = [f"LeakyRNN_softplus_diag_{args.size}_all_{s}"
                for s in parse_seeds(args.seeds)]

    done, skipped, failed = [], [], []
    for net in nets:
        if resolve_model_dir(net) is None:
            print(f"[skip] {net}: model dir not found (drive mounted?)")
            skipped.append(net)
            continue
        out_dir = OUTPUT_ROOT / net
        summary = out_dir / "summary.json"
        if summary.exists() and not args.refind:
            print(f"[fp ] {net}: summary exists, skipping FP finding")
        else:
            print(f"[fp ] {net}: finding fixed points "
                  f"(n_inits={args.n_inits}, max_iters={args.max_iters})")
            find_for_model(net=net, rules=None, n_inits=args.n_inits,
                           noise_scale=0.1, tol_q=TOL_Q, max_iters=args.max_iters,
                           batch_size=200, seed=0, out_dir=out_dir)
        print(f"[stat] {net}: participation tests")
        # fresh cache per net (signed eigvecs depend on the FP set)
        cache = out_dir / "participation_cache_v2.npz"
        if args.refind and cache.exists():
            cache.unlink()
        try:
            run_net(net)
            done.append(net)
        except Exception as e:            # don't let one bad net kill the ensemble
            print(f"[FAIL] {net}: {type(e).__name__}: {e}")
            failed.append(net)

    print(f"\ncompleted {len(done)} nets; skipped {len(skipped)}; failed {len(failed)}")
    if failed:
        print("failed nets:", ", ".join(failed))
    print("aggregate with:  conda run -n myenv python stepnet/aggregate_motif_ensemble.py "
          f"--glob '*_{args.size}_*'")


if __name__ == "__main__":
    main()
