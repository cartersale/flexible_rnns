"""Restart-ensemble replication of the causal lesion effect (Tier C, needs TF).

The lesion deficit for a group depends only on WHICH units are lesioned, so we
lesion each distinct unit-set at most once per network (frozenset-keyed cache)
and reuse it across every partition that contains that cell. For each network we
then evaluate the lesion metrics -- deficit-activity alignment, max single-task
drop, specificity -- on all 20 WL restarts, the consensus, and 20 size-matched
random partitions, giving the same three reference points as the TF-free driver.

One row per (network, partition_kind, index) with the aggregated lesion metrics,
merged downstream with ``ensemble_metrics_nolesion.csv``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

STEP_DIR = Path(__file__).resolve().parent
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

import tools  # noqa: E402
from network import Model  # noqa: E402
from ensemble_analyze import lesion_metrics  # noqa: E402
from run_lesion_causal_batch import unit_activity_by_task  # noqa: E402
from structural_cell_lesions import evaluate_condition  # noqa: E402

REPO = STEP_DIR.parent
ENS = REPO / "results" / "signed_input_wl_paper" / "restart_ensemble"
ORIG = Path("/Volumes/X9_Pro/flexible_multitask_data/multitask_processed/suppfig4")
STAGED = Path("/Volumes/X9_Pro/flexible_multitask_data/new_runs/staged")


def resolve(model: str) -> Path | None:
    for store in (ORIG, STAGED):
        if (store / model).exists() and (store / model / "model.ckpt.index").exists():
            return store / model
    return None


# Local SSD staging: evaluate_condition restores the checkpoint on EVERY lesion,
# and restoring from the external USB drive is the run's bottleneck (~1.5s each).
# Copy the (small) checkpoint + hp locally once per net so restore() hits the SSD.
STAGE_ROOT = Path(tempfile.gettempdir()) / "flexmult_ckpt_stage"


def stage_checkpoint(src: Path, model: str) -> Path:
    dst = STAGE_ROOT / model
    dst.mkdir(parents=True, exist_ok=True)
    needed = glob.glob(str(src / "model.ckpt.*")) + [str(src / "hp.json")]
    for f in needed:
        fp = Path(f)
        target = dst / fp.name
        if fp.exists() and (not target.exists() or target.stat().st_size != fp.stat().st_size):
            shutil.copy2(fp, target)
    return dst


def run_net(seed: int, *, n_rep: int, batch: int, min_cell: int) -> list[dict]:
    model = f"LeakyRNN_softplus_diag_256_all_{seed}"
    md = resolve(model)
    ens_path = ENS / f"{model}_ensemble.npz"
    if md is None or not ens_path.exists():
        print(f"SKIP {model}")
        return []
    # np.load is monkeypatched by several modules into a chain that ends in
    # task.py's loader (forces allow_pickle=True); calling it here double-injects
    # the kwarg. Use task.py's saved pristine loader directly.
    from task import np_load_old
    z = np_load_old(ens_path, allow_pickle=True)
    k = int(z["k"])
    ind_active = np.asarray(z["ind_active"], dtype=np.int64)
    restart_labels = np.asarray(z["restart_labels"], dtype=np.int64)
    consensus_labels = np.asarray(z["consensus_labels"], dtype=np.int64)
    random_labels = np.asarray(z["random_labels"], dtype=np.int64)

    local_md = stage_checkpoint(md, model)
    print(f"[{model}] staged checkpoint -> {local_md}", flush=True)
    hp = tools.load_hp(str(local_md))
    rules = list(hp.get("rule_trains") or hp["rules"])
    model_obj = Model(str(local_md))

    partitions = [("restart", r, restart_labels[r]) for r in range(len(restart_labels))]
    partitions.append(("consensus", -1, consensus_labels))
    partitions += [("random", r, random_labels[r]) for r in range(len(random_labels))]

    rows: list[dict] = []
    deficit_cache: dict[frozenset, np.ndarray] = {}
    n_eval = 0

    with tf.Session() as sess:
        A = unit_activity_by_task(model_obj, sess, hp, rules, n_rep, batch)
        perf_intact, _ = evaluate_condition(model_obj, sess, hp, rules, None, n_rep, batch)

        def deficit_fn(units: np.ndarray) -> np.ndarray:
            nonlocal n_eval
            key = frozenset(int(u) for u in units)
            hit = deficit_cache.get(key)
            if hit is not None:
                return hit
            perf, _ = evaluate_condition(model_obj, sess, hp, rules, units, n_rep, batch)
            d = perf_intact - perf
            deficit_cache[key] = d
            n_eval += 1
            return d

        for kind, idx, labels in partitions:
            t0 = time.time()
            m = lesion_metrics(labels, ind_active, deficit_fn, A, min_cell=min_cell)
            row = {"model": model, "seed": seed, "k": k, "kind": kind, "index": idx,
                   "intact_perf": float(perf_intact.mean())}
            row.update(m)
            rows.append(row)
            print(f"  {kind:9} {idx:>3}  align={m['lesion_align']:+.3f} "
                  f"spec={m['lesion_specificity']:.3f} maxdrop={m['lesion_maxdrop']:.3f} "
                  f"ncell={m['lesion_ncell']} [evals={n_eval}] ({time.time()-t0:.1f}s)", flush=True)
    tf.reset_default_graph()
    print(f"[{model}] distinct lesions evaluated: {n_eval}", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    p.add_argument("--n-rep", type=int, default=4)
    p.add_argument("--batch", type=int, default=200)
    p.add_argument("--min-cell", type=int, default=1)
    p.add_argument("--out", default=str(ENS / "ensemble_metrics_lesion.csv"))
    args = p.parse_args()

    all_rows: list[dict] = []
    for seed in args.seeds:
        all_rows += run_net(seed, n_rep=args.n_rep, batch=args.batch, min_cell=args.min_cell)
        # incremental write so a long run is never lost
        if all_rows:
            fieldnames = sorted({kk for r in all_rows for kk in r})
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in all_rows:
                    w.writerow({kk: r.get(kk, "") for kk in fieldnames})
    print(f"\nwrote {args.out} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
