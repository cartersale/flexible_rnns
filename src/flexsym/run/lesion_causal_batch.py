"""Causal lesion test across seeds: are connectivity roles the computational units?

Reuses structural_cell_lesions machinery (lesion each structural cell / activity
cluster / size-matched random group, measure per-task performance drop) and ADDS
the causal-mechanistic layer:

  * per-unit x per-task ACTIVITY profile (intact), so each group gets an activity
    vector a_g and a deficit vector d_g over the 15 tasks;
  * FUNCTIONAL ALIGNMENT r(a_g, d_g) -- "you break what you compute". Structural
    cells should align more than size-matched random groups;
  * SPECIFICITY (max single-task drop / total) -- structural more selective than random;
  * saves full per-group deficit + activity matrices for downstream motif-category
    and memory-mode analyses.

Runs the 10 x 256 networks at operating k on the materialized transpose partition.
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

STEP_DIR = Path(__file__).resolve().parent
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

import tools  # noqa: E402
from network import Model, get_perf  # noqa: E402
from task import generate_trials  # noqa: E402
from structural_cell_lesions import build_groups, evaluate_condition, load_partitions  # noqa: E402

REPO = STEP_DIR.parent
ORIG = Path("/Volumes/X9_Pro/flexible_multitask_data/multitask_processed/suppfig4")
STAGED = Path("/Volumes/X9_Pro/flexible_multitask_data/new_runs/staged")
OPK = {0: 41, 1: 51, 2: 38, 3: 14, 4: 14, 5: 14, 6: 14, 7: 14, 8: 14, 9: 15}
OUTDIR = REPO / "results/signed_input_wl_paper/lesion_causal"


def resolve(model):
    for s in (ORIG, STAGED):
        if (s / model).exists() and (s / model / "model.ckpt.index").exists():
            return s / model
    return None


def unit_activity_by_task(model, sess, hp, rules, n_rep, batch):
    """Intact per-unit x per-task mean activation A[n_rnn, n_rules]."""
    model.restore()  # intact
    n_rnn = hp["n_rnn"]
    A = np.zeros((n_rnn, len(rules)))
    brep = max(1, batch // n_rep)
    for ri, rule in enumerate(rules):
        acc = []
        for _ in range(n_rep):
            trial = generate_trials(rule, hp, "random", batch_size=brep)
            fd = tools.gen_feed_dict(model, trial, hp)
            h = sess.run(model.h, feed_dict=fd)          # [T, B, n_rnn]
            acc.append(h.mean(axis=(0, 1)))
        A[:, ri] = np.mean(acc, axis=0)
    return A


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def run_one(model_dir, sweep, seed, n_rep, batch, n_random):
    k, labels_struct, labels_act, ind_active, variant = load_partitions(Path(sweep), None)
    hp = tools.load_hp(str(model_dir))
    rules = list(hp.get("rule_trains") or hp["rules"])
    groups = build_groups(labels_struct, labels_act, ind_active, n_random, seed)

    model = Model(str(model_dir))
    with tf.Session() as sess:
        A = unit_activity_by_task(model, sess, hp, rules, n_rep, batch)  # [n_rnn, R]
        perf_intact, _ = evaluate_condition(model, sess, hp, rules, None, n_rep, batch)
        recs = []
        for g in groups:
            perf, _ = evaluate_condition(model, sess, hp, rules, g["units"], n_rep, batch)
            deficit = perf_intact - perf                 # positive = perf lost
            a_g = A[g["units"], :].mean(axis=0)          # group activity profile
            recs.append({"type": g["type"], "gid": g["group_id"], "src": g["source_partition"],
                         "size": len(g["units"]), "deficit": deficit, "activity": a_g})
    return k, rules, perf_intact, recs, A, labels_struct, ind_active


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    n_rep, batch, n_random = 6, 240, 5
    agg = []
    for seed, k in OPK.items():
        model = f"LeakyRNN_softplus_diag_256_all_{seed}"
        md = resolve(model)
        sweep = REPO / f"results/signed_input_wl_paper/{model}/transpose_selected_k_{k}/signed_input_wl_sweep.npz"
        if md is None or not sweep.exists():
            print(f"SKIP {model} (md={md}, sweep={sweep.exists()})", flush=True)
            continue
        print(f"=== {model} (k={k}) ===", flush=True)
        kk, rules, perf_intact, recs, A, labels, ind = run_one(md, sweep, seed, n_rep, batch, n_random)

        # per-group alignment + specificity
        for r in recs:
            d = r["deficit"]
            r["align"] = corr(r["activity"], d)
            r["max_drop"] = float(d.max())
            r["total_drop"] = float(d.sum())
            r["specificity"] = float(d.max() / d.sum()) if d.sum() > 1e-9 else np.nan

        def by(t):
            return [r for r in recs if r["type"] == t]

        s_align = np.array([r["align"] for r in by("structural")], dtype=float)
        r_align = np.array([r["align"] for r in by("random")], dtype=float)
        s_spec = np.array([r["specificity"] for r in by("structural")], dtype=float)
        r_spec = np.array([r["specificity"] for r in by("random")], dtype=float)
        s_max = np.array([r["max_drop"] for r in by("structural")], dtype=float)
        r_max = np.array([r["max_drop"] for r in by("random")], dtype=float)
        row = {
            "seed": seed, "k": kk, "intact_perf": float(perf_intact.mean()),
            "align_structural": float(np.nanmean(s_align)), "align_random": float(np.nanmean(r_align)),
            "spec_structural": float(np.nanmean(s_spec)), "spec_random": float(np.nanmean(r_spec)),
            "maxdrop_structural": float(np.nanmean(s_max)), "maxdrop_random": float(np.nanmean(r_max)),
        }
        agg.append(row)
        print(f"  align struct={row['align_structural']:.3f} vs random={row['align_random']:.3f} | "
              f"spec struct={row['spec_structural']:.2f} vs random={row['spec_random']:.2f} | "
              f"maxdrop struct={row['maxdrop_structural']:.3f} vs random={row['maxdrop_random']:.3f}", flush=True)

        # save per-seed detail (deficit + activity matrices)
        np.savez(OUTDIR / f"lesion_detail_{model}.npz",
                 rules=np.asarray(rules, dtype=object), perf_intact=perf_intact,
                 struct_deficit=np.array([r["deficit"] for r in by("structural")]),
                 struct_activity=np.array([r["activity"] for r in by("structural")]),
                 struct_size=np.array([r["size"] for r in by("structural")]),
                 random_deficit=np.array([r["deficit"] for r in by("random")]),
                 random_activity=np.array([r["activity"] for r in by("random")]),
                 activity_deficit=np.array([r["deficit"] for r in by("activity")]),
                 A_unit_task=A, labels=labels, ind_active=ind)
        tf.reset_default_graph()

    out = OUTDIR / "lesion_causal_summary.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys())); w.writeheader(); w.writerows(agg)
    import numpy as _np
    from scipy.stats import wilcoxon
    print("\n=== across seeds (seed = replication unit) ===")
    for metric in ["align", "spec", "maxdrop"]:
        s = _np.array([r[f"{metric}_structural"] for r in agg])
        rnd = _np.array([r[f"{metric}_random"] for r in agg])
        diff = s - rnd
        try:
            p = wilcoxon(diff).pvalue
        except Exception:
            p = float("nan")
        print(f"  {metric:8}: structural={s.mean():.3f}  random={rnd.mean():.3f}  "
              f"struct>rand {int((diff>0).sum())}/{len(diff)}  wilcoxon p={p:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
