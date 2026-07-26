"""Motif taxonomy for every network with fixed-point results, in one CSV.

plot_motif_validation.py writes a per-network motif_taxonomy.csv as a side effect of
its (expensive) plotting. This runs only the classification, over all networks, and
writes a single combined table plus the seeds 0-2 validation-gate counts.

Gate definitions match plot_motif_validation.py exactly:
  memory delays   -> line_attractor flag, epochs delay1/delay2 with epoch_type memory
  decision stim   -> n_saddle > 0, for DECISION_RULES stim epochs
  sensory stim    -> condition_motif == point_attractor, non-decision stim epochs

Writes results/motif_fixed_points/motif_taxonomy_all_nets.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

STEP_DIR = Path(__file__).resolve().parent
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from motif_classify import classify_net  # noqa: E402
from plot_motif_validation import DECISION_RULES  # noqa: E402

REPO = STEP_DIR.parent
ROOT = REPO / "results/motif_fixed_points"
OUT = ROOT / "motif_taxonomy_all_nets.csv"


def main():
    nets = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("LeakyRNN"))
    rows = []
    for net in nets:
        size = int(net.name.split("_")[3])
        seed = int(net.name.split("_all_")[1])
        res = classify_net(net)
        for cm in res.values():
            rows.append({
                "net": net.name, "size": size, "seed": seed,
                "rule": cm.rule, "epoch": cm.epoch, "epoch_type": cm.epoch_type,
                "motif": cm.condition_motif,
                "line_attractor": bool(cm.line_attractor),
                "n_fp": cm.extra["n_fp"], "n_saddle": cm.extra["n_saddle"],
                "n_rotation": cm.extra["n_rotation"],
                "n_point_attractor": cm.extra["n_point_attractor"],
                "is_decision_rule": cm.rule in DECISION_RULES,
            })
        print(f"{net.name}: {len(res)} epochs", flush=True)

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}  ({len(rows)} epochs, {len(nets)} networks)")

    for label, seeds in (("seeds 0-2 (released nets)", {0, 1, 2}),
                         ("all networks", set(range(10)))):
        sub = [r for r in rows if r["seed"] in seeds]
        mem = [r for r in sub if r["epoch_type"] == "memory"
               and r["epoch"] in ("delay1", "delay2")]
        ds = [r for r in sub if r["is_decision_rule"] and r["epoch_type"] == "stim"]
        ss = [r for r in sub if not r["is_decision_rule"] and r["epoch_type"] == "stim"]
        print(f"\n=== {label} ===")
        print(f"  memory delays -> line attractor: "
              f"{sum(r['line_attractor'] for r in mem)}/{len(mem)}")
        print(f"  decision stim -> >=1 saddle:     "
              f"{sum(r['n_saddle'] > 0 for r in ds)}/{len(ds)}")
        print(f"  sensory stim  -> point attractor:"
              f" {sum(r['motif'] == 'point_attractor' for r in ss)}/{len(ss)}")


if __name__ == "__main__":
    main()
