"""Build the motif participation cache for the tanh nets from their fixed-point outputs.

Runs after motif_fixed_points has produced results/motif_fixed_points/<net>/<rule>__<epoch>.npz.
Classifies each condition's motif (fp_inventory / motif_classify) and writes
participation_cache_v2.npz, using ind_active from the WL sweep written by tanh_lumpability. This is
the only step between fixed-point finding and the concentration analysis.
"""
from __future__ import annotations
import argparse
import numpy as np

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_participation_stats import collect_motifs

FP_ROOT = paths.REPO_ROOT / "results/motif_fixed_points"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation", default="tanh")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    for s in args.seeds.split(","):
        net = f"LeakyRNN_{args.activation}_diag_256_all_{s}"
        wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
        ind = np.asarray(wl["ind_active"], int)
        recs = collect_motifs(FP_ROOT / net, ind, use_cache=False)
        from collections import Counter
        tally = Counter(m["motif"] for m in recs)
        print(f"[{net}] {len(recs)} motif conditions: {dict(tally)}", flush=True)


if __name__ == "__main__":
    main()
