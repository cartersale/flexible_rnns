"""Paper Figure 1 -- the surrogate-matrix control panel.

Step-R^2 of the trained recurrent matrix against three destructive surrogates, one
point per network in the cohort (network count read from the data, not hardcoded).

  trained        the real W_rec (transpose variant, active units)
  row-shuffle    entries permuted within each row: every unit's incoming strength
                 profile is preserved, only who-connects-to-whom is destroyed
  full-shuffle   all entries permuted: only the global value distribution survives
  gaussian       i.i.d. normal matched in Frobenius norm -- the untrained geometry

Every surrogate got the full pipeline in run/control_ksweep.py: a fresh signed-WL
partition optimised on the surrogate itself, scored against its own size-matched
random-partition null. A low score means the structure is absent, not that the
surrogate was handed the wrong partition.

Statistic: `r2_topk` (mean step-R^2 over the k leading eigenvectors), plotted as the
excess over each matrix's own matched null. `r2_best` is deliberately NOT used here --
per docs/paper_1_results_overview.md 1.4 it fails as a comparative statistic across
matrices (the spectrum surrogate beats trained on it 92% of the time).

Reads results/signed_input_wl_paper/lumpability/control_ksweep.csv; writes
figures/fig1/pieces/fig1_controls_r2topk_k<k>.{svg,png}.

    python -m flexsym.figures.paper_fig1_controls
    python -m flexsym.figures.paper_fig1_controls --k 30 --stat raw
"""
from __future__ import annotations

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flexsym import paths
from flexsym.figures.figstyle import GRID, INK, MUTED, SIZE_COL

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

CSV = (paths.RESULTS_ROOT / "signed_input_wl_paper" / "lumpability"
       / "control_ksweep.csv")
PIECES = paths.REPO_ROOT / "figures" / "fig1" / "pieces"

#: k = 20 to match the other Figure 1 panels; it is one of the 11 control k values.
K = 20
#: the destructive surrogates, in order of how much of W_rec they keep.
TYPES = [("trained", "trained"), ("row-shuffle", "row\nshuffle"),
         ("full-shuffle", "full\nshuffle"), ("gaussian", "Gaussian")]
BOX = "#dfe4ea"


def load(k: int, stat: str):
    rows = list(csv.DictReader(paths.require(CSV, "control k-sweep").open()))
    out = {}
    for key, _ in TYPES:
        sel = [r for r in rows if r["type"] == key and int(r["k"]) == k]
        vals = np.array([float(r["r2_topk"]) - (float(r["r2_topk_null_mean"])
                                                if stat == "excess" else 0.0)
                         for r in sel])
        sizes = np.array([int(r["size"]) for r in sel])
        models = [r["model"] for r in sel]
        order = np.argsort(models)                     # same network order per group
        out[key] = dict(v=vals[order], size=sizes[order],
                        model=[models[i] for i in order])
    return out


def panel(data, k, stat, stem):
    labels = [lab for _, lab in TYPES]
    vals = [data[key]["v"] for key, _ in TYPES]
    # cohort facts come from the data, so the panel cannot mislabel itself when
    # the cohort changes (it previously hardcoded 33 networks and a 1024 legend
    # entry, neither of which survives the released/main cohort split)
    n_nets = len(data[TYPES[0][0]]["v"])
    sizes_present = sorted(set(data[TYPES[0][0]]["size"].tolist()))

    fig = plt.figure(figsize=(5.2, 3.3))
    ax = fig.add_axes([0.155, 0.165, 0.820, 0.740])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    bp = ax.boxplot(vals, widths=0.50, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.4),
                    boxprops=dict(facecolor=BOX, edgecolor=MUTED, lw=0.8),
                    whiskerprops=dict(color=MUTED, lw=0.8),
                    capprops=dict(color=MUTED, lw=0.8))
    for b in bp["boxes"]:
        b.set_alpha(0.85)

    rng = np.random.default_rng(0)
    for i, (key, _) in enumerate(TYPES):
        v, sz = data[key]["v"], data[key]["size"]
        x = i + 1 + rng.uniform(-0.155, 0.155, len(v))
        ax.scatter(x, v, s=13, c=[SIZE_COL[s] for s in sz], alpha=0.85,
                   linewidths=0.4, edgecolors="white", zorder=4)

    if stat == "excess":
        ax.axhline(0.0, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.set_ylabel("step-$R^2$ excess over matched null", fontsize=8.5, color=INK)
    else:
        ax.set_ylabel("step-$R^2$ (mean over $k$ leading eigenvectors)",
                      fontsize=8.5, color=INK)

    ax.set_xticklabels(labels, fontsize=8.5, color=INK, linespacing=1.25)
    ax.tick_params(colors=MUTED, labelsize=8, length=2.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)

    ax.set_title(f"$k$ = {k},  {n_nets} networks", fontsize=8.5, color=MUTED, pad=6)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=4.2, mec="white", mew=0.4,
                          color=SIZE_COL[s], label=f"{s}") for s in sizes_present]
    leg = ax.legend(handles=handles, title="units", loc="upper right", fontsize=6.8,
                    title_fontsize=6.8, frameon=False, handletextpad=0.3,
                    labelspacing=0.25, borderpad=0.1)
    leg.get_title().set_color(MUTED)
    for t in leg.get_texts():
        t.set_color(MUTED)

    PIECES.mkdir(parents=True, exist_ok=True)
    for ext, kw in ((".svg", {}), (".png", dict(dpi=400))):
        fig.savefig(PIECES / f"{stem}{ext}", bbox_inches="tight", transparent=True, **kw)
    plt.close(fig)
    print(f"wrote {PIECES / stem}.svg / .png")


def report(data):
    ref = data["trained"]
    print(f"  trained: median {np.median(ref['v']):.3f}  "
          f"[{ref['v'].min():.3f}, {ref['v'].max():.3f}]  n = {len(ref['v'])}")
    for key, _ in TYPES[1:]:
        d = data[key]
        assert d["model"] == ref["model"]              # paired, network by network
        gap = ref["v"] - d["v"]
        print(f"  {key:13s} median {np.median(d['v']):.3f}   "
              f"trained wins {100 * (gap > 0).mean():.0f}% of {len(gap)} nets, "
              f"median gap {np.median(gap):+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--stat", default="excess", choices=["excess", "raw"],
                    help="excess over each matrix's own matched null, or raw r2_topk")
    args = ap.parse_args()

    data = load(args.k, args.stat)
    print(f"control k-sweep, k = {args.k}, statistic = r2_topk ({args.stat})")
    report(data)
    panel(data, args.k, args.stat, f"fig1_controls_r2topk_{args.stat}_k{args.k}")


if __name__ == "__main__":
    main()
