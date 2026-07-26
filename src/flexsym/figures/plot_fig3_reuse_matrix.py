"""Fig 3 pooled task-epoch reuse matrix -> fig3/panelC_reuse_matrix.svg"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
STEP = Path(__file__).resolve().parent
sys.path.insert(0, str(STEP))
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from figstyle import INK, MUTED, GRID, YELLOW, BLUE, ORANGE, save

OUT = STEP.parent / "results/figures/fig3"
DATA = STEP.parent / "results/signed_input_wl_paper/epoch_reuse_matrix.json"
TYPECOL = {"stimulus": YELLOW, "memory": BLUE, "response": ORANGE}
ABBR = {"stim1": "stim", "delay1": "delay", "go1": "go"}


def main():
    d = json.load(open(DATA))
    M = np.array(d["matrix"], dtype=float)
    epochs = d["epochs"]; types = d["types"]
    n = len(epochs)
    labels = [e.replace("/", " ").replace("stim1", "stim").replace("delay1", "delay").replace("go1", "go") for e in epochs]
    Mm = M.copy(); np.fill_diagonal(Mm, np.nan)   # hide self-ARI (=1) so off-diagonal structure shows
    vmax = np.nanpercentile(Mm, 96)

    fig, ax = plt.subplots(figsize=(4.8, 4.4)); fig.subplots_adjust(left=0.28, right=0.86, top=0.80, bottom=0.26)
    cmap = plt.cm.get_cmap("magma_r").copy(); cmap.set_bad("#eceae6")
    im = ax.imshow(Mm, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    # block boundaries between types
    bnds = [i for i in range(1, n) if types[i] != types[i-1]]
    for b in bnds:
        ax.axhline(b-0.5, color="white", lw=2.5); ax.axvline(b-0.5, color="white", lw=2.5)
        ax.axhline(b-0.5, color=INK, lw=1.0); ax.axvline(b-0.5, color=INK, lw=1.0)
    # ticks + type-colored labels
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=5.6, rotation=90)
    ax.set_yticklabels(labels, fontsize=5.6)
    for tk, t in zip(ax.get_xticklabels(), types):
        tk.set_color(TYPECOL[t])
    for tk, t in zip(ax.get_yticklabels(), types):
        tk.set_color(TYPECOL[t])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # type color strips + labels along top
    seg = {}
    for i, t in enumerate(types):
        seg.setdefault(t, [i, i])[1] = i
    for t, (a, b) in seg.items():
        ax.add_patch(Rectangle((a-0.5, -1.4), b-a+1, 0.7, facecolor=TYPECOL[t], edgecolor="white", clip_on=False, zorder=5))
        ax.text((a+b)/2, -1.75, t, ha="center", va="bottom", fontsize=7.5, color=INK, clip_on=False)

    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.03)
    cb.set_label("sub-grouping similarity (ARI)", fontsize=7.5, color=MUTED)
    cb.ax.tick_params(labelsize=6.5, colors=MUTED); cb.outline.set_visible(False)

    ax.set_title("Task-epochs sharing a computation\nreuse the same within-cell sub-grouping",
                 fontsize=10, color=INK, y=1.14)
    ax.text(0.5, -0.32, "pooled over all cells and 10 networks; dark diagonal blocks = reuse within a computation type",
            transform=ax.transAxes, ha="center", fontsize=6.6, color=MUTED)
    save(fig, OUT / "panelC_reuse_matrix.svg")


if __name__ == "__main__":
    main()
