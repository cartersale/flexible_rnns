"""Fig 3 pooled pair-fate Sankey -> fig3/panelB_sankey.svg

Flows = excess reuse (above-chance probability that a unit pair grouped together
during one computation stays grouped during another), pooled over all cells and
all 10 seeds. Same-computation (diagonal) flows = reuse; different-computation =
reorganization.
"""
from __future__ import annotations

from flexsym import paths
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Rectangle
from flexsym.figures.figstyle import INK, MUTED, GRID, YELLOW, BLUE, ORANGE, GRAY, save

OUT = paths.REPO_ROOT / "results/figures/fig3"
DATA = paths.REPO_ROOT / "results/signed_input_wl_paper/pair_fate_3x3.json"
TYPECOL = {"stimulus": YELLOW, "memory": BLUE, "response": ORANGE}


def ribbon(ax, x0, y0, x1, y1, w, color, alpha):
    """Filled S-curve ribbon of constant width w from (x0,y0) to (x1,y1) (left/right edges)."""
    xm = (x0 + x1) / 2
    top = [(x0, y0 + w), (xm, y0 + w), (xm, y1 + w), (x1, y1 + w)]
    bot = [(x1, y1), (xm, y1), (xm, y0), (x0, y0)]
    verts = [top[0]] + top[1:] + [bot[0]] + bot[1:] + [top[0]]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
             MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=2))


def main():
    d = json.load(open(DATA))
    T = d["types"]
    # excess reuse (recompute if absent)
    if "excess_reuse" in d:
        E = d["excess_reuse"]
    else:
        br = d["base_rate"]; E = {a: {b: d["P_stay_together"][a][b]-br[b] for b in T} for a in T}
    W = np.array([[max(E[a][b], 0) for b in T] for a in T])   # widths (clip tiny negatives)

    fig, ax = plt.subplots(figsize=(5.2, 3.8)); fig.subplots_adjust(left=0.16, right=0.84, top=0.82, bottom=0.08)
    x0, x1, nodew = 0.12, 0.88, 0.045
    gap = 0.05
    src_tot = W.sum(1); dst_tot = W.sum(0)
    scale = (1 - 2*gap*(len(T)-1)) / max(src_tot.sum(), dst_tot.sum())

    # node y-extents (stacked top->bottom), sources left, dests right
    def stack(tots):
        ys = {}; y = 1.0
        for i, t in enumerate(T):
            h = tots[i]*scale; ys[t] = (y-h, y); y -= h + gap
        return ys
    S = stack(src_tot); D = stack(dst_tot)

    # draw ribbons: within each source, stack outgoing by dest; within each dest, stack incoming by source
    src_cur = {t: S[t][1] for t in T}   # current top y at source (fill downward)
    dst_cur = {t: D[t][1] for t in T}
    for i, a in enumerate(T):
        for j, b in enumerate(T):
            w = W[i, j]*scale
            if w <= 0:
                continue
            ya = src_cur[a] - w; yb = dst_cur[b] - w
            same = (a == b)
            ribbon(ax, x0+nodew, ya, x1-nodew, yb, w,
                   TYPECOL[a] if same else GRAY, 0.85 if same else 0.28)
            src_cur[a] -= w; dst_cur[b] -= w

    # node bars + labels
    for side, ys, xx, lab in [("src", S, x0, "grouped during"), ("dst", D, x1, "still grouped during")]:
        for t in T:
            y0, y1 = ys[t]
            ax.add_patch(Rectangle((xx-nodew if side == "dst" else xx, y0), nodew, y1-y0,
                         facecolor=TYPECOL[t], edgecolor="white", lw=1, zorder=4))
            tx = xx - nodew - 0.02 if side == "src" else xx + nodew + 0.02
            ha = "right" if side == "src" else "left"
            ax.text(tx, (y0+y1)/2, t, ha=ha, va="center", fontsize=8.5, color=INK)
    ax.text(x0, 1.045, "grouped during\nthis computation...", ha="center", fontsize=7.3, color=MUTED)
    ax.text(x1, 1.045, "...stays grouped\nduring this one", ha="center", fontsize=7.3, color=MUTED)

    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1.14); ax.axis("off")
    # legend
    ax.add_patch(Rectangle((0.30, -0.02), 0.03, 0.03, facecolor=BLUE, alpha=0.85, clip_on=False))
    ax.text(0.345, -0.005, "same computation (reuse)", fontsize=7, color=INK, va="center")
    ax.add_patch(Rectangle((0.30, -0.075), 0.03, 0.03, facecolor=GRAY, alpha=0.4, clip_on=False))
    ax.text(0.345, -0.06, "different computation (reorganize)", fontsize=7, color=INK, va="center")
    ax.set_title("Unit pairs reuse their grouping for the same computation\n(pooled over all cells and 10 networks)",
                 fontsize=10, color=INK, y=1.12)
    save(fig, OUT / "panelB_sankey.svg")


if __name__ == "__main__":
    main()
