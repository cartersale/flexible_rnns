"""Fig 3A (cell regrouping) and Fig 4A (RSA method) schematics -> fig3/, fig4/*.svg"""
from __future__ import annotations

from flexsym import paths
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Circle, Rectangle
from flexsym.figures.figstyle import INK, MUTED, GRID, BLUE, YELLOW, GREEN, ORANGE, VIOLET, GRAY, save

F3 = paths.REPO_ROOT / "results/figures/fig3"
F4 = paths.REPO_ROOT / "results/figures/fig4"

# 6 units in one role cell, fixed positions (2x3 grid)
UPOS = {0: (-0.5, 0.35), 1: (0.0, 0.35), 2: (0.5, 0.35),
        3: (-0.5, -0.35), 4: (0.0, -0.35), 5: (0.5, -0.35)}


def _cell(ax, cx, cy, colors, title, scale=1.0):
    ax.add_patch(Ellipse((cx, cy), 1.7*scale, 1.25*scale, facecolor="#eef1f4", edgecolor="#cfd4d9", lw=1.3, zorder=1))
    for i, (x, y) in UPOS.items():
        ax.add_patch(Circle((cx+x*scale, cy+y*scale), 0.15*scale, facecolor=colors[i], edgecolor="white", lw=1.2, zorder=4))
    ax.text(cx, cy+0.95*scale, title, ha="center", fontsize=8.5, color=INK)


def fig3A():
    fig, ax = plt.subplots(figsize=(6.4, 2.7)); fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.03)
    ax.set_xlim(0, 9.8); ax.set_ylim(-1.7, 1.7); ax.axis("off")
    _cell(ax, 1.3, 0.05, [GRAY]*6, "one role cell")
    A = [BLUE, BLUE, BLUE, ORANGE, ORANGE, ORANGE]           # task A: {0,1,2}/{3,4,5}
    B = [BLUE, GREEN, ORANGE, BLUE, GREEN, ORANGE]           # task B: {0,3}/{1,4}/{2,5}
    ax.add_patch(FancyArrowPatch((2.5, 0.3), (3.6, 0.85), arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.4))
    ax.add_patch(FancyArrowPatch((2.5, -0.2), (3.6, -0.85), arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.4))
    ax.text(2.85, 0.75, "task A", fontsize=7.5, color=MUTED, ha="center")
    ax.text(2.85, -0.75, "task B", fontsize=7.5, color=MUTED, ha="center")
    _cell(ax, 5.0, 0.85, A, "", scale=0.8)
    _cell(ax, 5.0, -0.85, B, "", scale=0.8)
    ax.text(7.6, 0.05, "the same units split into\ndifferent functional\nsub-populations for\ndifferent computations",
            ha="center", va="center", fontsize=7.6, color=INK)
    ax.set_title("Task-driven reorganization within a role cell", fontsize=10.5, color=INK, y=1.0)
    save(fig, F3 / "panelA_schematic.svg")


def fig4A():
    fig, ax = plt.subplots(figsize=(6.6, 2.7)); fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.04)
    ax.set_xlim(0, 11.5); ax.set_ylim(-0.3, 3.0); ax.axis("off")
    rng = np.random.default_rng(3)
    m = 6

    def simmat(cx, cy, M, title, cmap, sub=""):
        s = 0.9; cell = s/m
        vmax = np.abs(M).max()
        for i in range(m):
            for j in range(m):
                v = M[i, j]/vmax
                ax.add_patch(Rectangle((cx+j*cell, cy-i*cell), cell*0.94, cell*0.94,
                             facecolor=plt.cm.get_cmap(cmap)(0.5+0.45*v), edgecolor="none", zorder=3))
        ax.text(cx+s/2, cy+0.16, title, ha="center", fontsize=8, color=INK)
        if sub:
            ax.text(cx+s/2, cy-s-0.16, sub, ha="center", fontsize=6.8, color=MUTED)

    # cell units (left)
    ax.text(0.7, 2.55, "units in\na role cell", ha="center", fontsize=8, color=INK)
    for i, (x, y) in UPOS.items():
        ax.add_patch(Circle((0.7+x*0.6, 1.5+y*0.9), 0.11, facecolor=GRAY, edgecolor="white", lw=1.0, zorder=4))

    # three similarity matrices
    base = rng.standard_normal((m, m)); base = (base+base.T)/2
    F = base + 0.6*rng.standard_normal((m, m)); F = (F+F.T)/2
    simmat(1.9, 2.5, F, "functional\nsimilarity", "RdBu_r", "(activity)")
    simmat(1.9, 1.1, base + 0.5*rng.standard_normal((m, m)), "full-context\nsimilarity", "PuOr", "(whole-network wiring)")
    simmat(3.4, 1.1, 0.4*base + rng.standard_normal((m, m)), "internal\nsimilarity", "PuOr", "(local wiring)")

    ax.add_patch(FancyArrowPatch((4.55, 1.55), (5.35, 1.55), arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.4))
    ax.text(4.95, 1.95, "partial\ncorrelation", ha="center", fontsize=6.8, color=MUTED)

    # inline scatter drawn in main-axes coords (box 5.7..8.2 x, 0.7..2.4 y)
    bx0, bx1, by0, by1 = 5.9, 8.1, 0.75, 2.35
    ax.add_patch(Rectangle((bx0, by0), bx1-bx0, by1-by0, fill=False, edgecolor=MUTED, lw=1.0))
    iu = np.triu_indices(m, 1)
    xg = base[iu]-base[iu].mean(); yg = F[iu]-F[iu].mean()
    xn = (xg-xg.min())/(np.ptp(xg)+1e-9); yn = (yg-yg.min())/(np.ptp(yg)+1e-9)
    px = bx0+0.12*(bx1-bx0)+xn*0.76*(bx1-bx0); py = by0+0.14*(by1-by0)+yn*0.72*(by1-by0)
    ax.scatter(px, py, s=13, color=BLUE, edgecolor="white", lw=0.4, zorder=5)
    b1, b0 = np.polyfit(px, py, 1); xs = np.array([px.min(), px.max()])
    ax.plot(xs, b0+b1*xs, color=INK, lw=1.4, zorder=6)
    ax.text((bx0+bx1)/2, by1+0.12, "unit pairs", ha="center", fontsize=6.8, color=MUTED)
    ax.text((bx0+bx1)/2, by0-0.16, "full-context similarity", ha="center", fontsize=6.5, color=MUTED)
    ax.text(bx0-0.12, (by0+by1)/2, "functional\nsimilarity", ha="center", va="center", rotation=90, fontsize=6.5, color=MUTED)

    ax.text(9.85, 1.75, r"$r(F,\,\mathrm{full}\mid\mathrm{internal})$", ha="center", fontsize=8.5, color=INK)
    ax.text(9.85, 1.15, "does whole-network wiring\npredict functional grouping\nbeyond local wiring?", ha="center", fontsize=6.8, color=MUTED, style="italic")

    ax.set_xlim(0, 11.8)
    ax.set_title("Similarity regression: connectivity vs function", fontsize=10.5, color=INK, y=1.0)
    save(fig, F4 / "panelA_schematic.svg")


if __name__ == "__main__":
    fig3A(); fig4A()
