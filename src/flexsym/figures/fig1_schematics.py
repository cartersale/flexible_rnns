"""Fig 1 schematic panels A (task paradigm) and B (partition method) -> fig1/*.svg"""
from __future__ import annotations

from flexsym import paths
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from flexsym.figures.figstyle import INK, MUTED, GRID, BLUE, YELLOW, GREEN, MAGENTA, ORANGE, AQUA, VIOLET, RED, GRAY, SURF, save, clean

OUT = paths.FIGURES_ROOT / "fig1"

# epoch-type colors (shared components)
EP = {"fixation": "#c9c8c3", "stimulus": YELLOW, "delay": BLUE, "response": ORANGE}

# ---- shared toy network for the method panels ----
HUES = [BLUE, GREEN, ORANGE]
GROUPS = [0, 0, 1, 1, 2, 2]
POS = {0: (-0.55, 0.85), 1: (0.05, 1.05), 2: (1.05, 0.35), 3: (1.05, -0.45),
       4: (-0.35, -0.85), 5: (-0.95, -0.15)}
EDGES = [(0, 1, 2.2), (1, 2, 2.6), (2, 3, 1.4), (3, 4, 1.6), (4, 5, 2.0),
         (5, 0, 1.5), (0, 4, 1.0), (2, 5, 1.2), (1, 3, 1.4), (3, 5, 0.9)]


def _draw_net(ax, cx, cy, node_colors, bubbles=None, node_r=0.14):
    from matplotlib.patches import Ellipse
    for (s, d, w) in EDGES:
        x0, y0 = POS[s]; x1, y1 = POS[d]
        ax.add_patch(FancyArrowPatch((cx+x0, cy+y0), (cx+x1, cy+y1), arrowstyle="-|>",
                     mutation_scale=6, color="#c2c1bc", lw=0.5+w*0.5, alpha=0.9, zorder=3,
                     connectionstyle="arc3,rad=0.12", shrinkA=5, shrinkB=5))
    if bubbles is not None:
        for g, hue in enumerate(HUES):
            mem = [i for i in POS if bubbles[i] == g]
            if not mem:
                continue
            mx, my = np.mean([cx+POS[i][0] for i in mem]), np.mean([cy+POS[i][1] for i in mem])
            ax.add_patch(Ellipse((mx, my), 0.95, 0.75, facecolor=hue, edgecolor=hue, alpha=0.15, lw=1.4, zorder=1))
    for i, (x, y) in POS.items():
        ax.add_patch(Circle((cx+x, cy+y), node_r, facecolor=node_colors[i], edgecolor="white", lw=1.1, zorder=5))


def panel_A():
    """Task paradigm: example tasks as epoch timelines; shared components color-coded."""
    # (task label, [(epoch_type, width), ...], mapping note)
    tasks = [
        ("Reaction-Pro",  [("fixation", 1.2), ("response", 1.6)], "respond to stimulus"),
        ("Delay-Pro",     [("fixation", 1.0), ("stimulus", 1.0), ("delay", 1.6), ("response", 1.2)], "remember, then respond"),
        ("Delay-Anti",    [("fixation", 1.0), ("stimulus", 1.0), ("delay", 1.6), ("response", 1.2)], "remember, respond opposite"),
        ("Delay-DM",      [("fixation", 1.0), ("stimulus", 1.4), ("delay", 1.6), ("response", 1.2)], "compare, then respond"),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.2)); fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.14)
    y = len(tasks)
    row_h = 0.62
    for name, epochs, note in tasks:
        x = 0.0
        for et, w in epochs:
            ax.add_patch(FancyBboxPatch((x, y-row_h/2), w, row_h, boxstyle="round,pad=0.0,rounding_size=0.05",
                                        facecolor=EP[et], edgecolor="white", linewidth=1.4, zorder=3))
            if w > 0.9:
                ax.text(x+w/2, y, et, ha="center", va="center", fontsize=7.2, color=INK, zorder=4)
            x += w + 0.04
        ax.text(-0.15, y, name, ha="right", va="center", fontsize=9, color=INK, fontweight="bold")
        ax.text(x+0.12, y, note, ha="left", va="center", fontsize=7.3, color=MUTED, style="italic")
        y -= 1
    ax.set_xlim(-2.0, 7.6); ax.set_ylim(0.3, len(tasks)+0.7)
    ax.axis("off")
    ax.set_title("15 tasks recombine a small set of task epochs", fontsize=11, color=INK, y=1.0)
    # legend of shared components
    lx = 0.0
    for et in ["fixation", "stimulus", "delay", "response"]:
        ax.add_patch(Rectangle((lx, 0.45), 0.3, 0.22, facecolor=EP[et], edgecolor="white", clip_on=False))
        ax.text(lx+0.36, 0.56, et, fontsize=7.3, color=MUTED, va="center")
        lx += 1.5
    save(fig, OUT / "panelA_task_paradigm.svg")


def panel_B():
    """Method as a graph story: network (W + graph) -> signed WL -> role cells (bubbles) -> quotient."""
    fig, ax = plt.subplots(figsize=(7.4, 2.9)); fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.02)
    ax.set_xlim(0, 12.4); ax.set_ylim(-0.4, 3.4); ax.axis("off")
    rng = np.random.default_rng(4)

    hues = [BLUE, GREEN, ORANGE]
    groups = [0, 0, 1, 1, 2, 2]           # 3 role cells of 2 units
    # node layout (shared by stage 1 and stage 2), clustered by group so bubbles are clean
    P = {0: (-0.55, 0.85), 1: (0.05, 1.05), 2: (1.05, 0.35), 3: (1.05, -0.45),
         4: (-0.35, -0.85), 5: (-0.95, -0.15)}
    edges = [(0, 1, 2.2), (1, 2, 2.6), (2, 3, 1.4), (3, 4, 1.6), (4, 5, 2.0),
             (5, 0, 1.5), (0, 4, 1.0), (2, 5, 1.2), (1, 3, 1.4), (3, 5, 0.9)]

    def draw_graph(cx, cy, node_col, bubbles=False):
        for (s, d, w) in edges:
            x0, y0 = P[s]; x1, y1 = P[d]
            ax.add_patch(FancyArrowPatch((cx+x0, cy+y0), (cx+x1, cy+y1), arrowstyle="-|>",
                         mutation_scale=7, color="#b8b7b2", lw=0.5+w*0.55, alpha=0.9, zorder=3,
                         connectionstyle="arc3,rad=0.12",
                         shrinkA=6, shrinkB=6))
        if bubbles:
            for g, hue in enumerate(hues):
                mem = [i for i in P if groups[i] == g]
                xs = [cx+P[i][0] for i in mem]; ys = [cy+P[i][1] for i in mem]
                mx, my = np.mean(xs), np.mean(ys)
                from matplotlib.patches import Ellipse
                ax.add_patch(Ellipse((mx, my), 0.95, 0.75, facecolor=hue, edgecolor=hue,
                             alpha=0.16, lw=1.6, zorder=1))
        for i, (x, y) in P.items():
            c = hues[groups[i]] if node_col == "group" else "#7d7c78"
            ax.add_patch(Circle((cx+x, cy+y), 0.14, facecolor=c, edgecolor="white", lw=1.2, zorder=5))

    # ---- Stage 1: network (graph + matrix) ----
    gx, gy = 1.3, 2.0
    draw_graph(gx, gy, node_col="gray")
    ax.text(gx-1.1, gy+0.6, "network", ha="center", fontsize=8.5, color=INK)
    # matrix below
    W = rng.standard_normal((6, 6)); W = np.clip(W/2.3, -1, 1)
    cell = 0.16; mx0, my0 = gx-0.48, 0.9
    for i in range(6):
        for j in range(6):
            ax.add_patch(Rectangle((mx0+j*cell, my0-i*cell), cell*0.92, cell*0.92,
                         facecolor=plt.cm.RdBu_r(0.5+0.5*W[i, j]), edgecolor="none", zorder=3))
    ax.text(mx0+3*cell, my0+cell*1.2, "$W_{\\rm rec}$", ha="center", fontsize=8, color=INK)

    # arrow 1
    ax.add_patch(FancyArrowPatch((2.9, 1.7), (4.0, 1.7), arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.5))
    ax.text(3.45, 2.05, "signed WL\nrefinement", ha="center", fontsize=7, color=MUTED)

    # ---- Stage 2: role cells (bubbles) ----
    draw_graph(5.5, 1.7, node_col="group", bubbles=True)
    ax.text(5.5, 3.0, "role cells", ha="center", fontsize=8.5, color=INK)

    # arrow 2
    ax.add_patch(FancyArrowPatch((7.3, 1.7), (8.4, 1.7), arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.5))
    ax.text(7.85, 2.0, "quotient", ha="center", fontsize=7, color=MUTED)

    # ---- Stage 3: quotient graph ----
    q = {0: (9.9, 2.4), 1: (11.4, 1.7), 2: (10.1, 0.7)}
    qedges = [(0, 1, 2.6), (1, 2, 1.6), (2, 0, 1.8), (0, 2, 1.0)]
    for (s, d, w) in qedges:
        x0, y0 = q[s]; x1, y1 = q[d]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                     color="#a8a7a2", lw=0.6+w*0.6, zorder=3, shrinkA=13, shrinkB=13,
                     connectionstyle="arc3,rad=0.12"))
    for g, (x, y) in q.items():
        ax.add_patch(Circle((x, y), 0.30, facecolor=hues[g], edgecolor="white", lw=1.6, zorder=5))
    ax.text(10.5, 3.0, "quotient graph", ha="center", fontsize=8.5, color=INK)

    ax.set_title("Role cells from connectivity: the recurrent graph and its reduced description",
                 fontsize=10.5, color=INK, y=1.03)
    save(fig, OUT / "panelB_method.svg")


def panel_B_refinement():
    """Colour refinement: iteratively relabel each unit by its neighbours' roles until stable."""
    fig, ax = plt.subplots(figsize=(6.8, 2.7)); fig.subplots_adjust(left=0.01, right=0.99, top=0.72, bottom=0.06)
    ax.set_xlim(0, 10.5); ax.set_ylim(-2.25, 1.5); ax.axis("off")
    G = "#8f8e8a"
    stages = [
        (1.3, [G]*6, "start:\none role"),
        (4.4, [HUES[g] for g in [0, 0, 1, 1, 0, 1]], "iteration 1"),
        (7.5, [HUES[g] for g in GROUPS], "stable:\nrole cells"),
    ]
    for k, (cx, cols, lab) in enumerate(stages):
        _draw_net(ax, cx, 0.15, cols, node_r=0.15)
        ax.text(cx, -1.25, lab, ha="center", fontsize=7.5, color=INK)
        if k < 2:
            ax.add_patch(FancyArrowPatch((cx+1.35, 0.15), (cx+1.75, 0.15), arrowstyle="-|>",
                         mutation_scale=13, color=MUTED, lw=1.5))
    ax.text(5.25, 1.35, "relabel each unit by the roles of the units it connects to", ha="center", fontsize=7, color=MUTED)
    # loop-back arrow
    ax.add_patch(FancyArrowPatch((7.5, -1.75), (4.4, -1.75), arrowstyle="-|>", mutation_scale=11, color="#c2c1bc", lw=1.3,
                 connectionstyle="arc3,rad=0.22"))
    ax.text(5.95, -2.15, "repeat until the labelling stops changing", ha="center", fontsize=6.8, color=MUTED, style="italic")
    ax.set_title("Signed colour refinement (Weisfeiler--Leman)", fontsize=10.5, color=INK, y=1.02)
    save(fig, OUT / "panelB_refinement.svg")


def panel_B_ksweep():
    """Sweep the number of cells k; the structural partition is far more equitable than chance."""
    fig, ax = plt.subplots(figsize=(4.2, 2.7)); fig.subplots_adjust(left=0.15, right=0.96, top=0.82, bottom=0.17)
    k = np.linspace(3, 60, 200)
    # schematic: gamma decreases with k for both; structural stays below the null
    null = 120 * np.exp(-k/34) + 40
    struct = null - (32 * np.exp(-((k-33)**2)/620))  # dips furthest below null near the operating scale
    ax.plot(k, null, "--", color=MUTED, lw=1.8, label="random partitions")
    ax.plot(k, struct, "-", color=BLUE, lw=2.2, label="signed-WL partition")
    ax.fill_between(k, struct, null, color=BLUE, alpha=0.12)
    kop = 33
    ax.axvline(kop, color=INK, lw=1.2, ls=":")
    ax.scatter([kop], [struct[np.argmin(np.abs(k-kop))]], s=45, color=BLUE, edgecolor="white", zorder=6)
    ax.text(kop+1.5, struct[np.argmin(np.abs(k-kop))]+6, "operating $k$", fontsize=7.5, color=INK)
    ax.annotate("more equitable\nthan chance", xy=(33, (struct[np.argmin(np.abs(k-33))]+null[np.argmin(np.abs(k-33))])/2),
                xytext=(44, 95), fontsize=7.2, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))
    ax.set_xlabel("number of role cells $k$", fontsize=9.5, color=INK)
    ax.set_ylabel(r"$\Gamma_{\rm EP}$  (non-equitability)", fontsize=9.5, color=INK)
    ax.set_title("Sweep $k$, compare $\\Gamma_{\\rm EP}$ to chance", fontsize=10, color=INK, pad=6)
    ax.set_yticks([]); ax.legend(loc="upper right", frameon=False, fontsize=7.8)
    clean(ax)
    save(fig, OUT / "panelB_ksweep.svg")


if __name__ == "__main__":
    panel_A(); panel_B(); panel_B_refinement(); panel_B_ksweep()
