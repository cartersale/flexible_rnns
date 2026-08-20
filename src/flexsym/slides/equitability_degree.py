"""Slide panel: raw equitability is largely degree-driven; lumpability is the training signature.

The strength-preserving row shuffle destroys the wiring pattern while keeping every unit's
total positive/negative input strength. If a measure cannot tell it apart from the trained
matrix, that measure is reading the strength (degree) distribution rather than training.

  A  2-D dissociation: equitability score (x) vs eigenvector step R^2 (y), 4 matrix types
     x 33 networks. Trained and row-shuffle are NOT separated horizontally but ARE vertically.
  B  trained-minus-shuffle gap on equitability, by network size -> straddles zero
  C  trained-minus-shuffle gap on step R^2, by network size     -> positive everywhere

Data: results/signed_input_wl_paper/training_control_all_sizes.csv (33 networks, 128-1024 units)
"""

from __future__ import annotations

from flexsym import paths

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = paths.REPO_ROOT
CTRL = paths.RESULTS_ROOT / "signed_input_wl_paper/training_control_all_sizes.csv"
OUT = paths.FIGURES_ROOT / "slides"

BG = "#ffffff"
INK = "#111111"
MUTED = "#5b5a57"
GRID = "#e3e2df"
BLUE = "#2a78d6"
ORANGE = "#eda100"
RED = "#eb6834"
GREY = "#9a9a97"

TYPES = ["trained", "row-shuffle", "full-shuffle", "gaussian"]
LABELS = ["trained", "row shuffle", "full shuffle", "matched Gaussian"]
COLORS = [BLUE, ORANGE, RED, GREY]
MARKERS = ["o", "D", "s", "^"]


def tidy(ax, title, fs=17):
    ax.set_title(title, fontsize=fs, color=INK, pad=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(1.2)
    ax.tick_params(colors=MUTED, labelsize=12.5, width=1.2, length=5)
    ax.set_axisbelow(True)


def load():
    rows = list(csv.DictReader(open(CTRL)))
    eq = {t: np.array([-float(r[f"{t}_equit_z"]) for r in rows]) for t in TYPES}  # score = -z
    lr = {t: np.array([float(r[f"{t}_lump_r2"]) for r in rows]) for t in TYPES}
    size = np.array([int(r["size"]) for r in rows])
    return eq, lr, size


def panel_scatter(ax, eq, lr):
    for t, lab, c, mk in zip(TYPES, LABELS, COLORS, MARKERS):
        ax.scatter(eq[t], lr[t], s=78, color=c, marker=mk, edgecolor=BG, linewidth=1.0,
                   alpha=0.85, zorder=4)
        ax.scatter([eq[t].mean()], [lr[t].mean()], s=340, color=c, marker=mk,
                   edgecolor=INK, linewidth=2.0, zorder=6)

    # the two comparisons that matter
    ax.axhspan(lr["row-shuffle"].mean(), lr["trained"].mean(), color=BLUE, alpha=0.07, zorder=1)
    ax.annotate("", xy=(eq["trained"].mean(), lr["trained"].mean()),
                xytext=(eq["row-shuffle"].mean(), lr["row-shuffle"].mean()),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2,
                                shrinkA=13, shrinkB=13), zorder=7)

    tidy(ax, "the two measures disagree")
    ax.set_xlabel("equitability score   ($-z$ vs size-matched null)", fontsize=15,
                  color=INK, labelpad=7)
    ax.set_ylabel("eigenvector step $R^2$", fontsize=15, color=INK, labelpad=8)
    ax.grid(True, color=GRID, linewidth=1.0)
    ax.set_ylim(0.15, 1.10)

    for i, (lab, c) in enumerate(zip(LABELS, COLORS)):
        ax.text(0.022, 0.985 - 0.062 * i, lab, transform=ax.transAxes, color=c,
                fontsize=15, fontweight="bold", ha="left", va="top", zorder=8)

    ax.text(0.98, 0.04, "same equitability,\nvery different lumpability",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=14.5, color=INK,
            linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.4", fc=BG, ec=GRID, lw=1.4), zorder=9)


SIZES = [128, 256, 512, 1024]


def panel_gap(ax, a, b, size, title, ylabel, fmt="{:+.2f}"):
    """Per-network trained-minus-shuffle gap, split by network size."""
    gap = a - b
    ax.axhline(0, color=INK, lw=1.6, zorder=3)
    rng = np.random.default_rng(0)
    for i, sz in enumerate(SIZES):
        g = gap[size == sz]
        if not len(g):
            continue
        jit = rng.uniform(-0.16, 0.16, len(g))
        ax.scatter(i + jit, g, s=52, color=[BLUE if v > 0 else ORANGE for v in g],
                   edgecolor=BG, linewidth=0.9, zorder=4)
        ax.plot([i - 0.26, i + 0.26], [g.mean()] * 2, color=INK, lw=2.6,
                solid_capstyle="butt", zorder=6)

    n_pos, n = int(np.sum(gap > 0)), len(gap)
    tidy(ax, title.format(n=n_pos, tot=n), fs=15)
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels([str(s) for s in SIZES], fontsize=12.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=13, color=INK, labelpad=5)
    ax.set_xlim(-0.5, len(SIZES) - 0.5)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    if gap.mean() > 0:
        ax.text(0.985, 0.045, "trained better $\\uparrow$", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=12.5, color=MUTED)
        ax.set_xlabel("network size (units)", fontsize=13.5, color=INK, labelpad=6)
    return n_pos


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eq, lr, size = load()
    n = len(size)

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0], height_ratios=[1, 1],
                          left=0.072, right=0.975, top=0.735, bottom=0.165,
                          wspace=0.34, hspace=0.58)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])

    panel_scatter(axA, eq, lr)
    up_eq = panel_gap(axB, eq["trained"], eq["row-shuffle"], size,
                      "equitability: trained higher in only {n}/{tot}",
                      "gap, equitability score")
    up_lr = panel_gap(axC, lr["trained"], lr["row-shuffle"], size,
                      "lumpability: trained higher in {n}/{tot}",
                      "gap, step $R^2$")

    fig.suptitle("Raw equitability is largely degree-driven —\n"
                 "lumpability is the training-created signature",
                 fontsize=26, color=INK, x=0.072, ha="left", y=0.975, va="top",
                 linespacing=1.3)
    fig.text(0.975, 0.018,
             "row shuffle destroys the wiring pattern but preserves every unit's total "
             "input strength   ·   33 networks, 128-1024 units",
             ha="right", va="bottom", fontsize=12.5, color=MUTED)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"slide_equitability_degree.{ext}", dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT / 'slide_equitability_degree.png'}")
    print(f"equitability: trained higher in {up_eq}/{n} networks "
          f"(trained {eq['trained'].mean():+.2f}, shuffle {eq['row-shuffle'].mean():+.2f})")
    print(f"step R^2:     trained higher in {up_lr}/{n} networks "
          f"(trained {lr['trained'].mean():.3f}, shuffle {lr['row-shuffle'].mean():.3f})")


if __name__ == "__main__":
    main()
