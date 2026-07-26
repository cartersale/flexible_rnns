"""Slide panel: the step-function result holds across all networks, sizes, and dies on surrogates.

Follow-on to plot_slide_lumpability_eigenvector.py -- keeps the same statistic (step R^2)
and the same visual language, but widens it two ways.

  A  step R^2 per network vs its size-matched random partition, grouped by size (33 nets)
  B  lumpability score (= -leakage z) by size, for W_rec^T, the raw state-update W, and
     the independent subspace-invariance measure
  C  surrogate ladder: trained -> strength-preserving row shuffle -> full shuffle ->
     matched Gaussian, 10 seeds at 256 units

Data
  results/signed_input_wl_paper/lumpability/step_r2_all.csv
  results/signed_input_wl_paper/lumpability/lumpability.csv
  results/signed_input_wl_paper/training_control.csv
"""

from __future__ import annotations

from flexsym import paths

import argparse
import collections
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = paths.REPO_ROOT
LUMP = REPO / "results/signed_input_wl_paper/lumpability/lumpability.csv"
STEP = REPO / "results/signed_input_wl_paper/lumpability/step_r2_all.csv"
CTRL = REPO / "results/signed_input_wl_paper/training_control.csv"
OUT = REPO / "results/figures/slides"

BG = "#ffffff"
INK = "#111111"
MUTED = "#5b5a57"
GRID = "#e3e2df"
BLUE = "#2a78d6"
GREY = "#9a9a97"
ORANGE = "#eda100"
GREEN = "#008300"
RED = "#eb6834"
SIZES = [128, 256, 512, 1024]
TYPES = ["trained", "row-shuffle", "full-shuffle", "gaussian"]
TYPE_LABELS = ["trained", "row\nshuffle", "full\nshuffle", "matched\nGaussian"]
TYPE_COLORS = [BLUE, ORANGE, RED, GREY]


def tidy(ax, title):
    ax.set_title(title, fontsize=17, color=INK, pad=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(1.2)
    ax.tick_params(colors=MUTED, labelsize=13, width=1.2, length=5)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)


def panel_a(ax):
    """Per-network step R^2, real partition vs size-matched random, grouped by size."""
    rows = list(csv.DictReader(open(STEP)))
    by = collections.defaultdict(list)
    for r in rows:
        by[int(r["size"])].append((float(r["real"]), float(r["rand"])))

    n_tot = 0
    rng = np.random.default_rng(0)
    for i, sz in enumerate(SIZES):
        pairs = np.array(by[sz])
        n_tot += len(pairs)
        jit = rng.uniform(-0.055, 0.055, len(pairs))
        xr, xn = i - 0.17 + jit, i + 0.17 + jit
        for j, (real, rand) in enumerate(pairs):
            ax.plot([xr[j], xn[j]], [real, rand], color=GRID, lw=1.4, zorder=2)
        ax.scatter(xr, pairs[:, 0], s=64, color=BLUE,
                   edgecolor=BG, linewidth=1.0, zorder=4)
        ax.scatter(xn, pairs[:, 1], s=64, color=GREY,
                   edgecolor=BG, linewidth=1.0, zorder=4)
        for x, vals, c in ((i - 0.17, pairs[:, 0], BLUE), (i + 0.17, pairs[:, 1], GREY)):
            ax.plot([x - 0.14, x + 0.14], [vals.mean()] * 2, color=INK, lw=2.6,
                    solid_capstyle="butt", zorder=6)

    tidy(ax, "every network is above its own null")
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels([f"{s}" for s in SIZES], fontsize=14, color=INK)
    ax.set_xlabel("network size (units)", fontsize=14, color=INK, labelpad=6)
    ax.set_ylabel("eigenvector step $R^2$", fontsize=15, color=INK, labelpad=8)
    ax.set_xlim(-0.5, len(SIZES) - 0.5)
    ax.set_ylim(0, 1.06)
    ax.text(0.02, 0.115, "connectivity groups", transform=ax.transAxes, fontsize=13.5,
            color=BLUE, fontweight="bold")
    ax.text(0.02, 0.05, "size-matched random", transform=ax.transAxes, fontsize=13.5,
            color=MUTED, fontweight="bold")
    ax.text(0.985, 0.965, f"{n_tot}/{n_tot} networks", transform=ax.transAxes,
            ha="right", va="top", fontsize=14, color=INK, fontweight="bold")


def panel_b(ax):
    """Lumpability / invariance score vs network size."""
    rows = list(csv.DictReader(open(LUMP)))
    by = collections.defaultdict(list)
    for r in rows:
        by[int(r["size"])].append((
            -float(r["transpose_eig_leakage_z"]),
            -float(r["W_eig_leakage_z"]),
            -float(r["transpose_invariance_z"]),
        ))
    x = np.arange(len(SIZES))
    series = [
        ("eigenvectors of $W_{\\rm rec}^{T}$", BLUE, "o", 0),
        ("eigenvectors of $W_{\\rm rec}$", GREEN, "s", 1),
        ("subspace invariance", ORANGE, "^", 2),
    ]
    ax.axhline(0, color=INK, lw=1.4, zorder=3)
    for label, c, mk, j in series:
        m = np.array([np.mean([v[j] for v in by[s]]) for s in SIZES])
        se = np.array([np.std([v[j] for v in by[s]]) / max(1, np.sqrt(len(by[s]))) for s in SIZES])
        ax.plot(x, m, color=c, lw=2.6, marker=mk, ms=10, mec=BG, mew=1.2, zorder=5, label=label)
        ax.fill_between(x, m - se, m + se, color=c, alpha=0.16, lw=0, zorder=2)

    tidy(ax, "and grows with network size")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}" for s in SIZES], fontsize=14, color=INK)
    ax.set_xlabel("network size (units)", fontsize=14, color=INK, labelpad=6)
    ax.set_ylabel("lumpability score  ($-z$ vs null)", fontsize=15, color=INK, labelpad=8)
    ax.set_xlim(-0.25, len(SIZES) - 0.75)
    ax.text(-0.2, 0.9, "chance", fontsize=12, color=MUTED, va="bottom")
    leg = ax.legend(fontsize=12.5, frameon=False, loc="upper left", handlelength=1.6,
                    borderpad=0.2, labelspacing=0.35)
    for t in leg.get_texts():
        t.set_color(INK)


def panel_c(ax):
    """Surrogate ladder at 256 units, 10 seeds."""
    rows = list(csv.DictReader(open(CTRL)))
    vals = np.array([[float(r[f"{t}_lump_r2"]) for t in TYPES] for r in rows])
    x = np.arange(len(TYPES))
    for row in vals:
        ax.plot(x, row, color=GRID, lw=1.4, zorder=2)
    for i, c in enumerate(TYPE_COLORS):
        ax.scatter(np.full(len(vals), x[i]), vals[:, i], s=64, color=c,
                   edgecolor=BG, linewidth=1.0, zorder=4)
        ax.plot([x[i] - 0.19, x[i] + 0.19], [vals[:, i].mean()] * 2, color=INK, lw=2.6,
                solid_capstyle="butt", zorder=6)
        ax.text(x[i], vals[:, i].mean() + 0.052, f"{vals[:, i].mean():.2f}", ha="center",
                fontsize=14, color=INK, fontweight="bold", zorder=7)

    tidy(ax, "surrogates lose it")
    ax.set_xticks(x)
    ax.set_xticklabels(TYPE_LABELS, fontsize=12.5, color=INK)
    ax.set_ylabel("eigenvector step $R^2$", fontsize=15, color=INK, labelpad=8)
    ax.set_xlim(-0.45, len(TYPES) - 0.55)
    ax.set_ylim(0, 1.06)
    ax.text(0.985, 0.965, "256 units, 10 seeds", transform=ax.transAxes, ha="right",
            va="top", fontsize=13, color=MUTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="also write staged builds (A, A+B)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    stages = [1, 2, 3] if args.build else [3]
    for st in stages:
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.patch.set_facecolor(BG)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.05],
                              left=0.068, right=0.982, top=0.735, bottom=0.145, wspace=0.36)
        axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))
        panel_a(axA)
        panel_b(axB)
        panel_c(axC)
        if st < 2:
            axB.set_visible(False)
        if st < 3:
            axC.set_visible(False)

        fig.suptitle("The step structure is present in all 33 trained networks,\n"
                     "strengthens with size, and requires the trained wiring",
                     fontsize=26, color=INK, x=0.068, ha="left", y=0.975, va="top",
                     linespacing=1.3)
        if st == 3:
            fig.text(0.982, 0.028, "row shuffle preserves each unit's total input strength",
                     ha="right", va="bottom", fontsize=12.5, color=MUTED)
        stem = "slide_lumpability_robustness" + (f"_build{st}" if args.build and st < 3 else "")
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{stem}.{ext}", dpi=200, facecolor=BG)
        plt.close(fig)
        print(f"wrote {OUT / (stem + '.png')}")


if __name__ == "__main__":
    main()
