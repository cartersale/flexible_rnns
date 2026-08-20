"""Lumpability figure: a leading eigenvector of W is piecewise-constant on cells.

Left: eigenvector component per unit, units ordered & colored by structural cell
      -> step function (flat within each cell). Right: the SAME eigenvector under
a random same-size grouping -> jagged, no steps. R^2 of the step approximation
annotated on each.
"""

from __future__ import annotations

from flexsym import paths

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from flexsym.structural.signed_input_wl_structural_roles import apply_matrix_variant, approx_wl_signed, load_w_rec  # noqa: E402

REPO = paths.REPO_ROOT
ACT_ROOT = paths.RESULTS_ROOT / "activity_clusters_paper"
ACT_NAME = "epoch_variance_unit_clusters_paper_max.npz"
ORIGINAL = paths.ORIGINAL_STORE
OUT = paths.FIGURES_ROOT; OUT.mkdir(parents=True, exist_ok=True)

INK = "#0b0b0b"; MUTED = "#52514e"; GRID = "#e6e5e2"
MODEL = "LeakyRNN_softplus_diag_256_all_1"
K = 51
VARIANT = "transpose"   # matrix the partition is equitable for; W-raw looks similar
# 8-hue validated categorical order, cycled across cells
HUES = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]


def step_r2(v, labels):
    """R^2 of approximating v by its per-group mean (1 = perfect step function)."""
    pred = np.zeros_like(v)
    for c in np.unique(labels):
        pred[labels == c] = v[labels == c].mean()
    ss_res = np.sum((v - pred) ** 2)
    ss_tot = np.sum((v - v.mean()) ** 2)
    return 1 - ss_res / ss_tot, pred


def pick_eigenvector(M, labels, topn=10):
    w, V = np.linalg.eig(M)
    order = np.argsort(-np.abs(w))[:topn]
    best, best_r2 = None, -np.inf
    for i in order:
        v = np.real(V[:, i])
        if np.linalg.norm(v) == 0:
            continue
        v = v / np.linalg.norm(v)
        r2, _ = step_r2(v, labels)
        if r2 > best_r2:
            best_r2, best = r2, v
    return best


def ordered_plot(ax, v, labels, title):
    r2, pred = step_r2(v, labels)
    # order cells by their mean value; units within a cell kept together
    cells = list(np.unique(labels))
    cell_mean = {c: v[labels == c].mean() for c in cells}
    cells_sorted = sorted(cells, key=lambda c: cell_mean[c])
    order, boundaries, ci = [], [], 0
    for c in cells_sorted:
        idx = np.where(labels == c)[0]
        order += list(idx)
        ci += len(idx)
        boundaries.append(ci)
    order = np.array(order)
    x = np.arange(len(order))
    colors = {c: HUES[i % len(HUES)] for i, c in enumerate(cells_sorted)}
    ax.bar(x, v[order], width=1.0, color=[colors[labels[u]] for u in order], zorder=3)
    # per-cell mean as a dark step overlay
    start = 0
    for b, c in zip(boundaries, cells_sorted):
        ax.plot([start - 0.5, b - 0.5], [cell_mean[c], cell_mean[c]], color=INK, linewidth=1.8, zorder=5)
        start = b
    for b in boundaries[:-1]:
        ax.axvline(b - 0.5, color="white", linewidth=0.6, zorder=4)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=2)
    ax.set_title(title, fontsize=11.5, color=INK, pad=8)
    ax.set_xlabel("units (ordered by group)", fontsize=10.5, color=INK)
    ax.text(0.03, 0.96, f"step R² = {r2:.2f}", transform=ax.transAxes, va="top", fontsize=11,
            color=INK, fontweight="bold", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.set_xlim(-0.5, len(order) - 0.5)


def main():
    md = ORIGINAL / MODEL
    ind = np.asarray(np.load(ACT_ROOT / MODEL / ACT_NAME, allow_pickle=True)["ind_active"], dtype=np.int64)
    M = apply_matrix_variant(load_w_rec(md), VARIANT)[np.ix_(ind, ind)]
    labels, _ = approx_wl_signed(np.maximum(M, 0), np.maximum(-M, 0), K, n_iter=20, seed=0, n_init=20)
    v = pick_eigenvector(M, labels)

    rng = np.random.default_rng(0)
    rand_labels = labels.copy(); rng.shuffle(rand_labels)  # same sizes, shuffled assignment

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.79, bottom=0.15, wspace=0.08)
    ordered_plot(axL, v, labels, "ordered by structural cell  →  step function")
    ordered_plot(axR, v, rand_labels, "same eigenvector, random grouping  →  no steps")
    axL.set_ylabel("eigenvector component", fontsize=10.5, color=INK)
    for ax in (axL, axR):
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)

    fig.suptitle("A leading eigenvector of W is flat within each connectivity group\n"
                 f"(the invariant-subspace signature of lumpability — {MODEL.split('_')[3]}-unit net)",
                 fontsize=12.5, color=INK, y=0.995)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"lumpability_eigenvector.{ext}", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    r2r, _ = step_r2(v, labels); r2n, _ = step_r2(v, rand_labels)
    print(f"wrote {OUT/'lumpability_eigenvector.png'}  real R²={r2r:.3f}  random R²={r2n:.3f}")


if __name__ == "__main__":
    main()
