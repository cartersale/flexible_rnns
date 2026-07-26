"""Slide panel: a leading eigenvector of W_rec is a step function on the connectivity groups.

Presentation version of plot_lumpability_eigenvector.py -- 16:9, large type, stacked
strips sharing a y-axis so the flat-vs-jagged contrast reads at projection scale.

Top strip:    units ordered by signed-WL group  -> plateaus, high step R^2.
Bottom strip: same eigenvector, size-matched random grouping -> no steps, low R^2.

Optional staged builds for slide animation (--build):
  0 raw eigenvector, unsorted, grey
  1 sorted and coloured by group
  2 group-mean step overlaid
  3 random-grouping strip revealed  (== the static figure)
"""

from __future__ import annotations

from flexsym import paths

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from flexsym.structural.signed_input_wl_structural_roles import apply_matrix_variant, approx_wl_signed, load_w_rec  # noqa: E402

REPO = paths.REPO_ROOT
ACT_ROOT = REPO / "results/activity_clusters_paper"
ACT_NAME = "epoch_variance_unit_clusters_paper_max.npz"
ORIGINAL = paths.ORIGINAL_STORE
OUT = REPO / "results/figures/slides"

BG = "#ffffff"
INK = "#111111"
MUTED = "#5b5a57"
GRID = "#e3e2df"
GREY = "#b9b8b4"
VARIANT = "transpose"
# 8-hue validated categorical order, cycled across groups
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
    """Leading eigenvector (by |lambda|) that is most piecewise-constant on the groups."""
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


def group_order(v, labels):
    """Unit order grouping units by label, groups sorted by their mean value."""
    groups = sorted(np.unique(labels), key=lambda c: v[labels == c].mean())
    order, boundaries, cursor = [], [], 0
    for c in groups:
        idx = np.where(labels == c)[0]
        order += list(idx)
        cursor += len(idx)
        boundaries.append(cursor)
    return np.array(order), boundaries, groups


def strip(ax, v, labels, title, stage=3, show_steps=True):
    r2, _ = step_r2(v, labels)
    order, boundaries, groups = group_order(v, labels)

    if stage == 0:  # unsorted, uncoloured
        x = np.arange(len(v))
        ax.bar(x, v, width=1.0, color=GREY, zorder=3)
    else:
        x = np.arange(len(order))
        colors = {c: HUES[i % len(HUES)] for i, c in enumerate(groups)}
        ax.bar(x, v[order], width=1.0, color=[colors[labels[u]] for u in order], zorder=3)
        for b in boundaries[:-1]:
            ax.axvline(b - 0.5, color=BG, linewidth=0.9, zorder=4)

    if stage >= 2 and show_steps:
        start = 0
        for b, c in zip(boundaries, groups):
            m = v[labels == c].mean()
            ax.plot([start - 0.5, b - 0.5], [m, m], color=INK, linewidth=3.0,
                    solid_capstyle="butt", zorder=6)
            start = b

    ax.axhline(0, color=MUTED, linewidth=1.0, zorder=2)
    ax.set_title(title, fontsize=21, color=INK, pad=12, loc="left")
    if stage >= 2 and show_steps:
        ax.text(0.985, 0.93, f"step $R^2$ = {r2:.2f}", transform=ax.transAxes,
                va="top", ha="right", fontsize=26, color=INK, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc=BG, ec=GRID, lw=1.4), zorder=8)
    ax.set_xlim(-0.5, len(v) - 0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(1.2)
    ax.tick_params(colors=MUTED, labelsize=14, width=1.2, length=5)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    return r2


def build_figure(v, labels, rand_labels, model, k, stage=3):
    fig, (axT, axB) = plt.subplots(2, 1, figsize=(13.33, 7.5), sharey=True)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.115, right=0.975, top=0.755, bottom=0.105, hspace=0.46)

    t_top = ("units unordered" if stage == 0 else
             "units ordered by connectivity group" if stage == 1 else
             "units ordered by connectivity group  →  flat within each group")
    r2r = strip(axT, v, labels, t_top, stage=stage)
    r2n = strip(axB, v, rand_labels,
                "same eigenvector, size-matched random grouping  →  no steps",
                stage=3, show_steps=True)
    if stage < 3:
        axB.set_visible(False)

    axB.set_xlabel("units", fontsize=18, color=INK, labelpad=8)
    for ax in (axT, axB):
        ax.set_ylabel("eigenvector\ncomponent", fontsize=17, color=INK, labelpad=10)

    fig.suptitle("A leading eigenvector of $W_{\\rm rec}$ is piecewise-constant\n"
                 "on the connectivity groups",
                 fontsize=28, color=INK, x=0.115, ha="left", y=0.975,
                 va="top", linespacing=1.3)
    n = model.split("_")[3]
    fig.text(0.975, 0.022, f"{n}-unit network,  k = {k} groups", fontsize=14, color=MUTED,
             ha="right", va="bottom")

    # headroom so the R^2 badges never sit on top of a bar
    lo, hi = axT.get_ylim()
    axT.set_ylim(lo, hi + 0.42 * (hi - lo))
    return fig, r2r, r2n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LeakyRNN_softplus_diag_256_all_0")
    ap.add_argument("--k", type=int, default=41)
    ap.add_argument("--seed", type=int, default=0, help="seed for WL init and the random grouping")
    ap.add_argument("--build", action="store_true", help="also write staged builds 0-2")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    md = ORIGINAL / args.model
    ind = np.asarray(np.load(ACT_ROOT / args.model / ACT_NAME, allow_pickle=True)["ind_active"],
                     dtype=np.int64)
    M = apply_matrix_variant(load_w_rec(md), VARIANT)[np.ix_(ind, ind)]
    labels, _ = approx_wl_signed(np.maximum(M, 0), np.maximum(-M, 0), args.k,
                                 n_iter=20, seed=args.seed, n_init=20)
    v = pick_eigenvector(M, labels)

    rng = np.random.default_rng(args.seed)
    rand_labels = labels.copy()
    rng.shuffle(rand_labels)  # same group sizes, shuffled assignment

    stages = [0, 1, 2, 3] if args.build else [3]
    for st in stages:
        fig, r2r, r2n = build_figure(v, labels, rand_labels, args.model, args.k, stage=st)
        stem = "slide_lumpability_eigenvector" + (f"_build{st}" if args.build and st < 3 else "")
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"{stem}.{ext}", dpi=200, facecolor=BG)
        plt.close(fig)
        print(f"wrote {OUT / (stem + '.png')}")
    print(f"n_active={len(ind)}  groups={len(np.unique(labels))}  R2_groups={r2r:.3f}  R2_random={r2n:.3f}")


if __name__ == "__main__":
    main()
