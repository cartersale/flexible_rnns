"""Fig 5 (causal lesions) panels -> results/figures/fig5/*.svg"""
from __future__ import annotations

from flexsym import paths
import sys, csv, glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from flexsym.figures.figstyle import INK, MUTED, GRID, BLUE, GRAY, clean, save

REPO = paths.REPO_ROOT
LES = paths.RESULTS_ROOT / "signed_input_wl_paper/lesion_causal"
OUT = paths.FIGURES_ROOT / "fig5"


def diag_scatter(ax, x, y, xylabel, title, p):
    lim = [min(x.min(), y.min()) - 0.03, max(x.max(), y.max()) + 0.03]
    ax.plot(lim, lim, "--", color=MUTED, lw=1.1, zorder=1)
    ax.scatter(x, y, s=48, color=BLUE, edgecolor="white", linewidth=1.0, zorder=4)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel(f"random groups", fontsize=10, color=INK)
    ax.set_ylabel(f"structural cells", fontsize=10, color=INK)
    ax.set_title(title, fontsize=11, color=INK, pad=8)
    ax.text(0.04, 0.96, f"{int((y>x).sum())}/10 above line\n$P$={p}", transform=ax.transAxes,
            va="top", fontsize=9, color=INK)
    clean(ax, grid="both") if False else clean(ax)
    ax.xaxis.grid(True, color=GRID, lw=0.8)


def main():
    rows = list(csv.DictReader(open(LES / "lesion_causal_summary.csv")))
    al_s = np.array([float(r["align_structural"]) for r in rows]); al_r = np.array([float(r["align_random"]) for r in rows])
    md_s = np.array([float(r["maxdrop_structural"]) for r in rows]); md_r = np.array([float(r["maxdrop_random"]) for r in rows])

    # (b) alignment scatter
    fig, ax = plt.subplots(figsize=(3.6, 3.6)); fig.subplots_adjust(left=0.18, bottom=0.16, top=0.86, right=0.96)
    diag_scatter(ax, al_r, al_s, "alignment", "Deficit follows activity\n(“break what you compute”)", "0.002")
    save(fig, OUT / "panelB_alignment.svg")

    # (c) max-drop scatter
    fig, ax = plt.subplots(figsize=(3.6, 3.6)); fig.subplots_adjust(left=0.18, bottom=0.16, top=0.86, right=0.96)
    diag_scatter(ax, md_r, md_s, "max deficit", "Larger task-specific deficit", "0.002")
    save(fig, OUT / "panelC_maxdrop.svg")

    # (a) deficit heatmap (canonical seed 0)
    det = np.load(LES / "lesion_detail_LeakyRNN_softplus_diag_256_all_0.npz", allow_pickle=True)
    deficit = np.asarray(det["struct_deficit"], dtype=float)  # [cell, task]
    rules = [str(x) for x in det["rules"]]
    order = np.argsort(-deficit.max(1))  # cells by peak deficit
    D = deficit[order]
    fig, ax = plt.subplots(figsize=(5.2, 4.2)); fig.subplots_adjust(left=0.22, bottom=0.16, top=0.90, right=0.98)
    vmax = np.percentile(np.abs(D), 98)
    im = ax.imshow(D.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(rules))); ax.set_yticklabels(rules, fontsize=6.5, color=INK)
    ax.set_xlabel("structural cell (ranked by peak deficit)", fontsize=9.5, color=INK)
    ax.set_title("Per-task performance drop from lesioning each cell", fontsize=10, color=INK, pad=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02); cb.set_label("performance drop", fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED); cb.outline.set_visible(False)
    save(fig, OUT / "panelA_deficit_heatmap.svg")

    # (d) memory-motif: per-seed activity-memory partial correlation
    mm = list(csv.DictReader(open(LES / "memory_motif_v2.csv")))
    act = np.array([float(r["act_r"]) for r in mm]); seeds = [int(r["seed"]) for r in mm]
    fig, ax = plt.subplots(figsize=(3.8, 3.6)); fig.subplots_adjust(left=0.17, bottom=0.16, top=0.86, right=0.96)
    rng = np.random.default_rng(0); jit = (rng.random(len(act)) - 0.5) * 0.3
    ax.axhline(0, color=INK, lw=1.2)
    ax.scatter(jit, act, s=44, color=BLUE, edgecolor="white", linewidth=1.0, zorder=4)
    ax.scatter([0], [act.mean()], s=180, marker="D", color=INK, edgecolor="white", linewidth=1.2, zorder=6)
    ax.text(0.16, act.mean(), f"mean {act.mean():.2f}", fontsize=9, color=INK, va="center")
    ax.set_xlim(-0.5, 0.7); ax.set_xticks([])
    ax.set_ylim(-0.05, 0.85)
    ax.set_ylabel("memory-cell $\\to$ memory-deficit\npartial $r$ (per network)", fontsize=9.5, color=INK)
    ax.set_title("Lesioning memory cells\nselectively breaks memory", fontsize=10.5, color=INK, pad=8)
    ax.text(0.5, 0.03, "10/10 networks", transform=ax.transAxes, ha="center", fontsize=8.5, color=MUTED)
    clean(ax)
    save(fig, OUT / "panelD_memory_motif.svg")


if __name__ == "__main__":
    main()
