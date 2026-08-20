"""Fig 3 (compositional reorganization) panels -> results/figures/fig3/*.svg"""
from __future__ import annotations

from flexsym import paths
import sys, csv, glob, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from flexsym.figures.figstyle import INK, MUTED, GRID, BLUE, YELLOW, GREEN, clean, save

REPO = paths.REPO_ROOT
BASE = paths.RESULTS_ROOT / "signed_input_wl_paper"
OUT = paths.FIGURES_ROOT / "fig3"


def main():
    # ---- (b) within-task vs cross-task ARI, paired per seed ----
    within, cross = [], []
    for seed in range(10):
        m = f"LeakyRNN_softplus_diag_256_all_{seed}"
        d = glob.glob(f"{BASE}/{m}/within_task_split_half_null_transpose_k_*/summary.json")
        if not d:
            continue
        import json
        s = json.load(open(d[0]))
        within.append(s["mean_split_half_ari"]); cross.append(s["mean_cross_task_ari"])
    within, cross = np.array(within), np.array(cross)
    fig, ax = plt.subplots(figsize=(3.4, 3.9)); fig.subplots_adjust(left=0.14, bottom=0.13, top=0.9, right=0.96)
    for w, c in zip(within, cross):
        ax.plot([0, 1], [w, c], color="#9a9a97", lw=1.0, alpha=0.6, zorder=2)
    ax.scatter([0]*len(within), within, s=42, color=BLUE, edgecolor="white", linewidth=0.8, zorder=4)
    ax.scatter([1]*len(cross), cross, s=42, color=YELLOW, edgecolor="white", linewidth=0.8, zorder=4)
    ax.scatter([0], [within.mean()], s=170, marker="D", color=BLUE, edgecolor="white", lw=1.2, zorder=6)
    ax.scatter([1], [cross.mean()], s=170, marker="D", color=YELLOW, edgecolor="white", lw=1.2, zorder=6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["within-task\n(same task,\nsplit trials)", "cross-task\n(different\ntasks)"], fontsize=9, color=INK)
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(0, 1.0)
    ax.set_ylabel("sub-partition similarity (ARI)", fontsize=10, color=INK)
    ax.set_title("Reliable within a task,\nreorganized across tasks", fontsize=10.5, color=INK, pad=8)
    ax.text(0.5, 0.04, "gap +0.43, 10/10, $P$<0.001", transform=ax.transAxes, ha="center", fontsize=8.3, color=MUTED)
    clean(ax)
    save(fig, OUT / "panelB_within_vs_cross.svg")

    # ---- (c) subpopulation repertoire ----
    rows = [r for r in csv.DictReader(open(BASE / "partition_count_distributions_by_size/partition_count_distributions_by_size.csv"))
            if r["measure"] == "functional_k_pooled" and int(r["network_size"]) == 256]
    ks = [int(r["k"]) for r in rows]; frac = [100*float(r["fraction"]) for r in rows]
    fig, ax = plt.subplots(figsize=(3.2, 3.4)); fig.subplots_adjust(left=0.16, bottom=0.15, top=0.9, right=0.96)
    ax.bar(ks, frac, width=0.7, color=BLUE, edgecolor="white", linewidth=1.0, zorder=3)
    for k, f in zip(ks, frac):
        ax.text(k, f+1.5, f"{f:.0f}%", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(ks); ax.set_xlabel("functional sub-populations per cell", fontsize=10, color=INK)
    ax.set_ylabel("% of cell$\\times$task subpartitions", fontsize=9.5, color=INK)
    ax.set_title("A small, discrete repertoire", fontsize=11, color=INK, pad=8)
    ax.set_ylim(0, 100)
    clean(ax)
    save(fig, OUT / "panelC_subpopulation_repertoire.svg")

    # ---- (d) shared vs non-shared component ----
    sc = list(csv.DictReader(open(BASE / "shared_component_reorg/shared_component_reorg.csv")))
    sh = np.array([float(r["shared"]) for r in sc]); no = np.array([float(r["nonshared"]) for r in sc])
    fig, ax = plt.subplots(figsize=(3.4, 3.9)); fig.subplots_adjust(left=0.14, bottom=0.13, top=0.9, right=0.96)
    for s, n in zip(sh, no):
        ax.plot([0, 1], [s, n], color="#9a9a97", lw=1.0, alpha=0.6, zorder=2)
    ax.scatter([0]*len(sh), sh, s=42, color=GREEN, edgecolor="white", linewidth=0.8, zorder=4)
    ax.scatter([1]*len(no), no, s=42, color="#9a9a97", edgecolor="white", linewidth=0.8, zorder=4)
    ax.scatter([0], [sh.mean()], s=170, marker="D", color=GREEN, edgecolor="white", lw=1.2, zorder=6)
    ax.scatter([1], [no.mean()], s=170, marker="D", color="#6b6a67", edgecolor="white", lw=1.2, zorder=6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["shared\ncomponent", "different\ncomponent"], fontsize=9.5, color=INK)
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(0, 0.65)
    ax.set_ylabel("sub-grouping similarity (ARI)", fontsize=10, color=INK)
    ax.set_title("Same configuration for the\nsame computation", fontsize=10.5, color=INK, pad=8)
    ax.text(0.5, 0.04, "10/10 networks, $P$=0.002", transform=ax.transAxes, ha="center", fontsize=8.3, color=MUTED)
    clean(ax)
    save(fig, OUT / "panelD_shared_vs_nonshared.svg")


if __name__ == "__main__":
    main()
