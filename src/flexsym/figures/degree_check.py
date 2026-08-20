"""Degree-artefact check figure (reviewer #5).

Panel A: group-size distribution (256 family) — a healthy spread with some
    singletons, no single giant cell.
Panel B: singleton node-strength ratio vs singleton mass, per network (all 33) —
    singletons carry 1.3-2.6x mean strength (degree sensitivity is real) but are
    only 1-7% of units (mild; no pathological collapse).
"""

from __future__ import annotations

from flexsym import paths

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


from flexsym.structural.signed_input_wl_structural_roles import apply_matrix_variant, approx_wl_signed, labels_to_H, load_w_rec  # noqa: E402

REPO = paths.REPO_ROOT
ACT_ROOT = paths.RESULTS_ROOT / "activity_clusters_paper"
ACT_NAME = "epoch_variance_unit_clusters_paper_max.npz"
ORIGINAL = paths.ORIGINAL_STORE
STAGED = paths.STAGED_STORE
OUT = paths.FIGURES_ROOT; OUT.mkdir(parents=True, exist_ok=True)
CSV = paths.RESULTS_ROOT / "signed_input_wl_paper/partition_stability_soft/partition_stability_all.csv"

INK = "#0b0b0b"; MUTED = "#52514e"; GRID = "#e6e5e2"; BAND = "#eef4fc"
C_MAIN = "#2a78d6"; C_SINGLE = "#eb6834"   # blue cells, orange singletons
SIZE_COL = {128: "#2a78d6", 256: "#008300", 512: "#e87ba4", 1024: "#eda100"}
from flexsym.structural.operating_k import OPK as OPK256


def resolve(model):
    for s in (ORIGINAL, STAGED):
        if (s / model).exists():
            return s / model
    return None


def cell_sizes_256():
    sizes = []
    for seed, k in OPK256.items():
        model = f"LeakyRNN_softplus_diag_256_all_{seed}"
        md = resolve(model); act = ACT_ROOT / model / ACT_NAME
        if md is None or not act.exists():
            continue
        ind = np.asarray(np.load(act, allow_pickle=True)["ind_active"], dtype=np.int64)
        W = apply_matrix_variant(load_w_rec(md), "transpose")[np.ix_(ind, ind)]
        labels, _ = approx_wl_signed(np.maximum(W, 0), np.maximum(-W, 0), k, n_iter=20, seed=0, n_init=20)
        s = labels_to_H(labels, k).sum(axis=0)
        sizes += [int(v) for v in s if v > 0]
    return np.array(sizes)


def main():
    sizes = cell_sizes_256()
    rows = [r for r in csv.DictReader(open(CSV)) if r["backend"] == "hard"]
    fr = np.array([100 * float(r["frac_units_in_singletons"]) for r in rows])
    sr = np.array([float(r["singleton_strength_ratio"]) for r in rows])
    sz = np.array([int(r["size"]) for r in rows])
    maxcf = np.array([float(r["max_cell_fraction"]) for r in rows])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.15, wspace=0.26)

    # ---- Panel A: group-size distribution ----
    bins = np.arange(0.5, sizes.max() + 1.5, 1)
    counts, edges = np.histogram(sizes, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [C_SINGLE if int(round(c)) == 1 else C_MAIN for c in centers]
    axA.bar(centers, counts, width=0.9, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    axA.axvline(np.median(sizes), color=INK, linewidth=1.4, linestyle="--", zorder=4)
    axA.text(np.median(sizes) + 0.6, counts.max() * 0.92, f"median = {int(np.median(sizes))} units",
             fontsize=9.5, color=INK)
    pct_cells_single = 100 * (sizes == 1).mean()
    pct_units_single = 100 * sizes[sizes == 1].sum() / sizes.sum()
    axA.annotate(f"{pct_cells_single:.0f}% of cells,\nbut only {pct_units_single:.0f}% of units",
                 xy=(1, counts[0] * 0.5), xytext=(3.2, counts.max() * 0.72), fontsize=9, color=C_SINGLE,
                 arrowprops=dict(arrowstyle="->", color=C_SINGLE, lw=1.2))
    axA.text(0.97, 0.5,
             f"largest cell ≤ {maxcf.max()*100:.0f}% of units\n(mean {maxcf.mean()*100:.0f}%) → no giant cell",
             transform=axA.transAxes, ha="right", fontsize=9.5, color=MUTED,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))
    axA.set_xlabel("structural cell size (units)", fontsize=10.5, color=INK)
    axA.set_ylabel("number of cells", fontsize=10.5, color=INK)
    axA.set_title("Group sizes: a healthy spread", fontsize=11.5, color=INK, pad=8)
    axA.legend(handles=[Line2D([0], [0], marker="s", color="none", markerfacecolor=C_SINGLE,
                               markersize=9, label="singletons (size 1)"),
                        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_MAIN,
                               markersize=9, label="multi-unit cells")],
               loc="upper right", frameon=False, fontsize=9, bbox_to_anchor=(1.0, 0.99))

    # ---- Panel B: singleton strength ratio vs mass ----
    axB.axhline(1.0, color=INK, linewidth=1.3, zorder=3)
    axB.text(7.4, 1.02, "no strength bias", fontsize=8.5, color=MUTED, ha="right", va="bottom")
    for size in [128, 256, 512, 1024]:
        m = sz == size
        axB.scatter(fr[m], sr[m], s=42, color=SIZE_COL[size], edgecolor="white",
                    linewidth=0.8, alpha=0.9, zorder=5, label=f"{size} units")
    axB.annotate("singletons carry 1.3–2.6×\nmean strength (33/33 nets)",
                 xy=(3.6, 2.0), xytext=(0.2, 2.6), fontsize=9, color=INK,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    axB.axvspan(0, 8, color=BAND, zorder=0)
    axB.text(0.2, 1.3, "…but only 1–7% of units\n(mild — no collapse)", fontsize=9, color=MUTED, va="top")
    axB.set_xlim(0, 8)
    axB.set_ylim(0.9, 2.75)
    axB.set_xlabel("units in singleton cells (% of active units)", fontsize=10.5, color=INK)
    axB.set_ylabel("singleton node-strength ratio\n(vs mean unit)", fontsize=10.5, color=INK)
    axB.set_title("Singletons carry more weight — but are few", fontsize=11.5, color=INK, pad=8)
    axB.legend(loc="upper right", frameon=False, fontsize=8.5, title="network size",
               title_fontsize=8.5)

    for ax in (axA, axB):
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED)

    fig.suptitle("The degree artefact is present but mild — no pathological giant-cell + singletons collapse",
                 fontsize=12, color=INK, y=0.965)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"degree_check.{ext}", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"wrote {OUT/'degree_check.png'}  (n cells={len(sizes)}, median={np.median(sizes):.0f})")


if __name__ == "__main__":
    main()
