"""Two method-fragility figures.

(1) degeneracy_landscape.png  — schematic: a cost landscape with many equal-depth
    valleys = many different groupings at nearly the same cost.
(2) restart_stability_scatter.png — data: per-network restart-ARI (partition
    reproducibility) vs gamma CV (cost reproducibility), hard vs soft back-end.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/figures"; OUT.mkdir(parents=True, exist_ok=True)
CSV = REPO / "results/signed_input_wl_paper/partition_stability_soft/partition_stability_all.csv"

C_HARD = "#2a78d6"   # blue
C_SOFT = "#eda100"   # yellow
C_G3 = ["#2a78d6", "#eda100", "#008300"]
INK = "#0b0b0b"; MUTED = "#52514e"; GRID = "#e6e5e2"; BAND = "#eef4fc"


# --------------------------------------------------------------------------- #
def landscape():
    x = np.linspace(0.35, 5.65, 1200)
    # 5 valleys at integer x, near-equal depth + tiny variation
    y = 0.42 * (1 - np.cos(2 * np.pi * x)) + 0.02 * np.sin(1.7 * x + 0.5)
    valleys_x = np.array([1, 2, 3, 4, 5])
    valleys_y = 0.02 * np.sin(1.7 * valleys_x + 0.5)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.66, bottom=0.12)
    # equal-cost band
    lo, hi = valleys_y.min(), valleys_y.max()
    ax.axhspan(lo - 0.012, hi + 0.012, color=BAND, zorder=0)
    ax.plot(x, y, color=INK, linewidth=2.4, zorder=3)
    ax.scatter(valleys_x, valleys_y, s=70, color=C_HARD, edgecolor="white", linewidth=1.5, zorder=5)

    # equal-cost bracket on the right
    ax.annotate("", xy=(5.5, hi + 0.012), xytext=(5.5, lo - 0.012),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.4))
    ax.text(5.58, (lo + hi) / 2, "cost varies\nonly ~3%", fontsize=9.5, color=MUTED, va="center")

    # grouping glyphs above 3 valleys: 8 units colored into different groupings
    groupings = {1: [0, 0, 1, 1, 2, 2, 0, 1],
                 3: [0, 1, 0, 1, 2, 0, 2, 1],
                 5: [1, 1, 0, 2, 2, 0, 0, 1]}
    gy = 1.22
    sq = 0.085
    for vx, g in groupings.items():
        for i, lab in enumerate(g):
            ax.add_patch(Rectangle((vx - 0.36 + i * sq, gy), sq * 0.86, sq * 1.5,
                                   facecolor=C_G3[lab], edgecolor="white", linewidth=0.8,
                                   clip_on=False, zorder=6))
        # connector down to the valley
        ax.plot([vx, vx], [gy - 0.02, 0.10], color=MUTED, linewidth=0.8, linestyle=":", zorder=2)
    ax.text(3, gy + 0.17, "different groupings  (restart ARI ≈ 0.40)",
            ha="center", fontsize=10, color=INK, clip_on=False)

    ax.set_xlim(0.2, 6.15)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("space of possible groupings", fontsize=10.5, color=INK)
    ax.set_ylabel("cost  (γ, non-equitability)", fontsize=10.5, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    fig.suptitle("A near-degenerate landscape: many groupings, nearly the same cost",
                 fontsize=12.5, color=INK, y=0.99)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"degeneracy_landscape.{ext}", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"wrote {OUT/'degeneracy_landscape.png'}")


# --------------------------------------------------------------------------- #
def scatter():
    rows = list(csv.DictReader(open(CSV)))
    data = {"hard": [], "soft": []}
    for r in rows:
        data[r["backend"]].append((float(r["ari_mean"]), 100 * float(r["gamma_cv"])))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    fig.subplots_adjust(left=0.11, right=0.96, top=0.80, bottom=0.13)

    # "reproducible cost" band (low CV)
    ax.axhspan(0, 5, color=BAND, zorder=0)
    ax.text(0.02, 4.6, "cost reproducible  (CV < 5%)", fontsize=9, color=MUTED, va="top")
    # "reproducible partition" reference (high ARI) — where points are NOT
    ax.axvspan(0.8, 1.0, color="#f3f1ec", zorder=0)
    ax.text(0.895, 8.6, "partition\nreproducible", fontsize=9, color=MUTED, ha="center", va="top")

    for bk, color, fill, lab in [("hard", C_HARD, True, "hard k-means"),
                                 ("soft", C_SOFT, False, "soft / GMM")]:
        pts = np.array(data[bk])
        ax.scatter(pts[:, 0], pts[:, 1], s=34,
                   facecolor=color if fill else "none", edgecolor=color,
                   linewidth=1.6, alpha=0.9, zorder=4, label=lab)
        mx, my = pts[:, 0].mean(), pts[:, 1].mean()
        ax.scatter([mx], [my], s=220, marker="X", color=color, edgecolor="white",
                   linewidth=1.4, zorder=6)
        ax.annotate(f"mean ARI {mx:.2f}", (mx, my), xytext=(mx, my - 1.1 if bk == "hard" else my + 0.9),
                    ha="center", fontsize=8.5, color=color, fontweight="bold")

    ax.annotate("partition NOT reproducible\n(ARI ≈ 0.40, far from 1.0)",
                xy=(0.55, 2.3), xytext=(0.56, 7.4), fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3), ha="left")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 9)
    ax.set_xlabel("restart agreement  (ARI across random starts)  →  reproducible", fontsize=10, color=INK)
    ax.set_ylabel("cost variability  (γ CV %, across restarts)", fontsize=10, color=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, bbox_to_anchor=(0.005, 0.90))
    fig.suptitle("Cost is reproducible; the specific grouping is not — and soft doesn't fix it",
                 fontsize=12, color=INK, y=0.95)
    fig.text(0.5, 0.015, "each point = one network (n=33, operating k). soft sits at even lower ARI than hard.",
             ha="center", fontsize=8, color=MUTED)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"restart_stability_scatter.{ext}", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"wrote {OUT/'restart_stability_scatter.png'}")


if __name__ == "__main__":
    landscape()
    scatter()
