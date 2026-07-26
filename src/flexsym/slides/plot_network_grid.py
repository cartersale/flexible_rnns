"""Slide figure: the network dataset as a size x seed grid, shaded by performance.

Rows = network size, cols = seed. Each cell shaded by final worst-task accuracy
(perf_min). Original seeds 0-2 vs new seeds 3-9 separated; 1024 new seeds shown
as in-training (not analyzed).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

DATA = Path("/private/tmp/claude-501/-Users-cartersale-Research-projects-w-rec-rnns-flexible-multitask/4e7bcf54-e1b0-4216-b3a3-e0f15d75d62e/scratchpad/net_perf.json")
OUT = Path("/Users/cartersale/Research/projects/w_rec_rnns/flexible_multitask/results/figures")
OUT.mkdir(parents=True, exist_ok=True)

# validated blue sequential ramp (dataviz reference palette), light -> dark
BLUE = LinearSegmentedColormap.from_list(
    "brand_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
INK = "#0b0b0b"
MUTED = "#52514e"
GRAY = "#d8d7d3"
SIZES = [128, 256, 512, 1024]
SEEDS = list(range(10))
METRIC = "perf_avg"
VMIN, VMAX = 0.90, 1.0


def main():
    recs = json.load(open(DATA))
    perf = {(r["size"], r["seed"]): r for r in recs}
    norm = Normalize(VMIN, VMAX)

    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    fig.subplots_adjust(left=0.11, right=0.88, top=0.78, bottom=0.14)
    gap = 0.06

    for ri, sz in enumerate(SIZES):
        y = len(SIZES) - 1 - ri  # 128 at top
        for ci, sd in enumerate(SEEDS):
            r = perf.get((sz, sd))
            x = ci
            missing = (r is None) or (r[METRIC] is None)
            if missing:
                ax.add_patch(Rectangle((x + gap, y + gap), 1 - 2 * gap, 1 - 2 * gap,
                                       facecolor=GRAY, edgecolor="white", linewidth=1.4, hatch="////"))
            else:
                p = r[METRIC]
                col = BLUE(norm(p))
                ax.add_patch(Rectangle((x + gap, y + gap), 1 - 2 * gap, 1 - 2 * gap,
                                       facecolor=col, edgecolor="white", linewidth=1.4))
                txtcol = "white" if norm(p) > 0.55 else INK
                ax.text(x + 0.5, y + 0.5, f"{p:.2f}", ha="center", va="center",
                        fontsize=8.5, color=txtcol)

    # divider between originals (0-2) and new (3-9)
    ax.plot([3, 3], [0, 4], color=INK, linewidth=1.6, zorder=5)
    ax.text(1.5, 4.28, "originals", ha="center", va="bottom", fontsize=9.5, color=MUTED)
    ax.text(6.5, 4.28, "new (this work)", ha="center", va="bottom", fontsize=9.5, color=MUTED)

    ax.set_xlim(-0.02, 10.02)
    ax.set_ylim(-0.02, 4.02)
    ax.set_xticks([c + 0.5 for c in range(10)])
    ax.set_xticklabels(SEEDS, fontsize=9.5, color=INK)
    ax.set_yticks([len(SIZES) - 1 - i + 0.5 for i in range(len(SIZES))])
    ax.set_yticklabels([f"{s}" for s in SIZES], fontsize=10, color=INK)
    ax.set_xlabel("seed", fontsize=10.5, color=INK)
    ax.set_ylabel("network size (units)", fontsize=10.5, color=INK)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    fig.suptitle("33 networks analyzed: 4 sizes × up to 10 seeds  (seed = unit of replication)",
                 fontsize=12, color=INK, y=0.96)

    # colorbar
    cax = fig.add_axes([0.90, 0.14, 0.022, 0.64])
    import matplotlib.cm as cm
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=BLUE), cax=cax)
    cb.set_label("final average accuracy across tasks (perf_avg)", fontsize=8.8, color=MUTED)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    cb.outline.set_visible(False)

    # legend note for hatched
    ax.add_patch(Rectangle((0.0, -0.95), 0.34, 0.34, facecolor=GRAY, edgecolor="white",
                           hatch="////", clip_on=False))
    ax.text(0.42, -0.78, "1024 new seeds: still training (not analyzed)",
            fontsize=8, color=MUTED, va="center")

    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"network_grid.{ext}", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"wrote {OUT/'network_grid.png'}")


if __name__ == "__main__":
    main()
