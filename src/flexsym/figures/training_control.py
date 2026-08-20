"""Training-control figure: is the structure created by training, or by degree?

Panel A: equitability z by matrix type — row-shuffle (keeps strength) matches
    trained → equitability is a degree artifact.
Panel B: lumpability step-R² by matrix type — trained > row-shuffle > full-shuffle
    > gaussian → lumpability survives the degree control (a real training effect),
    though degree gets most of the way.
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
CSV = paths.RESULTS_ROOT / "signed_input_wl_paper/training_control.csv"
OUT = paths.FIGURES_ROOT; OUT.mkdir(parents=True, exist_ok=True)

TYPES = ["trained", "row-shuffle", "full-shuffle", "gaussian"]
LABELS = ["trained", "row-shuffle\n(keeps strength)", "full-shuffle", "gaussian\n(untrained)"]
COLORS = ["#2a78d6", "#eda100", "#eb6834", "#9a9a97"]
INK = "#0b0b0b"; MUTED = "#52514e"; GRID = "#e6e5e2"


def main():
    rows = list(csv.DictReader(open(CSV)))

    def col(metric):
        return {t: np.array([float(r[f"{t}_{metric}"]) for r in rows]) for t in TYPES}

    eq = col("equit_z"); lu = col("lump_r2")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.6, 4.6))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.80, bottom=0.17, wspace=0.24)
    x = np.arange(len(TYPES)); rng = np.random.default_rng(0)

    def panel(ax, data, ylabel, title, invert=False):
        means = [data[t].mean() for t in TYPES]
        ax.bar(x, means, 0.62, color=COLORS, edgecolor="white", linewidth=1.2, zorder=3)
        for i, t in enumerate(TYPES):
            pts = data[t]; jit = (rng.random(pts.size) - 0.5) * 0.34
            ax.scatter(x[i] + jit, pts, s=15, color="white", edgecolor=INK, linewidth=0.6,
                       alpha=0.85, zorder=6)
            ax.text(i, means[i] + (0.02 if not invert else -0.4) * (1 if means[i] >= 0 else -1),
                    f"{means[i]:.2f}" if not invert else f"{means[i]:.1f}",
                    ha="center", va="bottom" if means[i] >= 0 else "top", fontsize=9, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=9, color=INK)
        ax.set_ylabel(ylabel, fontsize=10.5, color=INK)
        ax.set_title(title, fontsize=11.5, color=INK, pad=8)
        ax.axhline(0, color=MUTED, linewidth=1.0, zorder=2)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED)

    panel(axA, eq, "equitability z  (more negative = more equitable)",
          "Equitability ≈ degree artefact", invert=True)
    axA.annotate("row-shuffle ≈ trained\n→ driven by unit strength,\n    not learned wiring",
                 xy=(1.2, eq["row-shuffle"].mean() + 0.3), xytext=(2.55, -12.5),
                 fontsize=9, color=INK, ha="right",
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))

    panel(axB, lu, "lumpability  (best-eigenvector step-R²)",
          "Lumpability survives the degree control")
    axB.text(2.5, 0.78, "trained > all controls\n→ a real training effect\n(but degree gets\n    most of the way)",
             ha="right", va="top", fontsize=9, color=INK)
    axB.set_ylim(0, 1.05)

    fig.suptitle("Did training create the structure?  Equitability: mostly no (degree).  "
                 "Lumpability: yes, modestly.\n(256-unit, 10 seeds; identical WL pipeline on each matrix)",
                 fontsize=11.5, color=INK, y=0.99)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"training_control.{ext}", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"wrote {OUT/'training_control.png'}")


if __name__ == "__main__":
    main()
