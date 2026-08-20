"""Figure panel: connectivity->function effect by task-epoch type (P3, full battery).

Grouped bars per epoch type (stimulus / delay-memory / response) x two control
levels: partial r(F, full | internal) and the fully-controlled r(F, full |
internal + W_in + random-projection). Per-seed dots overlaid, 95% CI whiskers,
zero baseline, Wilcoxon significance vs 0 (seed as unit of replication).
"""

from __future__ import annotations

from flexsym import paths

import csv
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.stats import t as tdist
from scipy.stats import wilcoxon

REPO = paths.REPO_ROOT
DATA = paths.RESULTS_ROOT / "signed_input_wl_paper/conn_func_regression_battery"
OUT = paths.FIGURES_ROOT
OUT.mkdir(parents=True, exist_ok=True)

EPOCH_TYPE = {"stim1": "stimulus", "delay1": "delay /\nmemory", "go1": "response"}
TYPES = ["stimulus", "delay /\nmemory", "response"]
# validated categorical hues (dataviz reference palette)
C_CTRL = "#2a78d6"   # blue  = fully controlled (beyond local + input + dimensionality)
C_LOCAL = "#eda100"  # yellow = beyond local wiring only
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e2"


def stars(p: float) -> str:
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def load() -> dict:
    """{epoch_type: {'local': [per-seed means], 'ctrl': [per-seed means]}}"""
    out = {t: {"local": [], "ctrl": []} for t in TYPES}
    for f in sorted(glob.glob(str(DATA / "conn_func_regression_LeakyRNN_softplus_diag_256_all_*.csv"))):
        rows = list(csv.DictReader(open(f)))
        for t in TYPES:
            rk = [r for r in rows if EPOCH_TYPE.get(r["period"].split("/")[1], "?") == t]
            if not rk:
                continue
            out[t]["local"].append(np.nanmean([float(r["r_full_given_internal"]) for r in rk]))
            out[t]["ctrl"].append(np.nanmean([float(r["r_full_given_int_win_rand"]) for r in rk]))
    return out


def ci95(x):
    x = np.asarray(x, float)
    sem = x.std(ddof=1) / np.sqrt(x.size)
    return tdist.ppf(0.975, x.size - 1) * sem


def main():
    d = load()
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.13, right=0.97)
    x = np.arange(len(TYPES))
    w = 0.36
    rng = np.random.default_rng(0)

    for offset, key, color in [
        (-w / 2 - 0.012, "local", C_LOCAL),
        (+w / 2 + 0.012, "ctrl", C_CTRL),
    ]:
        means = [np.mean(d[t][key]) for t in TYPES]
        cis = [ci95(d[t][key]) for t in TYPES]
        ax.bar(x + offset, means, w, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        ax.errorbar(x + offset, means, yerr=cis, fmt="none", ecolor=INK, elinewidth=1.4, capsize=3, zorder=5)
        for i, t in enumerate(TYPES):
            pts = np.asarray(d[t][key], float)
            jit = (rng.random(pts.size) - 0.5) * (w * 0.5)
            ax.scatter(x[i] + offset + jit, pts, s=15, color=color, edgecolor="white",
                       linewidth=0.6, alpha=0.9, zorder=6)
        # value + significance stacked above the controlled bar's whisker
        if key == "ctrl":
            for i, t in enumerate(TYPES):
                p = wilcoxon(d[t]["ctrl"]).pvalue
                top = means[i] + cis[i]
                ax.text(x[i] + offset, top + 0.012, f"{means[i]:+.2f}", ha="center", va="bottom",
                        fontsize=8.5, color=MUTED)
                ax.text(x[i] + offset, top + 0.052, stars(p), ha="center", va="bottom",
                        fontsize=10.5, color=INK, fontweight="bold")

    ax.axhline(0, color=INK, linewidth=1.3, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(TYPES, fontsize=10.5, color=INK)
    ax.set_ylabel("partial r  (functional ~ full-context connectivity)", fontsize=10, color=INK)
    ax.set_ylim(-0.13, 0.44)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)

    fig.suptitle("Whole-network connectivity predicts within-cell function across epochs\n"
                 "— uniquely so during memory",
                 fontsize=12, color=INK, y=0.985, ha="center")
    ax.legend(handles=[Patch(facecolor=C_LOCAL, label="beyond local wiring"),
                       Patch(facecolor=C_CTRL, label="beyond local + input weights + dimensionality")],
              loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=1, frameon=False, fontsize=9,
              handlelength=1.1, handleheight=1.1)
    fig.text(0.5, 0.035, "n = 10 seeds (256-unit); dots = per-seed means, whiskers = 95% CI;  "
                         "** p<0.01, n.s. (Wilcoxon vs 0, seed as replication unit)",
             ha="center", fontsize=7.8, color=MUTED)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"epoch_type_context_effect.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="#fcfcfb")
    print(f"wrote {OUT/'epoch_type_context_effect.png'}")
    # also print the numbers behind the figure
    for t in TYPES:
        pc = wilcoxon(d[t]["ctrl"]).pvalue
        print(f"  {t.replace(chr(10),' '):14} local={np.mean(d[t]['local']):+.3f}  ctrl={np.mean(d[t]['ctrl']):+.3f}  ctrl_wilcoxon_p={pc:.4f}")


if __name__ == "__main__":
    main()
