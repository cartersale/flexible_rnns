"""Fig 1 (role structure exists & reproducible) panels -> results/figures/fig1/*.svg"""
from __future__ import annotations

from flexsym import paths
import sys, csv, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
from flexsym.figures.figstyle import INK, MUTED, GRID, BAND, BLUE, SIZE_COL, clean, save

REPO = paths.REPO_ROOT
BASE = paths.RESULTS_ROOT / "signed_input_wl_paper"
OUT = paths.FIGURES_ROOT / "fig1"
CAN = "LeakyRNN_softplus_diag_256_all_0"
SWEEP = BASE / f"{CAN}/transpose_selected_k_41/signed_input_wl_sweep.npz"


def panel_quotient():
    s = np.load(SWEEP, allow_pickle=True)
    B = np.asarray(s["B_pos_chosen"], dtype=float) - np.asarray(s["B_neg_chosen"], dtype=float)
    fig, ax = plt.subplots(figsize=(3.9, 3.5)); fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    vmax = np.percentile(np.abs(B), 98)
    im = ax.imshow(B, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), interpolation="nearest")
    ax.set_title("Signed quotient: mean connectivity\nbetween role groups", fontsize=10, color=INK, pad=8)
    ax.set_xlabel("role group", fontsize=9.5, color=INK); ax.set_ylabel("role group", fontsize=9.5, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03); cb.set_label("net recurrent weight", fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED); cb.outline.set_visible(False)
    save(fig, OUT / "panelC_quotient.svg")


def panel_equitability_z():
    """Equitability z as a function of k, mean +/- across seeds, per size (sweep, not single k)."""
    import os
    sizes = [128, 256, 512, 1024]
    fig, ax = plt.subplots(figsize=(4.2, 3.4)); fig.subplots_adjust(left=0.15, right=0.96, top=0.9, bottom=0.15)

    def zcurve(sz):
        c = {}
        for seed in range(10):
            p = BASE / f"LeakyRNN_softplus_diag_{sz}_all_{seed}/transpose/signed_input_wl_sweep.csv"
            if not p.exists():
                continue
            for r in csv.DictReader(open(p)):
                sd = float(r["gamma_null_sd"])
                if sd > 0:
                    c.setdefault(int(r["k"]), []).append((float(r["gamma_struct"])-float(r["gamma_null_mean"]))/sd)
        ks = sorted(c)
        return np.array(ks), np.array([np.mean(c[k]) for k in ks]), np.array([np.std(c[k])/np.sqrt(len(c[k])) for k in ks])

    ax.axhline(0, color=INK, lw=1.1)
    ax.axhline(-3, color=MUTED, ls="--", lw=1.0); ax.text(56, -3.6, "$z=-3$", fontsize=7, color=MUTED)
    for sz in sizes:
        ks, m, se = zcurve(sz)
        if len(ks) == 0:
            continue
        ax.plot(ks, m, "-", color=SIZE_COL[sz], lw=2.0, zorder=4, label=f"{sz}")
        ax.fill_between(ks, m-se, m+se, color=SIZE_COL[sz], alpha=0.15, zorder=2)
    ax.set_xlabel("number of role cells $k$", fontsize=10, color=INK)
    ax.set_ylabel("equitability $z$ vs chance", fontsize=9.5, color=INK)
    ax.set_title("More equitable than chance across\nall scales (128–512); reverses at 1024", fontsize=10, color=INK, pad=8)
    ax.legend(title="units", frameon=False, fontsize=8, title_fontsize=8, loc="center right")
    ax.invert_yaxis()
    clean(ax); save(fig, OUT / "panelD_equitability_z.svg")


def panel_reproducibility():
    rows = [r for r in csv.DictReader(open(BASE / "partition_stability_soft/partition_stability_all.csv"))]
    data = {"hard": [], "soft": []}
    for r in rows:
        data[r["backend"]].append((float(r["ari_mean"]), 100*float(r["gamma_cv"])))
    fig, ax = plt.subplots(figsize=(4.2, 3.6)); fig.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.14)
    ax.axhspan(0, 5, color=BAND, zorder=0); ax.axvspan(0.8, 1.0, color="#f3f1ec", zorder=0)
    ax.text(0.02, 4.6, "cost reproducible", fontsize=8.5, color=MUTED, va="top")
    ax.text(0.9, 8.6, "partition\nreproducible", fontsize=8, color=MUTED, ha="center", va="top")
    for bk, col, fill, lab in [("hard", BLUE, True, "hard"), ("soft", "#eda100", False, "soft")]:
        pts = np.array(data[bk])
        ax.scatter(pts[:, 0], pts[:, 1], s=30, facecolor=col if fill else "none", edgecolor=col, lw=1.4, alpha=0.9, zorder=4, label=lab)
        ax.scatter([pts[:, 0].mean()], [pts[:, 1].mean()], s=170, marker="X", color=col, edgecolor="white", lw=1.2, zorder=6)
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 9)
    ax.set_xlabel("restart agreement (ARI)  $\\to$ reproducible", fontsize=9.5, color=INK)
    ax.set_ylabel("cost variability ($\\gamma$ CV %)", fontsize=9.5, color=INK)
    ax.set_title("Cost reproducible, partition not", fontsize=10.5, color=INK, pad=8)
    ax.legend(loc="upper left", frameon=False, fontsize=9, bbox_to_anchor=(0.005, 0.88))
    clean(ax); save(fig, OUT / "panelE_reproducibility.svg")


def panel_contingency():
    s = np.load(SWEEP, allow_pickle=True)
    C = np.asarray(s["contingency_chosen"], dtype=float)
    # order rows/cols to emphasize block structure
    ri = np.argsort(-C.max(1)); ci = np.argsort(-C.max(0))
    C = C[np.ix_(ri, ci)]
    fig, ax = plt.subplots(figsize=(3.7, 3.4)); fig.subplots_adjust(left=0.13, right=0.9, top=0.9, bottom=0.13)
    im = ax.imshow(C, cmap="Blues", aspect="auto", interpolation="nearest")
    ax.set_xlabel("activity cluster", fontsize=9.5, color=INK); ax.set_ylabel("structural role group", fontsize=9.5, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Structure vs activity: related\nbut not identical (NMI≈0.58)", fontsize=10, color=INK, pad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03); cb.set_label("units", fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED); cb.outline.set_visible(False)
    save(fig, OUT / "panelF_contingency.svg")


if __name__ == "__main__":
    panel_quotient(); panel_equitability_z(); panel_reproducibility(); panel_contingency()
