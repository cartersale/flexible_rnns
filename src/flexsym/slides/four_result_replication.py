"""The four-result arc, replicated across all ten 256-unit networks (released 0-2 + the
l1-refixed 3-9). One panel per result, each network a point, released vs fixed distinguished.

  A  lumpable coordinates      lumpability score (-eigenvector-leakage z) per net
  B  quotient reduced model    A_cond of the quotient's motif prediction vs its random-quotient null
  C  division of labour        response-epoch readout potency: memory ring vs matched non-ring plane
  D  structural bottleneck     motif energy: participation ratio over quotient modes, conn vs random

Data (all seeds 0-9, 256-unit):
  results/signed_input_wl_paper/lumpability/lumpability.csv
  results/motif_fixed_points/quotient_prediction.json
  results/motif_fixed_points/rotation_readout_control.csv
  results/motif_fixed_points/quotient_mode_concentration.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flexsym import paths

RES = paths.RESULTS_ROOT
FP = RES / "motif_fixed_points"
OUT = paths.FIGURES_ROOT / "slides"

BG = "#ffffff"
INK = "#111111"
MUTED = "#5b5a57"
GRID = "#e3e2df"
BLUE = "#2a78d6"
DARK = "#0f3f73"
GREY = "#9a9a97"
RED = "#e34948"
GREEN = "#008300"
K = "20"
SEEDS = list(range(10))


def tidy(ax, title, fs=15.5):
    ax.set_title(title, fontsize=fs, color=INK, pad=9, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(1.1)
    ax.tick_params(colors=MUTED, labelsize=11.5, width=1.1, length=4)
    ax.set_axisbelow(True)


def net(s):
    return f"LeakyRNN_softplus_diag_256_all_{s}"


def load():
    lump = {int(r["seed"]): -float(r["transpose_eig_leakage_z"])
            for r in csv.DictReader(open(RES / "signed_input_wl_paper/lumpability/lumpability.csv"))
            if r["size"] == "256"}
    qp = json.load(open(FP / "quotient_prediction.json"))
    qcond = {s: qp[net(s)]["per_k"][K]["pred_m3"] for s in SEEDS if net(s) in qp}
    rot = {}
    rows = list(csv.DictReader(open(FP / "rotation_readout_control.csv")))
    for s in SEEDS:
        r = [x for x in rows if x["net"].endswith(f"_{s}")]
        if r:
            rot[s] = (float(np.median([float(x["ring_resp_potency"]) for x in r])),
                      float(np.median([float(x["match_resp_potency"]) for x in r])))
    mc = json.load(open(FP / "quotient_mode_concentration.json"))
    conc = {s: mc[net(s)][K] for s in SEEDS if net(s) in mc}
    return lump, qcond, rot, conc


def split(xs, vals):
    """Return (released x/y, fixed x/y) for coloring seeds 0-2 vs 3-9."""
    rel = [(x, v) for x, v in zip(xs, vals) if x < 3]
    fix = [(x, v) for x, v in zip(xs, vals) if x >= 3]
    return rel, fix


def panel_bars(ax, data, title, ylabel, chance=0.0, chance_label="chance"):
    xs = sorted(data)
    vals = [data[s] for s in xs]
    colors = [DARK if s < 3 else BLUE for s in xs]
    ax.bar(range(len(xs)), vals, color=colors, edgecolor=BG, linewidth=0.8, zorder=4)
    if chance is not None:
        ax.axhline(chance, color=INK, lw=1.3, zorder=3)
    tidy(ax, title)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, fontsize=11)
    ax.set_xlabel("network (seed)", fontsize=12.5, color=INK, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=12.5, color=INK, labelpad=5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9)


def panel_paired(ax, pairs, title, ylabel, c_hi=BLUE, c_lo=GREY,
                 lab_hi="connectivity", lab_lo="random"):
    """pairs[s] = (high, low); connectivity vs its null, per seed, paired."""
    xs = sorted(pairs)
    for i, s in enumerate(xs):
        hi, lo = pairs[s]
        ax.plot([i, i], [lo, hi], color=GRID, lw=2.2, zorder=2)
        ax.scatter([i], [hi], s=46, color=c_hi, edgecolor=BG, linewidth=0.8, zorder=5,
                   marker="o" if s < 3 else "o")
        ax.scatter([i], [lo], s=40, color=c_lo, edgecolor=BG, linewidth=0.8, zorder=4)
    tidy(ax, title)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, fontsize=11)
    ax.set_xlabel("network (seed)", fontsize=12.5, color=INK, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=12.5, color=INK, labelpad=5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9)
    ax.scatter([], [], s=46, color=c_hi, label=lab_hi)
    ax.scatter([], [], s=40, color=c_lo, label=lab_lo)
    leg = ax.legend(fontsize=11, frameon=False, loc="upper right", handletextpad=0.3,
                    labelspacing=0.25)
    for t in leg.get_texts():
        t.set_color(INK)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lump, qcond, rot, conc = load()

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.975, top=0.80, bottom=0.10,
                          wspace=0.26, hspace=0.42)
    axA, axB, axC, axD = (fig.add_subplot(gs[i]) for i in [(0, 0), (0, 1), (1, 0), (1, 1)])

    # A: lumpable coordinates
    panel_bars(axA, lump, "1 · lumpable coordinates",
               "lumpability score\n($-$leakage z vs null)")
    axA.set_ylim(0, max(lump.values()) * 1.18)
    axA.text(0.02, 0.94, f"{sum(v > 0 for v in lump.values())}/10 positive",
             transform=axA.transAxes, ha="left", va="top", fontsize=12,
             color=INK, fontweight="bold")

    # B: quotient reduced model — A_cond (conn) vs random-quotient null
    pairsB = {s: (qcond[s]["obs"], qcond[s]["rand"]) for s in qcond}
    panel_paired(axB, pairsB, "2 · quotient reduced model",
                 "$A_{\\rm cond}$ (motif direction,\n$r{=}3$)",
                 lab_hi="connectivity", lab_lo="random quotient")
    axB.text(0.02, 0.06, f"conn > rand in {sum(v[0] > v[1] for v in pairsB.values())}/10",
             transform=axB.transAxes, ha="left", va="bottom", fontsize=12,
             color=INK, fontweight="bold")

    # C: division of labour — ring vs matched non-ring at response epoch
    pairsC = {s: rot[s] for s in rot}
    panel_paired(axC, pairsC, "3 · division of labour (rotation)",
                 "readout potency\n(response epoch)",
                 c_hi=GREEN, lab_hi="memory ring", lab_lo="matched non-ring")
    axC.text(0.02, 0.94, f"ring > matched in {sum(v[0] > v[1] for v in pairsC.values())}/10",
             transform=axC.transAxes, ha="left", va="top", fontsize=12,
             color=INK, fontweight="bold")

    # D: structural bottleneck — participation ratio, conn (low=shared) vs random
    pairsD = {s: (conc[s]["pr_conn"], conc[s]["pr_rand_mean"]) for s in conc}
    panel_paired(axD, pairsD, "4 · structural bottleneck (reuse)",
                 "participation ratio\nover quotient modes",
                 c_hi=RED, lab_hi="connectivity (shared)", lab_lo="random")
    axD.text(0.02, 0.06, f"conn < rand in {sum(v[0] < v[1] for v in pairsD.values())}/10",
             transform=axD.transAxes, ha="left", va="bottom", fontsize=12,
             color=INK, fontweight="bold")

    fig.suptitle("The four-result arc replicates across all ten networks",
                 fontsize=25, color=INK, x=0.07, ha="left", y=0.955, va="top")
    fig.text(0.07, 0.865,
             "256-unit softplus RNNs · dark = 3 released, light = 7 retrained (l1-corrected) · "
             "connectivity beats its null in every network",
             fontsize=12.5, color=MUTED, ha="left", va="top")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"four_result_replication.{ext}", dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT / 'four_result_replication.png'}")
    print(f"  R1 lumpable  : {sum(v>0 for v in lump.values())}/10 positive")
    print(f"  R2 quotient  : {sum(v[0]>v[1] for v in pairsB.values())}/10 conn>rand")
    print(f"  R3 rotation  : {sum(v[0]>v[1] for v in pairsC.values())}/10 ring>matched")
    print(f"  R4 bottleneck: {sum(v[0]<v[1] for v in pairsD.values())}/10 conn<rand PR")


if __name__ == "__main__":
    main()
