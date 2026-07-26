"""Slide 2 of 2 on dynamical motifs: what was recovered, across every network.

  A  motif inventory per network -- stacked counts of the classified task-epoch motifs
  B  epoch type -> motif, pooled over networks, row-normalised: the mapping from what a
     task epoch demands to the motif that implements it
  C  |Im(lambda)| of each epoch's dominant mode: near-real throughout, i.e. no rotations

Networks: all 256-unit nets with fixed-point results (1024 excluded by request; no
fixed-point runs exist at other sizes).
Data: results/motif_fixed_points/motif_taxonomy_all_nets.csv and <net>/*.npz
"""

from __future__ import annotations

from flexsym import paths

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

from flexsym.motifs.motif_classify import IM_THRESH  # noqa: E402  (|Im lambda| above this => rotation)

REPO = paths.REPO_ROOT
ROOT = REPO / "results/motif_fixed_points"
TAX = ROOT / "motif_taxonomy_all_nets.csv"
OUT = REPO / "results/figures/slides"
CACHE = ROOT / "dominant_mode_imag.npz"

BG = "#ffffff"
INK = "#111111"
MUTED = "#5b5a57"
GRID = "#e3e2df"
BLUE = "#2a78d6"
RED = "#e34948"
GREEN = "#008300"
GREY = "#9a9a97"
PURPLE = "#4a3aa7"

MOTIFS = ["line_attractor", "point_attractor", "saddle_decision", "rotation", "mixed"]
MLABEL = {"line_attractor": "line attractor", "point_attractor": "point attractor",
          "saddle_decision": "saddle", "rotation": "rotation", "mixed": "mixed"}
MSHORT = {"line_attractor": "line", "point_attractor": "point",
          "saddle_decision": "saddle", "rotation": "rot.", "mixed": "mixed"}
MCOLOR = {"line_attractor": BLUE, "point_attractor": GREEN, "saddle_decision": RED,
          "rotation": PURPLE, "mixed": GREY}

ETYPES = [("memory delay", lambda r: r["epoch_type"] == "memory"
           and r["epoch"] in ("delay1", "delay2")),
          ("decision stimulus", lambda r: r["is_decision_rule"] == "True"
           and r["epoch_type"] == "stim"),
          ("sensory stimulus", lambda r: r["is_decision_rule"] != "True"
           and r["epoch_type"] == "stim"),
          ("fixation", lambda r: r["epoch"] == "fix1")]


def tidy(ax, title, fs=17):
    ax.set_title(title, fontsize=fs, color=INK, pad=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(1.2)
    ax.tick_params(colors=MUTED, labelsize=12.5, width=1.2, length=5)
    ax.set_axisbelow(True)


def panel_inventory(ax, rows):
    seeds = sorted({int(r["seed"]) for r in rows})
    x = np.arange(len(seeds))
    bottom = np.zeros(len(seeds))
    for m in MOTIFS:
        v = np.array([sum(1 for r in rows if int(r["seed"]) == s and r["motif"] == m)
                      for s in seeds], dtype=float)
        if v.sum() == 0:
            continue
        ax.bar(x, v, bottom=bottom, width=0.72, color=MCOLOR[m], label=MLABEL[m],
               edgecolor=BG, linewidth=0.8, zorder=3)
        bottom += v
    tidy(ax, "motifs recovered, per network")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds], fontsize=12)
    ax.set_xlabel("network (seed)", fontsize=14, color=INK, labelpad=6)
    ax.set_ylabel("task epochs classified", fontsize=14, color=INK, labelpad=6)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    leg = ax.legend(fontsize=12.5, frameon=False, loc="upper center", ncol=2,
                    bbox_to_anchor=(0.5, 1.30), handlelength=1.3, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(INK)
    return bottom


def panel_mapping(ax, rows):
    present = [m for m in MOTIFS if any(r["motif"] == m for r in rows)]
    M = np.zeros((len(ETYPES), len(present)))
    tot = []
    for i, (_, pred) in enumerate(ETYPES):
        sub = [r for r in rows if pred(r)]
        tot.append(len(sub))
        c = Counter(r["motif"] for r in sub)
        for j, m in enumerate(present):
            M[i, j] = c[m] / max(len(sub), 1)

    ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(ETYPES)):
        for j in range(len(present)):
            if M[i, j] > 0.005:
                ax.text(j, i, f"{M[i, j]*100:.0f}", ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color=BG if M[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([MSHORT[m] for m in present], fontsize=13, color=INK)
    ax.set_yticks(range(len(ETYPES)))
    ax.set_yticklabels([f"{n}\n({t})" for (n, _), t in zip(ETYPES, tot)], fontsize=12.5,
                       color=INK)
    ax.set_title("epoch type → motif  (%)", fontsize=17, color=INK, pad=11, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=MUTED)


def dominant_imag(rows):
    """|Im(lambda)| of the dominant mode of each epoch's best-converged point."""
    if CACHE.exists():
        return np.load(CACHE)["imag"]
    vals = []
    for r in rows:
        p = ROOT / r["net"] / f"{r['rule']}__{r['epoch']}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        ev = np.asarray(d["eigvals"])
        q = np.asarray(d["qstar"], dtype=float)
        e = ev[int(np.argmin(q))]
        vals.append(abs(e[int(np.argmax(np.abs(e)))].imag))
    imag = np.asarray(vals)
    np.savez(CACHE, imag=imag)
    return imag


def panel_rotation(ax, imag):
    ax.hist(imag, bins=np.linspace(0, max(IM_THRESH * 2.4, imag.max() * 1.05), 34),
            color=GREY, edgecolor=BG, linewidth=0.7, zorder=3)
    ax.axvline(IM_THRESH, color=PURPLE, ls="--", lw=2.0, zorder=4)
    ax.text(IM_THRESH + 0.005, 0.93, "rotation\nthreshold", transform=ax.get_xaxis_transform(),
            fontsize=12.5, color=PURPLE, va="top", ha="left", linespacing=1.25)
    tidy(ax, "spectra are near-real")
    ax.set_xlabel("$|\\,\\mathrm{Im}\\,\\lambda\\,|$ of the dominant mode", fontsize=14,
                  color=INK, labelpad=6)
    ax.set_ylabel("task epochs", fontsize=14, color=INK, labelpad=6)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    frac = float(np.mean(imag < IM_THRESH))
    ax.text(0.985, 0.60, f"{frac*100:.0f}% below\nthreshold", transform=ax.transAxes,
            ha="right", va="top", fontsize=14, color=INK, fontweight="bold",
            linespacing=1.3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(TAX)) if int(r["size"]) != 1024]

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0],
                          left=0.062, right=0.978, top=0.655, bottom=0.135, wspace=0.40)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

    totals = panel_inventory(axA, rows)
    panel_mapping(axB, rows)
    imag = dominant_imag(rows)
    panel_rotation(axC, imag)

    n_nets = len({r["net"] for r in rows})
    counts = Counter(r["motif"] for r in rows)
    fig.suptitle("Line and point attractors in every network — saddles vary",
                 fontsize=26, color=INK, x=0.062, ha="left", y=0.975, va="top")
    fig.text(0.062, 0.885,
             f"{len(rows)} classified task epochs across {n_nets} networks (256 units)  ·  "
             f"{counts['line_attractor']} line attractors, "
             f"{counts['point_attractor']} point attractors, "
             f"{counts['saddle_decision']} saddles",
             fontsize=14, color=MUTED, ha="left", va="top")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"slide_motif_inventory.{ext}", dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT / 'slide_motif_inventory.png'}")
    print(f"  epochs/net: {totals.min():.0f}-{totals.max():.0f}   counts: {dict(counts)}")
    print(f"  |Im lambda| dominant: median {np.median(imag):.4f}, "
          f"{np.mean(imag < IM_THRESH)*100:.0f}% below {IM_THRESH}, max {imag.max():.3f}")


if __name__ == "__main__":
    main()
