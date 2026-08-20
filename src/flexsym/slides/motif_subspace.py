"""Slide: the directions of the dynamical motifs lie in the connectivity subspace.

  A  per-network median subspace-containment score (motif direction vs size-matched
     random partitions), positive in every network
  B  the decisive comparison -- connectivity vs activity partition matched on k AND on
     the group-size distribution, each minus its own size-matched random baseline,
     across matched scales
  C  containment holds for all three motif types

Containment is ||Pp||/||p|| for motif direction p and group projector P; it depends only
on the subspace a partition spans, not on the labelling.

Data: results/motif_fixed_points/subspace_role_vs_activity.json
      results/motif_fixed_points/<net>/participation_per_motif.csv
"""

from __future__ import annotations

from flexsym import paths

import collections
import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = paths.REPO_ROOT
ROOT = paths.RESULTS_ROOT / "motif_fixed_points"
JSON = ROOT / "subspace_role_vs_activity.json"
OUT = paths.FIGURES_ROOT / "slides"

BG = "#ffffff"
INK = "#111111"
MUTED = "#5b5a57"
GRID = "#e3e2df"
BLUE = "#2a78d6"
GREEN = "#008300"
RED = "#e34948"
GREY = "#9a9a97"

TYPES = [("line_attractor", "line", BLUE),
         ("point_attractor", "point", GREEN),
         ("saddle_decision", "saddle", RED)]


def tidy(ax, title, fs=17):
    ax.set_title(title, fontsize=fs, color=INK, pad=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(1.2)
    ax.tick_params(colors=MUTED, labelsize=12.5, width=1.2, length=5)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)


def load_motifs():
    per = collections.defaultdict(list)
    for p in sorted(glob.glob(str(ROOT / "*/participation_per_motif.csv"))):
        seed = int(p.split("_all_")[1].split("/")[0])
        for r in csv.DictReader(open(p)):
            per[seed].append(r)
    return per


def panel_per_net(ax, per):
    seeds = sorted(per)
    med = np.array([np.median([float(r["sub_z_role"]) for r in per[s]]) for s in seeds])
    order = np.argsort(-med)
    x = np.arange(len(seeds))
    ax.axhline(0, color=INK, lw=1.6, zorder=3)
    ax.bar(x, med[order], width=0.68, color=BLUE, edgecolor=BG, linewidth=0.8, zorder=4)
    for xi, s in zip(x, np.array(seeds)[order]):
        ax.text(xi, -0.35, str(s), ha="center", va="top", fontsize=11.5, color=MUTED)
    tidy(ax, "every network, above chance")
    ax.set_xticks([])
    ax.set_xlabel("network (seed), ranked", fontsize=14, color=INK, labelpad=18)
    ax.set_ylabel("subspace-containment score\n(median over motifs)", fontsize=14,
                  color=INK, labelpad=6)
    ax.set_xlim(-0.7, len(seeds) - 0.3)
    ax.set_ylim(0, med.max() * 1.30)
    ax.text(0.985, 0.98, f"{int(np.sum(med > 0))}/{len(med)} networks\nmedian "
            f"{np.median(med):+.1f}   $P$ = 0.002",
            transform=ax.transAxes, ha="right", va="top", fontsize=13.5, color=INK,
            fontweight="bold", linespacing=1.4)
    return med


def panel_vs_activity(ax, agg):
    ks = sorted(int(k) for k in agg)
    role = np.array([agg[str(k)]["median_delta_role"] for k in ks])
    act = np.array([agg[str(k)]["median_delta_activity"] for k in ks])
    frac = np.array([agg[str(k)]["frac_pos"] for k in ks])

    ax.axhline(0, color=INK, lw=1.6, zorder=3)
    ax.plot(ks, role, color=BLUE, lw=3.0, marker="o", ms=10, mec=BG, mew=1.2, zorder=6,
            label="connectivity groups")
    ax.plot(ks, act, color=GREY, lw=3.0, marker="s", ms=9, mec=BG, mew=1.2, zorder=5,
            label="activity clusters")
    ax.fill_between(ks, act, role, color=BLUE, alpha=0.10, lw=0, zorder=2)

    tidy(ax, "activity captures nothing")
    ax.set_xlabel("number of groups $k$  (matched between partitions)", fontsize=14,
                  color=INK, labelpad=6)
    ax.set_ylabel("excess containment over each\npartition's own size-matched random",
                  fontsize=14, color=INK, labelpad=6)
    ax.set_xlim(ks[0] - 3, ks[-1] + 15)
    ax.text(ks[-1] + 1.5, role[-1], "connectivity\ngroups", color=BLUE, fontsize=13.5,
            fontweight="bold", va="center", ha="left", linespacing=1.25)
    ax.text(ks[-1] + 1.5, act[-1], "activity\nclusters", color=MUTED, fontsize=13.5,
            fontweight="bold", va="center", ha="left", linespacing=1.25)
    lo = min(ks[i] for i in range(len(ks)) if frac[i] == 1.0)
    ax.text(0.04, 0.40, f"connectivity > activity\nin 10/10 networks\nat every $k \\geq$ {lo}",
            transform=ax.transAxes, ha="left", va="top", fontsize=13, color=INK,
            fontweight="bold", linespacing=1.4)
    return role, act


def panel_by_type(ax, per):
    seeds = sorted(per)
    x = np.arange(len(TYPES))
    rng = np.random.default_rng(0)
    ax.axhline(0, color=INK, lw=1.6, zorder=3)
    meds = []
    for i, (key, lab, c) in enumerate(TYPES):
        v = np.array([np.median([float(r["sub_z_role"]) for r in per[s]
                                 if r["motif"] == key])
                      for s in seeds if any(r["motif"] == key for r in per[s])])
        meds.append(v)
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=62, color=c,
                   edgecolor=BG, linewidth=0.9, zorder=4)
        ax.plot([i - 0.26, i + 0.26], [np.median(v)] * 2, color=INK, lw=2.8,
                solid_capstyle="butt", zorder=6)
        ax.text(i + 0.30, np.median(v), f"{np.median(v):+.1f}", ha="left", va="center",
                fontsize=14, color=INK, fontweight="bold", zorder=7)
    tidy(ax, "for all three motif types")
    ax.set_xticks(x)
    ax.set_xticklabels([t[1] for t in TYPES], fontsize=13.5, color=INK)
    ax.set_ylabel("subspace-containment score", fontsize=14, color=INK, labelpad=6)
    ax.set_xlim(-0.45, len(TYPES) - 0.25)
    return meds


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.load(open(JSON))
    per = load_motifs()

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.35, 0.95],
                          left=0.068, right=0.978, top=0.665, bottom=0.155, wspace=0.44)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

    med = panel_per_net(axA, per)
    role, act = panel_vs_activity(axB, d["aggregate"])
    panel_by_type(axC, per)

    frac = np.median([float(r["sub_frac"]) for r in per[0]])
    fig.suptitle("The motif directions lie inside the connectivity subspace",
                 fontsize=26, color=INK, x=0.068, ha="left", y=0.975, va="top")
    fig.text(0.068, 0.885,
             "fraction of each motif direction inside the group subspace, "
             f"$\\|Pp\\|/\\|p\\|$, vs size-matched random partitions\n"
             f"{int(frac*100)}% for a typical motif in the canonical network  ·  "
             "10 networks, 256 units",
             fontsize=14, color=MUTED, ha="left", va="top", linespacing=1.5)
    fig.text(0.978, 0.022,
             "a subspace statistic — invariant to which unit sits in which group",
             ha="right", va="bottom", fontsize=12.5, color=MUTED)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"slide_motif_subspace.{ext}", dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT / 'slide_motif_subspace.png'}")
    print(f"  per-net median score: {np.median(med):+.2f}, positive in "
          f"{int(np.sum(med > 0))}/{len(med)}  (range {med.min():+.2f} to {med.max():+.2f})")
    print(f"  delta_role {role.min():.3f}-{role.max():.3f}   "
          f"delta_activity {act.min():.3f}-{act.max():.3f}")


if __name__ == "__main__":
    main()
