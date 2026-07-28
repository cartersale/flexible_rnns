"""Result 3 (decisive): the memory ring rotates into the readout, and it is task-specific.

The connectivity quotient cannot distinguish a motif mode from a matched-eigenvalue non-motif
mode (the eig-matched control). What singles out the motif is the TASK: as the fixation input
drops (delay -> response), the memory ring rotates from output-null into output-potent space --
while a dynamically-matched non-ring plane of the same response Jacobian stays dark.

  A  the transition: readout potency of the ring at the delay epoch (null) vs the response
     epoch (potent), vs a matched non-ring plane and a random plane -- pooled over tasks.
  B  per task (canonical net): ring vs matched non-ring at the response epoch.
  C  per network: ring vs matched, median over that net's delayed-response tasks -- the
     replication across all ten seeds (released + l1-refixed).

Data: results/motif_fixed_points/rotation_readout_control.csv (seeds 0-9, 256-unit).
"""

from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flexsym import paths

SRC = paths.REPO_ROOT / "results/motif_fixed_points/rotation_readout_control.csv"
OUT = paths.FIGURES_ROOT / "slides"
CANON = "LeakyRNN_softplus_diag_256_all_0"

BG = "#ffffff"
INK = "#111111"
MUTED = "#5b5a57"
GRID = "#e3e2df"
GREEN = "#008300"
BLUE = "#2a78d6"
DARK = "#0f3f73"
GREY = "#9a9a97"
PALE = "#c9c8c5"


def tidy(ax, title, fs=16):
    ax.set_title(title, fontsize=fs, color=INK, pad=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(1.1)
    ax.tick_params(colors=MUTED, labelsize=12, width=1.1, length=4)
    ax.set_axisbelow(True)


def load():
    rows = list(csv.DictReader(open(SRC)))
    for r in rows:
        r["seed"] = int(r["net"].split("_")[-1])
        for c in ("ring_delay_potency", "ring_resp_potency", "match_resp_potency",
                  "random_potency"):
            r[c] = float(r[c])
    return rows


def panel_transition(ax, rows):
    """Pooled readout potency: delay ring, response ring, matched non-ring, random."""
    cats = [("ring\n(delay)", [r["ring_delay_potency"] for r in rows], PALE),
            ("ring\n(response)", [r["ring_resp_potency"] for r in rows], GREEN),
            ("matched\nnon-ring", [r["match_resp_potency"] for r in rows], GREY),
            ("random\nplane", [r["random_potency"] for r in rows], PALE)]
    for i, (lab, vals, col) in enumerate(cats):
        v = np.array(vals)
        ax.bar(i, np.median(v), width=0.62, color=col, edgecolor=BG, linewidth=0.8, zorder=3)
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)),
                   v, s=10, color=INK, alpha=0.28, zorder=4)
    tidy(ax, "the ring rotates into the readout")
    ax.set_xticks(range(4))
    ax.set_xticklabels([c[0] for c in cats], fontsize=11.5, color=INK)
    ax.set_ylabel("readout potency", fontsize=13, color=INK, labelpad=5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9)
    ax.annotate("", xy=(1, np.median([r["ring_resp_potency"] for r in rows])),
                xytext=(0, np.median([r["ring_delay_potency"] for r in rows]) + 0.004),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.0,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(0.5, np.median([r["ring_resp_potency"] for r in rows]) * 0.9,
            "fixation\ndrops", fontsize=11, color=INK, ha="center", va="top",
            style="italic", linespacing=1.2)


def panel_per_task(ax, rows):
    """Canonical net: ring vs matched non-ring, per delayed-response task."""
    r = sorted((x for x in rows if x["net"] == CANON),
               key=lambda z: -z["ring_resp_potency"])
    x = np.arange(len(r))
    for i, row in enumerate(r):
        ax.plot([i, i], [row["match_resp_potency"], row["ring_resp_potency"]],
                color=GRID, lw=2.0, zorder=2)
    ax.scatter(x, [row["ring_resp_potency"] for row in r], s=42, color=GREEN,
               edgecolor=BG, linewidth=0.8, zorder=5, label="memory ring")
    ax.scatter(x, [row["match_resp_potency"] for row in r], s=38, color=GREY,
               edgecolor=BG, linewidth=0.8, zorder=4, label="matched non-ring")
    tidy(ax, "every delayed-response task (canonical net)")
    ax.set_xticks(x)
    ax.set_xticklabels([row["rule"] for row in r], fontsize=8.5, rotation=45, ha="right")
    ax.set_ylabel("readout potency\n(response epoch)", fontsize=13, color=INK, labelpad=5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9)
    leg = ax.legend(fontsize=11.5, frameon=False, loc="upper right", labelspacing=0.25)
    for t in leg.get_texts():
        t.set_color(INK)


def panel_per_net(ax, rows):
    """Per network: median ring vs matched, released vs fixed."""
    by = defaultdict(list)
    for r in rows:
        by[r["seed"]].append(r)
    seeds = sorted(by)
    ring = [np.median([x["ring_resp_potency"] for x in by[s]]) for s in seeds]
    match = [np.median([x["match_resp_potency"] for x in by[s]]) for s in seeds]
    x = np.arange(len(seeds))
    for i in x:
        ax.plot([i, i], [match[i], ring[i]], color=GRID, lw=2.0, zorder=2)
    ax.scatter(x, ring, s=46, color=[DARK if s < 3 else GREEN for s in seeds],
               edgecolor=BG, linewidth=0.8, zorder=5)
    ax.scatter(x, match, s=40, color=GREY, edgecolor=BG, linewidth=0.8, zorder=4)
    tidy(ax, "…in every network")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, fontsize=11)
    ax.set_xlabel("network (seed)", fontsize=13, color=INK, labelpad=4)
    ax.set_ylabel("readout potency\n(median over tasks)", fontsize=13, color=INK, labelpad=5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9)
    n_pos = sum(r > m for r, m in zip(ring, match))
    ntask = sum(x["ring_resp_potency"] > x["match_resp_potency"] for x in rows)
    ax.text(0.98, 0.94, f"ring > matched:\n{n_pos}/10 nets, {ntask}/{len(rows)} tasks",
            transform=ax.transAxes, ha="right", va="top", fontsize=12, color=INK,
            fontweight="bold", linespacing=1.3)
    ax.scatter([], [], s=46, color=GREEN, label="memory ring (dark = released)")
    ax.scatter([], [], s=40, color=GREY, label="matched non-ring")
    leg = ax.legend(fontsize=10.5, frameon=False, loc="lower right", labelspacing=0.25)
    for t in leg.get_texts():
        t.set_color(INK)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.82, 1.15, 1.0],
                          left=0.065, right=0.978, top=0.70, bottom=0.20, wspace=0.36)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))
    panel_transition(axA, rows)
    panel_per_task(axB, rows)
    panel_per_net(axC, rows)

    fig.suptitle("The task recruits the memory ring into the readout",
                 fontsize=25, color=INK, x=0.065, ha="left", y=0.955, va="top")
    fig.text(0.065, 0.865,
             "as fixation drops (delay→response), the memory ring rotates from output-null into "
             "output-potent space, beating a matched-eigenvalue\nnon-ring plane of the same "
             "response Jacobian (not just a random plane) — the task-specific signature the "
             "connectivity quotient cannot provide",
             fontsize=12.5, color=MUTED, ha="left", va="top", linespacing=1.4)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"rotation_readout.{ext}", dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT / 'rotation_readout.png'}")


if __name__ == "__main__":
    main()
