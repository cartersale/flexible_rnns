"""Slide: cross-task reuse is real, but not specific to the connectivity partition.

The reuse statistic: for each motif, a participation vector over the groups (each
group's summed squared loading on the motif direction); correlate participation
between motif pairs; R = mean(same-computation pairs) - mean(different-computation
pairs). Positive R means the same coordinates recur for the same computation.

  A  the original result -- R per network at its operating k, positive in 10/10
  B  the control -- at matched k, the SAME statistic on a size-matched random
     partition. Random matches or exceeds connectivity at every scale.
  C  per-network paired connectivity vs random: connectivity wins in a minority.

Data: results/motif_fixed_points/reuse_random_partition_control.json
"""

from __future__ import annotations

from flexsym import paths

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = paths.REPO_ROOT
SRC = REPO / "results/motif_fixed_points/reuse_random_partition_control.json"
OUT = REPO / "results/figures/slides"

BG = "#ffffff"
INK = "#111111"
MUTED = "#5b5a57"
GRID = "#e3e2df"
BLUE = "#2a78d6"
GREY = "#9a9a97"
RED = "#e34948"
NETS = [f"LeakyRNN_softplus_diag_256_all_{s}" for s in range(10)]
KMATCH = 20          # representative matched scale for the paired panel


def tidy(ax, title, fs=16.5):
    ax.set_title(title, fontsize=fs, color=INK, pad=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(1.2)
    ax.tick_params(colors=MUTED, labelsize=12.5, width=1.2, length=5)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)


def panel_original(ax, d):
    R = np.array([d["per_net"][n]["R_raw"] for n in NETS])
    order = np.argsort(-R)
    x = np.arange(len(R))
    ax.axhline(0, color=INK, lw=1.6, zorder=3)
    ax.bar(x, R[order], width=0.68, color=BLUE, edgecolor=BG, linewidth=0.8, zorder=4)
    tidy(ax, "the effect, as first measured")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in np.arange(10)[order]], fontsize=11.5)
    ax.set_xlabel("network (seed), ranked", fontsize=14, color=INK, labelpad=6)
    ax.set_ylabel("reuse  $R$\n(same $-$ different computation)", fontsize=14,
                  color=INK, labelpad=6)
    ax.set_ylim(0, R.max() * 1.32)
    ax.text(0.985, 0.97, f"positive in {int((R > 0).sum())}/10\nmedian {np.median(R):+.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=14, color=INK,
            fontweight="bold", linespacing=1.4)
    return R


def panel_vs_random(ax, d):
    agg = d["aggregate"]
    ks = sorted(int(k) for k in agg)
    conn = np.array([agg[str(k)]["R_connectivity"] for k in ks])
    rand = np.array([agg[str(k)]["R_random"] for k in ks])
    act = np.array([agg[str(k)]["R_activity"] for k in ks])

    ax.plot(ks, rand, color=RED, lw=3.0, marker="D", ms=9, mec=BG, mew=1.2, zorder=6,
            label="size-matched random")
    ax.plot(ks, conn, color=BLUE, lw=3.0, marker="o", ms=10, mec=BG, mew=1.2, zorder=5,
            label="connectivity groups")
    ax.plot(ks, act, color=GREY, lw=2.4, marker="s", ms=8, mec=BG, mew=1.2, zorder=4,
            label="activity clusters")
    ax.fill_between(ks, conn, rand, color=RED, alpha=0.10, lw=0, zorder=2)

    tidy(ax, "…is matched by a random partition")
    ax.set_xlabel("number of groups $k$  (matched)", fontsize=14, color=INK, labelpad=6)
    ax.set_ylabel("reuse  $R$", fontsize=14, color=INK, labelpad=6)
    ax.set_xlim(ks[0] - 3, ks[-1] + 4)
    ax.set_ylim(0.075, max(rand.max(), conn.max()) * 1.10)
    leg = ax.legend(fontsize=13, frameon=False, loc="lower right", handlelength=1.7,
                    labelspacing=0.35, borderpad=0.2)
    for t in leg.get_texts():
        t.set_color(INK)
    return conn, rand


def panel_paired(ax, d):
    c = np.array([d["per_net"][n]["per_k"][str(KMATCH)]["R_connectivity"] for n in NETS])
    r = np.array([d["per_net"][n]["per_k"][str(KMATCH)]["R_random_conn_sizes"] for n in NETS])
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.08, 0.08, len(c))
    for i in range(len(c)):
        ax.plot([jit[i], 1 + jit[i]], [c[i], r[i]],
                color=BLUE if c[i] > r[i] else RED, lw=1.8, alpha=0.6, zorder=2)
    ax.scatter(jit, c, s=62, color=BLUE, edgecolor=BG, linewidth=0.9, zorder=4)
    ax.scatter(1 + jit, r, s=62, color=RED, edgecolor=BG, linewidth=0.9, zorder=4)
    for x, v in ((0, c), (1, r)):
        ax.plot([x - 0.22, x + 0.22], [v.mean()] * 2, color=INK, lw=2.8,
                solid_capstyle="butt", zorder=6)
        ax.text(x + 0.27, v.mean(), f"{v.mean():.2f}", ha="left", va="center",
                fontsize=14.5, color=INK, fontweight="bold", zorder=7)
    tidy(ax, f"{int((c > r).sum())}/10 at $k$ = {KMATCH}")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["connectivity", "size-matched\nrandom"], fontsize=13, color=INK)
    ax.set_ylabel("reuse  $R$", fontsize=14, color=INK, labelpad=6)
    ax.set_xlim(-0.45, 1.55)
    return c, r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.load(open(SRC))

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 0.9],
                          left=0.088, right=0.978, top=0.655, bottom=0.155, wspace=0.44)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

    R = panel_original(axA, d)
    conn, rand = panel_vs_random(axB, d)
    c, r = panel_paired(axC, d)

    agg = d["aggregate"]
    ks = sorted(int(k) for k in agg)
    fracs = [agg[str(k)]["frac_conn_gt_rand"] for k in ks]

    fig.suptitle("Cross-task reuse is real — but not specific to the connectivity groups",
                 fontsize=25, color=INK, x=0.088, ha="left", y=0.975, va="top")
    fig.text(0.088, 0.885,
             "same-computation motif pairs share group participation more than "
             "different-computation pairs ($R>0$),\nbut a size-matched random partition "
             "of the same units does just as well",
             fontsize=14, color=MUTED, ha="left", va="top", linespacing=1.5)
    fig.text(0.978, 0.022,
             f"connectivity beats random in {int(min(fracs)*10)}–{int(max(fracs)*10)} "
             "of 10 networks depending on scale — never a majority",
             ha="right", va="bottom", fontsize=12.5, color=MUTED)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"slide_reuse_fails_control.{ext}", dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT / 'slide_reuse_fails_control.png'}")
    print(f"  R_raw median {np.median(R):+.3f}, positive {int((R > 0).sum())}/10")
    print(f"  matched-k: connectivity {conn.mean():.3f} vs random {rand.mean():.3f} "
          f"(mean over k)")
    print(f"  k={KMATCH}: connectivity higher in {int((c > r).sum())}/10")
    print(f"  frac_conn_gt_rand by k: "
          + ", ".join(f"k={k}:{agg[str(k)]['frac_conn_gt_rand']:.1f}" for k in ks))


if __name__ == "__main__":
    main()
