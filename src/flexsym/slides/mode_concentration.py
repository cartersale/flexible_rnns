"""Result 4: the structural bottleneck — task motifs share a few connectivity-quotient modes.

Expressed in the connectivity quotient's eigenmode coordinates, the motifs of different tasks
pile onto a shared handful of modes; a size-matched random quotient spreads the same motifs
over ~3x as many. This is the properly-controlled reuse claim (the group-participation version
failed its random control).

  A  energy spectrum (canonical net, k=20): fraction of total motif energy on each quotient
     mode, sorted -- connectivity concentrated on ~3 modes vs random spread flat.
  B  participation ratio per network (effective # modes the motifs occupy): connectivity
     (shared, low) vs random, all ten seeds (released + l1-refixed).
  C  fraction of motif energy on the top-3 modes, per network: connectivity vs random.

Data: results/motif_fixed_points/quotient_mode_concentration.json + (for A) recomputed
      from the WL sweep + participation cache of the canonical network.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flexsym import paths
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.run.motif_quotient_prediction import condition_jacobians, load_conditions
from flexsym.run.motif_quotient_mode_concentration import quotient_modes

SRC = paths.REPO_ROOT / "results/motif_fixed_points/quotient_mode_concentration.json"
OUT = paths.FIGURES_ROOT / "slides"
CANON = "LeakyRNN_softplus_diag_256_all_0"
K = "20"

BG = "#ffffff"
INK = "#111111"
MUTED = "#5b5a57"
GRID = "#e3e2df"
RED = "#e34948"
DARK = "#0f3f73"
GREY = "#9a9a97"


def tidy(ax, title, fs=16):
    ax.set_title(title, fontsize=fs, color=INK, pad=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(1.1)
    ax.tick_params(colors=MUTED, labelsize=12, width=1.1, length=4)
    ax.set_axisbelow(True)


def energy_spectrum(net, k):
    """Fraction of total motif energy per quotient mode, connectivity vs a random quotient."""
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], float)
    ind = np.asarray(wl["ind_active"], int)
    jbars = condition_jacobians(net, ind, W)
    conds = load_conditions(net, ind, W, jbars)
    P = np.array([c["v"] for c in conds])
    ls = wl["labels_struct_by_k"]
    role = np.asarray(ls[_row_for_k(ls, k)], int)
    M, _ = quotient_modes(W, role)
    E = ((P @ M) ** 2).sum(0)
    E = np.sort(E)[::-1] / E.sum()
    rng = np.random.default_rng(0)
    _, sizes = np.unique(role, return_counts=True)
    Er = []
    for _ in range(50):
        perm = np.repeat(np.arange(len(sizes)), sizes)
        rng.shuffle(perm)
        Mr, _ = quotient_modes(W, perm)
        e = ((P @ Mr) ** 2).sum(0)
        Er.append(np.sort(e)[::-1] / e.sum())
    return E, np.mean(Er, 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.load(open(SRC))
    nets = [f"LeakyRNN_softplus_diag_256_all_{s}" for s in range(10)]

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0],
                          left=0.065, right=0.978, top=0.70, bottom=0.16, wspace=0.34)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

    # A: energy spectrum, canonical net
    E, Er = energy_spectrum(CANON, int(K))
    r = np.arange(1, min(15, len(E)) + 1)
    axA.bar(r - 0.19, E[:len(r)] * 100, width=0.38, color=RED, label="connectivity", zorder=3)
    axA.bar(r + 0.19, Er[:len(r)] * 100, width=0.38, color=GREY, label="random quotient",
            zorder=3)
    tidy(axA, "motifs pile onto a few modes")
    axA.set_xlabel("quotient mode (sorted by energy)", fontsize=13, color=INK, labelpad=4)
    axA.set_ylabel("% of total motif energy", fontsize=13, color=INK, labelpad=5)
    axA.set_xticks(r[::2])
    axA.yaxis.grid(True, color=GRID, linewidth=0.9)
    leg = axA.legend(fontsize=12, frameon=False, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    axA.text(0.97, 0.62, f"top-3: {E[:3].sum()*100:.0f}% vs {Er[:3].sum()*100:.0f}%",
             transform=axA.transAxes, ha="right", va="top", fontsize=12.5, color=INK,
             fontweight="bold")

    # B: participation ratio per network
    seeds = list(range(10))
    prc = [d[nets[s]][K]["pr_conn"] for s in seeds]
    prr = [d[nets[s]][K]["pr_rand_mean"] for s in seeds]
    x = np.arange(10)
    for i in x:
        axB.plot([i, i], [prc[i], prr[i]], color=GRID, lw=2.0, zorder=2)
    axB.scatter(x, prr, s=40, color=GREY, edgecolor=BG, linewidth=0.8, zorder=4)
    axB.scatter(x, prc, s=46, color=[DARK if s < 3 else RED for s in seeds],
                edgecolor=BG, linewidth=0.8, zorder=5)
    tidy(axB, "shared few modes, every network")
    axB.set_xticks(x)
    axB.set_xticklabels(seeds, fontsize=11)
    axB.set_xlabel("network (seed)", fontsize=13, color=INK, labelpad=4)
    axB.set_ylabel("participation ratio\n(effective # modes)", fontsize=13, color=INK,
                   labelpad=5)
    axB.yaxis.grid(True, color=GRID, linewidth=0.9)
    axB.text(0.02, 0.06, f"conn < random in {sum(c < r for c, r in zip(prc, prr))}/10",
             transform=axB.transAxes, ha="left", va="bottom", fontsize=12, color=INK,
             fontweight="bold")
    axB.scatter([], [], s=46, color=RED, label="connectivity (dark=released)")
    axB.scatter([], [], s=40, color=GREY, label="random")
    leg = axB.legend(fontsize=10.5, frameon=False, loc="upper left", labelspacing=0.25)
    for t in leg.get_texts():
        t.set_color(INK)

    # C: top-3 energy fraction per network
    tc = [d[nets[s]][K]["top_conn"] * 100 for s in seeds]
    tr = [d[nets[s]][K]["top_rand_mean"] * 100 for s in seeds]
    for i in x:
        axC.plot([i, i], [tr[i], tc[i]], color=GRID, lw=2.0, zorder=2)
    axC.scatter(x, tc, s=46, color=[DARK if s < 3 else RED for s in seeds],
                edgecolor=BG, linewidth=0.8, zorder=5)
    axC.scatter(x, tr, s=40, color=GREY, edgecolor=BG, linewidth=0.8, zorder=4)
    tidy(axC, "top-3 modes carry most of it")
    axC.set_xticks(x)
    axC.set_xticklabels(seeds, fontsize=11)
    axC.set_xlabel("network (seed)", fontsize=13, color=INK, labelpad=4)
    axC.set_ylabel("% motif energy on top-3 modes", fontsize=13, color=INK, labelpad=5)
    axC.set_ylim(0, 100)
    axC.yaxis.grid(True, color=GRID, linewidth=0.9)

    fig.suptitle("A shared bottleneck: tasks reuse the same few quotient modes",
                 fontsize=25, color=INK, x=0.065, ha="left", y=0.955, va="top")
    fig.text(0.065, 0.855,
             "each task motif expressed in the connectivity quotient's eigenmode coordinates · "
             "the motifs concentrate on a shared low-dimensional\nset that a size-matched random "
             "quotient does not — the properly-controlled version of shared dynamical motifs",
             fontsize=12.5, color=MUTED, ha="left", va="top", linespacing=1.4)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"mode_concentration.{ext}", dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT / 'mode_concentration.png'}")
    print(f"  top-3 energy canonical: conn {E[:3].sum()*100:.0f}% vs rand {Er[:3].sum()*100:.0f}%")
    print(f"  PR conn<rand in {sum(c<r for c,r in zip(prc,prr))}/10")


if __name__ == "__main__":
    main()
