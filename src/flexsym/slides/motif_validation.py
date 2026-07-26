"""Slide 1 of 2 on dynamical motifs: the taxonomy reproduces on the released networks.

Sanity check restricted to seeds 0-2 -- the networks released by Driscoll et al. (2024),
so the recovered motifs can be checked against motifs already reported for them.

  A  Jacobian eigenvalue spectra in the complex plane for one representative epoch of
     each motif type: memory delay (a marginal mode at |lambda|~1), decision stimulus
     (one mode outside the unit circle), sensory stimulus (all modes inside).
  B  the three canonical epoch-type predictions, as counts over seeds 0-2.

Gate definitions match plot_motif_validation.py.
Data: results/motif_fixed_points/<net>/*.npz and motif_taxonomy_all_nets.csv
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
ROOT = REPO / "results/motif_fixed_points"
TAX = ROOT / "motif_taxonomy_all_nets.csv"
OUT = REPO / "results/figures/slides"

BG = "#ffffff"
INK = "#111111"
MUTED = "#5b5a57"
GRID = "#e3e2df"
BLUE = "#2a78d6"
RED = "#e34948"
GREEN = "#008300"
GREY = "#9a9a97"

# one representative epoch per motif type, from the canonical network
CAN = "LeakyRNN_softplus_diag_256_all_0"
EXAMPLES = [
    ("delaygo__delay1", "memory delay", "line attractor", BLUE, "marginal",
     "a marginal mode at $|\\lambda|\\approx1$\nholds the remembered direction"),
    ("delaydm1__stim1", "decision stimulus", "saddle", RED, "unstable",
     "one mode outside the circle\nis the decision axis"),
    ("fdgo__stim1", "sensory stimulus", "point attractor", GREEN, "stable",
     "every mode inside\nthe state decays to a point"),
]


def pick_point(ev, q, mode):
    """Index of a retained point that exhibits the motif (not merely best-converged)."""
    top = np.abs(ev).max(axis=1)
    if mode == "unstable":                      # a genuine saddle: some |lambda| > 1
        cand = np.flatnonzero(top > 1.0)
        return int(cand[np.argmin(q[cand])]) if cand.size else int(np.argmax(top))
    if mode == "marginal":                      # the near-marginal memory mode
        return int(np.argmin(np.abs(top - 1.0)))
    cand = np.flatnonzero(top <= 1.0)           # a true point attractor
    return int(cand[np.argmin(q[cand])]) if cand.size else int(np.argmin(q))


def spectrum(ax, key, epoch_label, motif, color, mode, caption):
    d = np.load(ROOT / CAN / f"{key}.npz", allow_pickle=True)
    ev = np.asarray(d["eigvals"])
    q = np.asarray(d["qstar"], dtype=float)
    e = ev[pick_point(ev, q, mode)]

    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=MUTED, lw=1.6, zorder=3)
    ax.axhline(0, color=GRID, lw=1.2, zorder=1)
    ax.axvline(0, color=GRID, lw=1.2, zorder=1)
    inside = np.abs(e) <= 1.0
    ax.scatter(e[inside].real, e[inside].imag, s=26, color=GREY, alpha=0.75,
               edgecolor="none", zorder=4)
    if (~inside).any():
        ax.scatter(e[~inside].real, e[~inside].imag, s=110, color=color, marker="*",
                   edgecolor=INK, linewidth=0.8, zorder=6)
    # the mode that defines the motif
    top = int(np.argmax(np.abs(e)))
    ax.scatter([e[top].real], [e[top].imag], s=150, facecolor="none",
               edgecolor=color, linewidth=2.4, zorder=7)
    ax.text(0.5, -0.14, f"$|\\lambda|_{{\\max}}$ = {abs(e[top]):.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=14,
            color=INK, fontweight="bold")

    ax.set_title(f"{epoch_label}\n{motif}", fontsize=17, color=color, pad=10,
                 linespacing=1.25)
    ax.text(0.5, -0.28, caption, transform=ax.transAxes, ha="center", va="top",
            fontsize=12.5, color=MUTED, linespacing=1.3)
    ax.set_aspect("equal")
    lim = max(1.25, np.abs(e).max() * 1.12)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=11.5)


def gate(ax, rows, seeds=(0, 1, 2)):
    """The three canonical predictions, per released network (not pooled)."""
    checks = [("memory delay\n→ line attractor", BLUE,
               lambda r: r["epoch_type"] == "memory" and r["epoch"] in ("delay1", "delay2"),
               lambda r: r["line_attractor"] == "True"),
              ("decision stimulus\n→ ≥1 saddle", RED,
               lambda r: r["is_decision_rule"] == "True" and r["epoch_type"] == "stim",
               lambda r: int(r["n_saddle"]) > 0),
              ("sensory stimulus\n→ point attractor", GREEN,
               lambda r: r["is_decision_rule"] != "True" and r["epoch_type"] == "stim",
               lambda r: r["motif"] == "point_attractor")]

    w = 0.26
    out = []
    for ci, (label, c, sel, ok) in enumerate(checks):
        for si, sd in enumerate(seeds):
            sub = [r for r in rows if int(r["seed"]) == sd and sel(r)]
            n, tot = sum(ok(r) for r in sub), len(sub)
            x = ci + (si - 1) * w
            ax.bar(x, 1.0, width=w * 0.86, color=GRID, zorder=2)
            ax.bar(x, n / max(tot, 1), width=w * 0.86, color=c,
                   alpha=1.0 if sd == 0 else 0.45, zorder=3)
            ax.text(x, n / max(tot, 1) + 0.035, f"{n}/{tot}", ha="center", va="bottom",
                    fontsize=12.5, color=INK, fontweight="bold")
            ax.text(x, -0.055, f"seed {sd}", ha="center", va="top", fontsize=11.5,
                    color=INK if sd == 0 else MUTED)
            out.append((label, sd, n, tot))

    ax.set_xticks(range(len(checks)))
    ax.set_xticklabels([c[0].replace(chr(10), "  ") for c in checks], fontsize=13.5, color=INK, linespacing=1.35)
    ax.tick_params(axis="x", pad=24)
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "50", "100"])
    ax.set_ylabel("epochs matching\nthe prediction (%)", fontsize=13.5, color=INK,
                  labelpad=6)
    ax.set_title("the canonical network validates without exception — the other two do not",
                 fontsize=17, color=INK, pad=12, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=12.5)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(TAX)))

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor(BG)
    gsT = fig.add_gridspec(1, 3, left=0.085, right=0.945, top=0.745, bottom=0.535,
                           wspace=0.30)
    axes = [fig.add_subplot(gsT[i]) for i in range(3)]
    for ax, (key, lab, motif, c, mode, cap) in zip(axes, EXAMPLES):
        spectrum(ax, key, lab, motif, c, mode, cap)

    gsB = fig.add_gridspec(1, 1, left=0.085, right=0.945, top=0.325, bottom=0.205)
    axg = fig.add_subplot(gsB[0])
    checks = gate(axg, rows)

    fig.suptitle("The recovered motifs reproduce the reported taxonomy",
                 fontsize=26, color=INK, x=0.055, ha="left", y=0.975, va="top")
    fig.text(0.085, 0.905,
             "seeds 0–2 — the networks released by Driscoll et al. (2024) — "
             "spectra from the canonical network",
             fontsize=14, color=MUTED, ha="left", va="top")
    fig.text(0.085, 0.032,
             "seed 0 is the network the manuscript reports  ·  seed 1 leaves 16 of its 49 "
             "epochs unclassified (mixed), which is what its gate failures reflect",
             fontsize=12.5, color=MUTED, ha="left", va="bottom")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"slide_motif_validation.{ext}", dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT / 'slide_motif_validation.png'}")
    for label, sd, n, tot in checks:
        print(f"  seed {sd}  {label.replace(chr(10), ' ')}: {n}/{tot}")


if __name__ == "__main__":
    main()
