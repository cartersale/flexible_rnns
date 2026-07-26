"""Slide: the connectivity quotient predicts motif directions, at every scale.

Main    -- A_cond across k for three quotients: connectivity, size-matched random, and
           activity. A_cond = ||P_VQ p|| / ||P_H p||: of the part of the motif direction
           that the partition's subspace can reach at all, how much the r quotient modes
           actually capture. The random quotient gets its own matched partition, group
           sizes, gains and mode selection, so it is a like-for-like null.
Inset   -- at k = 20, the raw prediction against the containment ceiling, on the scale of
           the whole unit direction.
Right   -- eigenvalue error |lambda_Q - lambda_J|, the sharper version of the same result.

k sweep, r = 3, seeds 0-2 (the released networks).
Data: results/motif_fixed_points/quotient_prediction.json (A_cond, lambda error)
      results/motif_fixed_points/quotient_controls.json  (A_raw, containment ceiling)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PRED = REPO / "results/motif_fixed_points/quotient_prediction.json"
CTRL = REPO / "results/motif_fixed_points/quotient_controls.json"
OUT = REPO / "results/figures/slides"

BG = "#ffffff"
INK = "#111111"
MUTED = "#5b5a57"
GRID = "#e3e2df"
BLUE = "#2a78d6"
PALE = "#a9cbf2"
GREY = "#9a9a97"
PGREY = "#c9c8c5"
RED = "#e34948"
DARK = "#0f3f73"
KEY = "20"


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


def load():
    p = json.load(open(PRED))
    c = json.load(open(CTRL))
    nets = sorted(n for n in p if n.startswith("LeakyRNN"))
    ks = sorted(int(k) for k in p[nets[0]]["per_k"])

    def stack(field, which):
        return np.array([[p[n]["per_k"][str(k)][field][which] for k in ks] for n in nets])

    acond = {w: stack("pred_m3", w) for w in ("obs", "rand", "act")}
    lam = {w: stack("lam_err", w) for w in ("obs", "rand", "act")}
    raw = np.array([c[n]["per_k"][KEY]["Q_pred|r3"]["raw"]["obs"] for n in nets])
    ceil = np.array([c[n]["per_k"][KEY]["containment"]["raw"]["obs"] for n in nets])
    return ks, acond, lam, raw, ceil


def panel_acond(ax, ks, a):
    series = [("obs", BLUE, "o", "connectivity"),
              ("act", GREY, "s", "activity"),
              ("rand", RED, "D", "size-matched random")]
    for w, col, mk, lab in series:
        for row in a[w]:                       # per-seed traces, faint
            ax.plot(ks, row, color=col, lw=1.2, alpha=0.30, zorder=3)
        ax.plot(ks, np.median(a[w], 0), color=col, lw=3.2, marker=mk, ms=9.5,
                mec=BG, mew=1.2, zorder=6, label=lab)

    tidy(ax, "connectivity beats both nulls at every scale")
    ax.set_xlabel("number of groups  $k$", fontsize=15, color=INK, labelpad=7)
    ax.set_ylabel("$A_{\\mathrm{cond}}$   (of what is reachable,\nhow much 3 quotient "
                  "modes capture)", fontsize=14.5, color=INK, labelpad=8)
    ax.set_xlim(ks[0] - 2, ks[-1] + 2)
    ax.set_ylim(0.34, 1.0)
    leg = ax.legend(fontsize=13.5, frameon=False, loc="upper right", handlelength=1.7,
                    labelspacing=0.3, borderpad=0.2)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.text(0.015, 0.975, "3/3 networks at every $k$", transform=ax.transAxes,
            fontsize=13.5, color=INK, fontweight="bold", ha="left", va="top")


def inset_ceiling(ax, raw, ceil):
    r_, c_ = np.median(raw), np.median(ceil)
    ax.barh(0, r_, height=0.5, color=BLUE, zorder=4)
    ax.barh(0, c_ - r_, left=r_, height=0.5, color=PALE, zorder=4)
    ax.barh(0, 1 - c_, left=c_, height=0.5, color=PGREY, zorder=4)
    ax.plot([c_, c_], [-0.25, 0.25], color=BG, lw=2.0, zorder=6)

    ax.text(r_ / 2, 0.40, f"$A_{{\\mathrm{{raw}}}}$ {r_:.3f}", ha="center", va="bottom",
            fontsize=12.5, color=DARK, fontweight="bold")
    ax.text(c_ + 0.015, 0.40, f"ceiling {c_:.3f}", ha="left", va="bottom",
            fontsize=12.5, color=BLUE, fontweight="bold")
    ax.text(0.5, -0.48, "the rest of $p$ lies outside col($H$)", ha="center", va="top",
            fontsize=11.5, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.95, 0.95)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"at $k$ = {KEY}:  prediction vs containment ceiling",
                 fontsize=12.5, color=INK, pad=6, loc="left")
    ax.patch.set_alpha(0.0)


def panel_lambda(ax, ks, lam):
    for w, col, mk in (("obs", BLUE, "o"), ("act", GREY, "s"), ("rand", RED, "D")):
        ax.plot(ks, np.median(lam[w], 0), color=col, lw=2.8, marker=mk, ms=8,
                mec=BG, mew=1.1, zorder=5)
    tidy(ax, "eigenvalue error", fs=15)
    ax.set_xlabel("$k$", fontsize=13.5, color=INK, labelpad=5)
    ax.set_ylabel("$|\\lambda_Q - \\lambda_J|$", fontsize=13.5, color=INK, labelpad=6)
    ax.set_xlim(ks[0] - 2, ks[-1] + 2)
    ax.set_ylim(0, float(np.max(np.median(lam["rand"], 0))) * 1.20)
    ax.text(0.97, 0.94, "lower = better", transform=ax.transAxes, fontsize=11.5,
            color=MUTED, ha="right", va="top")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ks, acond, lam, raw, ceil = load()

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor(BG)
    gsL = fig.add_gridspec(1, 1, left=0.085, right=0.615, top=0.680, bottom=0.135)
    gsR = fig.add_gridspec(1, 1, left=0.725, right=0.968, top=0.680, bottom=0.455)
    axA = fig.add_subplot(gsL[0])
    axL = fig.add_subplot(gsR[0])

    panel_acond(axA, ks, acond)
    axI = axA.inset_axes([0.045, 0.10, 0.54, 0.215])
    inset_ceiling(axI, raw, ceil)
    panel_lambda(axL, ks, lam)

    r_, c_ = np.median(raw), np.median(ceil)
    cd = np.median(acond["obs"][:, ks.index(int(KEY))])
    lc, lr = np.median(lam["obs"], 0), np.median(lam["rand"], 0)

    fig.suptitle("A 20×20 quotient of the wiring predicts the motif directions",
                 fontsize=26, color=INK, x=0.085, ha="left", y=0.975, va="top")
    fig.text(0.085, 0.885,
             "$r=3$ quotient modes  ·  seeds 0–2, per-network traces faint, median bold  ·  "
             "the random quotient gets its own\nmatched partition, group sizes, gains and "
             "mode selection",
             fontsize=13.5, color=MUTED, ha="left", va="top", linespacing=1.5)

    fig.text(0.725, 0.340, f"$A_{{\\mathrm{{raw}}}}$ = {r_:.3f}", fontsize=26,
             color=DARK, fontweight="bold", ha="left", va="top")
    fig.text(0.728, 0.252, "of the whole direction", fontsize=12.5, color=MUTED,
             ha="left", va="top")
    fig.text(0.725, 0.196, f"$A_{{\\mathrm{{cond}}}}$ = {cd:.3f}", fontsize=26,
             color=BLUE, fontweight="bold", ha="left", va="top")
    fig.text(0.728, 0.108, f"of what is reachable  ($k$={KEY}, $r$=3)",
             fontsize=12.5, color=MUTED, ha="left", va="top")
    fig.text(0.968, 0.035,
             f"eigenvalue error {lc.min():.3f}–{lc.max():.3f} connectivity "
             f"vs {lr.min():.3f}–{lr.max():.3f} random",
             fontsize=11.5, color=MUTED, ha="right", va="bottom")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"slide_quotient_decomposition.{ext}", dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT / 'slide_quotient_decomposition.png'}")
    print(f"  A_cond k={KEY}: {cd:.3f}   A_raw {r_:.3f}  ceiling {c_:.3f}")
    print(f"  lambda err conn {lc.min():.3f}-{lc.max():.3f}, rand {lr.min():.3f}-{lr.max():.3f}")


if __name__ == "__main__":
    main()
