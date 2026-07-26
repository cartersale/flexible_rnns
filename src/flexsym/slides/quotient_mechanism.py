"""Slide: what the quotient buys, how many modes it needs, and what carries the geometry.

A  compression -- alignment against the number of free parameters in the predictor.
   The quotient (k^2 + k = 420 numbers) is compared with three full-rank predictors
   built from the whole n x n matrix (41,616 numbers).
B  mode saturation -- A_cond as r grows. Connectivity saturates; the random quotient
   keeps climbing because more modes simply fill more of the k-dimensional subspace.
C  mechanism -- ablations of the conditioning gains. Structure without gains still
   beats random; gains without structure lose to it.

k = 20, seeds 0-2, median over networks.
Panel A reports A_raw (absolute); panels B and C report A_cond (share of the reachable
part), which is the metric the ablations and the saturation argument are posed in.
Data: results/motif_fixed_points/quotient_controls.json
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
CTRL = REPO / "results/motif_fixed_points/quotient_controls.json"
OUT = REPO / "results/figures/slides"

BG = "#ffffff"
INK = "#111111"
MUTED = "#5b5a57"
GRID = "#e3e2df"
BLUE = "#2a78d6"
PALE = "#a9cbf2"
MID = "#6ea8e8"
GREY = "#9a9a97"
RED = "#e34948"
ORANGE = "#eda100"
GREEN = "#008300"
DARK = "#0f3f73"
K = "20"
NPAR_Q = 20 * 20 + 20          # quotient entries + one gain per group
NPAR_FULL = 204 ** 2           # canonical network: 204 active units


def tidy(ax, title, fs=16):
    ax.set_title(title, fontsize=fs, color=INK, pad=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(1.2)
    ax.tick_params(colors=MUTED, labelsize=12, width=1.2, length=5)
    ax.set_axisbelow(True)


def load():
    d = json.load(open(CTRL))
    nets = sorted(n for n in d if n.startswith("LeakyRNN"))
    med = lambda f: float(np.median([f(d[n]) for n in nets]))
    A = {
        "quotient": med(lambda x: x["per_k"][K]["Q_pred|r3"]["raw"]["obs"]),
        "full_J": med(lambda x: x["full_space"]["full_J|r3"]),
        "full_W": med(lambda x: x["full_space"]["full_W|r3"]),
        "svd": med(lambda x: x["full_space"]["W_svd|r3"]),
        "randdir": med(lambda x: x["per_k"][K]["rand_dir|r3"]["raw"]["obs"]),
        "ceiling": med(lambda x: x["per_k"][K]["containment"]["raw"]["obs"]),
    }
    rs = list(range(1, 11))
    B = {
        "conn": np.array([med(lambda x, r=r: x["per_k"][K][f"Q_pred|r{r}"]["cond"]["obs"])
                          for r in rs]),
        "rand": np.array([med(lambda x, r=r: x["per_k"][K][f"Q_pred|r{r}"]["cond"]["rand"])
                          for r in rs]),
    }
    C = {v: med(lambda x, v=v: x["per_k"][K][f"{v}|r3"]["cond"]["obs"])
         for v in ("Q_pred", "scalar_gain", "no_gain", "gain_only")}
    C["random"] = med(lambda x: x["per_k"][K]["Q_pred|r3"]["cond"]["rand"])
    return A, rs, B, C


def panel_compression(ax, A):
    pts = [("full Jacobian", NPAR_FULL, A["full_J"], INK, "s"),
           ("full $W_{\\rm rec}$", NPAR_FULL, A["full_W"], ORANGE, "^"),
           ("$W_{\\rm rec}$ SVD", NPAR_FULL, A["svd"], GREEN, "v")]
    ax.axhline(A["ceiling"], color=BLUE, ls=":", lw=1.8, zorder=2)
    ax.text(3.2e4, A["ceiling"] + 0.012, "containment ceiling", fontsize=11,
            color=BLUE, ha="right", va="bottom")
    ax.axhline(A["randdir"], color=GREY, ls="--", lw=1.6, zorder=2)
    ax.text(3.2e4, A["randdir"] + 0.012, "random directions in col($H$)", fontsize=11,
            color=MUTED, ha="right", va="bottom")

    for lab, x, y, c, mk in pts:
        ax.scatter([x], [y], s=150, color=c, marker=mk, edgecolor=BG, linewidth=1.2,
                   zorder=5)
        ax.text(x * 0.80, y, f"{lab}  {y:.3f}", fontsize=12, color=c, ha="right",
                va="center", fontweight="bold")
    ax.scatter([NPAR_Q], [A["quotient"]], s=320, color=BLUE, marker="o",
               edgecolor=INK, linewidth=2.0, zorder=7)
    ax.annotate(f"quotient\n{A['quotient']:.3f}", xy=(NPAR_Q, A["quotient"]),
                xytext=(NPAR_Q * 1.5, A["quotient"] - 0.135), fontsize=13.5,
                color=BLUE, fontweight="bold", ha="left", va="top", linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.4))

    tidy(ax, "1% of the parameters")
    ax.set_xscale("log")
    ax.set_xlim(180, 1.3e5)
    ax.set_ylim(0.14, 0.90)
    ax.set_xlabel("free parameters in the predictor", fontsize=13.5, color=INK,
                  labelpad=6)
    ax.set_ylabel("$A_{\\mathrm{raw}}$   ($r = 3$)", fontsize=13.5, color=INK, labelpad=6)
    ax.set_xticks([1e3, 1e4, 1e5])
    ax.grid(True, color=GRID, linewidth=1.0)
    ax.text(0.97, 0.055, f"{NPAR_Q:,} vs {NPAR_FULL:,} parameters",
            transform=ax.transAxes, fontsize=12, color=MUTED, ha="right", va="bottom")


def panel_saturation(ax, rs, B):
    ax.plot(rs, B["conn"], color=BLUE, lw=3.0, marker="o", ms=8, mec=BG, mew=1.1,
            zorder=6, label="connectivity")
    ax.plot(rs, B["rand"], color=RED, lw=3.0, marker="D", ms=7.5, mec=BG, mew=1.1,
            zorder=5, label="size-matched random")
    ax.axvline(3, color=MUTED, ls="--", lw=1.6, zorder=3)
    ax.text(3.25, 0.975, "$r = 3$", transform=ax.get_xaxis_transform(),
            fontsize=13, color=MUTED, ha="left", va="top")
    ax.annotate("", xy=(10, B["conn"][-1]), xytext=(10, B["rand"][-1]),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.4))

    tidy(ax, "saturates at $r = 3$")
    ax.set_xlabel("number of quotient modes  $r$", fontsize=13.5, color=INK, labelpad=6)
    ax.set_ylabel("$A_{\\mathrm{cond}}$", fontsize=13.5, color=INK, labelpad=6)
    ax.set_xlim(0.5, 10.9)
    ax.set_ylim(0.35, 1.0)
    ax.set_xticks([1, 3, 5, 10])
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    leg = ax.legend(fontsize=12.5, frameon=False, loc="lower right", handlelength=1.6,
                    labelspacing=0.3, borderpad=0.2)
    for t in leg.get_texts():
        t.set_color(INK)


def panel_mechanism(ax, C):
    rows = [("full gains", C["Q_pred"], BLUE),
            ("one global\nscalar gain", C["scalar_gain"], MID),
            ("no gains\n(quotient only)", C["no_gain"], PALE),
            ("gains only\n(no quotient)", C["gain_only"], GREY)]
    y = np.arange(len(rows))[::-1]
    for (lab, v, col), yi in zip(rows, y):
        ax.barh(yi, v, height=0.62, color=col, zorder=4)
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", ha="left", fontsize=13,
                color=INK, fontweight="bold")
        ax.text(-0.02, yi, lab, va="center", ha="right", fontsize=12.5, color=INK,
                linespacing=1.25)
    ax.axvline(C["random"], color=RED, ls="--", lw=2.0, zorder=6)
    ax.text(C["random"], -0.62, f"random {C['random']:.3f}", fontsize=12.5, color=RED,
            ha="center", va="top", fontweight="bold")

    tidy(ax, "structure, not gains")
    ax.set_yticks([])
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.7, len(rows) - 0.25)
    ax.set_xlabel("$A_{\\mathrm{cond}}$", fontsize=13.5, color=INK, labelpad=6)
    ax.xaxis.grid(True, color=GRID, linewidth=1.0)
    for s in ("left",):
        ax.spines[s].set_visible(False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    A, rs, B, C = load()

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 0.92, 1.05],
                          left=0.062, right=0.978, top=0.615, bottom=0.145, wspace=0.42)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))
    panel_compression(axA, A)
    panel_saturation(axB, rs, B)
    panel_mechanism(axC, C)

    fig.suptitle("A tiny coarse-graining, three modes, and the structure — not the gains",
                 fontsize=25, color=INK, x=0.062, ha="left", y=0.975, va="top")
    fig.text(0.062, 0.885,
             "$k=20$, seeds 0–2, median over networks  ·  panel A reports $A_{\\rm raw}$ "
             "(absolute); B and C report $A_{\\rm cond}$ (share of the reachable part)",
             fontsize=13, color=MUTED, ha="left", va="top")

    fig.text(0.062, 0.055,
             "beats the full connectivity matrix's\nown top modes outright",
             fontsize=12.5, color=MUTED, ha="left", va="top", linespacing=1.4)
    fig.text(0.435, 0.055,
             "more modes stop helping once\nthe geometry is captured",
             fontsize=12.5, color=MUTED, ha="left", va="top", linespacing=1.4)
    fig.text(0.978, 0.055,
             "gains without the quotient lose\nto a random partition",
             fontsize=12.5, color=MUTED, ha="right", va="top", linespacing=1.4)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"slide_quotient_mechanism.{ext}", dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT / 'slide_quotient_mechanism.png'}")
    print(f"  A: quotient {A['quotient']:.3f} ({NPAR_Q} par) vs full_J {A['full_J']:.3f} "
          f"({NPAR_FULL} par) -> {A['quotient']/A['full_J']*100:.0f}%; "
          f"full_W {A['full_W']:.3f}, svd {A['svd']:.3f}")
    print(f"  B: conn {B['conn'][0]:.3f}->{B['conn'][2]:.3f}->{B['conn'][-1]:.3f}, "
          f"rand {B['rand'][0]:.3f}->{B['rand'][-1]:.3f}")
    print(f"  C: {', '.join(f'{k} {v:.3f}' for k, v in C.items())}")


if __name__ == "__main__":
    main()
