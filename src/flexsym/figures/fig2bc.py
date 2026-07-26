"""Fig 2b/c: lumpability scales with size (b) and the delay-period Jacobian (c)."""
from __future__ import annotations

from flexsym import paths
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = paths.REPO_ROOT
OUT = REPO / "results/figures"; OUT.mkdir(parents=True, exist_ok=True)
LUMP = REPO / "results/signed_input_wl_paper/lumpability/lumpability.csv"
JAC = REPO / "results/signed_input_wl_paper/motif_jacobian_lumpability.csv"
C_T = "#2a78d6"; C_W = "#eda100"; C_RAND = "#9a9a97"; INK = "#0b0b0b"; MUTED = "#52514e"; GRID = "#e6e5e2"; BAND = "#eef4fc"


def main():
    lump = list(csv.DictReader(open(LUMP)))
    jac = list(csv.DictReader(open(JAC)))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 4.5))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.80, bottom=0.15, wspace=0.28)
    rng = np.random.default_rng(0)

    # ---- (b) lumpability (-leakage z) by size, transpose vs raw W ----
    sizes = [128, 256, 512, 1024]
    x = np.arange(len(sizes)); w = 0.38
    for off, key, col, lab in [(-w/2-0.01, "transpose_eig_leakage_z", C_T, "recurrent-input matrix"),
                               (+w/2+0.01, "W_eig_leakage_z", C_W, "raw dynamics matrix")]:
        vals = {sz: [-float(r[key]) for r in lump if int(r["size"]) == sz] for sz in sizes}
        means = [np.mean(vals[sz]) for sz in sizes]
        axA.bar(x+off, means, w, color=col, edgecolor="white", linewidth=1.1, zorder=3, label=lab)
        for i, sz in enumerate(sizes):
            p = np.array(vals[sz]); j = (rng.random(p.size)-0.5)*w*0.5
            axA.scatter(x[i]+off+j, p, s=13, color="white", edgecolor=INK, linewidth=0.5, alpha=0.8, zorder=6)
    axA.axhline(0, color=MUTED, lw=1.0)
    axA.text(0.02, 0.03, "chance", transform=axA.transAxes, fontsize=8, color=MUTED)
    axA.set_xticks(x); axA.set_xticklabels([f"{s}" for s in sizes], fontsize=10, color=INK)
    axA.set_xlabel("network size (units)", fontsize=10.5, color=INK)
    axA.set_ylabel("eigenvector lumpability  ($-$leakage $z$ vs chance)", fontsize=10, color=INK)
    axA.set_title("Lumpability strengthens with size", fontsize=11.5, color=INK, pad=8)
    axA.legend(loc="upper left", frameon=False, fontsize=8.8)
    axA.text(0.97, 0.03, "1024: n=3", transform=axA.transAxes, ha="right", fontsize=7.5, color=MUTED)

    # ---- (c) delay-period Jacobian: linear vs Jacobian vs random ----
    lin = np.array([float(r["r2_linear_WT"]) for r in jac])
    jc = np.array([float(r["r2_delay_jacobian"]) for r in jac])
    rd = np.array([float(r["r2_random_partition"]) for r in jac])
    slow_eig = np.mean([float(r["slow_J_eig_abs"]) for r in jac])
    slow_r2 = np.mean([float(r["slow_mode_step_r2"]) for r in jac])
    labs = ["linear\nrecurrent", "delay\nJacobian", "random\npartition"]
    cols = [C_T, "#1baf7a", C_RAND]
    data = [lin, jc, rd]
    xb = np.arange(3)
    axB.axhspan(0, rd.mean(), color=BAND, zorder=0)
    for i, (d, c) in enumerate(zip(data, cols)):
        axB.bar(xb[i], d.mean(), 0.6, color=c, edgecolor="white", linewidth=1.1, zorder=3)
        j = (rng.random(d.size)-0.5)*0.32
        axB.scatter(xb[i]+j, d, s=15, color="white", edgecolor=INK, linewidth=0.5, alpha=0.85, zorder=6)
        axB.text(xb[i], d.mean()+0.03, f"{d.mean():.2f}", ha="center", fontsize=9, color=INK)
    axB.set_xticks(xb); axB.set_xticklabels(labs, fontsize=9.5, color=INK)
    axB.set_ylabel("eigenvector step-$R^2$ on cells", fontsize=10.5, color=INK)
    axB.set_title("The nonlinear memory-motif Jacobian is lumpable", fontsize=11, color=INK, pad=8)
    axB.set_ylim(0, 1.0)
    axB.text(0.03, 0.20, f"slow memory mode:\n$\\lambda\\approx{slow_eig:.2f}$, cell-organized ($R^2={slow_r2:.2f}$)",
             transform=axB.transAxes, fontsize=8.8, color=INK, va="top",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))

    for ax in (axA, axB):
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED)
    fig.suptitle("Connectivity roles are lumpable invariant subspaces of the dynamics",
                 fontsize=12, color=INK, y=0.965)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"fig2bc_lumpability.{ext}", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"wrote {OUT/'fig2bc_lumpability.png'}")


if __name__ == "__main__":
    main()
