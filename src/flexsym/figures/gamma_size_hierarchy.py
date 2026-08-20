"""Figure: gamma is size-weighted -> role structure is a size hierarchy.

Per structural cell (256-unit family, 10 seeds, operating k) decompose the signed
equitability defect into per-group residual energy:

    S_c = sum_{i in c} (||Res_pos_i||^2 + ||Res_neg_i||^2)     (Res = WH - H*quotient)
    gamma^2 total  = sum_c S_c              -> gamma-contribution% of group c
    f_A^2  total   = sum_c S_c / m_c        -> size-neutral f_A-contribution%

Panel A: contribution% vs group size (gamma climbs with size; f_A flat).
Panel B: Lorenz / concentration curve (gamma signal concentrated in a few large
groups; f_A near the equality diagonal).
"""

from __future__ import annotations

from flexsym import paths

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


from flexsym.structural.signed_input_wl_structural_roles import (  # noqa: E402
    apply_matrix_variant, approx_wl_signed, labels_to_H, load_w_rec,
)

REPO = paths.REPO_ROOT
ACT_ROOT = paths.RESULTS_ROOT / "activity_clusters_paper"
ACT_NAME = "epoch_variance_unit_clusters_paper_max.npz"
ORIGINAL = paths.ORIGINAL_STORE
STAGED = paths.STAGED_STORE
OUT = paths.FIGURES_ROOT; OUT.mkdir(parents=True, exist_ok=True)

C_G = "#2a78d6"    # gamma (size-weighted)
C_F = "#eda100"    # f_A (size-neutral)
INK = "#0b0b0b"; MUTED = "#52514e"; GRID = "#e6e5e2"
from flexsym.structural.operating_k import OPK


def resolve(model):
    for s in (ORIGINAL, STAGED):
        if (s / model).exists():
            return s / model
    return None


def per_group_contrib(W_pos, W_neg, labels, k):
    """Return (size_c, gamma_contrib%, fA_contrib%) per non-empty group.

    Correct decomposition is by residual COLUMN (cell) energy:
        gamma^2 = sum_c ||R[:,c]||^2 = sum_c m_c * e_c   (col energy carries the m_c weight)
        f_A^2   = sum_c e_c = sum_c ||R[:,c]||^2 / m_c   (size-neutralized)
    """
    H = labels_to_H(labels, k)
    sizes = H.sum(axis=0)
    col_e = np.zeros(k)
    for W in (W_pos, W_neg):
        WH = W @ H
        D_inv = 1.0 / np.maximum(sizes, 1e-12)
        cell_means = (H.T @ WH) * D_inv[:, None]
        Res = WH - H @ cell_means
        col_e += (Res ** 2).sum(axis=0)  # energy per CELL (column)
    keep = sizes > 0
    m = sizes[keep]
    e = col_e[keep]                       # e here = m_c * e_c  (gamma column energy)
    g = 100 * e / e.sum()                 # gamma-weighted contribution
    fa_raw = e / m                        # f_A: neutralize size
    fa = 100 * fa_raw / fa_raw.sum()
    return m, g, fa


def collect():
    sizes, gc, fc = [], [], []
    for seed, k in OPK.items():
        model = f"LeakyRNN_softplus_diag_256_all_{seed}"
        md = resolve(model); act = ACT_ROOT / model / ACT_NAME
        if md is None or not act.exists():
            continue
        ind = np.asarray(np.load(act, allow_pickle=True)["ind_active"], dtype=np.int64)
        W = apply_matrix_variant(load_w_rec(md), "transpose")[np.ix_(ind, ind)]
        labels, _ = approx_wl_signed(np.maximum(W, 0), np.maximum(-W, 0), k, n_iter=20, seed=0, n_init=20)
        m, g, fa = per_group_contrib(np.maximum(W, 0), np.maximum(-W, 0), labels, k)
        sizes += list(m); gc += list(g); fc += list(fa)
    return np.array(sizes), np.array(gc), np.array(fc)


def lorenz(contrib):
    x = np.sort(contrib)[::-1]
    cum = np.cumsum(x) / x.sum()
    frac_groups = np.arange(1, len(x) + 1) / len(x)
    return np.concatenate([[0], frac_groups]), np.concatenate([[0], cum])


def gini(x):
    x = np.sort(x); n = len(x); cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def main():
    sizes, gc, fc = collect()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 4.4))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.16, wspace=0.28)

    # ---- Panel A: contribution vs size ----
    rg = pearsonr(sizes, gc)[0]
    rf = pearsonr(sizes, fc)[0]
    axA.scatter(sizes, gc, s=16, color=C_G, alpha=0.45, edgecolor="none", label="γ (size-weighted)")
    axA.scatter(sizes, fc, s=16, color=C_F, alpha=0.45, edgecolor="none", label="f$_A$ (size-neutral)")
    for x, y, c in [(sizes, gc, C_G), (sizes, fc, C_F)]:
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        axA.plot(xs, b0 + b1 * xs, color=c, linewidth=2, zorder=5)
    axA.set_xlabel("group size (units)", fontsize=10.5, color=INK)
    axA.set_ylabel("% of total equitability defect", fontsize=10.5, color=INK)
    axA.set_title("γ over-weights large groups", fontsize=11.5, color=INK, pad=8)
    axA.text(0.04, 0.94, f"γ:  r = {rg:+.2f}\nf$_A$: r = {rf:+.2f}", transform=axA.transAxes,
             fontsize=10, va="top", color=INK,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))
    axA.legend(loc="upper right", frameon=False, fontsize=9)

    # ---- Panel B: Lorenz concentration ----
    gx, gy = lorenz(gc); fx, fy = lorenz(fc)
    axB.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1.2, label="even (no hierarchy)")
    axB.plot(gx, gy, color=C_G, linewidth=2.4, label="γ (size-weighted)")
    axB.plot(fx, fy, color=C_F, linewidth=2.4, label="f$_A$ (size-neutral)")
    # top-k annotation: fraction of groups carrying 50% of gamma
    k50 = gx[np.searchsorted(gy, 0.5)]
    axB.set_xlabel("fraction of groups (largest-contributing first)", fontsize=10.5, color=INK)
    axB.set_ylabel("cumulative % of signal", fontsize=10.5, color=INK)
    axB.set_title("→ signal concentrated in a few large roles", fontsize=11.5, color=INK, pad=8)
    axB.text(0.045, 0.955,
             f"Gini  γ = {gini(gc):.2f},  f$_A$ = {gini(fc):.2f}\n"
             f"top {k50*100:.0f}% of groups → 50% of γ",
             transform=axB.transAxes, fontsize=9.5, color=INK, va="top",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))
    axB.set_xlim(0, 1); axB.set_ylim(0, 1.02)
    axB.legend(loc="lower right", frameon=False, fontsize=9)

    for ax in (axA, axB):
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED)

    fig.suptitle("Super-equitable structure is real but organized as a size hierarchy\n"
                 "(256-unit, 10 seeds, operating k)", fontsize=12, color=INK, y=0.985)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"gamma_size_hierarchy.{ext}", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"wrote {OUT/'gamma_size_hierarchy.png'}")
    print(f"n cells={len(sizes)}  γ~size r={pearsonr(sizes,gc)[0]:+.3f}  fA~size r={pearsonr(sizes,fc)[0]:+.3f}")
    print(f"Gini γ={gini(gc):.3f} fA={gini(fc):.3f}  top {lorenz(gc)[0][np.searchsorted(lorenz(gc)[1],0.5)]*100:.0f}% groups -> 50% γ")


if __name__ == "__main__":
    main()
