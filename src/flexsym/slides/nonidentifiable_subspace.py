"""Slide panel: the labelling is non-identifiable, the subspace it spans is not.

For each network we have R independent WL restarts at the operating k. Three questions:

  A  is the COST reproducible?      coefficient of variation of Gamma_signed across restarts
  B  is the LABELLING reproducible? pairwise adjusted Rand index between restart partitions
  C  is the SUBSPACE reproducible?  mean squared principal-angle cosine between the
                                    group-constant subspaces of restart pairs, against the
                                    same quantity for size-matched random partitions

Principal angles: for orthonormal bases U, V of the two group-constant subspaces
(U = H(H^T H)^{-1/2}), the singular values of U^T V are the cosines of the principal
angles; we summarise a pair by the mean squared cosine (1 = identical subspace).

Data: results/signed_input_wl_paper/restart_ensemble_all_sizes/*.npz
      (falls back to restart_ensemble/ if the all-sizes sweep has not been run)
"""

from __future__ import annotations

from flexsym import paths

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score

REPO = paths.REPO_ROOT
ALL_SIZES = paths.RESULTS_ROOT / "signed_input_wl_paper/restart_ensemble_all_sizes"
FALLBACK = paths.RESULTS_ROOT / "signed_input_wl_paper/restart_ensemble"
OUT = paths.FIGURES_ROOT / "slides"

BG = "#ffffff"
INK = "#111111"
MUTED = "#5b5a57"
GRID = "#e3e2df"
BLUE = "#2a78d6"
GREY = "#9a9a97"
ORANGE = "#eda100"
GREEN = "#008300"
SIZES = [128, 256, 512, 1024]


def tidy(ax, title, fs=16.5):
    ax.set_title(title, fontsize=fs, color=INK, pad=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(1.2)
    ax.tick_params(colors=MUTED, labelsize=12.5, width=1.2, length=5)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)


def basis(labels):
    """Orthonormal basis of the group-constant subspace for a labelling."""
    ks = np.unique(labels)
    H = np.zeros((len(labels), len(ks)))
    for j, c in enumerate(ks):
        H[labels == c, j] = 1.0
    return H / np.sqrt(H.sum(0, keepdims=True))


def mean_sq_cos(a, b):
    """Mean squared principal-angle cosine between two group-constant subspaces.

    np.errstate: matmul on these bases raises a spurious BLAS FPU-flag warning; the
    bases are orthonormal and finite, and the result matches a QR-based basis to 4e-16.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        s = np.linalg.svd(basis(a).T @ basis(b), compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1) ** 2))


def collect(paths):
    """Per-network: gamma CV, mean pairwise restart ARI, restart & random subspace overlap."""
    recs = []
    for p in sorted(paths):
        z = np.load(p, allow_pickle=True)
        L, Rn, g = z["restart_labels"], z["random_labels"], z["restart_gamma"]
        R = len(L)
        pairs = [(i, j) for i in range(R) for j in range(i + 1, R)]
        recs.append({
            "model": str(z["model"]) if "model" in z.files else p.stem,
            "size": int(str(p.stem).split("_")[3]),
            "seed": int(z["seed"]),
            "k": int(z["k"]),
            "gamma_cv": float(g.std(ddof=1) / g.mean()),
            "ari": float(np.mean([adjusted_rand_score(L[i], L[j]) for i, j in pairs])),
            "sub_restart": float(np.mean([mean_sq_cos(L[i], L[j]) for i, j in pairs])),
            "sub_random": float(np.mean([mean_sq_cos(Rn[i], Rn[j]) for i, j in pairs])),
        })
        print(f"  {recs[-1]['model']}: cv={recs[-1]['gamma_cv']:.3f} "
              f"ari={recs[-1]['ari']:.2f} sub={recs[-1]['sub_restart']:.2f} "
              f"(rand {recs[-1]['sub_random']:.2f})", flush=True)
    return recs


def strip(ax, recs, key, color, ylabel, title, ref=None, ref_label=None, pct=False):
    """One metric, per network, split by size."""
    rng = np.random.default_rng(0)
    vals = np.array([r[key] for r in recs]) * (100 if pct else 1)
    size = np.array([r["size"] for r in recs])
    for i, sz in enumerate(SIZES):
        v = vals[size == sz]
        if not len(v):
            continue
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=58, color=color,
                   edgecolor=BG, linewidth=0.9, zorder=4)
        ax.plot([i - 0.27, i + 0.27], [v.mean()] * 2, color=INK, lw=2.6,
                solid_capstyle="butt", zorder=6)
    if ref is not None:
        ax.axhline(ref, color=MUTED, ls="--", lw=1.6, zorder=3)
        ax.text(len(SIZES) - 0.45, ref, f" {ref_label}", fontsize=12, color=MUTED,
                va="center", ha="left")
    tidy(ax, title)
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels([str(s) for s in SIZES], fontsize=12.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=13.5, color=INK, labelpad=6)
    ax.set_xlim(-0.5, len(SIZES) - 0.5)
    return vals


def panel_subspace(ax, recs):
    """Restart vs size-matched-random subspace overlap, paired per network."""
    rng = np.random.default_rng(0)
    a = np.array([r["sub_restart"] for r in recs])
    b = np.array([r["sub_random"] for r in recs])
    jit = rng.uniform(-0.09, 0.09, len(a))
    for i in range(len(a)):
        ax.plot([0 + jit[i], 1 + jit[i]], [a[i], b[i]], color=GRID, lw=1.4, zorder=2)
    ax.scatter(0 + jit, a, s=62, color=GREEN, edgecolor=BG, linewidth=0.9, zorder=4)
    ax.scatter(1 + jit, b, s=62, color=GREY, edgecolor=BG, linewidth=0.9, zorder=4)
    for x, v in ((0, a), (1, b)):
        ax.plot([x - 0.24, x + 0.24], [v.mean()] * 2, color=INK, lw=2.8,
                solid_capstyle="butt", zorder=6)
        ax.text(x + 0.30, v.mean(), f"{v.mean():.2f}", ha="left", va="center",
                fontsize=16, color=INK, fontweight="bold", zorder=7)
    tidy(ax, "…but the subspace is")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["restart\npairs", "size-matched\nrandom pairs"],
                       fontsize=13, color=INK)
    ax.set_ylabel("subspace overlap\n(mean sq. principal-angle cosine)",
                  fontsize=13, color=INK, labelpad=6)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 1.02)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["auto", "all", "256"], default="auto")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    paths = sorted(ALL_SIZES.glob("*_ensemble.npz")) if args.source != "256" else []
    if args.source == "auto" and len(paths) < 33:
        have = {p.name for p in paths}
        paths += [p for p in sorted(FALLBACK.glob("*_ensemble.npz")) if p.name not in have]
    if args.source == "256":
        paths = sorted(FALLBACK.glob("*_ensemble.npz"))
    print(f"{len(paths)} ensembles")
    recs = collect(paths)

    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor(BG)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.05],
                          left=0.075, right=0.978, top=0.735, bottom=0.155, wspace=0.42)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

    cv = strip(axA, recs, "gamma_cv", BLUE, "coefficient of variation of $\\Gamma$  (%)",
               "the cost is reproducible…", pct=True)
    axA.set_ylim(0, max(6.0, cv.max() * 1.2))
    ari = strip(axB, recs, "ari", ORANGE, "pairwise restart ARI",
                "…the labelling is not…", ref=1.0, ref_label="identical")
    axB.set_ylim(0, 1.05)
    a, b = panel_subspace(axC, recs)
    for ax in (axA, axB):
        ax.set_xlabel("network size (units)", fontsize=13.5, color=INK, labelpad=6)

    n = len(recs)
    fig.suptitle("The partition is non-identifiable — but the subspace it spans is stable",
                 fontsize=25, color=INK, x=0.075, ha="left", y=0.975, va="top")
    fig.text(0.075, 0.885,
             f"20 independent restarts per network · {n} networks · "
             "every downstream claim is posed at the subspace level",
             fontsize=14, color=MUTED, ha="left", va="top")
    fig.text(0.978, 0.022, f"restart overlap exceeds random in {int(np.sum(a > b))}/{n} networks",
             ha="right", va="bottom", fontsize=12.5, color=MUTED)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"slide_nonidentifiable_subspace.{ext}", dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"wrote {OUT / 'slide_nonidentifiable_subspace.png'}")
    print(f"gamma CV      {cv.mean():.2f}%   ARI {ari.mean():.2f}   "
          f"subspace {a.mean():.2f} vs random {b.mean():.2f}  "
          f"({int(np.sum(a > b))}/{n} networks)")


if __name__ == "__main__":
    main()
