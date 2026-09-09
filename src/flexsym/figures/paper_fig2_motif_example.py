"""Paper Figure 2 -- one concrete motif: the memory line attractor.

A single (task, epoch) condition, shown three ways:

  A  the recovered slow-point manifold in the top three PCs, with real delay-epoch
     trajectories overlaid -- the line-attractor geometry
  B  the per-neuron loading on the motif's near-unit mode, sorted
  C  the same loadings in connectivity-role order, with the role bands behind them

The loading of neuron i is its component in the Jacobian's real eigenvector nearest
+1 -- the direction along which the memory is held. It is computed per slow point
exactly as `motif_classify.detect_line_attractor` does (`_slow_direction`),
sign-aligned across points, then averaged. Panel B carries an error bar (the spread
over slow points), so it shows how consistent each neuron's loading is along the
manifold; panel C drops the bars and joins the points within each role instead, so
the eye reads the step between roles rather than the within-point scatter. Values
are scaled so the largest mean loading is 1.

Panel C uses the same signed-WL partition, cell order and role colours as Figure 1,
imported from paper_fig1_partition, so a role is the same colour in both figures.

    python -m flexsym.figures.paper_fig2_motif_example
    python -m flexsym.figures.paper_fig2_motif_example --rule delaygo --epoch delay1
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401  (registers 3d proj)

from flexsym import paths
from flexsym.figures.figstyle import INK, MUTED, ORANGE
from flexsym.figures.paper_fig1_partition import (
    K, NET, coarse_grain, load_sweep, npload, role_colors,
)
from flexsym.motifs.motif_classify import _slow_direction

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

#: DMS, delay: a memory epoch whose slow-point set is the cleanest 1-D continuous
#: attractor in this network (participation ratio 3.45, 969 slow points forming a
#: single closed curve) and whose loadings are typically role-structured
#: (eta^2_role = 0.56). It is also not one of the conditions Figure 1 uses.
RULE, EPOCH = "dmsgo", "delay1"

FP_ROOT = paths.RESULTS_ROOT / "motif_fixed_points"
PIECES = paths.REPO_ROOT / "figures" / "fig2" / "pieces"
Q_SLOW = 1e-6                                  # same threshold the classifier uses
MAX_POINTS = 300                               # ...and the same subsample cap
N_TRAJ = 10


# ------------------------------------------------------------------ the motif

def load_motif(net, rule, epoch):
    d = npload(paths.require(FP_ROOT / net / f"{rule}__{epoch}.npz",
                             f"fixed points for {rule}|{epoch}"))
    slow = np.asarray(d["qstar"], dtype=float) < Q_SLOW
    x = np.asarray(d["xstar"], dtype=float)[slow]
    J = np.asarray(d["J_xstar"], dtype=float)[slow]
    if len(x) > MAX_POINTS:
        idx = np.random.default_rng(0).choice(len(x), MAX_POINTS, replace=False)
        x, J = x[idx], J[idx]
    return x, J, str(d["epoch_type"])


def loadings(J, mode="participation"):
    """Per-neuron loading on the motif's near-unit mode: mean and sd over slow points.

    `participation` is |u| averaged over points -- the vector `motif_classify` builds
    and every participation statistic in the paper is computed on. It is the right
    summary here because the slow-point set is a closed curve: the tangent direction
    rotates as you travel around it, so the *signed* eigenvector components average
    towards zero with a large spread, while their magnitudes do not.

    `signed` keeps the sign, aligning each point's eigenvector to the mean. It is only
    interpretable for a straight (non-rotating) line attractor -- kept for that case.
    """
    U = np.array([_slow_direction(*np.linalg.eig(j)) for j in J])
    U[~np.isfinite(U).all(1)] = 0.0

    if mode == "participation":
        U = np.abs(U)
    else:
        ref = U[np.argmax(np.linalg.norm(U, axis=1))]
        for _ in range(2):                      # two passes: seed, then refine on the mean
            U *= np.sign(U @ ref)[:, None]
            ref = U.mean(0)
    # point = the participation entry itself (the mean, as motif_classify defines it);
    # bar = the 25-75% spread over slow points, which is robust to the few points where
    # the eigen-solve picks a different near-unit mode
    lo, hi = np.percentile(U, [25, 75], axis=0)
    return U.mean(0), np.c_[lo, hi].T


def trajectories(net, rule, epoch, n=N_TRAJ):
    cache = npload(paths.require(paths.CACHE_ROOT / f"{net}_b512.npz", "activity cache"))
    key = f"{rule}|{epoch}"
    if key not in cache.files:
        return None
    H = np.asarray(cache[key], dtype=float)      # (n_cond, T, N)
    return H[np.linspace(0, len(H) - 1, min(n, len(H))).astype(int)]


def eta2_role(net, rule, epoch):
    """Share of participation variance explained by connectivity role, if computed."""
    import csv
    path = FP_ROOT / net / "participation_per_motif.csv"
    if not path.exists():
        return None
    for r in csv.DictReader(path.open()):
        if r["key"] == f"{rule}__{epoch}":
            return float(r["eta2_role"])
    return None


# --------------------------------------------------------------------- panels

def panel_geometry(ax, x, H):
    mu = x.mean(0)
    _, _, Vt = np.linalg.svd(x - mu, full_matrices=False)
    B = Vt[:3].T
    P = (x - mu) @ B

    if H is not None:
        for h in H:
            T = (h - mu) @ B
            ax.plot(T[:, 0], T[:, 1], T[:, 2], color=ORANGE, lw=0.8, alpha=0.75, zorder=2)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=7, color=INK, depthshade=False, zorder=3)

    # frame on the slow points, so a trajectory excursion cannot dominate the view
    lim = float(np.abs(P).max()) * 1.12
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)

    ax.set_xlabel("PC1", fontsize=8.5, color=INK, labelpad=-11)
    ax.set_ylabel("PC2", fontsize=8.5, color=INK, labelpad=-11)
    ax.set_zlabel("PC3", fontsize=8.5, color=INK, labelpad=-11)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.tick_params(length=0)
    ax.view_init(elev=22, azim=-58)
    ax.grid(True)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor("white"); pane.pane.set_alpha(0.0)
        pane.pane.set_edgecolor("#d9d8d5")
        pane._axinfo["grid"]["color"] = "#e6e5e2"
        pane._axinfo["grid"]["linewidth"] = 0.6


def _loading_axis(ax, mean, sd, n, xlabel, mode, ylabel=False, bars=True):
    lo = -1.15 if mode == "signed" else -0.04
    ticks = [-1, 0, 1] if mode == "signed" else [0, 0.5, 1]
    labels = ["−1", "0", "1"] if mode == "signed" else ["0", "0.5", "1"]
    xs = np.arange(n)
    if bars:
        ax.vlines(xs, sd[0], sd[1], color=INK, lw=0.55, alpha=0.8, zorder=3)
    ax.scatter(xs, mean, s=5.5, color=INK, zorder=4, linewidths=0)

    ax.set_xlim(-1.5, n + 0.5)
    ax.set_ylim(lo, 1.15)
    ax.set_xticks([0, n - 1]); ax.set_xticklabels(["1", str(n)])
    ax.set_yticks(ticks if ylabel else [])
    if ylabel:
        ax.set_yticklabels(labels)
        ax.set_ylabel("Loading" if mode == "signed" else "Participation",
                      fontsize=9, color=INK, labelpad=1)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED, labelpad=3)
    ax.tick_params(labelsize=8, colors=MUTED, length=2.5, pad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.8)


def panel_sorted(ax, mean, sd, mode):
    o = np.argsort(-mean)
    _loading_axis(ax, mean[o], sd[:, o], len(mean), "Neurons (sorted)", mode, ylabel=True)


def panel_by_role(ax, mean, sd, cells, colors, k, mode):
    """Loadings in role order: points joined within a role, no across-point spread.

    The line is broken at every role boundary, so a segment traces the loadings of
    one role's units and nothing is drawn across the gap between two roles.
    """
    edges = np.r_[0, np.where(np.diff(cells) != 0)[0] + 1, len(cells)]
    for j in range(len(edges) - 1):
        ax.axvspan(edges[j] - 0.5, edges[j + 1] - 0.5,
                   color=colors[cells[edges[j]]], alpha=0.34, lw=0, zorder=0)
        a, b = edges[j], edges[j + 1]
        if b - a > 1:
            ax.plot(np.arange(a, b), mean[a:b], color=INK, lw=0.7, alpha=0.85,
                    solid_capstyle="round", zorder=3)
    _loading_axis(ax, mean, sd, len(mean), f"Roles (1 ... {k})", mode, bars=False)


# ------------------------------------------------------------------ assembling

def save(fig, stem):
    PIECES.mkdir(parents=True, exist_ok=True)
    for ext, kw in ((".svg", {}), (".png", dict(dpi=400))):
        fig.savefig(PIECES / f"{stem}{ext}", bbox_inches="tight", transparent=True, **kw)
    plt.close(fig)
    print(f"wrote {PIECES / stem}.svg / .png")


def combined(x, H, mean, sd, cells, colors, k, mode, stem, eta2=None):
    fig = plt.figure(figsize=(7.4, 2.35))
    ax3 = fig.add_axes([0.005, 0.02, 0.245, 0.96], projection="3d")
    axS = fig.add_axes([0.360, 0.215, 0.245, 0.600])
    axR = fig.add_axes([0.715, 0.215, 0.245, 0.600])

    panel_geometry(ax3, x, H)
    panel_sorted(axS, mean, sd, mode)
    panel_by_role(axR, mean, sd, cells, colors, k, mode)

    what = "Loadings" if mode == "signed" else "Participation"
    fig.text(0.8375, 0.985, f"{what} ordered by", ha="center", va="top",
             fontsize=9.5, color=INK)
    fig.text(0.8375, 0.895, "connectivity role", ha="center", va="top",
             fontsize=9.5, color=INK)
    if eta2 is not None:
        axR.text(0.985, 0.965, f"$\\eta^2_{{role}}$ = {eta2:.2f}", transform=axR.transAxes,
                 ha="right", va="top", fontsize=7.5, color=MUTED)
    save(fig, stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=NET)
    ap.add_argument("--rule", default=RULE)
    ap.add_argument("--epoch", default=EPOCH)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--mode", default="participation", choices=["participation", "signed"],
                    help="|u| averaged over slow points (the paper's participation "
                         "vector), or the sign-aligned signed eigenvector")
    args = ap.parse_args()

    W, labels = load_sweep(args.net, args.k)
    D = coarse_grain(W, labels, "seriate")
    ind_active = np.asarray(
        npload(paths.RESULTS_ROOT / "signed_input_wl_paper" / args.net
               / "transpose" / "signed_input_wl_sweep.npz")["ind_active"], dtype=int)

    x, J, etype = load_motif(args.net, args.rule, args.epoch)
    m, s = loadings(J, args.mode)
    m, s = m[ind_active][D["units"]], s[:, ind_active][:, D["units"]]  # active, role order
    scale = np.abs(m).max() or 1.0
    m, s = m / scale, s / scale

    H = trajectories(args.net, args.rule, args.epoch)
    print(f"{args.net}  {args.rule}|{args.epoch} ({etype})")
    print(f"  {len(x)} slow points, {len(m)} active units, k = {args.k}"
          + (f", {len(H)} trajectories" if H is not None else ", no trajectory cache"))
    e2 = eta2_role(args.net, args.rule, args.epoch)
    print(f"  loading range {m.min():.3f}-{m.max():.3f}, "
          f"median across-point IQR {np.median(s[1] - s[0]):.3f}"
          + (f", eta^2_role = {e2:.3f}" if e2 is not None else ""))

    colors = role_colors(D["k"])
    tag = f"{args.net}_{args.rule}_{args.epoch}_k{args.k}_{args.mode}"
    combined(x, H, m, s, D["cells"], colors, D["k"], args.mode,
             f"fig2_motif_example_{tag}", eta2=eta2_role(args.net, args.rule, args.epoch))

    fig = plt.figure(figsize=(2.2, 2.2))
    panel_geometry(fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d"), x, H)
    save(fig, f"fig2_motif_geometry_{tag}")

    fig = plt.figure(figsize=(2.3, 1.9))
    panel_sorted(fig.add_axes([0.19, 0.21, 0.79, 0.75]), m, s, args.mode)
    save(fig, f"fig2_loadings_sorted_{tag}")

    fig = plt.figure(figsize=(2.3, 1.9))
    panel_by_role(fig.add_axes([0.06, 0.21, 0.92, 0.75]), m, s, D["cells"],
                  colors, D["k"], args.mode)
    save(fig, f"fig2_loadings_by_role_{tag}")


if __name__ == "__main__":
    main()
