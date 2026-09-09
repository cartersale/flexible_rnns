"""Paper Figure 1 -- the coarse-graining panel.

One trained W_rec, reordered by the inferred structural partition H, next to the
reduced operator it induces.

  left   W (transpose variant, active units) with rows/columns sorted by cell,
         thin rules at the cell boundaries
  right  the quotient Q_struct = D^-1 H' W H -- the k x k operator that acts on
         cell-mean activity

Nothing is redrawn or idealised: W and the labels come straight out of the cached
signed-WL sweep, and Q is computed from them with the same `quotient_matrix` the
analyses use. Cell order is cosmetic only (correlation seriation of the quotient
row/column profiles); it does not change W, H or Q.

Writes, into figures/fig1/pieces/, both as .svg and .png:

  fig1_wrec_partitioned_<net>_k<k>      left piece alone
  fig1_quotient_<net>_k<k>              right piece alone (k x k, own scale)
  fig1_quotient_sized_<net>_k<k>        alternative right piece: the same
                                        coarse-graining drawn with cell widths
                                        proportional to cell size and on W's
                                        colour scale, i.e. literally the left
                                        panel averaged within blocks
  fig1_partition_quotient_<net>_k<k>    the two assembled side by side

    python -m flexsym.figures.paper_fig1_partition
    python -m flexsym.figures.paper_fig1_partition --k 41 --order size
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from flexsym import paths
from flexsym.figures.figstyle import MUTED, INK
from flexsym.run.lumpability_r2_ksweep import TOP_N, leading_eigvecs, step_r2
from flexsym.structural.signed_input_wl_structural_roles import (
    labels_to_H,
    quotient_matrix,
)

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

#: representative network: released Driscoll seed, 256 units, 204 of them active.
#: Main-cohort example network (the former 256 seed 0 default is a released
#: Driscoll network and is not in RESULTS_ROOT under the main cohort).
NET = "LeakyRNN_softplus_diag_256_all_5"
#: display scale. k = 20 sits on the plateau of the R^2(k) curve for this network
#: (r2_best 0.790 vs 0.134 for size-matched random partitions, z = +29) while
#: still giving ~10 units per cell, so the blocks are wide enough to read.
K = 20

WL_ROOT = paths.RESULTS_ROOT / "signed_input_wl_paper"
PIECES = paths.REPO_ROOT / "figures" / "fig1" / "pieces"

CMAP = "RdBu_r"
#: symmetric colour limits are set at a percentile of |value| rather than the max:
#: W_rec is inhibition-dominated with a long tail, and at the 99th percentile the
#: block structure washes out. The 95th saturates ~5% of entries and is stated in
#: the caption.
CLIP_W = 95.0
CLIP_Q = 99.0
RULE = "#ffffff"       # cell-boundary rules inside the matrix
ROLE_CMAP = "turbo"    # ramp the role swatches so role order reads as an order


# ------------------------------------------------------------------- load data

def npload(path):
    try:
        return np.load(path, allow_pickle=True)
    except TypeError:      # flexsym.core.task monkeypatches np.load
        return np.load(path)


def load_sweep(net: str, k: int):
    """W (transpose variant, active units) and the k-cell structural labels."""
    path = paths.require(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz",
                         f"signed-WL sweep for {net}")
    d = npload(path)
    W = np.asarray(d["W_rec"], dtype=float)
    ks = np.array([int(np.asarray(r[0])) for r in d["rows"]])
    hit = np.where(ks == k)[0]
    if not len(hit):
        raise SystemExit(f"k={k} not in the cached sweep (k = {ks.min()}..{ks.max()})")
    labels = np.asarray(d["labels_struct_by_k"][int(hit[0])], dtype=int)
    return W, labels


# ---------------------------------------------------------------- coarse-grain

def cell_order(Q: np.ndarray, sizes: np.ndarray, how: str) -> np.ndarray:
    """Cosmetic ordering of the k cells. Does not touch H or Q themselves."""
    if how == "size":
        return np.argsort(-sizes)
    if how == "diag":
        return np.argsort(-np.diag(Q))
    if how == "seriate":
        # cluster cells by their joint outgoing/incoming quotient profile so that
        # cells with similar connectivity end up adjacent
        F = np.c_[Q, Q.T]
        return leaves_list(linkage(pdist(F, "correlation"), "average"))
    raise ValueError(f"unknown ordering {how!r}")


def coarse_grain(W: np.ndarray, labels: np.ndarray, how: str):
    k = int(labels.max()) + 1
    H = labels_to_H(labels, k)
    Q = quotient_matrix(W, H)                     # D^-1 H' W H
    sizes = H.sum(axis=0).astype(int)

    order = cell_order(Q, sizes, how)
    rank = np.empty(k, dtype=int)
    rank[order] = np.arange(k)
    cells = rank[labels]                          # relabelled cell id per unit
    units = np.argsort(cells, kind="stable")      # units grouped by cell

    Wo = W[np.ix_(units, units)]
    Qo = Q[np.ix_(order, order)]
    # block means of W, on W's own scale (the sized view of the same reduction)
    Mo = Qo / sizes[order][None, :]
    return dict(W=Wo, Q=Qo, M=Mo, cells=cells[units], units=units,
                sizes=sizes[order], k=k)


# ------------------------------------------------------------------- primitives

def _frame(ax, lw=0.7):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_linewidth(lw)


def _cbar(im, ax, v, label, orientation="horizontal", fmt="{:.1f}"):
    cb = plt.colorbar(im, cax=ax, orientation=orientation)
    cb.set_ticks([-v, 0, v])
    cb.set_ticklabels(["−" + fmt.format(v), "0", fmt.format(v)])
    cb.ax.tick_params(labelsize=6.5, colors=MUTED, length=2, pad=1.5)
    cb.outline.set_visible(False)
    if label:
        cb.set_label(label, fontsize=6.5, color=MUTED, labelpad=2)
    return cb


def role_colors(k: int) -> np.ndarray:
    """One colour per structural role, ramped so role order reads as an order."""
    return plt.get_cmap(ROLE_CMAP)(np.linspace(0.03, 0.97, k))


def draw_ribbon(ax, cells, colors, orient="h"):
    """Per-unit role strip along an edge of the matrix; widths = cell sizes."""
    strip = colors[cells][None, :, :] if orient == "h" else colors[cells][:, None, :]
    ax.imshow(strip, aspect="auto", interpolation="nearest")
    for b in np.where(np.diff(cells) != 0)[0] + 0.5:
        (ax.axvline if orient == "h" else ax.axhline)(b, color="white", lw=0.5)
    _frame(ax, lw=0.5)


def draw_wrec(ax, D, vmax, cbar_ax=None, cbar_orientation="horizontal"):
    """W reordered by the partition, with rules at the cell boundaries."""
    im = ax.imshow(D["W"], cmap=CMAP, vmin=-vmax, vmax=vmax, interpolation="nearest")
    for b in np.where(np.diff(D["cells"]) != 0)[0] + 0.5:
        ax.axhline(b, color=RULE, lw=0.35, alpha=0.65)
        ax.axvline(b, color=RULE, lw=0.35, alpha=0.65)
    _frame(ax)
    if cbar_ax is not None:
        _cbar(im, cbar_ax, vmax, "weight", cbar_orientation)
    return im


def draw_role_legend(ax, colors, k, n_show=3):
    """Role 1, Role 2, Role 3, ..., Role k -- as in the reference layout."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    entries = [(i, f"Role {i + 1}") for i in range(min(n_show, k))]
    if k > n_show + 1:
        entries.append((None, "..."))
    if k > n_show:
        entries.append((k - 1, f"Role {k}"))

    slot = 1.0 / len(entries)
    for j, (i, text) in enumerate(entries):
        x = j * slot + slot / 2
        if i is None:
            ax.text(x, 0.5, text, ha="center", va="center",
                    fontsize=9, color=MUTED)
            continue
        ax.add_patch(Rectangle((x - 0.062, 0.30), 0.032, 0.40,
                               facecolor=colors[i], edgecolor="white", lw=0.6,
                               transform=ax.transAxes, clip_on=False))
        ax.text(x - 0.022, 0.5, text, ha="left", va="center",
                fontsize=8, color=INK)


def draw_quotient(ax, D, cbar_ax=None):
    """The k x k reduced operator, one pixel per (cell, cell) pair."""
    v = float(np.percentile(np.abs(D["Q"]), CLIP_Q))
    im = ax.imshow(D["Q"], cmap=CMAP, vmin=-v, vmax=v, interpolation="nearest")
    for b in np.arange(0.5, D["k"] - 1):
        ax.axhline(b, color=RULE, lw=0.5, alpha=0.9)
        ax.axvline(b, color=RULE, lw=0.5, alpha=0.9)
    _frame(ax)
    if cbar_ax is not None:
        _cbar(im, cbar_ax, v, "cell-to-cell weight")
    return im


def draw_quotient_sized(ax, D, cbar_ax=None):
    """Same reduction, cell widths proportional to cell size.

    This is the block-mean of the left panel: each tile is the mean weight from
    one cell to another, drawn over exactly the rows and columns it averages, so
    its boundaries line up with the rules in the left panel. Block means are much
    smaller than individual weights, so this gets its own colour limit.
    """
    edges = np.r_[0, np.cumsum(D["sizes"])]
    vmax = float(np.percentile(np.abs(D["M"]), CLIP_Q))
    im = ax.pcolormesh(edges, edges, D["M"], cmap=CMAP, vmin=-vmax, vmax=vmax,
                       edgecolors=RULE, linewidth=0.35)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(edges[-1], edges[0])
    ax.set_aspect("equal")
    _frame(ax)
    if cbar_ax is not None:
        _cbar(im, cbar_ax, vmax, "mean weight")
    return im


# ------------------------------------------------------------------- assembling

def save(fig, stem):
    PIECES.mkdir(parents=True, exist_ok=True)
    for ext, kw in ((".svg", {}), (".png", dict(dpi=400))):
        fig.savefig(PIECES / f"{stem}{ext}", bbox_inches="tight",
                    facecolor="none", transparent=True, **kw)
    plt.close(fig)
    print(f"wrote {PIECES / stem}.svg / .png")


def piece_wrec(D, vmax, stem):
    """The left piece on its own, with its role ribbons."""
    colors = role_colors(D["k"])
    fig = plt.figure(figsize=(2.9, 3.15))
    x0, y0, s, rib, gap = 0.075, 0.155, 0.855, 0.030, 0.012
    ax = fig.add_axes([x0, y0, s, s * 2.9 / 3.15])
    draw_wrec(ax, D, vmax)
    draw_ribbon(fig.add_axes([x0, y0 + s * 2.9 / 3.15 + gap, s, rib]),
                D["cells"], colors, "h")
    draw_ribbon(fig.add_axes([x0 - rib * 2.9 / 3.15 - gap, y0,
                              rib * 2.9 / 3.15, s * 2.9 / 3.15]),
                D["cells"], colors, "v")
    _cbar(ax.images[0], fig.add_axes([x0 + 0.22, 0.045, 0.42, 0.028]),
          vmax, "weight")
    save(fig, stem)


def piece_quotient(D, stem):
    fig = plt.figure(figsize=(2.0, 2.25))
    ax = fig.add_axes([0.0, 0.16, 1.0, 0.84])
    cax = fig.add_axes([0.24, 0.05, 0.52, 0.038])
    draw_quotient(ax, D, cax)
    save(fig, stem)


def piece_quotient_sized(D, stem):
    fig = plt.figure(figsize=(2.0, 2.25))
    ax = fig.add_axes([0.0, 0.16, 1.0, 0.84])
    cax = fig.add_axes([0.24, 0.05, 0.52, 0.038])
    draw_quotient_sized(ax, D, cax)
    save(fig, stem)


# ------------------------------------------------- eigenvector / step-fit panel

def step_fit(W: np.ndarray, D: dict, top_n: int = TOP_N, pick: str = "best"):
    """A leading eigenvector of W in role order, and its per-cell mean.

    `leading_eigvecs` and `step_r2` are imported from run/lumpability_r2_ksweep,
    so the panel shows one of the same leading real spectral directions the sweep
    scores. The sweep's reported statistic is the block R^2 (step_r2_block); the
    per-vector R^2 annotated here is the one-dimensional case of it, and for a
    column of a complex block it is a display value only, since the individual
    columns turn with the eigenvector's phase.
    """
    w = np.linalg.eigvals(W)
    order = np.argsort(-np.abs(w))[:top_n]
    V = leading_eigvecs(W, W.shape[0])[:top_n]

    cells, units = D["cells"], D["units"]
    r2 = [step_r2(v[units], cells) for v in V]
    i = int(np.argmax(r2)) if pick == "best" else int(pick)

    v = V[i][units]
    v = v * np.sign(v[np.argmax(np.abs(v))])       # sign is arbitrary; fix it
    v = v / np.abs(v).max()                        # so the axis reads 1 / -1

    q = np.zeros_like(v)
    for c in np.unique(cells):
        m = cells == c
        q[m] = v[m].mean()
    return dict(v=v, q=q, r2=r2[i], rank=i, lam=w[order[i]], all_r2=np.array(r2))


def _role_bands(ax, cells, colors, alpha=0.34):
    """Shade the background by role, as wide as each cell."""
    edges = np.r_[0, np.where(np.diff(cells) != 0)[0] + 1, len(cells)]
    for j in range(len(edges) - 1):
        ax.axvspan(edges[j] - 0.5, edges[j + 1] - 0.5,
                   color=colors[cells[edges[j]]], alpha=alpha, lw=0, zorder=0)


def draw_eigvec(ax, y, cells, colors, ylabels=False, n=None):
    _role_bands(ax, cells, colors)
    x = np.arange(len(y))
    ax.plot(x, y, color=INK, lw=0.7, zorder=3)
    ax.scatter(x, y, s=5.5, color=INK, zorder=4, linewidths=0)

    ax.set_xlim(-0.5, len(y) - 0.5)
    ax.set_ylim(-1.12, 1.12)
    ax.set_xticks([0, len(y) - 1]); ax.set_xticklabels(["1", str(n or len(y))])
    ax.set_yticks([-1, 0, 1] if ylabels else [])
    if ylabels:
        ax.set_yticklabels(["−1", "0", "1"])
    ax.tick_params(labelsize=8, colors=MUTED, length=2.5, pad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.8)


def eigvec_panel(W, D, k, stem, top_n=TOP_N, pick="best"):
    """Eigenvector v ~= its best piecewise-constant approximation Hq."""
    F = step_fit(W, D, top_n, pick)
    colors = role_colors(k)
    cells = D["cells"]

    fig = plt.figure(figsize=(6.4, 2.5))
    axL = fig.add_axes([0.085, 0.215, 0.375, 0.560])
    axR = fig.add_axes([0.595, 0.215, 0.375, 0.560])
    draw_eigvec(axL, F["v"], cells, colors, ylabels=True)
    draw_eigvec(axR, F["q"], cells, colors, ylabels=False)

    fig.text(0.2725, 0.855, "Eigenvector $\\mathbf{v}$", ha="center", va="bottom",
             fontsize=10, color=INK)
    fig.text(0.7825, 0.925, "Best piecewise-constant", ha="center", va="bottom",
             fontsize=10, color=INK)
    fig.text(0.7825, 0.855, "approximation $H\\mathbf{q}$", ha="center", va="bottom",
             fontsize=10, color=INK)
    fig.text(0.5275, 0.470, "≈", ha="center", va="center", fontsize=22, color=INK)
    fig.text(0.2725, 0.045, "Neurons (ordered by role)", ha="center", va="bottom",
             fontsize=9, color=MUTED)
    fig.text(0.7825, 0.045,
             f"step-$R^2$ = {F['r2']:.2f}   ($k$ = {k}, $\\lambda$ = {F['lam'].real:.1f}"
             + (f"{F['lam'].imag:+.1f}i" if abs(F["lam"].imag) > 1e-9 else "") + ")",
             ha="center", va="bottom", fontsize=8, color=MUTED)

    save(fig, stem)
    print(f"  eigenvector rank {F['rank'] + 1} of top {top_n} by |lambda|: "
          f"step-R2 = {F['r2']:.3f} (top-{top_n} range "
          f"{F['all_r2'].min():.3f}-{F['all_r2'].max():.3f})")


def combined(D, vmax, net, k, stem, legend=True):
    """W with role ribbons -> projection by H -> the k x k quotient."""
    n = D["W"].shape[0]
    colors = role_colors(k)

    W_IN, H_IN = 6.4, 3.5
    fig = plt.figure(figsize=(W_IN, H_IN))
    ar = W_IN / H_IN                      # multiply x-fractions to get square axes

    leg_h = 0.135 if legend else 0.0
    # --- left: ribbons + matrix -------------------------------------------
    rib = 0.026                            # ribbon thickness, figure fraction of H
    mh = 0.755 - leg_h * 0.55              # matrix height
    mw = mh / ar                           # ...as a width, so the axes is square
    x0, y0 = 0.075, leg_h + 0.045
    axW = fig.add_axes([x0, y0, mw, mh])
    draw_wrec(axW, D, vmax)
    draw_ribbon(fig.add_axes([x0, y0 + mh + 0.012, mw, rib]), D["cells"], colors, "h")
    draw_ribbon(fig.add_axes([x0 - rib / ar - 0.010, y0, rib / ar, mh]),
                D["cells"], colors, "v")

    # --- middle: shared weight scale, then the projection arrow -----------
    cW = fig.add_axes([0.462, y0 + 0.30 * mh, 0.016, 0.34 * mh])
    _cbar(axW.images[0], cW, vmax, "", "vertical")
    arrow = FancyArrowPatch((0.556, y0 + 0.47 * mh), (0.622, y0 + 0.47 * mh),
                            transform=fig.transFigure,
                            arrowstyle="-|>,head_length=6,head_width=3.6",
                            lw=3.0, color=INK, shrinkA=0, shrinkB=0)
    fig.add_artist(arrow)
    fig.text(0.592, y0 + 0.635 * mh, "Projection", ha="center", va="center",
             fontsize=9.5, color=INK)
    fig.text(0.592, y0 + 0.560 * mh, "($H$)", ha="center", va="center",
             fontsize=9.5, color=INK)

    # --- right: the quotient ----------------------------------------------
    qh = 0.68 * mh
    axQ = fig.add_axes([0.660, y0 + 0.13 * mh, qh / ar, qh])
    draw_quotient(axQ, D)
    cQ = fig.add_axes([0.660 + 0.06 * qh / ar, y0 + 0.13 * mh - 0.055,
                       0.88 * qh / ar, 0.020])
    _cbar(axQ.images[0], cQ, float(np.percentile(np.abs(D["Q"]), CLIP_Q)), "")

    # --- titles and legend -------------------------------------------------
    fig.text(x0 + mw / 2, y0 + mh + rib + 0.055,
             f"$W_{{rec}}$ reordered by $H$  ({n} × {n})",
             ha="center", va="bottom", fontsize=9.5, color=INK)
    fig.text(0.660 + qh / ar / 2, y0 + 0.13 * mh + qh + 0.085, "Quotient",
             ha="center", va="bottom", fontsize=10, color=INK)
    fig.text(0.660 + qh / ar / 2, y0 + 0.13 * mh + qh + 0.025,
             f"$Q_{{struct}}$ ({k} × {k})",
             ha="center", va="bottom", fontsize=10, color=INK)

    if legend:
        axL = fig.add_axes([0.075, 0.025, 0.855, leg_h - 0.050])
        axL.add_patch(Rectangle((0, 0), 1, 1, transform=axL.transAxes,
                                facecolor="none", edgecolor=MUTED, lw=0.8,
                                zorder=0))
        draw_role_legend(axL, colors, k)

    save(fig, stem)


# ------------------------------------------------------------------------ main

def report(W, D):
    """Numbers worth quoting in the caption."""
    eW = np.linalg.eigvals(W)
    eQ = np.linalg.eigvals(D["Q"])
    eW = eW[np.argsort(-np.abs(eW))][:4]
    eQ = eQ[np.argsort(-np.abs(eQ))][:4]
    print(f"  cells: {D['k']}, sizes {D['sizes'].min()}-{D['sizes'].max()} "
          f"(median {int(np.median(D['sizes']))})")
    print("  leading |lambda| of W:", np.round(np.abs(eW), 2))
    print("  leading |lambda| of Q:", np.round(np.abs(eQ), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=NET)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--eig", default="best",
                    help="'best' (max step-R2 over the top-|lambda| eigenvectors, "
                         "the reported r2_best statistic) or a 0-based rank")
    ap.add_argument("--order", default="seriate",
                    choices=["seriate", "size", "diag"],
                    help="cosmetic ordering of the cells")
    args = ap.parse_args()

    W, labels = load_sweep(args.net, args.k)
    D = coarse_grain(W, labels, args.order)
    vmax = float(np.percentile(np.abs(W), CLIP_W))

    tag = f"{args.net}_k{args.k}"
    print(f"{args.net}  W {W.shape}  k={args.k}  order={args.order}  "
          f"|W| clip at p{CLIP_W:g} = {vmax:.2f} "
          f"({100 * (np.abs(W) > vmax).mean():.1f}% of entries saturated)")
    report(W, D)

    eigvec_panel(W, D, args.k, f"fig1_eigvec_stepfit_{tag}", pick=args.eig)
    piece_wrec(D, vmax, f"fig1_wrec_partitioned_{tag}")
    piece_quotient(D, f"fig1_quotient_{tag}")
    piece_quotient_sized(D, f"fig1_quotient_sized_{tag}")
    combined(D, vmax, args.net, args.k, f"fig1_partition_quotient_{tag}")


if __name__ == "__main__":
    main()
