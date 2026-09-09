"""Paper Figure 1, panel A -- schematic, drawn from real data.

One fixed recurrent connectivity (W_rec) is driven by different task
conditions; each condition realises a different dynamical motif in the same
network. Everything shown is measured, not drawn:

  * the directed graph and the matrix are the trained recurrent weights of
    LeakyRNN_softplus_diag_256_all_0, ordered by its structural roles
  * the timecourses are real trial inputs with real epoch boundaries
  * the motif portraits are the recovered fixed/slow points, real trial
    trajectories, and the local linearised flow of the trained Jacobian

Run flexsym.figures.paper_fig1_data first to build figures/fig1/data/.

    python -m flexsym.figures.paper_fig1
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle

from flexsym import paths
from flexsym.figures.figstyle import INK, MUTED, BLUE, YELLOW, ORANGE, AQUA, VIOLET, RED

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

OUT = paths.REPO_ROOT / "figures" / "fig1"
PIECES = OUT / "pieces"
DATA = OUT / "data" / "fig1_panelA.npz"

POS_W, NEG_W = "#3f7fc4", "#c9524f"          # excitatory / inhibitory weights
EP = {"fix1": "#d5d4cf", "stim1": YELLOW, "stim2": "#f2c14e",
      "delay1": BLUE, "delay2": "#7fb0e8", "go1": ORANGE}
EP_NAME = {"fix1": "fixation", "stim1": "stimulus", "stim2": "stimulus 2",
           "delay1": "delay", "delay2": "delay 2", "go1": "response"}

# (data key, display task name, rule, highlighted epoch, motif label, hue)
ROWS = [
    ("point",  "Reaction-Pro",  "reactgo",   "go1",    "point attractor",              AQUA),
    ("ring",   "Delay-Anti",    "delayanti", "delay1", "ring attractor",               VIOLET),
    ("line",   "DMC",           "dmcgo",     "delay1", "line attractor",               BLUE),
    ("saddle", "Delay-DM1",     "delaydm1",  "stim1",  "saddle\n(decision boundary)",  RED),
    ("rot",    "Delay-Pro",     "delaygo",   "delay1", "rotation\n(into output space)", ORANGE),
]


def load():
    return np.load(paths.require(DATA, "fig1 panel A cache "
                                       "(run flexsym.figures.paper_fig1_data)"))


# ------------------------------------------------------- directed W_rec graph

def draw_wrec_graph(ax, D, n_nodes=16, n_per_sign=18):
    """A directed subgraph of the trained recurrent weights.

    Nodes are the units with the strongest recurrent coupling, on one ring in
    structural-role order. Edges are the strongest real weights among them,
    taken separately per sign so both are legible, and drawn as curved arrows
    so i->j and j->i stay distinguishable. Self-connections (kept strong by the
    diagonal initialisation) are drawn as outward loops.
    """
    W, roles = D["W"], D["roles"]
    strength = np.abs(W).sum(0) + np.abs(W).sum(1)
    sel = np.argsort(-strength)[:n_nodes]
    sel = sel[np.argsort(roles[sel], kind="stable")]
    Ws = W[np.ix_(sel, sel)]

    n = len(sel)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    P = np.c_[np.cos(th), np.sin(th)]

    off = Ws.copy()
    np.fill_diagonal(off, 0.0)
    wmax = np.abs(off).max()
    edges = []
    for sign in (1, -1):
        m = np.where(np.sign(off) == sign, np.abs(off), 0.0)
        edges += list(np.argsort(-m, axis=None)[:n_per_sign])

    for f in edges:
        i, j = np.unravel_index(f, off.shape)
        w = off[i, j]
        if w == 0:
            continue
        a = abs(w) / wmax
        ax.add_patch(FancyArrowPatch(
            P[i], P[j], arrowstyle="-|>,head_length=3.2,head_width=2.0",
            color=POS_W if w > 0 else NEG_W, lw=0.30 + 1.05 * a,
            alpha=0.35 + 0.5 * a, zorder=2,
            connectionstyle="arc3,rad=0.16", shrinkA=6.5, shrinkB=6.5))

    dmax = np.abs(np.diag(Ws)).max() or 1.0
    for i in range(n):
        w = Ws[i, i]
        if abs(w) < 0.35 * dmax:
            continue
        u = P[i] / np.linalg.norm(P[i])
        t = np.array([-u[1], u[0]]) * 0.055
        ax.add_patch(FancyArrowPatch(
            P[i] + t, P[i] - t, arrowstyle="-|>,head_length=2.6,head_width=1.7",
            color=POS_W if w > 0 else NEG_W, lw=0.4 + 0.9 * abs(w) / dmax,
            alpha=0.8, zorder=3, connectionstyle="arc3,rad=-2.6",
            shrinkA=0.5, shrinkB=0.5))

    for (x, y) in P:
        ax.add_patch(Circle((x, y), 0.072, facecolor="#3f3e3b", edgecolor="white",
                            lw=0.9, zorder=6))

    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)
    ax.axis("off")


def draw_wrec_matrix(ax, D, cbar_ax=None):
    W, roles = D["W"], D["roles"]
    v = float(np.percentile(np.abs(W), 99))
    im = ax.imshow(W, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")

    # thin rules at the structural-role boundaries the matrix is sorted by
    b = np.where(np.diff(roles) != 0)[0] + 0.5
    for x in b:
        ax.axhline(x, color="#ffffff", lw=0.30, alpha=0.55)
        ax.axvline(x, color="#ffffff", lw=0.30, alpha=0.55)

    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_linewidth(0.7)
    if cbar_ax is not None:
        cb = plt.colorbar(im, cax=cbar_ax)
        cb.set_ticks([-v, 0, v])
        cb.set_ticklabels([f"-{v:.1f}", "0", f"{v:.1f}"])
        cb.ax.tick_params(labelsize=6, colors=MUTED, length=2)
        cb.outline.set_visible(False)
    return im


# ---------------------------------------------------------- real trial inputs

def draw_timeseries(ax, D, rule, hi_epoch, hue, show_time=False):
    x = D[f"x__{rule}"]                       # (T, n_input)
    eps = D[f"ep__{rule}"]
    T = x.shape[0]
    t = np.arange(T)

    for name, a, b in eps:
        if name not in EP:
            continue
        ax.axvspan(a, b, color=EP[name], alpha=0.42 if name == hi_epoch else 0.17,
                   lw=0, zorder=0)
        if name == hi_epoch:
            ax.plot([a, b], [-0.30, -0.30], color=hue, lw=2.6, solid_capstyle="butt",
                    clip_on=False, zorder=8)

    # fixation channel (0) and the four circular stimulus channels (1:5)
    ax.plot(t, x[:, 0], color="#6f6e6a", lw=1.2, zorder=4)
    for c in range(1, 5):
        if np.ptp(x[:, c]) < 1e-6:
            continue
        ax.plot(t, x[:, c], color=hue, lw=1.15, alpha=0.9, zorder=5)

    ax.set_xlim(0, T - 1)
    ax.set_ylim(-0.12, 1.55)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if show_time:
        ax.annotate("", xy=(1.0, -0.46), xytext=(0.0, -0.46), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))
        ax.text(1.0, -0.62, "time", transform=ax.transAxes, ha="right", va="top",
                fontsize=7.2, color=MUTED)


# ------------------------------------------------- measured motif portraits

def _trajlines(ax, T, color, lw=0.75, alpha=0.8):
    segs = [np.c_[T[i, :-1], T[i, 1:]].reshape(-1, 2, 2)
            for i in range(T.shape[0])]
    ax.add_collection(LineCollection(np.concatenate(segs), colors=color, lw=lw,
                                     alpha=alpha, zorder=3))


def draw_motif(ax, D, key, hue):
    pts = D[f"{key}__pts"]
    traj = D.get(f"{key}__traj")
    meta = D[f"{key}__meta"][0]

    if f"{key}__flow_g" in D:
        g, U, V = D[f"{key}__flow_g"], D[f"{key}__flow_U"], D[f"{key}__flow_V"]
        sp = np.hypot(U, V)
        ax.streamplot(g, g, U, V, color="#d6d5d1",
                      linewidth=0.35 + 0.7 * sp / (sp.max() or 1.0),
                      density=0.62, arrowsize=0.6, minlength=0.30, zorder=1)

    if traj is not None:
        _trajlines(ax, traj, "#84837f")

    if key == "rot":
        # memory manifold and response manifold, in the plane they jointly span
        tb = D.get(f"{key}__traj_b")
        if tb is not None:
            _trajlines(ax, tb, "#b89476", alpha=0.75)
        ax.scatter(pts[:, 0], pts[:, 1], s=2.6, color=BLUE, alpha=0.75,
                   linewidths=0, zorder=5)
        ax.scatter(D[f"{key}__pts_b"][:, 0], D[f"{key}__pts_b"][:, 1], s=2.6,
                   color=hue, alpha=0.85, linewidths=0, zorder=6)
        r = 0.45 * np.abs(pts).max()
        ax.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=0, theta2=float(meta["angle"]),
                         color=INK, lw=1.0, zorder=7))
        ax.text(r * 1.18, r * 0.62, f"{meta['angle']:.0f}°", fontsize=6.6, color=INK,
                zorder=8, ha="left", va="bottom")
    else:
        stable = D[f"{key}__stable"].astype(bool)
        s = 34.0 if len(pts) < 5 else (9.0 if len(pts) < 120 else 3.2)
        ax.scatter(pts[stable, 0], pts[stable, 1], s=s, color=hue, alpha=0.85,
                   linewidths=0, zorder=5)
        if (~stable).any():
            ax.scatter(pts[~stable, 0], pts[~stable, 1], s=max(s, 11), facecolor="white",
                       edgecolor=hue, linewidths=0.8, zorder=6)

    m = np.abs(np.r_[pts, traj.reshape(-1, 2) if traj is not None else pts]).max()
    ax.set_xlim(-1.12 * m, 1.12 * m); ax.set_ylim(-1.12 * m, 1.12 * m)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ------------------------------------------------------------------- assembly

def panel_A(path=None):
    D = load()
    fig = plt.figure(figsize=(7.9, 5.8))
    gs = fig.add_gridspec(
        nrows=len(ROWS), ncols=4,
        width_ratios=[1.62, 0.26, 1.85, 1.28],
        left=0.010, right=0.845, top=0.860, bottom=0.120, hspace=0.55, wspace=0.09,
    )

    # --- left: the one fixed network ---------------------------------------
    ax_net = fig.add_subplot(gs[0:2, 0])
    draw_wrec_graph(ax_net, D)
    ax_net.text(0.5, 1.02, "one fixed recurrent network", transform=ax_net.transAxes,
                ha="center", va="bottom", fontsize=8.8, color=INK)
    ax_net.text(0.5, 0.00, "strongest recurrent weights among\n16 units, ordered by structural role",
                transform=ax_net.transAxes, ha="center", va="top", fontsize=6.3, color=MUTED)

    ax_mat = fig.add_subplot(gs[2:5, 0])
    box = ax_mat.get_position()
    # square in inches, not in figure fraction, or imshow letterboxes inside the
    # axes and the colorbar ends up detached from the image
    fw, fh = fig.get_size_inches()
    side_in = min(box.width * fw * 0.88, box.height * fh * 0.90)
    w, h = side_in / fw, side_in / fh
    ax_mat.set_position([box.x0 + (box.width - w) / 2 - 0.012,
                         box.y0 + (box.height - h) / 2 - 0.008, w, h])
    mb = ax_mat.get_position()
    cax = fig.add_axes([mb.x1 + 0.006, mb.y0 + mb.height * 0.18, 0.0085, mb.height * 0.64])
    draw_wrec_matrix(ax_mat, D, cbar_ax=cax)
    ax_mat.text(0.5, -0.035, r"$W_{\mathrm{rec}}$: 204 active units,"
                             "\nsorted into 41 structural roles",
                transform=ax_mat.transAxes, ha="center", va="top", fontsize=7.2, color=INK)

    # --- fan-out -----------------------------------------------------------
    ax_fan = fig.add_subplot(gs[:, 1]); ax_fan.axis("off")
    ax_fan.set_xlim(0, 1); ax_fan.set_ylim(0, 1)
    for i in range(len(ROWS)):
        y = 1.0 - (i + 0.5) / len(ROWS)
        ax_fan.add_patch(FancyArrowPatch(
            (0.02, 0.5), (0.92, y), arrowstyle="-|>", mutation_scale=8.5,
            color="#b9b8b3", lw=1.0, zorder=2))

    # --- rows --------------------------------------------------------------
    for i, (key, task, rule, hi, label, hue) in enumerate(ROWS):
        ax_t = fig.add_subplot(gs[i, 2])
        draw_timeseries(ax_t, D, rule, hi, hue, show_time=(i == len(ROWS) - 1))
        ax_t.text(0.004, 1.06, task, transform=ax_t.transAxes, ha="left", va="bottom",
                  fontsize=8.2, color=INK)
        ax_t.text(1.0, 1.06, f"{EP_NAME[hi]} epoch", transform=ax_t.transAxes,
                  ha="right", va="bottom", fontsize=6.8, color=MUTED, style="italic")

        ax_m = fig.add_subplot(gs[i, 3])
        draw_motif(ax_m, D, key, hue)
        ax_m.text(1.02, 0.5, label, transform=ax_m.transAxes, ha="left", va="center",
                  fontsize=7.4, color=INK)

        if i == 0:
            ax_t.text(0.5, 1.62, "task conditions over time", transform=ax_t.transAxes,
                      ha="center", va="bottom", fontsize=8.8, color=INK)
            ax_m.text(0.5, 1.14, "dynamical motif", transform=ax_m.transAxes,
                      ha="center", va="bottom", fontsize=8.8, color=INK)

    _legend(fig)
    _save(fig, path or OUT / "fig1_panelA_schematic.svg")


def _legend(fig, y=0.036):
    """Epoch swatches, then what the marks in the motif portraits mean."""
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.075]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    yy = 0.45

    x = 0.022
    for name in ["fix1", "stim1", "delay1", "go1"]:
        ax.add_patch(Rectangle((x, yy - 0.13), 0.016, 0.26, facecolor=EP[name],
                               alpha=0.5, edgecolor="none"))
        ax.text(x + 0.021, yy, EP_NAME[name], fontsize=6.6, color=MUTED, va="center")
        x += 0.082

    x = 0.395
    ax.scatter([x], [yy], s=8, color="#4a4945", linewidths=0, clip_on=False)
    ax.text(x + 0.010, yy, "fixed / slow points", fontsize=6.6, color=MUTED, va="center")

    x = 0.525
    ax.scatter([x], [yy], s=22, facecolor="white", edgecolor="#4a4945", linewidths=0.9,
               clip_on=False)
    ax.text(x + 0.011, yy, "unstable point", fontsize=6.6, color=MUTED, va="center")

    x = 0.635
    ax.plot([x, x + 0.020], [yy, yy], color="#9c9b97", lw=0.9)
    ax.text(x + 0.026, yy, "trial trajectories", fontsize=6.6, color=MUTED, va="center")

    x = 0.760
    ax.annotate("", xy=(x + 0.020, yy), xytext=(x, yy),
                arrowprops=dict(arrowstyle="-|>", color="#c9c8c4", lw=0.9))
    ax.text(x + 0.026, yy, "linearised flow of the trained Jacobian",
            fontsize=6.6, color=MUTED, va="center")


def pieces():
    """Each element on its own canvas, for hand assembly."""
    D = load()

    f = plt.figure(figsize=(2.4, 2.4)); a = f.add_axes([0, 0, 1, 1])
    draw_wrec_graph(a, D)
    _save(f, PIECES / "wrec_directed_graph.svg")

    f = plt.figure(figsize=(2.6, 2.4))
    a = f.add_axes([0.02, 0.02, 0.86, 0.96])
    c = f.add_axes([0.90, 0.16, 0.025, 0.66])
    draw_wrec_matrix(a, D, cbar_ax=c)
    _save(f, PIECES / "wrec_matrix.svg")

    for i, (key, task, rule, hi, label, hue) in enumerate(ROWS):
        f = plt.figure(figsize=(2.8, 0.76))
        a = f.add_axes([0.01, 0.16, 0.98, 0.78])
        draw_timeseries(a, D, rule, hi, hue)
        _save(f, PIECES / f"timeseries_{i + 1}_{rule}_{hi}.svg")

        f = plt.figure(figsize=(1.7, 1.7))
        a = f.add_axes([0.02, 0.02, 0.96, 0.96])
        draw_motif(a, D, key, hue)
        _save(f, PIECES / f"motif_{i + 1}_{key}.svg")


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", facecolor="white")
    fig.savefig(path.with_suffix(".png"), dpi=220, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    panel_A()
    pieces()
