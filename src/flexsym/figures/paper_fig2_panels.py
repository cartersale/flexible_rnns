"""Paper Figure 2 -- connectivity-defined roles predict motif geometry (Results §2-3).

Panels, written as SVG into figures/fig2/pieces/:

  A  motif inventory: task x epoch mosaic, coloured by motif type
  B  worked example: per-unit motif participation, in role order
  C  geometry dissociation: step-R^2 of the motif direction on roles vs activity clusters
  D  commonality: unique / shared variance in motif participation
  E  A_cond vs k for m = 1, 3, 5, against the random-partition null and chance
  F  quotient ladder: Q_struct -> Q_pred -> Q_task, per network
  G  eigenvalue prediction: lambda_Q vs lambda_J per condition
  H  mode routing: containment-alignment correlation, and the span of the routed modes

Every panel is population-level (471 classified conditions over ten 256-unit networks)
except B, which is the anchoring example. Nothing is recomputed here: A/C/D come from
the participation and taxonomy CSVs, E/F from quotient_prediction.json, G from
quotient_lambda_pairs.csv (flexsym.run.quotient_lambda_pairs), H from
lumpability/quotient_mode_motif_alignment.csv.

    python -m flexsym.figures.paper_fig2_panels           # all panels
    python -m flexsym.figures.paper_fig2_panels --only C G
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from flexsym import paths
from flexsym.figures.figstyle import (
    INK, MUTED, GRID, BAND, BLUE, YELLOW, GREEN, ORANGE, AQUA, VIOLET, RED, GRAY, clean,
)

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

FP_ROOT = paths.RESULTS_ROOT / "motif_fixed_points"
WL_ROOT = paths.RESULTS_ROOT / "signed_input_wl_paper"
PIECES = paths.REPO_ROOT / "figures" / "fig2" / "pieces"

K = 20                                   # display / operating resolution
EX_NET, EX_COND = "LeakyRNN_softplus_diag_256_all_10", "dmsgo__delay1"

#: motif-type colours, shared by panels A, C and G (and Figures 3-4)
MOTIF_COL = {"line_attractor": BLUE, "saddle_decision": RED,
             "point_attractor": AQUA, "mixed": GRAY}
MOTIF_LAB = {"line_attractor": "line attractor", "saddle_decision": "decision saddle",
             "point_attractor": "point attractor", "mixed": "unclassified"}
EPOCHS = ["fix1", "stim1", "delay1", "stim2", "delay2"]
EPOCH_LAB = {"fix1": "fixation", "stim1": "stimulus", "delay1": "delay",
             "stim2": "stimulus 2", "delay2": "delay 2"}


def save(fig, name):
    """SVG into pieces/. Set FIG_PREVIEW=<dir> to also drop a PNG there for review."""
    PIECES.mkdir(parents=True, exist_ok=True)
    out = PIECES / f"{name}.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    preview = os.environ.get("FIG_PREVIEW")
    if preview:
        d = Path(preview); d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.png", dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def participation_table():
    """The 471 classified conditions, one row each, pooled over the ten networks."""
    out = []
    for p in sorted(FP_ROOT.glob("*/participation_per_motif.csv")):
        for r in rows(p):
            r["net"] = p.parent.name
            out.append(r)
    return out


def jitter(n, w, seed=0):
    return (np.random.default_rng(seed).random(n) - 0.5) * w


# --------------------------------------------------------------------- panel A

def panel_A():
    tax = rows(FP_ROOT / "motif_taxonomy_all_nets.csv")
    rules = sorted({r["rule"] for r in tax})
    cell = {}
    for r in tax:
        cell.setdefault((r["rule"], r["epoch"]), []).append(r["motif"])

    fig, ax = plt.subplots(figsize=(4.3, 4.6))
    fig.subplots_adjust(left=0.28, right=0.98, top=0.86, bottom=0.14)
    for i, rule in enumerate(rules):
        for j, ep in enumerate(EPOCHS):
            got = cell.get((rule, ep))
            if not got:
                continue
            mode, n = Counter(got).most_common(1)[0]
            agree = n / len(got)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, lw=0.6,
                                       facecolor=MOTIF_COL[mode], edgecolor="white",
                                       alpha=0.25 + 0.75 * agree, zorder=3))
            if agree < 0.999:
                ax.text(j, i, f"{agree:.0%}", ha="center", va="center", fontsize=5.2,
                        color="white", zorder=5)
    ax.set_xlim(-0.5, len(EPOCHS) - 0.5); ax.set_ylim(len(rules) - 0.5, -0.5)
    ax.set_xticks(range(len(EPOCHS)))
    ax.set_xticklabels([EPOCH_LAB[e] for e in EPOCHS], fontsize=7.5, color=INK,
                       rotation=35, ha="right")
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels(rules, fontsize=6.8, color=INK)
    ax.tick_params(length=0, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    counts = Counter(r["motif"] for r in tax)
    ax.legend(handles=[Patch(facecolor=MOTIF_COL[m], label=f"{MOTIF_LAB[m]} ({counts[m]})")
                       for m in ["line_attractor", "saddle_decision", "point_attractor", "mixed"]],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False,
              fontsize=7, handlelength=1.1)
    ax.set_title("Recovered motifs match the computation\neach epoch requires",
                 fontsize=10, color=INK, pad=8)
    ax.text(0.5, -0.30, "colour = most common motif over 10 networks;  shade = agreement",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.0, color=MUTED)
    save(fig, "fig2_panelA_motif_inventory")


# --------------------------------------------------------------------- panel B

def panel_B_units():
    from flexsym.motifs.motif_classify import npload
    from flexsym.run.motif_reuse_control import _row_for_k

    wl = npload(WL_ROOT / EX_NET / "transpose" / "signed_input_wl_sweep.npz")
    r = _row_for_k(wl["labels_struct_by_k"], K)
    labels = np.asarray(wl["labels_struct_by_k"][r], dtype=int)
    d = npload(FP_ROOT / EX_NET / "participation_cache_v2.npz")
    rec = next(m for m in list(d["records"]) if m["key"] == EX_COND)
    p = np.asarray(rec["p_act"], dtype=float)
    p = p / p.max()

    eta2 = next((float(x["eta2_role"]) for x in
                 rows(FP_ROOT / EX_NET / "participation_per_motif.csv")
                 if x["key"] == EX_COND), np.nan)

    cells = sorted(np.unique(labels), key=lambda c: -p[labels == c].mean())
    order, bnd, ci = [], [], 0
    for c in cells:
        idx = np.where(labels == c)[0]
        order += list(idx); ci += len(idx); bnd.append(ci)
    order = np.array(order)
    cmap = plt.get_cmap("turbo")
    col = {c: cmap(0.08 + 0.84 * i / max(len(cells) - 1, 1)) for i, c in enumerate(cells)}

    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.82, bottom=0.19)
    x = np.arange(len(order))
    ax.bar(x, p[order], width=1.0, color=[col[labels[u]] for u in order], zorder=3)
    st = 0
    for b, c in zip(bnd, cells):
        ax.plot([st - 0.5, b - 0.5], [p[labels == c].mean()] * 2, color=INK, lw=1.5,
                zorder=6)
        st = b
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xlabel("units, ordered by connectivity role", fontsize=9, color=INK)
    ax.set_ylabel("motif participation", fontsize=9, color=INK)
    ax.text(0.985, 0.94, f"$\\eta^2_{{\\rm role}}$ = {eta2:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5, color=INK)
    rule, epoch = EX_COND.split("__")
    ax.set_title(f"Motif participation is organized by role\n"
                 f"({rule} | {epoch}, line attractor)", fontsize=9.6, color=INK, pad=6)
    clean(ax)
    save(fig, "fig2_panelB_participation_units")


def _example_roles(k=K):
    """Role labels, per-role mean participation and the quotient, for the example."""
    from flexsym.motifs.motif_classify import npload
    from flexsym.run.motif_reuse_control import _row_for_k
    from flexsym.run.motif_quotient_prediction import indicator, quotient

    wl = npload(WL_ROOT / EX_NET / "transpose" / "signed_input_wl_sweep.npz")
    r = _row_for_k(wl["labels_struct_by_k"], k)
    labels = np.asarray(wl["labels_struct_by_k"][r], dtype=int)
    W = np.asarray(wl["W_rec"], dtype=float)
    d = npload(FP_ROOT / EX_NET / "participation_cache_v2.npz")
    rec = next(m for m in list(d["records"]) if m["key"] == EX_COND)
    p = np.asarray(rec["p_act"], dtype=float)
    p = p / p.max()

    cells = np.unique(labels)
    means = np.array([p[labels == c].mean() for c in cells])
    order = np.argsort(-means)                     # roles ranked by participation
    H, sizes = indicator(labels)
    Q = quotient(W, H, sizes)[np.ix_(order, order)]
    iqr = np.array([np.percentile(p[labels == c], [25, 75]) for c in cells[order]])
    return labels, p, cells[order], means[order], sizes[order], Q, iqr


def panel_B():
    """Role-level view: the quotient graph, with role participation as node size."""
    import networkx as nx

    labels, p, cells, means, sizes, Q, iqr = _example_roles()
    k = len(cells)
    cmap = plt.get_cmap("turbo")
    col = [cmap(0.08 + 0.84 * i / max(k - 1, 1)) for i in range(k)]

    eta2 = next((float(x["eta2_role"]) for x in
                 rows(FP_ROOT / EX_NET / "participation_per_motif.csv")
                 if x["key"] == EX_COND), np.nan)

    # Layout from the FULL weighted quotient (so no role is left unplaced), edges
    # drawn only where the coupling is strong enough to read.
    A = (np.abs(Q) + np.abs(Q).T) / 2
    np.fill_diagonal(A, 0.0)
    A = A / A.max()
    G = nx.Graph()
    G.add_nodes_from(range(k))
    for i in range(k):
        for j in range(i + 1, k):
            G.add_edge(i, j, w=float(A[i, j]))
    pos = nx.spring_layout(G, weight="w", seed=3, iterations=600, k=1.8 / np.sqrt(k))
    P = np.array([pos[i] for i in range(k)])
    P = P - P.mean(0)
    P = P / np.abs(P).max()

    thresh = np.percentile(A[A > 0], 82)
    fig = plt.figure(figsize=(6.9, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.24,
                          left=0.02, right=0.97, top=0.85, bottom=0.16)
    axg, axs = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    for i in range(k):
        for j in range(i + 1, k):
            if A[i, j] < thresh:
                continue
            f = (A[i, j] - thresh) / max(1 - thresh, 1e-9)
            axg.plot(P[[i, j], 0], P[[i, j], 1], color=GRAY, lw=0.35 + 1.3 * f,
                     alpha=0.14 + 0.34 * f, zorder=2, solid_capstyle="round")
    axg.scatter(P[:, 0], P[:, 1], s=45 + 620 * means, color=col, alpha=0.88,
                linewidths=1.0, edgecolors="white", zorder=5)
    axg.set_aspect("equal"); axg.axis("off")
    axg.set_xlim(-1.35, 1.35); axg.set_ylim(-2.30, 1.35)
    axg.set_title("connectivity roles, sized by\nmotif participation", fontsize=9.4,
                  color=INK, pad=4)

    # node-size key
    for x_, f in zip(np.linspace(-0.85, 0.25, 4), [0.15, 0.4, 0.7, 1.0]):
        axg.scatter([x_], [-1.55], s=45 + 620 * f * means.max(), color=GRAY, alpha=0.5,
                    linewidths=0, zorder=6)
    axg.annotate("", xy=(0.75, -1.55), xytext=(-1.05, -1.55),
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    axg.text(-1.15, -1.55, "low", fontsize=6.4, color=MUTED, ha="right", va="center")
    axg.text(0.82, -1.55, "high", fontsize=6.4, color=MUTED, ha="left", va="center")
    axg.text(0.0, -1.93, "edges and layout: recurrent coupling between roles "
             "($Q_{\\rm struct}$);\npositions are a force-directed layout, not a measurement",
             fontsize=6.0, color=MUTED, ha="center", va="top", linespacing=1.35)

    x = np.arange(1, k + 1)
    for i in range(k):
        axs.plot([x[i], x[i]], [iqr[i, 0], iqr[i, 1]], color=col[i], lw=3.0, alpha=0.25,
                 zorder=3, solid_capstyle="round")
        axs.plot([x[i], x[i]], [0, means[i]], color=col[i], lw=1.2, zorder=4)
    axs.scatter(x, means, s=34, facecolor="white", edgecolor=col, linewidths=1.6,
                zorder=6)
    axs.set_xlabel("structural role (ranked by participation)", fontsize=9.2, color=INK)
    axs.set_ylabel("mean motif participation $p_m$", fontsize=9.2, color=INK)
    axs.set_xlim(0.4, k + 0.6); axs.set_ylim(0, max(iqr[:, 1].max(), means.max()) * 1.12)
    axs.set_xticks([1, 5, 10, 15, 20][:k])
    axs.tick_params(labelsize=8, colors=MUTED)
    axs.text(0.97, 0.93, f"$\\eta^2_{{\\rm role}}$ = {eta2:.2f}", transform=axs.transAxes,
             ha="right", va="top", fontsize=9.5, color=INK)
    axs.text(0.97, 0.83, "bars: interquartile range of units in the role",
             transform=axs.transAxes, ha="right", va="top", fontsize=6.2, color=MUTED)
    clean(axs)

    rule, epoch = EX_COND.split("__")
    fig.suptitle(f"Motif participation concentrates on a few connectivity roles "
                 f"({rule} | {epoch})", fontsize=10.0, color=INK, y=0.99)
    save(fig, "fig2_panelB_role_participation")


# --------------------------------------------------------------------- panel C

def panel_C():
    d = participation_table()
    x = np.array([float(r["sub_step_r2"]) for r in d])
    y = np.array([float(r["sub_step_r2_activity"]) for r in d])
    null = np.median([float(r["sub_null_r2"]) for r in d])
    mot = [r["motif"] for r in d]

    fig, ax = plt.subplots(figsize=(3.9, 3.9))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.86, bottom=0.15)
    lim = (0, 0.8)
    ax.plot(lim, lim, color=MUTED, lw=1.0, ls="--", zorder=2)
    ax.axhspan(0, null, color=BAND, zorder=0)
    ax.axvspan(0, null, color=BAND, zorder=0)
    ax.text(0.008, null - 0.012, "matched-null level", ha="left", va="top",
            fontsize=6.4, color=MUTED, transform=ax.get_yaxis_transform())
    for m in ["line_attractor", "saddle_decision", "point_attractor"]:
        k = np.array([mm == m for mm in mot])
        ax.scatter(x[k], y[k], s=13, color=MOTIF_COL[m], alpha=0.75, lw=0.3,
                   edgecolor="white", zorder=4, label=MOTIF_LAB[m])
    net = np.array([r["net"] for r in d])
    for nn in np.unique(net):
        m = net == nn
        ax.scatter([np.median(x[m])], [np.median(y[m])], s=26, marker="D",
                   color="white", edgecolor=INK, lw=0.8, zorder=6)
    ax.scatter([np.median(x)], [np.median(y)], s=150, marker="X", color=INK,
               edgecolor="white", lw=1.2, zorder=7)
    ax.text(0.035, 0.70, f"median  ({np.median(x):.2f}, {np.median(y):.2f})",
            transform=ax.transAxes, fontsize=7.0, color=INK)
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_xlabel("step $R^2$ on connectivity roles", fontsize=9.5, color=INK)
    ax.set_ylabel("step $R^2$ on activity clusters", fontsize=9.5, color=INK)
    ax.set_title("Motif direction is piecewise-constant on roles,\n"
                 "not on activity clusters", fontsize=9.8, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left", handletextpad=0.2,
              borderpad=0.2)
    ax.text(0.97, 0.03, f"{len(d)} conditions · diamonds = network medians\n"
            f"$z$ = 7.8 vs 0.8",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8, color=MUTED)
    clean(ax, grid=None)
    save(fig, "fig2_panelC_geometry_dissociation")


# --------------------------------------------------------------------- panel D

def panel_D():
    d = participation_table()
    ur = np.array([float(r["cv_unique_role"]) for r in d])
    ua = np.array([float(r["cv_unique_activity"]) for r in d])
    sh = np.array([float(r["cv_shared"]) for r in d])
    net = np.array([r["net"] for r in d])
    both = np.array([float(r["cv_M2_both"]) for r in d])

    fig, (axT, axB) = plt.subplots(2, 1, figsize=(4.2, 3.2),
                                   gridspec_kw=dict(height_ratios=[1.0, 1.5]))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.80, bottom=0.16, hspace=0.55)

    parts = [(np.median(ur), GREEN, "unique to roles"),
             (np.median(sh), GRAY, "shared"),
             (np.median(ua), YELLOW, "unique to activity")]
    left = 0.0
    for val, c, lab in parts:
        axT.barh(0, val, left=left, height=0.55, color=c, edgecolor="white", lw=1.0,
                 zorder=3, label=lab)
        axT.text(left + val / 2, 0, f"{val:.2f}", ha="center", va="center",
                 fontsize=7.4, color="white", zorder=5)
        left += val
    axT.set_xlim(0, max(np.median(both) * 1.12, left * 1.12))
    axT.set_ylim(-0.5, 0.5); axT.set_yticks([])
    axT.set_xlabel("cross-validated $R^2$ of the combined model", fontsize=8, color=INK)
    axT.legend(frameon=False, fontsize=7, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.55), handlelength=1.0, columnspacing=1.0)
    for s in ["top", "right", "left"]:
        axT.spines[s].set_visible(False)
    axT.spines["bottom"].set_color(MUTED); axT.tick_params(colors=MUTED, labelsize=7)

    for i, (v, c, lab) in enumerate([(ur, GREEN, "unique to\nroles"),
                                     (ua, YELLOW, "unique to\nactivity")]):
        axB.scatter(v, i + jitter(len(v), 0.5, seed=i), s=8, color=c, alpha=0.45,
                    lw=0.2, edgecolor="white", zorder=4)
        per_net = [np.median(v[net == nn]) for nn in np.unique(net)]
        axB.scatter(per_net, [i] * len(per_net), s=24, marker="D", color="white",
                    edgecolor=INK, lw=0.7, zorder=6)
        axB.plot([np.median(v)] * 2, [i - 0.32, i + 0.32], color=INK, lw=2.0, zorder=7)
    axB.axvline(0, color=MUTED, lw=0.9)
    axB.set_yticks([0, 1]); axB.set_yticklabels(["unique to\nroles", "unique to\nactivity"],
                                                fontsize=7.6, color=INK)
    axB.set_ylim(-0.6, 1.6)
    axB.set_xlabel("per-condition unique $R^2$", fontsize=8.5, color=INK)
    axB.text(0.02, 0.02, "both $p < .001$ · diamonds = network medians",
             transform=axB.transAxes, ha="left", va="bottom", fontsize=6.4, color=MUTED)
    clean(axB, grid="x")
    fig.suptitle("Roles explain motif participation beyond\nthe activity partition",
                 fontsize=9.8, color=INK, y=1.06)
    save(fig, "fig2_panelD_commonality")


# ------------------------------------------------------------------ panels E/F

def quotient_json():
    return json.load(open(FP_ROOT / "quotient_prediction.json"))


def panel_E():
    d = quotient_json()
    ks = sorted({int(k) for v in d.values() for k in v["per_k"]})
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    fig.subplots_adjust(left=0.16, right=0.96, top=0.85, bottom=0.15)

    for m, col in [(1, GRAY), (3, BLUE), (5, VIOLET)]:
        med, lo, hi, kk, rnd = [], [], [], [], []
        for k in ks:
            vals = [v["per_k"][str(k)][f"pred_m{m}"]["obs"] for v in d.values()
                    if str(k) in v["per_k"]]
            if not vals:
                continue
            kk.append(k)
            med.append(np.median(vals)); lo.append(np.percentile(vals, 25))
            hi.append(np.percentile(vals, 75))
            rnd.append(np.median([v["per_k"][str(k)][f"pred_m{m}"]["rand"]
                                  for v in d.values() if str(k) in v["per_k"]]))
        ax.plot(kk, med, "-o", ms=3.4, color=col, lw=2.0, zorder=5, label=f"$m$ = {m}")
        ax.fill_between(kk, lo, hi, color=col, alpha=0.15, zorder=3)
        # each m against ITS OWN size-matched random-partition null
        ax.plot(kk, rnd, "--", color=col, lw=1.2, alpha=0.75, zorder=4)

    ax.plot(ks, np.sqrt(3 / np.array(ks, dtype=float)), ls=":", lw=1.2, color=MUTED,
            zorder=4)
    ax.text(ks[-1] - 1, np.sqrt(3 / ks[-1]) + 0.02, "chance ($m$ = 3)", ha="right",
            va="bottom", fontsize=6.8, color=MUTED)
    ax.text(0.985, 0.45, "dashed: size-matched\nrandom partition",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=INK)
    ax.set_xlabel("number of roles $k$", fontsize=9.5, color=INK)
    ax.set_ylabel("$A_{\\rm cond}$", fontsize=10, color=INK)
    ax.set_ylim(0.2, 1.0)
    ax.set_title("Three quotient modes capture the motif\ndirection at every resolution",
                 fontsize=9.8, color=INK, pad=8)
    ax.legend(frameon=False, fontsize=8, loc="lower left", handletextpad=0.4)
    clean(ax)
    save(fig, "fig2_panelE_acond_vs_k")


def panel_F():
    d = quotient_json()
    keys = [("struct_m3", "$Q_{\\rm struct}$\nconnectivity", BLUE),
            ("pred_m3", "$Q_{\\rm pred}$\n+ role gains", GREEN),
            ("task_m3", "$Q_{\\rm task}$\nupper bound", VIOLET)]
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    fig.subplots_adjust(left=0.19, right=0.96, top=0.85, bottom=0.17)
    rng = np.random.default_rng(0)
    for i, (key, lab, col) in enumerate(keys):
        v = np.array([n["per_k"][str(K)][key]["obs"] for n in d.values()
                      if str(K) in n["per_k"]])
        ax.bar(i, np.median(v), 0.62, color=col, edgecolor="white", lw=1.0, zorder=3)
        ax.scatter(i + (rng.random(v.size) - 0.5) * 0.32, v, s=13, color="white",
                   edgecolor=INK, lw=0.5, zorder=6, alpha=0.9)
        ax.text(i, 0.035, f"{np.median(v):.2f}", ha="center", fontsize=9.5,
                color="white", zorder=7)
    rnd = np.median([n["per_k"][str(K)]["pred_m3"]["rand"] for n in d.values()
                     if str(K) in n["per_k"]])
    act = np.median([n["per_k"][str(K)]["pred_m3"]["act"] for n in d.values()
                     if str(K) in n["per_k"]])
    ax.axhspan(min(rnd, act), max(rnd, act), color="#e8e6e1", zorder=1)
    ax.axhline(rnd, color=INK, ls="--", lw=1.1, zorder=4)
    ax.text(2.46, max(rnd, act) + 0.015,
            f"random {rnd:.2f} / activity {act:.2f}", ha="right", fontsize=6.8,
            color=INK)
    ax.axhline(np.sqrt(3 / K), color=MUTED, ls=":", lw=1.1, zorder=4)
    ax.text(-0.45, np.sqrt(3 / K) + 0.012, "chance", fontsize=6.8, color=MUTED)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l, _ in keys], fontsize=7.8,
                                                color=INK)
    ax.set_ylabel("$A_{\\rm cond}$  ($m$ = 3, $k$ = 20)", fontsize=9.5, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_title("Role-mean gains close the gap\nto the upper bound", fontsize=9.8,
                 color=INK, pad=8)
    clean(ax)
    save(fig, "fig2_panelF_quotient_ladder")


# --------------------------------------------------------------------- panel G

def panel_G():
    """Eigenvalue recovery.

    NOT drawn as a lambda_Q vs lambda_J scatter with an identity line: lambda_J is
    tightly clustered near 1 (median 1.02, IQR 1.00-1.04) because every motif-carrying
    mode is near-critical, so such a scatter shows range restriction (r = 0.17), not
    predictive accuracy. What is real, and what this panel shows, is that the quotient
    lands on the near-critical value with a small downward bias, and closer than either
    control in 98% of conditions.
    """
    d = rows(FP_ROOT / "quotient_lambda_pairs.csv")
    lj = np.array([float(r["lam_J"]) for r in d])
    lq = np.array([float(r["lam_Q"]) for r in d])
    seed = np.array([int(r["seed"]) for r in d])
    dl = {"connectivity\nroles": np.array([float(r["dlam"]) for r in d]),
          "activity\nclusters": np.array([float(r["dlam_act"]) for r in d]),
          "random\npartition": np.array([float(r["dlam_rand"]) for r in d])}

    fig = plt.figure(figsize=(6.6, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.34,
                          left=0.10, right=0.97, top=0.82, bottom=0.19)
    axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # ---- left: both eigenvalues are near-critical; the quotient sits just below
    axL.axvline(1.0, color=MUTED, ls="--", lw=1.0, zorder=2)
    axL.text(1.0, 1.62, "critical", ha="center", fontsize=6.6, color=MUTED)
    for i, (v, c, lab) in enumerate([(lq, GREEN, "$\\lambda_Q$  quotient mode"),
                                     (lj, INK, "$\\lambda_J$  full Jacobian")]):
        axL.scatter(v, i + jitter(len(v), 0.46, seed=i), s=7, color=c, alpha=0.4,
                    lw=0.2, edgecolor="white", zorder=4)
        axL.plot([np.median(v)] * 2, [i - 0.34, i + 0.34], color=INK, lw=2.2, zorder=6)
        axL.text(np.median(v), i + 0.46, f"{np.median(v):.2f}", ha="center",
                 fontsize=7.6, color=INK, zorder=7)
    axL.annotate("", xy=(np.median(lq), 0.72), xytext=(np.median(lj), 0.72),
                 arrowprops=dict(arrowstyle="<|-", color=RED, lw=1.3,
                                 shrinkA=0, shrinkB=0, mutation_scale=10), zorder=8)
    axL.text(0.5 * (np.median(lq) + np.median(lj)), 0.86,
             f"bias {np.median(lq - lj):+.2f}", ha="center", va="bottom", fontsize=6.8,
             color=RED, zorder=9,
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))
    axL.set_yticks([0, 1])
    axL.set_yticklabels(["$\\lambda_Q$\nquotient", "$\\lambda_J$\nJacobian"],
                        fontsize=8.0, color=INK)
    axL.set_ylim(-0.6, 1.75); axL.set_xlim(0.55, 1.3)
    axL.set_xlabel("eigenvalue", fontsize=9.5, color=INK)
    axL.set_title("Motif modes are near-critical, and\nthe quotient lands there",
                  fontsize=9.4, color=INK, pad=8)
    clean(axL, grid="x")

    # ---- right: the comparison that carries the claim
    for i, (lab, v) in enumerate(dl.items()):
        c = [GREEN, YELLOW, GRAY][i]
        axR.scatter(v, i + jitter(len(v), 0.42, seed=i), s=7, color=c, alpha=0.35,
                    lw=0.2, edgecolor="white", zorder=4)
        per_net = [np.nanmedian(v[seed == sd]) for sd in np.unique(seed)]
        axR.scatter(per_net, [i] * len(per_net), s=26, marker="D", color="white",
                    edgecolor=INK, lw=0.7, zorder=6)
        axR.plot([np.nanmedian(v)] * 2, [i - 0.36, i + 0.36], color=INK, lw=2.4,
                 zorder=7)
        axR.text(np.nanmedian(v), i + 0.42, f"{np.nanmedian(v):.2f}", ha="center",
                 fontsize=7.6, color=INK)
    axR.set_yticks(range(3)); axR.set_yticklabels(list(dl), fontsize=7.8, color=INK)
    axR.set_ylim(-0.7, 2.75); axR.set_xlim(0, 0.5)
    axR.set_xlabel("$|\\lambda_Q - \\lambda_J|$   (best-aligned mode)", fontsize=9.2,
                   color=INK)
    axR.set_title("Closer than either control", fontsize=9.4, color=INK, pad=8)
    better_r = 100 * np.mean(dl["connectivity\nroles"] < dl["random\npartition"])
    better_a = 100 * np.mean(dl["connectivity\nroles"] < dl["activity\nclusters"])
    axR.text(0.98, 0.03, f"roles closer in {better_r:.0f}% of conditions vs random,\n"
             f"{better_a:.0f}% vs activity · 10/10 networks · diamonds = network medians",
             transform=axR.transAxes, ha="right", va="bottom", fontsize=6.0, color=MUTED,
             linespacing=1.35)
    clean(axR, grid="x")
    save(fig, "fig2_panelG_eigenvalue_recovery")


# --------------------------------------------------------------------- panel H

def panel_H():
    d = [r for r in rows(WL_ROOT / "lumpability" / "quotient_mode_motif_alignment.csv")
         if int(r["k_req"]) == K]
    rho = np.array([float(r["rho_c_a"]) for r in d])
    rho0 = np.array([float(r["rho_c_a_null_part"]) for r in d])
    span = {"$k$ most-contained\nmodes": np.array([float(r["span_routed"]) for r in d]),
            "$k$ slowest\nmodes": np.array([float(r["span_slow"]) for r in d]),
            "$k$ random\nmodes": np.array([float(r["span_rand"]) for r in d])}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.2, 3.2))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.18, wspace=0.42)

    for i, (v, c, lab) in enumerate([(rho, GREEN, "connectivity roles"),
                                     (rho0, GRAY, "size-matched null")]):
        axL.scatter(i + jitter(len(v), 0.45, seed=i), v, s=8, color=c, alpha=0.45,
                    lw=0.2, edgecolor="white", zorder=4)
        axL.plot([i - 0.32, i + 0.32], [np.median(v)] * 2, color=INK, lw=2.2, zorder=6)
        axL.text(i + 0.38, np.median(v), f"{np.median(v):.2f}", ha="left",
                 va="center", fontsize=8.0, color=INK)
    axL.axhline(0, color=MUTED, lw=0.9)
    axL.set_xticks([0, 1]); axL.set_xticklabels(["connectivity\nroles",
                                                 "size-matched\nnull"], fontsize=7.8,
                                                color=INK)
    axL.set_xlim(-0.6, 1.6)
    axL.set_ylabel("$\\rho$ (containment, alignment)", fontsize=9.0, color=INK)
    axL.set_title("Motifs occupy modes contained\nin the role subspace", fontsize=9.4,
                  color=INK, pad=6)
    clean(axL)

    for i, (lab, v) in enumerate(span.items()):
        c = [GREEN, BLUE, GRAY][i]
        axR.bar(i, np.median(v), 0.62, color=c, edgecolor="white", lw=1.0, zorder=3)
        axR.text(i, np.median(v) + 0.025, f"{np.median(v):.2f}", ha="center",
                 fontsize=8.0, color=INK)
    axR.set_xticks(range(3)); axR.set_xticklabels(list(span), fontsize=7.4, color=INK)
    axR.set_ylim(0, 1.05)
    axR.set_ylabel("motif direction captured", fontsize=9.0, color=INK)
    axR.set_title("and those modes span it", fontsize=9.4, color=INK, pad=6)
    axR.text(0.5, -0.34, "containment is not timescale: the slowest modes span the motif\n"
             "equally well, and containment is uncorrelated with $|\\lambda|$ ($\\rho$ = $-$0.04)",
             transform=axR.transAxes, ha="center", va="top", fontsize=6.0, color=MUTED,
             linespacing=1.35)
    clean(axR)
    save(fig, "fig2_panelH_mode_routing")


PANELS = {"A": panel_A, "B": panel_B, "B0": panel_B_units, "C": panel_C,
          "D": panel_D, "E": panel_E, "F": panel_F, "G": panel_G, "H": panel_H}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="panel letters")
    args = ap.parse_args()
    for name in (args.only or PANELS):
        PANELS[name.upper()]()


if __name__ == "__main__":
    main()
