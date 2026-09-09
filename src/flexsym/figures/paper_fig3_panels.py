"""Paper Figure 3 -- state-dependent gain reconfigures a shared scaffold (Results §4).

The section's result is partly negative -- quotient modes are NOT reusable operators
that identify a motif type across tasks -- so the figure has to make the negative
legible before the positive replacement lands: the conserved object is the structural
substrate, and gain state selects the configuration.

Panels, written as SVG into figures/fig3/pieces/:

  A  the decomposition, from data: one fixed Q_struct, three condition gain vectors,
     three different spectra
  B  operator reuse fails: quotient modes separate same-type from different-type
     cross-task pairs worse than unit-level participation does
  C  gain-swap matrix: condition i's motif direction against a quotient built from
     condition j's gains, one network
  D  swap ladder: unit > role ~ activity > random partition > shuffled gains

  E  composition (supplement): a composed motif lies in the span of its two
     constituents barely more than a size-matched null

Unlike Figure 2, panels here use each network's own operating resolution (OPK,
k = 33-44), because that is where the gain analyses were run.

    python -m flexsym.figures.paper_fig3_panels
    python -m flexsym.figures.paper_fig3_panels --only C
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, Rectangle

from flexsym import paths
from flexsym.figures.figstyle import (
    INK, MUTED, GRID, BAND, BLUE, YELLOW, GREEN, ORANGE, AQUA, VIOLET, RED, GRAY, clean,
)
from flexsym.motifs.motif_classify import npload
from flexsym.motifs.motif_participation import WL_ROOT
from flexsym.run.gain_configuration import (
    diag_percentile, participation_ratio, susceptibility, swap_matrix,
)
from flexsym.run.motif_quotient_prediction import (
    condition_jacobians, indicator, load_conditions, quotient,
)
from flexsym.run.motif_reuse_control import _row_for_k
from flexsym.structural.operating_k import OPK, model_name

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

FP_ROOT = paths.RESULTS_ROOT / "motif_fixed_points"
PIECES = paths.REPO_ROOT / "figures" / "fig3" / "pieces"
EX_SEED = 10                       # example network, as in Figure 2

MOTIF_COL = {"line_attractor": BLUE, "saddle_decision": RED,
             "point_attractor": AQUA, "mixed": GRAY}
MOTIF_LAB = {"line_attractor": "line attractor", "saddle_decision": "decision saddle",
             "point_attractor": "point attractor", "mixed": "unclassified"}
MOTIF_ORDER = ["line_attractor", "saddle_decision", "point_attractor"]


def save(fig, name):
    PIECES.mkdir(parents=True, exist_ok=True)
    out = PIECES / f"{name}.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    preview = os.environ.get("FIG_PREVIEW")
    if preview:
        d = Path(preview); d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def jitter(n, w, seed=0):
    return (np.random.default_rng(seed).random(n) - 0.5) * w


def load_net(seed=EX_SEED):
    """W, role labels at the operating k, and the condition list for one network."""
    net = model_name(seed)
    k = OPK[seed]
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    W = np.asarray(wl["W_rec"], dtype=float)
    ind = np.asarray(wl["ind_active"], dtype=int)
    labels = np.asarray(wl["labels_struct_by_k"][_row_for_k(wl["labels_struct_by_k"], k)],
                        dtype=int)
    conds = load_conditions(net, ind, W, condition_jacobians(net, ind, W))
    return net, k, W, labels, conds


def leverage_json():
    return json.load(open(FP_ROOT / "gain_leverage_geometry.json"))


def rel_allocation(gl, tname, cname):
    """Per-network relative allocation, real and size-matched null.

    Exactly the aggregation `gain_leverage_geometry` prints: leverage on the class
    divided by leverage on the bulk, medianed over contrasts within a network. The
    reported enrichment is the ratio of the medians across networks.
    """
    R, P = [], []
    for v in gl.values():
        rr = [c[f"{tname}_{cname}"]["response"] / c[f"{tname}_other"]["response"]
              for c in v["contrasts"] if c[f"{tname}_other"]["response"] > 0]
        pp = [c[f"{tname}_{cname}"]["response_permsize"]
              / c[f"{tname}_other"]["response_permsize"]
              for c in v["contrasts"] if c[f"{tname}_other"]["response_permsize"] > 0]
        R.append(np.nanmedian(rr) if rr else np.nan)
        P.append(np.nanmedian(pp) if pp else np.nan)
    return np.array(R), np.array(P)


def bulk_ratio(gl, tname):
    """Per-network absolute ratio for the irrelevant bulk (depletion)."""
    out = []
    for v in gl.values():
        r = [c[f"{tname}_other"]["response"] / c[f"{tname}_other"]["response_permsize"]
             for c in v["contrasts"] if c[f"{tname}_other"]["response_permsize"] > 0]
        out.append(np.nanmedian(r) if r else np.nan)
    return np.array(out)


def gain_json():
    return json.load(open(FP_ROOT / "gain_configuration.json"))


def composition_json():
    return json.load(open(FP_ROOT / "operator_composition.json"))


# --------------------------------------------------------------------- panel A

def panel_A():
    net, k, W, labels, conds = load_net()
    H, sizes = indicator(labels)
    Qs = quotient(W, H, sizes)

    # one condition per motif type, each the median-|gbar| example of its type
    picks = []
    for m in MOTIF_ORDER:
        same = [c for c in conds if c["motif"] == m]
        if not same:
            continue
        gb = [np.linalg.norm((H.T @ c["g"]) / sizes) for c in same]
        picks.append(same[int(np.argsort(gb)[len(gb) // 2])])

    fig = plt.figure(figsize=(7.1, 3.6))
    gs = fig.add_gridspec(len(picks), 3, width_ratios=[1.25, 1.05, 1.0],
                          wspace=0.40, hspace=0.55,
                          left=0.02, right=0.97, top=0.80, bottom=0.14)
    axQ = fig.add_subplot(gs[:, 0])
    vmax = np.percentile(np.abs(Qs), 98)
    axQ.imshow(Qs, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax),
               interpolation="nearest")
    axQ.set_xticks([]); axQ.set_yticks([])
    for s_ in axQ.spines.values():
        s_.set_visible(False)
    axQ.set_title("$Q_{\\rm struct}$\nfixed across all conditions", fontsize=9.0,
                  color=INK, pad=6)
    axQ.set_xlabel(f"{k} roles", fontsize=8.0, color=MUTED)

    gmean = np.array([(H.T @ c["g"]) / sizes for c in conds]).mean(0)
    axS = fig.add_subplot(gs[:, 2])
    for i, c in enumerate(picks):
        col = MOTIF_COL[c["motif"]]
        gb = (H.T @ c["g"]) / sizes
        axg = fig.add_subplot(gs[i, 1])
        axg.bar(np.arange(len(gb)), gb, width=0.9, color=col, zorder=3)
        axg.plot(np.arange(len(gb)), gmean, color=MUTED, lw=0.8, zorder=5)
        axg.set_xlim(-0.5, len(gb) - 0.5); axg.set_xticks([])
        axg.tick_params(labelsize=6.2, colors=MUTED)
        axg.set_ylabel("$\\bar{g}$", fontsize=8.0, color=INK, labelpad=1)
        rule, epoch = c["key"].split("__")
        axg.text(0.0, 1.04, f"{rule} | {epoch}", transform=axg.transAxes,
                 fontsize=7.0, color=col, va="bottom")
        for s_ in ["top", "right"]:
            axg.spines[s_].set_visible(False)
        for s_ in ["left", "bottom"]:
            axg.spines[s_].set_color(MUTED)
        if i == 0:
            axg.set_title("role-mean gain $\\bar{g}_c$", fontsize=9.0, color=INK,
                          pad=16)
            axg.text(1.0, 1.04, "grey: mean over conditions", transform=axg.transAxes,
                     ha="right", va="bottom", fontsize=6.2, color=MUTED)

        Qc = (1 - c["alpha"]) * np.eye(len(sizes)) + c["alpha"] * (gb[:, None] * Qs)
        mag = np.sort(np.abs(np.linalg.eigvals(Qc)))[::-1][:12]
        axS.plot(np.arange(1, len(mag) + 1), mag, "-o", ms=3.2, lw=1.6, color=col,
                 zorder=5, label=f"{c['key'].split('__')[0]}")

    axS.axhline(1.0, color=INK, ls="--", lw=1.0, zorder=3)
    axS.text(12, 1.003, "critical", ha="right", va="bottom", fontsize=6.8, color=INK)
    axS.set_xlabel("mode rank", fontsize=8.6, color=INK)
    axS.set_ylabel("$|\\lambda|$ of $Q_c$", fontsize=9.0, color=INK)
    axS.set_title("spectrum of $Q_c$", fontsize=9.0, color=INK, pad=16)
    axS.set_xlim(0.5, 12.5)
    clean(axS)

    fig.suptitle("One structural substrate, reconfigured by gain state",
                 fontsize=10.5, color=INK, y=1.02)
    fig.text(0.5, -0.02, "$Q_c = (1-\\alpha)I + \\alpha\\,{\\rm diag}(\\bar{g}_c)"
             "\\,Q_{\\rm struct}$  —  only $\\bar{g}_c$ changes across conditions",
             ha="center", fontsize=7.6, color=MUTED)
    save(fig, "fig3_panelA_decomposition")


# --------------------------------------------------------------------- panel B

def panel_B():
    d = composition_json()
    node = np.array([n["A"]["auc_node"] for n in d.values()])
    op = np.array([n["A"]["auc_operator"] for n in d.values()])
    act = np.array([n["A"]["auc_activity"] for n in d.values()])
    null = np.array([n["A"]["auc_null_mean"] for n in d.values()])
    lo_op = np.array([n["A"]["low_node"]["auc_operator"] for n in d.values()])

    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    fig.subplots_adjust(left=0.17, right=0.96, top=0.84, bottom=0.19)
    ax.axhspan(0.5, np.median(null), color=BAND, zorder=0)
    ax.axhline(0.5, color=INK, lw=1.1, zorder=2)
    ax.text(-0.44, 0.505, "chance", ha="left", fontsize=6.8, color=INK, va="bottom")
    ax.text(-0.44, np.median(null) - 0.004, "random-partition null", ha="left",
            va="top", fontsize=6.8, color=MUTED)

    xs = [0, 1, 2, 3]
    data = [node, op, lo_op, act]
    cols = [INK, VIOLET, VIOLET, YELLOW]
    for a, b in zip(node, op):                     # pair adjacent columns only
        ax.plot([0, 1], [a, b], color=GRAY, lw=0.8, alpha=0.7, zorder=3)
    for x, v, c in zip(xs, data, cols):
        ax.scatter(x + jitter(len(v), 0.22, seed=x), v, s=16, color="white",
                   edgecolor=c, lw=1.2, zorder=5)
        ax.plot([x - 0.30, x + 0.30], [np.median(v)] * 2, color=c, lw=2.4, zorder=6)
        ax.text(x - 0.34, np.median(v), f"{np.median(v):.2f}", ha="right", va="center",
                fontsize=8.0, color=c, zorder=7,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    ax.set_xticks(xs)
    ax.set_xticklabels(["unit-level\nparticipation", "quotient\nmodes",
                        "quotient modes\nlowest-overlap", "activity\nclusters"],
                       fontsize=7.2, color=INK)
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(0.42, 0.80)
    ax.set_ylabel("AUC, same-type vs different-type\ncross-task pairs", fontsize=9.0,
                  color=INK)
    ax.set_title("Quotient modes do not identify a motif\nacross tasks",
                 fontsize=9.8, color=INK, pad=8)
    ax.text(0.5, 0.03, f"quotient modes better in "
            f"{int((op > node).sum())} of {len(op)} networks",
            transform=ax.transAxes, ha="center", fontsize=6.8, color=MUTED)
    clean(ax)
    save(fig, "fig3_panelB_operator_reuse")


# --------------------------------------------------------------------- panel C

def panel_C():
    """The swap matrix, shown as the statistic actually measures it.

    Raw ||B(gbar_j)'v_i|| / ||P_H v_i|| is dominated by row-level differences between
    conditions, so a heatmap of it looks uniform and hides the diagonal. `diag_percentile`
    scores the matched entry by its WITHIN-ROW percentile, so the panel plots that:
    entry (i,j) is the fraction of condition i's other gain states that its own row beats.
    The diagonal is then exactly the quantity being averaged.
    """
    net, k, W, labels, conds = load_net()
    order = sorted(range(len(conds)),
                   key=lambda i: (MOTIF_ORDER.index(conds[i]["motif"])
                                  if conds[i]["motif"] in MOTIF_ORDER else 99,
                                  conds[i]["key"]))
    conds = [conds[i] for i in order]
    S = swap_matrix(conds, labels, W)
    pct = diag_percentile(S)
    n = len(conds)

    P = np.full_like(S, np.nan)
    for i in range(n):
        row = S[i]
        off = np.delete(row, i)
        off = off[np.isfinite(off)]
        if not len(off):
            continue
        for j in range(n):
            if np.isfinite(row[j]):
                P[i, j] = (off < row[j]).mean() + 0.5 * (off == row[j]).mean()
    d = np.diag(P)

    fig = plt.figure(figsize=(5.0, 3.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.34], wspace=0.10,
                          left=0.16, right=0.86, top=0.83, bottom=0.16)
    ax = fig.add_subplot(gs[0]); axr = fig.add_subplot(gs[1], sharey=None)

    im = ax.imshow(P, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    for i in range(n):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=ORANGE,
                               lw=0.8, zorder=5))
    ax.set_xlabel("gain state used  ($\\bar{g}_j$)", fontsize=9.0, color=INK)
    ax.set_ylabel("motif direction of condition $i$", fontsize=9.0, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for m in MOTIF_ORDER:
        idx = [i for i, c in enumerate(conds) if c["motif"] == m]
        if not idx:
            continue
        lo, hi = min(idx) - 0.5, max(idx) + 0.5
        ax.add_patch(Rectangle((-2.2, lo), 1.2, hi - lo, color=MOTIF_COL[m],
                               clip_on=False, zorder=4))
        ax.add_patch(Rectangle((lo, n + 0.9), hi - lo, 1.2, color=MOTIF_COL[m],
                               clip_on=False, zorder=4))
    ax.set_xlim(-2.4, n - 0.5); ax.set_ylim(n + 2.4, -0.5)

    axr.barh(np.arange(n), d, height=0.9, color=BLUE, zorder=3)
    axr.axvline(0.5, color=INK, lw=1.0, zorder=5)
    axr.plot([np.nanmedian(d)] * 2, [-0.5, n - 0.5], color=ORANGE, lw=1.4, zorder=6)
    axr.set_ylim(n + 2.4, -0.5); axr.set_xlim(0, 1.0)
    axr.set_yticks([]); axr.set_xticks([0, 0.5, 1.0])
    axr.set_xticklabels(["0", "0.5", "1"], fontsize=6.6, color=MUTED)
    axr.set_xlabel("own gains\n(diagonal)", fontsize=7.4, color=INK)
    for sp in ["top", "right", "left"]:
        axr.spines[sp].set_visible(False)
    axr.spines["bottom"].set_color(MUTED)

    cb = fig.colorbar(im, ax=axr, shrink=0.62, pad=0.30)
    cb.set_label("within-row percentile", fontsize=7.4, color=MUTED)
    cb.ax.tick_params(labelsize=6.4, colors=MUTED); cb.outline.set_visible(False)

    ax.set_title("A condition's own gain state recovers its motif best",
                 fontsize=9.8, color=INK, pad=8, x=0.72)
    fig.text(0.16, -0.005, f"seed {EX_SEED}, $k$ = {k}, {n} conditions  ·  "
             f"median diagonal percentile {pct:.2f} (chance 0.50, orange line)",
             ha="left", va="top", fontsize=6.6, color=MUTED)
    save(fig, "fig3_panelC_gain_swap_matrix")


# --------------------------------------------------------------------- panel D

def panel_D():
    d = gain_json()
    reps = [("unit", "per-unit\ngains", GREEN), ("role", "role-mean\ngains", BLUE),
            ("activity", "activity\nclusters", YELLOW),
            ("random_partition", "random\npartition", GRAY),
            ("shuffled_gains", "shuffled\nrole gains", ORANGE)]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    fig.subplots_adjust(left=0.15, right=0.97, top=0.85, bottom=0.20)
    ax.axhline(0.5, color=INK, lw=1.1, zorder=4)
    ax.text(4.45, 0.507, "chance", ha="right", fontsize=6.8, color=INK, va="bottom")
    for i, (key, lab, col) in enumerate(reps):
        v = np.array([n["swap"][key]["pct"] for n in d.values()])
        ax.bar(i, np.median(v), 0.6, color=col, edgecolor="white", lw=1.0, zorder=3)
        ax.scatter(i + jitter(len(v), 0.3, seed=i), v, s=13, color="white",
                   edgecolor=INK, lw=0.5, zorder=6, alpha=0.9)
        ax.text(i, 0.035, f"{np.median(v):.2f}", ha="center", fontsize=9.0,
                color="white", zorder=7)
    ax.set_xticks(range(len(reps)))
    ax.set_xticklabels([l for _, l, _ in reps], fontsize=7.4, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("own condition recovered\n(within-row percentile)", fontsize=9.0,
                  color=INK)
    ax.set_title("Gain state specifies the configuration —\n"
                 "but roles discard some of it", fontsize=9.8, color=INK, pad=8)
    clean(ax)
    save(fig, "fig3_panelD_swap_ladder")


# --------------------------------------------------------------------- panel E

def panel_E():
    d = json.load(open(FP_ROOT / "direct_sum_residual.json"))
    obs, null, rs = [], [], []
    for n in d.values():
        for t in n["triples"].values():
            for r, v in t["by_r"].items():
                obs.append(v["rho_AB"]); null.append(v["null"]); rs.append(int(r))
    obs, null, rs = np.array(obs), np.array(null), np.array(rs)

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    fig.subplots_adjust(left=0.17, right=0.96, top=0.85, bottom=0.17)
    uniq = sorted(set(rs))
    for i, r in enumerate(uniq):
        m = rs == r
        ax.scatter(i + jitter(m.sum(), 0.3, seed=i) - 0.16, obs[m], s=11, color=VIOLET,
                   alpha=0.5, lw=0.2, edgecolor="white", zorder=4)
        ax.scatter(i + jitter(m.sum(), 0.3, seed=i + 50) + 0.16, null[m], s=11,
                   color=GRAY, alpha=0.5, lw=0.2, edgecolor="white", zorder=4)
        ax.plot([i - 0.34, i + 0.02], [np.median(obs[m])] * 2, color=VIOLET, lw=2.2,
                zorder=6)
        ax.plot([i - 0.02, i + 0.34], [np.median(null[m])] * 2, color=INK, lw=2.2,
                zorder=6)
        ax.text(i, 1.005, f"{np.median(obs[m]) - np.median(null[m]):+.2f}",
                ha="center", fontsize=7.0, color=INK, transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(uniq)))
    ax.set_xticklabels([str(r) for r in uniq], fontsize=8.0, color=INK)
    ax.set_xlabel("truncation rank of the shared subspace removed", fontsize=8.6,
                  color=INK)
    ax.set_ylabel("composed motif captured by\nits two constituents", fontsize=9.0,
                  color=INK)
    ax.set_title("Composition is barely above a\nsize-matched null", fontsize=9.8,
                 color=INK, pad=12)
    ax.scatter([], [], s=18, color=VIOLET, label="observed")
    ax.scatter([], [], s=18, color=GRAY, label="size-matched null")
    ax.legend(frameon=False, fontsize=7.2, loc="lower left", handletextpad=0.2)
    clean(ax)
    save(fig, "fig3_panelE_composition")


# --------------------------------------------------------------------- panel F

def panel_F():
    """The susceptibility map itself, and how concentrated it is."""
    net, k, W, labels, conds = load_net()
    H, sizes = indicator(labels)
    Qs = quotient(W, H, sizes)
    alpha = float(np.median([c["alpha"] for c in conds]))
    gref = np.median(np.array([(H.T @ c["g"]) / sizes for c in conds]), axis=0)
    S, lam, _ = susceptibility(Qs, gref, alpha)
    order = np.argsort(-np.abs(lam))[:10]
    M = S[:, order]
    rows = np.argsort(-np.abs(M).max(1))
    M = M[rows]

    fig = plt.figure(figsize=(6.6, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.42,
                          left=0.10, right=0.95, top=0.82, bottom=0.17)
    ax, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    vmax = np.percentile(np.abs(M), 99)
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax),
                   aspect="auto", interpolation="nearest")
    ax.set_xlabel("collective mode (top 10 by $|\\lambda|$)", fontsize=8.8, color=INK)
    ax.set_ylabel(f"structural role (of {len(sizes)}, sorted by leverage)",
                  fontsize=8.8, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cb.set_label("$S^{\\rm ev}_{ij} = \\partial|\\lambda_j| / \\partial\\bar{g}_i$",
                 fontsize=7.6, color=MUTED)
    cb.ax.tick_params(labelsize=6.4, colors=MUTED); cb.outline.set_visible(False)
    ax.set_title("Leverage sits on a minority of roles", fontsize=9.6, color=INK, pad=8)

    gj = gain_json()
    ks = np.array([n["susceptibility"]["k"] for n in gj.values()], dtype=float)
    series = [("pr_role", "connectivity\nroles", GREEN),
              ("pr_activity", "activity\nclusters", YELLOW),
              ("pr_random", "random\npartition", GRAY)]
    vals = [np.array([n["susceptibility"][key] for n in gj.values()]) for key, _, _ in series]
    for i in range(len(ks)):
        axr.plot(range(3), [v[i] for v in vals], color=GRAY, lw=0.8, alpha=0.6, zorder=3)
    for x, (v, (_, lab, c)) in enumerate(zip(vals, series)):
        axr.scatter([x] * len(v) + jitter(len(v), 0.16, seed=x), v, s=16, color="white",
                    edgecolor=c, lw=1.2, zorder=5)
        axr.plot([x - 0.3, x + 0.3], [np.median(v)] * 2, color=c, lw=2.4, zorder=6)
        axr.text(x + 0.34, np.median(v), f"{np.median(v):.1f}", ha="left", va="center",
                 fontsize=8.4, color=c, zorder=7)
    axr.plot([-0.3, 2.3], [np.median(ks)] * 2, color=INK, ls=":", lw=1.1, zorder=4)
    axr.text(2.3, np.median(ks) - 0.4, f"all {np.median(ks):.0f} roles", ha="right",
             va="top", fontsize=6.8, color=MUTED)
    axr.set_xticks(range(3)); axr.set_xticklabels([l for _, l, _ in series],
                                                 fontsize=7.6, color=INK)
    axr.set_xlim(-0.5, 2.5)
    axr.set_ylabel("participation ratio of leverage", fontsize=9.0, color=INK)
    axr.set_title("More concentrated than either control", fontsize=9.6, color=INK,
                  pad=8)
    axr.text(0.02, 0.885, "lower = more concentrated\n10 of 10 networks",
             transform=axr.transAxes, ha="left", va="top", fontsize=6.8, color=MUTED,
             linespacing=1.35)
    clean(axr)
    save(fig, "fig3_panelF_susceptibility_map")


# --------------------------------------------------------------------- panel G

def panel_G():
    """First-order validity: does S_ev predict the observed eigenvalue change?"""
    gj = gain_json()
    rows = [r for n in gj.values() for r in n["leverage"]]
    obs = np.array([r["r_pred_obs"] for r in rows])
    null = np.array([r["r_null"] for r in rows])
    per_net = [(np.nanmedian([r["r_pred_obs"] for r in n["leverage"]]),
                np.nanmedian([r["r_null"] for r in n["leverage"]])) for n in gj.values()]

    fig, ax = plt.subplots(figsize=(3.7, 3.5))
    fig.subplots_adjust(left=0.19, right=0.96, top=0.83, bottom=0.15)
    ax.axhline(0, color=MUTED, lw=0.9, zorder=2)
    for i, (v, c) in enumerate([(obs, GREEN), (null, GRAY)]):
        ax.scatter(i + jitter(len(v), 0.34, seed=i), v, s=9, color=c, alpha=0.4,
                   lw=0.2, edgecolor="white", zorder=4)
        ax.plot([i - 0.32, i + 0.32], [np.nanmedian(v)] * 2, color=INK, lw=2.4, zorder=7)
    for i, vals in enumerate([[a for a, _ in per_net], [b for _, b in per_net]]):
        ax.text(i - 0.38, np.median(vals), f"{np.median(vals):.2f}", ha="right",
                va="center", fontsize=9.0, color=INK, zorder=9,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    ax.scatter([0] * len(per_net), [a for a, _ in per_net], s=26, marker="D",
               color="white", edgecolor=INK, lw=0.7, zorder=6)
    ax.scatter([1] * len(per_net), [b for _, b in per_net], s=26, marker="D",
               color="white", edgecolor=INK, lw=0.7, zorder=6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["$S^{\\rm ev}\\cdot\\Delta\\bar{g}$",
                                               "role-permuted"], fontsize=8.6, color=INK)
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(-0.6, 1.05)
    ax.set_ylabel("$r$ with measured $\\Delta|\\lambda_j|$", fontsize=9.2, color=INK)
    ax.set_title("The map predicts the network's own\ngain-induced changes",
                 fontsize=9.6, color=INK, pad=8)
    ax.text(0.5, 0.10, f"{len(rows)} condition contrasts · diamonds and numbers are\n"
            "network medians · positive in 10 of 10 networks", transform=ax.transAxes,
            ha="center", fontsize=6.6, color=MUTED, linespacing=1.35)
    clean(ax)
    save(fig, "fig3_panelG_leverage_validity")


# --------------------------------------------------------------------- panel H

def panel_H():
    """Allocation: gain changes go to roles with leverage over the modes that matter."""
    gl = leverage_json()
    cats = [("ev", "star", "eigenvalue\nmotif mode", BLUE),
            ("ev", "critical", "eigenvalue\nnear-critical", BLUE),
            ("rot", "star", "rotation\nmotif mode", VIOLET),
            ("rot", "critical", "rotation\nnear-critical", VIOLET)]
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    fig.subplots_adjust(left=0.15, right=0.96, top=0.80, bottom=0.21)
    for i, (t, cn, lab, col) in enumerate(cats):
        R, P = rel_allocation(gl, t, cn)
        for a, b in zip(P, R):
            ax.plot([i - 0.17, i + 0.17], [a, b], color=GRAY, lw=0.7, alpha=0.6,
                    zorder=3)
        ax.scatter([i - 0.17] * len(P) + jitter(len(P), 0.1, seed=i), P, s=13,
                   color="white", edgecolor=GRAY, lw=1.0, zorder=5)
        ax.scatter([i + 0.17] * len(R) + jitter(len(R), 0.1, seed=i + 9), R, s=13,
                   color="white", edgecolor=col, lw=1.2, zorder=5)
        ax.plot([i - 0.30, i - 0.04], [np.nanmedian(P)] * 2, color=GRAY, lw=2.2,
                zorder=6)
        ax.plot([i + 0.04, i + 0.30], [np.nanmedian(R)] * 2, color=col, lw=2.4,
                zorder=6)
        ax.text(i, 1.045, f"{np.nanmedian(R) / np.nanmedian(P):.2f}$\\times$",
                transform=ax.get_xaxis_transform(), ha="center", fontsize=8.6,
                color=col)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([l for _, _, l, _ in cats], fontsize=7.4, color=INK)
    ax.set_xlim(-0.55, len(cats) - 0.45)
    ax.set_ylabel("leverage on the class ÷\nleverage on the bulk", fontsize=9.0,
                  color=INK)
    ax.set_title("Gain changes are directed at the roles with\nleverage over the modes "
                 "that carry computation", fontsize=9.6, color=INK, pad=20)
    ax.scatter([], [], s=16, color="white", edgecolor=GRAY, lw=1.0,
               label="size-matched null")
    ax.scatter([], [], s=16, color="white", edgecolor=INK, lw=1.2, label="observed")
    ax.legend(frameon=False, fontsize=7.0, loc="lower right", handletextpad=0.2)
    clean(ax)
    save(fig, "fig3_panelH_allocation")


# --------------------------------------------------------------------- panel I

def panel_I():
    """What reconfiguration spares, and what it avoids."""
    gl = leverage_json()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(5.8, 3.3))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.80, bottom=0.18, wspace=0.42)

    # left: rotation spared, eigenvalue suppressed, for the motif-carrying mode
    ev = np.array([np.nanmedian([c["ev_star"]["response"]
                                 / c["ev_star"]["response_permsize"]
                                 for c in n["contrasts"]
                                 if c["ev_star"]["response_permsize"] > 0])
                   for n in gl.values()])
    rot = np.array([np.nanmedian([c["rot_star"]["response"]
                                  / c["rot_star"]["response_permsize"]
                                  for c in n["contrasts"]
                                  if c["rot_star"]["response_permsize"] > 0])
                    for n in gl.values()])
    axL.axhline(1.0, color=INK, lw=1.1, zorder=3)
    axL.text(-0.45, 1.012, "size-matched null", ha="left", fontsize=6.8, color=INK)
    for a, b in zip(ev, rot):
        axL.plot([0, 1], [a, b], color=GRAY, lw=0.8, alpha=0.6, zorder=4)
    for x, (v, c) in enumerate([(ev, BLUE), (rot, VIOLET)]):
        axL.scatter([x] * len(v) + jitter(len(v), 0.14, seed=x), v, s=17, color="white",
                    edgecolor=c, lw=1.3, zorder=6)
        axL.plot([x - 0.28, x + 0.28], [np.nanmedian(v)] * 2, color=c, lw=2.4, zorder=7)
        axL.text(x - 0.33, np.nanmedian(v), f"{np.nanmedian(v):.2f}", ha="right",
                 va="center", fontsize=8.6, color=c, zorder=8,
                 bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    axL.set_xticks([0, 1]); axL.set_xticklabels(["eigenvalue\nleverage",
                                                 "rotation\nleverage"], fontsize=8.0,
                                                color=INK)
    axL.set_xlim(-0.5, 1.5)
    axL.set_ylabel("allocation relative to null", fontsize=9.0, color=INK)
    axL.set_title("Reconfiguration reorients the motif mode\nrather than destabilising it",
                  fontsize=9.2, color=INK, pad=8)
    axL.text(0.5, 0.965, f"rotation spared in {int((rot > ev).sum())} of 10 networks",
             transform=axL.transAxes, ha="center", va="top", fontsize=6.8, color=MUTED)
    clean(axL)

    # right: the bulk is avoided
    for i, (t, lab, c) in enumerate([("ev", "eigenvalue\nleverage", BLUE),
                                     ("rot", "rotation\nleverage", VIOLET)]):
        v = bulk_ratio(gl, t)
        axR.bar(i, np.nanmedian(v), 0.55, color=c, edgecolor="white", lw=1.0, zorder=3)
        axR.scatter(i + jitter(len(v), 0.28, seed=i), v, s=13, color="white",
                    edgecolor=INK, lw=0.5, zorder=6, alpha=0.9)
        axR.text(i, 0.04, f"{np.nanmedian(v):.2f}", ha="center", fontsize=9.5,
                 color="white", zorder=7)
    axR.axhline(1.0, color=INK, lw=1.1, zorder=4)
    axR.text(-0.45, 1.015, "size-matched null", ha="left", fontsize=6.8, color=INK)
    axR.set_xticks([0, 1]); axR.set_xticklabels(["eigenvalue\nleverage",
                                                 "rotation\nleverage"], fontsize=8.0,
                                                color=INK)
    axR.set_xlim(-0.5, 1.5); axR.set_ylim(0, 1.25)
    axR.set_ylabel("allocation to the irrelevant bulk", fontsize=9.0, color=INK)
    axR.set_title("and avoids the irrelevant bulk", fontsize=9.2, color=INK, pad=8)
    axR.text(0.5, 0.965, "10 of 10 networks below null", transform=axR.transAxes,
             ha="center", va="top", fontsize=6.8, color=MUTED)
    clean(axR)
    save(fig, "fig3_panelI_leverage_geometry")


PANELS = {"A": panel_A, "B": panel_B, "C": panel_C, "D": panel_D, "E": panel_E,
          "F": panel_F, "G": panel_G, "H": panel_H, "I": panel_I}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    for name in (args.only or PANELS):
        PANELS[name.upper()]()


if __name__ == "__main__":
    main()
