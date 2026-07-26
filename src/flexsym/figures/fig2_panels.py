"""Fig 2 (lumpable invariant subspaces) panels -> results/figures/fig2/*.svg"""
from __future__ import annotations

from flexsym import paths
import sys, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from flexsym.figures.figstyle import INK, MUTED, GRID, BAND, BLUE, YELLOW, GREEN, ORANGE, AQUA, VIOLET, RED, GRAY, clean, save
from flexsym.structural.signed_input_wl_structural_roles import apply_matrix_variant, approx_wl_signed, load_w_rec

REPO = paths.REPO_ROOT
BASE = REPO / "results/signed_input_wl_paper"
OUT = REPO / "results/figures/fig2"
HUES = [BLUE, GREEN, "#e87ba4", YELLOW, AQUA, ORANGE, VIOLET, RED]


def _load_W_labels(model, k):
    """Active-unit W (transpose variant) and labels at k, from the stored sweep npz (no drive)."""
    import csv as _csv
    for d in [f"transpose_selected_k_{k}", "transpose_selected_k"]:
        p = BASE / f"{model}/{d}/signed_input_wl_sweep.npz"
        if p.exists():
            s = np.load(p, allow_pickle=True)
            W = np.asarray(s["W_rec"], dtype=float)
            cols = [str(x) for x in np.asarray(s["row_columns"], dtype=object)]
            kmin = int(np.asarray(s["rows"], dtype=object)[0, cols.index("k")])
            lab = np.asarray(s["labels_struct_by_k"][k - kmin], dtype=np.int64)
            return W, lab
    return None, None


def _step_r2(v, lab):
    pred = np.zeros_like(v)
    for c in np.unique(lab):
        pred[lab == c] = v[lab == c].mean()
    return 1 - ((v-pred)**2).sum()/((v-v.mean())**2).sum()


def panel_eigenvector():
    # representative example (median step-R2 among 256-unit nets); computed from stored W
    EX, EXK = "LeakyRNN_softplus_diag_256_all_2", 38
    W, labels = _load_W_labels(EX, EXK)
    w, V = np.linalg.eig(W)
    order = np.argsort(-np.abs(w))[:12]
    best = max(order, key=lambda i: _step_r2(np.real(V[:, i])/np.linalg.norm(np.real(V[:, i])), labels))
    v = np.real(V[:, best]); v = v/np.linalg.norm(v)

    fig = plt.figure(figsize=(7.4, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.32,
                          left=0.08, right=0.97, top=0.82, bottom=0.16)
    axL = fig.add_subplot(gs[0]); axR = fig.add_subplot(gs[1])

    # ---- left: representative staircase ----
    cells = sorted(np.unique(labels), key=lambda c: v[labels == c].mean())
    order2, bnd, ci = [], [], 0
    for c in cells:
        idx = np.where(labels == c)[0]; order2 += list(idx); ci += len(idx); bnd.append(ci)
    order2 = np.array(order2); x = np.arange(len(order2))
    colmap = {c: HUES[i % len(HUES)] for i, c in enumerate(cells)}
    axL.bar(x, v[order2], width=1.0, color=[colmap[labels[u]] for u in order2], zorder=3)
    st = 0
    for b, c in zip(bnd, cells):
        axL.plot([st-0.5, b-0.5], [v[labels == c].mean()]*2, color=INK, lw=1.6, zorder=5); st = b
    axL.axhline(0, color=MUTED, lw=0.8)
    axL.text(0.03, 0.95, f"step $R^2$={_step_r2(v, labels):.2f}", transform=axL.transAxes, va="top",
             fontsize=10, color=INK, fontweight="bold")
    axL.set_xlabel("units (ordered by connectivity group)", fontsize=9.5, color=INK)
    axL.set_ylabel("eigenvector component", fontsize=9.5, color=INK)
    axL.set_title("A dynamical mode is flat within each group\n(representative network)", fontsize=9.8, color=INK, pad=6)
    clean(axL); axL.set_xlim(-0.5, len(order2)-0.5)

    # ---- right: step-R2 across all 33 networks, real vs random ----
    rows = list(csv.DictReader(open(BASE / "lumpability/step_r2_all.csv")))
    real = np.array([float(r["real"]) for r in rows]); rand = np.array([float(r["rand"]) for r in rows])
    szs = np.array([int(r["size"]) for r in rows])
    rng = np.random.default_rng(0)
    from flexsym.figures.figstyle import SIZE_COL
    axR.scatter(np.zeros(len(rand))+(rng.random(len(rand))-0.5)*0.28, rand, s=16, color=GRAY,
                edgecolor="white", lw=0.4, zorder=4)
    for i, sz in enumerate([128, 256, 512, 1024]):
        mrk = szs == sz
        axR.scatter(np.ones(mrk.sum())+(rng.random(mrk.sum())-0.5)*0.28, real[mrk], s=16,
                    color=SIZE_COL[sz], edgecolor="white", lw=0.4, zorder=4, label=f"{sz}")
    for xx, d in [(0, rand), (1, real)]:
        axR.plot([xx-0.28, xx+0.28], [d.mean()]*2, color=INK, lw=2.2, zorder=6)
    axR.set_xticks([0, 1]); axR.set_xticklabels(["random\ngrouping", "connectivity\ngroups"], fontsize=8.5, color=INK)
    axR.set_xlim(-0.5, 1.5); axR.set_ylim(0, 1.0)
    axR.set_ylabel("step $R^2$", fontsize=9.5, color=INK)
    axR.set_title("all 33 networks", fontsize=9.8, color=INK, pad=6)
    axR.legend(title="units", frameon=False, fontsize=7, title_fontsize=7, loc="center right", handletextpad=0.2)
    clean(axR)

    fig.suptitle("Connectivity groups are invariant subspaces of the dynamics", fontsize=11, color=INK, y=0.99)
    save(fig, OUT / "panelA_eigenvector.svg")


def panel_by_size():
    lump = list(csv.DictReader(open(BASE / "lumpability/lumpability.csv")))
    sizes = [128, 256, 512, 1024]; x = np.arange(4); w = 0.38
    fig, ax = plt.subplots(figsize=(3.8, 3.4)); fig.subplots_adjust(left=0.16, bottom=0.14, top=0.9, right=0.96)
    rng = np.random.default_rng(0)
    for off, key, col, lab in [(-w/2-0.01, "transpose_eig_leakage_z", BLUE, "recurrent-input matrix"),
                               (+w/2+0.01, "W_eig_leakage_z", YELLOW, "dynamics matrix")]:
        vals = {s: [-float(r[key]) for r in lump if int(r["size"]) == s] for s in sizes}
        ax.bar(x+off, [np.mean(vals[s]) for s in sizes], w, color=col, edgecolor="white", lw=1.0, zorder=3, label=lab)
        for i, s in enumerate(sizes):
            p = np.array(vals[s]); ax.scatter(x[i]+off+(rng.random(p.size)-0.5)*w*0.5, p, s=11, color="white", edgecolor=INK, lw=0.5, zorder=6, alpha=0.8)
    ax.axhline(0, color=MUTED, lw=1.0); ax.text(0.02, 0.02, "chance", transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(sizes, fontsize=9.5, color=INK)
    ax.set_xlabel("network size (units)", fontsize=10, color=INK)
    ax.set_ylabel("lumpability ($-$leakage $z$)", fontsize=9.5, color=INK)
    ax.set_title("Stronger with size", fontsize=11, color=INK, pad=8)
    ax.legend(loc="upper left", frameon=False, fontsize=8.3)
    clean(ax); save(fig, OUT / "panelB_lumpability_by_size.svg")


def panel_jacobian():
    jac = list(csv.DictReader(open(BASE / "motif_jacobian_lumpability.csv")))
    lin = np.array([float(r["r2_linear_WT"]) for r in jac]); jc = np.array([float(r["r2_delay_jacobian"]) for r in jac])
    rd = np.array([float(r["r2_random_partition"]) for r in jac])
    slow = np.mean([float(r["slow_J_eig_abs"]) for r in jac]); slowr2 = np.mean([float(r["slow_mode_step_r2"]) for r in jac])
    fig, ax = plt.subplots(figsize=(3.6, 3.4)); fig.subplots_adjust(left=0.16, bottom=0.16, top=0.9, right=0.96)
    ax.axhspan(0, rd.mean(), color=BAND, zorder=0); rng = np.random.default_rng(0)
    for i, (d, c) in enumerate(zip([lin, jc, rd], [BLUE, AQUA, GRAY])):
        ax.bar(i, d.mean(), 0.6, color=c, edgecolor="white", lw=1.0, zorder=3)
        ax.scatter(i+(rng.random(d.size)-0.5)*0.3, d, s=12, color="white", edgecolor=INK, lw=0.5, zorder=6, alpha=0.85)
        ax.text(i, d.mean()+0.03, f"{d.mean():.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(["linear\nrecurrent", "delay\nJacobian", "random"], fontsize=9, color=INK)
    ax.set_ylabel("eigenvector step-$R^2$", fontsize=9.5, color=INK); ax.set_ylim(0, 1.0)
    ax.set_title("Nonlinear memory Jacobian\nis lumpable", fontsize=10.5, color=INK, pad=8)
    ax.text(0.03, 0.16, f"slow mode $\\lambda\\approx{slow:.2f}$\ncell-organized ($R^2={slowr2:.2f}$)", transform=ax.transAxes, fontsize=8, color=INK, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID))
    clean(ax); save(fig, OUT / "panelC_jacobian.svg")


def panel_training_control():
    rows = list(csv.DictReader(open(BASE / "training_control.csv")))
    TYPES = ["trained", "row-shuffle", "full-shuffle", "gaussian"]
    LAB = ["trained", "strength\nshuffle", "full\nshuffle", "gaussian"]
    COL = [BLUE, YELLOW, ORANGE, GRAY]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.8, 3.3)); fig.subplots_adjust(left=0.1, right=0.98, top=0.85, bottom=0.17, wspace=0.32)
    rng = np.random.default_rng(0); x = np.arange(4)
    for ax, key, ylab, ttl in [(axA, "equit_z", "equitability $z$", "Equitability $\\approx$ degree"),
                               (axB, "lump_r2", "lumpability step-$R^2$", "Lumpability survives")]:
        vals = {t: np.array([float(r[f"{t}_{key}"]) for r in rows]) for t in TYPES}
        ax.bar(x, [vals[t].mean() for t in TYPES], 0.62, color=COL, edgecolor="white", lw=1.0, zorder=3)
        for i, t in enumerate(TYPES):
            ax.scatter(x[i]+(rng.random(len(vals[t]))-0.5)*0.34, vals[t], s=11, color="white", edgecolor=INK, lw=0.5, zorder=6, alpha=0.85)
        ax.axhline(0, color=MUTED, lw=1.0)
        ax.set_xticks(x); ax.set_xticklabels(LAB, fontsize=8.3, color=INK)
        ax.set_ylabel(ylab, fontsize=9.5, color=INK); ax.set_title(ttl, fontsize=10.5, color=INK, pad=6)
        clean(ax)
    axB.set_ylim(0, 1.0)
    fig.suptitle("Did training create the structure?", fontsize=11, color=INK, y=0.99)
    save(fig, OUT / "panelD_training_control.svg")


if __name__ == "__main__":
    panel_eigenvector(); panel_by_size(); panel_jacobian(); panel_training_control()
