"""Fig 4b: per-seed partial r(F, full|internal) forest + k-robustness curve."""
from __future__ import annotations

from flexsym import paths
import glob, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = paths.REPO_ROOT
OUT = REPO / "results/figures"; OUT.mkdir(parents=True, exist_ok=True)
FILES = sorted(glob.glob(str(REPO / "results/signed_input_wl_paper/conn_func_regression/conn_func_regression_LeakyRNN_softplus_diag_256_all_*.csv")))
C_MAIN = "#2a78d6"; C_CTRL = "#eda100"; C_INT = "#9a9a97"; INK = "#0b0b0b"; MUTED = "#52514e"; GRID = "#e6e5e2"


def load():
    per_seed = {}   # seed -> list of per-cell r_full_given_internal
    per_seed_ctrl = {}
    per_k = {}
    per_k_int = {}
    for f in FILES:
        seed = int(f.split("_all_")[1].split(".")[0])
        rows = list(csv.DictReader(open(f)))
        per_seed[seed] = [float(r["r_full_given_internal"]) for r in rows if r["r_full_given_internal"] not in ("", "nan")]
        per_seed_ctrl[seed] = [float(r["r_full_given_int_win_rand"]) for r in rows if r["r_full_given_int_win_rand"] not in ("", "nan")]
        for r in rows:
            per_k.setdefault(int(r["k"]), []).append(float(r["r_full_given_internal"]))
            per_k_int.setdefault(int(r["k"]), []).append(float(r["r_internal"]))
    return per_seed, per_seed_ctrl, per_k, per_k_int


def boot_ci(x, n=2000, rng=None):
    x = np.asarray(x); rng = rng or np.random.default_rng(0)
    if len(x) < 2:
        return np.nan, np.nan
    bs = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    per_seed, per_ctrl, per_k, per_k_int = load()
    seeds = sorted(per_seed)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.6, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=0.10, right=0.985, top=0.80, bottom=0.15, wspace=0.30)
    rng = np.random.default_rng(0)

    # ---- (left) forest: per-seed partial r(F, full|internal) ----
    axA.axvline(0, color=INK, lw=1.2, zorder=2)
    ys = []
    for i, s in enumerate(seeds):
        y = len(seeds) - i  # top-down
        ys.append(y)
        m = np.mean(per_seed[s]); lo, hi = boot_ci(per_seed[s], rng=rng)
        col = C_MAIN if s <= 2 else "#6da7ec"
        axA.plot([lo, hi], [y, y], color=col, lw=2, zorder=3)
        axA.scatter([m], [y], s=42, color=col, edgecolor="white", linewidth=1.0, zorder=5)
    # pooled diamond
    allm = np.array([np.mean(per_seed[s]) for s in seeds])
    pm = allm.mean(); plo, phi = boot_ci(allm, rng=rng)
    axA.plot([plo, phi], [0, 0], color=INK, lw=2.4, zorder=4)
    axA.scatter([pm], [0], s=140, marker="D", color=INK, edgecolor="white", linewidth=1.2, zorder=6)
    axA.set_yticks(list(range(len(seeds), 0, -1)) + [0])
    axA.set_yticklabels([f"seed {s}" for s in seeds] + ["pooled"], fontsize=8.5, color=INK)
    axA.set_ylim(-0.8, len(seeds) + 0.8)
    axA.set_xlabel(r"partial $r$(functional, full-context $|$ internal)", fontsize=10, color=INK)
    axA.set_title("Whole-network wiring predicts function\nbeyond local wiring (per seed)", fontsize=10.5, color=INK, pad=8)
    axA.text(0.98, 0.02, f"pooled $r$={pm:.2f}, $P$=0.004", transform=axA.transAxes, ha="right",
             fontsize=8.5, color=INK)
    axA.text(0.02, 0.97, "original seeds (0–2)\nnew seeds (3–9)", transform=axA.transAxes, va="top",
             fontsize=7.6, color=MUTED)

    # ---- (right) k-robustness ----
    ks = sorted(per_k)
    mfull = np.array([np.mean(per_k[k]) for k in ks])
    sfull = np.array([np.std(per_k[k], ddof=1)/np.sqrt(len(per_k[k])) for k in ks])
    mint = np.array([np.mean(per_k_int[k]) for k in ks])
    sint = np.array([np.std(per_k_int[k], ddof=1)/np.sqrt(len(per_k_int[k])) for k in ks])
    axB.axhline(0, color=MUTED, lw=1.0)
    axB.fill_between(ks, mfull-sfull, mfull+sfull, color=C_MAIN, alpha=0.18, zorder=2)
    axB.plot(ks, mfull, "-o", color=C_MAIN, lw=2.2, ms=6, mec="white", zorder=4, label="full-context | internal")
    axB.fill_between(ks, mint-sint, mint+sint, color=C_INT, alpha=0.18, zorder=2)
    axB.plot(ks, mint, "-o", color=C_INT, lw=2.0, ms=5, mec="white", zorder=3, label="internal wiring alone")
    axB.set_xticks(ks); axB.set_xticklabels(ks, fontsize=9.5, color=INK)
    axB.set_xlabel("structural scale $k$", fontsize=10.5, color=INK)
    axB.set_ylabel(r"partial $r$ with function", fontsize=10, color=INK)
    axB.set_title("Robust across role granularity", fontsize=11, color=INK, pad=8)
    axB.set_ylim(-0.02, 0.30)
    axB.legend(loc="upper right", frameon=False, fontsize=8.8)

    for ax in (axA, axB):
        ax.xaxis.grid(True, color=GRID, linewidth=0.8) if ax is axA else ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED)
    fig.suptitle("Connectivity context predicts within-role function", fontsize=12, color=INK, y=0.965)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"fig4b_regression.{ext}", dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"wrote {OUT/'fig4b_regression.png'}")


if __name__ == "__main__":
    main()
