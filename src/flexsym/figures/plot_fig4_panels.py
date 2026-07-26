"""Fig 4 (context predicts function) panels -> results/figures/fig4/*.svg"""
from __future__ import annotations
import sys, csv, glob
from pathlib import Path
import numpy as np
STEP = Path(__file__).resolve().parent
sys.path.insert(0, str(STEP))
import matplotlib.pyplot as plt
from figstyle import INK, MUTED, GRID, BLUE, YELLOW, GRAY, clean, save

REPO = STEP.parent
BASE = REPO / "results/signed_input_wl_paper"
OUT = REPO / "results/figures/fig4"
ET = {"stim1": "stimulus", "delay1": "delay /\nmemory", "go1": "response"}


def boot_ci(x, rng, n=2000):
    x = np.asarray(x)
    if len(x) < 2:
        return np.nan, np.nan
    bs = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    files = sorted(glob.glob(str(BASE / "conn_func_regression/conn_func_regression_LeakyRNN_softplus_diag_256_all_*.csv")))
    per_seed, per_k, per_k_int = {}, {}, {}
    for f in files:
        seed = int(f.split("_all_")[1].split(".")[0]); rows = list(csv.DictReader(open(f)))
        per_seed[seed] = [float(r["r_full_given_internal"]) for r in rows if r["r_full_given_internal"] not in ("", "nan")]
        for r in rows:
            per_k.setdefault(int(r["k"]), []).append(float(r["r_full_given_internal"]))
            per_k_int.setdefault(int(r["k"]), []).append(float(r["r_internal"]))
    seeds = sorted(per_seed); rng = np.random.default_rng(0)

    # (b1) forest
    fig, ax = plt.subplots(figsize=(3.8, 3.9)); fig.subplots_adjust(left=0.18, right=0.96, top=0.87, bottom=0.13)
    ax.axvline(0, color=INK, lw=1.2)
    for i, s in enumerate(seeds):
        y = len(seeds)-i; m = np.mean(per_seed[s]); lo, hi = boot_ci(per_seed[s], rng)
        col = BLUE if s <= 2 else "#6da7ec"
        ax.plot([lo, hi], [y, y], color=col, lw=2, zorder=3); ax.scatter([m], [y], s=40, color=col, edgecolor="white", lw=1.0, zorder=5)
    allm = np.array([np.mean(per_seed[s]) for s in seeds]); plo, phi = boot_ci(allm, rng)
    ax.plot([plo, phi], [0, 0], color=INK, lw=2.2); ax.scatter([allm.mean()], [0], s=130, marker="D", color=INK, edgecolor="white", lw=1.2, zorder=6)
    ax.set_yticks(list(range(len(seeds), 0, -1))+[0]); ax.set_yticklabels([f"seed {s}" for s in seeds]+["pooled"], fontsize=8, color=INK)
    ax.set_ylim(-0.8, len(seeds)+0.8); ax.set_xlabel(r"partial $r$ (function, full $|$ internal)", fontsize=9.5, color=INK)
    ax.set_title("Whole-network wiring predicts\nfunction beyond local wiring", fontsize=10, color=INK, pad=8)
    ax.text(0.98, 0.02, f"pooled $r$={allm.mean():.2f}, $P$=0.004", transform=ax.transAxes, ha="right", fontsize=8.3, color=INK)
    clean(ax, grid="x")
    save(fig, OUT / "panelB1_forest.svg")

    # (b2) k-robustness
    ks = sorted(per_k)
    mf = np.array([np.mean(per_k[k]) for k in ks]); sf = np.array([np.std(per_k[k], ddof=1)/np.sqrt(len(per_k[k])) for k in ks])
    mi = np.array([np.mean(per_k_int[k]) for k in ks]); si = np.array([np.std(per_k_int[k], ddof=1)/np.sqrt(len(per_k_int[k])) for k in ks])
    fig, ax = plt.subplots(figsize=(3.6, 3.4)); fig.subplots_adjust(left=0.16, right=0.96, top=0.87, bottom=0.15)
    ax.axhline(0, color=MUTED, lw=1.0)
    ax.fill_between(ks, mf-sf, mf+sf, color=BLUE, alpha=0.18); ax.plot(ks, mf, "-o", color=BLUE, lw=2.2, ms=6, mec="white", label="full-context | internal")
    ax.fill_between(ks, mi-si, mi+si, color=GRAY, alpha=0.18); ax.plot(ks, mi, "-o", color=GRAY, lw=2.0, ms=5, mec="white", label="internal alone")
    ax.set_xticks(ks); ax.set_xticklabels(ks, fontsize=9.5, color=INK); ax.set_xlabel("structural scale $k$", fontsize=10, color=INK)
    ax.set_ylabel(r"partial $r$ with function", fontsize=9.5, color=INK); ax.set_ylim(-0.02, 0.30)
    ax.set_title("Robust across role granularity", fontsize=10.5, color=INK, pad=8)
    ax.legend(loc="upper right", frameon=False, fontsize=8.3)
    clean(ax); save(fig, OUT / "panelB2_k_robustness.svg")

    # (c) epoch-type effect
    bf = sorted(glob.glob(str(BASE / "conn_func_regression_battery/conn_func_regression_LeakyRNN_softplus_diag_256_all_*.csv")))
    TYPES = ["stimulus", "delay /\nmemory", "response"]
    loc = {t: [] for t in TYPES}; ctrl = {t: [] for t in TYPES}
    for f in bf:
        rows = list(csv.DictReader(open(f)))
        for t in TYPES:
            rk = [r for r in rows if ET.get(r["period"].split("/")[1], "?") == t]
            if rk:
                loc[t].append(np.nanmean([float(r["r_full_given_internal"]) for r in rk]))
                ctrl[t].append(np.nanmean([float(r["r_full_given_int_win_rand"]) for r in rk]))
    from scipy.stats import t as tdist, wilcoxon
    fig, ax = plt.subplots(figsize=(4.4, 3.6)); fig.subplots_adjust(left=0.14, right=0.97, top=0.83, bottom=0.16)
    x = np.arange(3); w = 0.36; rng2 = np.random.default_rng(1)
    for off, data, col, lab in [(-w/2-0.01, loc, YELLOW, "beyond local wiring"), (+w/2+0.01, ctrl, BLUE, "beyond local + input + dim.")]:
        means = [np.mean(data[t]) for t in TYPES]
        cis = [tdist.ppf(0.975, len(data[t])-1)*np.std(data[t], ddof=1)/np.sqrt(len(data[t])) for t in TYPES]
        ax.bar(x+off, means, w, color=col, edgecolor="white", lw=1.1, zorder=3, label=lab)
        ax.errorbar(x+off, means, yerr=cis, fmt="none", ecolor=INK, elinewidth=1.3, capsize=3, zorder=5)
        for i, t in enumerate(TYPES):
            p = np.array(data[t]); ax.scatter(x[i]+off+(rng2.random(p.size)-0.5)*w*0.5, p, s=11, color="white", edgecolor=INK, lw=0.5, zorder=6, alpha=0.85)
        if col == BLUE:
            for i, t in enumerate(TYPES):
                pv = wilcoxon(data[t]).pvalue
                star = "**" if pv < 0.01 else "*" if pv < 0.05 else "n.s."
                ax.text(x[i]+off, means[i]+cis[i]+0.02, star, ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.axhline(0, color=INK, lw=1.1); ax.set_xticks(x); ax.set_xticklabels(TYPES, fontsize=9.5, color=INK)
    ax.set_ylabel(r"partial $r$ with function", fontsize=9.5, color=INK); ax.set_ylim(-0.05, 0.42)
    ax.set_title("Uniquely robust during memory", fontsize=11, color=INK, pad=8)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    clean(ax); save(fig, OUT / "panelC_epoch_type.svg")


if __name__ == "__main__":
    main()
