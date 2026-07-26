"""Identifiability statistics for the restart-ensemble replication.

Consumes the per-(network, partition) effect tables produced by
``run_ensemble_nolesion.py`` (+ ``run_ensemble_lesion.py``) and answers the two
questions that decide whether "the connectivity partition" language survives
partition non-uniqueness:

  1. SIGN STABILITY. For each downstream effect, over the 10x20 (network x
     restart) grid: does the effect keep its expected sign, and does each
     network's restart-mean beat its size-matched random null? Rendered as a
     10x20 grid heatmap per effect.

  2. VARIANCE DECOMPOSITION. Fit ``effect ~ 1 + (1|network) + (1|network:restart)``
     (one value per network x restart => one-way random effects with network as
     group, restart as replicate) and report the intraclass correlation
     ICC_network = between-network / (between-network + between-restart). High
     ICC (>0.6) => the effect is a property of the NETWORK and the specific
     partition is an implementation detail (keeps "the"). Low ICC => the effect
     is partition-specific (retreat to "a connectivity-derived partition").

  3. Gamma DIAGNOSTIC. Within network, regress per-restart effect on per-restart
     Gamma_signed. Negative slope (lower cost -> stronger effect) licenses
     reporting the best-Gamma partition; null slope => the cost landscape is
     functionally flat.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
ENS = REPO / "results" / "signed_input_wl_paper" / "restart_ensemble"

# effect key -> (display name, expected-sign direction: +1 means larger=stronger)
EFFECTS = {
    "reorg_gap": ("Reorganization gap", +1),
    "partial_r": ("Connectivity->function partial r", +1),
    "lesion_align": ("Lesion deficit-activity alignment", +1),
    "gamma_z": ("Equitability z (reference)", -1),  # more negative = more equitable
    "step_r2_jacobian": ("Jacobian step-R^2 (reference)", +1),
}


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def merge() -> list[dict]:
    base = {(r["model"], r["kind"], r["index"]): dict(r)
            for r in load_rows(ENS / "ensemble_metrics_nolesion.csv")}
    for r in load_rows(ENS / "ensemble_metrics_lesion.csv"):
        key = (r["model"], r["kind"], r["index"])
        base.setdefault(key, dict(r)).update(
            {k: v for k, v in r.items() if k.startswith("lesion") or k == "intact_perf"}
        )
    return list(base.values())


def fval(r: dict, key: str) -> float:
    try:
        v = float(r.get(key, ""))
        return v
    except (ValueError, TypeError):
        return float("nan")


def grid_by_kind(rows: list[dict], effect: str, kind: str):
    """Return (models_sorted, grid[n_model x max_index+1]) for a partition kind."""
    models = sorted({r["model"] for r in rows}, key=lambda m: int(m.split("_")[-1]))
    idxs = sorted({int(r["index"]) for r in rows if r["kind"] == kind and int(r["index"]) >= 0})
    grid = np.full((len(models), len(idxs)), np.nan)
    for r in rows:
        if r["kind"] != kind:
            continue
        i = int(r["index"])
        if i < 0:
            continue
        gi = models.index(r["model"])
        gj = idxs.index(i)
        grid[gi, gj] = fval(r, effect)
    return models, grid


def one_way_icc(grid: np.ndarray) -> dict:
    """ICC(1) from a networks x replicates matrix (rows=groups). NaN-robust."""
    rows = [g[np.isfinite(g)] for g in grid]
    rows = [g for g in rows if g.size >= 2]
    if len(rows) < 2:
        return {"icc_network": float("nan"), "n_groups": len(rows)}
    grand = np.concatenate(rows).mean()
    group_means = np.array([g.mean() for g in rows])
    ns = np.array([g.size for g in rows])
    a = len(rows)
    N = int(ns.sum())
    ss_between = float(np.sum(ns * (group_means - grand) ** 2))
    ss_within = float(np.sum([((g - g.mean()) ** 2).sum() for g in rows]))
    df_b = a - 1
    df_w = N - a
    ms_b = ss_between / df_b
    ms_w = ss_within / df_w if df_w > 0 else float("nan")
    # unbiased average group size for unbalanced design
    n0 = (N - np.sum(ns**2) / N) / (a - 1)
    var_between = (ms_b - ms_w) / n0
    var_within = ms_w
    icc = var_between / (var_between + var_within) if (var_between + var_within) > 0 else float("nan")
    return {
        "icc_network": float(max(0.0, icc)) if np.isfinite(icc) else float("nan"),
        "var_between_network": float(max(0.0, var_between)),
        "var_between_restart": float(var_within),
        "ms_between": ms_b, "ms_within": ms_w, "n0": float(n0), "n_groups": a,
    }


def beats_null(rows: list[dict], effect: str, direction: int):
    """Per network: restart-mean vs the 20 size-matched random partitions."""
    models = sorted({r["model"] for r in rows}, key=lambda m: int(m.split("_")[-1]))
    out = []
    for m in models:
        restart = np.array([fval(r, effect) for r in rows
                            if r["model"] == m and r["kind"] == "restart"], dtype=float)
        rand = np.array([fval(r, effect) for r in rows
                         if r["model"] == m and r["kind"] == "random"], dtype=float)
        restart = restart[np.isfinite(restart)]
        rand = rand[np.isfinite(rand)]
        if restart.size == 0 or rand.size < 2:
            continue
        rm = restart.mean()
        z = direction * (rm - rand.mean()) / rand.std(ddof=1) if rand.std(ddof=1) > 0 else float("nan")
        # one-sided empirical p: fraction of random at least as extreme (in effect dir)
        if direction > 0:
            p = (np.sum(rand >= rm) + 1) / (rand.size + 1)
        else:
            p = (np.sum(rand <= rm) + 1) / (rand.size + 1)
        out.append({"model": m, "restart_mean": float(rm), "random_mean": float(rand.mean()),
                    "random_sd": float(rand.std(ddof=1)), "z": float(z), "p": float(p),
                    "beats": bool(p < 0.05)})
    return out


def gamma_regression(rows: list[dict], effect: str):
    """Within-network (network-demeaned) correlation of effect vs Gamma_signed."""
    xs, ys, per_net = [], [], []
    models = sorted({r["model"] for r in rows}, key=lambda m: int(m.split("_")[-1]))
    for m in models:
        g = np.array([fval(r, "gamma_restart") for r in rows
                      if r["model"] == m and r["kind"] == "restart"], dtype=float)
        e = np.array([fval(r, effect) for r in rows
                      if r["model"] == m and r["kind"] == "restart"], dtype=float)
        ok = np.isfinite(g) & np.isfinite(e)
        g, e = g[ok], e[ok]
        if g.size >= 3 and g.std() > 0 and e.std() > 0:
            r_net = float(np.corrcoef(g, e)[0, 1])
            per_net.append({"model": m, "r": r_net, "n": int(g.size)})
            xs.append(g - g.mean())
            ys.append(e - e.mean())
    if not xs:
        return {"pooled_within_r": float("nan"), "per_net": per_net}
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    r = float(np.corrcoef(X, Y)[0, 1]) if X.std() > 0 and Y.std() > 0 else float("nan")
    slope = float(np.polyfit(X, Y, 1)[0]) if X.std() > 0 else float("nan")
    # permutation p on pooled demeaned pairs
    rng = np.random.default_rng(0)
    null = np.array([np.corrcoef(X, rng.permutation(Y))[0, 1] for _ in range(2000)])
    p = float((np.sum(np.abs(null) >= abs(r)) + 1) / (len(null) + 1))
    return {"pooled_within_r": r, "pooled_within_slope": slope, "p": p,
            "per_net": per_net, "mean_per_net_r": float(np.mean([d["r"] for d in per_net]))}


def plot_sign_grid(models, grid, direction, name, path):
    signed = np.sign(grid) * (1 if direction > 0 else -1)  # +1 = expected sign
    fig, ax = plt.subplots(figsize=(7, 3.2))
    im = ax.imshow(signed, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xlabel("restart index")
    ax.set_ylabel("network (seed)")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.split("_")[-1] for m in models])
    frac = float(np.nanmean(signed > 0))
    ax.set_title(f"{name}: expected-sign grid  (green=expected, {frac:.0%} of {np.isfinite(grid).sum()} cells)")
    fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], shrink=0.8, label="sign (+1 expected)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return frac


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(ENS / "stability_stats"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = merge()
    available = [e for e in EFFECTS if any(np.isfinite(fval(r, e)) for r in rows)]
    print(f"merged {len(rows)} rows; effects available: {available}\n")

    summary = {}
    for effect in available:
        name, direction = EFFECTS[effect]
        models, grid = grid_by_kind(rows, effect, "restart")
        frac = plot_sign_grid(models, grid, direction, name, out / f"signgrid_{effect}.png")
        icc = one_way_icc(grid * direction)  # orient so larger=stronger for ICC scale-invariance
        icc_raw = one_way_icc(grid)
        bn = beats_null(rows, effect, direction)
        greg = gamma_regression(rows, effect)
        # consensus values
        cons = {r["model"].split("_")[-1]: fval(r, effect) for r in rows if r["kind"] == "consensus"}

        summary[effect] = {
            "name": name, "direction": direction,
            "sign_expected_fraction": frac,
            "icc_network": icc_raw["icc_network"],
            "var_between_network": icc_raw.get("var_between_network"),
            "var_between_restart": icc_raw.get("var_between_restart"),
            "n_networks_beating_null_p05": int(sum(d["beats"] for d in bn)),
            "n_networks_tested": len(bn),
            "median_beats_null_z": float(np.nanmedian([d["z"] for d in bn])) if bn else float("nan"),
            "gamma_within_r": greg["pooled_within_r"],
            "gamma_within_p": greg.get("p"),
            "gamma_mean_per_net_r": greg.get("mean_per_net_r"),
            "beats_null_per_net": bn,
            "gamma_per_net": greg["per_net"],
            "consensus_by_seed": cons,
        }
        print(f"=== {name} [{effect}] ===")
        print(f"  expected sign: {frac:.0%} of grid cells")
        print(f"  ICC_network = {icc_raw['icc_network']:.3f}  "
              f"(var_net={icc_raw.get('var_between_network'):.3g} vs "
              f"var_restart={icc_raw.get('var_between_restart'):.3g})")
        print(f"  beats size-matched random null: {sum(d['beats'] for d in bn)}/{len(bn)} networks "
              f"(median z={np.nanmedian([d['z'] for d in bn]):+.2f})")
        print(f"  within-network Gamma->effect r = {greg['pooled_within_r']:+.3f} "
              f"(p={greg.get('p')}, mean per-net r={greg.get('mean_per_net_r'):+.3f})\n")

    # ICC bar chart across effects
    eff_order = [e for e in available if e in ("reorg_gap", "partial_r", "lesion_align",
                                               "step_r2_jacobian", "gamma_z")]
    if eff_order:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        vals = [summary[e]["icc_network"] for e in eff_order]
        ax.bar(range(len(eff_order)), vals, color="#4477aa")
        ax.axhline(0.6, ls="--", c="k", lw=1, label="ICC=0.6 ('property of network')")
        ax.set_xticks(range(len(eff_order)))
        ax.set_xticklabels([EFFECTS[e][0].split(" (")[0] for e in eff_order], rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("ICC_network")
        ax.set_ylim(0, 1)
        ax.set_title("Between-network vs between-restart variance (ICC)")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "icc_by_effect.png", dpi=200)
        plt.close(fig)

    with (out / "stability_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True, default=float)
    print(f"wrote {out}/stability_summary.json and figures")


if __name__ == "__main__":
    main()
