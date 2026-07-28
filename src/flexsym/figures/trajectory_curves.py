"""The four trajectory curves against training step, diag init vs randortho control.

  (a) subspace agreement with the final network, against the restart-noise floor
  (b) incremental drift between adjacent snapshots
  (c) task performance
  (d) motif stabilisation -- slow points per epoch, and motif direction movement

Panels (a)-(c) come from trajectory_subspace.csv (every snapshot), panel (d) from
trajectory_motifs.csv (~5 snapshots). Lines are the mean over seeds within an
init; individual seeds are drawn faint behind them.

    python -m flexsym.figures.trajectory_curves
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from flexsym import paths
from flexsym.run.trajectory_subspace import REPORT_FLOOR_MSCA

RES = paths.RESULTS_ROOT / 'trajectory'
OUT = paths.FIGURES_ROOT
INK = '#0b0b0b'; MUTED = '#52514e'; GRID = '#e6e5e2'
COLORS = {'diag': '#2a78d6', 'randortho': '#eda100'}
LABELS = {'diag': 'diagonal init', 'randortho': 'random-orthogonal init'}


def _f(v):
    """CSV cell -> float or None ('' and 'None' both mean missing)."""
    return None if v in ('', 'None', None) else float(v)


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f'missing {path}; run the analysis scripts first.')
    with path.open() as f:
        return list(csv.DictReader(f))


def by_init_seed(rows, xkey, ykey, reduce_fn=None):
    """{init: {seed: (x, y)}}, sorted by x, missing y dropped.

    `reduce_fn` collapses several rows sharing an x (panel d has one row per
    rule x epoch per snapshot) into a single value.
    """
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        y = _f(r[ykey])
        x = float(r[xkey])
        # Drop the step-0 snapshot explicitly. The x-axis is log, so matplotlib
        # would drop it anyway -- but silently, leaving a curve that looks like it
        # starts at the first training snapshot when the init point exists in the
        # CSV. Better to exclude it on purpose and footnote it.
        if y is None or x <= 0:
            continue
        grouped[r['init']][int(r['seed'])][x].append(y)
    out = {}
    for init, seeds in grouped.items():
        out[init] = {}
        for seed, xs in seeds.items():
            x = np.array(sorted(xs))
            y = np.array([(reduce_fn or (lambda v: v[0]))(xs[xi]) for xi in x])
            out[init][seed] = (x, y)
    return out


def plot_series(ax, series, logx=True):
    """Faint per-seed lines plus the across-seed mean on the common x grid."""
    for init in sorted(series):
        c = COLORS.get(init, MUTED)
        per_seed = series[init]
        for x, y in per_seed.values():
            ax.plot(x, y, color=c, lw=0.8, alpha=0.28, zorder=2)
        # Nets share a snapshot grid by construction (fixed budget, same
        # --n-snapshots), but intersect rather than assume it: a net analysed
        # mid-training has fewer snapshots and would otherwise break the mean.
        common = sorted(set.intersection(*[set(x) for x, _ in per_seed.values()]))
        if not common:
            continue
        M = np.array([[y[list(x).index(xi)] for xi in common] for x, y in per_seed.values()])
        ax.plot(common, M.mean(0), color=c, lw=2.0, zorder=5,
                label=f'{LABELS.get(init, init)} (n={len(per_seed)})')
    if logx:
        ax.set_xscale('log')
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def main():
    sub = load(RES / 'trajectory_subspace.csv')
    mot_path = RES / 'trajectory_motifs.csv'
    mot = load(mot_path) if mot_path.exists() else []

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.92, bottom=0.08,
                        wspace=0.24, hspace=0.32)
    (axA, axB), (axC, axD) = axes

    # --- (a) subspace agreement with the final network ----------------------
    plot_series(axA, by_init_seed(sub, 'trials', 'msca_vs_final'))
    # Each init's own restart-noise floor -- the scale that decides whether S_t
    # has converged. The report's 0.63 is a diag-net number and is shown only as
    # a reference, not as the test.
    floors = defaultdict(list)
    for r in sub:
        floors[r['init']].append(float(r['floor_msca']))
    for init, vals in sorted(floors.items()):
        m = float(np.mean(vals))
        axA.axhline(m, color=COLORS.get(init, MUTED), ls='--', lw=1.1, alpha=0.75)
        axA.text(0.01, m, f'{init} floor {m:.2f}', transform=axA.get_yaxis_transform(),
                 ha='left', va='bottom', fontsize=7.5, color=COLORS.get(init, MUTED))
    axA.axhline(REPORT_FLOOR_MSCA, color=MUTED, ls=':', lw=1.1)
    axA.text(0.99, REPORT_FLOOR_MSCA, f'report floor {REPORT_FLOOR_MSCA}',
             transform=axA.get_yaxis_transform(), ha='right', va='bottom',
             fontsize=7.5, color=MUTED)
    axA.set_ylabel('mean squared principal-angle cosine\nwith the final subspace',
                   fontsize=9.5, color=INK)
    axA.set_title('(a) Subspace agreement with the final network',
                  fontsize=11, color=INK, pad=8, loc='left')
    axA.legend(loc='center right', frameon=False, fontsize=8.5)

    # --- (b) incremental drift ----------------------------------------------
    plot_series(axB, by_init_seed(sub, 'trials', 'msca_incremental'))
    axB.set_ylabel('msca between adjacent snapshots', fontsize=9.5, color=INK)
    axB.set_title('(b) Incremental drift', fontsize=11, color=INK, pad=8, loc='left')
    axB.text(0.02, 0.04, 'snapshots are log-spaced, so the step gap grows along x;\n'
                         'see log_step_gap in the CSV to normalise per unit training',
             transform=axB.transAxes, fontsize=7.5, color=MUTED)

    # --- (c) task performance ------------------------------------------------
    plot_series(axC, by_init_seed(sub, 'trials', 'perf_min'))
    for init, per_seed in by_init_seed(sub, 'trials', 'perf_avg').items():
        common = sorted(set.intersection(*[set(x) for x, _ in per_seed.values()]))
        M = np.array([[y[list(x).index(xi)] for xi in common] for x, y in per_seed.values()])
        axC.plot(common, M.mean(0), color=COLORS.get(init, MUTED), lw=1.4, ls='--',
                 zorder=4, label=f'{LABELS.get(init, init)}, mean over tasks')
    axC.set_ylabel('performance', fontsize=9.5, color=INK)
    axC.set_title('(c) Task performance (solid: worst task; dashed: mean)',
                  fontsize=11, color=INK, pad=8, loc='left')
    axC.legend(loc='lower right', frameon=False, fontsize=7.5)

    # --- (d) motif stabilisation --------------------------------------------
    if mot:
        # One row per rule x epoch per snapshot: average over them for the count,
        # and average the direction movement the same way.
        plot_series(axD, by_init_seed(mot, 'trials', 'n_fixed_points',
                                      reduce_fn=lambda v: float(np.mean(v))))
        axD.set_ylabel('slow points found per epoch (mean)', fontsize=9.5, color=INK)
        axD2 = axD.twinx()
        for init, per_seed in by_init_seed(
                mot, 'trials', 'angle_vs_final_deg',
                reduce_fn=lambda v: float(np.mean(v))).items():
            common = sorted(set.intersection(*[set(x) for x, _ in per_seed.values()]))
            M = np.array([[y[list(x).index(xi)] for xi in common]
                          for x, y in per_seed.values()])
            axD2.plot(common, M.mean(0), color=COLORS.get(init, MUTED), lw=1.6,
                      ls='--', zorder=4)
        axD2.set_ylabel('motif direction movement vs final (deg)',
                        fontsize=9.5, color=MUTED)
        axD2.spines['top'].set_visible(False)
        axD.legend(loc='upper right', frameon=False, fontsize=8.5)
        axD.text(0.02, 0.04, 'solid: slow-point count (left)\ndashed: direction angle (right)',
                 transform=axD.transAxes, fontsize=7.5, color=MUTED)
    else:
        axD.text(0.5, 0.5, 'no trajectory_motifs.csv\nrun flexsym.run.trajectory_motifs',
                 ha='center', va='center', transform=axD.transAxes,
                 fontsize=9.5, color=MUTED)
        axD.set_xscale('log')
    axD.set_title('(d) Motif stabilisation', fontsize=11, color=INK, pad=8, loc='left')

    for ax in (axA, axB, axC, axD):
        ax.set_xlabel('training trials', fontsize=9.5, color=INK)

    fig.suptitle('When does the connectivity coordinate system settle, '
                 'relative to the motifs inside it?', fontsize=12.5, color=INK)
    fig.text(0.5, 0.005, 'x is log-scaled, so the step-0 (initialisation) snapshot '
                         'is omitted from every panel; it is present in the CSVs.',
             ha='center', fontsize=7.5, color=MUTED)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'trajectory_curves.{ext}', dpi=200)
    print(f"wrote {OUT / 'trajectory_curves.png'} (+ .pdf)")


if __name__ == '__main__':
    main()
