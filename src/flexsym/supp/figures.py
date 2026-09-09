"""Generate docs/figs/supp/*.svg for supplementary.tex.

    python -m flexsym.supp.figures

Each figure is also written as PDF, because pdflatex cannot embed SVG; the SVG
is the editable deliverable and the PDF is what \\includegraphics pulls in.

Figures whose inputs live on the X9_Pro volume (results/main/motif_fixed_points
is a symlink to it) are skipped with a message when the volume is not mounted.
"""
import json
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

from flexsym import paths

ROOT = paths.REPO_ROOT
R = paths.RESULTS_STORE
FP = R/'main'/'motif_fixed_points'
OUT = ROOT/'docs'/'figs'/'supp'; OUT.mkdir(parents=True, exist_ok=True)

# Categorical slots 1-5 of the validated default palette (light mode); assigned
# in fixed order, never cycled.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8880', '#e3e2dd'

mpl.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 110,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
    'axes.labelcolor': INK2, 'text.color': INK,
    'xtick.color': INK2, 'ytick.color': INK2,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
    'lines.linewidth': 1.4, 'legend.frameon': False,
    'svg.fonttype': 'none',          # keep text as text in the SVG
    'pdf.fonttype': 42,
})


def save(fig, stem):
    for ext in ('svg', 'pdf'):
        fig.savefig(OUT/f'{stem}.{ext}', bbox_inches='tight',
                    facecolor='white', transparent=False)
    plt.close(fig)
    print('wrote', f'{stem}.svg', '+ .pdf')


def panel_label(ax, letter):
    ax.text(-0.17, 1.14, letter, transform=ax.transAxes, fontsize=8.5,
            fontweight='bold', va='bottom', ha='left', color=INK)


# ---- S: released-cohort lumpability ---------------------------------------

# ---- S3: generalization across network size and architecture --------------
# Panels A-C: the principal effects of Figs. 1-3 at N = 128, 256, 512.
# Panel D: the same lumpability test on the released architectures.
SIZES = [(128, SERIES[0]), (256, SERIES[1]), (512, SERIES[2])]
FPJ = lambda stem, size: FP/(stem if size == 256 else f'{stem[:-5]}_{size}.json')

# 6.5in is \textwidth in the supplement, so the figure is drawn at final size
fig, axes = plt.subplots(1, 5, figsize=(6.5, 2.0), constrained_layout=True)

# A: lumpability across k, by network size
ax = axes[0]
sw = pd.read_csv(R/'main'/'signed_input_wl_paper'/'lumpability'/'step_r2_ksweep.csv')
for size, col in SIZES:
    g = sw[sw['size'] == size]
    q = g.groupby('k_req').r2_topk.quantile([.25, .5, .75]).unstack()
    ax.fill_between(q.index, q[.25], q[.75], color=col, alpha=0.16, lw=0)
    ax.plot(q.index, q[.5], color=col, label=f'$N$={size}')
nul = sw.groupby('k_req').r2_topk_null_mean.median()
ax.plot(nul.index, nul.values, color=MUTED, lw=1.0, ls=(0, (3, 2)), label='null')
ax.set_xlim(0, 62); ax.set_ylim(0, 0.62)
ax.set_xlabel('number of roles $k$'); ax.set_ylabel(r'$R^2_{\mathrm{step}}$')
ax.set_title('connectivity modes', loc='left', pad=3, color=INK)
ax.legend(loc='upper left', fontsize=5.2, handlelength=1.2, borderpad=0.15,
          labelspacing=0.2)
panel_label(ax, 'A')

# B: quotient alignment against k, by size (m = 3 leading role-level modes)
ax = axes[1]
for size, col in SIZES:
    d = json.load(open(FPJ('quotient_prediction.json', size)))
    nets = [n for n in d if n.startswith('LeakyRNN')]
    ks = sorted({int(k) for n in nets for k in d[n]['per_k']})
    obs = [np.median([d[n]['per_k'][str(k)]['struct_m3']['obs']
                      for n in nets if str(k) in d[n]['per_k']]) for k in ks]
    rnd = [np.median([d[n]['per_k'][str(k)]['struct_m3']['rand']
                      for n in nets if str(k) in d[n]['per_k']]) for k in ks]
    ax.plot(ks, obs, color=col, marker='o', ms=2.6, label=f'$N$={size}')
    ax.plot(ks, rnd, color=col, lw=0.9, ls=(0, (3, 2)), alpha=0.75)
ax.set_xlim(0, 62); ax.set_ylim(0, 1.0)
ax.set_xlabel('number of roles $k$'); ax.set_ylabel(r'$A_{\mathrm{cond}}$')
ax.set_title('motif direction', loc='left', pad=3, color=INK)
ax.text(60, 0.06, 'dashed: matched\nrandom partition', fontsize=5.6,
        color=INK2, ha='right', va='bottom')
panel_label(ax, 'B')

# C: gain swap, by size
ax = axes[2]
for j, (size, col) in enumerate(SIZES):
    gc = json.load(open(FPJ('gain_configuration.json', size)))
    pct = [gc[n]['swap']['role']['pct'] for n in gc if n.startswith('LeakyRNN')]
    jitter = np.random.default_rng(j).normal(0, 0.03, len(pct))
    ax.scatter(np.full(len(pct), j) + jitter, pct, s=9, facecolor='none',
               edgecolor=col, lw=0.7, zorder=3)
    ax.plot([j - 0.26, j + 0.26], [np.median(pct)] * 2, color=col, lw=1.8, zorder=4)
ax.axhline(0.5, color=INK2, lw=0.6, ls=(0, (3, 2)))
ax.set_xticks(range(len(SIZES))); ax.set_xticklabels([str(s) for s, _ in SIZES])
ax.set_xlim(-0.55, len(SIZES) - 0.45); ax.set_ylim(0.3, 1.0)
ax.set_xlabel('recurrent units $N$'); ax.set_ylabel('percentile')
ax.set_title('gain swap', loc='left', pad=3, color=INK)
ax.text(len(SIZES) - 0.5, 0.515, 'chance', fontsize=5.8, color=INK2,
        ha='right', va='bottom')
panel_label(ax, 'C')

# D: targeted modulation, by size (its own chance rate, 1/(k-1) per condition)
ax = axes[3]
for j, (size, col) in enumerate(SIZES):
    tm = json.load(open(FPJ('gain_targeted_modulation.json', size)))
    s1 = tm['summary']['1.0']
    ax.bar([j], [s1['hit']], width=0.52, color=col, alpha=0.85, lw=0, zorder=3)
    ax.plot([j - 0.3, j + 0.3], [s1['chance']] * 2, color=INK2, lw=1.0,
            ls=(0, (2, 1.6)), zorder=4)
    ax.text(j, s1['hit'] + 0.02, f"{s1['hit']*100:.0f}%", ha='center',
            va='bottom', fontsize=6, color=INK)
ax.set_xticks(range(len(SIZES))); ax.set_xticklabels([str(s) for s, _ in SIZES])
ax.set_xlim(-0.55, len(SIZES) - 0.45); ax.set_ylim(0, 0.72)
ax.set_xlabel('recurrent units $N$'); ax.set_ylabel('selection rate')
ax.set_title('targeted modulation', loc='left', pad=3, color=INK)
ax.text(len(SIZES) - 0.5, 0.055, 'chance', fontsize=5.8, color=INK2,
        ha='right', va='bottom')
panel_label(ax, 'D')

# E: released architectures, same lumpability test
ax = axes[4]
ARCH = [('LeakyRNN_softplus_diag',      'RNN/softplus/diag'),
        ('LeakyRNN_softplus_randgauss', 'RNN/softplus/rand'),
        ('LeakyRNN_tanh_diag',          'RNN/tanh/diag'),
        ('LeakyGRU_softplus_diag',      'GRU/softplus/diag'),
        ('LeakyGRU_softplus_randgauss', 'GRU/softplus/rand')]
rl = pd.read_csv(R/'released'/'signed_input_wl_paper'/'lumpability'/'step_r2_ksweep.csv')
for i, (arch, label) in enumerate(ARCH):
    g = rl[rl.arch == arch].groupby('k').r2_topk.median()
    ax.plot(g.index, g.values, color=SERIES[i], lw=1.2, label=label)
nul = rl.groupby('k').r2_topk_null_mean.median()
ax.plot(nul.index, nul.values, color=MUTED, lw=1.0, ls=(0, (3, 2)))
ax.set_xlim(0, 62); ax.set_ylim(0, 0.62)
ax.set_xlabel('number of roles $k$'); ax.set_ylabel(r'$R^2_{\mathrm{step}}$')
ax.set_title('released architectures', loc='left', pad=3, color=INK)
ax.legend(loc='upper left', fontsize=4.4, handlelength=1.1, borderpad=0.12,
          labelspacing=0.18)
panel_label(ax, 'E')

fig.set_constrained_layout_pads(w_pad=0.045, h_pad=0.02, wspace=0.02)
save(fig, 'fig_generalization')


# ---- S: restart reproducibility (labels vs subspace) ----------------------
rs = pd.read_csv(R/'main'/'signed_input_wl_paper'/'lumpability'/'restart_ari_vs_r2.csv')
KS = sorted(rs.k.unique())

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
ax = axes[0]
for i, k in enumerate(KS):
    g = rs[rs.k == k]
    ax.scatter(g.ari_mean, g.overlap_mean, s=22, facecolor=SERIES[i],
               edgecolor='white', linewidth=0.4, zorder=3, label=f'$k$={k}')
    ax.scatter(g.ari_rand_mean, g.overlap_rand_mean, s=16, facecolor='none',
               edgecolor=SERIES[i], linewidth=0.7, alpha=0.6, zorder=2)
ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.set_xlim(-0.05, 0.62); ax.set_ylim(0, 0.68)
ax.set_xlabel('adjusted Rand index between restarts')
ax.set_ylabel('subspace overlap between restarts')
ax.set_title('labels disagree, subspaces do not', loc='left', pad=3, color=INK)
ax.text(0.02, 0.235, 'size-matched random\npartitions (open)', fontsize=5.8,
        color=INK2, ha='left', va='top')
ax.legend(loc='lower right', fontsize=6, handletextpad=0.3, borderpad=0.3,
          labelspacing=0.3)
panel_label(ax, 'A')

ax = axes[1]
rng = np.random.default_rng(1)
for i, k in enumerate(KS):
    g = rs[rs.k == k]
    x = i + rng.uniform(-0.13, 0.13, len(g))
    ax.errorbar(x, g.r2_topk_mean, yerr=g.r2_topk_sd, fmt='o', ms=3,
                color=SERIES[i], ecolor=SERIES[i], elinewidth=0.8, capsize=1.5,
                mfc=SERIES[i], mec='white', mew=0.3, alpha=0.9, zorder=3)
    ax.plot([i-0.3, i+0.3], [g.r2_topk_rand_mean.median()]*2, color=MUTED,
            lw=1.4, ls=(0, (3, 2)), zorder=2)
ax.set_xticks(range(len(KS))); ax.set_xticklabels([f'$k$={k}' for k in KS])
ax.set_xlim(-0.5, len(KS)-0.5)
ax.set_ylabel(r'$R^2_{\mathrm{step}}$ across restarts')
ax.set_title('the derived statistic is stable', loc='left', pad=3, color=INK)
ax.text(len(KS)-0.52, rs[rs.k == KS[-1]].r2_topk_rand_mean.median() - 0.035,
        'matched null', fontsize=5.8, color=INK2, ha='right', va='top')
panel_label(ax, 'B')
fig.subplots_adjust(wspace=0.30)
save(fig, 'fig_restart_reproducibility')

# ---- figures that need the X9_Pro volume ----------------------------------
if not FP.exists():
    print('\nSKIPPED (X9_Pro not mounted, results/main/motif_fixed_points '
          'unreadable): fig_motif_direction, fig_reuse_composition')
    raise SystemExit

NETS = [f'LeakyRNN_softplus_diag_256_all_{s}' for s in range(3, 13)]
MOTIF = [('line_attractor', 'line attractor', SERIES[0]),
         ('saddle_decision', 'decision saddle', SERIES[1]),
         ('point_attractor', 'point attractor', SERIES[2])]

# ---- S: motif direction on roles vs activity clusters ----------------------
pm = pd.concat([pd.read_csv(FP/n/'participation_per_motif.csv').assign(net=n)
                for n in NETS], ignore_index=True)
null_med = pm.sub_null_r2.median()

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2),
                         gridspec_kw={'width_ratios': [1.25, 1]})
ax = axes[0]
lim = (0, 0.78)
ax.plot(lim, lim, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.axvspan(0, null_med, color=GRID, alpha=0.55, lw=0, zorder=0)
ax.axhspan(0, null_med, color=GRID, alpha=0.55, lw=0, zorder=0)
for key, label, col in MOTIF:
    g = pm[pm.motif == key]
    ax.scatter(g.sub_step_r2_activity, g.sub_step_r2, s=11, facecolor=col,
               edgecolor='white', linewidth=0.35, alpha=0.85, zorder=3,
               label=f'{label} ($n$={len(g)})')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r'$R^2_{\mathrm{step}}$ on activity clusters')
ax.set_ylabel(r'$R^2_{\mathrm{step}}$ on structural roles')
ax.set_title('signed motif direction, 471 conditions', loc='left', pad=3,
             color=INK)
ax.text(0.055, 0.735, 'size-matched\nnull level', fontsize=5.8, color=INK2,
        ha='left', va='top')
ax.legend(loc='lower right', fontsize=6, handletextpad=0.3, borderpad=0.3,
          labelspacing=0.35)
panel_label(ax, 'A')

ax = axes[1]
rng = np.random.default_rng(0)
for i, (key, label, col) in enumerate(MOTIF):
    g = pm[pm.motif == key]
    d = g.sub_z_role - g.sub_z_activity
    y = i + rng.uniform(-0.16, 0.16, len(d))
    ax.scatter(d, y, s=9, facecolor=col, edgecolor='white', linewidth=0.3,
               alpha=0.8, zorder=3)
    ax.plot([np.median(d)]*2, [i-0.3, i+0.3], color=INK, lw=1.6, zorder=4)
    ax.text(-3.5, i + 0.34, label, fontsize=6.2, color=col, ha='left',
            va='bottom')
ax.axvline(0, color=INK2, lw=0.7, zorder=2)
ax.set_yticks([]); ax.set_ylim(-0.6, 2.8)
ax.set_xlabel(r'$z_{\mathrm{role}} - z_{\mathrm{activity}}$')
ax.set_title('advantage over the activity partition', loc='left', pad=3,
             color=INK)
panel_label(ax, 'B')
fig.subplots_adjust(wspace=0.28)
plt.close(fig)   # dropped from the supplement (three figures only)

# ---- S: operator reuse and composition ------------------------------------
oc = json.load(open(FP/'operator_composition.json'))
ds = json.load(open(FP/'direct_sum_residual.json'))

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9),
                         gridspec_kw={'width_ratios': [1.15, 0.8, 1.15]})
REPS = [('auc_node', 'unit\nparticipation'), ('auc_activity', 'activity\nclusters'),
        ('auc_operator', 'quotient\nmodes'), ('auc_null_mean', 'random\npartition')]
ax = axes[0]
x = np.arange(len(REPS))
for n in NETS:
    ax.plot(x, [oc[n]['A'][k] for k, _ in REPS], color=MUTED, lw=0.6,
            alpha=0.75, marker='o', ms=2.2, mfc=MUTED, mec='none', zorder=2)
for j, (k, _) in enumerate(REPS):
    v = [oc[n]['A'][k] for n in NETS]
    ax.plot([j-0.22, j+0.22], [np.median(v)]*2, color=SERIES[0], lw=2, zorder=3)
ax.axhline(0.5, color=INK2, lw=0.7, ls=(0, (4, 3)))
ax.set_xticks(x); ax.set_xticklabels([lb for _, lb in REPS], fontsize=6)
ax.set_ylabel('AUC, same vs different motif type')
ax.set_title('all cross-task pairs', loc='left', pad=3, color=INK)
ax.set_ylim(0.45, 0.78)
panel_label(ax, 'A')

ax = axes[1]
for n in NETS:
    ax.plot([0, 1], [oc[n]['A']['low_node']['auc_node'],
                     oc[n]['A']['low_node']['auc_operator']],
            color=MUTED, lw=0.6, alpha=0.75, marker='o', ms=2.2, mfc=MUTED,
            mec='none', zorder=2)
for j, k in enumerate(('auc_node', 'auc_operator')):
    v = [oc[n]['A']['low_node'][k] for n in NETS]
    ax.plot([j-0.18, j+0.18], [np.median(v)]*2, color=SERIES[0], lw=2, zorder=3)
ax.axhline(0.5, color=INK2, lw=0.7, ls=(0, (4, 3)))
ax.set_xticks([0, 1]); ax.set_xticklabels(['unit\nparticipation', 'quotient\nmodes'],
                                          fontsize=6)
ax.set_xlim(-0.4, 1.4); ax.set_ylim(0.45, 0.78)
ax.set_title('lowest-overlap pairs', loc='left', pad=3, color=INK)
panel_label(ax, 'B')

ax = axes[2]
rs = sorted({r for n in NETS for t in ds[n]['triples'].values() for r in t['by_r']},
            key=int)
q = {}
for r in rs:
    v = [t['by_r'][r]['rho_AB'] - t['by_r'][r]['null']
         for n in NETS for t in ds[n]['triples'].values() if r in t['by_r']]
    q[int(r)] = np.percentile(v, [25, 50, 75])
xr = sorted(q)
ax.fill_between(xr, [q[r][0] for r in xr], [q[r][2] for r in xr],
                color=SERIES[1], alpha=0.18, lw=0)
ax.plot(xr, [q[r][1] for r in xr], color=SERIES[1], marker='o', ms=3)
ax.axhline(0, color=INK2, lw=0.7)
ax.set_xlabel('truncation rank $r$')
ax.set_ylabel('energy in constituents\n$-$ matched null')
ax.set_title('composition', loc='left', pad=3, color=INK)
panel_label(ax, 'C')
fig.subplots_adjust(wspace=0.42)
save(fig, 'fig_reuse_composition')
