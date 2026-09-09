"""Generate docs/tables/*.tex for supplementary.tex from results/main.

    python -m flexsym.supp.tables
Every table in the supplement is written by this script; edit the script, not
the .tex files.
"""
import json, numpy as np, pandas as pd
from pathlib import Path

from flexsym import paths

ROOT = paths.REPO_ROOT
R = paths.RESULTS_STORE/'main'
FP = R/'motif_fixed_points'
WL = R/'signed_input_wl_paper'
L = WL/'lumpability'
OUT = ROOT/'docs'/'tables'; OUT.mkdir(parents=True, exist_ok=True)
NETS = [f'LeakyRNN_softplus_diag_256_all_{s}' for s in range(3,13)]
SEEDS = list(range(3,13))

def W(name, s): (OUT/name).write_text(s.rstrip()+'\n'); print('wrote', name)
def f(x, n=2):
    return '--' if x is None or (isinstance(x,float) and not np.isfinite(x)) else f'{x:.{n}f}'
def med(a): a=np.asarray(a,float); return float(np.nanmedian(a))

# ---- S1 network inventory --------------------------------------------------
rows=[]
for size in (128,256,512):
    for seed in SEEDS:
        net=f'LeakyRNN_softplus_diag_{size}_all_{seed}'
        z=np.load(R/'activity_clusters_paper'/net/'epoch_variance_unit_clusters_paper_max.npz',allow_pickle=True)
        na=int(np.asarray(z['ind_active']).size); bk=int(z['best_k'])
        opk=max(3,min(min(bk,max(3,min(int(round(0.2*na)),na-1))),na-1))
        rows.append((size,seed,na,100*na/size,bk,float(z['best_score']),opk))
inv=pd.DataFrame(rows,columns=['size','seed','n_active','pct','best_k','sil','opk'])
lines=[r'\begin{tabular}{rrrrrrr}',r'\toprule',
 r'$N$ & seed & active units & \% active & activity $k^{\ast}$ & silhouette & operating $k$\\',r'\midrule']
for _,r in inv.iterrows():
    lines.append(f"{r['size']:.0f} & {r['seed']:.0f} & {r['n_active']:.0f} & {r['pct']:.1f} & {r['best_k']:.0f} & {r['sil']:.3f} & {r['opk']:.0f}\\\\")
    if r['seed']==12 and r['size']!=512: lines.append(r'\midrule')
lines += [r'\bottomrule',r'\end{tabular}']
W('tab_networks.tex','\n'.join(lines))

# ---- S2 tasks --------------------------------------------------------------
EPOCH_LABEL={'fix1':'fixation','stim1':'stimulus 1','delay1':'delay 1',
             'stim2':'stimulus 2','delay2':'delay 2','go1':'response'}

TASKS=[('fdgo','Go','pro, no delay',2000,'fixation 0--500; stimulus 1 500--1500; response 1500--2000'),
('reactgo','RT Go','pro, reaction time',2500,'fixation 0--2000; response 2000--2500'),
('delaygo','Dly Go','pro, delayed',2500,'fixation 0--500; stimulus 1 500--1000; delay 1 1000--2000; response 2000--2500'),
('fdanti','Anti','anti, no delay',2000,'fixation 0--500; stimulus 1 500--1500; response 1500--2000'),
('reactanti','RT Anti','anti, reaction time',2500,'fixation 0--2000; response 2000--2500'),
('delayanti','Dly Anti','anti, delayed',2500,'fixation 0--500; stimulus 1 500--1000; delay 1 1000--2000; response 2000--2500'),
('delaydm1','DM 1','compare mod.\\ 1',3000,'fixation 0--500; stimulus 1 500--1000; delay 1 1000--2000; stimulus 2 2000--2500; delay 2 2500--2700; response 2700--3000'),
('delaydm2','DM 2','compare mod.\\ 2',3000,'fixation 0--500; stimulus 1 500--1000; delay 1 1000--2000; stimulus 2 2000--2500; delay 2 2500--2700; response 2700--3000'),
('contextdelaydm1','Ctx DM 1','attend mod.\\ 1',3500,'fixation 0--500; stimulus 1 500--1000; delay 1 1000--2000; stimulus 2 2000--2500; delay 2 2500--3000; response 3000--3500'),
('contextdelaydm2','Ctx DM 2','attend mod.\\ 2',3500,'fixation 0--500; stimulus 1 500--1000; delay 1 1000--2000; stimulus 2 2000--2500; delay 2 2500--3000; response 3000--3500'),
('multidelaydm','Multi DM','integrate both mod.',3500,'fixation 0--500; stimulus 1 500--1000; delay 1 1000--2000; stimulus 2 2000--2500; delay 2 2500--3000; response 3000--3500'),
('dmsgo','DMS','match $\\to$ respond',2700,'fixation 0--500; stimulus 1 500--1000; delay 1 1000--2200; response 2200--2700'),
('dmsnogo','DNMS','match $\\to$ withhold',2700,'fixation 0--500; stimulus 1 500--1000; delay 1 1000--2200; response 2200--2700'),
('dmcgo','DMC','category match',2700,'fixation 0--500; stimulus 1 500--1000; delay 1 1000--2200; response 2200--2700'),
('dmcnogo','DNMC','category non-match',2700,'fixation 0--500; stimulus 1 500--1000; delay 1 1000--2200; response 2200--2700')]
lines=[r'\begin{tabular}{llp{3.1cm}rp{6.4cm}}',r'\toprule',
 r'task & abbreviation & computational demand & $T$ (ms) & epochs (ms)\\',r'\midrule']
for a,b,c,t,e in TASKS:
    lines.append(f"{a} & {b} & {c} & {t} & {e}\\\\")
lines+= [r'\bottomrule',r'\end{tabular}']
W('tab_tasks.tex','\n'.join(lines))

# ---- S3 motif taxonomy -----------------------------------------------------
tax=pd.read_csv(FP/'motif_taxonomy_all_nets.csv')
ct=pd.crosstab([tax.epoch_type,tax.epoch],tax.motif)
order=['line_attractor','saddle_decision','point_attractor','mixed']
ct=ct.reindex(columns=order,fill_value=0)
lines=[r'\begin{tabular}{llrrrrr}',r'\toprule',
 r'epoch type & epoch & line attractor & saddle & point attractor & unclassified & total\\',r'\midrule']
for (et,ep),r in ct.iterrows():
    lines.append(f"{'memory' if et=='memory' else 'stimulus'} & {EPOCH_LABEL.get(ep,ep)} & {r[order[0]]} & {r[order[1]]} & {r[order[2]]} & {r[order[3]]} & {r.sum()}\\\\")
tot=ct.sum()
lines+= [r'\midrule', f"\\multicolumn{{2}}{{l}}{{total}} & {tot[order[0]]} & {tot[order[1]]} & {tot[order[2]]} & {tot[order[3]]} & {tot.sum()}\\\\",
         r'\bottomrule',r'\end{tabular}']
W('tab_taxonomy.tex','\n'.join(lines))

# by decision rule
ct2=pd.crosstab([tax.epoch_type,tax.is_decision_rule],tax.motif).reindex(columns=order,fill_value=0)
lines=[r'\begin{tabular}{llrrrrr}',r'\toprule',
 r'epoch type & decision task & line attractor & saddle & point attractor & unclassified & total\\',r'\midrule']
for (et,dr),r in ct2.iterrows():
    lines.append(f"{'memory' if et=='memory' else 'stimulus'} & {'yes' if dr else 'no'} & {r[order[0]]} & {r[order[1]]} & {r[order[2]]} & {r[order[3]]} & {r.sum()}\\\\")
lines+=[r'\bottomrule',r'\end{tabular}']
W('tab_taxonomy_decision.tex','\n'.join(lines))

# ---- S4 lumpability k sweep -----------------------------------------------
k=pd.read_csv(L/'step_r2_ksweep.csv')
ks=[3,5,8,10,15,20,25,30,40,50,60]
lines=[r'\begin{tabular}{r'+'rrr'*3+r'}',r'\toprule',
 r'& \multicolumn{3}{c}{$N=128$} & \multicolumn{3}{c}{$N=256$} & \multicolumn{3}{c}{$N=512$}\\',
 r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
 r'$k$ & $R^2$ & null & $z$ & $R^2$ & null & $z$ & $R^2$ & null & $z$\\',r'\midrule']
for kk in ks:
    cells=[]
    for s in (128,256,512):
        q=k[(k['k']==kk)&(k['size']==s)]
        cells += [f(q.r2_topk.median()), f(q.r2_topk_null_mean.median()), f(q.z_topk.median(),1)]
    lines.append(f"{kk} & "+" & ".join(cells)+r"\\")
lines+=[r'\bottomrule',r'\end{tabular}']
W('tab_lump_ksweep.tex','\n'.join(lines))

# ---- S5 surrogates ---------------------------------------------------------
c=pd.read_csv(L/'control_ksweep.csv')
t=c[c.type=='trained'].set_index(['model','k'])
names={'trained':'trained','row-shuffle':'row shuffle (strength preserving)','full-shuffle':'full shuffle (value distribution)',
 'gaussian':'Gaussian (Frobenius matched)','lowrank-r5':'low rank $r=5$ + matched residual',
 'lowrank-r20':'low rank $r=20$ + matched residual','spectrum':'spectrum preserving (random Schur basis)'}
lines=[r'\begin{tabular}{lrrrr}',r'\toprule',
 r'surrogate & median $R^2_{\mathrm{step}}$ & median $z$ & trained higher & median excess\\',r'\midrule']
for ty,lab in names.items():
    s=c[c.type==ty].set_index(['model','k'])
    if ty=='trained':
        lines.append(f"{lab} & {f(s.r2_topk.median())} & {f(s.z_topk.median(),1)} & -- & --\\\\"); lines.append(r'\midrule'); continue
    j=t[['r2_topk']].join(s[['r2_topk']],rsuffix='_s',how='inner').dropna()
    lines.append(f"{lab} & {f(j.r2_topk_s.median())} & {f(s.z_topk.median(),1)} & {100*(j.r2_topk>j.r2_topk_s).mean():.1f}\\% & {f((j.r2_topk-j.r2_topk_s).median())}\\\\")
lines+=[r'\bottomrule',r'\end{tabular}']
W('tab_surrogates.tex','\n'.join(lines))

# ---- S6 equitability -------------------------------------------------------
e=pd.read_csv(L/'equitability_control_ksweep.csv')
tt=e[e.type=='trained'].set_index(['model','k']); rs=e[e.type=='row-shuffle'].set_index(['model','k'])
j=tt[['equit_score','gamma']].join(rs[['equit_score','gamma']],rsuffix='_r',how='inner')
lines=[r'\begin{tabular}{lrrr}',r'\toprule',
 r'matrix & $\Gamma_{\text{signed}}$ & structure score $S$ & higher $S$ than counterpart\\',r'\midrule',
 f"trained & {f(j.gamma.median())} & ${j.equit_score.mean():.1f}\\pm{j.equit_score.std():.1f}$ & {100*(j.equit_score>j.equit_score_r).mean():.1f}\\%\\\\",
 f"row shuffle & {f(j.gamma_r.median())} & ${j.equit_score_r.mean():.1f}\\pm{j.equit_score_r.std():.1f}$ & {100*(j.equit_score_r>j.equit_score).mean():.1f}\\%\\\\",
 r'\bottomrule',r'\end{tabular}']
W('tab_equitability.tex','\n'.join(lines))

# ---- S7 participation by motif type ---------------------------------------
bm=pd.read_csv(WL/'motif_participation'/'aggregate_by_motif.csv').set_index('motif')
lab={'all':'all','line_attractor':'line attractor','saddle_decision':'decision saddle','point_attractor':'point attractor'}
cols=[('n_cond_total','$n$',0),('eta2_role_median','$\\eta^2_{\\text{role}}$',2),('z_role_median','$z$',1),
 ('cv_M1_activity_median','$R^2_{\\text{act}}$',2),('cv_M3_role_median','$R^2_{\\text{role}}$',2),
 ('cv_M2_both_median','$R^2_{\\text{both}}$',2),('cv_unique_role_median','unique role',2),
 ('cv_unique_activity_median','unique act.',2),('cv_shared_median','shared',2),
 ('sub_step_r2_median','$R^2_{\\text{step}}$ role',2),('sub_z_role_median','$z$',1),
 ('sub_step_r2_activity_median','$R^2_{\\text{step}}$ act.',2),('sub_z_activity_median','$z$',1)]
lines=[r'\begin{tabular}{l'+'r'*len(cols)+r'}',r'\toprule',
 ' & '.join(['motif']+[c[1] for c in cols])+r'\\',r'\midrule']
for m in ['all','line_attractor','saddle_decision','point_attractor']:
    r=bm.loc[m]
    lines.append(' & '.join([lab[m]]+[('%d'%r[c] if n==0 else f(r[c],n)) for c,_,n in cols])+r'\\')
lines+=[r'\bottomrule',r'\end{tabular}']
W('tab_participation.tex','\n'.join(lines))

# ---- S8 quotient prediction across k ---------------------------------------
qp=json.load(open(FP/'quotient_prediction.json'))
ks=[8,12,16,20,25,30,40,50]
def grab(tag,m,kk,field):
    return [qp[n]['per_k'][str(kk)][f'{tag}_m{m}'][field] for n in NETS if str(kk) in qp[n]['per_k']]
lines=[r'\begin{tabular}{rrrrrrr}',r'\toprule',
 r'$k$ & $Q_{\text{struct}}$ & $Q_{\text{pred}}$ & $Q_{\text{task}}$ & random & activity & chance\\',r'\midrule']
for kk in ks:
    lines.append(f"{kk} & "+" & ".join([f(med(grab('struct',3,kk,'obs'))),f(med(grab('pred',3,kk,'obs'))),
        f(med(grab('task',3,kk,'obs'))),f(med(grab('pred',3,kk,'rand'))),f(med(grab('pred',3,kk,'act'))),
        f(np.sqrt(3/kk))])+r'\\')
lines+=[r'\bottomrule',r'\end{tabular}']
W('tab_quotient_ksweep.tex','\n'.join(lines))

# m-dependence at k=20
lines=[r'\begin{tabular}{rrrrrrr}',r'\toprule',
 r'$m$ & $Q_{\text{struct}}$ & $Q_{\text{pred}}$ & $Q_{\text{task}}$ & random & activity & chance\\',r'\midrule']
for m in (1,3,5):
    lines.append(f"{m} & "+" & ".join([f(med(grab('struct',m,20,'obs'))),f(med(grab('pred',m,20,'obs'))),
        f(med(grab('task',m,20,'obs'))),f(med(grab('pred',m,20,'rand'))),f(med(grab('pred',m,20,'act'))),
        f(np.sqrt(m/20))])+r'\\')
lines+=[r'\bottomrule',r'\end{tabular}']
W('tab_quotient_m.tex','\n'.join(lines))

# ---- S9 eigenvalue-matched control ----------------------------------------
em=json.load(open(FP/'quotient_eigmatched_control.json'))
lines=[r'\begin{tabular}{lrrrr}',r'\toprule',
 r'motif & $n$ & motif & eigenvalue-matched & gap\\',r'\midrule']
for mt,lb in [('line_attractor','line attractor'),('saddle_decision','decision saddle'),('point_attractor','point attractor')]:
    ns=[em[n]['per_k']['20']['by_type'][mt] for n in NETS if mt in em[n]['per_k']['20']['by_type']]
    lines.append(f"{lb} & {sum(x['n'] for x in ns)} & {f(med([x['motif'] for x in ns]))} & {f(med([x['eig_matched'] for x in ns]))} & {f(med([x['gap'] for x in ns]))}\\\\")
allm=[em[n]['per_k']['20'] for n in NETS]
lines+=[r'\midrule',
 f"all & {sum(x['n_cond'] for x in allm)} & {f(med([x['motif'] for x in allm]))} & {f(med([x['eig_matched'] for x in allm]))} & {f(med([x['gap'] for x in allm]))}\\\\",
 r'\bottomrule',r'\end{tabular}']
W('tab_eigmatched.tex','\n'.join(lines))

# ---- S10 held-out ----------------------------------------------------------
ho=json.load(open(FP/'quotient_heldout.json'))
lines=[r'\begin{tabular}{lrrr}',r'\toprule',
 r'split & conditions & $A_{\text{cond}}$ & random partition\\',r'\midrule']
for key,lb in [('in_sample','in sample (upper bound)'),
               ('interleaved','interleaved fixed points'),
               ('arc','held-out manifold arc'),
               ('cross_task','held-out task, same motif type'),
               ('cross_type','different motif type (negative control)')]:
    o=[ho[n]['20'][key]['obs'] for n in NETS]; r=[ho[n]['20'][key]['rand'] for n in NETS]
    nn=sum(ho[n]['20'][key]['n'] for n in NETS)
    lines.append(f"{lb} & {nn} & {f(med(o))} & {f(med(r))}\\\\")
lines+=[r'\midrule']
for mt,lb in [('line_attractor','line attractor'),('saddle_decision','decision saddle'),
              ('point_attractor','point attractor')]:
    v=[ho[n]['20']['arc_by_type'][mt] for n in NETS if mt in ho[n]['20']['arc_by_type']]
    lines.append(f"\\quad arc, {lb} & {sum(x['n'] for x in v)} & {f(med([x['obs'] for x in v]))} & --\\\\")
lines+=[r'\bottomrule',r'\end{tabular}']
W('tab_heldout.tex','\n'.join(lines))
print('heldout ks', list(ho[NETS[0]]))

# ---- S11 reuse -------------------------------------------------------------
oc=json.load(open(FP/'operator_composition.json'))
lines=[r'\begin{tabular}{rrrrrr}',r'\toprule',
 r'seed & $k$ & unit participation & quotient modes & activity clusters & random partition\\',r'\midrule']
for s,n in zip(SEEDS,NETS):
    a=oc[n]['A']
    lines.append(f"{s} & {oc[n]['k']} & {f(a['auc_node'])} & {f(a['auc_operator'])} & {f(a['auc_activity'])} & {f(a['auc_null_mean'])}\\\\")
lines+=[r'\midrule',
 f"median & -- & {f(med([oc[n]['A']['auc_node'] for n in NETS]))} & {f(med([oc[n]['A']['auc_operator'] for n in NETS]))} & {f(med([oc[n]['A']['auc_activity'] for n in NETS]))} & {f(med([oc[n]['A']['auc_null_mean'] for n in NETS]))}\\\\",
 f"low-overlap pairs & -- & {f(med([oc[n]['A']['low_node']['auc_node'] for n in NETS]))} & {f(med([oc[n]['A']['low_node']['auc_operator'] for n in NETS]))} & -- & --\\\\",
 r'\bottomrule',r'\end{tabular}']
W('tab_reuse.tex','\n'.join(lines))

# ---- S12 gain swap ---------------------------------------------------------
gc=json.load(open(FP/'gain_configuration.json'))
lines=[r'\begin{tabular}{rrrrrrr}',r'\toprule',
 r'seed & $k$ & role & activity & per unit & role-permuted & random partition\\',r'\midrule']
for s,n in zip(SEEDS,NETS):
    sw=gc[n]['swap']
    lines.append(f"{s} & {gc[n]['k']} & {f(sw['role']['pct'])} & {f(sw['activity']['pct'])} & {f(sw['unit']['pct'])} & {f(sw['shuffled_gains']['pct'])} & {f(sw['random_partition']['pct'])}\\\\")
g=lambda key: med([gc[n]['swap'][key]['pct'] for n in NETS])
lines+=[r'\midrule',
 f"median & -- & {f(g('role'))} & {f(g('activity'))} & {f(g('unit'))} & {f(g('shuffled_gains'))} & {f(g('random_partition'))}\\\\",
 r'\bottomrule',r'\end{tabular}']
W('tab_gainswap.tex','\n'.join(lines))

# ---- S13 susceptibility concentration + first-order validity ---------------
lines=[r'\begin{tabular}{rrrrrrr}',r'\toprule',
 r'seed & $k$ & role & activity & random & $r(S^{\rm ev}\!\cdot\!\Delta\bar g,\ \Delta|\lambda|)$ & permuted\\',r'\midrule']
for s,n in zip(SEEDS,NETS):
    su=gc[n]['susceptibility']; lv=gc[n]['leverage']
    r_obs=med([x['r_pred_obs'] for x in lv]); r_null=med([x['r_null'] for x in lv])
    lines.append(f"{s} & {su['k']} & {f(su['pr_role'],1)} & {f(su['pr_activity'],1)} & {f(su['pr_random'],1)} & {f(r_obs)} & {f(r_null)}\\\\")
allr=[x['r_pred_obs'] for n in NETS for x in gc[n]['leverage']]
alln=[x['r_null'] for n in NETS for x in gc[n]['leverage']]
lines+=[r'\midrule',
 f"median & -- & {f(med([gc[n]['susceptibility']['pr_role'] for n in NETS]),1)} & {f(med([gc[n]['susceptibility']['pr_activity'] for n in NETS]),1)} & {f(med([gc[n]['susceptibility']['pr_random'] for n in NETS]),1)} & {f(med(allr))} & {f(med(alln))}\\\\",
 r'\bottomrule',r'\end{tabular}']
W('tab_susceptibility.tex','\n'.join(lines))
print('n contrasts total', len(allr))

# ---- S14 allocation --------------------------------------------------------
gl=json.load(open(FP/'gain_leverage_geometry.json'))
def ratios(sens,cls,denom=None,nullkey='response_permsize'):
    """Median over networks of (observed response)/(null response) per contrast.

    nullkey selects the permutation null: 'response_permsize' is the
    size-stratified permutation, 'response_perm' the unrestricted one.
    """
    out=[]
    for n in NETS:
        vals=[]
        for c in gl[n]['contrasts']:
            k=f'{sens}_{cls}'
            if k not in c: continue
            obs=c[k]['response']; nul=c[k][nullkey]
            if denom:
                kd=f'{sens}_{denom}'
                obs=obs/c[kd]['response'] if c[kd]['response'] else np.nan
                nul=nul/c[kd][nullkey] if c[kd][nullkey] else np.nan
            if np.isfinite(obs) and np.isfinite(nul) and nul>0: vals.append(obs/nul)
        if vals: out.append(med(vals))
    return out
lines=[r'\begin{tabular}{llrrrr}',r'\toprule',
 r'sensitivity & mode class & size-matched null & unrestricted null & relative to bulk & networks $>1$\\',r'\midrule']
for sens,sl in [('ev','eigenvalue'),('rot','rotation')]:
    for cls,cl in [('star','motif-carrying'),('critical','near-critical'),('other','irrelevant bulk')]:
        a=ratios(sens,cls); rel=ratios(sens,cls,'other') if cls!='other' else None
        u=ratios(sens,cls,nullkey='response_perm')
        lines.append(f"{sl} & {cl} & {f(med(a))} & {f(med(u))} & {f(med(rel)) if rel else '--'} & {sum(x>1 for x in a)}/10\\\\")
    lines.append(r'\midrule')
lines=lines[:-1]+[r'\bottomrule',r'\end{tabular}']
W('tab_allocation.tex','\n'.join(lines))

# ---- S15 targeted modulation ----------------------------------------------
tm=json.load(open(FP/'gain_targeted_modulation.json'))
rows=pd.DataFrame(tm['rows'])
lines=[r'\begin{tabular}{lrrrrrrr}',r'\toprule',
 r'$\delta$ & exact & top 3 & median rank & fold vs median role & pred/actual & within $2\times$ & $r$ across roles\\',r'\midrule']
for key,lb in [('1.0','$\\delta$'),('0.25','$\\delta/4$')]:
    s=tm['summary'][key]
    lines.append(f"{lb} & {100*s['hit']:.0f}\\% & {100*s['top3']:.0f}\\% & {s['rank']:.1f} & {s['fold']:.1f}$\\times$ & {f(s['ratio'])} & {100*s['within2']:.0f}\\% & {f(s['rcol'])}\\\\")
lines.append(f"chance & {100*tm['summary']['1.0']['chance']:.1f}\\% & 7\\% & {tm['summary']['1.0']['rank_null']:.1f} & 1.0$\\times$ & -- & -- & --\\\\")
lines+=[r'\bottomrule',r'\end{tabular}']
W('tab_targeted.tex','\n'.join(lines))

# per-net targeted breakdown
r1=rows[rows.frac==1.0].copy()
r1['seed']=r1['key'].map(lambda x:x) if 'seed' not in r1 else r1['seed']
print('targeted row cols', rows.columns.tolist())
if 'net' in rows.columns: gcol='net'
else: gcol=None
if gcol:
    lines=[r'\begin{tabular}{rrrrrr}',r'\toprule',
     r'seed & conditions & exact & top 3 & median rank & $r$ across roles\\',r'\midrule']
    for n,g in r1.groupby(gcol):
        seed=n.split('_')[-1]
        lines.append(f"{seed} & {len(g)} & {100*(g['rank']==0).mean():.0f}\\% & {100*(g['rank']<3).mean():.0f}\\% & {g['rank'].median():.1f} & {f(g['r_col'].median())}\\\\")
    lines+=[r'\bottomrule',r'\end{tabular}']
    W('tab_targeted_pernet.tex','\n'.join(lines))
    # role-size control
    print('corr rank/size', np.corrcoef(r1['size'],np.abs(r1['act_star']))[0,1])
    print('median overlap', r1['ov_star'].median(), 'median lam', r1['lam'].median(), r1['lam'].quantile([.25,.75]).tolist())

# ---- S16 partition family --------------------------------------------------
pf=json.load(open(FP/'partition_family_comparison.json'))
lines=[r'\begin{tabular}{lrrr}',r'\toprule',
 r'partition & leakage & containment & top-3 alignment\\',r'\midrule']
for fam,lb in [('wl','signed WL (structural roles)'),('spectral','spectral clustering'),
               ('modularity','modularity (Louvain)'),('random','size-matched random')]:
    lines.append(f"{lb} & {f(med([pf[n]['20'][fam]['leakage'] for n in NETS]))} & {f(med([pf[n]['20'][fam]['containment'] for n in NETS]))} & {f(med([pf[n]['20'][fam]['top3'] for n in NETS]))}\\\\")
lines+=[r'\bottomrule',r'\end{tabular}']
W('tab_partition_family.tex','\n'.join(lines))

# ---- extra numbers ---------------------------------------------------------
sub=json.load(open(FP/'subspace_role_vs_activity.json'))['aggregate']['20']
conc=json.load(open(FP/'quotient_mode_concentration.json'))
print('subspace role vs activity k20', sub)
print('alpha median', med([qp[n]['alpha'] for n in NETS]))
print('lam_err k20 pred', med([qp[n]['per_k']['20']['lam_err']['obs'] for n in NETS]))
print('mode concentration k20 pr_conn', med([conc[n]['20']['pr_conn'] for n in NETS]),
      'rand', med([conc[n]['20']['pr_rand_mean'] for n in NETS]),
      'top', med([conc[n]['20']['top_conn'] for n in NETS]), med([conc[n]['20']['top_rand_mean'] for n in NETS]))

# ---- S1b network performance ----------------------------------------------
# Training ran a fixed step budget with no performance stopping criterion, so
# final performance has to be reported. The spread is across TASKS (go/anti at
# ceiling, the delayed-decision family near 0.95), not across sizes or seeds,
# so the table is task-by-size; the per-network range goes in the prose.
GRID_ROOT = paths.MODEL_ROOT/'flexsym_grid'
RULES = [t[0] for t in TASKS]
ABBR = {t[0]: t[1] for t in TASKS}
perf = {}
for size in (128, 256, 512):
    for seed in SEEDS:
        net = f'LeakyRNN_softplus_diag_{size}_all_{seed}'
        lg = json.load(open(GRID_ROOT/str(size)/net/'log.json'))
        perf[(size, seed)] = {r: float(lg['perf_'+r][-1]) for r in RULES}
        perf[(size, seed)]['_avg'] = float(lg['perf_avg'][-1])
        perf[(size, seed)]['_min'] = float(lg['perf_min'][-1])

def _cell(size, key):
    v = np.array([perf[(size, s)][key] for s in SEEDS], float)
    return f'{np.median(v):.3f} ({v.min():.3f})'

lines = [r'\begin{tabular}{llccc}', r'\toprule',
         r'task & abbrev. & $N=128$ & $N=256$ & $N=512$\\', r'\midrule']
for r in RULES:
    lines.append(f"{r} & {ABBR[r]} & " + " & ".join(_cell(s, r) for s in (128, 256, 512)) + r"\\")
lines += [r'\midrule',
          r'\multicolumn{2}{l}{\textbf{mean over tasks}} & '
          + " & ".join(_cell(s, '_avg') for s in (128, 256, 512)) + r'\\',
          r'\multicolumn{2}{l}{\textbf{worst task}} & '
          + " & ".join(_cell(s, '_min') for s in (128, 256, 512)) + r'\\',
          r'\bottomrule', r'\end{tabular}']
W('tab_performance.tex', '\n'.join(lines))

avg_all = [perf[k]['_avg'] for k in perf]
min_all = [perf[k]['_min'] for k in perf]
print(f"performance: mean-over-tasks range {min(avg_all):.3f}-{max(avg_all):.3f}; "
      f"worst single task-network {min(min_all):.3f}")

# ---- S13 released-cohort lumpability --------------------------------------
RL = paths.RESULTS_STORE/'released'/'signed_input_wl_paper'/'lumpability'/'step_r2_ksweep.csv'
rl = pd.read_csv(RL)
ARCH_LABEL = {
    'LeakyRNN_softplus_diag':      ('leaky RNN', 'softplus', 'diagonal'),
    'LeakyRNN_softplus_randgauss': ('leaky RNN', 'softplus', 'random'),
    'LeakyRNN_tanh_diag':          ('leaky RNN', 'tanh',     'diagonal'),
    'LeakyGRU_softplus_diag':      ('GRU',       'softplus', 'diagonal'),
    'LeakyGRU_softplus_randgauss': ('GRU',       'softplus', 'random'),
}
ORDER = ['LeakyRNN_softplus_diag', 'LeakyRNN_softplus_randgauss', 'LeakyRNN_tanh_diag',
         'LeakyGRU_softplus_diag', 'LeakyGRU_softplus_randgauss']
lines = [r'\begin{tabular}{lllrrrrrrr}', r'\toprule',
         r'cell & activation & init. & nets & $R^{2}_{\rm topk}$ & null & $z$ '
         r'& $R^{2}_{\rm best}$ & $z$ & $z>0$\\', r'\midrule']
for a in ORDER:
    g = rl[rl.arch == a]
    cell, act, ini = ARCH_LABEL[a]
    lines.append(
        f"{cell} & {act} & {ini} & {g.model.nunique()} & "
        f"{g.r2_topk.median():.3f} & {g.r2_topk_null_mean.median():.3f} & {g.z_topk.median():.1f} & "
        f"{g.r2_best.median():.3f} & {g.z_best.median():.1f} & "
        f"{(g.z_topk > 0).sum()}/{len(g)}\\\\")
lines += [r'\midrule',
          f"\\multicolumn{{3}}{{l}}{{\\textbf{{all released}}}} & {rl.model.nunique()} & "
          f"{rl.r2_topk.median():.3f} & {rl.r2_topk_null_mean.median():.3f} & {rl.z_topk.median():.1f} & "
          f"{rl.r2_best.median():.3f} & {rl.z_best.median():.1f} & "
          f"{(rl.z_topk > 0).sum()}/{len(rl)}\\\\",
          r'\bottomrule', r'\end{tabular}']
W('tab_released_lumpability.tex', '\n'.join(lines))

k5 = rl[rl.k >= 5]
print(f"released lumpability: {len(rl)} rows, {rl.model.nunique()} nets; "
      f"z_topk>0 in {(rl.z_topk>0).sum()}/{len(rl)} (k>=5: {(k5.z_topk>0).sum()}/{len(k5)}, "
      f"min z {k5.z_topk.min():.2f}); z_best>0 in {(rl.z_best>0).sum()}/{len(rl)}, "
      f"min {rl.z_best.min():.2f}")
print("  negative rows:", [(r.model, int(r.k), round(r.z_topk, 2)) for r in rl[rl.z_topk <= 0].itertuples()])

# ---- S17 quotient controls -------------------------------------------------
# Promised in the main text: gain ablation, connectivity ablation, alternative
# mode-selection rules (A_cond, k=20, r=3) and compression against
# partition-free predictors (A_raw, which is what full_space stores).
qcj = json.load(open(FP/'quotient_controls.json'))
def qcv(key, field='obs', r=3, kk='20', stat='cond'):
    return med([qcj[n]['per_k'][kk][f'{key}|r{r}'][stat][field] for n in NETS
                if f'{key}|r{r}' in qcj[n]['per_k'][kk]])
def qcf(key, r=3):
    return med([qcj[n]['full_space'][f'{key}|r{r}'] for n in NETS])
def arow(lb, key):
    return (f"{lb} & {f(qcv(key))} & {f(qcv(key,'rand'))} & {f(qcv(key,'act'))} & "
            f"{f(qcv(key,'z'),1)}\\\\")
lines=[r'\begin{tabular}{lrrrr}', r'\toprule',
       r'quotient & $A_{\text{cond}}$ & random & activity & $z$\\', r'\midrule',
       arow(r'$Q_{\rm pred}$ (role-mean gains)', 'Q_pred'),
       r'\midrule',
       arow(r'no gain ($Q_{\rm struct}$ alone)', 'no_gain'),
       arow(r'scalar (mean) gain', 'scalar_gain'),
       arow(r'gains without $Q_{\rm struct}$', 'gain_only'),
       arow(r'role-mean gains permuted across roles', 'perm_gain'),
       arow(r'gains from another condition', 'cross_gain'),
       arow(r'norm-matched random direction', 'rand_dir'),
       r'\midrule',
       arow(r'mode chosen by best match to the motif', 'sel_match'),
       arow(r'mode chosen on the motif itself', 'sel_motif'),
       r'\bottomrule', r'\end{tabular}']
W('tab_quotient_ablation.tex','\n'.join(lines))

lines=[r'\begin{tabular}{lrrrr}', r'\toprule',
       r'predictor & $r=1$ & $r=3$ & $r=5$ & $r=10$\\', r'\midrule']
for lb, key, full in [(r'$Q_{\rm pred}$ lifted from roles', 'Q_pred', False),
                      (r'random directions in $\mathrm{col}(H)$', 'rand_dir', False),
                      (r'full mean Jacobian', 'full_J', True),
                      (r'full mean Jacobian, eigenvalue-matched', 'full_J_match', True),
                      (r'full $W_{\rm rec}$', 'full_W', True),
                      (r'leading singular vectors of $W_{\rm rec}$', 'W_svd', True)]:
    vals = [qcf(key, r) if full else qcv(key, 'obs', r, stat='raw') for r in (1,3,5,10)]
    lines.append(f"{lb} & " + " & ".join(f(v) for v in vals) + r'\\')
lines+=[r'\midrule',
        f"containment ceiling & \\multicolumn{{4}}{{r}}{{{f(med([qcj[n]['per_k']['20']['containment']['raw']['obs'] for n in NETS]))}}}\\\\",
        r'\bottomrule', r'\end{tabular}']
W('tab_quotient_compression.tex','\n'.join(lines))

# ---- S18 mode routing ------------------------------------------------------
mr = pd.read_csv(L/'quotient_mode_motif_alignment.csv')
mr20 = mr[mr.k_req == 20]
def mrm(c): return float(np.nanmedian(mr20[c]))
lines=[r'\begin{tabular}{lrr}', r'\toprule',
       r'statistic & observed & null\\', r'\midrule',
       f"$\\rho$(containment, motif alignment) & {f(mrm('rho_c_a'))} & {f(mrm('rho_c_a_null_part'))}\\\\",
       f"\\quad partialling out $|\\lambda|$ & {f(mrm('rho_c_a_partial_lam'))} & {f(mrm('rho_c_a_null_part_partial'))}\\\\",
       f"\\quad across conditions & -- & {f(mrm('rho_c_a_null_cross'))}\\\\",
       f"$\\rho$(containment, $|\\lambda|$) & {f(mrm('rho_c_lam'))} & --\\\\",
       r'\midrule',
       f"motif energy in the $k$ most-contained modes & {f(mrm('span_routed'))} & {f(mrm('span_routed_null_part'))}\\\\",
       f"motif energy in the $k$ slowest modes & {f(mrm('span_slow'))} & --\\\\",
       f"motif energy in $k$ modes at random & {f(mrm('span_rand'))} & --\\\\",
       r'\bottomrule', r'\end{tabular}']
W('tab_routing.tex','\n'.join(lines))
print('routing n rows k20', len(mr20), 'nets', mr20.model.nunique(),
      'frac rho_c_a>0', float((mr20.rho_c_a > 0).mean()),
      'frac partial>0', float((mr20.rho_c_a_partial_lam > 0).mean()))

# ---- S19 composition -------------------------------------------------------
comp = [(n, key, d) for n in NETS for key, d in oc[n]['C'].items()]
def cm(field): return med([d[field] for _, _, d in comp])
lines=[r'\begin{tabular}{lrr}', r'\toprule',
       r'statistic & value & null\\', r'\midrule',
       f"composed conditions & {len(comp)} & --\\\\",
       f"energy in the two constituents & {f(cm('rho_AB'))} & {f(cm('null'))}\\\\",
       f"\\quad $z$ against the size-matched null & {f(cm('z'))} & --\\\\",
       f"best single constituent & {f(med([max(d['rho_A'], d['rho_B']) for _,_,d in comp]))} & --\\\\",
       f"gain over the best single constituent & {f(cm('gain_over_best_single'))} & --\\\\",
       f"balance between constituents & {f(cm('balance'))} & --\\\\",
       f"angle between constituents (deg) & {f(cm('angle_AB_deg'),1)} & --\\\\",
       r'\bottomrule', r'\end{tabular}']
W('tab_composition.tex','\n'.join(lines))
print('composition: z>0 in', sum(d['z'] > 0 for _,_,d in comp), 'of', len(comp),
      '; per-net median z', [round(med([d['z'] for d in oc[n]['C'].values()]),2) for n in NETS])

# ---- extra numbers for the prose ------------------------------------------
sr = pd.read_csv(L/'step_r2_ksweep.csv')
print('r2_best by size at k=20:',
      {int(sz): (round(g.r2_best.median(),3), round(g.z_best.median(),1))
       for sz, g in sr[sr.k == 20].groupby('size')})
print('r2_best overall: median', round(sr.r2_best.median(),3),
      'min z_best', round(sr.z_best.min(),1), 'z_best>0', int((sr.z_best>0).sum()), '/', len(sr))
print('eigmatched: gap_z med', round(med([em[n]['per_k']['20']['gap_z'] for n in NETS]),2),
      'nets with gap>0', sum(em[n]['per_k']['20']['gap'] > 0 for n in NETS),
      'frac cond motif>ctrl', round(med([em[n]['per_k']['20']['n_motif_gt_ctrl'] /
                                         em[n]['per_k']['20']['n_cond'] for n in NETS]),3))
print('fd_vs_analytic_r range', min(gl[n]['fd_vs_analytic_r'] for n in NETS),
      max(gl[n]['fd_vs_analytic_r'] for n in NETS))
from scipy.stats import pearsonr, spearmanr
r1t = rows[rows.frac == 1.0]
print('targeted role size vs realised shift  r/rho',
      round(pearsonr(r1t['size'], r1t['act_star'].abs())[0],3),
      round(spearmanr(r1t['size'], r1t['act_star'].abs())[0],3))
print('targeted role size vs predicted shift r/rho',
      round(pearsonr(r1t['size'], r1t['pred_star'].abs())[0],3),
      round(spearmanr(r1t['size'], r1t['pred_star'].abs())[0],3))
print('gain swap medians', {k: round(med([gc[n]['swap'][k]['pct'] for n in NETS]),3)
      for k in ('role','activity','unit','shuffled_gains','random_partition')})
print('reuse medians', {k: round(med([oc[n]['A'][k] for n in NETS]),3)
      for k in ('auc_node','auc_operator','auc_activity','auc_null_mean')},
      'low-overlap', {k: round(med([oc[n]['A']['low_node'][k] for n in NETS]),3)
      for k in ('auc_node','auc_operator')})
