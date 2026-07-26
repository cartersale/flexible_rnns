# Figure map

Each figure panel → the module that produces it → the results it consumes.
Everything is run as `python -m flexsym.figures.<module>`.

**Important:** the `fig1`–`fig5` set maps to the project's earlier five-result
framing, not the current three claims. Claims 2 and 3 (motifs, quotient) have
**no paper figure set** — they exist only as slides. See "Coverage gap" below.

Shared styling lives in `figures.figstyle`.

**Modules write to `figN/`, but the curated set has since been re-sorted by
hand** — eight panels now live in `results/figures/supp/`. Re-running a figure
module writes back to `figN/`, so it will *not* overwrite the curated `supp/`
copy. Where the two differ, the table's "curated" column is authoritative for
what the current figure set actually looks like.

---

## Paper figures (current set)

### Fig 1 — the role structure exists — *claim 1*

`figures.fig1_panels` → `results/figures/fig1/`

| Panel | File | Curated in | Consumes |
|---|---|---|---|
| C | `panelC_quotient.svg` | **supp/** | `signed_input_wl_paper/<net>/` partition |
| D | `panelD_equitability_z.svg` | fig1 | γ z across seeds/sizes |
| E | `panelE_reproducibility.svg` | fig1 | `partition_stability/` restart ARI |
| F | `panelF_contingency.svg` | **supp/** | role vs activity partition contingency |

`figures.fig1_schematics` → `panelB_method` (kept in fig1), plus
`panelA_task_paradigm`, `panelB_refinement`, `panelB_ksweep` (all now in supp/).
`figures.real_task_paradigm` → a data-derived alternative to panel A, drawn from
the actual trial generator rather than by hand.

### Fig 2 — lumpability — *claim 1, the strongest part*

`figures.fig2_panels` → `results/figures/fig2/` — the one figure whose on-disk
set matches its module exactly.

| Panel | File | Consumes |
|---|---|---|
| A | `panelA_eigenvector.svg` | `lumpability/step_r2_all.csv` |
| B | `panelB_lumpability_by_size.svg` | same, grouped by size |
| C | `panelC_jacobian.svg` | `run.motif_jacobian_lumpability` output |
| D | `panelD_training_control.svg` | `training_control_all_sizes.csv` |

Panel D carries the negative result — equitability is degree-driven,
lumpability is the training signature. `figures.fig2bc` is a standalone variant
of panels B/C.

### Fig 3 — compositional reorganisation — *superseded framing*

Curated `fig3/`: `panelA_schematic`, `panelB_within_vs_cross`,
`panelC_reuse_matrix`, `panelD_shared_vs_nonshared`.

| Module | Writes | Note |
|---|---|---|
| `figures.fig3_panels` | `panelB_within_vs_cross`, `panelC_subpopulation_repertoire`, `panelD_shared_vs_nonshared` | `panelC_subpopulation_repertoire` moved to supp/ |
| `figures.fig3_reuse_matrix` | `panelC_reuse_matrix.svg` ← `epoch_reuse_matrix.json` | promoted into fig3 as panel C |
| `figures.fig34_schematics` | `panelA_schematic.svg` (fig3 + fig4) | |
| `figures.fig3_sankey` | `panelB_sankey.svg` ← `pair_fate_3x3.json` | **not in the current set** — generated but dropped |

### Fig 4 — context predicts function — *superseded, weak*

Curated `fig4/`: `panelA_schematic`, `panelB1_forest`, `panelC_epoch_type`.
`figures.fig4_panels` also writes `panelB2_k_robustness`, now in supp/.
Consumes `fair_context_predictor/`, `conn_func_regression/`.

Depicts the effect described in findings.md as weak and unreplicated. Do not
promote these panels without re-reading that section.
`figures.fig4b`, `figures.epoch_type_context_effect` are standalone variants.

### Fig 5 — causal lesions

Curated `fig5/`: `panelA_deficit_heatmap`, `panelB_alignment`,
`panelD_memory_motif`. `figures.fig5_panels` also writes `panelC_maxdrop`, now
in supp/. Consumes `run.lesion_causal_batch`, `run.lesion_memory_motif_v2`.

### Supplementary

`results/figures/supp/` holds the eight panels demoted from fig1–fig5:
`panelA_task_paradigm`, `panelB_ksweep`, `panelB_refinement`,
`panelB2_k_robustness`, `panelC_maxdrop`, `panelC_quotient`,
`panelC_subpopulation_repertoire`, `panelF_contingency`.

`results/figures/_legacy/` holds 40 SVGs from the prior figure generation.

---

## Diagnostic figures

Written to `results/figures/` (not a numbered subdirectory):

| Module | Shows |
|---|---|
| `figures.degree_check` | group-size distribution; degree-artefact check |
| `figures.training_control` | equitability vs lumpability by surrogate |
| `figures.gamma_size_hierarchy` | γ is size-weighted → the 1024 reversal |
| `figures.lumpability_eigenvector` | a single eigenvector, stepped on cells |
| `figures.degeneracy_figures` | cost-landscape near-degeneracy (why the partition is non-unique) |
| `figures.motif_validation` | validation gate + taxonomy table for recovered motifs |

---

## Slides

`python -m flexsym.slides.<module>` → `results/figures/slides/`

| Module | Claim | Shows |
|---|---|---|
| `equitability_degree` | 1 | raw equitability is degree-driven; lumpability is the signature |
| `lumpability_eigenvector` | 1 | eigenvector is a step function on groups |
| `lumpability_robustness` | 1 | holds across all sizes; dies on surrogates |
| `nonidentifiable_subspace` | 1 | labelling is non-identifiable, the subspace is not |
| `motif_validation` | 2 | taxonomy reproduces on released networks |
| `motif_inventory` | 2 | per-network motif inventory |
| `reuse_fails_control` | 2 | **the negative control** — reuse is real but not partition-specific |
| `motif_subspace` | 3 | motif directions lie in the connectivity subspace |
| `quotient_decomposition` | 3 | quotient predicts motif directions across k |
| `quotient_mechanism` | 3 | how many modes are needed; what carries the geometry |
| `network_grid` | — | dataset as size × seed grid ⚠️ needs `results/net_perf.json` |

---

## Coverage gap

The slide set is currently the **only** figure coverage for claims 2 and 3,
including the strongest result in the project (quotient predicts motif
direction) and the most important negative control (reuse fails its
random-partition control).

If the three-claim framing is the one going into the manuscript, a `fig6`/`fig7`
pair promoting `slides.quotient_decomposition` + `slides.motif_subspace` and
`slides.motif_inventory` is the missing piece — and `fig3`/`fig4` should be
reconsidered, since both depict superseded or weak results.
