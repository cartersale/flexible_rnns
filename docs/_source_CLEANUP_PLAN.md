# stepnet/ cleanup plan

Goal: a clean, methods-traceable `stepnet/` for the paper. Scripts that don't
feed the paper's results move to `stepnet/_archive/` (git-ignored — kept locally,
out of the tracked repo). Nothing is deleted; archiving is fully reversible.

Method: every `.py` was classified by (a) module docstring, (b) the `results/`
and `figs/` paths it produces/consumes, (c) git recency, cross-referenced with
`ANALYSIS.md`, `WALKTHROUGH_NOTES.md` (the 5-result story R1–R5), and the new
dynamical-motif pipeline. An import check confirmed no kept file imports any
archived file (one catch: `structural_cell_epoch_variance.py` is imported by
`run_alignment_stability.py`, so it stays).

## Tiers (121 root files)

| Tier | What | Count | Disposition |
|---|---|---|---|
| A | Core infra (model/task/train/tools + shared libs + `analysis/`) | 23 (+16 in `analysis/`) | **keep** |
| B | Dynamical-motif pipeline (new) | 10 | **keep** |
| C | Structural-paper R1–R5 (analyses, current `plot_fig*` figures, controls, reviewer replies) | 42 | **keep** |
| D | Exploratory / superseded | 40 | **archive** |
| E | Uncertain (needs author call) | 6 | **decide** |

## Key facts that shaped this

- **Two figure generations.** The short `plot_fig1..5_panels` family (untracked,
  newest) is the CURRENT figure set, maps 1:1 to R1–R5, writes `results/figures/`.
  The verbose `plot_figure*` family (2026-06-30) is the prior generation, writes a
  *different* tree (`results/paper_figures/`) — superseded, safe to archive.
- **Persistent homology** (`ph_*`, `plot_ph_motifs`, `persistent_homology_motifs`)
  is not among R1–R5 and is self-contained (its library has no callers outside the
  cluster). Dead direction unless it's going in the paper.
- **Shapley** (`shapley_*`, `run_shapley_all_cells`, `plot_shapley_cell_importance`)
  is not among R1–R5 and nothing kept depends on it. Dead direction.
- **Abandoned R4 line:** the progressive-context alignment scripts were superseded
  by the fair-context predictor + connectivity regression (current R4).

## Archive buckets (Tier D + E), all verified safe (no kept importer)

**B1 — Prior-generation figure scripts (6):** plot_figure1_structural_panels,
plot_figure2_compositional_panels, plot_figure3_subpartition_panels,
plot_figure3_all_epoch_panels, plot_figure4_all_epoch_panels,
plot_figure4_embedding_predictor_panels.

**B2 — Persistent homology (6):** persistent_homology_motifs, ph_method_gate,
ph_category_diagnostic, ph_reference_motif_library, ph_broken_symmetry_pilot,
plot_ph_motifs.

**B3 — Shapley (4):** shapley_all_cells, shapley_cell_importance,
run_shapley_all_cells, plot_shapley_cell_importance.

**B4 — Superseded exploratory analyses & one-off plots (24):**
run_cross_task_fractionation, softplus_memory_progressive_context_alignment,
run_progressive_context_wrapper, quotient_graph_distance_ordering,
plot_progressive_context_alignment, softplus_all_epoch_full_context_alignment,
all_cell_subpartition_analysis, all_task_subpartition_comparison,
plot_memory_subpartition_composition, plot_k_sweep_validation_result,
plot_signed_wl_cost_curve, plot_softplus_cell_geometry_details,
plot_softplus_cell_pca_trajectories, plot_structural_cell_trajectories,
plot_structural_cell_epoch_variance_figures, plot_structural_gamma_null_summary,
plot_cell_fractionation_examples, plot_cell_unit_reorganization,
structural_cell_task_fingerprints, run_task_fingerprints,
aggregate_structural_cell_epochs, compare_computational_components_k33,
compare_shared_vs_nonshared_task_epochs, signed_input_wl_posthoc_checks.

**B5 — Uncertain / author call (6):** subpartition_lesion_selectivity (recent;
possible R5 extension), random_partition_activity_covariance_null (possible
helper), hp_sweep, leave_one_out, pre_transfer, run_analyses (original-repo
scaffolds, no docstrings).

## Restore

Everything moves to `stepnet/_archive/` (git-ignored). To restore any file:
`git mv`? No — these aren't tracked in the archive; just `mv stepnet/_archive/X.py stepnet/`.
A manifest (`stepnet/_archive/ARCHIVED.md`) records what moved, from where, and why.

## Follow-up cleanup (after archiving)

1. Update `ANALYSIS.md` — it documents the old pipeline (incl. PH/Shapley/
   progressive-context). Trim to the kept scripts, or split into
   `docs/methods_structural.md` + `docs/methods_motif.md`.
2. Add a **methods map**: a table mapping each paper result/figure → the exact
   scripts that produce it (R1→plot_fig1_panels + producers, … motif→…). This is
   what reviewers/co-authors need and what makes the repo self-documenting.
3. Consider committing the untracked keepers (Tier B + C) so the paper repo has
   the real pipeline under version control (currently much of it is uncommitted).
4. Document the `external/` dependency (FixedPointFinder + recurrent-whisperer)
   setup in the README so the motif pipeline is reproducible.
