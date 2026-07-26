# Repo migration plan

Goal: replace this fork-of-a-fork with a clean, self-contained repo for the
symmetry paper. Upstream code is vendored (5 files), not inherited.

- **Old repo** (this one): `~/Research/projects/w_rec_rnns/flexible_multitask` —
  stays on disk untouched as a reference until the new repo is verified.
- **New repo**: `~/Research/projects/w_rec_rnns/flexmult_symmetry`
- **Package**: `flexsym`
- **Remote**: `git@github.com:cartersale/flexible_multitask_symmetry.git` (the
  current `symmetry` remote), fresh history.

## Decisions taken

| Question | Decision |
|---|---|
| Relation to upstream | New repo, vendor the 5 load-bearing files |
| Upstream artifacts (`analysis/` 145M, `figs/` 26M, `transfer_learn/`, `build/`, egg-info) | Drop entirely — public at `lauradriscoll/flexible_multitask` |
| Layout | Package + entry-point split (see deviation note below) |
| Manuscript | Out of scope — not in the code repo, not tracked here |

**Deviation on layout.** A literal `src/` + `scripts/` split does not work here:
~12 `run_*.py` files are imported by other scripts *and* are entry points
(`run_motif_fixed_points`, `run_motif_quotient_prediction`, `run_training_control`,
`run_lumpability_check`, `run_lesion_causal_batch`,
`run_connectivity_functional_regression`, `run_motif_participation_stats`,
`run_motif_reuse_control`, `run_motif_reuse_diagnostics`,
`run_motif_jacobian_lumpability`, `run_motif_quotient_controls`,
`restart_ensemble_partitions`). Splitting them forces sys.path hacks back in.

Instead: **everything lives in the package**, entry points keep their
`if __name__ == "__main__"` block and are invoked as
`python -m flexsym.run.motif_fixed_points`. Cross-imports become ordinary
package imports. Same organisational benefit, no dual-role problem.

---

## Phase 0 — Safety snapshot ✅ DONE

All 76 previously-untracked files and the 40 archive deletions are committed as
`13e0100 pre-migration`, pushed to `symmetry/main`, and tagged `pre-migration`.
`541b9c9` untracks the `stepnet/data` seed-99 checkpoints (moved to the data
volume) and gitignores `stepnet/data/` + `data/`. Working tree is clean.

Rollback point for everything below: `git reset --hard pre-migration`.

## Phase 1 — Scaffold the new repo

```
flexmult_symmetry/
  README.md              # what this is, relation to Driscoll 2024, how to run
  pyproject.toml         # replaces setup.py; installs `flexsym`
  .env.example           # DATA_ROOT, MODEL_ROOT, RESULTS_ROOT
  .gitignore             # results/ figs/ external/ data/ .env __pycache__/
  docs/
    methods.md           # from ANALYSIS.md, trimmed to kept scripts
    figure_map.md        # result/figure -> exact module that produces it
    findings.md          # from WALKTHROUGH_NOTES.md (R1-R5 + motif story)
    provenance.md        # upstream commit pinned for each vendored file
  src/flexsym/
    paths.py             # THE fix for the /Volumes hardcoding (Phase 3)
    core/                # vendored upstream, UNMODIFIED except imports
    training/
    structural/
    motifs/
    figures/
    run/                 # entry-point modules
    slides/
  configs/
  external/              # gitignored; FixedPointFinder setup documented in README
```

`git init`, no upstream remote. `origin` = the symmetry repo.

## Phase 2 — Move the code

### `core/` — vendored upstream (5 files)

`network.py`, `task.py`, `train.py`, `tools.py`, `analysis/variance.py`
(+ `__init__.py`). Verified closure: `variance` imports only task/network/tools;
nothing else upstream is reachable from your code.

`docs/provenance.md` records the upstream commit and states these are
unmodified apart from import rewrites.

**Dropped upstream leftovers** (in `stepnet/` but unused by your work):
`general_model_train.py`, `hp_sweep.py`, `leave_one_out.py`, `pre_transfer.py`,
`run_analyses.py`, `save_{pre,post}_train_model_params.py`, `tools_lnd.py`, and
the rest of `stepnet/analysis/` (clustering, dimensionality, taskset, …).

### `training/`

`train_softplus_diag.py`, `run_seeds_parallel.py`,
`compute_variance_epoch_to_file.py`, `cluster_epoch_variance.py`

### `structural/` — signed-WL partition machinery

`signed_input_wl_structural_roles.py`, `structural_partition_stability.py`,
`restart_ensemble_partitions.py`, `ensemble_analyze.py`, `ensemble_stats.py`,
`structural_cell_epoch_variance.py`, `structural_cell_activity_coherence.py`,
`structural_cell_cross_task_fractionation.py`, `structural_cell_lesions.py`,
`structural_cell_subpartition_pilot.py`, `within_task_split_half_null.py`,
`random_partition_activity_covariance_null.py`,
`softplus_memory_full_context_alignment.py`,
`softplus_memory_subpartition_sweep.py`, `subpartition_lesion_selectivity.py`

### `motifs/` — dynamical-motif pipeline

`fp_inventory.py`, `motif_classify.py`, `motif_participation.py`,
`motif_rotation_from_response.py`, `aggregate_motif_ensemble.py`

### `run/` — entry points (~35)

All `run_*.py` (prefix dropped: `run/motif_fixed_points.py`), plus
`batch_softplus_analysis.py`, `batch_256_transpose_analysis.py`.

### `figures/` — paper panels

`figstyle.py` + `plot_fig1_panels`, `plot_fig1_schematics`, `plot_fig2_panels`,
`plot_fig2bc`, `plot_fig3_panels`, `plot_fig3_reuse_matrix`, `plot_fig3_sankey`,
`plot_fig34_schematics`, `plot_fig4_panels`, `plot_fig4b`, `plot_fig5_panels`
(prefix dropped: `figures/fig1_panels.py`).

Diagnostic/supplementary plots also land here: `plot_degree_check`,
`plot_gamma_size_hierarchy`, `plot_lumpability_eigenvector`,
`plot_motif_validation`, `plot_training_control`,
`plot_epoch_type_context_effect`, `plot_real_task_paradigm`,
`plot_degeneracy_figures`.

### `slides/` — presentation figures (11)

All `plot_slide_*.py` + `plot_network_grid.py`.

### Not carried over

`stepnet/_archive/` (40 already-archived scripts) stays in the old repo only.
`stepnet/data/` — resolved: model checkpoints, moved to the data volume and
gitignored. Reached via `paths.py` (`MODEL_ROOT`), never carried into the repo.

## Phase 3 — Fix paths and imports (the real work)

This is where the reproducibility win is, and the main risk.

1. **`paths.py`.** 33 files hardcode `/Volumes/X9_Pro/...`; the dominant
   literals are `multitask_processed/suppfig4` (25 files) and
   `new_runs/staged` (22 files). Replace with resolved constants read from
   `.env` (`DATA_ROOT`, `MODEL_ROOT`, `RESULTS_ROOT`) with sane defaults, so
   the repo runs on a machine without the X9_Pro drive mounted.
2. **Kill the `sys.path` hacks.** 65 files carry them; all become unnecessary
   once `flexsym` is `pip install -e .`.
3. **Rewrite imports.** 20 files use bare `import tools` / `from network import
   Model` → `from flexsym.core import tools`. Cross-script imports → package
   paths.
4. **Preserve the `results/` symlinks** to `/Volumes/X9_Pro` (37G fixed points,
   1.6G responses). Layout unchanged; only how code *addresses* them changes.

Done as one scripted pass, not by hand, so it's reviewable as a single diff.

## Phase 4 — Verify before trusting

1. `python -m compileall src/` — syntax clean.
2. Import every module (`for m in ...: importlib.import_module(m)`) — catches
   every missed rewrite. Must be 100%.
3. Re-run 3 cheap analyses end-to-end and **byte-diff their outputs against the
   existing `results/`** — proves the move changed nothing. Candidates:
   `run_lumpability_check`, `run_wl_vs_strength_ari`, `run_equitability_soft`.
4. Regenerate one figure (`figures/fig2_panels`) and compare to the current SVG.

Migration is not "done" until step 3 passes.

## Phase 5 — Document, then commit

1. `docs/figure_map.md` — the table reviewers and co-authors need: each paper
   result/figure → the exact module that produces it → the results path it
   reads. Build this *from the verified pipeline*, not from memory.
2. README: setup, `.env`, the `external/` FixedPointFinder dependency, data
   access (Allen Institute buckets + which parts live on X9_Pro), and an
   explicit "this builds on Driscoll et al. 2024" pointer.
3. Initial commit, push to `symmetry`.
4. Only after Phase 4 passes and the push lands: retire the old clone (rename
   to `flexible_multitask_OLD`, keep on disk).

## Risks

- **Silent behaviour change from the path refactor** — a default that points
  somewhere plausible but wrong, so scripts run and produce subtly different
  numbers. This is the main risk. Phase 4 step 3 is the guard; do not skip it.
- **Import rewrite misses a dynamic import** (`importlib`, string module names).
  Phase 4 step 2 catches static ones; grep for `importlib`/`__import__` too.
- **Data volume unmounted** — 33 files assume `/Volumes/X9_Pro`. `paths.py`
  should fail loudly with a clear message, not silently fall back.
