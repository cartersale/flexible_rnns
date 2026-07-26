# Symmetry & Structural Cell Analysis Pipeline

Documentation for the extended analysis scripts in `stepnet/` on the `codex/flexible-multitask-symmetry-publish` branch. These scripts characterize **structural cells** — groups of recurrent units identified via signed Weisfeiler-Lehman (WL) graph partitioning — and analyze how they implement shared and task-specific computations.

---

## Overview

The pipeline flows roughly as follows:

```
Train network (stepnet/)
    ↓
Compute epoch-wise variance per unit
    ↓
Cluster units by activity profile → activity clusters
    ↓
Signed WL partitioning of W_rec → structural cells
    ↓
Per-cell analyses:
  • Epoch variance heatmaps
  • Functional sub-partitions (within-cell)
  • Context alignment (how much wiring context explains sub-partitions)
  • Cross-task fractionation (do sub-partitions change by task?)
  • Topological characterization (persistent homology)
  • Shapley importance (which cells enable which computations)
    ↓
Null models + post-hoc validation
    ↓
Plotting scripts
```

---

## 1. Variance & Epoch Analysis

These scripts compute how much each unit's activity varies across task epochs, then cluster units by those variance profiles.

### `compute_variance_epoch_to_file.py`
Runs the trained model to compute per-unit, per-task-epoch activation variance and writes the result to disk.
- **Input:** model checkpoint directory
- **Output:** `variance_epoch.pkl` — dict with `h_var_all`, `h_var_end`, `keys`

### `cluster_epoch_variance.py`
Clusters active units by their variance profiles across task epochs using hierarchical clustering with silhouette-based selection of the number of clusters.
- **Input:** `variance_epoch.pkl`
- **Output:** `epoch_variance_unit_clusters*.npz`, summary JSON

### `structural_cell_epoch_variance.py`
Averages per-unit normalized epoch variance over the units in each structural cell, producing a (n_cells × n_epochs) matrix.
- **Input:** activity cluster NPZ, structural sweep NPZ
- **Output:** `structural_cell_epoch_variance_k_{k}.npz`, CSV, heatmap figure

### `aggregate_structural_cell_epochs.py`
Collapses the (n_cells × n_rule_epoch) matrix by epoch type (e.g., `delay1`, `stim1`), averaging across all tasks that share the same epoch name. Accepts original/transpose/no_diag variants.
- **Input:** base results directory containing variant subdirectories
- **Output:** aggregated NPZ, CSV, PNG/PDF heatmap per variant

---

## 2. Signed WL Structural Roles

The core graph-partitioning analysis. Identifies structural cells by finding approximate equitable partitions of the signed recurrent weight graph.

### `signed_input_wl_structural_roles.py`
Computes approximate signed Weisfeiler-Lehman equitable partitions of `W_rec`. Sweeps over k (number of cells), computing gamma_EP (signed cost) for the structural partition, activity partition (from `cluster_epoch_variance.py`), and a random null. Also exports utility functions used by downstream scripts: `load_w_rec`, `apply_matrix_variant`, `gamma_signed`, `labels_to_H`.
- **Input:** model directory, activity cluster NPZ
- **Output:** `signed_input_wl_sweep.npz`, `signed_input_wl_sweep.csv`, contingency CSV, gamma/ARI figures

### `signed_input_wl_posthoc_checks.py`
Post-hoc robustness validation for the WL partition. Tests three perturbations: (1) removing the largest activity cluster, (2) splitting it internally, (3) coarsening the structural partition via quotient similarity.
- **Input:** model dir, activity cluster NPZ, transpose sweep directory
- **Output:** contingency CSVs per check, `posthoc_checks_summary.json`

---

## 3. Structural Cell Analysis

Per-cell analyses that characterize what structural cells do and how their computations relate to task demands.

### `structural_cell_activity_coherence.py`
Computes within-cell activity coherence for each structural cell and task period: mean pairwise correlation, top eigenvalue fraction, participation rank, and entropy rank. Also provides `compute_hidden_by_epoch` and `load_structural_labels` utility functions used throughout the pipeline.
- **Input:** model dir, activity cluster NPZ, structural sweep NPZ, cell variance NPZ
- **Output:** coherence NPZ, per-metric CSVs, heatmap figures, comparison CSV

### `structural_cell_subpartition_pilot.py`
Pilot analysis for functional sub-partitions within a single structural cell. Clusters units by activity trajectory (functional partition) and by internal recurrent wiring (structural partition), then compares them across MemoryPro, MemoryAnti, and DelayDM1 periods. Exports `choose_functional_partition` and `internal_structural_partition` used by many downstream scripts.
- **Input:** model dir, structural sweep NPZ, cell variance NPZ
- **Output:** partition strip figure, unit label CSV, comparison CSV, summary JSON

### `structural_cell_cross_task_fractionation.py`
Computes functional sub-partitions separately for each task epoch within each structural cell, then measures pairwise ARI/NMI between task-specific partitions. Classifies each cell as "stable", "intermediate", or "dynamic".
- **Input:** model dir, structural labels NPZ
- **Output:** `fractionation_matrix.csv`, `stability_classification.csv`, heatmap and stability figures, `fractionation_summary.json`

### `structural_cell_task_fingerprints.py`
Builds a fingerprint matrix of mean activity per structural cell × task epoch, clusters cells hierarchically, and classifies them into functional types (task-general, task-selective, epoch-selective, etc.).
- **Input:** model dir, structural labels NPZ
- **Output:** `fingerprints.npz`, `fingerprint_metrics.csv`, `cell_type_classification.csv`, heatmap PNG, summary JSON

---

## 4. Memory Sub-partition Analyses

Focused analyses on memory-selective structural cells, testing whether shared memory computations preserve the same within-cell functional organization.

### `softplus_memory_subpartition_sweep.py`
For each memory-selective structural cell, computes functional sub-partitions during MemoryPro, MemoryAnti, and DelayDM1 stimulus epochs, then measures ARI between them. Tests the hypothesis that shared memory cells maintain the same internal partition across tasks. Exports `unit_features_for_period`.
- **Input:** model dir, structural sweep NPZ, cell variance NPZ
- **Output:** `softplus_memory_subpartition_sweep.csv`, geometry PNG, summary JSON

### `softplus_memory_full_context_alignment.py`
Tests whether memory-cell functional sub-partitions align better with full-context wiring sub-partitions (signed recurrent input profiles from the whole network) than with size-matched shuffle nulls. Exports `choose_full_context_partition`.
- **Input:** model dir, structural sweep NPZ, cell variance NPZ
- **Output:** alignment CSV, unit-label CSV, summary CSV and JSON with z-scores/p-values

### `softplus_memory_progressive_context_alignment.py`
Traces how functional-structural alignment evolves as more structural cells are progressively added to the wiring context — starting from within-cell connections only and expanding to the full network.
- **Input:** model dir, structural sweep NPZ, cell variance NPZ
- **Output:** `progressive_context_alignment.json` with per-step ARI/NMI curves

---

## 5. Sub-partition Generalization

Extends memory sub-partition findings to all structural cells and all task epochs.

### `all_cell_subpartition_analysis.py`
For each cell × task epoch, tests whether trajectory-based functional partitions align with full-context and internal wiring partitions vs. size-matched shuffle nulls (ARI/NMI + z-scores).
- **Input:** model dir, activity cluster NPZ, structural sweep NPZ, cell variance NPZ
- **Output:** detail CSV, cell summary CSV, period-summary CSV, summary JSON

### `all_task_subpartition_comparison.py`
Extends `all_cell_subpartition_analysis.py` by additionally computing pairwise ARI/NMI between task-epoch sub-partitions within each cell, identifying which task epochs share the same functional organization.
- **Input:** model dir, activity cluster NPZ, structural sweep NPZ, cell variance NPZ
- **Output:** wiring comparison CSVs, pairwise comparison CSV, grouped summaries

### `compare_computational_components_k33.py`
Tests whether known computational components (e.g., all go1-response tasks, all delay1-memory tasks) reuse the same structural cells more than cross-component comparisons do, via Pearson r, cosine similarity, and Jaccard of top-k cells.
- **Input:** cell-variance NPZs for original/transpose/no_diag variants
- **Output:** pair-level CSV, summary CSV, bar figure, heatmap

### `compare_shared_vs_nonshared_task_epochs.py`
Compares structural-cell activity vectors for specific task-epoch pairs expected to be shared (e.g., MemPro delay vs. MemAnti delay) vs. different (e.g., ReactionGo go vs. DelayGo delay).
- **Input:** cell-variance NPZ
- **Output:** comparison CSV, heatmap PNG/PDF, bar-chart PNG/PDF

---

## 6. Persistent Homology

Topological characterization of within-cell state clouds during task epochs.

### `persistent_homology_motifs.py`
Core library for topological analysis. Provides: state cloud construction, Vietoris-Rips persistence (via ripser), diagram summaries (Betti numbers, lifetimes, entropy), null models (feature-shuffle H1, H0), motif topology characterization via `motif_topology`, canonical motif templates, and diagram matching via sliced-Wasserstein distance. No TF model loading — pure analysis library.

### `ph_method_gate.py`
Phase-0 validation. Confirms the PH pipeline recovers known topology on synthetic ground-truth clouds (circle, two blobs, sphere), and optionally confirms ring attractor detection on full-network MemoryPro activity. Exits with non-zero status on failure — run this before other PH scripts.
- **Output:** pass/fail status; optional diagnostic PNG

### `ph_category_diagnostic.py`
Characterizes full-network state clouds for memory-period candidates (delaygo, dmcgo, dmsgo) using H0 gap ratio, null-calibrated component count, k-means silhouette, and PCA scatter. Determines which tasks produce ring vs. multi-point attractors before committing to a Betti estimator.
- **Input:** model directory
- **Output:** diagnostic PNG, printed JSON

### `ph_reference_motif_library.py`
Builds a reference motif library by characterizing full-network state clouds for candidate task periods (ring, multi-point, plane, point candidates). Runs `motif_topology`, saves normalized persistence diagrams, and proposes reference sets for downstream matching.
- **Input:** model directory
- **Output:** `reference_motif_library.npz`, `reference_motif_library.json`, scatter PNG/PDF

### `ph_broken_symmetry_pilot.py`
Applies persistent homology to within-cell state clouds for a "broken symmetry" structural cell (one whose functional sub-partition diverges from its internal wiring partition). Computes H1 significance vs. shuffle null, bootstraps loop lifetime, and matches against the reference library.
- **Input:** model dir, structural sweep NPZ, cell variance NPZ
- **Output:** diagrams NPZ (clouds + H0/H1 per period), per-period CSV, summary JSON

---

## 7. Shapley / Cell Importance

Quantifies each structural cell's contribution to enabling functional-structural alignment via Monte Carlo Shapley values.

### `shapley_cell_importance.py`
Computes Shapley values for each structural cell's contribution to functional-structural alignment of memory cells. For each memory cell, measures how much adding a context cell improves ARI between the functional partition and the wiring-based partition.
- **Input:** model dir, structural sweep NPZ, cell variance NPZ
- **Output:** `shapley_cell_importance.json`

### `shapley_all_cells.py`
Extends Shapley analysis to all structural cells with meaningful functional substructure (not just memory cells), testing across multiple task epochs and selecting the best epoch per cell.
- **Input:** model dir, structural sweep NPZ
- **Output:** `shapley_all_cells.json`

---

## 8. Context Ordering Analysis

### `quotient_graph_distance_ordering.py`
Alternative progressive context analysis that orders context cells by their distance in the inter-cell quotient graph (built from block averages of `W_rec`). Tests whether alignment emerges from local-to-distant information flow. Supports shortest-path, weighted-shortest-path, and resistance distances.
- **Input:** model dir, structural sweep NPZ, cell variance NPZ
- **Output:** `quotient_distance_alignment.json`

---

## 9. Null Models

### `random_partition_activity_covariance_null.py`
Generates a null distribution for within-cell activity coherence metrics by randomly shuffling structural labels (preserving cell sizes) across active units many times, then comparing true structural partition metrics to the null. Also tests whether shared vs. non-shared task-period coherence deltas exceed chance.
- **Input:** model dir, activity cluster NPZ, structural sweep NPZ, structural coherence NPZ
- **Output:** global null CSV, delta null CSV, NPZ, summary JSON

---

## 10. Plotting Scripts

All plotting scripts are read-only with respect to model weights — they consume pre-computed result files and produce figures.

| Script | What it plots |
|--------|--------------|
| `plot_signed_wl_cost_curve.py` | gamma_EP cost vs. k, z-score profile with chosen k highlighted |
| `plot_structural_gamma_null_summary.py` | Null distribution histogram, contingency heatmap, overlap metric bar chart |
| `plot_structural_cell_epoch_variance_figures.py` | Panel comparison of structural-cell variance matrices across original/transpose/no_diag variants |
| `plot_structural_cell_trajectories.py` | Per-cell mean activation trajectories over epoch time, line + heatmap, coherence annotated |
| `plot_softplus_cell_pca_trajectories.py` | PCA scatter grids colored by functional sub-partition for selected cells across MemoryPro/MemAnti/DelayDM1 |
| `plot_softplus_cell_geometry_details.py` | Multi-panel geometry detail: 3D PCA, polar angular geometry, partition comparison, distance distributions |
| `plot_memory_subpartition_composition.py` | Two-column heatmap of shared (MemPro/MemAnti) vs. different (MemPro/DelayDM1) ARI per memory cell |
| `plot_ph_motifs.py` | Per-period three-panel figures: PCA scatter of within-cell cloud, persistence barcode, persistence diagram |
| `plot_progressive_context_alignment.py` | Alignment-vs.-context-size curves: individual cells, mean±SEM, ARI+NMI panels, incremental gain |
| `plot_shapley_cell_importance.py` | Shapley bar plots sorted by magnitude, histogram, horizontal bar chart of top-N cells per memory cell |
| `plot_cell_fractionation_examples.py` | Per-cell ARI/NMI heatmaps, sub-partition complexity bar, ARI distribution histogram, stability spectrum |
| `plot_cell_unit_reorganization.py` | Unit×feature heatmap with cluster boundaries, PCA projections, unit-tracking alluvial matrix, cluster-switch histogram |

---

## 11. Batch Runners & Wrappers

### `batch_softplus_analysis.py`
Full softplus analysis pipeline runner for seeds 1 and 2 (optionally seed 0). Runs in order: variance computation → activity clustering → signed-WL sweep (three matrix variants) → structural cell epoch variance → epoch aggregation → figure generation → component comparison → activity coherence → random-covariance null → memory sub-partition sweep. Resumable via skip logic on expected output files.

### `batch_256_transpose_analysis.py`
Config-driven batch runner for all 256-unit networks using the transpose signed-WL variant. Discovers models from a glob, selects k via saturation or active-fraction policy, then runs the full pipeline: variance → clustering → WL sweep → k-materialization → cell variance → functional sub-partition → full-context alignment. Checkpoints each model's status to JSON.

### `run_progressive_context_wrapper.py`
Resolves paths from `.env` and calls `softplus_memory_progressive_context_alignment.py` then `plot_progressive_context_alignment.py`.

### `run_shapley_all_cells.py`
Resolves paths and invokes `shapley_all_cells.py` with sensible defaults.

### `run_cross_task_fractionation.py`
Resolves paths and invokes `structural_cell_cross_task_fractionation.py`.

### `run_task_fingerprints.py`
Resolves paths and invokes `structural_cell_task_fingerprints.py`.

### `run_cell_unit_viz.sh`
Calls `plot_cell_unit_reorganization.py` for four example cells (14, 9, 12, 27) using hardcoded paths for the canonical softplus model.

### `run_progressive_context_analysis.sh`
Runs progressive context alignment then plotting using `$DATA_DIR` and `$HOME_DIR` environment variables.

---

## 12. Config Files

### `configs/batch_256_transpose_config.json`
Configures `batch_256_transpose_analysis.py` for all `Leaky*_softplus_*_256_all_*` models (seed 0 excluded). Uses `matrix_variant: "transpose"`, k range 3–60, saturation-based k selection (within 10% of minimum gamma z-score), 1000 WL nulls, and enables functional sub-partition with `n_null: 5000`.

### `configs/batch_rnn_softplus_diag_config.json`
Configures `batch_256_transpose_analysis.py` for `LeakyRNN_softplus_diag_*_all_*` models across all sizes and seeds (none excluded). Uses `matrix_variant: "transpose"` and `k_selection.policy: "active_fraction"` (20% of active units, rounded) rather than saturation-based selection.
