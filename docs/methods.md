# Methods

What each analysis does, organised by the three claims. For what the analyses
*found*, see [findings.md](findings.md); for figure provenance, see
[figure_map.md](figure_map.md).

Every module is run as `python -m flexsym.<subpackage>.<module>`.

## The three claims

1. **Connectivity defines a dynamical role structure.** The recurrent weight
   matrix partitions into groups of units that wire to the rest of the circuit
   the same way, and those groups are the natural coordinate system of the
   linear dynamics.
2. **The network's computational motifs are organised by that structure.** The
   fixed-point motifs the network uses to compute recruit role groups
   non-uniformly.
3. **The quotient predicts the motifs.** The coarse-grained circuit obtained by
   collapsing each role group to one node predicts the *directions* the motifs
   occupy in state space.

Claims 1 and 2 rest on connectivity and dynamics respectively; claim 3 is the
bridge, and is the strongest positive result in the repo.

---

## Shared foundations

### The partition object

`structural.signed_input_wl_structural_roles` — the core method. Signed
Weisfeiler–Leman refinement on `W_rec`: iteratively relabel each unit by the
multiset of labels it connects to, treating excitatory and inhibitory weights
separately, then k-means in the resulting embedding. Produces a partition into
`k` **role cells** and the equitability cost γ (distance from a perfectly
equitable partition; lower is better).

γ is reported as a z-score against size-matched random partitions. Negative z =
more equitable than chance.

**The partition is not unique.** Restart agreement is ARI ≈ 0.34–0.47
(`structural.structural_partition_stability`, `run.partition_stability_batch`)
while the *cost* is highly reproducible. Every downstream claim must therefore
be reported over an ensemble or a consensus, never a single run:

- `structural.restart_ensemble_partitions` — build the R-restart ensemble (256)
- `run.restart_ensemble_all_sizes` — same for 128/512/1024
- `structural.ensemble_analyze` — partition-agnostic metric infrastructure
- `structural.ensemble_stats` — ICC and beats-random-null across the ensemble
- `run.ensemble_nolesion` / `run.ensemble_lesion` — TF-free / TF metrics over it

### Training the network grid

`training.train_softplus_diag` (single net) driven by
`training.run_seeds_parallel` (size × seed grid across cores). Activity and
epoch variance via `training.compute_variance_epoch_to_file` and
`training.cluster_epoch_variance`, which produce the Driscoll-style
activity clusters used as the comparison partition throughout.

Batch drivers: `run.batch_softplus_analysis`, `run.batch_256_transpose_analysis`
(config-driven, checkpointed, read `configs/*.json`).

---

## Claim 1 — Connectivity defines a dynamical role structure

### 1a. The structure exists

`run.equitability_soft` — γ z-score across all seeds under **hard** (k-means)
and **soft** (GMM) assignment. Answers "is the structure a hard-assignment
artefact?" by holding the null fixed and swapping only the assignment method.

`run.wl_vs_strength_ari` — how much of the partition is just node-strength
quantiles. A specificity check: if signed-WL merely recovered degree bins, ARI
against a strength-quantile partition would be high.

### 1b. Lumpability — the theory bridge

`run.lumpability_check` — the highest-value theoretical check. Equitable
partitions are not merely "role equivalence": they are **lumpable**, i.e.
invariant subspaces of the linear dynamics, and eigenvectors are
piecewise-constant on cells. With indicator `H` and projector
`P = H(HᵀH)⁻¹Hᵀ`, two consequences are tested directly:

1. *Invariant subspace*: `‖(I−P)MH‖ / ‖MH‖` small (the normalised γ defect).
2. *Piecewise-constant eigenvectors*: `leakage(v) = ‖v−Pv‖ / ‖v‖` small,
   summarised as **step-R²**.

Both are z-scored against random same-size partitions. Run on the matrix the
partition was optimised for (transpose) and on raw `W` for the dynamics reading.

`run.motif_jacobian_lumpability` — extends the same test from the *linear*
`W_rec` to the **nonlinear** memory-delay Jacobian, i.e. do the cells scaffold
the dynamics the network actually runs, not just the weight matrix.

### 1c. Controls — the part that changed the story

`run.training_control` / `run.training_control_all_sizes` — is the structure
*created by training*, or a by-product of the degree distribution? Compares the
trained matrix against three surrogates: **row-shuffle** (preserves node
strength), **full-shuffle**, and **Gaussian**. This control is what
demoted equitability and promoted lumpability (see findings).

`structural.random_partition_activity_covariance_null` — random-partition null
for within-cell activity covariance.

Supporting diagnostics: `figures.degree_check`, `figures.gamma_size_hierarchy`,
`figures.degeneracy_figures` (cost-landscape near-degeneracy).

---

## Claim 2 — Motifs are organised by that structure

### 2a. Recover the motifs

`run.motif_fixed_points` — the entry point for the dynamical side. For each
task and each analysable epoch, hold input constant at task-relevant values,
sample initial states from real trial trajectories (plus noise to escape the
trajectory manifold), and minimise `q(h) = ½‖F(h)−h‖²` with Matt Golub's
FixedPointFinder. Saves fixed/slow points, their Jacobians, and eigenvalues.

Requires `external/` (FixedPointFinder, recurrent-whisperer) and TF; outputs are
large and live on the data volume via the `results/motif_fixed_points` symlink.

`motifs.motif_classify` — classify each `(rule, epoch)` into a motif from the
Jacobian spectrum: line attractor, point attractor, saddle/decision, rotation,
mixed. `motifs.fp_inventory` gives the per-network inventory.

`run.motif_taxonomy_all_nets` — the taxonomy for every network in one CSV.
`figures.motif_validation` — validation-gate plots confirming the recovered
taxonomy reproduces Driscoll's motifs before anything is built on it.

### 2b. Link motifs to role groups

`motifs.motif_participation` + `run.motif_participation_stats` — per-unit motif
participation vs role-group membership. Is participation in a given motif
concentrated within role cells beyond chance (`z_role`), and does the role
partition explain variance the activity partition does not (`unique_role`)?

`run.motif_ensemble` + `motifs.aggregate_motif_ensemble` — the whole pipeline
across the network ensemble, with **seed as the unit of replication**
(Wilcoxon across nets, not pooled motifs).

### 2c. The controls that constrain the claim

`run.motif_reuse_control` — **the decisive control.** Cross-task motif reuse
(`R` = correlation of participation vectors for the same motif in different
tasks) is compared against a **size-matched random partition**. Reuse being
positive is not evidence for the connectivity partition unless it beats this.

`run.motif_reuse_diagnostics` — three follow-up analyses on cross-task pairs.
`run.epoch_reuse_matrix`, `run.pair_fate` — pooled task-epoch reuse matrix and
within-cell pair-fate transitions.

---

## Claim 3 — The quotient predicts the motifs

### 3a. Containment, then prediction

`run.motif_subspace_rva` — role vs activity at the **subspace** level with
matched `k` and matched group-size distributions. Establishes that motif
directions lie in the role-defined subspace (containment).

`run.motif_quotient_prediction` — the headline. Containment says the motif
direction is *inside* the subspace; this asks whether the **quotient** — the
coarse-grained circuit itself, not just the grouping — predicts the direction.
Reports, per `k`, the alignment achievable from the top-*m* quotient modes
(`pred_m1/m3/m5`) against two matched baselines: a size-matched **random**
partition and the **activity** partition.

`run.motif_quotient_heldout` — is the prediction *prospective*? The quotient in
the above is fit with all conditions visible; this re-estimates under
interleaved, cross-task, and cross-type held-out splits.

`run.motif_quotient_controls` — what the quotient gives up. Compression cost
against the full Jacobian, full `W`, and the `W` SVD at matched rank.

### 3b. Does the subspace predict behaviour?

`run.subspace_predicts_dynamics` — does the wiring-only, group-constant
subspace predict memory performance beyond raw capacity?

`motifs.motif_rotation_from_response` — reproduces Driscoll's rotation motif
(the ring rotating into output-potent space) as a cross-check against the
published result.

---

## Causal tests (cut across claims 1 and 2)

`structural.structural_cell_lesions` — lesion a role cell, measure which tasks
break. `run.lesion_causal_batch` runs it across seeds.

`run.lesion_memory_motif` / `run.lesion_memory_motif_v2` — targeted test with a
pre-committed prediction: cells participating in the slow (near-marginal) memory
mode should selectively break memory tasks. v2 is the improved design.

`structural.subpartition_lesion_selectivity` — lesion *functional
sub-partitions* inside a cell, a sharper question than whole-cell lesions.

---

## Superseded lines (kept, but not part of the three claims)

These come from the earlier five-result framing and are retained because they
document controls that changed the story:

- `run.shared_component_reorg`, `structural.within_task_split_half_null`,
  `run.within_task_split_half_null`, `structural.structural_cell_cross_task_fractionation`
  — compositional reorganisation of cells across tasks.
- `run.fair_context_predictor`, `run.connectivity_functional_regression`,
  `structural.softplus_memory_full_context_alignment`,
  `structural.softplus_memory_subpartition_sweep`, `run.alignment_stability`
  — whether whole-network connectivity *context* predicts within-cell
  reorganisation.

See findings.md for why these are no longer headline claims.
