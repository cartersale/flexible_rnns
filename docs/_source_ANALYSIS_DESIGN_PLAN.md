# Analysis & Robustness Design Plan

Structural-cell / symmetry-breaking project. Branch `codex/flexible-multitask-symmetry-publish`.

This plan organizes the analyses around the three-part story, integrating the
reviewer's comments and the empirical results we have run so far. Each part
states the **defensible claim**, the **threats** to it (reviewer + our data),
the **checks** that settle it, and a **decision rule** — what result would
confirm, weaken, or kill the claim. The point is to decide what to run *and why*
before running it.

## The three-part story

1. **A connectivity-defined role structure underlies trained recurrent networks.**
2. **Role cells reorganize compositionally across tasks.**
3. **Connectivity *context*, not local wiring, predicts task-dependent reorganization.**

## Executive read (where each part stands today)

| Part | Status | One-line reason |
|---|---|---|
| 1 | **Solid, needs scoping + framing** | Equitability z ≈ −6 to −9 for 128/256/512, robust across seeds *and* hard/soft assignment (27/30 seeds z<−3). But 1024 reverses, the *specific* partition is non-unique, and k/node-set are activity-derived. |
| 2 | **Unvalidated, highest risk** | The "reorganization" signal has no within-task noise floor yet. On tiny cells (5–8 units) it may be indistinguishable from noise. |
| 3 | **At risk on two fronts** | The full-context predictor is an unfair (rank-mismatched, near-circular) test, *and* our stability sweep shows the headline z swings −2→+5.6 across equally-valid partitions with a null consensus. |

**The single most important structural fact we have learned:** the signed-WL
partition is **not unique** (restart ARI ≈ 0.40) even though its *cost* is highly
reproducible (CV ≈ 0.03). The cost landscape is near-degenerate — many different
partitions are almost equally equitable. Soft/GMM assignment does not fix this.
Every downstream claim must therefore be reported **over an ensemble of
partitions (or a consensus)**, never from a single run.

**The linchpin check:** Parts 2 and 3 both operate on *functional sub-partitions
within a cell*. If those sub-partitions are noise, then reorganization (Part 2)
is noise **and** wiring-predicts-reorganization (Part 3) is predicting noise.
The **within-task split-half null** is therefore the shared foundation for both
parts — it is the highest-value single analysis in this plan.

---

## Gate 0 — Fix the partition object (foundational; everything depends on it)

**Why:** We established the partition is one of many near-degenerate solutions,
and k/node-set are activity-derived (reviewer #3). Before any downstream claim,
we need a *defined, defensible* operating object and to know the claims survive
reasonable choices of it.

**Checks**
- **G0.1 Consensus partition.** Adopt the co-association consensus over N≥20
  restarts as the canonical operating partition (script already computes it).
  Report it, plus the restart distribution, everywhere downstream.
- **G0.2 Connectivity-only k.** Reproduce the main results with k chosen from the
  γ-saturation curve (connectivity), not `min(k_activity, 0.2·n_active)`. Show a
  k-range robustness band, not a single k. (Directly answers reviewer #3 and the
  k=33/41 discrepancy in reviewer #5.)
- **G0.3 Cell-size / degree sanity.** Report the cell-size distribution and the
  singleton–strength relationship (done: singletons carry 1.4–1.9× node strength,
  33/33 nets, but only 2–5% of units — mild, not the "one giant cell" pathology).

**Decision rule:** If the equitability and (later) alignment claims only hold at
one activity-derived k, we must requalify Part 1's "independent of activity."
If they hold across a connectivity-selected k-band, the claim strengthens.

**Status:** partial. Restart/consensus machinery built (`structural_partition_stability.py`).
Connectivity-only-k rerun not yet done.

---

## Part 1 — Connectivity-defined role structure exists

**Defensible claim:** Trained multitask RNN weight matrices contain robust,
reproducible **super-equitable** mesoscale structure — the cost of the best
signed equitable partition is far below chance — consistent across seeds and
sizes *within a range (128–512)*, and robust to the assignment method. The
*existence of the structure* is the claim, not the identity of a single
partition.

**Threats**
- Non-uniqueness of the partition (our data) → reframe from "the cells" to "the
  structure/cost."
- Activity-derived k and node set (reviewer #3) → Gate 0.
- The **1024 reversal** (reviewer #5): z goes *positive* (worse than chance) for
  all 3 seeds, and — we now know — **this persists under soft assignment**
  (hard +3.3, soft +3.6). So it is *not* a hard-k-means artifact.
- "Only works with non-negative activations" is misstated (reviewer #6).

**Checks**
- **P1.1 Equitability across seeds (DONE).** z ≈ −6 to −9 for 128/256/512, 27/30
  seeds z<−3, assignment-robust. Report as the backbone of Part 1.
- **P1.2 Explain the 1024 reversal.** Since soft doesn't fix it, test the
  "operating-k-too-large" hypothesis: sweep k downward at 1024 and check whether
  equitability recovers at smaller k (equitability is known to go positive at
  small *and* mismatched k). If a connectivity-selected k restores z<0, the
  reversal is a k-selection issue, not a failure of structure. If not, scope the
  claim honestly to ≤512.
- **P1.3 Recoverability without unit-axis alignment (reviewer #6, novelty).**
  Show the connectivity partition is recovered in a **tanh** network (where
  Yang/Driscoll unit-axis clustering fails) or under a random rotation of the
  activity basis. This is the clean novelty over Driscoll and reframes the
  non-negativity point correctly.
- **P1.4 Lumpability bridge (reviewer #6, highest-value theory).** Test the
  empirical prediction of equitable-partition theory: leading eigenvectors of
  W_rec are approximately **piecewise-constant on cells**, and the quotient
  (lumped) dynamics approximate the full dynamics. This grounds the
  connectivity→dynamics link in a theorem (invariant subspaces) instead of a
  soft alignment number.

**Decision rule:** Part 1 holds if equitability is super-chance across a
connectivity-selected k-band for 128–512 (already ~there) and P1.4 shows cells
are approximately lumpable. 1024 is scoped or explained via P1.2.

**Status:** P1.1 done. P1.2/P1.3/P1.4 not started.

---

## Part 2 — Compositional reorganization across tasks

**Defensible claim:** Within a role cell, the functional sub-grouping of units
**changes across tasks** by more than it changes across independent samples of
the *same* task. (I.e. reorganization exceeds the within-task noise floor.)

**Threats (this is the weakest part)**
- **No within-task baseline (reviewer #2, make-or-break).** Low cross-task ARI is
  currently read as reorganization, but ARI on 5–8 units split into ≤6 clusters
  is mostly noise; there is no within-task control anywhere in the code.
- **k=1 silhouette degeneracy (reviewer #2).** Two constant labellings score
  ARI=1 ("stable"); a split-vs-no-split scores ~0 ("dynamic"). So stable/dynamic
  is driven by whether a noisy silhouette crossed zero, not by reorganization.
- **Arbitrary 0.3/0.7 cutoffs (reviewer #2).**
- Partition non-uniqueness (our data): which units are even *in* a cell, and
  which cells are "memory cells," varies across restarts (we saw memory-cell
  count swing 5–8 on one network).

**Checks**
- **P2.1 Within-task split-half null (THE make-or-break).** Split trials of one
  task in half, cluster each half, measure ARI. This is the null the cross-task
  ARI must beat. Report cross-task ARI as a **ratio or z against the split-half
  distribution**, not a raw number vs arbitrary cutoffs.
- **P2.2 Fix the k=1 degeneracy** so "stable" is not inflated by both partitions
  collapsing to one cluster.
- **P2.3 Cutoff sensitivity sweep** — show any stable/dynamic split survives a
  range of thresholds, or drop the categorical framing.
- **P2.4 Ensemble + per-seed.** Run P2.1–P2.3 over the consensus partition and
  across restart partitions and seeds; treat seed as the unit of replication.

**Decision rule:** If cross-task ARI ≤ within-task split-half ARI (P2.1), the
"reorganization" is noise → **drop Part 2 or restate it as an upper bound.** If
cross-task ARI reliably exceeds the within-task floor across seeds, Part 2 is
earned. *Run P2.1 before investing in anything else in Part 2.*

**Status:** not started. Given the fragility we have already seen, treat this as
the pivotal experiment for whether the middle of the story survives.

---

## Part 3 — Connectivity context predicts reorganization

**Defensible claim:** The task-dependent functional sub-grouping within a cell is
better predicted by each unit's **whole-network (full-context) connectivity** than
by its **local (within-cell) wiring**, beyond a **rank-/dimension-matched** random
baseline, and this holds within-task and across seeds.

**Threats (at risk on two fronts)**
- **Unfair test (reviewer #1, headline).** Full-context uses ~512 features vs
  internal ≤12 on the same 5–8 points; internal weights are a strict subset, so
  full-context cannot lose on information. And it is near-circular: activity is
  generated by the full recurrent drive, so "units with similar full-network
  input have similar activity" partly restates the dynamics.
- **Partition-dependence (our data).** Across 20 restart partitions the headline
  `ari_z` swings **−1.9 → +5.6** (mean 1.6 hard / 2.1 soft), only 40–55% of
  partitions individually significant, and the **consensus partition is null**
  (−0.31 hard, +0.28 soft). Even the most partition-stable network is fragile.
  Soft assignment does not fix it.
- **Pseudoreplication (reviewer #4).** The P<0.001 sign test pools every
  cell×epoch as independent; cells share the full-context profile and epochs
  within a task are correlated.

**Checks**
- **P3.1 Fair, matched predictor (reviewer #1).** Match internal vs full-context
  on information: either project full-context down to the internal rank, or
  benchmark full-context against a **rank-matched random projection of W**. The
  real result is full-context beating a rank-matched random projection.
  (Scaffolding exists: `random_projection` / `internal_noise_padded` controls and
  `internal_rank`/`internal_width` modes in `softplus_memory_full_context_alignment.py`.)
- **P3.2 Add W_in and bias as candidate predictors (reviewer #1).** The
  publishable claim is that the *recurrent input profile* dominates over input
  weights and bias in setting the within-cell grouping. Test it explicitly.
- **P3.3 Aggregate over partitions (our data).** Report the effect as the mean
  `ari_z` over restart partitions (or on the consensus), with the full
  distribution — never a single-run z.
- **P3.4 Per-seed statistics (reviewer #4).** One number per cell (or per seed)
  before testing; mixed model with cell and seed as grouping factors; seed as
  the unit of replication.
- **P3.5 Within-task stability.** Show the functional partition being predicted is
  itself stable within-task (shares the P2.1 machinery).

**Decision rule:** If full-context ≤ rank-matched random projection (P3.1), Part 3
as stated collapses — the "context predicts function" result was a
dimensionality artifact. If the effect survives P3.1 but the partition-aggregated,
per-seed estimate (P3.3–P3.4) straddles zero, Part 3 is not a per-network claim —
weaken to a weak aggregate effect. If it survives all of the above, it is the
ambitious result worth fighting for.

**Status:** conclusion-stability characterized (fragile). Fair test (P3.1/P3.2)
and proper aggregation (P3.3/P3.4) not yet run.

---

## Cross-cutting requirements (apply to all parts)

- **C1 Statistics / replication (reviewer #4).** Seed is the unit of replication.
  Compute per-seed effects, then test across seeds; use mixed models with
  cell+seed random effects; never pool cell×epoch as independent. We now have
  10 seeds/size for 128/256/512 (the new runs) to make this real.
- **C2 Report over the ensemble, not a single partition.** Consequence of the
  non-uniqueness finding. Consensus + restart distribution everywhere.
- **C3 Causal validation (reviewer #7).** Lesion each role cell (consensus
  partition) and predict which tasks/epochs break, Driscoll-style. This turns the
  correlational story into mechanism. `structural_cell_lesions.py` exists; run it
  on the consensus partition, per seed.
- **C4 Framing fixes (reviewer #6).** (a) Correct the non-negativity claim: the
  unit-axis identifiability needs non-negativity; the *structure* is present in
  tanh nets, just rotated — a connectivity method recovers it regardless (ties to
  P1.3). (b) Ground the connectivity→dynamics bridge in lumpability/invariant
  subspaces (P1.4).

---

## Recommended sequencing (with rationale)

1. **Gate 0** (consensus partition + connectivity-only k). Everything downstream
   consumes the operating partition; fix it first.
2. **P2.1 within-task split-half null.** The linchpin — it validates (or kills)
   the functional sub-partition that *both* Parts 2 and 3 depend on. Highest
   information-per-hour. Do this before P3's fair test, because if the sub-partition
   is noise, P3 is moot.
3. **P3.1/P3.2 fair matched predictor.** The headline claim's make-or-break; only
   worth running once P2.1 says the functional partition is real.
4. **P3.3/P3.4 aggregation + per-seed stats**, then **P2.1–P2.4** full across
   seeds. Turns surviving effects into properly-replicated numbers.
5. **P1.2 (1024), P1.3 (tanh/rotation), P1.4 (lumpability)** — finalize Part 1's
   scope and theory in parallel; lower risk.
6. **C3 lesions** — causal capstone on the consensus partition.

**Make-or-break gates:** P2.1 (is reorganization above the within-task floor?)
and P3.1 (does full-context beat a rank-matched random projection?). If either
fails, the corresponding part must be restructured before drafting.

---

## Reviewer-concern → check crosswalk

| Reviewer concern | Addressed by |
|---|---|
| #1 unfair full-context test | P3.1, P3.2 |
| #2 no within-cell stability baseline | P2.1, P2.2, P2.3 |
| #3 "independent of activity" overclaimed | G0.2, P1 reframe |
| #4 pseudoreplication | C1, P3.4, P2.4 |
| #5 method fragility (restarts, soft, 1024, cell sizes, k wobble) | G0.1, G0.3, P1.1 (done), P1.2, C2 |
| #6 framing (non-negativity, lumpability) | P1.3, P1.4, C4 |
| #7 no causal test | C3 |
