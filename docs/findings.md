# Findings

> ## ⚠️ Numbers involving seeds 3–9 are STALE (as of 2026-07-26)
>
> The 256-unit seeds 3–9 were **retrained on 2026-07-25** (checkpoints rewritten
> 14:19–15:02). Every result file in `results/` predates that. The numbers below
> were computed on the superseded checkpoints.
>
> This is not a small drift. On the current checkpoints the partition object
> itself is different:
>
> | | n_active | k |
> |---|---|---|
> | seeds 3–9, **stale** | 69–75 | 14 |
> | seeds 3–9, **current** | 212–220 | 42–44 |
> | seeds 0–2 (released, unchanged) | 188–256 | 38–51 |
>
> The old values reflect nets that had barely trained — only ~70 of 256 units
> carried meaningful epoch variance, against ~215 now. `perf_min` likewise moved
> from 0.781–0.904 to 0.926–0.947.
>
> **Exposure by claim:**
> - **Claim 1** — the 33-net training control and the 33-net step-R² pool seeds
>   3–9. Affected; direction unknown until recomputed.
> - **Claim 2** — all 10-net participation and reuse statistics pool seeds 3–9,
>   and previously mixed k=14 nets with k=38–51 nets. Most exposed.
> - **Claim 3** — uses seeds 0–2 only. **Not affected.**
> - The fair-context-predictor section pools all 10. Affected.
>
> A batch-confound claim previously recorded here (that released and new nets
> were incommensurable) was **an artefact of the stale checkpoints** and has been
> removed — the two batches are comparable on the current nets.
>
> Recompute chain: `cluster_epoch_variance` → `restart_ensemble_partitions` →
> {lumpability, training control, reuse, lesions} → `motif_fixed_points` →
> {inventory, participation, quotient}.

Status of each claim, with the numbers as they stand in `results/`. Recomputed
from the result files during the repo migration (2026-07-26), not copied from
prior notes.

**Read the negative results first.** Three controls changed what this project
can claim, and each one demoted something that looked like a headline.

| | Claim | Status |
|---|---|---|
| 1 | Connectivity defines a dynamical role structure | **Holds**, but via lumpability, not equitability |
| 2 | Motifs are organised by that structure | **Holds** for participation; **fails** for cross-task reuse |
| 3 | The quotient predicts the motifs | **Holds**, strongest result — but only 3 networks |

---

## Claim 1 — Connectivity defines a dynamical role structure

### What holds

**Eigenvectors of `W_rec` are piecewise-constant on role cells.** Step-R² = 0.825
median (range 0.71–0.91) vs 0.279 for random same-size partitions — **33/33
networks**, every size 128–1024. This is the theoretically grounded version of
the claim (equitable partitions are invariant subspaces) and it is the most
robust result in claim 1.

**Lumpability is what training creates.** Against surrogates that preserve
node strength (33 networks, `training_control_all_sizes.csv`):

| Surrogate | median step-R² |
|---|---|
| Trained | **0.825** |
| Row-shuffle (preserves node strength) | 0.706 |
| Full-shuffle | 0.445 |
| Gaussian | 0.318 |

Trained beats row-shuffle in **31/33** networks.

**It extends to the nonlinear dynamics.** The memory-delay Jacobian shows the
same structure (`run.motif_jacobian_lumpability`), so the cells organise the
dynamics the network actually runs, not only the weight matrix.

### What failed — equitability is largely a degree artefact

Raw equitability z looks strong (median −5.78 across 33 nets). But against a
**row-shuffle that preserves node strength**, the trained network is *more*
equitable in only **12/33** networks — the surrogate's median z is −8.70,
i.e. *better* than trained.

**Consequence: lead with lumpability, not γ.** Equitability magnitude is mostly
explained by the degree distribution. An earlier "coarse hierarchy / competing
quotients" reading built on unnormalised γ did not survive a class-size-invariant
statistic and has been dropped.

### What failed — the partition is not unique

Restart agreement is ARI **0.34–0.47** depending on size (0.47 at 256, 0.34 at
128), while the cost is highly reproducible. The cost landscape is
near-degenerate: many different partitions are nearly equally equitable, and
soft/GMM assignment does not fix it.

**Consequence:** claims are about *the structure*, never "these exact cells,"
and every downstream number is reported over a restart ensemble or consensus.

### Scope limit — the 1024 reversal

At n=1024 the equitability z goes **positive** (+4.19, worse than chance) while
step-R² is at its **highest** (0.851). The structure is clearly present; the
reversal is a property of the size-weighted statistic. Consistent with
equitability being the weaker, more artefact-prone measure.

---

## Claim 2 — Motifs are organised by that structure

### The motif inventory reproduces Driscoll

490 `(network, rule, epoch)` conditions across **10 networks**:

| Motif | n |
|---|---|
| Line attractor | 259 |
| Point attractor | 127 |
| Mixed | 59 |
| Saddle / decision | 43 |
| Rotation | 1 |

Memory epochs (309) are dominated by line attractors, stimulus epochs (180) by
point attractors and saddles — the expected taxonomy, which gates everything
downstream.

### What holds — motif participation is organised by role cells

Across 10 networks (seed as the unit of replication, Wilcoxon):

| Metric | median | nets positive | p |
|---|---|---|---|
| `z_role` (participation concentration) | 3.23 | — | — |
| `subspace_z_role` | 2.52 | **10/10** | 0.002 |
| `unique_role` (variance the activity partition misses) | 0.051 | 9/10 | 0.006 |
| `unique_activity` (the converse) | 0.070 | 8/10 | 0.020 |

Role structure explains motif participation beyond the activity partition — and
the converse also holds, so the two views are **complementary, not redundant**.
Effect is strongest for line attractors (`z_role` 3.48) and weakest for
saddle/decision (2.60, 7/9 nets, p=0.13 — not significant).

### What failed — cross-task motif reuse is not partition-specific

Cross-task reuse is real: `R` = 0.196 median (range 0.145–0.283, 10 nets), and
`line_reuse_minus_baseline` is positive in 10/10 nets (p=0.002).

But against a **size-matched random partition**, the connectivity partition does
**not** win at any k:

| k | median z | Δ>0 in |
|---|---|---|
| 8 | −0.75 | 2/10 |
| 16 | −0.25 | 4/10 |
| 30 | −0.41 | 4/10 |
| 50 | −0.13 | 4/10 |

**Consequence:** cross-task motif reuse is a *unit-level* fact about the
network, not evidence that the connectivity partition organises reuse. It
cannot be cited in support of claim 2. Participation (above) survives its
control; reuse does not.

---

## Claim 3 — The quotient predicts the motifs

**The strongest positive result in the repo**, and the one that most directly
supports the thesis: it is the *structure* of the coarse-grained circuit doing
the work, not the gains.

The top-3 quotient modes recover the motif direction almost completely, and beat
both matched baselines at **every** k tested (`pred_m3`, 3 networks):

| k | quotient | random partition | activity partition | median z | beats activity |
|---|---|---|---|---|---|
| 8 | **0.933** | 0.753 | 0.829 | +2.64 | 3/3 |
| 12 | **0.914** | 0.681 | 0.725 | +3.80 | 3/3 |
| 20 | **0.889** | 0.640 | 0.646 | +3.60 | 3/3 |
| 30 | **0.858** | 0.643 | 0.623 | +3.68 | 3/3 |
| 50 | **0.819** | 0.676 | 0.680 | +3.17 | 3/3 |

The margin over both baselines is stable across an 8→50 range of k, so this is
not a single-k effect. Structural alignment (`struct_m3`) shows the same
pattern more weakly (0.59–0.86, z +1.5 to +2.9).

### The limit you must state

**This is n=3 networks (seeds 0–2), not 10.** Claim 2 rests on 10 networks;
claim 3 does not. Extending `run.motif_quotient_prediction` to the full
10-network ensemble is the single highest-value outstanding run in the repo —
it is the difference between "strongest result" and "strongest result, properly
replicated."

`run.motif_quotient_heldout` (prospective splits) and `run.motif_quotient_controls`
(compression cost vs full J / full W / W-SVD) have been computed but are not yet
summarised here.

---

## Superseded: context predicts reorganisation

From the earlier five-result framing. The claim was that a unit's within-cell
functional membership is predicted by its whole-network connectivity beyond its
local wiring.

Under the **rank-matched fair test** (`run.fair_context_predictor`, 10 nets),
the point estimate survives but the significance does not:

- `full − rank-matched random projection`: median **+0.082**, positive in 8/10
- `full − internal wiring`: median +0.025, positive in 7/10
- **fraction of individual tests where full beats random at p<0.05: median 0.00**,
  non-zero in only 2/10 networks

So the direction is consistent across networks while essentially no individual
test reaches significance — an underpowered effect, not a per-network claim.
Note this is more nuanced than a flat failure: an earlier reading recorded it as
simply not surviving, which the 10-net aggregate does not support. Treat it as
**weak and unreplicated**, and do not put weight on it.

---

## Outstanding

1. **Extend claim 3 to all 10 networks** — highest value by a wide margin.
2. **No paper figure set exists for claims 2 and 3.** The `fig1`–`fig5` panels
   map to the old five-result story; the motif and quotient work exists only as
   `slides/`. See figure_map.md.
3. Summarise `quotient_heldout` and `quotient_controls`.
4. `slides.network_grid` cannot run until `results/net_perf.json` is regenerated.
5. tanh / off-axis recovery — the cleanest novelty over Driscoll (a connectivity
   method should recover the structure where unit-axis clustering fails) — has
   still not been run.
