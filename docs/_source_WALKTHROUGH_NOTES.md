# Walkthrough notes — connectivity-defined role structure in multitask RNNs
*Prep for talking a neuroscientist through the paper. Numbers are current as of this draft.*

## The one-liner
A trained multitask RNN's **wiring** is organized into groups of units that share a computational role — readable from the weight matrix alone, before you look at any activity. These groups are the **natural coordinates of the network's dynamics** (the substrate its reusable "dynamical motifs" live in), they **recombine across tasks** to do different computations, and **lesioning one removes the computations it carries.** Flexible computation = task-dependent reconfiguration of a fixed, connectivity-defined role structure.

## The gap we fill (say this early)
Driscoll 2024 and Yang 2019 define the computational units of these networks from **activity** — task variance, fixed points, clusters — which requires running the task. But the computation is produced by the **fixed connectivity**. We ask: is the organization already in the wiring? Answer: yes, and it's the dynamical substrate. The novel wedge over Driscoll: **connectivity-first, before-task, and it doesn't need the roles to be aligned to single-unit axes.**

## Jargon → neuroscience translation (use these phrasings)
- **Equitable partition / "role cell"** → a group of units that connect to the rest of the circuit *in the same way* — like a connectivity-defined cell type (defined by projection/input pattern, not by response tuning). **Key subtlety:** it is NOT a densely-interconnected cluster; members needn't wire to each other, they wire *similarly to everyone else*.
- **Signed Weisfeiler–Leman refinement** → iteratively relabel each unit by the labels of the units it connects to, until stable. "Define cell types by who you talk to." Signed = excitation and inhibition handled separately.
- **Quotient** → the coarse-grained circuit: collapse each role group to one node.
- **Invariant subspace / lumpability** → you can describe the population dynamics at the level of the groups without losing anything; the groups are the natural coordinate system of the activity. The eigenvectors (natural modes) are flat within a group.
- **γ (Gamma)** → how far a grouping is from a perfect equitable partition; lower = better. Compared to shuffled groups → a z-score.

## The five results (spine + the numbers that matter)

**R1 — the structure exists and is reproducible.**
- Role groups from W_rec are far more equitable than chance **across every scale** (k = 3–60), all sizes 128–512 (z well below −3, P<0.001). Not tied to one k.
- Honest point to volunteer: the *exact* grouping isn't unique (many near-equal solutions; restart agreement ARI ≈ 0.40) but the *cost* is highly reproducible (≈3%). So we talk about "the structure," report over a consensus — not "these exact cells."
- Related to but distinct from activity clusters (NMI ≈ 0.58).

**R2 — the groups are the dynamical substrate (this is the strongest, most novel result).**
- The network's dynamical modes (eigenvectors of W) are **piecewise-constant on the groups** — step-R² 0.89 vs 0.20 for a random grouping; holds in all 33 networks; gets *stronger* with size.
- Extends to the **nonlinear memory-delay dynamics** (Jacobian): the slow, near-marginal "line-attractor" mode that holds a memory is organized by the groups.
- **Training control:** lumpability is created by training (0.85 trained > 0.68 strength-shuffle > 0.34 random). Volunteer this: the *raw equitability magnitude* is largely explained by the unit-strength (degree) distribution — it's the **lumpability** (eigenvector structure) that is the real training signature. Lead with lumpability, not γ.
- This is the bridge to Driscoll: the groups are the coordinates his dynamical motifs live in.

**R3 — the groups reorganize compositionally.**
- Within a group, the functional sub-grouping of units is **reliable within a task** (split-half ARI 0.71) but **reorganizes across tasks** (cross-task 0.29); gap positive in 10/10 networks, P<0.001. Not clustering noise (we built the within-task null to prove it).
- Small discrete repertoire: 2–3 sub-configurations per group.
- **Compositional, not arbitrary:** the same sub-grouping recurs for the same computation across different tasks (e.g., both memory delays) and differs for different computations (shared vs non-shared component 0.36 vs 0.23, 10/10, P=0.002).

**R4 — whole-network context, not local wiring, sets the reorganization.**
- A unit's within-group functional membership is predicted by its **whole-network connectivity beyond its local wiring** (partial r 0.13, 9/10, P=0.004; survives controlling for input weights + dimensionality, 0.10, P=0.02). Robust across k.
- General across epochs but **strongest/uniquely robust during memory** (recurrent dynamics dominate when there's no external input).
- Honest caveat to volunteer: some correlation is built in (input drives activity); the non-trivial part is beating *local wiring* and a *rank-matched random projection*, which it does. This is the most modest of the results — present it as supporting, not headline.

**R5 — the groups are causally the computational units.**
- Lesion a group → it breaks the tasks it participates in more than random groups (deficit-activity alignment 0.48 vs 0.40, 10/10, P=0.002) and produces bigger task-specific deficits (0.70 vs 0.60, 10/10, P=0.002).
- The specificity nuance is a *feature*: structural lesions hit a **coherent set of related tasks** (not one), because groups are compositional — corroborates R3.
- Memory motif is causally localized: lesion the cells that hold a memory → selectively abolish memory (r 0.57, 10/10). **Volunteer the limitation:** those cells were identified by *activity*; the connectivity-defined slow mode alone did *not* predict the causal memory cells (the memory subspace is distributed). So the tightest wiring→motif→behavior link is only partial.

## Caveats to volunteer (don't get caught out)
1. Raw equitability ≈ degree artefact; **lumpability** is the training signature. (R2 control.)
2. Partition non-unique → consensus, not "these exact cells." (R1.)
3. R4 effect is modest and memory-dominated; has an inherent partial-circularity we control for.
4. Memory motif localized by activity, not connectivity slow-mode.
5. Invariant-subspace argument is exact for linear dynamics, approximate for the trained nonlinear net.
6. 1024-unit nets: equitability reverses (positive) — but lumpability is *strongest* there, so structure is present; the reversal is a size-weighted-statistic artefact.

## Questions a neuroscientist will likely ask (and answers)
- **"Isn't this just cell types?"** Related in spirit — it's a connectivity-defined typing — but the payoff is dynamical: the types are invariant subspaces, so they predict what the circuit *computes*, not just how it's wired. And they're derived without a genetic/morphological label, purely from the weight matrix.
- **"Does it match the activity-based functional clusters?"** Partially (NMI ≈ 0.58) — related but not identical. The connectivity view is a scaffold that's flexibly deployed; that mismatch is the point (R3).
- **"Would this hold in a real brain / connectome?"** It's a prediction: the equitable-partition structure of a measured connectome should constrain the motifs a circuit can express. Caveat honestly with Beiran & Litwin-Kumar 2025 (connectome alone often insufficient) — we position role structure as a *necessary intermediate feature*, not a sufficient predictor.
- **"The memory result — connectivity didn't predict the causal memory cells, activity did. Doesn't that undercut the thesis?"** Fair. The thesis (groups = substrate, groups causally functional) stands; what we *can't* yet claim is single-cell slow-mode necessity — the memory subspace is distributed across cells. It's a limitation, not a contradiction.
- **"Why these 15 tasks?"** Driscoll's battery — reuses a small set of epochs (memory/stim/response) across tasks, which is exactly what lets us separate shared from task-specific computation.
- **"What do I do with this as an experimentalist?"** Two things: (1) a connectivity-first way to find computational units without task data; (2) a concrete lesion prediction — ablate a connectivity role → lose a specific coherent set of computations.

## What's solid vs provisional
- **Solid:** R1 (existence), R2 (lumpability + control), R3 (reorganization + compositional), R5 core (causal). All 10 seeds, per-seed stats.
- **Provisional / to strengthen:** tanh off-axis recovery (the cleanest novelty vs Driscoll — not yet run); 1024 nets still training; transfer-learning motif reuse (potential 6th result).
