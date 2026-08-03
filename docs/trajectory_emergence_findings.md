# When the partition becomes an object, and when it locks

Two prior questions to `docs/trajectory_partition_findings.md`, in the order they
have to be asked. E1: is there a partition at all, at each point in training?
E2: does the partition lock while the weights are still moving?

**Date:** 2026-08-02
**Code:** `flexsym.run.trajectory_emergence` at `77b129d` **+ uncommitted changes**
(the module itself is untracked, as is the equitability/lumpability block in
`trajectory_subspace.py` that it imports its unit set and k from).
**Data:** `results/trajectory/trajectory_emergence.{csv,json}` — 230 rows
(10 nets × 23 snapshots), run 2026-07-31.

```bash
python -m flexsym.run.trajectory_emergence                       # all 10
python -m flexsym.run.trajectory_emergence --init diag --seeds 0 # one net
```

R = 20 WL restarts at 23 checkpoints is the cost driver. The recorded run launched
all 10 nets as concurrent processes at 18:01:31 and finished between 18:05 and
18:10 — 3.5–9 min per net, ~9 min wall clock. Logs in `logs/emergence/`.

## Networks

The same 10 softplus LeakyRNNs as the subspace tier — `n_rnn=256`, ruleset `all`,
23 log-spaced snapshots from step 0 to 1,562,500, two arms (**diag**, **randortho**)
× seeds 0–4. Active unit set and k are held at the **final** checkpoint's values
throughout, exactly as in `trajectory_subspace`, so the two modules describe the
same object.

## What is measured

| measure | column | meaning |
|---|---|---|
| **E1** restart agreement | `restart_msca` | mean pairwise (1/k)‖UᵢᵀUⱼ‖²_F over 20 independent WL refinements of the **same** W(t) — the reproducibility of the method on that matrix |
| **E1** random floor | `random_msca` | the same statistic over 20 size-matched random partitions, recomputed **per net** (it depends on k, and k varies 30–46) |
| **E1** margin | `margin_sigma` | (restart − floor) / floor sd; `clears_floor` at ≥ 3σ |
| **E1** gate | `t_E1` | first checkpoint that clears and never falls back |
| supporting | `offdiag_frac` | ‖offdiag(W)‖_F / ‖W‖_F — when recurrent structure exists at all |
| **E2** lock | `A` | agreement of span(H_t) with span(H_final) |
| **E2** ceiling | `ceiling` | = `restart_msca` at that checkpoint. A cannot exceed the reproducibility of the method that produced the reference |
| **E2** clock | `w_remaining` | ‖W(t)−W_final‖_F / ‖W(0)−W_final‖_F — fraction of total weight change still to come |
| **E2** control | `A_shuffled` | the same curve on a row-shuffled trajectory (one permutation per net, applied at every checkpoint) |

The two reference lines matter more here than usual. Read against 1, every A curve
looks unconverged; read against each net's own ceiling, most of them are done well
before training is.

---

## Result 1 — every net clears the E1 gate, and none of them falls back

`t_E1` is defined so that a late dip disqualifies every earlier crossing. Nothing
dips: the `clears_floor` sequence is a clean run of failures followed by a clean
run of successes in all 10 nets. So E2 is interpretable everywhere, and the
`never_clears` verdict is not exercised by this grid.

| net | k | n_active | t_E1 | perf_avg at t_E1 | margin at t_E1 | max margin |
|---|---|---|---|---|---|---|
| diag 0 | 43 | 215 | 33,179 | 0.743 | 53.5σ | 53.5σ |
| diag 1 | 42 | 209 | 23,377 | 0.605 | 15.6σ | 49.9σ |
| diag 2 | 42 | 210 | 33,179 | 0.709 | 31.0σ | 39.7σ |
| diag 3 | 42 | 211 | 23,377 | 0.639 | 3.1σ | 37.0σ |
| diag 4 | 30 | 213 | 33,179 | 0.618 | 57.6σ | 57.6σ |
| randortho 0 | 45 | 223 | 1,419 | 0.440 | 3.9σ | 40.2σ |
| randortho 1 | 46 | 229 | 2,859 | 0.565 | 3.9σ | 36.8σ |
| randortho 2 | 44 | 222 | 1,419 | 0.263 | 3.7σ | 37.2σ |
| randortho 3 | 44 | 222 | 2,015 | 0.581 | 3.1σ | 40.9σ |
| randortho 4 | 31 | 233 | 1,000 | 0.426 | 3.5σ | 58.1σ |

The two arms are separated by more than an order of magnitude — randortho at
1,000–2,859 steps, diag at 23,377–33,179 — with no overlap.

## Result 2 — that arm difference is off-diagonal mass, not a fact about partitions

The obvious reading of Result 1 is that random-orthogonal initialisation finds its
partition ~20× sooner. The `offdiag_frac` column says something duller and more
convincing.

| step | diag `offdiag_frac` | diag ceiling | randortho `offdiag_frac` | randortho ceiling |
|---|---|---|---|---|
| 0 | 0.000 | 0.206 | 0.998 | 0.205 |
| 1,000 | 0.088 | 0.206 | 0.998 | 0.217 |
| 5,760 | 0.088 | 0.206 | 0.998 | 0.351 |
| 11,604 | 0.147 | 0.206 | 0.999 | 0.505 |
| 16,470 | 0.319 | 0.206 | 0.999 | 0.534 |
| 23,377 | 0.668 | 0.228 | 0.999 | 0.542 |
| 33,179 | 0.880 | 0.552 | 0.999 | 0.551 |
| 47,093 | 0.951 | 0.636 | 0.999 | 0.538 |

(arm means)

A random orthogonal matrix is dense from step 0 — `offdiag_frac` = 0.998 at every
checkpoint, never a variable. The diag arm has to *build* its off-diagonal mass,
and it does so over exactly the window in which E1 clears: 0.09 → 0.15 → 0.32 →
0.67 → 0.88 across steps 8,176–33,179, with `t_E1` at 23,377 or 33,179 in all five
seeds. **The diag arm's late E1 is not a slower search for a partition; there was
no recurrent structure to partition until then.** Once both arms have a dense
matrix their ceilings are within 0.02 of each other.

The corollary is that `t_E1` should not be reported as an independent property of
the two initialisations. It is close to a restatement of when `W_rec` stops being
diagonal.

## Result 3 — the partition is at ~84% of its own ceiling with half the weight change still to come

This is E2's answer. `A_at_w_half` is A linearly interpolated at the checkpoint
where half the total weight change remains.

| net | A(0) | floor | A @ w=0.5 | ceiling @ w=0.5 | ratio | A_shuffled @ w=0.5 |
|---|---|---|---|---|---|---|
| diag 0 | 0.216 | 0.216 | 0.458 | 0.507 | 0.90 | 0.333 |
| diag 1 | 0.225 | 0.216 | 0.439 | 0.541 | 0.81 | 0.353 |
| diag 2 | 0.212 | 0.216 | 0.412 | 0.545 | 0.76 | 0.330 |
| diag 3 | 0.218 | 0.214 | 0.451 | 0.502 | 0.90 | 0.393 |
| diag 4 | 0.168 | 0.166 | 0.398 | 0.488 | 0.82 | 0.342 |
| randortho 0 | 0.214 | 0.217 | 0.412 | 0.481 | 0.86 | 0.366 |
| randortho 1 | 0.237 | 0.215 | 0.438 | 0.513 | 0.85 | 0.368 |
| randortho 2 | 0.200 | 0.213 | 0.408 | 0.489 | 0.83 | 0.383 |
| randortho 3 | 0.208 | 0.212 | 0.422 | 0.506 | 0.83 | 0.370 |
| randortho 4 | 0.158 | 0.157 | 0.368 | 0.454 | 0.81 | 0.249 |

Mean ratio 0.84 (range 0.76–0.90). By the last non-trivial checkpoint (step
1,100,863, w ≈ 0.27) it is 0.93 (range 0.86–1.06).

The A(0) control passes cleanly and is what licenses reading those ratios as
saturation: **A(0) sits at the random floor in all 10 nets** (0.158–0.237 against
floors of 0.157–0.216, i.e. within noise of chance in every case). The subspace
genuinely moved from nothing, so arriving near the ceiling is a real trajectory
and not an artefact of starting there.

`dA/dw` — subspace drift per unit weight change — falls between the early and late
halves of the post-E1 window in **8 of 10** nets, typically by 2–4× (e.g. randortho 2:
+2.31 → +0.17; diag 2: +1.04 → +0.26). The two exceptions are diag 1 (early half is
negative, contaminated by the degeneracy plateau below) and randortho 3 (+0.29 →
+0.35, essentially flat). This is the mechanistic version of the same claim, and
unlike raw crossing times it does not depend on the log spacing of the snapshot grid.

## Result 4 — the row-shuffled control separates, but by less than the headline suggests

The degree-preserving control is the one that decides whether the early lock is
equitable structure or just incoming-strength structure. It separates, consistently
but modestly.

At w = 0.5 the trained curve is above the shuffled curve in **10 of 10** nets, mean
gap +0.072 (sd 0.032, paired t = 7.1 on 9 df). That is a real effect and it is not
seed-fragile.

But the gap is small against the span it lives in. The floor→ceiling distance at
that point is ~0.29, so +0.072 is about a quarter of it, and the shuffled trajectory
has itself covered 43–80% of the way from floor to ceiling by w ≈ 0.27. Pooled over
every post-E1 checkpoint the gap is +0.034 and positive in only 124 of 155 — in
randortho seeds 2, 3 and 4 the shuffled curve is at or above the trained one at
roughly half the individual checkpoints.

**So a substantial fraction of the early lock is degree structure.** The honest
statement is that the trained subspace locks somewhat earlier and somewhat harder
than a within-row permutation of the same weights, not that the control is flat
while the real thing rises. Any downstream claim resting on E2 should quote the gap,
not just the ratio in Result 3.

## Result 5 — restart reproducibility peaks mid-training and then declines

The ceiling is not monotone. It rises out of the floor, overshoots, and settles
back:

| arm | peak ceiling | at step | final ceiling | drop |
|---|---|---|---|---|
| diag | 0.634–0.656 | 33,179–66,841 | 0.470–0.543 | 0.10–0.16 |
| randortho | 0.522–0.588 | 16,470–33,179 | 0.452–0.553 | 0.01–0.09 |

WL is *most* reproducible on these matrices shortly after structure first appears,
and less reproducible on the fully trained matrix — by 16–26% in the diag arm and
2–16% in randortho, where seeds 1 and 3 barely decline at all. That is the same shape,
in an independent statistic, as Result 4 of the subspace tier — where `gamma_z`
peaks at 33,179 (diag) and 11,604 (randortho) and then reverts by roughly two-thirds.
Two different measurements of the partition both say mid-training is where it is
cleanest.

This also means the ceiling is doing real work in Result 3: A rises while the
ceiling falls, so a fixed reference line would have overstated late saturation and
understated the mid-training approach.

## Result 6 — step 0 is degenerate in both arms, for two different reasons

The step-0 degeneracy check, against 200 random partitions of W(0), with the final
checkpoint's Γ as the scale:

| arm | Γ_WL | mean Γ_rand | sd Γ_rand | Γ_final | Γ_WL / Γ_final | z |
|---|---|---|---|---|---|---|
| diag (all 5) | 0.000 | 0.000 | 0.000 | 154–189 | 0.000 | undefined |
| randortho 0 | 14.48 | 14.52 | 0.058 | 164.6 | 0.088 | −0.77 |
| randortho 1 | 14.89 | 14.89 | 0.068 | 162.7 | 0.092 | −0.01 |
| randortho 2 | 14.37 | 14.48 | 0.064 | 162.2 | 0.089 | −1.78 |
| randortho 3 | 14.27 | 14.44 | 0.071 | 165.0 | 0.086 | −2.41 |
| randortho 4 | 15.75 | 15.75 | 0.069 | 187.5 | 0.084 | −0.00 |

For diag the prediction is exactly met: under W = γI every partition is equitable,
so Γ_WL, mean Γ_rand and sd Γ_rand are all identically 0. The z is undefined because
the null has no variance — that is the correct answer for a diagonal matrix, not a
failure.

For randortho the matrix is dense but structureless: Γ_WL is 8.4–9.2% of the trained
scale and **statistically indistinguishable from a random partition of the same
matrix** (z from −0.01 to −2.41; none reaches −3). So the WL partition at step 0 is
not equitable in either arm. Restart agreement nonetheless leaves the floor within
~1,000–3,000 steps in the randortho arm — E1 asks whether the method is reproducible,
which is a weaker and earlier condition than whether the partition is good.

---

## Caveats

**randortho's `t_E1` is a marginal crossing; diag's is not.** Every randortho net
clears at 3.1–3.9σ against a 3.0σ threshold, while its own maximum margin later in
training is 37–58σ. Re-deciding the gate at other thresholds:

| net | σ=3 | σ=5 | σ=10 | σ=20 |
|---|---|---|---|---|
| diag 0/2/4 | 33,179 | 33,179 | 33,179 | 33,179 |
| diag 1 | 23,377 | 23,377 | 23,377 | 33,179 |
| diag 3 | 23,377 | 33,179 | 33,179 | 1,100,863 |
| randortho 0 | 1,419 | 2,859 | 8,176 | never |
| randortho 1 | 2,859 | 4,058 | 4,058 | never |
| randortho 2 | 1,419 | 4,058 | 5,760 | 775,616 |
| randortho 3 | 2,015 | 2,859 | 8,176 | 775,616 |
| randortho 4 | 1,000 | 1,419 | 2,859 | 5,760 |

The diag arm is stable to σ=10; the randortho values move by up to 6× over the same
range and three nets stop clearing at all by σ=20. The **arm ordering** in Result 1
survives every threshold tried, but the randortho numbers should be quoted as "within
the first few thousand steps", not as point estimates. `FLOOR_SIGMA` is a module
constant and `margin_sigma` is in the CSV, so this can be re-decided without re-running.

**The diag early plateau is degeneracy, not convergence** — the same caveat as the
subspace tier, and it shows up here as `A` being *exactly* constant to 6 decimals for
the first 10–11 checkpoints (through step 16,470 or 23,377), one distinct value per
seed. A near-diagonal `W_rec` makes every unit identical to WL, so the partition is
decided by k-means tie-breaking. Those points carry no information and the flat A
must not be read as an early lock.

**`A_norm` (fraction of the floor→ceiling span) is undefined before the ceiling
separates from the floor.** Where ceiling ≈ floor the denominator is ~0 and the
normalised values explode to ±50 or worse, including sign flips. Result 4's normalised
statements are restricted to post-E1 checkpoints for this reason; the raw A, ceiling
and floor columns are the ones to use early.

**A can slightly exceed the ceiling.** randortho 0 reaches A/ceiling = 1.056 at step
1,100,863. This is not a bug: the ceiling is a 20-restart estimate on that checkpoint's
matrix while A compares against the final partition, so the two are not hard-ordered
when the ceiling estimate is noisy. It does mean ratios near 1.0 should be read as
"at the ceiling within estimation noise", not as exact saturation.

**The final checkpoint is trivial by construction** — A = 1 and A_shuffled = 1, since
each is compared with itself. It is excluded from every trend, gap and dA/dw statement
above; the last informative point is step 1,100,863 (w ≈ 0.27, and 0.35 in randortho 4).

**The shuffled control has no ceiling of its own.** `A_shuffled` is agreement between
the shuffled W(t) partition and the shuffled W(final) partition, but restart
reproducibility was never computed on the shuffled matrices. Result 4 therefore compares
two raw agreement curves. If WL is systematically less reproducible on row-shuffled
matrices, part of the +0.072 gap is a ceiling difference rather than a locking
difference. Computing the shuffled restart ensemble is the cheapest way to close this,
and it would sharpen the one result here that is currently equivocal.

**seed 4 in both arms has k = 30/31 against 42–46 elsewhere**, carried over from the
`k_from_clustering` policy. Those two nets have the lowest floors (0.157/0.166 vs
~0.215) and the lowest ceilings, as expected for smaller k — the ratios in Result 3
are unremarkable for them (0.81, 0.82), so the k difference appears to be absorbed
correctly by the per-net floor and ceiling. Same standing caveat as the subspace doc:
treat it as a k effect until shown otherwise.

**WL occasionally returns fewer than k clusters, but never after `t_E1`.** `k_realised`
dips to 40–42 against a requested 42–43 in diag seeds 1–3, entirely at pre-E1
checkpoints. Post-E1, every restart in every net returns exactly k.

## Open

`trajectory_motifs` (fixed points at ~5 snapshots) is still unrun, and it is the tier
that tests the premise these two results set up: the coordinate system is at ~84% of
its ceiling with half the weight change remaining, while the subspace tier's
`msca_vs_final` is still drifting at the end of training. Whether the *motifs* settle
before or after the coordinate system does is the question neither tier can answer.

`figures/trajectory_curves.py` panel D still renders its "no trajectory_motifs.csv"
placeholder.
