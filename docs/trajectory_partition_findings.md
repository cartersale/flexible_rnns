# Partition quality across training — equitability and lumpability

How the signed-WL partition and its quality evolve over training, measured at
every checkpoint of the 5 × 2 trajectory grid.

**Date:** 2026-07-31
**Code:** `flexsym.run.trajectory_subspace` at `77b129d` **+ uncommitted changes**
(the equitability/lumpability block described below was added for this run and is
not yet committed).
**Data:** `results/trajectory/trajectory_subspace.{csv,json}` — 230 rows
(10 nets × 23 snapshots).

```bash
python -m flexsym.run.trajectory_subspace     # 5.5 min for the whole grid
```

## Networks

10 softplus LeakyRNNs, `n_rnn=256`, ruleset `all`, trained by
`flexsym.training.train_trajectory` to the full 1e8-trial budget with a 23-point
log-spaced snapshot grid (steps 0 → 1,562,500). Two arms differing only in
`w_rec_init`: **diag** (seeds 0–4) and **randortho** (seeds 0–4). All 10 carry a
`TRAJECTORY_COMPLETE` sentinel; wall clock 44.3–44.4 h per net.

Final performance: `perf_avg` 0.918–0.954, `perf_min` 0.669–0.844. Both arms
train normally; nothing here is a failed run.

## What is measured

At every snapshot the signed-WL partition is recomputed on that checkpoint's own
`W_rec`, with the **active unit set and k held fixed** at the values from the
net's final checkpoint (`operating_k.k_from_clustering`). Letting those float
would confound "the subspace moved" with "we partitioned a different set of units
at a different granularity".

| measure | column | meaning |
|---|---|---|
| Subspace agreement | `msca_vs_final` | mean squared principal-angle cosine between S_t and S_final |
| **Equitability, raw** | `gamma` | the signed-WL cost the partition is optimised on |
| **Equitability, scaled** | `gamma_rel` | same defect over the norm it is a defect of |
| **Equitability, vs chance** | `gamma_z` | z against 1000 size-matched label shuffles; negative = more equitable than chance |
| **Lumpability, invariance** | `*_invariance`, `*_invariance_z` | ‖(I−P)MP‖/‖MP‖; 0 = the group-constant subspace is M-invariant |
| **Lumpability, eigenvectors** | `*_eig_leakage`, `*_eig_leakage_z` | mean ‖v−Pv‖/‖v‖ over the top-\|λ\| eigenvectors |

Lumpability is reported for two matrices: `W_*` on the raw recurrent matrix (the
dynamics the motifs live in) and `transpose_*` on the transpose variant the
partition is actually fitted to. Nulls: 1000 shuffles for equitability
(matching `run.equitability_soft`), 200 for lumpability (matching
`run.lumpability_check`).

Both lumpability fast paths are checked against the reference implementations in
`run.lumpability_check` to 1e-9 by `_selfcheck()`, which runs before every
invocation.

---

## Result 1 — the raw score and the null-normalised score move in *opposite* directions

Over diag training, `gamma_rel` rises 0.03 → 0.39 while `gamma_z` falls 0 → −6.

These are not in conflict. The absolute equitability defect grows because the
matrix itself gets richer; relative to what a random partition of the same matrix
achieves, the structural partition pulls steadily further ahead. **The raw score
alone reports the opposite of what is happening** — it is not comparable across
checkpoints, because `gamma` is a Frobenius norm and the weight scale is not
constant during training. Read `gamma_z`.

## Result 2 — the two inits take completely different routes to the same endpoint

Onset of above-chance equitability, per net (first snapshot with `gamma_z < −3`):

| init | seed | onset_step | perf_avg_at_onset |
|---|---|---|---|
| diag | 0 | 33179 | 0.743 |
| diag | 1 | 23377 | 0.605 |
| diag | 2 | 33179 | 0.709 |
| diag | 3 | 33179 | 0.724 |
| diag | 4 | 33179 | 0.618 |
| randortho | 0 | 1000 | 0.302 |
| randortho | 1 | 1000 | 0.225 |
| randortho | 2 | 1000 | 0.165 |
| randortho | 3 | 2015 | 0.581 |
| randortho | 4 | 1000 | 0.426 |

**diag** sits at chance on all three measures for its first 11 snapshots, then
transitions abruptly: within one snapshot (23,377 → 33,179) the arm mean
`gamma_z` goes −5.5 → −19.6, `W_leak_z` −2.2 → −11.2, `W_inv_z` −5.2 → −20.5.
4 of 5 seeds cross at exactly step 33,179.

**randortho** is already above chance at its first non-zero snapshot (step 1,000,
4 of 5 seeds) and deepens smoothly, with no transition.

The endpoint is init-independent:

| arm mean at step 1,562,500 | `gamma_z` | `W_leak_z` | `W_inv_z` | `T_leak_z` | `T_inv_z` |
|---|---|---|---|---|---|
| diag | −6.22 | −13.74 | −13.83 | −20.56 | −14.33 |
| randortho | −6.45 | −13.55 | −13.38 | −21.95 | −15.01 |

Same destination, different path.

## Result 3 — lumpability settles long before the subspace stops moving

From step 134,652 to the end — the last eight snapshots, an order of magnitude of
training — the lumpability z's stay inside a narrow band with no trend:

| arm | `W_leak_z` range | `W_inv_z` range |
|---|---|---|
| diag | −15.7 … −13.7 | −13.8 … −10.8 |
| randortho | −16.5 … −13.6 | −13.4 … −9.4 |

Over that same span `msca_vs_final` climbs from ~0.30 to 1.0, and stays **below
each net's own restart-noise floor** (0.441–0.552) until the last snapshot or two.

So the WL subspace is still drifting at the end of training while lumpability has
been essentially settled since ~134k steps. This is directly relevant to the
premise `trajectory_motifs` is meant to test — but that tier has not been run,
and it is what actually tests it.

## Result 4 — the equitability advantage peaks mid-training and partly reverts

Unlike the lumpability measures, `gamma_z` is strongly non-monotonic in both arms:

| arm | peak `gamma_z` | at step | weakest after peak | final |
|---|---|---|---|---|
| diag | −19.6 | 33,179 | −3.4 | −6.2 |
| randortho | −14.1 | 11,604 | −2.0 | −6.4 |

The partition is *most* equitable-above-chance shortly after structure first
appears, loses roughly two-thirds of that margin, then recovers partway by the end.
Both arms show the same shape at different times. Lumpability does not do this —
it rises and stays. Whatever drives the excursion is specific to the WL cost, and
this write-up does not explain it.

---

## Full trajectories (mean over 5 seeds)

### diag

| step | perf_avg | msca_vs_final | gamma_rel | gamma_z | W_leak_z | W_inv_z | T_leak_z | T_inv_z |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.091 | 0.208 | 0.000 |  | -1.368 | 0.997 | -1.368 | 0.997 |
| 1000 | 0.113 | 0.208 | 0.033 | 0.072 | -0.043 | 0.102 | -0.350 | -0.083 |
| 1419 | 0.127 | 0.208 | 0.033 | 0.073 | -0.093 | 0.224 | -0.363 | -0.060 |
| 2015 | 0.164 | 0.208 | 0.033 | -0.076 | 0.474 | 0.401 | -0.279 | -0.138 |
| 2859 | 0.233 | 0.208 | 0.034 | -0.057 | 0.228 | 0.381 | 0.138 | -0.229 |
| 4058 | 0.346 | 0.208 | 0.035 | 0.060 | 0.420 | -0.133 | -0.234 | -0.068 |
| 5760 | 0.438 | 0.208 | 0.038 | 0.296 | -0.773 | -0.234 | 0.068 | 0.270 |
| 8176 | 0.444 | 0.208 | 0.050 | 0.253 | -0.037 | -0.393 | 0.049 | 0.131 |
| 11604 | 0.477 | 0.208 | 0.079 | 0.086 | 0.121 | -0.383 | -0.557 | 0.156 |
| 16470 | 0.520 | 0.208 | 0.138 | 0.176 | -0.382 | -0.158 | 0.439 | 0.183 |
| 23377 | 0.585 | 0.211 | 0.220 | -5.501 | -2.220 | -5.248 | -3.156 | -7.311 |
| 33179 | 0.703 | 0.261 | 0.222 | -19.593 | -11.187 | -20.540 | -17.041 | -24.134 |
| 47093 | 0.709 | 0.298 | 0.269 | -14.872 | -12.488 | -17.072 | -18.810 | -25.229 |
| 66841 | 0.731 | 0.292 | 0.307 | -9.283 | -13.054 | -14.589 | -19.393 | -22.603 |
| 94870 | 0.813 | 0.313 | 0.322 | -7.043 | -12.933 | -13.109 | -20.434 | -18.648 |
| 134652 | 0.920 | 0.340 | 0.342 | -3.357 | -14.402 | -10.817 | -19.977 | -12.861 |
| 191117 | 0.932 | 0.354 | 0.350 | -3.584 | -14.478 | -11.635 | -19.715 | -12.529 |
| 271261 | 0.928 | 0.381 | 0.349 | -3.818 | -15.705 | -13.134 | -19.586 | -12.809 |
| 385011 | 0.933 | 0.385 | 0.355 | -3.543 | -14.433 | -11.410 | -20.260 | -12.664 |
| 546462 | 0.939 | 0.425 | 0.363 | -4.297 | -15.179 | -12.670 | -20.889 | -13.796 |
| 775616 | 0.931 | 0.442 | 0.359 | -4.865 | -14.769 | -12.765 | -20.389 | -14.599 |
| 1100863 | 0.934 | 0.483 | 0.377 | -5.355 | -15.639 | -13.600 | -21.469 | -14.767 |
| 1562500 | 0.942 | 1.000 | 0.388 | -6.219 | -13.735 | -13.827 | -20.555 | -14.334 |

### randortho

| step | perf_avg | msca_vs_final | gamma_rel | gamma_z | W_leak_z | W_inv_z | T_leak_z | T_inv_z |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.101 | 0.203 | 0.402 | -1.027 | 0.141 | -0.102 | -0.771 | -0.841 |
| 1000 | 0.247 | 0.201 | 0.396 | -3.961 | -0.455 | -3.866 | -2.390 | -6.253 |
| 1419 | 0.336 | 0.204 | 0.400 | -3.853 | -1.067 | -4.023 | -4.643 | -7.870 |
| 2015 | 0.546 | 0.208 | 0.381 | -5.079 | -2.352 | -7.152 | -5.595 | -12.089 |
| 2859 | 0.466 | 0.206 | 0.370 | -5.708 | -2.131 | -7.216 | -5.811 | -13.770 |
| 4058 | 0.520 | 0.213 | 0.360 | -7.508 | -3.870 | -8.914 | -7.959 | -17.073 |
| 5760 | 0.532 | 0.208 | 0.335 | -9.009 | -6.939 | -11.427 | -9.863 | -17.501 |
| 8176 | 0.576 | 0.220 | 0.308 | -10.121 | -9.429 | -11.774 | -15.737 | -17.925 |
| 11604 | 0.633 | 0.232 | 0.282 | -14.125 | -11.909 | -11.183 | -18.246 | -20.143 |
| 16470 | 0.624 | 0.254 | 0.276 | -12.176 | -12.459 | -9.006 | -22.049 | -19.169 |
| 23377 | 0.746 | 0.257 | 0.300 | -7.402 | -14.596 | -8.813 | -21.131 | -18.721 |
| 33179 | 0.817 | 0.279 | 0.314 | -3.575 | -15.243 | -7.176 | -23.739 | -16.044 |
| 47093 | 0.847 | 0.285 | 0.327 | -2.028 | -17.512 | -8.539 | -23.812 | -13.813 |
| 66841 | 0.886 | 0.290 | 0.334 | -2.208 | -15.125 | -9.557 | -21.095 | -12.650 |
| 94870 | 0.883 | 0.309 | 0.336 | -2.225 | -15.762 | -9.273 | -22.187 | -11.947 |
| 134652 | 0.900 | 0.305 | 0.344 | -2.464 | -16.285 | -9.464 | -23.158 | -11.687 |
| 191117 | 0.925 | 0.321 | 0.347 | -2.552 | -16.010 | -9.408 | -24.825 | -12.330 |
| 271261 | 0.919 | 0.342 | 0.351 | -2.940 | -16.532 | -9.688 | -22.305 | -12.568 |
| 385011 | 0.944 | 0.374 | 0.356 | -3.239 | -15.012 | -9.380 | -24.997 | -13.256 |
| 546462 | 0.932 | 0.387 | 0.360 | -3.045 | -15.991 | -10.260 | -23.319 | -13.695 |
| 775616 | 0.937 | 0.419 | 0.371 | -4.522 | -14.325 | -10.325 | -24.046 | -14.190 |
| 1100863 | 0.936 | 0.473 | 0.378 | -5.585 | -14.167 | -12.412 | -22.923 | -15.678 |
| 1562500 | 0.932 | 1.000 | 0.385 | -6.450 | -13.551 | -13.379 | -21.953 | -15.008 |

## Final snapshot, per net

| init | seed | k | n_active | floor_msca | gamma | gamma_rel | gamma_z | W_leak_z | W_inv_z | T_leak_z | T_inv_z |
|---|---|---|---|---|---|---|---|---|---|---|---|
| diag | 0 | 43 | 215 | 0.534 | 155.442 | 0.387 | -8.440 | -12.090 | -13.069 | -23.560 | -17.278 |
| diag | 1 | 42 | 209 | 0.513 | 154.555 | 0.384 | -7.372 | -12.781 | -15.443 | -17.771 | -16.546 |
| diag | 2 | 42 | 210 | 0.543 | 156.798 | 0.384 | -7.351 | -15.565 | -15.006 | -19.148 | -14.899 |
| diag | 3 | 42 | 211 | 0.536 | 157.760 | 0.385 | -5.660 | -12.940 | -12.955 | -20.773 | -13.548 |
| diag | 4 | 30 | 213 | 0.470 | 188.858 | 0.398 | -2.273 | -15.299 | -12.664 | -21.524 | -9.399 |
| randortho | 0 | 45 | 223 | 0.493 | 164.635 | 0.370 | -6.231 | -10.448 | -12.927 | -18.707 | -14.323 |
| randortho | 1 | 46 | 229 | 0.552 | 162.678 | 0.379 | -7.782 | -14.335 | -16.832 | -22.040 | -17.337 |
| randortho | 2 | 44 | 222 | 0.500 | 162.179 | 0.390 | -7.655 | -12.919 | -14.058 | -25.216 | -17.947 |
| randortho | 3 | 44 | 222 | 0.528 | 165.014 | 0.402 | -5.594 | -13.703 | -12.953 | -22.833 | -16.483 |
| randortho | 4 | 31 | 233 | 0.441 | 187.511 | 0.385 | -4.986 | -16.352 | -10.123 | -20.970 | -8.952 |

---

## Caveats

**The diag early plateau is degeneracy, not learning.** `msca_vs_final` is
*exactly* constant across the first 11 diag snapshots — one distinct value per
seed to four decimals (0.2161, 0.2249, 0.2116, 0.2183, 0.1678). A near-diagonal
`W_rec` makes every unit look identical to WL, so the partition is decided by
k-means tie-breaking on identical rows and is seed-determined. Read those points
as "the initialisation has no structure to find". Do **not** read the flat `msca`
as "the subspace has already converged".

**Five NaNs, all at diag step 0.** `gamma` is exactly 0.0 there and every label
shuffle returns the same value, so the null has zero variance and the z is
undefined. That is the honest answer for a diagonal matrix, not a failure.
`W_invariance_z` and `transpose_invariance_z` are NaN in 4 of those 5 rows for
the same reason.

**seed4 in both arms has k = 30/31 against 42–46 elsewhere**, selected by the
`k_from_clustering` active_fraction policy from its own clustering. Those two
nets are also the weakest on `gamma_z` (diag seed4 = −2.27) and on
`transpose_invariance_z` (−9.40 / −8.95). This is most likely a k effect rather
than a property of those networks — check before treating seed4 as an outlier.

**Every net's restart floor is well below the 0.63 reference** carried from
`docs/methods_results_report.tex` — 0.441–0.552 here, recomputed per net. The
per-net floor is the one to judge `msca_vs_final` against.

**`trajectory_motifs` has not been run.** Result 3 states when partition quality
stops changing; whether the *motifs* settle after the coordinate system does is
the fixed-point tier's question, and it is still open.

## Reproducing

```bash
python -m flexsym.run.trajectory_subspace                          # all 10
python -m flexsym.run.trajectory_subspace --init diag --seeds 0    # one net, ~33 s
```

The first run per net computes `variance_epoch.pkl` and the activity clustering
that fix the unit set and k (a TF pass over `model.ckpt`); both are cached beside
the snapshots, so re-runs skip it. `--recompute-reference` forces it.
Null counts are `--n-null-gamma` (default 1000) and `--n-null-lump` (default 200);
`--m-eig` sets the number of top-|λ| eigenvectors (0 = k).
