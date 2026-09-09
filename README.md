# flexsym

Code for *Connectivity predicts where computation forms and where it can be
changed* (Sale, Coraggio, Kaplan & Richardson).

Grouping units by the total input each receives from every other group defines
structural **roles** from the recurrent weight matrix alone, without running any
task. This repository trains the networks the paper analyses, infers those roles,
recovers the fixed-point motifs the networks compute with, and produces every
number, table and figure the manuscript reports.

The networks and the 15-task suite build on Driscoll, Shenoy & Sussillo (2024).
The five upstream files this depends on are vendored under `flexsym/core/`; the
rest of that project is not reproduced here.

## Setup

```bash
conda create -n flexsym python=3.10
conda activate flexsym
pip install -e .

cp .env.example .env    # then edit the roots for your machine
```

`.env` defines where the bulk data lives. Nothing in the code hardcodes an
absolute path: every root resolves through `flexsym.paths`, and a relative value
resolves against the repository root rather than the working directory, so a
module writes to the same place wherever it is invoked from. Roots are not
validated at import, so importing a module does not require the data volume to
be mounted.

Fixed-point finding additionally needs FixedPointFinder and recurrent-whisperer,
which are not pip-installable. They are reached through `sys.path` from
`external/`:

```bash
mkdir -p external && cd external
git clone https://github.com/mattgolub/fixed-point-finder.git
git clone https://github.com/mattgolub/recurrent-whisperer.git
```

## Data

The networks released with Driscoll et al. (2024) are at the Allen Institute
database, buckets `multitask_shared` (networks) and `multitask_processed`
(processed data, including fixed point locations):
https://open.quiltdata.com/b/aind-trained-networks/tree/

The size × seed grid the main text analyses is trained locally by
`flexsym.training`. Checkpoints and fixed-point output run to tens of GB and are
expected on a separate volume pointed at by `MODEL_ROOT`.

Analysis output is written under `results/`, split by cohort so that results
computed on different sets of checkpoints cannot be pooled by accident:
`results/main/` for the locally trained grid, which backs the main text, and
`results/released/` for the Driscoll networks, which back the supplement.
`RESULTS_ROOT` in `.env` selects between them.

## Layout

```
src/flexsym/
  paths.py       every filesystem root, resolved from .env
  core/          vendored upstream: network, task, train, tools, variance
  training/      the size x seed training grid and its epoch-variance clustering
  structural/    signed-WL role partition, operating k, activity coherence
  motifs/        motif classification and per-unit participation
  run/           analysis entry points, one per reported result
  figures/       the panels of Figures 1-3
  supp/          every supplementary table and figure
```

Entry points are modules, not scripts:

```bash
python -m flexsym.run.motif_fixed_points --help
python -m flexsym.figures.paper_fig2_panels
```

## Reproducing the results

The stages are ordered; each reads what the previous one wrote.

1. **Train the grid.** `flexsym.training.run_seeds_parallel` fans
   `train_softplus_diag` across cores, one process per (size, seed).
2. **Cluster units by epoch variance.**
   `flexsym.training.compute_variance_epoch_to_file`, then
   `flexsym.training.cluster_epoch_variance`. This produces the activity-defined
   clustering the role partition is compared against.
3. **Infer the roles.** `flexsym.structural.signed_input_wl_structural_roles`
   computes the signed Weisfeiler-Leman partition over a k-sweep.
   `flexsym.run.batch_256_transpose_analysis` drives stages 2 and 3 over a cohort
   with per-network k selection, checkpointed per model and per step:

   ```bash
   python -m flexsym.run.batch_256_transpose_analysis \
       --config configs/batch_256_transpose_config.json
   ```

   `flexsym.structural.operating_k` is the single source of truth for each
   network's operating k.
4. **Recover the motifs.** `flexsym.run.motif_fixed_points` finds fixed and slow
   points and their Jacobians; `flexsym.motifs.motif_classify` assigns each
   task × epoch condition a motif type. This stage needs the external
   dependencies above.
5. **Run the analyses.** The modules under `flexsym.run`, listed below.
6. **Build the manuscript.** `flexsym.supp.tables` and `flexsym.supp.figures`
   write the supplement; `flexsym.figures.paper_fig*` write the main figures.

## What produces what

Main figures, into `figures/figN/pieces/`:

| Figure | Module |
|---|---|
| 1, schematic panel | `figures.paper_fig1_data`, then `figures.paper_fig1` |
| 1, coarse-graining | `figures.paper_fig1_partition` |
| 1, surrogate controls | `figures.paper_fig1_controls` |
| 2 | `figures.paper_fig2_panels`, `figures.paper_fig2_motif_example` |
| 3 | `figures.paper_fig3_panels` |

Supplementary tables, written by `flexsym.supp.tables` from the result each
analysis leaves in `results/`:

| Table | Analysis |
|---|---|
| networks, tasks, performance | the training grid and its logs |
| taxonomy, taxonomy decision | `run.motif_taxonomy_all_nets` |
| lumpability k-sweep | `run.lumpability_r2_ksweep` |
| surrogates | `run.control_ksweep` |
| equitability | `run.equitability_control_ksweep` |
| released cohort | `run.lumpability_r2_ksweep`, `run.tanh_lumpability` |
| participation | `run.motif_participation_aggregate`, `run.motif_participation_stats` |
| partition family | `run.partition_family_comparison`, `run.motif_subspace_rva` |
| quotient k-sweep, quotient m | `run.motif_quotient_prediction` |
| eigenvalue recovery | `run.quotient_lambda_pairs` |
| eigenvalue-matched control | `run.motif_quotient_eigmatched_control` |
| held-out gains | `run.motif_quotient_heldout` |
| quotient ablation, compression | `run.motif_quotient_controls` |
| routing, composition | `run.quotient_mode_motif_alignment` |
| operator reuse | `run.motif_operator_composition` |
| gain swap, susceptibility | `run.gain_configuration` |
| allocation | `run.gain_leverage_geometry` |
| targeted modulation | `run.gain_targeted_modulation` |

Supplementary figures are written by `flexsym.supp.figures`:

| Figure | Analysis |
|---|---|
| generalization across size and cohort | `run.lumpability_r2_ksweep`, `run.motif_quotient_prediction`, `run.gain_configuration`, `run.gain_targeted_modulation` |
| restart reproducibility | `run.restart_ari_vs_r2` |
| reuse and composition | `run.motif_participation_stats`, `run.motif_operator_composition`, `run.motif_direct_sum_residual` |

## Manuscript

The manuscript sources are not tracked here. The code that generates its tables
and figures is: `flexsym.supp` is the only place a reported value is formatted,
so every number in the supplement can be traced to the result file it came from.
