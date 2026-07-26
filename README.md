# flexsym

Connectivity symmetry and dynamical motifs in multitask RNNs.

This repo asks whether the approximate symmetries of a trained RNN's recurrent
weight matrix — its signed-WL equitable partition into role cells — constitute a
real organising principle for the network's computation. It builds on the
networks and task suite released with Driscoll et al. (2024); see
[docs/provenance.md](docs/provenance.md) for exactly what is vendored from
upstream and what was modified.

The work is organised around three claims:

1. **Connectivity defines a dynamical role structure.** `W_rec` partitions into
   groups of units that wire to the rest of the circuit the same way, and those
   groups are an invariant subspace of the dynamics.
2. **The network's computational motifs are organised by that structure.** The
   fixed-point motifs the network computes with recruit role groups
   non-uniformly.
3. **The quotient predicts the motifs.** Collapsing each role group to one node
   yields a coarse-grained circuit that predicts the directions the motifs
   occupy in state space.

[docs/findings.md](docs/findings.md) records where each claim stands, **including
the three controls that demoted results which previously looked like headlines**.
Read it before building on any number from this repo.

## Setup

```bash
conda create -n flexsym python=3.10
conda activate flexsym
pip install -e .

cp .env.example .env    # then edit the roots for your machine
```

`.env` defines where the bulk data lives (see `.env.example`). Nothing in the
code hardcodes an absolute path — everything routes through `flexsym.paths`.

### External dependencies

Fixed-point finding needs FixedPointFinder and recurrent-whisperer, which are
not pip-installable here. They live under `external/` (gitignored) and are
reached via `sys.path`:

```bash
mkdir -p external && cd external
git clone https://github.com/mattgolub/fixed-point-finder.git
git clone https://github.com/mattgolub/recurrent-whisperer.git
```

## Data

Trained networks and processed data from the original paper are at the Allen
Institute database — buckets `multitask_shared` (networks) and
`multitask_processed` (processed data, including fixed point locations):
https://open.quiltdata.com/b/aind-trained-networks/tree/

The size × seed training grid used here is generated locally by
`flexsym.training`. The fixed-point outputs are large (tens of GB) and are
expected to live on a separate volume pointed at by `MODEL_ROOT`.

## Layout

```
src/flexsym/
  paths.py       resolves every data/results root from .env
  core/          vendored upstream: network, task, train, tools, variance
  training/      training the size x seed network grid
  structural/    signed-WL partition, stability, ensembles, lesions
  motifs/        fixed-point inventory, motif classification, participation
  figures/       paper panels and diagnostic plots
  slides/        presentation figures
  run/           analysis entry points
```

Entry points are modules, not scripts — run them with `-m`:

```bash
python -m flexsym.run.motif_fixed_points --help
python -m flexsym.figures.fig2_panels
```

## Documentation

- [docs/methods.md](docs/methods.md) — what each analysis does
- [docs/figure_map.md](docs/figure_map.md) — each figure → the module that makes it
- [docs/findings.md](docs/findings.md) — current results and their status
- [docs/provenance.md](docs/provenance.md) — relation to the upstream repo
