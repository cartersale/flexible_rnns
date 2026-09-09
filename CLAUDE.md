# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project

Code for *Connectivity predicts where computation forms and where it can be
changed* (Sale, Coraggio, Kaplan & Richardson). Units are grouped by the total
input each receives from every other group, which defines structural **roles**
from the recurrent weight matrix alone. The analyses ask whether those roles
organise the fixed-point motifs the networks compute with, whether a role-level
quotient operator predicts motif geometry, and whether gain applied to a
predicted role moves a target mode.

Networks and the 15-task suite build on Driscoll, Shenoy & Sussillo (2024). The
five load-bearing upstream files are vendored under `flexsym/core/`.

See `README.md` for setup, the pipeline order, and the map from each reported
table and figure to the module that produces it. Read that map before adding an
analysis, and extend it when you do.

## Environment

Python 3.10, TensorFlow 2 in TF1 compatibility mode:

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

The working conda environment is `myenv`. Install with `pip install -e .`.

## Paths

`flexsym.paths` is the only place a filesystem root is defined; it reads them
from `.env`. Do not add an absolute path anywhere else, including config JSON.
Relative roots resolve against the repository root, not the working directory.

Roots are deliberately not validated at import time, so importing a module never
requires the data volume to be mounted. Call `paths.require()` or
`paths.ensure_dir()` at the point of use so a missing volume fails with a message
naming the root.

`results/` is split by cohort: `main/` is the locally trained grid behind the
main text, `released/` the Driscoll networks behind the supplement. They are
never pooled. `RESULTS_ROOT` selects between them.

## Conventions

- Entry points are modules, not scripts: `python -m flexsym.run.<name>`.
- Network identity is a string encoding type, activation, initialisation, size,
  ruleset and seed: `LeakyRNN_softplus_diag_256_all_0`.
- `flexsym.structural.operating_k` is the single source of truth for each
  network's operating k. Do not inline a k literal.
- The quotient in raw role coordinates satisfies `W H = H Q`, so an eigenvector
  of `Q` lifts to units as `H z`. In orthonormal coordinates `S = H D^-1/2` the
  operator is `Q~ = D^1/2 Q D^-1/2` and the lift is through `S`. Lifting an
  eigenvector of the raw `Q` through `S` crosses the two coordinate systems.
- `flexsym.supp` is the only place a reported value is formatted. A number in the
  manuscript should be traceable to the result file it came from.
- `np.load` is monkeypatched when `flexsym.core.task` is imported; passing
  `allow_pickle=True` explicitly after that import raises `TypeError`.

## Scope

Only code backing the main text and supplement lives on `main`. Exploratory
analyses, superseded figure plans, and presentation material were removed; they
remain in history at `c1fba09`. Manuscript sources are untracked.

## Writing

`docs/wiriting_heuristics.md` governs every manuscript edit: minimal
intervention, no defensive clarification, no code identifiers or repository
process in the prose.
