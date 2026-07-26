# Provenance of vendored code

This repo builds on the code released with:

> Driscoll, L.N., Shenoy, K., Sussillo, D. (2024). *Flexible multitask
> computation in recurrent networks utilizes shared dynamical motifs.*
> Nature Neuroscience. https://www.nature.com/articles/s41593-024-01668-6

Upstream repository: https://github.com/lauradriscoll/flexible_multitask

Vendored at upstream commit `2482af08fd7f286a04dac125df1edad32059ce1e`
(`main`, "Update README.md with new link"). The vendored files themselves were
last changed upstream in `69ed62f5196bd9eb474ee73c9953b104c1c37500`
(2024-03-29, "main figs all updated with tf2 and py3").

## What was vendored

Copied from upstream `stepnet/` into `src/flexsym/core/`:

| File | Status |
|---|---|
| `network.py` | Unmodified apart from import rewrites |
| `task.py` | Unmodified apart from import rewrites |
| `tools.py` | Unmodified apart from import rewrites |
| `variance.py` (from `stepnet/analysis/`) | Unmodified apart from import rewrites |
| `train.py` | **Forked** — see below |

"Import rewrites" means bare `import tools` → `from flexsym.core import tools`,
required by the package layout. No behavioural changes.

### `train.py` is a fork, not a copy

Diverges from upstream by roughly +83/−7 lines. The changes:

- **Device selection** — use a GPU if one is visible, otherwise fall back to CPU
  so the same code runs on headless CPU-only nodes.
- **Thread-pool control** — CPU intra/inter-op thread counts read from the
  environment, so a parallel launcher can cap each worker process (e.g.
  `TF_INTRA_THREADS=2` across 5 workers on a 10-core machine).
- **Resume from checkpoint** — restart from `model.ckpt` + `log.json` (weights,
  optimizer slots, and performance history), keeping wall-clock monotonic.
- **Wall-clock deadline** — `max_hours` safety net so a large net on CPU that
  never reaches `target_perf` terminates rather than running forever, with an
  explicit did-not-converge flag.
- **Final save** — persist the converged model even when training ends between
  `display_step` boundaries.

These exist to make the size × seed training grid runnable on the local
hardware. Behaviour on a converging GPU run is intended to match upstream.

## What was deliberately NOT vendored

Upstream's `analysis/`, `figs/`, `transfer_learn/`, `build/`, `utils/`, and the
remainder of `stepnet/analysis/` (clustering, dimensionality, taskset, …) are
not used by any code here. They remain available in the upstream repository.

Also dropped, as unused upstream scaffolding: `general_model_train.py`,
`hp_sweep.py`, `leave_one_out.py`, `pre_transfer.py`, `run_analyses.py`,
`save_pre_train_model_params.py`, `save_post_train_model_params.py`,
`tools_lnd.py`.

## External dependencies (not vendored, not on PyPI)

`external/` holds FixedPointFinder and recurrent-whisperer, reached via
`sys.path` rather than pip. See the README for setup.
