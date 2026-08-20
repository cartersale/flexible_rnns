"""Single source of truth for the operating k of each 256-unit network.

This used to be an `OPK = {0: 41, ..., 3: 14, ...}` literal copy-pasted into
eight modules. When the 256 seeds 3-9 were retrained on 2026-07-25 every copy
went stale at once: they held k=14, derived from checkpoints where only ~70 of
256 units were active, while the current nets want k=42-44. Nothing failed
loudly -- the analyses just silently kept partitioning at the old granularity.

The k is now derived from each network's activity clustering, using the same
rule the batch configs use (`configs/batch_rnn_softplus_diag*.json`, policy
"active_fraction"):

    k = clamp(min(activity_best_k, round(0.20 * n_active)), K_MIN, n_active - 1)

It reproduces the released seeds 0-2 exactly (41, 51, 38), which is the check
that licenses trusting it on the retrained seeds.

Usage -- drop-in for the old literal:

    from flexsym.structural.operating_k import OPK
    for seed, k in OPK.items(): ...

`OPK` resolves lazily and caches, so importing this module does not touch the
filesystem and a missing activity file only raises when that seed is asked for.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Iterator

import numpy as np

from flexsym import paths

ACTIVITY_ROOT = paths.RESULTS_ROOT / "activity_clusters_paper"
ACTIVITY_NAME = "epoch_variance_unit_clusters_paper_max.npz"

K_MIN = 3
ACTIVE_FRACTION = 0.20
SEEDS_256 = tuple(range(10))


def model_name(seed: int, size: int = 256) -> str:
    return f"LeakyRNN_softplus_diag_{size}_all_{seed}"


def k_from_clustering(best_k: int, n_active: int) -> int:
    """The active_fraction policy, applied to one clustering result.

    Split out from `operating_k` so callers holding a clustering that does not
    live under ACTIVITY_ROOT -- e.g. the per-checkpoint trajectory runs, whose
    clusterings sit beside their snapshots -- select k by the same rule instead
    of re-deriving it. Re-deriving is exactly how the eight copies of the old
    OPK literal went stale at once.
    """
    structural_k = max(K_MIN, min(int(round(ACTIVE_FRACTION * n_active)), n_active - 1))
    return max(K_MIN, min(min(int(best_k), structural_k), n_active - 1))


def operating_k(model: str) -> int | None:
    """Operating k for `model`, or None if it has no activity clustering yet."""
    path = ACTIVITY_ROOT / model / ACTIVITY_NAME
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    n_active = int(np.asarray(z["ind_active"], dtype=np.int64).size)
    return k_from_clustering(int(z["best_k"]), n_active)


class _OperatingK(Mapping):
    """Lazy seed -> k mapping over the 256-unit networks."""

    def __init__(self, seeds=SEEDS_256, size: int = 256):
        self._seeds = tuple(seeds)
        self._size = size
        self._cache: dict[int, int] = {}

    def __getitem__(self, seed: int) -> int:
        if seed not in self._cache:
            k = operating_k(model_name(seed, self._size))
            if k is None:
                raise KeyError(
                    f"no activity clustering for seed {seed} at "
                    f"{ACTIVITY_ROOT / model_name(seed, self._size) / ACTIVITY_NAME}. "
                    f"Run: python -m flexsym.training.cluster_epoch_variance ..."
                )
            self._cache[seed] = k
        return self._cache[seed]

    def __iter__(self) -> Iterator[int]:
        # only seeds that actually have a clustering, so callers can iterate safely
        return iter(s for s in self._seeds
                    if operating_k(model_name(s, self._size)) is not None)

    def __len__(self) -> int:
        return sum(1 for _ in iter(self))

    def __repr__(self) -> str:
        return f"OPK({dict(self)})"


OPK = _OperatingK()
