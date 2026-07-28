"""Turn a connectome edge list into a weight matrix the quotient code can eat.

The RNN side works on a dense W_rec where W[i, j] is the weight from j to i.
This module produces the same convention from connectome edge lists so that
`gamma_ep`, `gamma_signed`, `quotient_matrix` and the lumpability projectors
apply unchanged.

Weight scaling is left explicit rather than baked in: raw synapse counts are
heavily long-tailed, and since our null models are degree-sensitive, the choice
between raw counts, log counts, and row-normalised input fractions is a real
analysis decision, not a formatting one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_matrix(
    ids: np.ndarray | pd.Series,
    edges: pd.DataFrame,
    *,
    pre_col: str,
    post_col: str,
    weight_col: str = "weight",
    scaling: str = "raw",
) -> np.ndarray:
    """Return dense W with ``W[i, j]`` = weight from ``ids[j]`` to ``ids[i]``.

    scaling:
        "raw"      synapse counts as-is
        "log1p"    log(1 + count), compressing the long tail
        "infrac"   each row scaled to sum to 1 (fraction of a neuron's input);
                   uses absolute values so signed graphs normalise sensibly
    """
    ids = np.asarray(ids)
    index = {int(v): i for i, v in enumerate(ids)}
    n = len(ids)
    W = np.zeros((n, n), dtype=float)

    pre = edges[pre_col].astype("int64").to_numpy()
    post = edges[post_col].astype("int64").to_numpy()
    wt = edges[weight_col].astype(float).to_numpy()

    dropped = 0
    for p, q, w in zip(pre, post, wt):
        i, j = index.get(int(q)), index.get(int(p))  # row = target, col = source
        if i is None or j is None:
            dropped += 1
            continue
        W[i, j] += w

    if dropped:
        print(f"  build_matrix: dropped {dropped} edges with endpoints outside the neuron set")

    if scaling == "raw":
        return W
    if scaling == "log1p":
        return np.sign(W) * np.log1p(np.abs(W))
    if scaling == "infrac":
        denom = np.abs(W).sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        return W / denom
    raise ValueError(f"unknown scaling {scaling!r}")


def split_sign(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a signed matrix into (W_pos, W_neg), both non-negative.

    This is the input form `gamma_signed` expects.
    """
    return np.clip(W, 0, None), np.clip(-W, 0, None)


def summarise(W: np.ndarray, name: str = "W") -> str:
    nz = W != 0
    density = nz.mean() if W.size else 0.0
    return (
        f"{name}: {W.shape[0]}x{W.shape[1]}  density={density:.3%}  "
        f"weight range [{W.min():.3g}, {W.max():.3g}]  "
        f"mean out-degree={nz.sum(axis=0).mean():.1f}"
    )
