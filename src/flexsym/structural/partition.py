"""Partitions of units into structural roles, and the role-constant subspace.

Two indicator constructions exist and are not interchangeable:

* ``indicator(labels)`` is compact: one column per label that occurs, in sorted
  label order, so every column is non-empty. It returns the cell sizes too.
* ``labels_to_H(labels, k)`` is fixed width: column ``c`` is label ``c`` for
  ``c < k``, and a label absent from ``labels`` gives an all-zero column.

The signed WL partition (``approx_wl_signed``) is the structural-role partition
used throughout.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.cluster import KMeans


# --------------------------------------------------------------------------- #
# indicator matrices
# --------------------------------------------------------------------------- #
def indicator(labels):
    """Compact indicator H (N, k) and cell sizes (k,), one column per label present."""
    cats, inv = np.unique(labels, return_inverse=True)
    H = np.zeros((len(labels), cats.size))
    H[np.arange(len(labels)), inv] = 1.0
    return H, H.sum(0)


def labels_to_H(labels: np.ndarray, k: int) -> np.ndarray:
    """Fixed-width indicator (N, k); labels outside [0, k) leave their row empty."""
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    H = np.zeros((n, k), dtype=float)
    valid = (labels >= 0) & (labels < k)
    H[np.arange(n)[valid], labels[valid]] = 1.0
    return H


def orthonormal_indicator(H, sizes):
    """S = H D^-1/2, orthonormal columns spanning col(H).

    S is the lift for eigenvectors of the orthonormal-coordinate quotient
    ``quotient_ortho``; eigenvectors of the raw ``quotient`` lift through H.
    """
    return H / np.sqrt(sizes)[None, :]


# --------------------------------------------------------------------------- #
# the role-constant subspace
# --------------------------------------------------------------------------- #
def project(X, H, sizes):
    """P_H X for X of shape (N, m), with P_H = H (H'H)^-1 H', without forming P_H.

    X must be two-dimensional; a 1-D vector broadcasts against the cell sizes.
    """
    return H @ ((H.T @ X) / sizes[:, None])


def pw_constant_projector(labels: np.ndarray, k: int) -> np.ndarray:
    """The N x N projector onto piecewise-constant vectors; empty cells dropped."""
    H = np.zeros((labels.size, k), dtype=float)
    H[np.arange(labels.size), labels] = 1.0
    sizes = H.sum(axis=0)
    keep = sizes > 0
    H = H[:, keep]
    # P = H (H^T H)^{-1} H^T ; H^T H is diagonal(cell sizes)
    return H @ np.diag(1.0 / H.sum(axis=0)) @ H.T


def containment_norm(v, H, sizes):
    """||P_H v||."""
    gs = H.T @ v
    return float(np.sqrt(np.sum(gs ** 2 / sizes)))


# --------------------------------------------------------------------------- #
# label utilities
# --------------------------------------------------------------------------- #
def checked_matmul(A: np.ndarray, B: np.ndarray, name: str) -> np.ndarray:
    """A @ B, raising if the product has any non-finite entry."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
        out = A @ B
    if not np.isfinite(out).all():
        n_bad = int(np.size(out) - np.isfinite(out).sum())
        raise FloatingPointError(f"{name} produced {n_bad} non-finite values.")
    return out


def random_labels_same_sizes(labels: np.ndarray, seed: int | None = None) -> np.ndarray:
    """A uniformly random relabelling with the same cell sizes."""
    rng = np.random.default_rng(seed)
    shuffled = np.array(labels, copy=True)
    rng.shuffle(shuffled)
    return shuffled


def canonicalize_labels(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary cluster ids to 0..n_clusters-1 by first occurrence."""
    mapping: dict[int, int] = {}
    out = np.empty_like(labels, dtype=np.int64)
    next_label = 0
    for i, label in enumerate(labels):
        label_int = int(label)
        if label_int not in mapping:
            mapping[label_int] = next_label
            next_label += 1
        out[i] = mapping[label_int]
    return out


# --------------------------------------------------------------------------- #
# signed WL partition
# --------------------------------------------------------------------------- #
def approx_wl_signed(
    W_pos: np.ndarray,
    W_neg: np.ndarray,
    k: int,
    *,
    n_iter: int = 20,
    seed: int = 0,
    n_init: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate signed colour refinement into k cells.

    Each unit is described by its total positive and total negative input from
    every current cell, [W_pos H, W_neg H]; k-means on that description gives
    the next partition. Iterates to a fixed point or ``n_iter``.
    """
    n = W_pos.shape[0]
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    H = labels_to_H(labels, k)

    for _ in range(n_iter):
        X = np.concatenate(
            [
                checked_matmul(W_pos, H, "W_pos @ H"),
                checked_matmul(W_neg, H, "W_neg @ H"),
            ],
            axis=1,
        )
        labels_new = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=seed,
        ).fit_predict(X)
        labels_new = canonicalize_labels(labels_new)

        if np.array_equal(labels_new, labels):
            break

        labels = labels_new
        H = labels_to_H(labels, k)

    return labels, H
