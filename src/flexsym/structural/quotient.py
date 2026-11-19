"""Quotient operators of a partition, and the equitability error.

In raw role coordinates the quotient is Q = (H'H)^-1 H' M H. For an equitable
partition M H = H Q exactly, so an eigenvector z of Q lifts to units as H z.

In orthonormal coordinates S = H D^-1/2 the same operator is
Q~ = D^-1/2 H' M H D^-1/2 = D^1/2 Q D^-1/2, similar to Q (same spectrum), and
M S = S Q~ for an equitable partition, so eigenvectors of Q~ lift through S.
Lifting an eigenvector of the raw Q through S crosses the two coordinate systems.

``quotient_matrix`` computes the raw quotient with the reciprocal-multiply
arithmetic of the WL code; it is kept separate from ``quotient`` so neither
caller's floating-point results change.
"""

from __future__ import annotations

import numpy as np

from flexsym.structural.partition import checked_matmul


def quotient(M, H, sizes):
    """(H'H)^-1 H' M H : row-averaged block sums, in raw role coordinates.

    For an equitable partition M H = H Q, so an eigenvector z of Q lifts to
    units as H z (not H D^-1/2 z; see ``quotient_ortho``).
    """
    return (H.T @ M @ H) / sizes[:, None]


def quotient_ortho(M, H, sizes):
    """D^-1/2 H' M H D^-1/2 : the same quotient in orthonormal role coordinates.

    With S = H D^-1/2 (S'S = I), an equitable partition gives M S = S Q~, where
    Q~ = D^1/2 Q D^-1/2 is similar to Q. Use this operator whenever the lift to
    unit space is through S.
    """
    r = np.sqrt(sizes)
    return (H.T @ M @ H) / r[:, None] / r[None, :]


def quotient_matrix(W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """D^-1 H' W H with a fixed-width H; empty cells give zero rows."""
    cell_sizes = H.sum(axis=0)
    D_inv_diag = 1.0 / np.maximum(cell_sizes, 1e-12)
    return checked_matmul(checked_matmul(H.T, W, "H.T @ W"), H, "H.T @ W @ H") * D_inv_diag[:, None]


def gamma_ep(W: np.ndarray, H: np.ndarray) -> float:
    """||W H - H Q||_F : zero iff the partition is equitable for W."""
    cell_sizes = H.sum(axis=0)
    D_inv_diag = 1.0 / np.maximum(cell_sizes, 1e-12)
    WH = checked_matmul(W, H, "W @ H")
    cell_means = checked_matmul(H.T, WH, "H.T @ WH") * D_inv_diag[:, None]
    projected = checked_matmul(H, cell_means, "H @ cell_means")
    return float(np.linalg.norm(WH - projected, ord="fro"))


def gamma_signed(W_pos: np.ndarray, W_neg: np.ndarray, H: np.ndarray) -> float:
    """Equitability error of the positive and negative parts, summed."""
    return gamma_ep(W_pos, H) + gamma_ep(W_neg, H)
