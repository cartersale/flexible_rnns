"""Planted equitable matrices with known quotients and spectra.

W = H Q D^-1 H' + E, where E has zero sum over every block row (E H = 0). Then
W H = H Q exactly, while W itself is not block-constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

SIZES = (3, 5, 7, 9)          # unequal on purpose: S = H D^-1/2 differs from H


@dataclass
class Planted:
    W: np.ndarray
    Q: np.ndarray
    labels: np.ndarray
    H: np.ndarray
    sizes: np.ndarray


def rotation_scaling(r: float, theta: float) -> np.ndarray:
    """2 x 2 block with eigenvalues r e^{+-i theta}."""
    c, s = np.cos(theta), np.sin(theta)
    return r * np.array([[c, -s], [s, c]])


def plant(Q: np.ndarray, sizes=SIZES, noise: float = 0.3, seed: int = 0) -> Planted:
    rng = np.random.default_rng(seed)
    labels = rng.permutation(np.repeat(np.arange(len(sizes)), sizes))
    n, k = labels.size, len(sizes)
    H = np.zeros((n, k))
    H[np.arange(n), labels] = 1.0
    sz = H.sum(0)
    W0 = H @ (Q / sz[None, :]) @ H.T
    R = rng.standard_normal((n, n))
    E = R - (R @ H / sz[None, :]) @ H.T          # subtract each block-row mean
    return Planted(W=W0 + noise * E, Q=Q, labels=labels, H=H, sizes=sz)


@pytest.fixture
def real_planted() -> Planted:
    """Distinct real quotient eigenvalues."""
    rng = np.random.default_rng(1)
    V = rng.standard_normal((4, 4))
    Q = V @ np.diag([1.4, -0.9, 0.6, 0.25]) @ np.linalg.inv(V)
    return plant(Q)


@pytest.fixture
def complex_planted() -> Planted:
    """Quotient with a leading complex pair 1.2 e^{+-i 0.7} and two real eigenvalues."""
    rng = np.random.default_rng(2)
    V = rng.standard_normal((4, 4))
    B = np.zeros((4, 4))
    B[:2, :2] = rotation_scaling(1.2, 0.7)
    B[2, 2], B[3, 3] = 0.5, -0.2
    Q = V @ B @ np.linalg.inv(V)
    return plant(Q, seed=3)


@pytest.fixture
def positive_planted() -> Planted:
    """Entrywise-positive W with well-separated quotient rows, for the WL partition."""
    Q = np.array([[8.0, 1.0, 1.0, 1.0],
                  [1.0, 1.0, 8.0, 1.0],
                  [1.0, 8.0, 1.0, 8.0],
                  [8.0, 8.0, 1.0, 1.0]])
    P = plant(Q, noise=0.02, seed=4)
    assert (P.W > 0).all()
    return P
