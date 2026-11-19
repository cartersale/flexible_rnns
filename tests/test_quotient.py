import numpy as np

from flexsym.structural.partition import indicator, labels_to_H, random_labels_same_sizes
from flexsym.structural.quotient import (
    gamma_ep,
    gamma_signed,
    quotient,
    quotient_matrix,
    quotient_ortho,
)


def test_quotient_recovers_planted_Q_and_intertwines(real_planted):
    P = real_planted
    H, sizes = indicator(P.labels)
    Q = quotient(P.W, H, sizes)
    np.testing.assert_allclose(Q, P.Q, atol=1e-12)
    np.testing.assert_allclose(P.W @ H, H @ Q, atol=1e-12)


def test_quotient_ortho_is_similar(real_planted):
    P = real_planted
    H, sizes = indicator(P.labels)
    Q, Qt = quotient(P.W, H, sizes), quotient_ortho(P.W, H, sizes)
    d = np.sqrt(sizes)
    np.testing.assert_allclose(Qt, d[:, None] * Q / d[None, :], atol=1e-12)
    np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(Qt)),
                               np.sort_complex(np.linalg.eigvals(Q)), atol=1e-10)
    S = H / d[None, :]
    np.testing.assert_allclose(P.W @ S, S @ Qt, atol=1e-12)


def test_quotient_matrix_matches_quotient(real_planted):
    P = real_planted
    H, sizes = indicator(P.labels)
    np.testing.assert_allclose(quotient_matrix(P.W, labels_to_H(P.labels, len(sizes))),
                               quotient(P.W, H, sizes), atol=1e-12)


def test_quotient_matrix_empty_cell_gives_zero_row():
    W = np.arange(9.0).reshape(3, 3)
    Q = quotient_matrix(W, labels_to_H(np.array([0, 0, 2]), 3))
    np.testing.assert_array_equal(Q[1], 0.0)


def test_trivial_and_discrete_partitions():
    W = np.random.default_rng(0).standard_normal((6, 6))
    H, sizes = indicator(np.zeros(6, int))
    np.testing.assert_allclose(quotient(W, H, sizes), [[W.sum() / 6]])
    H, sizes = indicator(np.arange(6))
    np.testing.assert_allclose(quotient(W, H, sizes), W)


def test_gamma_zero_iff_equitable(real_planted):
    P = real_planted
    k = len(P.sizes)
    assert gamma_ep(P.W, labels_to_H(P.labels, k)) < 1e-10
    shuffled = random_labels_same_sizes(P.labels, seed=0)
    assert gamma_ep(P.W, labels_to_H(shuffled, k)) > 0.1


def test_gamma_signed_sums_parts(real_planted):
    P = real_planted
    H = labels_to_H(P.labels, len(P.sizes))
    Wp, Wn = np.maximum(P.W, 0), np.maximum(-P.W, 0)
    assert np.isclose(gamma_signed(Wp, Wn, H), gamma_ep(Wp, H) + gamma_ep(Wn, H))
