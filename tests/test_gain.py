import numpy as np

from flexsym.structural.gain import (
    gain_operator,
    match_modes,
    recover_alpha_gain,
    role_gain,
    rotation_sensitivity,
    susceptibility,
    track,
    unit_cols,
)
from flexsym.structural.partition import indicator
from flexsym.structural.quotient import quotient

ALPHA = 0.2


def test_gain_operator_form():
    M = np.arange(9.0).reshape(3, 3)
    g = np.array([1.0, 2.0, 0.5])
    np.testing.assert_allclose(gain_operator(M, g, ALPHA),
                               (1 - ALPHA) * np.eye(3) + ALPHA * np.diag(g) @ M)
    np.testing.assert_allclose(gain_operator(M, np.ones(3), 1.0), M)


def test_role_gain_of_piecewise_constant_gains(real_planted):
    H, sizes = indicator(real_planted.labels)
    gv = np.array([0.4, 1.3, 0.9, 2.0])
    np.testing.assert_allclose(role_gain(H @ gv, H, sizes), gv)


def test_quotient_of_jacobian_is_predicted_quotient(complex_planted):
    """g constant within roles + equitable W  =>  quot(J(W, g)) = J(Q, g_bar), exactly."""
    P = complex_planted
    H, sizes = indicator(P.labels)
    gv = np.array([0.4, 1.3, 0.9, 2.0])
    g = H @ gv
    J = gain_operator(P.W, g, ALPHA)
    Qs = quotient(P.W, H, sizes)
    Qp = gain_operator(Qs, role_gain(g, H, sizes), ALPHA)
    np.testing.assert_allclose(quotient(J, H, sizes), Qp, atol=1e-12)
    np.testing.assert_allclose(J @ H, H @ Qp, atol=1e-12)
    lam, V = np.linalg.eig(Qp)
    for j in range(len(lam)):
        u = H @ V[:, j]
        np.testing.assert_allclose(J @ u, lam[j] * u, atol=1e-10)


def test_recover_alpha_gain_exact():
    rng = np.random.default_rng(0)
    n = 30
    W = rng.uniform(0.1, 1.0, (n, n)) * rng.choice([-1, 1], (n, n))
    g = rng.uniform(0.2, 2.0, n)
    alpha, g_hat = recover_alpha_gain(gain_operator(W, g, ALPHA), W)
    assert np.isclose(alpha, ALPHA, atol=1e-12)
    np.testing.assert_allclose(g_hat, g, atol=1e-10)


def test_unit_cols():
    V = np.array([[3.0, 0.0], [4.0, 0.0]])
    np.testing.assert_allclose(unit_cols(V), [[0.6, 0.0], [0.8, 0.0]])


def test_track_inverts_a_permutation():
    rng = np.random.default_rng(1)
    Vb = unit_cols(rng.standard_normal((8, 5)) + 1j * rng.standard_normal((8, 5)))
    perm = rng.permutation(5)
    m = track(Vb, Vb[:, perm])
    np.testing.assert_array_equal(perm[m], np.arange(5))


def test_match_modes_recovers_permutation():
    rng = np.random.default_rng(2)
    V1 = np.linalg.qr(rng.standard_normal((8, 5)))[0]
    perm = rng.permutation(5)
    r, c, ov = match_modes(V1, -V1[:, perm])          # sign does not matter
    np.testing.assert_array_equal(perm[c], r)
    np.testing.assert_allclose(ov, 1.0)


def test_susceptibility_diagonal_closed_form():
    """Qs = diag(q): lambda_j = 1-a + a g_j q_j, so d|lambda_j|/dg_i = delta_ij a q_j sign(lambda_j)."""
    q = np.array([2.0, -3.0, 0.5, -0.8])
    g = np.array([1.0, 0.7, 1.5, 3.0])
    S, lam, _ = susceptibility(np.diag(q), g, ALPHA)
    np.testing.assert_allclose(lam, 1 - ALPHA + ALPHA * g * q)
    np.testing.assert_allclose(S, np.diag(ALPHA * q * np.sign(lam)), atol=1e-14)


def test_rotation_sensitivity_diagonal():
    q = np.array([2.0, -3.0, 0.5, -0.8])
    g = np.array([1.0, 0.7, 1.5, 3.0])
    S, _, _ = susceptibility(np.diag(q), g, ALPHA)
    Rot, dLam, _, _ = rotation_sensitivity(np.diag(q), g, ALPHA)
    np.testing.assert_allclose(Rot, 0.0, atol=1e-12)          # eigenvectors do not move
    np.testing.assert_allclose(dLam, S, atol=1e-8)            # |lambda| is linear here


def test_susceptibility_agrees_with_finite_difference(complex_planted):
    P = complex_planted
    H, sizes = indicator(P.labels)
    Qs = quotient(P.W, H, sizes)
    g = np.array([0.4, 1.3, 0.9, 2.0])
    S, lam, _ = susceptibility(Qs, g, ALPHA)
    _, dLam, lam2, _ = rotation_sensitivity(Qs, g, ALPHA)
    np.testing.assert_array_equal(lam, lam2)
    np.testing.assert_allclose(dLam, S, atol=1e-4)
