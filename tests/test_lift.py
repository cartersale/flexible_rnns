import numpy as np
import pytest

from conftest import rotation_scaling
from flexsym.structural.lift import (
    block_prefix,
    eig_real_blocks,
    leading_blocks,
    leading_eigvecs,
    lift_ortho,
    lifted_basis,
    lifted_eigbasis,
    lifted_real_modes,
    schur_basis,
    select_modes,
)
from flexsym.structural.partition import indicator, orthonormal_indicator
from flexsym.structural.quotient import quotient, quotient_ortho


def invariance_residual(W, B):
    """||(I - B B') W B|| for orthonormal B: zero iff span(B) is W-invariant."""
    WB = W @ B
    return np.linalg.norm(WB - B @ (B.T @ WB))


def test_select_modes_rules():
    lam = np.array([0.3, -1.2, 1.05, 0.9])
    np.testing.assert_array_equal(select_modes(lam, "top", 2), [1, 2])
    np.testing.assert_array_equal(select_modes(lam, "unit", 2), [2, 3])
    np.testing.assert_array_equal(select_modes(lam, "unstable", 2), [2, 3])
    np.testing.assert_array_equal(select_modes(lam, "match", 1, lam_J=0.31), [0])
    with pytest.raises(ValueError):
        select_modes(lam, "bogus", 1)


def test_lifted_eigenvector_is_eigenvector_of_W(real_planted):
    P = real_planted
    H, sizes = indicator(P.labels)
    lam, V = np.linalg.eig(quotient(P.W, H, sizes))
    for j in range(len(lam)):
        u = H @ V[:, j]
        np.testing.assert_allclose(P.W @ u, lam[j] * u, atol=1e-10)


def test_lift_must_match_coordinates(real_planted):
    """Raw-Q eigenvectors lift through H; Q~ eigenvectors lift through S. Crossing fails."""
    P = real_planted
    H, sizes = indicator(P.labels)
    S = orthonormal_indicator(H, sizes)
    lam, V = np.linalg.eig(quotient(P.W, H, sizes))
    lt, Vt = np.linalg.eig(quotient_ortho(P.W, H, sizes))
    j = int(np.argmax(np.abs(lam)))
    jt = int(np.argmax(np.abs(lt)))
    good = S @ Vt[:, jt]
    np.testing.assert_allclose(P.W @ good, lt[jt] * good, atol=1e-10)
    crossed = S @ V[:, j]
    rel = np.linalg.norm(P.W @ crossed - lam[j] * crossed) / np.linalg.norm(crossed)
    assert rel > 1e-2


def test_lifted_basis_real_spectrum(real_planted):
    P = real_planted
    H, sizes = indicator(P.labels)
    Q = quotient(P.W, H, sizes)
    B, lam = lifted_basis(Q, H, 2)
    assert B.shape == (len(P.labels), 2)
    np.testing.assert_allclose(B.T @ B, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(np.sort(np.abs(lam)), [0.9, 1.4], atol=1e-10)
    assert invariance_residual(P.W, B) < 1e-10


def test_lifted_basis_complex_pair_gives_its_plane(complex_planted):
    P = complex_planted
    H, sizes = indicator(P.labels)
    Q = quotient(P.W, H, sizes)
    B1, lam1 = lifted_basis(Q, H, 1)
    assert B1.shape[1] == 2 and np.isclose(abs(lam1[0]), 1.2)
    assert invariance_residual(P.W, B1) < 1e-10


def test_lifted_basis_both_pair_members_give_the_same_plane(complex_planted):
    """Selecting both members of a pair must not add a spurious direction."""
    P = complex_planted
    H, sizes = indicator(P.labels)
    Q = quotient(P.W, H, sizes)
    B1, _ = lifted_basis(Q, H, 1)
    B2, lam2 = lifted_basis(Q, H, 2)
    assert B2.shape[1] == 2
    np.testing.assert_allclose(lam2[0], np.conj(lam2[1]))
    np.testing.assert_allclose(B2 @ B2.T, B1 @ B1.T, atol=1e-10)
    B3, _ = lifted_basis(Q, H, 3)            # pair + the next real mode
    assert B3.shape[1] == 3
    assert invariance_residual(P.W, B3) < 1e-10


def test_lifted_basis_unchanged_without_a_repeated_pair(real_planted, complex_planted):
    """No pair selected twice: identical to stacking every selected eigenvector."""
    for P, m in ((real_planted, 3), (complex_planted, 1)):
        H, sizes = indicator(P.labels)
        Q = quotient(P.W, H, sizes)
        lam, V = np.linalg.eig(Q)
        idx = select_modes(lam, "top", m)
        U = H @ np.real(V[:, idx])
        if np.any(np.abs(np.imag(V[:, idx])) > 1e-12):
            U = np.hstack([U, H @ np.imag(V[:, idx])])
        Bq, R = np.linalg.qr(U)
        np.testing.assert_array_equal(lifted_basis(Q, H, m)[0], Bq[:, np.abs(np.diag(R)) > 1e-10])


def test_lifted_basis_without_lift_matches_identity_lift(complex_planted):
    W = complex_planted.W
    B0, l0 = lifted_basis(W, None, 3)
    Bi, li = lifted_basis(W, np.eye(len(W)), 3)
    np.testing.assert_allclose(B0, Bi, atol=1e-14)
    np.testing.assert_array_equal(l0, li)


def test_lifted_basis_precomputed_eig(real_planted):
    P = real_planted
    H, sizes = indicator(P.labels)
    Q = quotient(P.W, H, sizes)
    a = lifted_basis(Q, H, 2, rule="unit")
    b = lifted_basis(Q, H, 2, rule="unit", eig=np.linalg.eig(Q))
    np.testing.assert_array_equal(a[0], b[0])


def test_eig_real_blocks_visits_each_pair_once(complex_planted):
    P = complex_planted
    H, sizes = indicator(P.labels)
    Q = quotient(P.W, H, sizes)
    C = eig_real_blocks(Q)
    assert C.shape == (4, 4)
    plane = C[:, :2]                          # leading complex pair: a real invariant plane
    coef, *_ = np.linalg.lstsq(plane, Q @ plane, rcond=None)
    np.testing.assert_allclose(plane @ coef, Q @ plane, atol=1e-10)
    np.testing.assert_allclose(np.abs(np.linalg.eigvals(coef)), [1.2, 1.2], atol=1e-10)


def test_lifted_eigbasis(complex_planted):
    P = complex_planted
    H, sizes = indicator(P.labels)
    M = lifted_eigbasis(quotient(P.W, H, sizes), H)
    np.testing.assert_allclose(M.T @ M, np.eye(4), atol=1e-12)
    for d in (2, 3, 4):                       # every leading prefix is W-invariant
        assert invariance_residual(P.W, M[:, :d]) < 1e-10


def test_lifted_real_modes_matches_eigbasis_for_real_spectrum(real_planted):
    P = real_planted
    H, sizes = indicator(P.labels)
    Q = quotient(P.W, H, sizes)
    M, mag = lifted_real_modes(Q, H)
    np.testing.assert_allclose(mag, [1.4, 0.9, 0.6, 0.25], atol=1e-10)
    M2 = lifted_eigbasis(Q, H)
    for d in (1, 2, 3):
        np.testing.assert_allclose(M[:, :d] @ M[:, :d].T, M2[:, :d] @ M2[:, :d].T, atol=1e-10)


def test_schur_basis_lifted_through_S(complex_planted):
    P = complex_planted
    H, sizes = indicator(P.labels)
    S = orthonormal_indicator(H, sizes)
    Z = schur_basis(quotient_ortho(P.W, H, sizes))
    np.testing.assert_allclose(Z.T @ Z, np.eye(4), atol=1e-12)
    M = lift_ortho(Z, S)
    np.testing.assert_allclose(M.T @ M, np.eye(4), atol=1e-12)
    assert invariance_residual(P.W, M[:, :2]) < 1e-10      # leading pair first


def block_spectrum_matrix():
    """Known spectrum: complex pair |lambda| = 2, real 1.5, real -0.5, pair |lambda| = 0.3."""
    rng = np.random.default_rng(5)
    B = np.zeros((6, 6))
    B[:2, :2] = rotation_scaling(2.0, 0.4)
    B[2, 2], B[3, 3] = 1.5, -0.5
    B[4:, 4:] = rotation_scaling(0.3, 1.1)
    V = rng.standard_normal((6, 6))
    return V @ B @ np.linalg.inv(V)


def test_leading_blocks_never_split_a_pair():
    M = block_spectrum_matrix()
    blocks = leading_blocks(M, 6)
    assert [b.shape[1] for b in blocks] == [2, 1, 1, 2]
    for b in blocks:
        np.testing.assert_allclose(b.T @ b, np.eye(b.shape[1]), atol=1e-12)
        coef, *_ = np.linalg.lstsq(b, M @ b, rcond=None)
        np.testing.assert_allclose(b @ coef, M @ b, atol=1e-10)
    assert [b.shape[1] for b in leading_blocks(M, 1)] == [2]
    assert [b.shape[1] for b in block_prefix(blocks, 3)] == [2, 1]
    assert [b.shape[1] for b in block_prefix(blocks, 4)] == [2, 1, 1]


def test_leading_eigvecs():
    V = leading_eigvecs(block_spectrum_matrix(), 3)
    assert V.shape == (3, 6)
    np.testing.assert_allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-12)
