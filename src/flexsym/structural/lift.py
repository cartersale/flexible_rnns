"""Eigenspaces of quotient and full operators, and their lift to unit space.

An eigenvector z of the raw quotient Q (``quotient.quotient``) lifts as H z; an
eigenvector of the orthonormal-coordinate quotient Q~ lifts as S z with
S = H D^-1/2. Quotients are non-normal, so lifted directions are orthonormalised
by QR in unit space, after the lift.

A complex-conjugate eigenpair is represented by the real and imaginary parts of
one member, i.e. its two-dimensional real invariant plane.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import schur


# --------------------------------------------------------------------------- #
# mode selection and the lifted basis
# --------------------------------------------------------------------------- #
def select_modes(lam, rule, r, lam_J=None):
    """Indices of r eigenvalues chosen by ``rule``.

    top: largest |lambda|; unit: nearest 1; unstable: largest Re(lambda);
    match: nearest ``lam_J``.
    """
    if rule == "top":
        key = -np.abs(lam)
    elif rule == "unit":
        key = np.abs(lam - 1.0)
    elif rule == "unstable":
        key = -np.real(lam)
    elif rule == "match":                       # closest to the motif's own lambda
        key = np.abs(lam - lam_J)
    else:
        raise ValueError(rule)
    return np.argsort(key)[:r]


def lifted_basis(Q, H, m, rule="top", lam_J=None, eig=None):
    """Orthonormal unit-space basis of m selected eigen-directions of Q.

    Q is a raw-coordinate quotient lifted through H, or, with ``H=None``, a full
    N x N operator used as is. Selected eigenvectors contribute their real parts
    and, if any is complex, their imaginary parts; the stack is QR-orthonormalised
    and rank-deficient columns are dropped. ``eig`` is an optional precomputed
    ``np.linalg.eig(Q)``.

    Returns (B (N, r) with r <= 2m, selected eigenvalues).

    Known defect, kept so results are unchanged: if both members of a conjugate
    pair are selected, the stack is [a, a, b, -b] and the rank-deficient Householder
    column leaves an arbitrary direction, so B need not span the pair's plane.
    """
    lam, V = np.linalg.eig(Q) if eig is None else eig
    idx = select_modes(lam, rule, m, lam_J)
    Vr, Vi = np.real(V[:, idx]), np.imag(V[:, idx])
    U = Vr if H is None else H @ Vr
    if np.any(np.abs(Vi) > 1e-12):
        U = np.hstack([U, Vi if H is None else H @ Vi])
    B, R = np.linalg.qr(U)
    return B[:, np.abs(np.diag(R)) > 1e-10], lam[idx]


# --------------------------------------------------------------------------- #
# full spectral bases of the quotient
# --------------------------------------------------------------------------- #
def eig_real_blocks(Q):
    """Real invariant directions of Q in raw role coordinates, ordered by |lambda|.

    Each real eigenvalue contributes one column; each complex-conjugate pair
    contributes the real and imaginary parts of one member. The columns are not
    orthonormal (Q is non-normal); orthonormalisation happens after the lift.
    """
    lam, V = np.linalg.eig(Q)
    order = np.argsort(-np.abs(lam))
    cols, used = [], set()
    for i in order:
        if i in used:
            continue
        if abs(lam[i].imag) < 1e-9:
            cols.append(np.real(V[:, i]))
        else:
            cols.append(np.real(V[:, i])); cols.append(np.imag(V[:, i]))
            j = np.argmin(np.abs(lam - np.conj(lam[i])))     # its conjugate partner
            used.add(j)
        used.add(i)
    return np.column_stack(cols)


def lifted_eigbasis(Q, H):
    """Orthonormal unit-space basis from all eigen-directions of the raw quotient Q.

    The real invariant directions of Q (``eig_real_blocks``) are lifted through H
    and orthonormalised by QR, keeping |lambda| order.
    """
    B, _ = np.linalg.qr(H @ eig_real_blocks(Q))
    return B[:, :Q.shape[0]]


def lifted_real_modes(Q, H):
    """Real parts of all eigenvectors of the raw quotient Q, lifted through H.

    Orthonormalised by QR in |lambda| order. Unlike ``lifted_eigbasis``, a
    complex pair contributes only the real parts of its two members.

    Returns (M (N, k), |lambda| sorted descending).
    """
    lam, V = np.linalg.eig(Q)
    order = np.argsort(-np.abs(lam))
    M = H @ np.real(V[:, order])                          # W H = H Q: eigen-directions lift via H
    M, _ = np.linalg.qr(M)                                # orthonormalise, keeps |lambda| order
    return M, np.abs(lam)[order]


def schur_basis(Q):
    """Real Schur vectors, reordered so leading columns carry the largest |lambda|."""
    T, Z = schur(Q, output="real")
    # eigenvalues from 1x1 / 2x2 diagonal blocks
    n = Q.shape[0]; i = 0; blocks = []
    while i < n:
        if i + 1 < n and abs(T[i + 1, i]) > 1e-9:
            ev = np.linalg.eigvals(T[i:i + 2, i:i + 2]); blocks.append((i, 2, max(np.abs(ev)))); i += 2
        else:
            blocks.append((i, 1, abs(T[i, i]))); i += 1
    cols = []
    for start, size, _ in sorted(blocks, key=lambda b: -b[2]):
        cols.extend(range(start, start + size))
    return Z[:, cols]


def lift_ortho(B, S):
    """Lift a basis in orthonormal role coordinates to units through S = H D^-1/2.

    Only correct for directions of the orthonormal-coordinate quotient Q~.
    """
    return S @ B


# --------------------------------------------------------------------------- #
# leading real spectral blocks of a full operator
# --------------------------------------------------------------------------- #
def leading_blocks(M: np.ndarray, n: int) -> list[np.ndarray]:
    """The leading real spectral blocks of M, as orthonormal bases, until n dimensions.

    Eigenvalues are taken in order of decreasing |lambda|. A real eigenvalue gives a
    one-column block, its unit eigenvector; a complex-conjugate pair gives a two-column
    orthonormal basis of its real invariant plane, and the partner is not visited again.
    A block is never split, so the last one may carry the total past n.

    Blocks rather than individual vectors because Re(z) and Im(z) of a complex
    eigenvector depend on the phase the eigensolver happens to return, while the plane
    they span does not.
    """
    w, V = np.linalg.eig(M)
    order = np.argsort(-np.abs(w))
    blocks, used, dims = [], set(), 0
    for i in order:
        if dims >= n:
            break
        if i in used:
            continue
        used.add(i)
        if abs(w[i].imag) < 1e-9:
            v = np.real(V[:, i])
            nv = np.linalg.norm(v)
            if not nv:
                continue
            B = (v / nv)[:, None]
        else:
            used.add(int(np.argmin(np.abs(w - np.conj(w[i])))))   # conjugate partner
            B, _ = np.linalg.qr(np.column_stack([np.real(V[:, i]), np.imag(V[:, i])]))
        blocks.append(B)
        dims += B.shape[1]
    return blocks


def block_prefix(blocks: list[np.ndarray], n: int) -> list[np.ndarray]:
    """The leading blocks covering at least n real dimensions, never splitting one."""
    out, dims = [], 0
    for B in blocks:
        if dims >= n:
            break
        out.append(B)
        dims += B.shape[1]
    return out


def leading_eigvecs(M: np.ndarray, n: int) -> np.ndarray:
    """The columns of the leading real spectral blocks, unit-normalised, as rows.

    For display and for code that wants individual directions. A single column of
    a complex block turns with the eigenvector's phase.
    """
    out = []
    for B in leading_blocks(M, n):
        out.extend(B.T)
    return np.asarray(out[:n])
