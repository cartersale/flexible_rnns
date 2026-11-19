"""Gain-dependent operators and their sensitivity to gain.

The cell is h+ = (1-a) h + a f(W h + ...), so its Jacobian at a state with gains
g = f'(u) is J = (1-a) I + a diag(g) W. The same form on the structural quotient
Q_s with role-mean gains g_bar gives the predicted quotient
Q = (1-a) I + a diag(g_bar) Q_s. If g is constant within roles and the partition
is equitable for W, the quotient of J is exactly that Q.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

TOL = 1e-12
EPS = 1e-5              # finite-difference gain step in rotation_sensitivity
W_THRESH = 1e-3         # |W| above which the row-ratio gain recovery is used


# --------------------------------------------------------------------------- #
# operators
# --------------------------------------------------------------------------- #
def gain_operator(M, g, alpha):
    """(1 - alpha) I + alpha diag(g) M.

    With M = W and unit gains g this is the full Jacobian; with M a raw quotient
    and role-mean gains it is the predicted quotient.
    """
    return (1 - alpha) * np.eye(len(g)) + alpha * (g[:, None] * M)


def role_gain(g, H, sizes):
    """Role-mean gains g_bar = (H'H)^-1 H' g."""
    return (H.T @ g) / sizes


def recover_alpha_gain(Jb, W):
    """Exact (alpha, g) from J = (1-alpha) I + alpha diag(g) W.

    r_i = alpha * g_i is the constant row ratio J[i,j]/W[i,j] over off-diagonal
    entries with |W| above threshold; alpha then follows from the diagonal.
    """
    n = W.shape[0]
    off = ~np.eye(n, dtype=bool)
    use = off & (np.abs(W) > W_THRESH)
    with np.errstate(all="ignore"):
        num = np.where(use, Jb * W, 0.0).sum(1)
        den = np.where(use, W * W, 0.0).sum(1)
    r = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)     # least squares
    one_minus_a = np.median(np.diag(Jb) - r * np.diag(W))
    alpha = float(np.clip(1.0 - one_minus_a, 1e-6, 1.0))
    return alpha, r / alpha


# --------------------------------------------------------------------------- #
# eigenvector bookkeeping
# --------------------------------------------------------------------------- #
def unit_cols(V):
    """Columns scaled to unit norm."""
    return V / np.maximum(np.linalg.norm(V, axis=0), TOL)


def track(Vb, Vp):
    """Match perturbed modes to base modes by |eigenvector overlap|."""
    r, c = linear_sum_assignment(-np.abs(Vb.conj().T @ Vp))
    return c[np.argsort(r)]


def match_modes(V1, V2):
    """Greedy-optimal mode correspondence by |right-eigenvector overlap|."""
    A = np.abs(np.real(V1).T @ np.real(V2))
    A /= np.maximum(np.linalg.norm(np.real(V1), axis=0)[:, None]
                    * np.linalg.norm(np.real(V2), axis=0)[None, :], TOL)
    r, c = linear_sum_assignment(-A)
    return r, c, A[r, c]


# --------------------------------------------------------------------------- #
# sensitivity of the predicted quotient to role gain
# --------------------------------------------------------------------------- #
def susceptibility(Qs, gbar, alpha):
    """d|lambda_j| / d g_i at the operating point gbar.

    Returns (Ssens (k roles x k modes, real), lam (k,), Vr (right eigvecs)).
    """
    k = len(gbar)
    lam, Vr = np.linalg.eig(gain_operator(Qs, gbar, alpha))
    Vl = np.linalg.inv(Vr).conj().T          # left eigenvectors, u_j' v_j = 1
    QsV = Qs @ Vr
    S = np.zeros((k, k))
    for j in range(k):
        dl = alpha * np.conj(Vl[:, j]) * QsV[:, j]          # d lambda_j / d g_i
        if abs(lam[j]) > TOL:
            S[:, j] = np.real(np.conj(lam[j]) / abs(lam[j]) * dl)   # d|lambda|
        else:
            S[:, j] = np.real(dl)
    return S, lam, Vr


def rotation_sensitivity(Qs, g, alpha, eps=EPS):
    """Finite-difference ||dv_j/dg_i|| and d|lambda_j|/dg_i, mode-matched.

    Returns (Rot (k,k), dLam_fd (k,k), lam, V).
    """
    k = len(g)
    lam, V = np.linalg.eig(gain_operator(Qs, g, alpha))
    V = unit_cols(V)
    Rot = np.zeros((k, k))
    dLam = np.zeros((k, k))
    for i in range(k):
        gp = g.copy()
        gp[i] += eps
        lp, Vp = np.linalg.eig(gain_operator(Qs, gp, alpha))
        Vp = unit_cols(Vp)
        A = np.abs(V.conj().T @ Vp)
        r, c = linear_sum_assignment(-A)
        for j, cc in zip(r, c):
            v0, v1 = V[:, j], Vp[:, cc]
            ph = np.vdot(v0, v1)
            if abs(ph) > TOL:
                v1 = v1 * np.conj(ph) / abs(ph)      # phase/sign alignment
            Rot[i, j] = np.linalg.norm(v1 - v0) / eps
            dLam[i, j] = (abs(lp[cc]) - abs(lam[j])) / eps
    return Rot, dLam, lam, V
