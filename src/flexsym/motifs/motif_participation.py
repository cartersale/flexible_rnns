"""Link per-unit dynamical-motif participation to connectivity role groups.

Consumes the motif classification (motif_classify.py) and the cached partitions
(signed-WL role groups + activity clusters) to run the three tests:

  Test 1 (critical): does role-group membership predict motif participation?
    per motif: eta^2 of p_m ~ role label, z vs size-matched label permutations.

  Test 2 (earns the claim): nested comparison against activity clusters.
    M0: p_m ~ 1 ; M1: p_m ~ activity ; M2: p_m ~ activity + role ; M3: p_m ~ role.
    cross-validated R^2 (hold out units, k-fold) + commonality decomposition
    (unique-to-role, unique-to-activity, shared). Headline: unique-to-role > 0.

  Test 3 (reuse, closest to the thesis): is the same motif type carried by the
    same role groups across tasks? correlate p_m across task instances of the
    same motif type vs across different motif types.

All participation vectors are restricted to the active-unit index set shared by
both partitions (ind_active) before regressing against labels.
"""

from __future__ import annotations

from flexsym import paths
from flexsym.structural.operating_k import operating_k

from pathlib import Path

import numpy as np


def npload(path):
    """np.load robust to task.py's global monkeypatch (force allow_pickle=True)."""
    try:
        return np.load(path, allow_pickle=True)
    except TypeError:
        return np.load(path)


STEP_DIR = Path(__file__).resolve().parent
REPO_DIR = paths.REPO_ROOT
WL_ROOT = REPO_DIR / "results" / "signed_input_wl_paper"
ACT_ROOT = REPO_DIR / "results" / "activity_clusters_paper"
ACT_NAME = "epoch_variance_unit_clusters_paper_max.npz"

# Operating k is derived from each network's activity clustering by
# flexsym.structural.operating_k -- the single source of truth. This module used to
# carry its own copy of the old literal (seed 0 -> 33, seeds 3-9 -> 14/15), which went
# stale when the 256 seeds 3-9 were retrained on 2026-07-25 and was missed when the
# other copies were removed. Do not reintroduce a literal here.


def _row_for_k(labels_by_k, k):
    """Pick the labels_struct_by_k row whose #clusters == k (nearest if absent)."""
    counts = np.array([len(np.unique(labels_by_k[r]))
                       for r in range(labels_by_k.shape[0])])
    exact = np.where(counts == k)[0]
    if exact.size:
        return int(exact[0])
    return int(np.argmin(np.abs(counts - k)))


def load_partitions(net: str, k: int | None = None):
    """Return dict with ind_active, role_labels, act_labels (all len n_active).

    role and activity labels are on the SAME ind_active index set.
    """
    wl = npload(WL_ROOT / net / "transpose" / "signed_input_wl_sweep.npz")
    act = npload(ACT_ROOT / net / ACT_NAME)
    ind_active = np.asarray(act["ind_active"], dtype=int)
    wl_active = np.asarray(wl["ind_active"], dtype=int)
    if not np.array_equal(ind_active, wl_active):
        raise ValueError(f"{net}: WL and activity ind_active differ")

    if k is None:
        k = operating_k(net) or int(wl["chosen_k"])
    labels_by_k = wl["labels_struct_by_k"]
    role_labels = np.asarray(labels_by_k[_row_for_k(labels_by_k, k)], dtype=int)
    act_labels = np.asarray(act["labels_active"], dtype=int)
    return {"ind_active": ind_active, "role_labels": role_labels,
            "act_labels": act_labels, "k": k}


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def eta_squared(values, labels):
    """One-way eta^2 of a scalar per-unit response by categorical group."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    grand = values.mean()
    ss_tot = np.sum((values - grand) ** 2)
    if ss_tot == 0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(labels):
        v = values[labels == g]
        ss_between += v.size * (v.mean() - grand) ** 2
    return float(ss_between / ss_tot)


def perm_z(values, labels, n_perm=1000, rng=None):
    """z-score of eta^2 vs size-matched label permutations."""
    rng = rng or np.random.default_rng(0)
    obs = eta_squared(values, labels)
    null = np.empty(n_perm)
    lab = np.asarray(labels).copy()
    for i in range(n_perm):
        null[i] = eta_squared(values, rng.permutation(lab))
    mu, sd = null.mean(), null.std()
    z = (obs - mu) / sd if sd > 0 else 0.0
    return {"eta2": obs, "z": float(z), "null_mean": float(mu),
            "null_std": float(sd), "p_perm": float((np.sum(null >= obs) + 1)
                                                    / (n_perm + 1))}


def _onehot(labels):
    labels = np.asarray(labels)
    cats = np.unique(labels)
    X = (labels[:, None] == cats[None, :]).astype(float)
    return X


RIDGE_ALPHAS = np.logspace(-2, 4, 13)


def _cv_r2_ridge(X, y, folds, rng, alphas=RIDGE_ALPHAS):
    """Out-of-fold R^2 of ridge y ~ X, alpha chosen by nested CV per fold.

    Ridge regularisation is essential here: the categorical designs have many
    one-hot columns (k ~ 33-45 groups) relative to the ~200 active units, so
    unregularised OLS overfits and the held-out R^2 (and hence the commonality
    decomposition) is dominated by variance rather than signal.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    y = np.asarray(y, dtype=float)
    kf = KFold(n_splits=folds, shuffle=True,
               random_state=int(rng.integers(1_000_000)))
    preds = np.empty_like(y)
    for tr, te in kf.split(X):
        m = RidgeCV(alphas=alphas)          # internal CV on the training fold only
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _step_r2(v, labels):
    """Piecewise-constant (cell-averaged) R^2 of a vector v on a partition.

    Identical to the lumpability step-R^2 used for W_rec eigenvectors, so this
    subspace test is the SAME measurement applied to motif directions.
    """
    v = np.asarray(v, dtype=float)
    pred = np.zeros_like(v)
    for c in np.unique(labels):
        m = labels == c
        pred[m] = v[m].mean()
    denom = ((v - v.mean()) ** 2).sum()
    return 1.0 - ((v - pred) ** 2).sum() / denom if denom > 0 else 0.0


def _subspace_frac(v, labels):
    """||P v|| / ||v|| : fraction of v in the group-constant subspace (incl. mean)."""
    v = np.asarray(v, dtype=float)
    num = 0.0
    for c in np.unique(labels):
        m = labels == c
        num += m.sum() * v[m].mean() ** 2
    den = (v ** 2).sum()
    return float(np.sqrt(num / den)) if den > 0 else 0.0


def subspace_lumpability(vecs, role_labels, n_rand=200, max_vecs=60, rng=None):
    """Do signed motif directions lie in the group-constant (lumpable) subspace?

    Membership-invariant: depends only on the subspace col(H), not the label
    assignment. Compares the mean piecewise-constant R^2 (and ||Pv||/||v|| frac)
    of the motif's signed eigenvectors on the role partition to size-matched
    random partitions -> z. Inherits the robustness of the eigenvector-lumpability
    (R2) result because it is the same measurement on Jacobian eigenvectors.
    """
    from flexsym.structural.signed_input_wl_structural_roles import random_labels_same_sizes

    rng = rng or np.random.default_rng(0)
    V = np.asarray(vecs, dtype=float)
    if V.ndim == 1:
        V = V[None, :]
    if V.shape[0] > max_vecs:
        V = V[rng.choice(V.shape[0], size=max_vecs, replace=False)]
    obs_r2 = float(np.mean([_step_r2(v, role_labels) for v in V]))
    obs_frac = float(np.mean([_subspace_frac(v, role_labels) for v in V]))
    null = np.empty(n_rand)
    for r in range(n_rand):
        rl = random_labels_same_sizes(role_labels, seed=int(rng.integers(1_000_000)))
        null[r] = np.mean([_step_r2(v, rl) for v in V])
    mu, sd = null.mean(), null.std()
    z = (obs_r2 - mu) / sd if sd > 0 else 0.0
    return {"step_r2": obs_r2, "subspace_frac": obs_frac, "z": float(z),
            "null_r2_mean": float(mu), "n_vecs": int(V.shape[0]),
            "p_perm": float((np.sum(null >= obs_r2) + 1) / (n_rand + 1))}


def nested_commonality(values, role_labels, act_labels, folds=5, rng=None):
    """Ridge CV R^2 for M0..M3 and commonality decomposition (role vs activity).

    unique_role = R^2(activity+role) - R^2(activity): motif variance role groups
    explain that the activity partition does NOT. This is the headline number.
    """
    rng = rng or np.random.default_rng(0)
    y = np.asarray(values, dtype=float)
    Xr = _onehot(role_labels)
    Xa = _onehot(act_labels)
    Xra = np.hstack([Xa, Xr])
    r2_role = _cv_r2_ridge(Xr, y, folds, rng)
    r2_act = _cv_r2_ridge(Xa, y, folds, rng)
    r2_both = _cv_r2_ridge(Xra, y, folds, rng)
    unique_role = r2_both - r2_act
    unique_act = r2_both - r2_role
    shared = r2_role + r2_act - r2_both
    return {"M0": 0.0, "M1_activity": r2_act, "M2_both": r2_both,
            "M3_role": r2_role, "unique_role": unique_role,
            "unique_activity": unique_act, "shared": shared}
