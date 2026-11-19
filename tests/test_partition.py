import numpy as np
from sklearn.metrics import adjusted_rand_score

from flexsym.structural.partition import (
    approx_wl_signed,
    canonicalize_labels,
    containment_norm,
    indicator,
    labels_to_H,
    orthonormal_indicator,
    project,
    pw_constant_projector,
    random_labels_same_sizes,
)


def test_indicator_noncontiguous_labels():
    H, sizes = indicator(np.array([5, 2, 5, 9]))
    np.testing.assert_array_equal(H, [[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(sizes, [1, 2, 1])


def test_labels_to_H_keeps_empty_and_drops_out_of_range():
    H = labels_to_H(np.array([0, 0, 2, 3]), 3)
    np.testing.assert_array_equal(H, [[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]])


def test_indicators_agree_when_every_label_is_present(real_planted):
    H, _ = indicator(real_planted.labels)
    np.testing.assert_array_equal(H, labels_to_H(real_planted.labels, len(real_planted.sizes)))


def test_orthonormal_indicator(real_planted):
    H, sizes = indicator(real_planted.labels)
    S = orthonormal_indicator(H, sizes)
    np.testing.assert_allclose(S.T @ S, np.eye(len(sizes)), atol=1e-14)
    np.testing.assert_allclose(S @ S.T, pw_constant_projector(real_planted.labels, len(sizes)),
                               atol=1e-14)


def test_project_is_the_piecewise_constant_projector(real_planted):
    H, sizes = indicator(real_planted.labels)
    P = pw_constant_projector(real_planted.labels, len(sizes))
    X = np.random.default_rng(0).standard_normal((len(real_planted.labels), 3))
    PX = project(X, H, sizes)
    np.testing.assert_allclose(PX, P @ X, atol=1e-13)
    np.testing.assert_allclose(project(PX, H, sizes), PX, atol=1e-13)    # idempotent
    for c in range(len(sizes)):                                           # cell means
        m = real_planted.labels == c
        np.testing.assert_allclose(PX[m], np.broadcast_to(X[m].mean(0), X[m].shape), atol=1e-13)


def test_pw_constant_projector_drops_empty_cells():
    P = pw_constant_projector(np.array([0, 0, 2]), 4)
    np.testing.assert_allclose(P, [[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 1]])


def test_containment_norm(real_planted):
    H, sizes = indicator(real_planted.labels)
    v = np.random.default_rng(1).standard_normal(len(real_planted.labels))
    assert np.isclose(containment_norm(v, H, sizes),
                      np.linalg.norm(project(v[:, None], H, sizes)))
    assert np.isclose(containment_norm(H @ [1.0, -2.0, 0.5, 3.0], H, sizes),
                      np.linalg.norm(H @ [1.0, -2.0, 0.5, 3.0]))


def test_random_labels_same_sizes(real_planted):
    lab = real_planted.labels
    r = random_labels_same_sizes(lab, seed=7)
    np.testing.assert_array_equal(np.bincount(r), np.bincount(lab))
    np.testing.assert_array_equal(r, random_labels_same_sizes(lab, seed=7))
    assert not np.array_equal(r, lab)


def test_canonicalize_labels():
    np.testing.assert_array_equal(canonicalize_labels(np.array([7, 7, 3, 9, 3])), [0, 0, 1, 2, 1])


def test_wl_recovers_planted_equitable_partition(positive_planted):
    P = positive_planted
    k = len(P.sizes)
    labels, H = approx_wl_signed(P.W, np.zeros_like(P.W), k, seed=0)
    assert adjusted_rand_score(P.labels, labels) == 1.0
    np.testing.assert_array_equal(H, labels_to_H(labels, k))
