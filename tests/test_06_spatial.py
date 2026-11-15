"""§06 Spatial analysis."""
import numpy as np

import classify as C
import core
import spatial as SP


def _xy(epochs):
    return C.get_Xy(epochs, (8, 30), (0.5, 2.5))


def test_csp_recovers_ground_truth_patterns(epochs, objs):
    X, y, _, _ = _xy(epochs)
    csp = SP.CSP(len(epochs.ch_names)).fit(X, y)
    for cls in ("left_hand", "right_hand"):
        t = np.array([objs["true_patterns"][cls][core.SYN_CHANNELS.index(n)] for n in epochs.ch_names])
        t = t - t.mean()                       # pipeline uses the average reference
        _, r = SP.match_pattern(t, csp.patterns_[:4])
        assert r > 0.9, (cls, r)


def test_csp_physiology_and_ordering(epochs):
    X, y, _, _ = _xy(epochs)
    csp = SP.CSP(6).fit(X, y)
    peaks = [epochs.ch_names[int(np.argmax(np.abs(csp.patterns_[k])))] for k in range(2)]
    assert set(peaks) == {"C3", "C4"}
    assert csp.eigenvalues_[0] > 0.5 > csp.eigenvalues_[1]      # extremes first


def test_filters_and_patterns_are_inverse(epochs):
    X, y, _, _ = _xy(epochs)
    csp = SP.CSP(4).fit(X, y)
    assert np.allclose(csp.filters_ @ csp.patterns_.T, np.eye(len(epochs.ch_names)), atol=1e-6)


def test_car_laplacian_whitening():
    rng = np.random.RandomState(0)
    X = rng.standard_normal((5, 8, 300))
    assert np.allclose(SP.car(X).mean(axis=1), 0)
    xyz = core.montage_xyz(["F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz"])
    L = SP.laplacian_matrix(xyz, 4)
    assert np.allclose(L.sum(1), 0)
    w, V, Cm = SP.pca_filters(X)
    W = SP.whitening_matrix(Cm)
    assert np.allclose(W @ Cm @ W.T, np.eye(8), atol=1e-8)


def test_riemann_distance_plane_separates(epochs):
    from pyriemann.estimation import Covariances
    X, y, _, _ = _xy(epochs)
    Cv = Covariances("oas").fit_transform(X)
    cls, M = SP.riemann_means(Cv, y)
    D = SP.riemann_distances(Cv, M)
    assert np.mean(cls[np.argmin(D, 1)] == y) > 0.7
