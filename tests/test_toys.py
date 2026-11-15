"""Concept-demo toy models (toys.py)."""
import numpy as np

import classify as C
import spatial as SP
import toys as T


def test_spin_filter_optimum_is_the_csp_eigenvector():
    XA, XB, CA, CB = T.two_class_2d(n=2000, erd=0.6, mix=0.3, shared=0.8, seed=1)
    thetas = np.linspace(0, np.pi, 3601)
    _, _, lam = T.projected_variance(CA, CB, thetas)
    best = thetas[np.argmax(lam)]
    lam_csp, _, ang = T.csp_2d(CA, CB)
    d = abs(best - ang[0]) % np.pi
    assert min(d, np.pi - d) < np.deg2rad(0.2)
    assert abs(lam.max() - lam_csp[0]) < 1e-4


def test_whitening_makes_pooled_covariance_identity_and_rotation_diagonalises():
    _, _, CA, CB = T.two_class_2d(seed=2)
    P, U, angle = T.whiten_rotate(CA, CB)
    assert np.allclose(P @ (CA + CB) @ P.T, np.eye(2), atol=1e-10)
    D = U.T @ P @ CA @ P.T @ U
    assert abs(D[0, 1]) < 1e-10
    assert np.allclose(T.whiten_rotate_transform(P, angle, 0), np.eye(2))
    assert np.allclose(T.whiten_rotate_transform(P, angle, 1), P)


def test_toy_csp_agrees_with_app_csp():
    XA, XB, CA, CB = T.two_class_2d(n=500, seed=3)
    trials = np.concatenate([XA.reshape(10, 50, 2), XB.reshape(10, 50, 2)]).transpose(0, 2, 1)
    y = np.r_[np.zeros(10), np.ones(10)]
    csp = SP.CSP(2, cov="empirical").fit(trials, y)
    _, filt, _ = T.csp_2d(CA, CB)
    cos = abs(filt[0] @ csp.filters_[0]) / np.linalg.norm(csp.filters_[0])
    assert cos > 0.97


def test_haufe_filter_vs_pattern():
    w, a, _, _ = T.haufe_demo(noise=1.5, n=20000, seed=0)
    assert abs(w[1] / w[0] + 1) < 0.05            # filter subtracts the noise-only channel
    assert abs(a[1]) < 0.05 and abs(a[0]) > 0.95   # pattern: the signal lives on channel 1


def test_riemannian_mean_does_not_swell():
    A = np.array([[4.0, 0.0], [0.0, 0.25]])
    B = np.array([[0.25, 0.0], [0.0, 4.0]])
    euc, geo = T.riemann_path(A, B, [0.5])
    assert np.linalg.det(geo[0]) < np.linalg.det(euc[0])
    assert abs(np.linalg.det(geo[0]) - np.sqrt(np.linalg.det(A) * np.linalg.det(B))) < 1e-9


def test_binomial_null_matches_chance_levels():
    for n, k in [(40, 2), (100, 4), (288, 2)]:
        _, pmf, thr, sim = T.binomial_null(n, k, 0.05, n_sim=4000, seed=0)
        assert abs(thr - C.chance_levels(n, k, 0.05)[1]) < 1e-12
        assert abs(pmf.sum() - 1) < 1e-9
        assert abs((sim > thr).mean() - 0.05) < 0.03


def test_lda_toy_threshold_and_roc():
    X, y, w, s = T.lda_toy(sep=3.0, spread=1.0, n=400, seed=0)
    cm, tpr, fpr, acc = T.threshold_stats(s, y, 0.0)
    assert cm.sum() == len(y) and acc > 0.85
    _, _, auc = T.roc(s, y)
    assert auc > 0.9
    cm_hi, tpr_hi, fpr_hi, _ = T.threshold_stats(s, y, 5.0)
    assert tpr_hi <= tpr and fpr_hi <= fpr        # raising the threshold trades hits for fewer false alarms


def test_knn_memorises_at_k1():
    X, y = T.moons_toy(n=200, noise=0.35, seed=0)
    tr, te, _ = T.knn_complexity_curve(X, y, [1, 15, 99], seed=0)
    assert tr[0] == 1.0 and te[0] < 1.0
    assert te[1] > te[2]                          # very large k underfits


def test_leakage_toy():
    honest, leaky = T.leakage_toy(n_trials=60, n_feat=500, k=10, n_rep=8, seed=0)
    assert leaky.mean() > honest.mean() + 0.15
    assert abs(honest.mean() - 0.5) < 0.1
