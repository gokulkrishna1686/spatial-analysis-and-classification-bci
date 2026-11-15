"""§07 Feature extraction."""
import numpy as np

import features as F


def test_union_shapes_match_names(epochs):
    import classify as C
    X, y, _, times = C.get_Xy(epochs, (8, 30), (0.5, 2.5))
    fp = {"bands": [[8, 12], [12, 30]], "ar_order": 3, "erp_windows": [[0.6, 0.8]]}
    kinds = ["logvar", "bandpower", "hjorth", "entropy", "ar", "erp", "tangent"]
    M = F.make_feature_union(kinds, epochs.info["sfreq"], times, fp).fit_transform(X, y)
    names = F.feature_names(kinds, epochs.ch_names, fp)
    assert M.shape == (len(y), len(names))
    assert np.isfinite(M).all()


def test_bandpower_finds_the_right_band():
    sf = 250.0
    t = np.arange(500) / sf
    X = np.sin(2 * np.pi * 10 * t)[None, None] * np.ones((2, 1, 1))
    bp = F.BandPower(sf, ((8, 12), (20, 30))).transform(X)
    assert bp[0, 0] > bp[0, 1] + 5          # log power much larger in 8-12 Hz


def test_hjorth_mobility_of_a_sine():
    sf, f = 250.0, 10.0
    t = np.arange(1000) / sf
    X = np.sin(2 * np.pi * f * t)[None, None]
    act, mob, cplx = F.Hjorth().transform(X)[0]
    assert abs(mob - 2 * np.sin(np.pi * f / sf)) < 0.01      # mobility = mean frequency (discrete)
    assert abs(cplx - 1.0) < 0.05                             # a pure sine has complexity ~1


def test_entropy_white_vs_sine():
    sf = 250.0
    rng = np.random.RandomState(0)
    t = np.arange(1000) / sf
    X = np.stack([np.sin(2 * np.pi * 10 * t), rng.standard_normal(1000)])[:, None]
    e = F.SpectralEntropy(sf).transform(X)
    assert e[1, 0] > e[0, 0] + 0.3
