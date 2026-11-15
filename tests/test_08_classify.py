"""§08 Classification — leak-free evaluation, chance levels, statistics."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

import classify as C


def _xy(epochs):
    return C.get_Xy(epochs, (8, 30), (0.5, 2.5))


def test_real_labels_beat_chance(epochs):
    cfg = {"feature": "csp", "clf": "lda", "cv": {"scheme": "kfold", "k": 5}, "n_perm": 30, "band": [8, 30], "window": [0.5, 2.5]}
    r = C.run_classification(epochs, cfg, 42)
    assert r["summary"]["accuracy"][0] > r["binom_threshold"]
    assert r["p_value"] < 0.05


def test_shuffled_labels_give_chance(epochs):
    X, y, meta, _ = _xy(epochs)
    ys = np.random.RandomState(0).permutation(y)
    splits = C.make_splits({"scheme": "kfold", "k": 5}, ys, meta, 0)
    pipe = C.make_pipeline({"feature": "csp", "clf": "lda"}, epochs.info["sfreq"])
    r = C.evaluate(pipe, X, ys, splits)
    null = C.permutation_null(pipe, X, ys, splits, 30, seed=1)
    acc = r["summary"]["accuracy"][0]
    assert abs(acc - 0.5) < 0.12
    assert C.perm_p_value(acc, null) > 0.05


def test_leaky_pipeline_beats_chance_on_noise_but_honest_does_not(epochs):
    X, y, meta, _ = _xy(epochs)
    splits = C.make_splits({"scheme": "kfold", "k": 5}, y, meta, 0)
    d = C.leakage_demo(X, y, splits, epochs.info["sfreq"], n_shuffles=6, k_best=10, seed=0)
    assert d.leaky.mean() > 0.6
    assert abs(d.honest.mean() - 0.5) < 0.08


class _Spy(BaseEstimator, ClassifierMixin):
    """Records which sample indices it was fitted on (the last column is the trial index)."""
    seen = []

    def __init__(self, dummy=1):
        self.dummy = dummy

    def fit(self, X, y):
        _Spy.seen.append(set(X[:, -1].astype(int)))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])


def test_nested_cv_never_touches_the_test_fold():
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    rng = np.random.RandomState(0)
    X = np.c_[rng.standard_normal((60, 3)), np.arange(60)]
    y = np.repeat([0, 1], 30)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        _Spy.seen = []
        GridSearchCV(_Spy(), {"dummy": [1, 2]}, cv=StratifiedKFold(3)).fit(X[tr], y[tr])
        assert len(_Spy.seen) == 2 * 3 + 1               # inner fits + refit
        for s in _Spy.seen:
            assert not (s & set(te)), "a fit saw test-fold samples"


def test_splits_are_disjoint_and_session_aware(epochs):
    X, y, meta, _ = _xy(epochs)
    for tr, te in C.make_splits({"scheme": "kfold", "k": 5}, y, meta, 0):
        assert not set(tr) & set(te)
    for tr, te in C.make_splits({"scheme": "loso_session"}, y, meta, 0):
        assert set(meta.session[tr]).isdisjoint(set(meta.session[te]))
    (tr, te), = C.make_splits({"scheme": "cross_session", "train": 1, "test": 2}, y, meta, 0)
    assert set(meta.session[tr]) == {1} and set(meta.session[te]) == {2}


def test_chance_and_itr_formulas():
    theo, thr = C.chance_levels(40, 2, 0.05)
    assert theo == 0.5 and 0.6 < thr < 0.7                  # Combrisson & Jerbi: ~62.5 % for n = 40
    assert C.itr_bits(1.0, 2, 60.0) == 1.0                  # 1 bit per 60 s trial
    assert C.itr_bits(0.5, 2, 4.0) == 0.0
    assert abs(C.itr_bits(0.8, 2, 1.0) - 60 * (1 + 0.8 * np.log2(0.8) + 0.2 * np.log2(0.2))) < 1e-9


def test_benchmark_stats_and_holm():
    import pandas as pd
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"a": 0.8 + rng.normal(0, 0.01, 10), "b": 0.6 + rng.normal(0, 0.01, 10), "c": 0.6 + rng.normal(0, 0.01, 10)})
    s = C.benchmark_stats(df, 80, 20)
    assert s["friedman"].pvalue < 0.05
    row = s["pairs"].query("A == 'a' and B == 'b'").iloc[0]
    assert row.corrected_t_p_holm < 0.05
    assert np.all(C.holm([0.01, 0.04, 0.03]) >= np.array([0.01, 0.04, 0.03]))


def test_all_recipes_run(epochs):
    X, y, meta, times = _xy(epochs)
    splits = C.make_splits({"scheme": "kfold", "k": 3}, y, meta, 0)
    for name, rec in C.RECIPES.items():
        pipe = C.make_pipeline(dict(rec, n_components=4), epochs.info["sfreq"], times, 0)
        r = C.evaluate(pipe, X, y, splits)
        assert 0 <= r["summary"]["accuracy"][0] <= 1, name
