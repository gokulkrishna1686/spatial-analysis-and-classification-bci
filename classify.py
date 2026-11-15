"""Leak-free classification: pipeline factory, CV protocols, metrics, chance
levels, permutation tests, benchmark statistics, time-resolved decoding,
learning curves and the leakage demonstration.

Every data-fitted step (CSP, scaling, tangent-space reference, feature
selection, hyper-parameter search) lives INSIDE an sklearn Pipeline, which is
cloned and fitted on the training indices of each fold only.
No Streamlit / Plotly here (embedded into the generated script)."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut, RepeatedStratifiedKFold,
                                     StratifiedKFold)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from spatial import CSP, bandpass, info_xyz, pre_spatial  # @local-import
from features import BandPower, FilterBankCSP, TangentSpaceFeat, make_feature_union  # @local-import

# --------------------------------------------------------------------------
# Data -> X, y
# --------------------------------------------------------------------------

def get_Xy(epochs, band=(8.0, 30.0), window=(0.5, 2.5), classes=None, pre=None):
    """Stateless per-trial steps only: optional CAR/Laplacian, band-pass, crop.
    Returns X, y, meta, times."""
    meta = epochs.metadata.reset_index(drop=True)
    X = epochs.get_data(picks="eeg")
    sf = epochs.info["sfreq"]
    if pre in ("CAR", "Laplacian"):
        X = pre_spatial(X, pre, info_xyz(epochs.info))
    if band:
        X = bandpass(X, sf, band[0], band[1])
    t = epochs.times
    m = (t >= window[0]) & (t <= window[1])
    X = X[:, :, m]
    y = meta["label"].to_numpy()
    if classes:
        keep = np.isin(y, classes)
        X, y, meta = X[keep], y[keep], meta[keep].reset_index(drop=True)
    return X, y, meta, t[m]


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------

CLASSIFIERS = {
    "lda": "Linear Discriminant Analysis (shrinkage)",
    "lr": "Logistic regression",
    "svm": "Support vector machine",
    "rf": "Random forest",
    "knn": "k-nearest neighbours",
    "nb": "Gaussian naive Bayes",
    "mlp": "Multi-layer perceptron",
    "mdm": "Riemannian minimum distance to mean",
}

FEATURE_SOURCES = {
    "csp": "CSP log-variance",
    "fbcsp": "Filter-bank CSP",
    "tangent": "Riemannian tangent space",
    "bandpower": "Band power",
    "xdawn": "xDAWN covariances → tangent space",
    "features": "Feature set (choose families)",
    "cov": "Covariance matrices (for MDM)",
}

RECIPES = {
    "CSP + LDA": {"feature": "csp", "clf": "lda", "note": "The classic motor-imagery baseline (Ramoser 2000)."},
    "FBCSP + LDA": {"feature": "fbcsp", "clf": "lda", "note": "CSP in 7 sub-bands + mutual-information selection (Ang 2008)."},
    "Tangent space + LR": {"feature": "tangent", "clf": "lr", "note": "Riemannian geometry, strong all-rounder (Barachant 2012)."},
    "MDM": {"feature": "cov", "clf": "mdm", "note": "Nearest Riemannian class mean. No feature engineering at all."},
    "xDAWN + LR": {"feature": "xdawn", "clf": "lr", "note": "ERP / P300 standard (Rivet 2009, Barachant 2014)."},
    "Band power + SVM": {"feature": "bandpower", "clf": "svm", "note": "Spectral features with an RBF-kernel SVM."},
}


def make_classifier(name, p, seed=42):
    p = p or {}
    if name == "lda":
        sh = p.get("shrinkage", "auto")
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None if sh == "none" else sh)
    if name == "lr":
        return LogisticRegression(C=float(p.get("C", 1.0)), max_iter=2000, random_state=seed)
    if name == "svm":
        return SVC(C=float(p.get("C", 1.0)), kernel=p.get("kernel", "rbf"), gamma=p.get("gamma", "scale"),
                   probability=True, random_state=seed)
    if name == "rf":
        md = p.get("max_depth", None)
        return RandomForestClassifier(n_estimators=int(p.get("n_estimators", 200)),
                                      max_depth=None if md in (None, 0) else int(md), random_state=seed)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=int(p.get("k", 7)))
    if name == "nb":
        return GaussianNB()
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(int(p.get("hidden", 32)),), alpha=float(p.get("alpha", 1e-3)),
                             max_iter=1500, random_state=seed)
    if name == "mdm":
        from pyriemann.classification import MDM
        return MDM(metric="riemann")
    raise ValueError(f"Unknown classifier {name}")


def make_pipeline(cfg, sfreq, times=None, seed=42, feat_params=None):
    """cfg: {'feature', 'clf', 'clf_params', 'n_components', 'cov'} -> sklearn Pipeline."""
    f, c = cfg.get("feature", "csp"), cfg.get("clf", "lda")
    nc = int(cfg.get("n_components", 6))
    steps = []
    if c == "mdm" or f == "cov":
        from pyriemann.estimation import Covariances
        steps = [("covariances", Covariances(cfg.get("cov_est", "oas")))]
        if c != "mdm":
            from pyriemann.tangentspace import TangentSpace
            steps += [("tangent_space", TangentSpace())]
    elif f == "csp":
        steps = [("csp", CSP(nc, cfg.get("cov", "ledoit_wolf")))]
    elif f == "fbcsp":
        steps = [("fbcsp", FilterBankCSP(sfreq, n_components=4)),
                 ("select", SelectKBest(mutual_info_classif, k=int(cfg.get("k_best", 10))))]
    elif f == "tangent":
        steps = [("tangent_space", TangentSpaceFeat("oas"))]
    elif f == "bandpower":
        steps = [("bandpower", BandPower(sfreq))]
    elif f == "xdawn":
        from pyriemann.estimation import XdawnCovariances
        from pyriemann.tangentspace import TangentSpace
        steps = [("xdawn_cov", XdawnCovariances(nfilter=max(1, int(cfg.get("n_components", 4)) // 2), estimator="oas", xdawn_estimator="lwf")),
                 ("tangent_space", TangentSpace())]
    elif f == "features":
        fp = feat_params or {}
        steps = [("features", make_feature_union(fp.get("kinds", ["logvar"]), sfreq, times, fp))]
    else:
        raise ValueError(f"Unknown feature source {f}")
    if c != "mdm":
        steps.append(("scaler", StandardScaler()))
        steps.append(("clf", make_classifier(c, cfg.get("clf_params"), seed)))
    else:
        steps.append(("clf", make_classifier("mdm", {}, seed)))
    return Pipeline(steps)


PARAM_GRIDS = {
    "lda": {"clf__shrinkage": ["auto", 0.1, 0.5]},
    "lr": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "svm": {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", 0.01, 0.1]},
    "rf": {"clf__max_depth": [3, 6, None]},
    "knn": {"clf__n_neighbors": [3, 7, 15]},
    "mlp": {"clf__alpha": [1e-4, 1e-2, 1.0]},
}


def maybe_nested(pipe, cfg, seed=42):
    """Wrap in GridSearchCV: hyper-parameters are chosen with an INNER CV that
    only ever sees the outer training fold."""
    grid = PARAM_GRIDS.get(cfg.get("clf"))
    if not cfg.get("nested") or not grid:
        return pipe
    return GridSearchCV(pipe, grid, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                        scoring="balanced_accuracy", n_jobs=1)


# --------------------------------------------------------------------------
# Cross-validation protocols (materialised splits, reused everywhere)
# --------------------------------------------------------------------------

def make_splits(cv, y, meta, seed=42):
    s = cv.get("scheme", "kfold")
    idx = np.arange(len(y))
    if s == "kfold":
        k = int(min(cv.get("k", 5), np.min(np.unique(y, return_counts=True)[1])))
        return [(tr, te) for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(idx, y)]
    if s == "repeated":
        k = int(min(cv.get("k", 5), np.min(np.unique(y, return_counts=True)[1])))
        return [(tr, te) for tr, te in RepeatedStratifiedKFold(n_splits=k, n_repeats=int(cv.get("repeats", 3)),
                                                               random_state=seed).split(idx, y)]
    if s in ("loso_session", "loso_subject"):
        g = meta["session" if s == "loso_session" else "subject"].to_numpy()
        if len(np.unique(g)) < 2:
            raise ValueError(f"Leave-one-{'session' if s == 'loso_session' else 'subject'}-out needs at least two "
                             f"{'sessions' if s == 'loso_session' else 'subjects'} in the data.")
        return list(LeaveOneGroupOut().split(idx, y, g))
    if s == "cross_session":
        g = meta["session"].to_numpy()
        a, b = cv.get("train") or int(np.min(g)), cv.get("test") or int(np.max(g))
        if a == b or not (g == a).any() or not (g == b).any():
            raise ValueError("Train-on-one/test-on-another needs two different sessions.")
        return [(idx[g == a], idx[g == b])]
    raise ValueError(f"Unknown CV scheme {s}")


def split_repeat_ids(splits, n):
    """Repeat number of each split (a new repeat starts when trials get re-used as test)."""
    seen, rep, out = np.zeros(n, bool), 0, []
    for _, te in splits:
        if seen[te].any():
            rep += 1
            seen[:] = False
        seen[te] = True
        out.append(rep)
    return out


# --------------------------------------------------------------------------
# Metrics and chance
# --------------------------------------------------------------------------

def itr_bits(acc, n_classes, trial_len):
    """Wolpaw information transfer rate in bits/min."""
    p, n = float(acc), int(n_classes)
    if p <= 1.0 / n:
        return 0.0
    b = np.log2(n) + p * np.log2(p)
    if p < 1:
        b += (1 - p) * np.log2((1 - p) / (n - 1))
    return float(b * 60.0 / trial_len)


def fold_metrics(y_true, y_pred, proba, classes, trial_len):
    m = {"accuracy": accuracy_score(y_true, y_pred),
         "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
         "kappa": cohen_kappa_score(y_true, y_pred),
         "f1": f1_score(y_true, y_pred, average="macro")}
    try:
        if len(classes) == 2:
            m["roc_auc"] = roc_auc_score(y_true == classes[1], proba[:, 1])
        else:
            m["roc_auc"] = roc_auc_score(y_true, proba, multi_class="ovr", labels=classes)
    except Exception:
        m["roc_auc"] = np.nan
    m["itr_bits_min"] = itr_bits(m["accuracy"], len(classes), trial_len)
    return m


def chance_levels(n_test, n_classes, alpha=0.05):
    """Theoretical chance and the binomial significance threshold
    (Combrisson & Jerbi 2015): the accuracy a random classifier exceeds with
    probability alpha for this many test trials."""
    theo = 1.0 / n_classes
    thr = stats.binom.ppf(1 - alpha, n_test, theo) / n_test
    return float(theo), float(thr)


def _proba(model, X, classes):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except Exception:
            pass
    d = model.decision_function(X)
    if d.ndim == 1:
        p1 = 1 / (1 + np.exp(-d))
        return np.c_[1 - p1, p1]
    e = np.exp(d - d.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def evaluate(pipe, X, y, splits, trial_len=4.0, progress=None):
    """Fit a fresh clone on each training fold, predict the test fold.
    Returns per-fold metrics, out-of-fold predictions and ROC data."""
    classes = np.unique(y)
    reps = split_repeat_ids(splits, len(y))
    rows, oof, rocs = [], [], []
    for i, (tr, te) in enumerate(splits):
        model = clone(pipe).fit(X[tr], y[tr])
        pred = model.predict(X[te])
        proba = _proba(model, X[te], classes)
        m = fold_metrics(y[te], pred, proba, classes, trial_len)
        m.update(fold=i + 1, repeat=reps[i], n_train=len(tr), n_test=len(te))
        if hasattr(model, "best_params_"):
            m["best_params"] = str(model.best_params_)
        rows.append(m)
        for j, t in enumerate(te):
            oof.append({"trial": int(t), "fold": i + 1, "repeat": reps[i], "y_true": y[t], "y_pred": pred[j],
                        **{f"p_{c}": float(proba[j, k]) for k, c in enumerate(classes)}})
        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(y[te] == classes[1], proba[:, 1])
            rocs.append((fpr, tpr))
        if progress:
            progress((i + 1) / len(splits), f"fold {i + 1}/{len(splits)}")
    folds = pd.DataFrame(rows)
    oof = pd.DataFrame(oof)
    keys = ["accuracy", "balanced_accuracy", "kappa", "f1", "roc_auc", "itr_bits_min"]
    summary = {k: (float(folds[k].mean()), float(folds[k].std(ddof=1)) if len(folds) > 1 else 0.0) for k in keys}
    first = oof[oof.repeat == 0]
    cm = confusion_matrix(first.y_true, first.y_pred, labels=classes)
    return {"folds": folds, "oof": oof, "summary": summary, "classes": [str(c) for c in classes],
            "confusion": cm, "rocs": rocs, "n_test_total": int(len(first))}


def score_metric(y_true, y_pred, metric="accuracy"):
    return balanced_accuracy_score(y_true, y_pred) if metric == "balanced_accuracy" else accuracy_score(y_true, y_pred)


def choose_metric(y):
    """Accuracy for balanced classes; balanced accuracy when one class dominates
    (otherwise 'always predict the majority' would look like skill)."""
    _, n = np.unique(y, return_counts=True)
    return "balanced_accuracy" if n.max() / n.sum() > 0.6 else "accuracy"


def permutation_null(pipe, X, y, splits, n_perm=100, seed=42, progress=None, metric="accuracy"):
    """Score distribution under the null hypothesis: labels shuffled,
    then the SAME leak-free CV is run again."""
    rng = np.random.RandomState(seed)
    null = []
    for p in range(n_perm):
        yp = rng.permutation(y)
        accs = []
        for tr, te in splits:
            model = clone(pipe).fit(X[tr], yp[tr])
            accs.append(score_metric(yp[te], model.predict(X[te]), metric))
        null.append(float(np.mean(accs)))
        if progress:
            progress((p + 1) / n_perm, f"permutation {p + 1}/{n_perm}")
    return np.array(null)


def perm_p_value(observed, null):
    return float((np.sum(null >= observed) + 1) / (len(null) + 1))


def fit_final(pipe, X, y):
    """Deployment model fitted on ALL trials. For export/interpretation only:
    its training accuracy is NOT an evaluation."""
    return clone(pipe).fit(X, y)


# --------------------------------------------------------------------------
# Benchmark statistics
# --------------------------------------------------------------------------

def holm(pvals):
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for CV fold scores."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    v = d.var(ddof=1)
    if v == 0:
        return 1.0
    t = d.mean() / np.sqrt((1.0 / k + n_test / n_train) * v)
    return float(2 * stats.t.sf(abs(t), k - 1))


def benchmark_stats(scores, n_train, n_test):
    """scores: DataFrame (folds x pipelines). Friedman + pairwise tests (Holm)."""
    names = list(scores.columns)
    out = {"friedman": None}
    if len(names) >= 3 and len(scores) >= 2:
        try:
            out["friedman"] = stats.friedmanchisquare(*[scores[n] for n in names])
        except Exception:
            out["friedman"] = None
    pairs, pw, pt = [], [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = scores[names[i]].to_numpy(), scores[names[j]].to_numpy()
            try:
                w = stats.wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
            except Exception:
                w = 1.0
            pairs.append((names[i], names[j]))
            pw.append(w)
            pt.append(corrected_ttest(a, b, n_train, n_test))
    pw_h, pt_h = (holm(pw), holm(pt)) if pairs else ([], [])
    out["pairs"] = pd.DataFrame({"A": [p[0] for p in pairs], "B": [p[1] for p in pairs],
                                 "wilcoxon_p": pw, "wilcoxon_p_holm": pw_h,
                                 "corrected_t_p": pt, "corrected_t_p_holm": pt_h})
    return out


# --------------------------------------------------------------------------
# Time-resolved decoding, learning curve, leakage demonstration
# --------------------------------------------------------------------------

def time_resolved(pipe, X_full, y, times, splits, win=1.0, step=0.25, progress=None):
    """Slide a window across the epoch; leak-free CV accuracy per window and
    out-of-fold class probabilities per trial per window (first repeat)."""
    classes = np.unique(y)
    sf = 1.0 / (times[1] - times[0])
    w = int(round(win * sf))
    starts = np.arange(0, X_full.shape[2] - w + 1, max(1, int(round(step * sf))))
    reps = split_repeat_ids(splits, len(y))
    acc = np.zeros((len(starts), len(splits)))
    probs = np.full((len(y), len(starts), len(classes)), np.nan)
    total = len(starts) * len(splits)
    for a, s0 in enumerate(starts):
        Xw = X_full[:, :, s0:s0 + w]
        for i, (tr, te) in enumerate(splits):
            m = clone(pipe).fit(Xw[tr], y[tr])
            acc[a, i] = accuracy_score(y[te], m.predict(Xw[te]))
            if reps[i] == 0:
                probs[te, a] = _proba(m, Xw[te], classes)
            if progress:
                progress((a * len(splits) + i + 1) / total, f"window {a + 1}/{len(starts)}")
    centers = times[starts] + win / 2
    return {"t": centers, "acc_mean": acc.mean(1), "acc_std": acc.std(1), "probs": probs, "classes": classes}


def learning_curve_manual(pipe, X, y, splits, fractions=(0.2, 0.35, 0.5, 0.7, 0.85, 1.0), seed=42, progress=None):
    rng = np.random.RandomState(seed)
    rows = []
    total = len(fractions) * len(splits)
    n = 0
    for f in fractions:
        for i, (tr, te) in enumerate(splits):
            # stratified subsample of the TRAINING fold only
            sub = np.concatenate([rng.permutation(tr[y[tr] == c])[:max(2, int(round(f * np.sum(y[tr] == c))))]
                                  for c in np.unique(y)])
            m = clone(pipe).fit(X[sub], y[sub])
            rows.append({"fraction": f, "n_train": len(sub), "fold": i + 1,
                         "test_acc": accuracy_score(y[te], m.predict(X[te])),
                         "train_acc": accuracy_score(y[sub], m.predict(X[sub]))})
            n += 1
            if progress:
                progress(n / total, f"{int(f * 100)}% of training data")
    return pd.DataFrame(rows)


def leakage_demo(X, y, splits, sfreq, n_shuffles=10, k_best=10, seed=42, progress=None):
    """REAL computation on SHUFFLED labels (there is nothing to learn).
    honest: band power -> feature selection -> LDA, all inside each fold.
    leaky : feature selection fitted on ALL trials before CV (classic mistake)."""
    rng = np.random.RandomState(seed)
    F = BandPower(sfreq, ((4, 8), (8, 10), (10, 12), (12, 16), (16, 20), (20, 24), (24, 30))).transform(X)
    honest_pipe = Pipeline([("scaler", StandardScaler()), ("select", SelectKBest(f_classif, k=k_best)),
                            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])
    rows = []
    for s in range(n_shuffles):
        ys = rng.permutation(y)
        a_h, a_l = [], []
        sel = SelectKBest(f_classif, k=k_best).fit(StandardScaler().fit_transform(F), ys)   # sees test trials!
        Fl = F[:, sel.get_support()]
        for tr, te in splits:
            a_h.append(accuracy_score(ys[te], clone(honest_pipe).fit(F[tr], ys[tr]).predict(F[te])))
            lk = Pipeline([("scaler", StandardScaler()), ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))])
            a_l.append(accuracy_score(ys[te], lk.fit(Fl[tr], ys[tr]).predict(Fl[te])))
        rows.append({"shuffle": s + 1, "honest": np.mean(a_h), "leaky": np.mean(a_l)})
        if progress:
            progress((s + 1) / n_shuffles, f"shuffle {s + 1}/{n_shuffles}")
    return pd.DataFrame(rows)


def run_classification(epochs, cp, seed=42, feat_params=None, progress=None):
    """End-to-end evaluation used by both the app and the generated script."""
    if cp.get("feature_kinds"):
        feat_params = dict(feat_params or {}, kinds=list(cp["feature_kinds"]))
    X, y, meta, times = get_Xy(epochs, cp.get("band", (8, 30)), cp.get("window", (0.5, 2.5)), cp.get("classes"), cp.get("pre"))
    splits = make_splits(cp.get("cv", {}), y, meta, seed)
    pipe = maybe_nested(make_pipeline(cp, epochs.info["sfreq"], times, seed, feat_params), cp, seed)
    res = evaluate(pipe, X, y, splits, cp.get("trial_len", 4.0),
                   progress=(lambda f, m: progress(0.5 * f, m)) if progress else None)
    metric = choose_metric(y)
    null = permutation_null(pipe, X, y, splits, int(cp.get("n_perm", 100)), seed,
                            progress=(lambda f, m: progress(0.5 + 0.5 * f, m)) if progress else None,
                            metric=metric) if int(cp.get("n_perm", 0)) > 0 else np.array([])
    acc = res["summary"][metric][0]
    theo, thr = chance_levels(res["n_test_total"], len(res["classes"]), cp.get("alpha", 0.05))
    res.update(null=null, p_value=perm_p_value(acc, null) if len(null) else np.nan, metric=metric,
               chance=theo, binom_threshold=thr, splits=splits, X=X, y=y, meta=meta, times=times, pipe=pipe)
    return res
