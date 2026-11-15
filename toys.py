"""Small, transparent toy models behind the concept demos on the Spatial and
Classification pages. Pure numpy / scipy / scikit-learn: no Streamlit, no
plotting, so every demo is unit-tested (tests/test_toys.py)."""
import numpy as np
from scipy import linalg, stats


# ===========================================================================
# Spatial analysis toys
# ===========================================================================

def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def two_class_2d(n=200, erd=0.6, mix=0.3, shared=0.6, seed=0):
    """Two 'electrodes' (C3, C4) watching two sources (left and right motor cortex).
    Left-hand imagery weakens the right-hemisphere source (C4 side), right-hand
    imagery the left one (ERD). `mix` = volume conduction between the electrodes,
    `shared` = a common noise source seen by both electrodes.
    Returns XA, XB (n x 2) and their covariance matrices."""
    rng = np.random.RandomState(seed)
    M = np.array([[1.0, mix], [mix, 1.0]])               # sources -> electrodes
    def cls(scale):
        S = rng.standard_normal((n, 2)) * scale
        common = rng.standard_normal((n, 1)) * shared
        return S @ M.T + common @ np.ones((1, 2))
    XA = cls(np.array([1.0, 1.0 - erd]))                 # "left hand": C4-side source desynchronises
    XB = cls(np.array([1.0 - erd, 1.0]))                 # "right hand": C3-side source desynchronises
    return XA, XB, np.cov(XA.T), np.cov(XB.T)


def projected_variance(CA, CB, thetas):
    """Variance of the filtered signal w(theta)^T x for each class, and the CSP
    criterion lambda = varA / (varA + varB)."""
    W = np.stack([np.cos(thetas), np.sin(thetas)], 1)
    vA = np.einsum("ti,ij,tj->t", W, CA, W)
    vB = np.einsum("ti,ij,tj->t", W, CB, W)
    return vA, vB, vA / (vA + vB)


def csp_2d(CA, CB):
    """CSP for two channels: generalised eigenproblem CA w = lambda (CA + CB) w.
    Returns eigenvalues (descending), filters (rows) and the filters' angles."""
    lam, V = linalg.eigh(CA, CA + CB)
    o = np.argsort(lam)[::-1]
    lam, V = lam[o], V[:, o]
    V = V / np.linalg.norm(V, axis=0)
    angles = np.mod(np.arctan2(V[1], V[0]), np.pi)
    return lam, V.T, angles


def whiten_rotate(CA, CB):
    """CSP = whitening of the pooled covariance + a rotation.
    Returns the whitening matrix P, rotation U (proper, det = +1) and the angle of U."""
    C = CA + CB
    w, E = linalg.eigh(C)
    P = E @ np.diag(1.0 / np.sqrt(w)) @ E.T
    _, U = linalg.eigh(P @ CA @ P.T)
    if np.linalg.det(U) < 0:
        U[:, 0] = -U[:, 0]
    angle = np.arctan2(U[1, 0], U[0, 0])
    return P, U, angle


def whiten_rotate_transform(P, angle, t):
    """Continuous transform for the animation: t in [0, 1] blends identity -> whitening,
    t in [1, 2] then rotates the whitened space to the CSP axes."""
    if t <= 1:
        return (1 - t) * np.eye(2) + t * P
    return rot(-angle * (t - 1)) @ P


def ellipse(C, k=2.0, n=100):
    """Points on the k-sigma ellipse of a 2x2 covariance matrix."""
    w, V = linalg.eigh(C)
    a = np.linspace(0, 2 * np.pi, n)
    pts = V @ (np.sqrt(np.maximum(w, 0))[:, None] * np.vstack([np.cos(a), np.sin(a)])) * k
    return pts[0], pts[1]


def volume_conduction(spread=0.5, seed=0, n=600, sfreq=250.0):
    """Two independent mu-like sources under C3 and C4, projected to a 10-20 cap.
    Larger `spread` = blurrier projection = electrodes mix the sources.
    Returns channel names, the two source topographies, C3/C4 signals and their correlation."""
    from core import SYN_CHANNELS, montage_xyz
    xyz = montage_xyz(SYN_CHANNELS)
    width = 0.02 + spread * 0.13          # metres; C3 and C4 are ~13 cm apart
    def blob(name):
        c = xyz[SYN_CHANNELS.index(name)]
        return np.exp(-0.5 * (np.linalg.norm(xyz - c, axis=1) / width) ** 2)
    a1, a2 = blob("C3"), blob("C4")
    rng = np.random.RandomState(seed)
    t = np.arange(n) / sfreq
    s1 = np.sin(2 * np.pi * 10 * t + 0.3) * (1 + 0.5 * np.sin(2 * np.pi * 0.7 * t)) + 0.3 * rng.standard_normal(n)
    s2 = np.sin(2 * np.pi * 11.5 * t + 1.7) * (1 + 0.5 * np.cos(2 * np.pi * 0.45 * t)) + 0.3 * rng.standard_normal(n)
    X = np.outer(a1, s1) + np.outer(a2, s2)
    i3, i4 = SYN_CHANNELS.index("C3"), SYN_CHANNELS.index("C4")
    r = float(np.corrcoef(X[i3], X[i4])[0, 1])
    return SYN_CHANNELS, a1, a2, t, X[i3], X[i4], s1, s2, r


def haufe_demo(noise=1.0, n=3000, seed=0):
    """Signal s only on channel 1; a common noise source on both channels.
    The best filter subtracts channel 2 (w ~ [1, -1]) even though channel 2
    carries no signal; the activation pattern a = C w / (w' C w) ~ [1, 0]
    shows where the signal really is (Haufe et al. 2014)."""
    rng = np.random.RandomState(seed)
    s = rng.standard_normal(n)
    z = rng.standard_normal(n) * noise
    X = np.vstack([s + z, z])                     # 2 x n
    C = np.cov(X)
    w = np.linalg.solve(C, np.cov(np.vstack([X, s]))[:2, 2])   # least-squares filter that recovers s
    a = C @ w / (w @ C @ w)
    return w / np.abs(w).max(), a / np.abs(a).max(), X, s


def riemann_path(A, B, ts):
    """Euclidean (straight-line) and Riemannian (geodesic) interpolation between SPD matrices."""
    A_h = linalg.sqrtm(A).real
    A_ih = np.linalg.inv(A_h)
    M = A_ih @ B @ A_ih
    w, V = linalg.eigh(M)
    geo = [A_h @ (V @ np.diag(w ** t) @ V.T) @ A_h for t in ts]
    euc = [(1 - t) * A + t * B for t in ts]
    return euc, geo


# ===========================================================================
# Classification toys
# ===========================================================================

def lda_toy(sep=2.0, spread=1.0, n=150, seed=0):
    """Two 2-D Gaussian classes with a shared covariance and the LDA solution
    w = Sigma^-1 (mu1 - mu0). Returns X, y, w and the projected scores."""
    rng = np.random.RandomState(seed)
    cov = spread ** 2 * np.array([[1.0, 0.55], [0.55, 1.0]])
    L = np.linalg.cholesky(cov)
    mu = np.array([sep / 2, -sep / 4])
    X0 = rng.standard_normal((n, 2)) @ L.T - mu
    X1 = rng.standard_normal((n, 2)) @ L.T + mu
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    m0, m1 = X0.mean(0), X1.mean(0)
    Sw = (np.cov(X0.T) + np.cov(X1.T)) / 2
    w = np.linalg.solve(Sw, m1 - m0)
    scores = X @ w - (m0 + m1) @ w / 2           # 0 = the LDA boundary
    return X, y, w, scores


def threshold_stats(scores, y, thr):
    """Confusion matrix, TPR and FPR when 'class 1' means score > thr."""
    pred = (scores > thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return np.array([[tn, fp], [fn, tp]]), tp / max(1, tp + fn), fp / max(1, fp + tn), (tp + tn) / len(y)


def roc(scores, y):
    from sklearn.metrics import roc_auc_score, roc_curve
    fpr, tpr, _ = roc_curve(y, scores)
    return fpr, tpr, float(roc_auc_score(y, scores))


def binomial_null(n_test, n_classes, alpha=0.05, n_sim=1000, seed=0):
    """Accuracy distribution of a classifier that guesses at random:
    exact binomial pmf, the alpha-threshold and a Monte-Carlo check."""
    p = 1.0 / n_classes
    k = np.arange(n_test + 1)
    pmf = stats.binom.pmf(k, n_test, p)
    thr = stats.binom.ppf(1 - alpha, n_test, p) / n_test
    sim = np.random.RandomState(seed).binomial(n_test, p, size=n_sim) / n_test
    return k / n_test, pmf, float(thr), sim


def binomial_p(acc, n_test, n_classes):
    """P(random classifier reaches at least this accuracy)."""
    return float(stats.binom.sf(np.ceil(acc * n_test) - 1, n_test, 1.0 / n_classes))


def moons_toy(n=200, noise=0.35, seed=0):
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X, y


def knn_complexity_curve(X, y, ks, seed=0):
    """Train / held-out accuracy of kNN for each k (one fixed 50/50 split)."""
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)
    tr, te = [], []
    for k in ks:
        m = KNeighborsClassifier(n_neighbors=int(k)).fit(Xtr, ytr)
        tr.append(m.score(Xtr, ytr))
        te.append(m.score(Xte, yte))
    return np.array(tr), np.array(te), (Xtr, Xte, ytr, yte)


def leakage_toy(n_trials=60, n_feat=500, k=10, n_rep=10, seed=0):
    """Pure noise features and random labels. 'Honest' selects the k best features
    inside each training fold; 'leaky' selects them once on ALL trials before CV."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    rng = np.random.RandomState(seed)
    honest, leaky = [], []
    for r in range(n_rep):
        X = rng.standard_normal((n_trials, n_feat))
        y = rng.permutation(np.r_[np.zeros(n_trials // 2, int), np.ones(n_trials - n_trials // 2, int)])
        cv = list(StratifiedKFold(5, shuffle=True, random_state=r).split(X, y))
        sel = SelectKBest(f_classif, k=min(k, n_feat)).fit(X, y).get_support()
        h, l_ = [], []
        for tr, te in cv:
            p = make_pipeline(SelectKBest(f_classif, k=min(k, n_feat)), LinearDiscriminantAnalysis())
            h.append(p.fit(X[tr], y[tr]).score(X[te], y[te]))
            l_.append(LinearDiscriminantAnalysis().fit(X[tr][:, sel], y[tr]).score(X[te][:, sel], y[te]))
        honest.append(np.mean(h))
        leaky.append(np.mean(l_))
    return np.array(honest), np.array(leaky)


def cv_folds(X, y, k=5, seed=0):
    """Stratified k-fold with an LDA: (train idx, test idx, test accuracy) per fold."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold
    out = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        out.append((tr, te, float(LinearDiscriminantAnalysis().fit(X[tr], y[tr]).score(X[te], y[te]))))
    return out
