"""Spatial filters and helpers: CSP (own, transparent implementation),
CAR, surface Laplacian, PCA/whitening, xDAWN and Riemannian geometry tools.
No Streamlit / Plotly here (embedded into the generated script)."""
import numpy as np
from scipy import linalg
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import ledoit_wolf, oas
from sklearn.metrics import roc_auc_score


def cov_matrix(x, estimator="ledoit_wolf"):
    """Covariance of one trial x (channels x samples), trace-normalised."""
    x = x - x.mean(axis=1, keepdims=True)
    if estimator == "ledoit_wolf":
        c = ledoit_wolf(x.T, assume_centered=True)[0]
    elif estimator == "oas":
        c = oas(x.T, assume_centered=True)[0]
    elif isinstance(estimator, (int, float)) or str(estimator).startswith("shrink"):
        a = float(str(estimator).replace("shrink", "") or 0.1) if not isinstance(estimator, (int, float)) else float(estimator)
        c = x @ x.T / x.shape[1]
        c = (1 - a) * c + a * np.trace(c) / len(c) * np.eye(len(c))
    else:
        c = x @ x.T / x.shape[1]
    return c / np.trace(c)


def class_covs(X, y, estimator="ledoit_wolf"):
    classes = np.unique(y)
    C = np.array([cov_matrix(x, estimator) for x in X])
    return classes, np.array([C[y == c].mean(axis=0) for c in classes]), C


class CSP(BaseEstimator, TransformerMixin):
    """Common Spatial Patterns.

    Binary: solves the generalised eigenproblem C_a w = λ (C_a + C_b) w.
    λ near 1 -> variance high for class a, λ near 0 -> high for class b.
    Multiclass: one-vs-rest, the top components of each class are kept.
    transform() returns log-variance of the first `n_components` components.
    """

    def __init__(self, n_components=6, cov="ledoit_wolf", log=True):
        self.n_components = n_components
        self.cov = cov
        self.log = log

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_, means, _ = class_covs(X, y, self.cov)
        n_ch = X.shape[1]
        if len(self.classes_) == 2:
            w, V = linalg.eigh(means[0], means[0] + means[1])
            # alternate extremes: most "class a", most "class b", ...
            hi = [i for i in np.argsort(w)[::-1]]
            lo = [i for i in np.argsort(w)]
            alt = []
            for a, b in zip(hi, lo):
                for i in (a, b):
                    if i not in alt:
                        alt.append(i)
            order = np.array(alt[:n_ch])
            self.eigenvalues_ = w[order]
            self.favours_ = np.where(self.eigenvalues_ >= 0.5, self.classes_[0], self.classes_[1])
            V = V[:, order]
        else:
            Ctot = means.mean(axis=0)
            cols, ev, fav = [], [], []
            per = int(np.ceil(n_ch / len(self.classes_)))
            for k, c in enumerate(self.classes_):
                w, Vk = linalg.eigh(means[k], Ctot)
                idx = np.argsort(w)[::-1][:per]
                cols.append(Vk[:, idx])
                ev += list(w[idx])
                fav += [c] * len(idx)
            # interleave classes so the first components cover every class
            V_all = np.concatenate(cols, axis=1)
            inter = [k * per + j for j in range(per) for k in range(len(self.classes_))]
            inter = [i for i in inter if i < V_all.shape[1]][:n_ch]
            V = V_all[:, inter]
            self.eigenvalues_ = np.array(ev)[inter]
            self.favours_ = np.array(fav)[inter]
        self.filters_ = V.T                                   # rows = spatial filters
        self.patterns_ = linalg.pinv(V)                       # rows = activation patterns (columns of A = W^-1, Haufe 2014)
        return self

    def transform(self, X):
        W = self.filters_[: self.n_components]
        Z = np.einsum("kc,nct->nkt", W, X)
        v = Z.var(axis=2)
        return np.log(v) if self.log else v

    def components(self, X, k=None):
        W = self.filters_ if k is None else self.filters_[k:k + 1]
        return np.einsum("kc,nct->nkt", W, X)


def info_xyz(info):
    """3-D EEG sensor positions from an MNE info (falls back to the standard 10-05 montage)."""
    import mne
    picks = mne.pick_types(info, eeg=True)
    xyz = np.array([info["chs"][i]["loc"][:3] for i in picks], float)
    if not (np.isfinite(xyz).all() and (np.abs(xyz).sum(1) > 0).all()):
        mont = mne.channels.make_standard_montage("standard_1005")
        pos = {k.lower(): v for k, v in mont.get_positions()["ch_pos"].items()}
        xyz = np.array([pos.get(info.ch_names[i].lower(), [0, 0, 0]) for i in picks], float)
    return xyz


def pre_spatial(X, method, xyz=None):
    """Stateless re-referencing applied to every trial: 'CAR' or 'Laplacian'."""
    if method == "CAR":
        return car(X)
    if method == "Laplacian" and xyz is not None:
        return apply_spatial_matrix(X, laplacian_matrix(xyz))
    return X


def car(X):
    """Common average reference (works on (..., channels, samples))."""
    return X - X.mean(axis=-2, keepdims=True)


def laplacian_matrix(xyz, n_neighbors=4):
    """Hjorth surface Laplacian: each channel minus the mean of its nearest neighbours."""
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
    n = len(xyz)
    L = np.eye(n)
    for i in range(n):
        nb = np.argsort(d[i])[1:n_neighbors + 1]
        L[i, nb] = -1.0 / len(nb)
    return L


def apply_spatial_matrix(X, M):
    return np.einsum("kc,nct->nkt", M, X)


def pca_filters(X, n_components=None):
    Xc = np.concatenate([x - x.mean(1, keepdims=True) for x in X], axis=1)
    C = Xc @ Xc.T / Xc.shape[1]
    w, V = linalg.eigh(C)
    o = np.argsort(w)[::-1]
    w, V = w[o], V[:, o]
    if n_components:
        w, V = w[:n_components], V[:, :n_components]
    return w, V.T, C


def whitening_matrix(C, eps=1e-12):
    w, V = linalg.eigh(C)
    return V @ np.diag(1.0 / np.sqrt(np.maximum(w, eps))) @ V.T


def bandpass(X, sfreq, lo, hi, order=4):
    hi = min(hi, sfreq / 2 - 1)
    sos = butter(order, [lo, hi], btype="band", fs=sfreq, output="sos")
    return sosfiltfilt(sos, X, axis=-1)


def separation_auc(values, y):
    """Class separability of a 1-D feature in [0, 1] (0 = none, 1 = perfect), for 2 classes."""
    classes = np.unique(y)
    if len(classes) != 2 or np.std(values) == 0:
        return 0.0
    a = roc_auc_score(y == classes[1], values)
    return float(abs(a - 0.5) * 2)


def fisher_ratio(Z, y):
    """Between-class / within-class scatter of a 2-D (or n-D) point cloud."""
    classes = np.unique(y)
    mu = Z.mean(0)
    sb = sum((y == c).sum() * np.sum((Z[y == c].mean(0) - mu) ** 2) for c in classes)
    sw = sum(np.sum((Z[y == c] - Z[y == c].mean(0)) ** 2) for c in classes)
    return float(sb / (sw + 1e-12))


def match_pattern(pattern, candidates):
    """Best |correlation| between `pattern` and each row of `candidates`."""
    r = [abs(np.corrcoef(pattern, c)[0, 1]) for c in candidates]
    i = int(np.argmax(r))
    return i, float(r[i])


def align_sign(ref, p):
    return p if np.dot(ref, p) >= 0 else -p


# ---- Riemannian helpers ---------------------------------------------------

def riemann_means(C, y):
    from pyriemann.geometry.mean import mean_riemann
    classes = np.unique(y)
    return classes, np.array([mean_riemann(C[y == c]) for c in classes])


def riemann_distances(C, means):
    from pyriemann.geometry.distance import distance_riemann
    return np.array([[distance_riemann(c, m) for m in means] for c in C])
