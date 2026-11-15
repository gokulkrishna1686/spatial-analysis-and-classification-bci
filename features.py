"""Feature extractors as scikit-learn transformers on epochs arrays
(n_trials, n_channels, n_samples). Stateless ones can be applied anywhere;
data-fitted ones (CSP, tangent space) are refit inside each CV training fold.
No Streamlit / Plotly here (embedded into the generated script)."""
import numpy as np
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin

from spatial import CSP, bandpass  # @local-import


class LogVar(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(X.var(axis=2) + 1e-30)


class BandPower(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=250.0, bands=((8, 12), (12, 16), (16, 24), (24, 30))):
        self.sfreq = sfreq
        self.bands = bands

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        f, p = welch(X, self.sfreq, nperseg=min(X.shape[2], int(self.sfreq)), axis=2)
        out = [np.log(p[:, :, (f >= lo) & (f < hi)].mean(axis=2) + 1e-30) for lo, hi in self.bands]
        return np.concatenate(out, axis=1)


class Hjorth(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d1 = np.diff(X, axis=2)
        d2 = np.diff(d1, axis=2)
        v0, v1, v2 = X.var(2) + 1e-30, d1.var(2) + 1e-30, d2.var(2) + 1e-30
        mob = np.sqrt(v1 / v0)
        comp = np.sqrt(v2 / v1) / mob
        return np.concatenate([np.log(v0), mob, comp], axis=1)


class SpectralEntropy(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq=250.0, fmin=1.0, fmax=40.0):
        self.sfreq, self.fmin, self.fmax = sfreq, fmin, fmax

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        f, p = welch(X, self.sfreq, nperseg=min(X.shape[2], int(self.sfreq)), axis=2)
        p = p[:, :, (f >= self.fmin) & (f <= self.fmax)]
        p = p / (p.sum(axis=2, keepdims=True) + 1e-30)
        return -(p * np.log2(p + 1e-30)).sum(axis=2) / np.log2(p.shape[2])


class ARCoeffs(BaseEstimator, TransformerMixin):
    """Autoregressive coefficients per channel (Yule-Walker)."""

    def __init__(self, order=4):
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from scipy.linalg import solve_toeplitz
        n, c, t = X.shape
        out = np.zeros((n, c, self.order))
        Xc = X - X.mean(axis=2, keepdims=True)
        for i in range(n):
            for j in range(c):
                x = Xc[i, j]
                r = np.correlate(x, x, "full")[t - 1:t + self.order] / t
                out[i, j] = solve_toeplitz(r[:-1], r[1:]) if r[0] > 0 else 0
        return out.reshape(n, -1)


class WaveletEnergy(BaseEstimator, TransformerMixin):
    """Log energy of Morlet wavelet coefficients at a few frequencies."""

    def __init__(self, sfreq=250.0, freqs=(10.0, 20.0)):
        self.sfreq, self.freqs = sfreq, freqs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from mne.time_frequency import tfr_array_morlet
        p = tfr_array_morlet(X, self.sfreq, np.array(self.freqs, float), n_cycles=5, output="power", verbose="ERROR")
        return np.log(p.mean(axis=3) + 1e-30).reshape(len(X), -1)


class ERPWindows(BaseEstimator, TransformerMixin):
    """Mean amplitude in time windows (classic P300 features). `times` = epoch time axis."""

    def __init__(self, times=None, windows=((0.25, 0.35), (0.35, 0.45), (0.45, 0.55))):
        self.times, self.windows = times, windows

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        t = np.asarray(self.times) if self.times is not None else np.linspace(0, 1, X.shape[2])
        out = [X[:, :, (t >= a) & (t < b)].mean(axis=2) * 1e6 for a, b in self.windows]
        return np.concatenate(out, axis=1)


class TangentSpaceFeat(BaseEstimator, TransformerMixin):
    """Covariance -> Riemannian tangent space (reference mean fitted on training data)."""

    def __init__(self, estimator="oas"):
        self.estimator = estimator

    def fit(self, X, y=None):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        self.cov_ = Covariances(self.estimator)
        self.ts_ = TangentSpace(metric="riemann").fit(self.cov_.fit_transform(X))
        return self

    def transform(self, X):
        return self.ts_.transform(self.cov_.transform(X))


class CSPFeat(CSP):
    pass


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """FBCSP: band-pass into several bands, CSP log-variance per band, concatenated."""

    def __init__(self, sfreq=250.0, bands=((4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)),
                 n_components=4, cov="ledoit_wolf"):
        self.sfreq, self.bands, self.n_components, self.cov = sfreq, bands, n_components, cov

    def _band(self, X, b):
        return bandpass(X, self.sfreq, b[0], b[1])

    def fit(self, X, y):
        self.csps_ = [CSP(self.n_components, self.cov).fit(self._band(X, b), y) for b in self.bands]
        return self

    def transform(self, X):
        return np.concatenate([c.transform(self._band(X, b)) for c, b in zip(self.csps_, self.bands)], axis=1)


FEATURE_INFO = {
    "logvar": "Log-variance per channel (power of the band-passed signal)",
    "bandpower": "Log band power per channel and band (Welch)",
    "csp": "CSP log-variance (spatially filtered, fitted on training folds)",
    "hjorth": "Hjorth activity, mobility, complexity",
    "entropy": "Spectral entropy (how 'flat' the spectrum is)",
    "ar": "Autoregressive coefficients",
    "wavelet": "Morlet wavelet energy at 10 and 20 Hz",
    "erp": "Mean amplitude in ERP time windows",
    "tangent": "Riemannian tangent-space vector of the covariance",
}


def make_feature_union(kinds, sfreq, times, fp):
    from sklearn.pipeline import FeatureUnion
    blocks = []
    for k in kinds:
        if k == "logvar":
            blocks.append((k, LogVar()))
        elif k == "bandpower":
            blocks.append((k, BandPower(sfreq, tuple(tuple(b) for b in fp.get("bands", [[8, 12], [12, 30]])))))
        elif k == "csp":
            blocks.append((k, CSP(fp.get("n_csp", 6))))
        elif k == "hjorth":
            blocks.append((k, Hjorth()))
        elif k == "entropy":
            blocks.append((k, SpectralEntropy(sfreq)))
        elif k == "ar":
            blocks.append((k, ARCoeffs(int(fp.get("ar_order", 4)))))
        elif k == "wavelet":
            blocks.append((k, WaveletEnergy(sfreq)))
        elif k == "erp":
            blocks.append((k, ERPWindows(times, tuple(tuple(w) for w in fp.get("erp_windows", [[0.3, 0.5]])))))
        elif k == "tangent":
            blocks.append((k, TangentSpaceFeat()))
    if not blocks:
        blocks = [("logvar", LogVar())]
    return FeatureUnion(blocks)


def feature_names(kinds, ch_names, fp, n_csp=6):
    names = []
    for k in kinds:
        if k == "logvar":
            names += [f"logvar·{c}" for c in ch_names]
        elif k == "bandpower":
            names += [f"bp{lo}-{hi}·{c}" for lo, hi in fp.get("bands", [[8, 12], [12, 30]]) for c in ch_names]
        elif k == "csp":
            names += [f"csp{i + 1}" for i in range(fp.get("n_csp", n_csp))]
        elif k == "hjorth":
            names += [f"{h}·{c}" for h in ("act", "mob", "cplx") for c in ch_names]
        elif k == "entropy":
            names += [f"sent·{c}" for c in ch_names]
        elif k == "ar":
            names += [f"ar{o + 1}·{c}" for c in ch_names for o in range(int(fp.get("ar_order", 4)))]
        elif k == "wavelet":
            names += [f"wav{f}·{c}" for c in ch_names for f in (10, 20)]
        elif k == "erp":
            names += [f"erp{a:.2f}-{b:.2f}·{c}" for a, b in fp.get("erp_windows", [[0.3, 0.5]]) for c in ch_names]
        elif k == "tangent":
            n = len(ch_names)
            names += [f"ts{i + 1}" for i in range(n * (n + 1) // 2)]
    return names
