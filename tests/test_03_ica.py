"""§03 Artifact removal / ICA."""
import mne
import numpy as np

import core


def _blink_raw(seed=0):
    rng = np.random.RandomState(seed)
    sf, n = 200.0, 200 * 60
    names = ["Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2"]
    brain = rng.standard_normal((len(names), n)) * 5e-6
    blink = np.zeros(n)
    for b in rng.randint(0, n - 100, 25):
        blink[b:b + 80] += np.hanning(80) * 150e-6
    pat = np.array([1.0, 1.0, 0.45, 0.4, 0.45, 0.1, 0.1, 0.1, 0.02, 0.02, 0.02, 0.0, 0.0])
    raw = mne.io.RawArray(brain + np.outer(pat, blink), mne.create_info(names, sf, "eeg"))
    raw.set_montage("standard_1005")
    return raw, blink


def test_ica_removes_synthetic_blink():
    raw, blink = _blink_raw()
    r, ica, labels = core.fit_ica(raw, {"method": "fastica", "n_components": 10, "eog_thr": 0.5}, seed=1)
    assert "eye" in labels.values()
    clean = core.apply_ica(r, ica, [k for k, v in labels.items() if v == "eye"])
    before = abs(np.corrcoef(raw.get_data(picks=["Fp1"])[0], blink)[0, 1])
    after = abs(np.corrcoef(clean.get_data(picks=["Fp1"])[0], blink)[0, 1])
    assert before > 0.8 and after < 0.3


def test_ica_with_nothing_excluded_keeps_data():
    raw, _ = _blink_raw(1)
    r, ica, _ = core.fit_ica(raw, {"method": "fastica", "n_components": 12}, seed=0)
    out = core.apply_ica(r, ica, [])
    assert np.allclose(out.get_data(), r.get_data(), atol=1e-7)


def test_bad_channel_is_interpolated():
    raw, _ = _blink_raw(2)
    raw._data[raw.ch_names.index("Cz")] = 0
    r, _, _ = core.fit_ica(raw, {"method": "fastica", "n_components": 8, "bads": ["Cz"]}, seed=0)
    assert r.info["bads"] == [] and np.std(r.get_data(picks=["Cz"])) > 1e-7


def test_ica_is_seeded(objs, params):
    ip = {"method": "fastica", "n_components": 8, "decim": 4}
    a = core.fit_ica(objs["raw_f"].copy().crop(0, 120), ip, seed=3)[1].get_components()
    b = core.fit_ica(objs["raw_f"].copy().crop(0, 120), ip, seed=3)[1].get_components()
    assert np.allclose(a, b)
