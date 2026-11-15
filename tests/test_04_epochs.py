"""§04 Epoching / ERP."""
import numpy as np

import core


def test_epoch_counts_and_metadata(epochs, params):
    n = params["data"]["trials_per_class"] * 2 * params["data"]["n_sessions"]
    assert len(epochs) == n
    assert set(epochs.metadata.columns) >= {"label", "subject", "session", "onset_s"}
    assert sorted(epochs.metadata.session.unique()) == [1, 2]


def test_baseline_makes_prestimulus_mean_zero(objs):
    e = core.make_epochs(objs["raw"], {"events": None, "tmin": -0.5, "tmax": 1.0, "baseline": [-0.5, 0.0], "reject_uv": None}, objs["segments"])
    pre = e.get_data()[:, :, e.times <= 0].mean(axis=2)
    assert np.abs(pre).max() < 1e-12


def test_rejection_drops_epochs(objs):
    ep = {"events": None, "tmin": -1.0, "tmax": 4.0, "baseline": None, "reject_uv": None}
    a = core.make_epochs(objs["raw"], ep, objs["segments"])
    b = core.make_epochs(objs["raw"], dict(ep, reject_uv=40.0), objs["segments"])
    assert len(b) < len(a)


def test_event_selection(objs):
    e = core.make_epochs(objs["raw"], {"events": ["left_hand"], "tmin": 0, "tmax": 1, "baseline": None, "reject_uv": None}, objs["segments"])
    assert set(e.metadata.label) == {"left_hand"}
