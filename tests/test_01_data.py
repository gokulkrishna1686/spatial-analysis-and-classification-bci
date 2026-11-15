"""§01 Data loading and quality check."""
import numpy as np
import pandas as pd

import core


def test_synthetic_has_events_sessions_and_montage(objs):
    raw = objs["raw0"]
    assert raw.info["sfreq"] == 250
    assert {"left_hand", "right_hand"} <= set(raw.annotations.description)
    assert len({s[3] for s in objs["segments"]}) == 2
    assert raw.get_montage() is not None
    assert objs["true_patterns"] and len(objs["true_patterns"]["left_hand"]) == len(core.SYN_CHANNELS)


def test_synthetic_is_deterministic():
    a, _, _ = core.make_synthetic(trials_per_class=5, n_sessions=1, seed=7)
    b, _, _ = core.make_synthetic(trials_per_class=5, n_sessions=1, seed=7)
    c, _, _ = core.make_synthetic(trials_per_class=5, n_sessions=1, seed=8)
    assert np.allclose(a.get_data(), b.get_data())
    assert not np.allclose(a.get_data(), c.get_data())


def test_quality_check_flags_flat_noisy_and_units(objs):
    raw = objs["raw0"].copy().crop(0, 60)
    raw._data[raw.ch_names.index("Fz")] = 0.0
    raw._data[raw.ch_names.index("P3")] *= 40
    q, notes = core.quality_check(raw)
    issues = dict(zip(q.channel, q.issue))
    assert issues["Fz"] == "flat" and issues["P3"] == "noisy"
    raw._data *= 1e6                               # pretend the file stored µV as V
    _, notes = core.quality_check(raw)
    assert any("µV" in n for n in notes)


def test_csv_upload_roundtrip(tmp_path):
    x = np.random.RandomState(0).standard_normal((1000, 3)) * 20       # µV values
    pd.DataFrame(x, columns=["C3", "Cz", "C4"]).to_csv(tmp_path / "rec.csv", index=False)
    raw = core.read_any(str(tmp_path / "rec.csv"), sfreq_csv=200)
    assert raw.ch_names == ["C3", "Cz", "C4"] and raw.info["sfreq"] == 200
    assert np.allclose(raw.get_data()[0], x[:, 0] * 1e-6)                 # converted to volts


def test_unsupported_file_gives_friendly_error(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    try:
        core.read_any(str(p))
    except ValueError as e:
        assert "Unsupported file type" in str(e)
    else:
        raise AssertionError("expected ValueError")
