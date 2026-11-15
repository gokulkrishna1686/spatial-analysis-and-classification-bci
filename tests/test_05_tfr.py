"""§05 Time-frequency."""
import numpy as np
from mne.time_frequency import tfr_array_morlet

from spatial import bandpass


def _mu_power(epochs, ch, label):
    x = epochs[label].get_data(picks=[ch])[:, 0]
    x = bandpass(x, epochs.info["sfreq"], 9, 13)
    t = epochs.times
    return (x[:, (t > 0.8) & (t < 3.0)] ** 2).mean(), (x[:, t < -0.2] ** 2).mean()


def test_contralateral_erd_in_synthetic_data(epochs):
    # right-hand imagery -> ERD over C3 (left hemisphere)
    task_r, base_r = _mu_power(epochs, "C3", "right_hand")
    task_l, _ = _mu_power(epochs, "C3", "left_hand")
    assert task_r < base_r            # power drops relative to baseline
    assert task_r < task_l            # and is lower than during left-hand imagery


def test_morlet_shape_and_peak_frequency():
    sf = 250.0
    t = np.arange(0, 2, 1 / sf)
    x = np.sin(2 * np.pi * 11 * t)[None, None]
    freqs = np.arange(6, 31, 1.0)
    p = tfr_array_morlet(x, sf, freqs, n_cycles=5, output="power", verbose="ERROR")
    assert p.shape == (1, 1, len(freqs), len(t))
    assert freqs[np.argmax(p[0, 0, :, 250])] == 11


def test_erd_percent_formula():
    base, task = 4.0, 3.0
    assert (task / base - 1) * 100 == -25.0
