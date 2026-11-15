"""§02 Filtering."""
import mne
import numpy as np

import core


def _sine_raw(freqs, sfreq=250.0, dur=20.0):
    t = np.arange(int(dur * sfreq)) / sfreq
    x = sum(np.sin(2 * np.pi * f * t) for f in freqs) * 1e-5
    raw = mne.io.RawArray(np.vstack([x, x]), mne.create_info(["C3", "C4"], sfreq, "eeg"))
    return raw, t


def _amp(x, f, sfreq):
    spec = np.abs(np.fft.rfft(x)) / len(x)
    fr = np.fft.rfftfreq(len(x), 1 / sfreq)
    return spec[np.argmin(np.abs(fr - f))]


def test_bandpass_removes_out_of_band_sine():
    raw, _ = _sine_raw([10.0, 70.0])
    out = core.apply_filter(raw, {"l_freq": 8, "h_freq": 30, "notch": None, "method": "fir", "phase": "zero", "reference": "none"})
    x0, x1 = raw.get_data()[0], out.get_data()[0]
    sl = slice(1000, -1000)
    assert _amp(x1[sl], 70, 250) < 0.01 * _amp(x0[sl], 70, 250)          # > 40 dB down
    assert _amp(x1[sl], 10, 250) > 0.9 * _amp(x0[sl], 10, 250)           # passband kept


def test_notch_removes_line_noise_iir():
    raw, _ = _sine_raw([10.0, 50.0])
    out = core.apply_filter(raw, {"l_freq": None, "h_freq": None, "notch": 50.0, "method": "iir", "phase": "zero", "iir_order": 4, "reference": "none"})
    sl = slice(1000, -1000)
    assert _amp(out.get_data()[0][sl], 50, 250) < 0.1 * _amp(raw.get_data()[0][sl], 50, 250)


def test_zero_phase_has_no_lag_but_causal_does():
    sf = 250.0
    x = np.zeros(5000)
    x[2500] = 1e-5                                         # an impulse
    raw = mne.io.RawArray(np.vstack([x, x]), mne.create_info(["C3", "C4"], sf, "eeg"))
    base = dict(l_freq=8, h_freq=16, notch=None, method="iir", iir_order=4, reference="none")
    peak = lambda ph: int(np.argmax(np.abs(core.apply_filter(raw, dict(base, phase=ph)).get_data()[0]))) - 2500
    assert abs(peak("zero")) <= 1                          # symmetric response centred on the event
    assert peak("causal") >= 5                             # causal filter responds only after it (delay)


def test_bad_settings_are_warned():
    w = core.filter_warnings({"l_freq": 30, "h_freq": 10, "notch": None, "method": "fir", "phase": "causal"}, 250)
    assert any("nothing would pass" in x for x in w)
    assert any("Causal" in x for x in w)
