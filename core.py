"""Core signal-processing steps: data loading, synthetic EEG, quality check,
filtering, ICA, epoching and the `build()` function that turns a parameter
dictionary into pipeline objects.

No Streamlit / Plotly in here: this file is embedded verbatim into the
auto-generated standalone script, so the app and the script share one code path.
"""
import copy
import json
import os

import numpy as np
import pandas as pd
import mne
from scipy import signal as sps

mne.set_log_level("ERROR")

# --------------------------------------------------------------------------
# Default parameters (the whole pipeline is described by this one dict)
# --------------------------------------------------------------------------

def default_params(seed=42):
    return {
        "version": 1,
        "seed": int(seed),
        "data": {"source": "synthetic", "paradigm": "MI", "n_classes": 2,
                 "trials_per_class": 40, "n_sessions": 2, "n_subjects": 1,
                 "erd": 0.25, "noise": 1.0, "line_noise": True, "blinks": True,
                 "subjects": [1], "path": None, "sfreq_csv": 250.0},
        "filter": {"l_freq": 1.0, "h_freq": 40.0, "notch": 50.0, "method": "fir",
                   "phase": "zero", "iir_order": 4, "reference": "average"},
        "ica": None,
        "epochs": {"events": None, "tmin": -1.0, "tmax": 4.0,
                   "baseline": [-1.0, 0.0], "reject_uv": None},
        "spatial": {"method": "CSP", "n_components": 6, "cov": "ledoit_wolf",
                    "band": [8.0, 30.0], "window": [0.5, 2.5]},
        "features": {"kinds": ["logvar", "bandpower"], "bands": [[8, 12], [12, 16], [16, 24], [24, 30]],
                     "band": [8.0, 30.0], "window": [0.5, 2.5], "ar_order": 4,
                     "erp_windows": [[0.25, 0.35], [0.35, 0.45], [0.45, 0.55]]},
        "classify": {"feature": "csp", "clf": "lda", "clf_params": {},
                     "cv": {"scheme": "kfold", "k": 5, "repeats": 1, "train": None, "test": None},
                     "nested": False, "n_perm": 100, "trial_len": 4.0, "alpha": 0.05},
        "applied": [],
    }


STAGES = ["data", "filter", "ica", "epochs"]

# --------------------------------------------------------------------------
# Synthetic EEG generator (instant, offline, with known ground truth)
# --------------------------------------------------------------------------

SYN_CHANNELS = ["Fp1", "Fp2", "F3", "Fz", "F4", "FC3", "FC1", "FC2", "FC4",
                "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                "CP3", "CP1", "CP2", "CP4", "P3", "Pz", "P4", "Oz"]


def montage_xyz(ch_names):
    mont = mne.channels.make_standard_montage("standard_1005")
    pos = {k.lower(): v for k, v in mont.get_positions()["ch_pos"].items()}
    return np.array([pos[c.lower()] for c in ch_names])


def _blob(xyz, center_ch, ch_names, width=0.032):
    c = xyz[[n.lower() for n in ch_names].index(center_ch.lower())]
    d = np.linalg.norm(xyz - c, axis=1)
    return np.exp(-0.5 * (d / width) ** 2)


def _pink(n_ch, n_s, rng):
    spec = np.fft.rfft(rng.standard_normal((n_ch, n_s)), axis=1)
    f = np.arange(spec.shape[1], dtype=float)
    f[0] = 1.0
    spec /= np.sqrt(f)
    x = np.fft.irfft(spec, n=n_s, axis=1)
    return x / x.std(axis=1, keepdims=True)


def _band_noise(n_s, sfreq, lo, hi, rng):
    sos = sps.butter(4, [lo, hi], btype="band", fs=sfreq, output="sos")
    x = sps.sosfiltfilt(sos, rng.standard_normal(n_s))
    return x / x.std()


def make_synthetic(paradigm="MI", n_classes=2, trials_per_class=40, n_sessions=2,
                   n_subjects=1, erd=0.45, noise=1.0, line_noise=True, blinks=True,
                   sfreq=250.0, seed=42):
    """Simulate a multi-session EEG recording.

    MI: mu/beta sources over C3 / C4 / Cz desynchronise (ERD) during imagery of
    the contralateral hand (or feet). P300: a centro-parietal positivity ~350 ms
    after rare 'target' stimuli. Blinks, an ECG lead and 50 Hz line noise are
    added so the artifact tools have something to find.
    Returns (raw, segments, true_patterns).
    """
    rng = np.random.RandomState(seed)
    ch = list(SYN_CHANNELS)
    xyz = montage_xyz(ch)
    n_ch = len(ch)
    classes_mi = ["left_hand", "right_hand", "feet"][:max(2, min(3, n_classes))]
    centers = {"left_hand": "C4", "right_hand": "C3", "feet": "Cz"}
    blink_pat = 0.5 * (_blob(xyz, "Fp1", ch, 0.05) + _blob(xyz, "Fp2", ch, 0.05))
    ecg_pat = np.linspace(1.0, 0.2, n_ch) * (xyz[:, 0] < 0.0) + 0.15
    blocks, segments, ann_on, ann_desc, ann_dur = [], [], [], [], []
    true_patterns = {}
    offset = 0
    for subj in range(1, n_subjects + 1):
        subj_shift = rng.normal(0, 0.004, 3)
        erd_s = float(np.clip(erd * rng.uniform(0.8, 1.2) if subj > 1 else erd, 0, 0.95))
        for sess in range(1, n_sessions + 1):
            # electrode cap shifts a bit between sessions -> non-stationarity
            xyz_s = xyz + subj_shift + rng.normal(0, 0.002 * (sess - 1), 3)
            if paradigm == "P300":
                n_t = trials_per_class
                n_n = 3 * trials_per_class
                labels = np.array(["target"] * n_t + ["nontarget"] * n_n)
                rng.shuffle(labels)
                soa = 0.8
                n_s = int((len(labels) * soa + 3.0) * sfreq)
                onsets = 1.5 + np.arange(len(labels)) * soa + rng.uniform(0, 0.1, len(labels))
            else:
                labels = np.repeat(classes_mi, trials_per_class)
                rng.shuffle(labels)
                trial = 7.0
                n_s = int((len(labels) * trial + 2.0) * sfreq)
                onsets = 2.0 + np.arange(len(labels)) * trial + rng.uniform(-0.2, 0.2, len(labels))
            t = np.arange(n_s) / sfreq
            X = 6.0 * noise * _pink(n_ch, n_s, rng)                       # sensor noise
            for _ in range(4):                                           # broad background sources
                c = xyz_s[rng.randint(n_ch)]
                pat = np.exp(-0.5 * (np.linalg.norm(xyz_s - c, axis=1) / 0.06) ** 2)
                X += 5.0 * noise * np.outer(pat, _pink(1, n_s, rng)[0])
            alpha = _band_noise(n_s, sfreq, 9, 11, rng) * (1 + 0.3 * np.sin(2 * np.pi * 0.05 * t))
            X += 7.0 * np.outer(_blob(xyz_s, "Oz", ch, 0.045), alpha)
            if paradigm != "P300":
                # widespread, task-unrelated central rhythm (volume conduction): it swamps
                # single electrodes, but a spatial filter can subtract it
                for _ in range(2):
                    amp_env = 1 + 0.6 * _band_noise(n_s, sfreq, 0.05, 0.4, rng)
                    X += 9.0 * np.outer(_blob(xyz_s, "Cz", ch, 0.075), amp_env * _band_noise(n_s, sfreq, 8.5, 13.5, rng))
            if paradigm == "P300":
                p3 = _blob(xyz_s, "Pz", ch, 0.045)
                n2 = _blob(xyz_s, "Fz", ch, 0.045)
                vis = _blob(xyz_s, "Oz", ch, 0.04)
                true_patterns["target"] = p3
                for on, lab in zip(onsets, labels):
                    i0 = int(on * sfreq)
                    tt = np.arange(int(0.8 * sfreq)) / sfreq
                    w_vis = 3.0 * np.exp(-0.5 * ((tt - 0.12) / 0.03) ** 2)
                    X[:, i0:i0 + len(tt)] += np.outer(vis, w_vis)
                    if lab == "target":
                        lat = 0.35 + rng.normal(0, 0.03)
                        w = (9.0 * erd / 0.45) * np.exp(-0.5 * ((tt - lat) / 0.07) ** 2)
                        w2 = -3.0 * np.exp(-0.5 * ((tt - 0.22) / 0.03) ** 2)
                        X[:, i0:i0 + len(tt)] += np.outer(p3, w) + np.outer(n2, w2)
                    ann_on.append(offset / sfreq + on)
                    ann_desc.append(str(lab))
                    ann_dur.append(0.0)
            else:
                env = {c: np.ones(n_s) for c in classes_mi}
                ramp = int(0.4 * sfreq)
                win = np.concatenate([np.linspace(0, 1, ramp), np.ones(int(3.0 * sfreq)), np.linspace(1, 0, ramp)])
                for on, lab in zip(onsets, labels):
                    i0 = int((on + 0.3) * sfreq)
                    seg = slice(i0, i0 + len(win))
                    for c in classes_mi:
                        if c == lab:
                            env[c][seg] -= erd_s * win              # ERD: contralateral power drops
                        else:
                            env[c][seg] += 0.12 * erd_s * win       # mild ERS elsewhere
                    ann_on.append(offset / sfreq + on)
                    ann_desc.append(str(lab))
                    ann_dur.append(4.0)
                for c in classes_mi:
                    pat = _blob(xyz_s, centers[c], ch)
                    true_patterns[c] = pat
                    mu = _band_noise(n_s, sfreq, 9.5, 12.5, rng) + 0.5 * _band_noise(n_s, sfreq, 18, 24, rng)
                    X += 11.0 * np.outer(pat, env[c] * mu)
            if blinks:
                n_b = int(n_s / sfreq * 0.2)
                for b in rng.randint(0, n_s - int(sfreq), n_b):
                    tt = np.arange(int(0.4 * sfreq)) / sfreq
                    w = 100.0 * np.exp(-0.5 * ((tt - 0.15) / 0.045) ** 2)     # ~100 µV, ~250 ms blink
                    X[:, b:b + len(tt)] += np.outer(blink_pat, w)
            # heart: QRS spikes, mostly in the ECG lead, slightly leaking into EEG
            ecg = np.zeros(n_s)
            beat = 0.5
            while beat < t[-1] - 1:
                i = int(beat * sfreq)
                tt = np.arange(int(0.1 * sfreq)) / sfreq
                ecg[i:i + len(tt)] += np.exp(-0.5 * ((tt - 0.04) / 0.008) ** 2) - 0.25 * np.exp(-0.5 * ((tt - 0.06) / 0.01) ** 2)
                beat += 60.0 / 70 + rng.normal(0, 0.03)
            X += 7.0 * np.outer(ecg_pat, ecg)
            if line_noise:
                X += 4.0 * np.outer(rng.uniform(0.3, 1.0, n_ch), np.sin(2 * np.pi * 50 * t))
            X *= (1.0 + 0.1 * (sess - 1))                                  # session gain change
            ecg_lead = 800.0 * ecg + 5 * rng.standard_normal(n_s)
            blocks.append(np.vstack([X, ecg_lead[None]]) * 1e-6)
            segments.append([offset, offset + n_s, subj, sess])
            offset += n_s
    data = np.concatenate(blocks, axis=1)
    info = mne.create_info(ch + ["ECG"], sfreq, ["eeg"] * n_ch + ["ecg"])
    raw = mne.io.RawArray(data, info)
    raw.set_montage("standard_1005")
    raw.set_annotations(mne.Annotations(ann_on, ann_dur, ann_desc))
    tp = {k: (v / np.linalg.norm(v)).tolist() for k, v in true_patterns.items()}
    return raw, segments, tp


# --------------------------------------------------------------------------
# Real datasets and uploaded files
# --------------------------------------------------------------------------

def _segments_from_raws(raws, subj_sess):
    segs, off = [], 0
    for r, (s, ss) in zip(raws, subj_sess):
        segs.append([off, off + r.n_times, s, ss])
        off += r.n_times
    return segs


def load_eegbci(subjects=(1,)):
    from mne.datasets import eegbci
    raws, keys = [], []
    for s in subjects:
        files = eegbci.load_data(s, [4, 8, 12], update_path=True, verbose="ERROR")
        for run_i, f in enumerate(files, start=1):
            r = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
            eegbci.standardize(r)
            r.set_montage("standard_1005", on_missing="ignore")
            r.annotations.rename({"T0": "rest", "T1": "left_hand", "T2": "right_hand"})
            raws.append(r)
            keys.append((int(s), run_i))
    segs = _segments_from_raws(raws, keys)
    raw = mne.concatenate_raws([r.copy() for r in raws])
    return raw, segs


def load_bnci2014_001(subjects=(1,)):
    from moabb.datasets import BNCI2014_001
    ds = BNCI2014_001()
    data = ds.get_data(subjects=list(subjects))
    inv = {v: k for k, v in ds.event_id.items()}
    raws, keys = [], []
    for s in subjects:
        for si, (sess_name, runs) in enumerate(sorted(data[s].items()), start=1):
            for run_name, r in sorted(runs.items()):
                r = r.copy().load_data()
                stim = [r.ch_names[i] for i in mne.pick_types(r.info, stim=True)]
                if not set(inv.values()) & set(r.annotations.description):
                    # older MOABB versions: events only in the stim channel
                    ev = mne.find_events(r, stim_channel=stim[0], verbose="ERROR")
                    ann = mne.annotations_from_events(ev, r.info["sfreq"], event_desc=inv,
                                                      first_samp=r.first_samp)
                    ann.duration[:] = 4.0
                    r.set_annotations(ann)
                if stim:
                    r.drop_channels(stim)
                if r.get_montage() is None:
                    r.set_montage("standard_1005", on_missing="ignore")
                raws.append(r)
                keys.append((int(s), si))
    segs = _segments_from_raws(raws, keys)
    return mne.concatenate_raws(raws), segs


def load_mne_sample():
    path = mne.datasets.sample.data_path(verbose="ERROR")
    f = os.path.join(str(path), "MEG", "sample", "sample_audvis_raw.fif")
    raw = mne.io.read_raw_fif(f, preload=True, verbose="ERROR")
    ev = mne.find_events(raw, stim_channel="STI 014", verbose="ERROR")
    desc = {1: "auditory/left", 2: "auditory/right", 3: "visual/left", 4: "visual/right", 5: "smiley", 32: "button"}
    raw.set_annotations(mne.annotations_from_events(ev, raw.info["sfreq"], event_desc=desc,
                                                    first_samp=raw.first_samp))
    raw.pick(["eeg", "eog"]).load_data()
    raw.info["bads"] = []
    raw.resample(150, verbose="ERROR")
    return raw, [[0, raw.n_times, 1, 1]]


def read_any(path, sfreq_csv=250.0):
    """Read an uploaded recording. Supports EDF/BDF/GDF/FIF/SET/VHDR/CSV/NPY."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".edf",):
        raw = mne.io.read_raw_edf(path, preload=True)
    elif ext == ".bdf":
        raw = mne.io.read_raw_bdf(path, preload=True)
    elif ext == ".gdf":
        raw = mne.io.read_raw_gdf(path, preload=True)
    elif ext == ".fif" or path.endswith(".fif.gz"):
        raw = mne.io.read_raw_fif(path, preload=True)
    elif ext == ".set":
        raw = mne.io.read_raw_eeglab(path, preload=True)
    elif ext == ".vhdr":
        raw = mne.io.read_raw_brainvision(path, preload=True)
    elif ext in (".csv", ".npy"):
        if ext == ".csv":
            df = pd.read_csv(path)
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] == 0:
                raise ValueError("The CSV has no numeric columns.")
            data, names = num.to_numpy().T, [str(c) for c in num.columns]
        else:
            data = np.load(path, allow_pickle=False)
            if data.ndim != 2:
                raise ValueError("The .npy file must be a 2-D array (channels x samples).")
            if data.shape[0] > data.shape[1]:
                data = data.T
            names = [f"EEG{i + 1:03d}" for i in range(data.shape[0])]
        data = np.asarray(data, float)
        if np.nanmedian(np.abs(data)) > 1e-2:   # looks like microvolts -> convert
            data = data * 1e-6
        raw = mne.io.RawArray(data, mne.create_info(names, float(sfreq_csv), "eeg"))
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use EDF, BDF, GDF, FIF, SET, VHDR, CSV or NPY.")
    if len(raw.annotations) == 0:
        stim = mne.pick_types(raw.info, stim=True)
        if len(stim):
            ev = mne.find_events(raw, stim_channel=raw.ch_names[stim[0]], verbose="ERROR")
            if len(ev):
                raw.set_annotations(mne.annotations_from_events(ev, raw.info["sfreq"], first_samp=raw.first_samp))
    if raw.get_montage() is None:
        try:
            raw.set_montage("standard_1005", on_missing="ignore", match_case=False)
        except Exception:
            pass
    return raw


def load_data(dp, seed=42):
    """Returns dict(raw, segments, true_patterns, label)."""
    src = dp.get("source", "synthetic")
    tp = None
    if src == "synthetic":
        raw, segs, tp = make_synthetic(dp.get("paradigm", "MI"), dp.get("n_classes", 2),
                                       dp.get("trials_per_class", 40), dp.get("n_sessions", 2),
                                       dp.get("n_subjects", 1), dp.get("erd", 0.45), dp.get("noise", 1.0),
                                       dp.get("line_noise", True), dp.get("blinks", True), seed=seed)
        label = f"Synthetic {dp.get('paradigm', 'MI')} · {dp.get('n_subjects', 1)} subj × {dp.get('n_sessions', 2)} sess"
    elif src == "eegbci":
        raw, segs = load_eegbci(dp.get("subjects", [1]))
        label = f"PhysioNet EEGBCI MI · subj {dp.get('subjects', [1])}"
    elif src == "bnci2014_001":
        raw, segs = load_bnci2014_001(dp.get("subjects", [1]))
        label = f"BCI Comp IV 2a · subj {dp.get('subjects', [1])}"
    elif src == "mne_sample":
        raw, segs = load_mne_sample()
        label = "MNE sample (auditory/visual EEG)"
    elif src == "upload":
        raw = read_any(dp["path"], dp.get("sfreq_csv", 250.0))
        segs = [[0, raw.n_times, 1, 1]]
        label = f"Upload · {os.path.basename(dp['path'])}"
    else:
        raise ValueError(f"Unknown data source {src}")
    bads = [b for b in (dp.get("bads") or []) if b in raw.ch_names]
    raw.info["bads"] = bads
    return {"raw": raw, "segments": segs, "true_patterns": tp, "label": label}


# --------------------------------------------------------------------------
# Quality check
# --------------------------------------------------------------------------

def quality_check(raw):
    picks = mne.pick_types(raw.info, eeg=True)
    x = raw.get_data(picks=picks)
    names = [raw.ch_names[i] for i in picks]
    sd = x.std(axis=1)
    lsd = np.log10(sd + 1e-20)
    med = np.median(lsd)
    mad = np.median(np.abs(lsd - med)) + 1e-9
    z = (lsd - med) / (1.4826 * mad)
    rows = []
    for n, s, zz in zip(names, sd, z):
        issue = ""
        if s < 1e-7:
            issue = "flat"
        elif zz > 3.5:
            issue = "noisy"
        rows.append({"channel": n, "std_uV": s * 1e6, "robust_z": zz, "issue": issue})
    df = pd.DataFrame(rows)
    notes = []
    msd = np.median(sd)
    if msd > 1e-3:
        notes.append(f"Median channel SD is {msd:.3g} V — that is huge for EEG. The file is probably in µV but labeled as V; divide by 1e6.")
    elif msd < 1e-8:
        notes.append(f"Median channel SD is {msd:.3g} V — suspiciously small. Check units (nV?).")
    has_pos = any(np.isfinite(raw.info["chs"][i]["loc"][:3]).all() and np.abs(raw.info["chs"][i]["loc"][:3]).sum() > 0 for i in picks)
    if not has_pos:
        notes.append("No electrode positions (montage) found — topomaps and interpolation are unavailable. Rename channels to 10-20 names or set a montage.")
    if len(raw.annotations) == 0:
        notes.append("No events/annotations found — epoching and classification need event markers.")
    try:
        f, p = sps.welch(x, raw.info["sfreq"], nperseg=int(min(x.shape[1], 2 * raw.info["sfreq"])))
        for ln in (50, 60):
            if ln < raw.info["sfreq"] / 2 - 2:
                i = np.argmin(np.abs(f - ln))
                ref = np.median(p[:, max(0, i - 6):i - 2], axis=1)
                if np.median(p[:, i] / (ref + 1e-30)) > 10:
                    notes.append(f"Strong {ln} Hz line noise — use a notch filter in step 02.")
    except Exception:
        pass
    return df, notes


# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------

def filter_warnings(fp, sfreq):
    w = []
    lo, hi, notch = fp.get("l_freq"), fp.get("h_freq"), fp.get("notch")
    nyq = sfreq / 2
    if lo and hi and lo >= hi:
        w.append("High-pass cutoff is above the low-pass cutoff: nothing would pass.")
    if hi and hi >= nyq:
        w.append(f"Low-pass {hi} Hz is at/above Nyquist ({nyq:.0f} Hz) — it will be ignored.")
    if notch and notch >= nyq:
        w.append("Notch frequency is above Nyquist.")
    if lo and lo > 2:
        w.append("High-pass above 2 Hz distorts slow ERPs (P300) — fine for motor imagery, risky for ERP work.")
    if lo is not None and 0 < lo < 0.1 and fp.get("method") == "fir":
        w.append("Very low high-pass with FIR needs a very long filter — slow and long edge effects.")
    if fp.get("phase") != "zero":
        w.append("Causal filtering shifts latencies (phase delay). Good for simulating online BCIs, bad for ERP latency analysis.")
    if fp.get("method") == "iir" and fp.get("iir_order", 4) > 6:
        w.append("High IIR orders can become numerically unstable / ring.")
    return w


def apply_filter(raw, fp):
    raw = raw.copy()
    sfreq = raw.info["sfreq"]
    lo = fp.get("l_freq") or None
    hi = fp.get("h_freq") or None
    if hi is not None and hi >= sfreq / 2:
        hi = None
    iir = dict(order=int(fp.get("iir_order", 4)), ftype="butter", output="sos")
    method = fp.get("method", "fir")
    if method == "fir":
        phase = "zero" if fp.get("phase") == "zero" else "minimum"
    else:
        phase = "zero" if fp.get("phase") == "zero" else "forward"
    picks = mne.pick_types(raw.info, eeg=True, eog=True)
    if fp.get("notch") and fp["notch"] < sfreq / 2:
        freqs = np.arange(fp["notch"], sfreq / 2 - 1, fp["notch"])
        if method == "fir":
            raw.notch_filter(freqs, picks=picks, method="fir", phase=phase)
        else:                                   # MNE's IIR notch takes one stop-band at a time
            for f in freqs:
                raw.notch_filter(f, picks=picks, method="iir", iir_params=iir, phase=phase)
    if lo or hi:
        raw.filter(lo, hi, picks=picks, method=method, phase=phase,
                   iir_params=iir if method == "iir" else None)
    ref = fp.get("reference", "none")
    if ref == "average":
        raw.set_eeg_reference("average", projection=False)
    elif ref and ref not in ("none", "average") and ref in raw.ch_names:
        raw.set_eeg_reference([ref], projection=False)
    return raw


def filter_response(fp, sfreq, n=1024):
    """Frequency/phase response of the band-pass part (for plotting)."""
    lo = fp.get("l_freq") or None
    hi = fp.get("h_freq") or None
    if hi is not None and hi >= sfreq / 2:
        hi = None
    if not lo and not hi:
        f = np.linspace(0, sfreq / 2, n)
        return f, np.zeros(n), np.zeros(n), None
    method = fp.get("method", "fir")
    if method == "fir":
        phase = "zero" if fp.get("phase") == "zero" else "minimum"
        h = mne.filter.create_filter(None, sfreq, lo, hi, method="fir", phase=phase, verbose="ERROR")
        w, H = sps.freqz(h, worN=n, fs=sfreq)
        imp = h
    else:
        p = mne.filter.create_filter(None, sfreq, lo, hi, method="iir",
                                     iir_params=dict(order=int(fp.get("iir_order", 4)), ftype="butter", output="sos"),
                                     verbose="ERROR")
        w, H = sps.sosfreqz(p["sos"], worN=n, fs=sfreq)
        if fp.get("phase") == "zero":
            H = np.abs(H) ** 2 + 0j          # filtfilt: squared magnitude, zero phase
        imp = sps.sosfilt(p["sos"], np.r_[1.0, np.zeros(int(sfreq))])
    mag = 20 * np.log10(np.abs(H) + 1e-12)
    ph = np.unwrap(np.angle(H))
    return w, mag, ph, imp


# --------------------------------------------------------------------------
# Artifacts & ICA
# --------------------------------------------------------------------------

def fit_ica(raw, ip, seed=42):
    raw = raw.copy()
    bads = sorted(set(raw.info["bads"]) | set(b for b in (ip.get("bads") or []) if b in raw.ch_names))
    raw.info["bads"] = bads
    if bads and ip.get("interpolate", True):
        raw.interpolate_bads(reset_bads=True)
    fit_raw = raw.copy()
    if (raw.info["highpass"] or 0) < 1.0:
        fit_raw.filter(1.0, None, picks="eeg", verbose="ERROR")
    n_eeg = len(mne.pick_types(raw.info, eeg=True))
    nc = int(min(ip.get("n_components", 15), n_eeg - 1))
    method = ip.get("method", "fastica")
    fit_params = dict(extended=True) if method == "infomax" else None
    ica = mne.preprocessing.ICA(n_components=nc, method=method, random_state=seed,
                                max_iter=int(ip.get("max_iter", 500)), fit_params=fit_params)
    reject = dict(eeg=ip["reject_uv"] * 1e-6) if ip.get("reject_uv") else None
    ica.fit(fit_raw, picks="eeg", reject=reject, decim=int(ip.get("decim", 2)))
    labels = auto_label_ica(ica, fit_raw, ip)
    return raw, ica, labels


def auto_label_ica(ica, raw, ip):
    """Label components: 'eye' if the component's time course correlates with a
    frontal/EOG channel in the blink band (1-5 Hz, below the mu rhythm) with
    |r| > eog_thr; 'heart' via MNE's ECG detector (z-score > ecg_thr);
    'muscle' via MNE's high-frequency-slope detector."""
    labels = {}
    eog_chs = [raw.ch_names[i] for i in mne.pick_types(raw.info, eog=True)] or         [c for c in ("Fp1", "Fp2", "Fpz", "AF7", "AF8") if c in raw.ch_names]
    try:
        if eog_chs:
            lo = raw.copy().filter(1.0, 5.0, picks="all", verbose="ERROR")
            src = ica.get_sources(lo).get_data()
            ref = lo.get_data(picks=eog_chs)
            real_eog = bool(mne.pick_types(raw.info, eog=True).size)
            maps = np.abs(ica.get_components())
            front = np.array([n.upper().startswith(("FP", "AF")) for n in ica.ch_names])
            for k in range(src.shape[0]):
                r = max(abs(np.corrcoef(src[k], e)[0, 1]) for e in ref)
                # with EEG proxies (no EOG channel) also demand a frontal topography: after an
                # average reference every big source leaks into Fp1/Fp2
                frontal = (not real_eog and front.any() and (~front).any()
                           and maps[front, k].mean() >= 1.5 * maps[~front, k].mean())
                if r > ip.get("eog_thr", 0.5) and (real_eog or frontal):
                    labels[int(k)] = "eye"
    except Exception:
        pass
    try:
        if len(mne.pick_types(raw.info, ecg=True)):
            idx, _ = ica.find_bads_ecg(raw, method="correlation", threshold=ip.get("ecg_thr", 3.0))
            for i in idx:
                labels.setdefault(int(i), "heart")
    except Exception:
        pass
    try:
        idx, _ = ica.find_bads_muscle(raw, threshold=ip.get("muscle_thr", 0.8))
        for i in idx:
            labels.setdefault(int(i), "muscle")
    except Exception:
        pass
    return labels


def apply_ica(raw, ica, exclude):
    out = raw.copy()
    ica.apply(out, exclude=list(exclude))
    return out


# --------------------------------------------------------------------------
# Epoching
# --------------------------------------------------------------------------

def event_names(raw):
    return sorted(set(raw.annotations.description) - {"BAD boundary", "EDGE boundary", "rest", "button", "smiley"})


def make_epochs(raw, ep, segments):
    names = ep.get("events") or event_names(raw)
    names = [n for n in names if n in set(raw.annotations.description)]
    if not names:
        raise ValueError("No usable event markers found in this recording.")
    event_id = {n: i + 1 for i, n in enumerate(sorted(names))}
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose="ERROR")
    bl = tuple(ep["baseline"]) if ep.get("baseline") else None
    if bl is not None:
        bl = (max(bl[0], ep["tmin"]), min(bl[1], ep["tmax"]))
    reject = dict(eeg=ep["reject_uv"] * 1e-6) if ep.get("reject_uv") else None
    epochs = mne.Epochs(raw, events, event_id, tmin=ep["tmin"], tmax=ep["tmax"], baseline=bl,
                        picks="eeg", reject=reject, preload=True, reject_by_annotation=False, verbose="ERROR")
    samp = epochs.events[:, 0] - raw.first_samp
    subj = np.ones(len(samp), int)
    sess = np.ones(len(samp), int)
    for a, b, s, ss in segments:
        m = (samp >= a) & (samp < b)
        subj[m] = s
        sess[m] = ss
    inv = {v: k for k, v in event_id.items()}
    epochs.metadata = pd.DataFrame({"label": [inv[c] for c in epochs.events[:, 2]],
                                    "subject": subj, "session": sess,
                                    "onset_s": samp / raw.info["sfreq"]})
    return epochs


# --------------------------------------------------------------------------
# Build the pipeline from parameters (with a simple stage cache)
# --------------------------------------------------------------------------

def _key(params, upto, drop_exclude=False):
    d = {k: copy.deepcopy(params.get(k)) for k in STAGES[:STAGES.index(upto) + 1]}
    if drop_exclude and d.get("ica"):
        d["ica"].pop("exclude", None)
    d["seed"] = params.get("seed")
    return upto + json.dumps(d, sort_keys=True, default=str)


def build(params, cache=None, progress=None):
    """Run data -> filter -> ICA -> epochs. Returns a dict of objects."""
    cache = {} if cache is None else cache
    seed = int(params.get("seed", 42))
    out = {}

    def stage(name, fn, key=None):
        k = key or _key(params, name)
        if k not in cache:
            if progress:
                progress(name)
            cache[k] = fn()
        return cache[k]

    d = stage("data", lambda: load_data(params["data"], seed))
    out.update(raw0=d["raw"], segments=d["segments"], true_patterns=d["true_patterns"], label=d["label"])
    raw = d["raw"]
    if params.get("filter"):
        raw = stage("filter", lambda: apply_filter(d["raw"], params["filter"]))
    out["raw_f"] = raw
    out["ica"], out["ica_labels"] = None, {}
    if params.get("ica"):
        base = raw
        r_int, ica, labels = stage("ica", lambda: fit_ica(base, params["ica"], seed),
                                   key=_key(params, "ica", drop_exclude=True))
        excl = params["ica"].get("exclude")
        if excl is None:
            excl = sorted(labels)
        raw = stage("ica", lambda: apply_ica(r_int, ica, excl), key="apply" + _key(params, "ica"))
        out["ica"], out["ica_labels"], out["ica_exclude"] = ica, labels, excl
    out["raw"] = raw
    out["epochs"] = stage("epochs", lambda: make_epochs(raw, params["epochs"], d["segments"]))
    # keep the cache small
    while len(cache) > 12:
        cache.pop(next(iter(cache)))
    return out


def clone_params(p):
    return copy.deepcopy(p)
