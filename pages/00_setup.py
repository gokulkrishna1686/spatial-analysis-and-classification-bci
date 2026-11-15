"""Setup: a deliberately minimal stand-in for the other teams' modules
(data loading, filtering, epoching). Defaults load automatically, so this page is optional."""
import copy

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import welch

import core
import ui
import viz

o = ui.page_objects()
slot = ui.header_row("00", "Setup", "Data and preprocessing",
                     "Optional. A sample recording is already loaded with sensible preprocessing. "
                     "Change the dataset or the basic settings here; the full preprocessing modules are built by the other teams.")

with ui.guard("Setup"):
    P = ui.params()
    keys = list(ui.DATASETS)
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        ds = st.selectbox("Dataset", keys, index=keys.index(ui.current_dataset_key()) if ui.current_dataset_key() in keys else 0,
                          format_func=ui.DATASETS.get, help="Synthetic data is instant. Real datasets download once (internet needed).")
        if ds != ui.current_dataset_key() and st.button("Load this dataset", type="primary"):
            S = ui.S()
            S["history"].append(copy.deepcopy(P))
            S["params"] = ui.dataset_params(ds, P)
            st.rerun()
        up = st.file_uploader("Or upload a recording (EDF, BDF, GDF, FIF, SET, VHDR, CSV, NPY)", accept_multiple_files=True,
                              type=["edf", "bdf", "gdf", "fif", "set", "fdt", "vhdr", "eeg", "vmrk", "csv", "npy"])
        if up and st.button("Use uploaded file"):
            p = core.default_params(P.get("seed", 42))
            p["data"].update(source="upload", path=ui.save_uploads(up))
            p["filter"]["notch"] = None
            ui.S()["history"].append(copy.deepcopy(P))
            ui.S()["params"] = p
            st.rerun()
    fp, ep = dict(P["filter"] or {}), dict(P["epochs"])
    with c2:
        a, b, c = st.columns(3)
        lo = a.number_input("High-pass (Hz)", 0.0, 20.0, float(fp.get("l_freq") or 0.0), 0.5)
        hi = b.number_input("Low-pass (Hz)", 5.0, 120.0, float(fp.get("h_freq") or 40.0), 1.0)
        notch = c.selectbox("Notch", [None, 50.0, 60.0], index=[None, 50.0, 60.0].index(fp.get("notch")) if fp.get("notch") in (None, 50.0, 60.0) else 0,
                            format_func=lambda v: "off" if v is None else f"{v:.0f} Hz")
        d, e = st.columns(2)
        tmin = d.number_input("Epoch start (s)", -2.0, 0.0, float(ep["tmin"]), 0.1)
        tmax = e.number_input("Epoch end (s)", 0.5, 8.0, float(ep["tmax"]), 0.1)
    new_f = dict(fp, l_freq=lo or None, h_freq=hi, notch=notch)
    new_e = dict(ep, tmin=tmin, tmax=tmax)
    if new_e.get("baseline"):
        new_e["baseline"] = [tmin, min(0.0, new_e["baseline"][1])]

    ept = o["epochs"]
    chs = [c_ for c_ in ("C3", "C4") if c_ in ept.ch_names] or ept.ch_names[:2]
    yv = ept.metadata.label.to_numpy()
    fig = go.Figure()
    post = ept.times > 0.5
    for c_ in ui.classes_of(ept):
        for k, ch in enumerate(chs):
            x = ept[c_].get_data(picks=[ch])[:, 0][:, post]
            f, p = welch(x, ept.info["sfreq"], nperseg=min(x.shape[1], int(ept.info["sfreq"])))
            keep = (f >= 1) & (f <= 45)
            f, p = f[keep], p[:, keep]
            fig.add_trace(go.Scatter(x=f, y=10 * np.log10(p.mean(0) * 1e12), name=f"{viz.pretty(c_)} · {ch}",
                                     line=dict(color=viz.class_color(c_, ui.classes_of(ept)), width=2, dash="solid" if k == 0 else "dot")))
    fig.update_layout(height=300, xaxis_title="frequency (Hz)", yaxis_title="power (dB)", xaxis_range=[1, 45])
    ui.kicker(f"Quick look: power spectrum per class at {', '.join(chs)} ({len(ept)} trials)")
    ui.plot(fig, "setup_psd")
    ui.html(f'<div class="figcap">{o["label"]} · {len(ept)} trials · {len(ept.ch_names)} channels · {ept.info["sfreq"]:.0f} Hz</div>')

    def _apply():
        ui.apply_many({"filter": new_f, "epochs": new_e}, "Preprocessing applied")
    ui.apply_control(slot, "filter", {"filter": new_f, "epochs": new_e}, "Apply preprocessing", key="setup_apply", on_apply=_apply,
                     current={"filter": P["filter"], "epochs": P["epochs"]})
