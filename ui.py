"""Shared UI: pipeline state (apply / undo / reset), chapter headers, explain
panels, export drawers, plot wrapper, friendly error guard, data source popover."""
import copy
import hashlib
import json
import os
import re
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import streamlit as st

import core
import viz

ROOT = Path(__file__).parent
UPLOADS = ROOT / "uploads"

# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

def S():
    if "P" not in st.session_state:
        st.session_state["P"] = {"params": core.default_params(), "history": [], "cache": {}, "runs": [],
                                 "figs": {}, "results": {}}
    return st.session_state["P"]


def params():
    return S()["params"]


def phash(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def objects():
    """Build (cached) pipeline objects for the current params. Falls back to the
    previous working configuration with a friendly message if loading fails."""
    s = S()
    try:
        with st.spinner("Preparing the pipeline…"):
            return core.build(s["params"], s["cache"])
    except Exception as e:
        bad = s["params"]
        if s["history"]:
            s["params"] = s["history"].pop()
        else:
            s["params"] = core.default_params(bad.get("seed", 42))
        st.session_state["_load_error"] = friendly_message(e)
        return core.build(s["params"], s["cache"])


def show_load_error():
    msg = st.session_state.pop("_load_error", None)
    if msg:
        st.markdown(f'<div class="note v"><b>That did not work —</b> {msg} The previous configuration was restored.</div>',
                    unsafe_allow_html=True)


def apply(step, value, msg=None):
    s = S()
    s["history"].append(copy.deepcopy(s["params"]))
    s["history"] = s["history"][-25:]
    s["params"][step] = copy.deepcopy(value)
    if step not in s["params"]["applied"]:
        s["params"]["applied"].append(step)
    st.toast(msg or f"Applied · {step}")


def apply_many(updates, msg=None):
    s = S()
    s["history"].append(copy.deepcopy(s["params"]))
    for k, v in updates.items():
        s["params"][k] = copy.deepcopy(v)
        if k in core.STAGES + ["spatial", "features", "classify"] and k not in s["params"]["applied"]:
            s["params"]["applied"].append(k)
    if msg:
        st.toast(msg)


def undo():
    s = S()
    if s["history"]:
        s["params"] = s["history"].pop()
        st.toast("Undid last step")


def reset():
    seed = params().get("seed", 42)
    st.session_state["P"] = {"params": core.default_params(seed), "history": [], "cache": {}, "runs": [],
                             "figs": {}, "results": {}}
    st.toast("Pipeline reset")


DATASETS = {
    "synthetic_mi": "Synthetic · motor imagery (instant)",
    "synthetic_mi3": "Synthetic · 3-class MI",
    "synthetic_p300": "Synthetic · P300 oddball",
    "eegbci": "PhysioNet EEGBCI · MI (download ≈ 8 MB)",
    "bnci2014_001": "BCI Competition IV 2a · MOABB (≈ 40 MB)",
    "mne_sample": "MNE sample · auditory/visual (≈ 1.5 GB!)",
}


def dataset_params(key, base):
    """Return a params dict configured for a dataset (sensible defaults downstream)."""
    p = core.default_params(base.get("seed", 42))
    d = p["data"]
    if key == "synthetic_mi3":
        d["n_classes"] = 3
    elif key == "synthetic_p300":
        d["paradigm"] = "P300"
    elif key in ("eegbci", "bnci2014_001", "mne_sample"):
        d["source"] = key
    erp = key in ("synthetic_p300", "mne_sample")
    if erp:
        p["epochs"].update(tmin=-0.2, tmax=0.8, baseline=[-0.2, 0.0])
        p["filter"].update(l_freq=0.5, h_freq=30.0)
        p["spatial"].update(method="xDAWN", band=[1.0, 20.0], window=[0.0, 0.8], n_components=4)
        p["features"].update(kinds=["erp"], band=[1.0, 20.0], window=[0.0, 0.8])
        p["classify"].update(feature="xdawn", clf="lr", band=[1.0, 20.0], window=[0.0, 0.8], n_components=4)
    if key == "eegbci":
        p["epochs"].update(tmin=-1.0, tmax=4.0, baseline=None)
        p["classify"]["cv"] = {"scheme": "kfold", "k": 5}
    if key == "bnci2014_001":
        p["epochs"].update(tmin=-0.5, tmax=4.0, baseline=None, events=["left_hand", "right_hand"])
    if key == "mne_sample":
        p["epochs"]["events"] = ["auditory/left", "visual/left"]
        p["filter"]["notch"] = 60.0
    if key in ("eegbci",):
        p["filter"]["notch"] = 60.0
    p["applied"] = ["data"]
    return p


def current_dataset_key():
    d = params()["data"]
    if d["source"] == "synthetic":
        if d.get("paradigm") == "P300":
            return "synthetic_p300"
        return "synthetic_mi3" if d.get("n_classes", 2) == 3 else "synthetic_mi"
    return d["source"]


def save_uploads(files):
    UPLOADS.mkdir(exist_ok=True)
    paths = []
    for f in files:
        p = UPLOADS / os.path.basename(f.name)
        p.write_bytes(f.getbuffer())
        paths.append(p)
    prio = [".vhdr", ".set", ".fif", ".edf", ".bdf", ".gdf", ".csv", ".npy"]
    paths.sort(key=lambda p: prio.index(p.suffix.lower()) if p.suffix.lower() in prio else 99)
    return str(paths[0]) if paths else None


def data_popover(key="dp"):
    """Compact 'sample data or your own file' control, available on every page."""
    label = st.session_state.get("_ds_label", "Data").split(" · ")[0]
    with st.popover(label, icon=":material/database:", use_container_width=True, help="Switch dataset or upload your own recording"):
        st.markdown('<div class="kicker">Built-in sample data</div>', unsafe_allow_html=True)
        cur = current_dataset_key()
        keys = list(DATASETS)
        choice = st.selectbox("Dataset", keys, index=keys.index(cur) if cur in keys else 0,
                              format_func=DATASETS.get, key=f"{key}_ds",
                              help="Synthetic data is generated instantly and has known ground truth. Real datasets download once to ~/mne_data.")
        subj = 1
        if choice in ("eegbci", "bnci2014_001"):
            subj = st.number_input("Subject", 1, 109 if choice == "eegbci" else 9, 1, key=f"{key}_subj")
        if st.button("Load sample", key=f"{key}_load", type="primary", use_container_width=True):
            p = dataset_params(choice, params())
            if choice in ("eegbci", "bnci2014_001"):
                p["data"]["subjects"] = [int(subj)]
            S()["history"].append(copy.deepcopy(params()))
            S()["params"] = p
            st.rerun()
        st.markdown('<div class="kicker" style="margin-top:10px">Or your own recording</div>', unsafe_allow_html=True)
        up = st.file_uploader("EDF · BDF · GDF · FIF · SET(+FDT) · VHDR(+EEG+VMRK) · CSV · NPY",
                              accept_multiple_files=True, key=f"{key}_up",
                              type=["edf", "bdf", "gdf", "fif", "set", "fdt", "vhdr", "eeg", "vmrk", "csv", "npy"])
        sf = st.number_input("Sampling rate (CSV / NPY only)", 1.0, 10000.0, 250.0, key=f"{key}_sf",
                             help="CSV/NPY files carry no sampling rate. CSV: one column per channel, header = channel names.")
        if up and st.button("Use uploaded file", key=f"{key}_use", use_container_width=True):
            path = save_uploads(up)
            p = core.default_params(params().get("seed", 42))
            p["data"].update(source="upload", path=path, sfreq_csv=float(sf))
            p["filter"]["notch"] = None
            p["applied"] = ["data"]
            S()["history"].append(copy.deepcopy(params()))
            S()["params"] = p
            st.rerun()


# --------------------------------------------------------------------------
# Visual building blocks
# --------------------------------------------------------------------------

def css():
    st.markdown("<style>" + (ROOT / "assets" / "style.css").read_text(encoding="utf-8") + "</style>",
                unsafe_allow_html=True)


def trace_svg(seed=0, color="#17191E", width=1200, height=26, accent=None, n=420):
    rng = np.random.RandomState(seed)
    x = np.cumsum(rng.standard_normal(n))
    x = x - np.convolve(x, np.ones(25) / 25, mode="same")
    t = np.arange(n)
    x += 2.2 * np.sin(2 * np.pi * t / 18.0) * (0.4 + 0.6 * np.exp(-((t - n * 0.62) / 40) ** 2))
    x = (x - x.min()) / (np.ptp(x) + 1e-9) * (height - 4) + 2
    pts = " ".join(f"{i * width / (n - 1):.1f},{v:.1f}" for i, v in enumerate(x))
    acc = ""
    if accent:
        acc = f'<line x1="{width * 0.62:.0f}" y1="0" x2="{width * 0.62:.0f}" y2="{height}" stroke="{accent}" stroke-width="1.5"/>'
    return (f'<svg class="trace" viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
            f'<polyline fill="none" stroke="{color}" stroke-width="1.1" points="{pts}" vector-effect="non-scaling-stroke"/>{acc}</svg>')


def _plain(text):
    return re.sub(r"<[^>]+>", "", text)


def chapter(num, kicker, title, lead, flagship=False):
    """Page header: step indicator, module name, the guiding question and a one-line lead."""
    step = {"06": "Part 1 of 2", "08": "Part 2 of 2"}.get(str(num), "Optional")
    badge = '<span class="badge core">Spatial analysis + Classification module</span>' if flagship else ""
    st.markdown(
        f'<div class="page-head"><div class="crumb">{step}{badge}</div>'
        f'<h1 class="ttl">{kicker}</h1>'
        f'<p class="lead"><span class="q">{_plain(title)}</span> {lead}</p></div>', unsafe_allow_html=True)


def header_row(num, kicker, title, lead, flagship=False):
    """Page header + (right column) dataset menu and an action slot.
    Returns the slot: pages put their Apply control there, so it is always in
    the same place (top right) on every page."""
    c1, c2 = st.columns([4.2, 1.35], vertical_alignment="top", gap="large")
    with c1:
        chapter(num, kicker, title, lead, flagship)
    with c2:
        data_popover(f"dp{num}")
        o = st.session_state.get("_objs_label")
        if o:
            st.markdown(f'<div class="ds-meta">{o}</div>', unsafe_allow_html=True)
        slot = st.container(key=f"actionslot_{num}")
    show_load_error()
    return slot


def _same(a, b):
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def apply_control(slot, step, value, label="Apply", key=None, on_apply=None, help=None, current=None):
    """Status chip + Apply button in the page's action slot.
    The chip tells the student whether what they see is already in the pipeline.
    Call it at the END of the page so it reflects this run's widget values."""
    with slot:
        cur = params().get(step) if current is None else current
        done = step in params().get("applied", [])
        if _same(cur, value):
            chip = '<span class="chip ok">In pipeline</span>' if done else '<span class="chip">Default settings, not applied</span>'
            changed = False
        else:
            chip = '<span class="chip warn">Unapplied changes</span>'
            changed = True
        html(chip)
        clicked = st.button(label, key=key or f"apply_{step}", type="primary" if (changed or not done) else "secondary",
                            use_container_width=True,
                            help=help or "Commit these settings to the shared pipeline. Later modules use them; Undo in the sidebar reverts.")
        if clicked:
            if on_apply:
                on_apply()
            else:
                apply(step, value)
            st.rerun()


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def kicker(text):
    html(f'<div class="kicker">{text}</div>')


def figcap(n, text):
    html(f'<div class="figcap"><b>Figure {n}.</b> {text[:1].upper() + text[1:]}</div>')


def tag(text, kind=""):
    return f'<span class="tag {kind}">{text}</span>'


def note(text, kind=""):
    html(f'<div class="note {kind}">{text}</div>')


def deck_header(left, right=""):
    html(f'<div class="card-h"><span>{left}</span><span class="card-h-r">{right}</span></div>')


def plot(fig, key, select=False, name=None, height=None, mode=("points",)):
    """Plotly wrapper: our template, no Streamlit theme, optional selection events.
    Figures are registered by name so the Export Center can bundle them."""
    if height:
        fig.update_layout(height=height)
    pin(fig)
    if name:
        S()["figs"][name] = fig
    cfg = {"displaylogo": False, "displayModeBar": False, "scrollZoom": False}
    if select:
        return st.plotly_chart(fig, use_container_width=True, theme=None, key=key, on_select="rerun",
                               selection_mode=mode, config=cfg)
    st.plotly_chart(fig, use_container_width=True, theme=None, key=key, config=cfg)
    return None


def pin(fig):
    """Streamlit's frontend injects its own background colours and font into
    layout fields that are unset (even with theme=None). Pin ours explicitly."""
    lay = fig.layout
    if lay.plot_bgcolor is None:
        lay.plot_bgcolor = "rgba(0,0,0,0)"
    if lay.paper_bgcolor is None:
        lay.paper_bgcolor = "rgba(0,0,0,0)"
    if lay.font.family is None:
        fig.update_layout(font=dict(family=viz.SANS, color=viz.INK, size=12))
    # clicking a point must not fade everything else (our own colours show the selection)
    for tr in fig.data:
        if hasattr(tr, "unselected") and tr.type in ("scatter", "scattergl", "bar", "box", "violin", "histogram"):
            try:
                tr.unselected = dict(marker=dict(opacity=1.0))
                tr.selected = dict(marker=dict(opacity=1.0))
            except Exception:
                pass
    if lay.clickmode is None:
        lay.clickmode = "event+select"
    if lay.hovermode is None:
        # bar charts: hovering/clicking anywhere in a bar's column picks that bar (big click targets)
        only_bars = len(fig.data) > 0 and all(t.type == "bar" and (t.orientation or "v") == "v" for t in fig.data)
        lay.hovermode = "x" if only_bars else "closest"
    return fig


def selected_points(event):
    try:
        return event.selection.points or []
    except Exception:
        try:
            return event["selection"]["points"] or []
        except Exception:
            return []


def explain(title, paragraphs, questions):
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    qs = "".join(f'<details><summary data-n="{i + 1}">{q}<span class="reveal">Show answer</span></summary>'
                 f'<div class="ans">{a}</div></details>' for i, (q, a) in enumerate(questions))
    html(f'<div class="explain"><div class="eh"><span class="l">Explanation</span><span class="t">{title}</span></div>'
         f'<div class="eb"><div>{body}</div><div><div class="eb-h">Check your understanding</div>{qs}</div></div></div>')


def export_drawer(step, code, step_params, files=()):
    with st.expander(f"Export code and parameters ({step})", icon=":material/download:"):
        c1, c2 = st.columns([2, 1])
        with c1:
            st.code(code, language="python")
        with c2:
            st.download_button("Parameters · JSON", json.dumps(step_params, indent=2, default=str),
                               file_name=f"{step}_params.json", mime="application/json",
                               use_container_width=True, key=f"exp_{step}_json")
            st.download_button("Code · .py", code, file_name=f"{step}.py", mime="text/x-python",
                               use_container_width=True, key=f"exp_{step}_py")
            for i, (label, data, fname, mime) in enumerate(files):
                st.download_button(label, data, file_name=fname, mime=mime, use_container_width=True,
                                   key=f"exp_{step}_{i}")
            st.caption("The full reproducible pipeline (script, notebook, report, ZIP) is in the Downloads panel at the bottom of the page.")


def apply_button(label, key, help=None):
    return st.button(label, key=key, type="primary", help=help or "Commit these settings to the shared pipeline. Later modules use them.",
                     use_container_width=True)


def run_with_progress(label, fn):
    bar = st.progress(0.0, text=label)

    def cb(frac, msg=""):
        bar.progress(min(1.0, float(frac)), text=f"{label} · {msg}")
    out = fn(cb)
    bar.empty()
    return out


def friendly_message(e):
    m = str(e) or e.__class__.__name__
    low = m.lower()
    if "urlopen" in low or "connection" in low or "timed out" in low or "resolve" in low or "http" in low:
        return "the dataset could not be downloaded (are you online?)."
    if "no usable event" in low or "no matching events" in low:
        return "no event markers matched — choose other settings in Setup or check the file's annotations."
    if "positive definite" in low or "singular" in low:
        return "a covariance matrix was singular — try a shrinkage estimator (Ledoit-Wolf / OAS) or fewer components."
    if "n_splits" in low or "cannot have number of splits" in low:
        return "there are too few trials per class for this many folds — lower k."
    return m[:400]


@contextmanager
def guard(where="this module"):
    try:
        yield
    except Exception as e:
        if e.__class__.__name__ in ("RerunException", "StopException", "RerunData"):
            raise
        html(f'<div class="note v"><b>Something went wrong in {where}.</b> {friendly_message(e)}<br>'
             f'<span class="small">Nothing in your pipeline was changed. Try other settings, Undo, or Reset in the sidebar.</span></div>')
        with st.expander("Technical details"):
            st.code("".join(traceback.format_exception(e))[-3000:], language="text")


def classes_of(epochs):
    return sorted(epochs.metadata["label"].unique().tolist())


def page_objects():
    """Every page calls this: auto-loads synthetic data + default steps if needed."""
    o = objects()
    st.session_state["_ds_label"] = o["label"]
    e = o["epochs"]
    st.session_state["_objs_label"] = (f"{len(e)} trials · {len(e.ch_names)} ch · {e.info['sfreq']:.0f} Hz")
    return o


def on_click(event, key, field="customdata"):
    """Return the clicked value only when the selection actually changed."""
    pts = selected_points(event)
    sig = json.dumps([[p.get(field), p.get("curve_number"), p.get("point_index")] for p in pts], default=str) if pts else None
    if sig != st.session_state.get("_sig_" + key):
        st.session_state["_sig_" + key] = sig
        if pts:
            return pts[0].get(field) if field != "point" else pts[0]
    return None


def memo(name, key, fn):
    """Tiny per-session memo for expensive computations."""
    store = S()["results"].setdefault("memo", {})
    k = f"{name}:{key}"
    if k not in store:
        if len(store) > 40:
            store.pop(next(iter(store)))
        store[k] = fn()
    return store[k]


def seg(label, options, default=None, key=None, label_visibility="collapsed", pills=False, help=None, format_func=str):
    """Segmented control (pills when pills=True). Under automated tests
    (BCI_APPTEST=1) a radio is used, because Streamlit's AppTest cannot drive
    single-select button groups."""
    if os.environ.get("BCI_APPTEST"):
        idx = options.index(st.session_state.get(key, default)) if st.session_state.get(key, default) in options else 0
        return st.radio(label, options, index=idx, key=key, horizontal=True, label_visibility=label_visibility, help=help,
                        format_func=format_func)
    fn = st.pills if pills else st.segmented_control
    return fn(label, options, default=default, key=key, label_visibility=label_visibility, help=help, format_func=format_func)


def rerun_local():
    """Rerun only the enclosing st.fragment when possible (much smoother:
    the rest of the page is not redrawn); fall back to a full rerun."""
    from streamlit.errors import StreamlitAPIException
    try:
        st.rerun(scope="fragment")
    except StreamlitAPIException:
        st.rerun()


def guarded_fragment(where):
    """Decorator: run a function as an st.fragment inside the friendly error guard."""
    def deco(fn):
        @st.fragment
        def wrapper():
            with guard(where):
                fn()
        return wrapper
    return deco


AUTO_STEPS = {"data", "filter", "epochs", "spatial", "features", "classify"}   # sensible defaults exist


def step_status(step, url=None):
    """(material icon, css-key prefix) for a pipeline step, used by the sidebar and the Start page."""
    p = params()
    if step and step in p.get("applied", []):
        icon, k = ":material/check_circle:", "st_done"
    elif step == "ica" and p.get("ica") is None:
        icon, k = ":material/radio_button_unchecked:", "st_none"
    elif step in AUTO_STEPS:
        icon, k = ":material/adjust:", "st_auto"
    else:
        icon, k = ":material/radio_button_unchecked:", "st_none"
    if url in ("spatial", "classify") and k != "st_done":
        icon, k = ":material/star:", "st_flag"
    return icon, k
