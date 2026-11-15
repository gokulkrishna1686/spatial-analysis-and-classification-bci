"""BCI Master Studio — entry point.  Run:  streamlit run app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(page_title="BCI Master Studio", page_icon=str(Path(__file__).parent / "static" / "favicon.svg"), layout="wide",
                   initial_sidebar_state="expanded")

import ui  # noqa: E402

# (file, sidebar label, url, pipeline step or None)
PAGES = [
    ("pages/00_setup.py", "Setup", "setup", None),
    ("pages/01_spatial.py", "Spatial analysis", "spatial", "spatial"),
    ("pages/02_classify.py", "Classification", "classify", "classify"),
]

pages = [st.Page(p, title=t.split("  ", 1)[-1], url_path=u, default=(u == "spatial")) for p, t, u, _ in PAGES]
st.session_state["_pages"] = {u: pg for (_, _, u, _), pg in zip(PAGES, pages)}
nav = st.navigation(pages, position="hidden")

ui.css()
P = ui.params()

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-logo"><svg width="18" height="18" viewBox="0 0 18 18"><polyline points="1,10 4,10 6,5 9,14 11,7 13,10 17,10" '
        'fill="none" stroke="#fff" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/></svg></div>'
        '<div><div class="brand-name">BCI Master Studio</div><div class="brand-sub">EEG processing &amp; classification</div></div></div>',
        unsafe_allow_html=True)
    st.markdown('<div class="rail-h">Module</div>', unsafe_allow_html=True)
    with st.container(key="rail"):
        for (path, title, url, step), page in zip(PAGES, pages):
            icon, k = (":material/tune:", "st_none") if url == "setup" else ui.step_status(step, url)
            with st.container(key=f"{k}_{url}"):
                st.page_link(page, label=title, icon=icon)
    st.markdown('<div class="legend-side"><span class="d" style="background:#1D7A4E"></span>applied &nbsp; '
                '<span style="color:#2F5BD3">★</span> not applied yet</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Undo", use_container_width=True, disabled=not ui.S()["history"],
                     help="Revert the last Apply"):
            ui.undo()
            st.rerun()
    with c2:
        if st.button("Reset", use_container_width=True, help="Back to the default synthetic pipeline"):
            ui.reset()
            st.rerun()
    seed = st.number_input("Random seed", 0, 2**31 - 1, int(P.get("seed", 42)),
                           help="One seed drives data simulation, ICA, CV splits, classifiers and permutations.")
    if seed != P.get("seed"):
        ui.S()["params"]["seed"] = int(seed)
        st.rerun()
    badge = st.empty()

nav.run()

# ---- previous / next station
urls = [u for _, _, u, _ in PAGES]
cur = nav.url_path if nav.url_path in urls else "spatial"
i = urls.index(cur)
with st.container(key="pager"):
    a, b = st.columns(2)
    if i > 0:
        with a:
            st.page_link(pages[i - 1], label=f"← {PAGES[i - 1][1]}", use_container_width=True)
    if i < len(PAGES) - 1:
        with b:
            st.page_link(pages[i + 1], label=f"{PAGES[i + 1][1]} →", use_container_width=True)

lab = st.session_state.get("_ds_label", "—")
applied = set(ui.params().get("applied", []))
steps = " → ".join(s for s in ["filter", "epochs", "spatial", "classify"] if s in applied) or "defaults only"
badge.markdown(f'<div class="ds-badge"><div class="k">Dataset</div><b>{lab}</b><div class="k" style="margin-top:6px">Applied</div>{steps}</div>',
               unsafe_allow_html=True)
st.markdown('<div class="footer">BCI Master Studio · leak-free by construction · seed '
            f'{ui.params().get("seed", 42)} · results are computed live, never hardcoded</div>', unsafe_allow_html=True)
