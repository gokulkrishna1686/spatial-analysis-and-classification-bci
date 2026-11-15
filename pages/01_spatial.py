"""Spatial analysis (part 1 of 2 of our module)."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import classify as C
import concepts_spatial
import downloads
import spatial as SP
import ui
import viz

o = ui.page_objects()
slot = ui.header_row("06", "Spatial analysis", "Where in the head is the <em>information</em>?",
              "Spatial filters combine electrodes into virtual channels that make the classes as different as possible. "
              "Pick a method, click a component and see which part of the scalp is doing the talking.", flagship=True)

METHODS = ["CSP", "Riemann", "xDAWN", "PCA", "Whitening", "CAR", "Laplacian"]
EST = {"empirical": "empirical", "ledoit_wolf": "Ledoit-Wolf", "oas": "OAS", "shrink0.3": "shrinkage 0.3"}
PYR_EST = {"empirical": "scm", "ledoit_wolf": "lwf", "oas": "oas", "shrink0.3": "lwf"}


def apply_recipe(new_sp):
    """Store the spatial recipe and pass it on to Classification (which refits it per training fold)."""
    cl = dict(ui.params()["classify"])
    cl.update(band=list(new_sp["band"]), window=list(new_sp["window"]), n_components=new_sp["n_components"], cov=new_sp["cov"])
    m = new_sp["method"]
    cl["feature"] = {"CSP": "csp", "Riemann": "tangent", "xDAWN": "xdawn"}.get(m, cl.get("feature", "csp"))
    cl["pre"] = m if m in ("CAR", "Laplacian") else None
    was_applied = "classify" in ui.params().get("applied", [])
    ui.apply_many({"spatial": new_sp, "classify": cl}, f"Spatial recipe sent to Classification · {m}")
    if not was_applied:                      # pre-fills Classification; it is not "applied" until saved there
        ui.params()["applied"].remove("classify")


def recipe_panel(new_sp, key, editable):
    """Apply button + what will be applied + where to go next. Shown in both modes."""
    ui.apply_control(slot, "spatial", new_sp, "Apply spatial recipe", key=key, on_apply=lambda: apply_recipe(new_sp),
                     help="Sends this recipe to Classification. Supervised filters (CSP, xDAWN, Riemann) are not fitted here: "
                          "Classification refits them inside every training fold.")
    with slot:
        r = new_sp
        txt = (f'<b>Recipe:</b> {r["method"]} · {r["band"][0]:g}–{r["band"][1]:g} Hz · '
               f'{r["window"][0]:g}–{r["window"][1]:g} s · {r["n_components"]} components')
        if not editable:
            txt += '<br>Change it under <b>Your data</b>.'
        ui.html(f'<div class="ds-meta">{txt}</div>')
        P = ui.params()
        if "spatial" in P.get("applied", []) and ui._same(P.get("spatial"), new_sp) and "_pages" in st.session_state:
            st.page_link(st.session_state["_pages"]["classify"], label="Continue in Classification →", icon=":material/arrow_forward:",
                         use_container_width=True)


def side_of(name, xy_pt):
    if name.lower().endswith("z"):
        return "midline"
    return "left hemisphere" if xy_pt[0] < 0 else "right hemisphere"


with ui.guard("Spatial analysis"):
    ep = o["epochs"]
    sp = dict(ui.params()["spatial"])
    names, xy = viz.sensor_xy(ep.info)
    xyz = SP.info_xyz(ep.info)
    classes_all = ui.classes_of(ep)
    sf = ep.info["sfreq"]
    seed = ui.params()["seed"]

    def your_data():
        """Real EEG: method rail, control bar, views and inspector."""
        with st.container(key="method_rail"):
            method = ui.seg("Method", METHODS, default=sp.get("method", "CSP") if sp.get("method") in METHODS else "CSP", key="sp_method") or "CSP"
        n_ch = len(names)

        # ------------------------------------------------------------------ control bar (one row)
        with st.container(key="deck_bar"):
            presets = {"mu": (8.0, 12.0), "beta": (13.0, 30.0), "mu+beta": (8.0, 30.0), "ERP": (1.0, 20.0)}
            cur_band = tuple(sp.get("band", (8.0, 30.0)))
            pk = next((k for k, v in presets.items() if v == cur_band), "custom")
            multi = len(classes_all) > 2
            cols = st.columns([2.0, 1.35, 0.95, 1.0, 0.95] + ([1.3] if multi else []), gap="small")
            with cols[0]:
                bp = ui.seg("Band", list(presets) + ["custom"], default=pk, key="sp_bandp", pills=True, label_visibility="visible",
                            help="Spatial filters are computed on band-passed data. mu 8–12 Hz, beta 13–30 Hz (motor imagery); ERP 1–20 Hz (P300).") or pk
                if bp == "custom":
                    band = st.slider("Custom band (Hz)", 1.0, min(60.0, sf / 2 - 1), (float(cur_band[0]), float(cur_band[1])), 0.5, key="sp_band",
                                     label_visibility="collapsed")
                else:
                    band = presets[bp]
            t0, t1 = float(ep.times[0]), float(ep.times[-1])
            w = sp.get("window", (0.5, 2.5))
            window = cols[1].slider("Time window (s)", t0, t1, (max(t0, float(w[0])), min(t1, float(w[1]))), 0.05, key="sp_win",
                                    help="Only this part of each epoch is used. For MI skip the first ~0.5 s after the cue (visual response).")
            ncomp = cols[2].slider("Components", 2, min(12, n_ch), min(int(sp.get("n_components", 6)), min(12, n_ch)), 1, key="sp_nc",
                                   help="How many spatial filters are passed on to feature extraction / classification.")
            est = cols[3].selectbox("Covariance", list(EST), index=list(EST).index(sp.get("cov", "ledoit_wolf")) if sp.get("cov", "ledoit_wolf") in EST else 1,
                                    format_func=EST.get, key="sp_est",
                                    help="Shrinkage (Ledoit-Wolf, OAS) stabilises covariances when trials are few or channels many.")
            pct = cols[4].slider("Trials %", 10, 100, 100, 5, key="sp_pct",
                                 help="Fewer trials make spatial patterns noisier. Slide down and watch them degrade.")
            if multi:
                with cols[5]:
                    A = st.selectbox("Class A", classes_all, 0, format_func=viz.pretty, key="sp_A")
                    B = st.selectbox("Class B", [c for c in classes_all if c != A], 0, format_func=viz.pretty, key="sp_B", label_visibility="collapsed")
                    pair = [A, B]
            else:
                pair = classes_all
        new_sp = {"method": method, "n_components": ncomp, "cov": est, "band": list(band), "window": list(window)}
        recipe_panel(new_sp, "sp_apply", editable=True)
        ui.html('<div class="bar-tags">' + ui.tag("Exploratory fit · all trials", "o") + ui.tag("Refit per fold in Classification", "t")
                + f'<span class="figcap" style="margin:0 0 0 8px">{method} · {band[0]:g}–{band[1]:g} Hz · {window[0]:g}–{window[1]:g} s · '
                + f'{" vs ".join(viz.pretty(c) for c in pair)}</span></div>')

        # data for this page (stateless steps only)
        pre = method if method in ("CAR", "Laplacian") else None
        X, y, meta, times = C.get_Xy(ep, band, window, pair)
        if pct < 100:
            rng = np.random.RandomState(seed)
            keep = np.sort(np.concatenate([rng.permutation(np.where(y == c)[0])[:max(3, int(np.sum(y == c) * pct / 100))] for c in np.unique(y)]))
            X, y, meta = X[keep], y[keep], meta.iloc[keep].reset_index(drop=True)
        cls = np.unique(y)
        cc = viz.class_colors(list(cls))
        key = ui.phash([ui.params()[k] for k in ("data", "filter", "ica", "epochs", "seed")] + [list(band), list(window), est, pair, pct])

        @ui.guarded_fragment("Spatial view")
        def stage_area():
            """Everything below the control bar. Clicks in here rerun only this fragment."""
            stage, insp = st.columns([3.1, 1.12], gap="medium")
            # ================================================================ CSP
            if method == "CSP":
                if len(cls) != 2:
                    ui.note("CSP needs two classes. Pick a pair in the control bar.", "v")
                    return
                csp = ui.memo("csp", key, lambda: SP.CSP(n_ch, est).fit(X, y))
                lam, pats, filt = csp.eigenvalues_, csp.patterns_, csp.filters_
                Z = np.log(np.einsum("kc,nct->nkt", filt, X).var(2))
                aucs = np.array([SP.separation_auc(Z[:, k], y) for k in range(n_ch)])
                st.session_state.setdefault("sp_comp", 0)
                comp = min(st.session_state["sp_comp"], n_ch - 1)
                with stage:
                    views = ["Power maps", "Patterns", "Pull apart", "3-D head", "Beat CSP", "Stability"]
                    tp = o.get("true_patterns") or {}
                    if tp and all(c in tp for c in cls):
                        views.append("Truth")
                    view = ui.seg("View", views, default="Power maps", key="sp_view") or "Power maps"

                    if view == "Power maps":
                        ui.kicker("What differs between the classes? Band power per electrode (click an electrode on the difference map)")
                        L = 10 * np.log10(X.var(2))                     # trials x channels, dB
                        mA, mB = L[y == cls[0]].mean(0), L[y == cls[1]].mean(0)
                        grand = (mA + mB) / 2
                        st.session_state.setdefault("pm_ch", "C3" if "C3" in names else names[0])
                        pm_ch = st.session_state["pm_ch"] if st.session_state["pm_ch"] in names else names[0]
                        g = viz.topo_grid([mA - grand, mB - grand], xy, names,
                                          [f"{viz.pretty(cls[0])}: power vs average (dB)", f"{viz.pretty(cls[1])}: power vs average (dB)"],
                                          cols=2, height=250, cmax_shared=True)
                        ui.plot(g, "pm_pair", name="class_power_maps")
                        pc1, pc2 = st.columns([1.1, 1])
                        with pc1:
                            ev = ui.plot(viz.topo_fig(mA - mB, xy, names, height=300, highlight=pm_ch, hover_unit=" dB",
                                                      title=f"difference: {viz.pretty(cls[0])} − {viz.pretty(cls[1])} (dB)"),
                                         "pm_diff", select=True, name="class_power_difference")
                            v = ui.on_click(ev, "pm_diff")
                            if v and v != pm_ch:
                                st.session_state["pm_ch"] = v
                                ui.rerun_local()
                        with pc2:
                            ci = names.index(pm_ch)
                            fv = go.Figure()
                            for c_ in cls:
                                fv.add_trace(go.Box(y=L[y == c_, ci], name=viz.pretty(c_), marker_color=cc[c_], boxpoints="all", jitter=0.4, pointpos=0))
                            fv.update_layout(height=300, title=f"{pm_ch}: power of every trial", yaxis_title="band power (dB)", showlegend=False)
                            ui.plot(fv, "pm_box")
                        ui.html('<div class="reading"><b>What to notice.</b> Over each motor cortex, power is <em>lower</em> when the opposite hand is imagined '
                                '(ERD): the blobs near C3 and C4 have opposite colours. But the box plot shows how much single trials overlap at one electrode. '
                                'CSP (next tab) combines electrodes to pull the classes further apart.</div>')

                    elif view == "Patterns":
                        ui.kicker("Eigen-ladder — click a component")
                        dev = lam - 0.5
                        fb = go.Figure()
                        fb.add_trace(go.Bar(x=[f"C{k + 1}" for k in range(n_ch)], y=dev, base=0.5, customdata=list(range(n_ch)),
                                            marker=dict(color=[cc[f] for f in csp.favours_],
                                                        opacity=[1.0 if k < ncomp else 0.28 for k in range(n_ch)],
                                                        line=dict(color=[viz.INK if k == comp else "rgba(0,0,0,0)" for k in range(n_ch)], width=2.5)),
                                            hovertemplate="%{x}<br>λ = %{y:.3f}<extra></extra>"))
                        fb.add_hline(0.5, line=dict(color=viz.INK, width=1))
                        fb.add_vrect(x0=ncomp - 0.5, x1=n_ch - 0.5, fillcolor=viz.RULE2, opacity=0.45, line_width=0)
                        fb.add_annotation(x=(ncomp + n_ch) / 2 - 0.5, y=0.5 + 0.9 * max(0.06, float(np.abs(dev).max()) * 1.3), text="discarded", showarrow=False,
                                          font=dict(family=viz.MONO, size=10, color=viz.MUTED))
                        rr = max(0.06, float(np.abs(dev).max()) * 1.3)
                        fb.update_layout(height=230, yaxis=dict(range=[0.5 - rr, 0.5 + rr], title="eigenvalue λ"), margin=dict(t=8, b=30), bargap=0.25)
                        ev = ui.plot(fb, "sp_ladder", select=True, name="csp_eigen_ladder")
                        v = ui.on_click(ev, "sp_ladder")
                        if v is not None and int(v) != comp:
                            st.session_state["sp_comp"] = int(v)
                            ui.rerun_local()
                        ui.figcap("6.1", f"λ = share of the component's variance that belongs to {viz.pretty(cls[0])}. "
                                  f"Bars above 0.5: more {viz.pretty(cls[0])} power; below: more {viz.pretty(cls[1])} power. Teal/red = favoured class.")
                        a, b = st.columns(2)
                        with a:
                            ui.kicker("Spatial filter W — what we compute with")
                            ui.plot(viz.topo_fig(filt[comp], xy, names, height=245), "sp_filt", name="csp_filter")
                        with b:
                            ui.kicker("Activation pattern A — what the brain shows")
                            ui.plot(viz.topo_fig(pats[comp], xy, names, height=245, highlight=names[int(np.argmax(np.abs(pats[comp])))]), "sp_pat", name="csp_pattern")
                        ui.figcap("6.2", "filters look noisy because they also cancel noise; patterns (Haufe 2014) are the interpretable picture")
                        fv = go.Figure()
                        for c_ in cls:
                            fv.add_trace(go.Violin(x=Z[y == c_, comp], name=viz.pretty(c_), line_color=cc[c_], fillcolor=viz.rgba(cc[c_], 0.25),
                                                   orientation="h", points="all", pointpos=0, jitter=0.5, marker=dict(size=3.5), meanline_visible=True))
                        fv.update_layout(height=200, xaxis_title=f"log-variance of C{comp + 1}", showlegend=False, margin=dict(t=6, l=90))
                        ui.plot(fv, "sp_violin", name="csp_logvar_violin")

                    elif view == "Pull apart":
                        ui.kicker("Press play — trials slide from electrode space into CSP space")
                        pairs_ch = [n for n in ("C3", "C4") if n in names]
                        if len(pairs_ch) < 2:
                            a_ = np.argsort([-SP.separation_auc(np.log(X[:, i].var(1)), y) for i in range(n_ch)])[:2]
                            pairs_ch = [names[a_[0]], names[a_[1]]]
                        L = np.log(X.var(2))
                        P0 = L[:, [names.index(pairs_ch[0]), names.index(pairs_ch[1])]]
                        P1 = Z[:, [0, 1]]
                        zs = lambda M: (M - M.mean(0)) / (M.std(0) + 1e-12)
                        P0, P1 = zs(P0), zs(P1)
                        frames, steps = [], np.linspace(0, 1, 26)
                        ease = lambda s: 3 * s ** 2 - 2 * s ** 3

                        def scat(Pm):
                            return [go.Scatter(x=Pm[y == c_, 0], y=Pm[y == c_, 1], mode="markers", name=viz.pretty(c_),
                                               marker=dict(size=8, color=cc[c_], line=dict(width=0.8, color=viz.PAPER), opacity=0.9)) for c_ in cls]
                        for s in steps:
                            Pm = (1 - ease(s)) * P0 + ease(s) * P1
                            fr = SP.fisher_ratio(Pm, y)
                            frames.append(go.Frame(data=scat(Pm), name=f"{s:.2f}",
                                                   layout=dict(annotations=[dict(x=0.99, y=0.02, xref="paper", yref="paper", xanchor="right", showarrow=False,
                                                                                 text=f"Fisher ratio  {fr:5.2f}", font=dict(family=viz.MONO, size=15, color=viz.VERM))])))
                        fm = go.Figure(data=scat(P0), frames=frames)
                        fm.update_layout(height=470, xaxis=dict(range=[-3.6, 3.6], title=f"{pairs_ch[0]}  →  CSP C1"),
                                         yaxis=dict(range=[-3.6, 3.6], title=f"{pairs_ch[1]}  →  CSP C2", scaleanchor="x"),
                                         annotations=[dict(x=0.99, y=0.02, xref="paper", yref="paper", xanchor="right", showarrow=False,
                                                           text=f"Fisher ratio  {SP.fisher_ratio(P0, y):5.2f}", font=dict(family=viz.MONO, size=15, color=viz.VERM))],
                                         updatemenus=[dict(type="buttons", x=0, y=-0.13, xanchor="left", showactive=False, bgcolor=viz.CARD, bordercolor=viz.INK,
                                                           font=dict(family=viz.MONO, size=11),
                                                           buttons=[dict(label="▶  PLAY", method="animate", args=[None, dict(frame=dict(duration=70, redraw=True), transition=dict(duration=60), fromcurrent=True)]),
                                                                    dict(label="↺  RESET", method="animate", args=[["0.00"], dict(frame=dict(duration=0, redraw=True), mode="immediate")])])],
                                         sliders=[dict(active=0, x=0.25, len=0.75, y=-0.08, currentvalue=dict(visible=False), tickcolor=viz.RULE,
                                                       font=dict(family=viz.MONO, size=9),
                                                       steps=[dict(label=("electrodes" if i == 0 else ("CSP" if i == len(steps) - 1 else "")), method="animate",
                                                                   args=[[f"{s:.2f}"], dict(frame=dict(duration=0, redraw=True), mode="immediate")]) for i, s in enumerate(steps)])],
                                         margin=dict(b=90))
                        ui.plot(fm, "sp_morph", name="csp_morph")
                        ui.figcap("6.3", f"left: log-power at {pairs_ch[0]}/{pairs_ch[1]} · right: CSP components 1 and 2 (z-scored). The Fisher ratio is between-class over within-class scatter")
                        ui.kicker("Click a trial to see its two CSP signals")
                        fs_ = go.Figure()
                        for c_ in cls:
                            idx = np.where(y == c_)[0]
                            fs_.add_trace(go.Scatter(x=P1[idx, 0], y=P1[idx, 1], mode="markers", name=viz.pretty(c_), customdata=idx,
                                                     marker=dict(size=8, color=cc[c_], line=dict(width=0.8, color=viz.PAPER))))
                        fs_.update_layout(height=260, xaxis_title="CSP C1", yaxis_title="CSP C2", margin=dict(t=6))
                        c1_, c2_ = st.columns([1, 1.4])
                        with c1_:
                            ev = ui.plot(fs_, "sp_trials", select=True)
                        v = ui.on_click(ev, "sp_trials")
                        if v is not None:
                            st.session_state["sp_trial"] = int(v)
                        tr = st.session_state.get("sp_trial", 0)
                        tr = tr if tr < len(y) else 0
                        with c2_:
                            sig = np.einsum("kc,ct->kt", filt[[0, 1]], X[tr]) * 1e6
                            fz = go.Figure()
                            for k_, col in zip((0, 1), (cc[csp.favours_[0]], cc[csp.favours_[1]])):
                                fz.add_trace(go.Scatter(x=times, y=sig[k_] - k_ * 4 * sig.std(), name=f"C{k_ + 1}", line=dict(color=col, width=1.2)))
                            fz.update_layout(height=260, title=f"trial {tr} · {viz.pretty(y[tr])}", xaxis_title="time (s)", yaxis=dict(showticklabels=False))
                            ui.plot(fz, "sp_trialsig")

                    elif view == "3-D head":
                        ui.kicker(f"Activation pattern of C{comp + 1} painted on the scalp — drag to rotate, SPIN to orbit")
                        ui.plot(viz.head3d(pats[comp], xyz, names, height=520), "sp_head", name="csp_head3d")
                        ui.figcap("6.4", "thin-plate interpolation of the pattern over a spherical head; nose points forward")

                    elif view == "Beat CSP":
                        ui.html('<div class="big-q">Can you hand-craft a spatial filter that separates the classes <em>as well as CSP</em>?</div>')
                        ui.note("Click electrodes to cycle their weight: <b>0 → +1 → −1 → 0</b>. Your virtual channel is Σ wᵢ·xᵢ. "
                                "Hint: physiology says motor ERD is focal over C3/C4, and a local Laplacian (a centre minus its neighbours) sharpens it.", "t")
                        W = st.session_state.setdefault("beat_w", {})
                        W = {k: v for k, v in W.items() if k in names}
                        pc = st.columns(4)
                        if pc[0].button("C3 Laplacian", key="bt_l3", use_container_width=True) and "C3" in names:
                            i = names.index("C3"); nb = np.argsort(np.linalg.norm(xyz - xyz[i], axis=1))[1:5]
                            W = {"C3": 1.0, **{names[j]: -0.25 for j in nb}}
                        if pc[1].button("C3 − C4", key="bt_bip", use_container_width=True) and "C3" in names and "C4" in names:
                            W = {"C3": 1.0, "C4": -1.0}
                        if pc[2].button("Clear", key="bt_clr", use_container_width=True):
                            W = {}
                        reveal = pc[3].toggle("Show CSP", key="bt_rev", help="Overlay the CSP filter of component 1 for comparison.")
                        st.session_state["beat_w"] = W
                        kc = st.session_state.setdefault("beat_k", 0)
                        a, b = st.columns([1.15, 1])
                        with a:
                            wv = np.array([W.get(n, 0.0) for n in names])
                            fig = go.Figure()
                            if reveal:
                                viz.add_topomap(fig, filt[0] / np.abs(filt[0]).max(), xy, names, sensor_size=0.1)
                                fig.data[-1].update(visible=False)
                            for s in viz.head_shapes():
                                fig.add_shape(**s)
                            col = [viz.VERM if v_ > 0 else (viz.TEAL if v_ < 0 else viz.INK) for v_ in wv]
                            fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers+text", text=[f"{n}" for n in names], customdata=names,
                                                     textposition="top center", textfont=dict(family=viz.MONO, size=9),
                                                     marker=dict(size=[24 if v_ else 13 for v_ in wv], color=col, symbol=["circle" if v_ >= 0 else "circle" for v_ in wv],
                                                                 line=dict(width=1.5, color=viz.PAPER)),
                                                     hovertext=[f"{n}: w = {v_:+.2f}" for n, v_ in zip(names, wv)], hoverinfo="text", showlegend=False))
                            fig.update_xaxes(visible=False, range=[-1.2, 1.2])
                            fig.update_yaxes(visible=False, range=[-1.2, 1.2], scaleanchor="x")
                            fig.update_layout(height=460, margin=dict(l=0, r=0, t=0, b=0), dragmode=False)
                            ev = ui.plot(fig, f"beat_{kc}", select=True)
                            ui.html('<div class="legend-row"><span><i style="background:#D6492A"></i>weight +1</span><span><i style="background:#12707A"></i>weight −1 (or −¼ in Laplacian)</span><span><i style="background:#1B2130"></i>0</span></div>')
                            v = ui.on_click(ev, f"beat_{kc}")
                            if v:
                                cur = W.get(v, 0.0)
                                nxt = {0.0: 1.0, 1.0: -1.0}.get(cur, 0.0) if cur in (0.0, 1.0, -1.0) else 0.0
                                if nxt == 0.0:
                                    W.pop(v, None)
                                else:
                                    W[v] = nxt
                                st.session_state["beat_w"] = W
                                st.session_state["beat_k"] = kc + 1
                                ui.rerun_local()
                        with b:
                            wv = np.array([W.get(n, 0.0) for n in names])
                            best_ch = max(range(n_ch), key=lambda i: SP.separation_auc(np.log(X[:, i].var(1)), y))
                            s_csp = aucs[0]
                            s_ch = SP.separation_auc(np.log(X[:, best_ch].var(1)), y)
                            if np.any(wv):
                                zf = np.log(np.einsum("c,nct->nt", wv, X).var(1) + 1e-30)
                                s_you = SP.separation_auc(zf, y)
                            else:
                                zf, s_you = None, 0.0
                            rows = [("You", s_you, viz.VERM), ("CSP C1", s_csp, viz.INK), (f"best electrode {names[best_ch]}", s_ch, viz.MUTED)]
                            ui.kicker("Separation score (|2·AUC − 1|)")
                            for lab, s, col in rows:
                                ui.html(f'<div class="kv" style="margin:2px 0"><div class="k">{lab}</div><div class="v">{s:.3f}</div></div>'
                                        f'<div class="score-bar"><span style="width:{s * 100:.1f}%;background:{col}"></span></div>')
                            ratio = s_you / s_csp if s_csp > 0 else 0
                            msg = ("Build a filter first." if not np.any(wv) else
                                   "You beat CSP on this data. Careful, both are fit on all trials, so this is exploratory." if ratio >= 1 else
                                   f"You reach <b>{ratio * 100:.0f}%</b> of CSP's separation.")
                            ui.html(f'<div class="reading">{msg}</div>')
                            if zf is not None:
                                fv = go.Figure()
                                for c_ in cls:
                                    fv.add_trace(go.Box(x=zf[y == c_], name=viz.pretty(c_), marker_color=cc[c_], boxpoints="all", jitter=0.4, pointpos=0, orientation="h"))
                                fv.update_layout(height=170, showlegend=False, xaxis_title="log-variance of your channel", margin=dict(t=4, l=90))
                                ui.plot(fv, "beat_box")
                            if st.button("Log this attempt", key="bt_log", use_container_width=True, disabled=not np.any(wv)):
                                st.session_state.setdefault("beat_log", []).append((" ".join(f"{k}{v_:+.2g}" for k, v_ in W.items()), s_you))
                            for lab, s in sorted(st.session_state.get("beat_log", []), key=lambda r: -r[1])[:5]:
                                ui.html(f'<div class="attempt"><span>{lab[:38]}</span><span>{s:.3f}</span></div>')

                    elif view == "Stability":
                        ui.kicker(f"C{comp + 1} refit on each training fold — do the patterns agree?")
                        splits = C.make_splits({"scheme": "kfold", "k": 5}, y, meta, seed)

                        def fold_pats():
                            out = []
                            for tr, _ in splits:
                                c_ = SP.CSP(n_ch, est).fit(X[tr], y[tr])
                                cand = c_.patterns_[:max(4, ncomp)]
                                i, r = SP.match_pattern(pats[comp], cand)
                                out.append((SP.align_sign(pats[comp], cand[i]), r, c_.eigenvalues_[i]))
                            return out
                        fp = ui.memo("stab", key + str(comp), fold_pats)
                        gf = viz.topo_grid([p for p, _, _ in fp], xy, names, [f"fold {i + 1} · r={r:.2f}" for i, (_, r, _) in enumerate(fp)], cols=5, height=190)
                        ui.plot(gf, "sp_stab", name="csp_stability")
                        rs = np.array([r for _, r, _ in fp])
                        fl = go.Figure(go.Scatter(x=[f"fold {i + 1}" for i in range(len(fp))], y=[l_ for _, _, l_ in fp], mode="markers+lines",
                                                  marker=dict(size=11, color=viz.VERM), line=dict(color=viz.RULE)))
                        fl.add_hline(lam[comp], line=dict(color=viz.INK, dash="dot"), annotation_text="all-trial λ", annotation_font=dict(family=viz.MONO, size=10))
                        fl.update_layout(height=210, yaxis_title="λ in fold", margin=dict(t=10))
                        ui.plot(fl, "sp_stab_l")
                        verdict = "stable" if rs.min() > 0.9 else ("fairly stable" if rs.min() > 0.75 else "unstable — more trials or stronger shrinkage needed")
                        ui.html(f'<div class="reading">Mean similarity to the all-trial pattern: <em>{rs.mean():.2f}</em> (min {rs.min():.2f}) → {verdict}.</div>')
                        ui.figcap("6.5", "each fold sees 80 % of the trials; |r| = absolute correlation with the all-trial pattern, signs aligned")

                    elif view == "Truth":
                        ui.html('<div class="big-q">The synthetic data were generated from <em>known</em> brain sources. Did CSP find them?</div>')
                        rev = st.toggle("Reveal the hidden sources", key="gt_rev")
                        eeg_names = [n for n in names]
                        from core import SYN_CHANNELS
                        cards = st.columns(len(cls))
                        for col_, c_ in zip(cards, cls):
                            with col_:
                                t_full = np.array(tp[c_])
                                tvec = np.array([t_full[SYN_CHANNELS.index(n)] if n in SYN_CHANNELS else 0 for n in eeg_names])
                                if (ui.params().get("filter") or {}).get("reference") == "average":
                                    tvec = tvec - tvec.mean()
                                i, r = SP.match_pattern(tvec, pats[:ncomp])
                                ui.kicker(f"source of {viz.pretty(c_)} ERD")
                                if rev:
                                    ui.plot(viz.topo_fig(tvec, xy, names, height=230, title="true pattern"), f"gt_t_{c_}")
                                else:
                                    ui.plot(viz.blank("hidden — toggle reveal", 230), f"gt_b_{c_}")
                                ui.plot(viz.topo_fig(SP.align_sign(tvec, pats[i]), xy, names, height=230, title=f"recovered · C{i + 1}"), f"gt_r_{c_}")
                                ui.html(f'<div class="insp-big">{r:.2f}<small> |r|</small></div><div class="figcap">match between true and recovered pattern</div>')
                        ui.note("A high |r| means CSP's activation pattern points at the true generator. A <b>filter</b> would correlate much worse, "
                                "which is exactly why we interpret patterns, not filters.", "t")

                with insp:
                    with st.container(key="inspector"):
                        ui.deck_header("Inspector", f"C{comp + 1}")
                        fav = csp.favours_[comp]
                        other = [c_ for c_ in cls if c_ != fav][0]
                        peak = names[int(np.argmax(np.abs(pats[comp])))]
                        ratio = (lam[comp] / (1 - lam[comp])) if lam[comp] >= 0.5 else ((1 - lam[comp]) / max(lam[comp], 1e-9))
                        ui.html(f'<div class="insp-big">C{comp + 1}<small> / {n_ch}</small></div>')
                        ui.html('<div class="kv">' + "".join(f'<div class="k">{k}</div><div class="v">{v}</div>' for k, v in [
                            ("eigenvalue λ", f"{lam[comp]:.3f}"), ("variance ratio", f"{ratio:.2f}×"), ("favours", viz.pretty(fav)),
                            ("peak electrode", peak), ("separation", f"{aucs[comp]:.3f}"), ("kept?", "yes" if comp < ncomp else "no")]) + "</div>")
                        ui.html(f'<div class="reading">Power in this component is <em>{ratio:.1f}×</em> higher during <b>{viz.pretty(fav)}</b>. '
                                f'In other words it drops (ERD) when the subject imagines <b>{viz.pretty(other)}</b>. The source peaks at <b>{peak}</b> '
                                f'({side_of(peak, xy[names.index(peak)])}).</div>')
                        nav = st.columns(2)
                        if nav[0].button("◀ prev", key="sp_prev", use_container_width=True, disabled=comp == 0):
                            st.session_state["sp_comp"] = comp - 1
                            ui.rerun_local()
                        if nav[1].button("next ▶", key="sp_next", use_container_width=True, disabled=comp >= n_ch - 1):
                            st.session_state["sp_comp"] = comp + 1
                            ui.rerun_local()
                        fmini = go.Figure(go.Bar(x=list(range(1, n_ch + 1)), y=aucs, marker_color=[viz.VERM if k == comp else viz.INK2 for k in range(n_ch)]))
                        fmini.update_layout(height=150, margin=dict(l=30, r=4, t=20, b=24), title=dict(text="separation per component", font=dict(size=11, family=viz.MONO)),
                                            yaxis=dict(range=[0, 1]))
                        ui.plot(fmini, "sp_mini")
                        ui.html(f'<div class="figcap">{len(y)} trials · {X.shape[2]} samples · band {band[0]:g}–{band[1]:g} Hz</div>')

            # ================================================================ Riemann
            elif method == "Riemann":
                from pyriemann.estimation import Covariances
                from pyriemann.tangentspace import TangentSpace
                from sklearn.decomposition import PCA

                def riem():
                    Cv = Covariances(PYR_EST[est]).fit_transform(X)
                    classes_, M = SP.riemann_means(Cv, y)
                    D = SP.riemann_distances(Cv, M)
                    T = TangentSpace().fit_transform(Cv)
                    E = PCA(2, random_state=seed).fit_transform(T)
                    return Cv, M, D, E
                Cv, M, D, E = ui.memo("riem", key, riem)
                st.session_state.setdefault("rm_trial", 0)
                tr = min(st.session_state["rm_trial"], len(y) - 1)
                with stage:
                    view = ui.seg("View", ["Distance plane", "Class means", "Tangent map"], default="Distance plane", key="rm_view") or "Distance plane"
                    if view == "Distance plane":
                        ui.kicker("Every trial at its Riemannian distance to each class mean — click one")
                        if len(cls) > 2:
                            ui.note("Showing the first two classes' means.", "")
                        pred = cls[np.argmin(D, 1)]
                        fd = go.Figure()
                        lim = float(np.percentile(D[:, :2], 99)) * 1.05
                        fd.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color=viz.INK, dash="dot"), hoverinfo="skip", showlegend=False))
                        fd.add_trace(go.Scatter(x=[0, lim, 0], y=[0, lim, lim], fill="toself", fillcolor=viz.rgba(cc[cls[0]], 0.06), line_width=0, hoverinfo="skip", showlegend=False))
                        fd.add_trace(go.Scatter(x=[0, lim, lim], y=[0, 0, lim], fill="toself", fillcolor=viz.rgba(cc[cls[1]], 0.06), line_width=0, hoverinfo="skip", showlegend=False))
                        for c_ in cls:
                            idx = np.where(y == c_)[0]
                            fd.add_trace(go.Scatter(x=D[idx, 0], y=D[idx, 1], mode="markers", name=viz.pretty(c_), customdata=idx,
                                                    marker=dict(size=[13 if i == tr else 8 for i in idx], color=cc[c_],
                                                                symbol=["x" if pred[i] != y[i] else "circle" for i in idx], line=dict(width=1, color=viz.PAPER))))
                        fd.add_annotation(x=lim * 0.18, y=lim * 0.92, text=f"closer to {viz.pretty(cls[0])}", showarrow=False, font=dict(family=viz.MONO, size=11, color=cc[cls[0]]))
                        fd.add_annotation(x=lim * 0.85, y=lim * 0.1, text=f"closer to {viz.pretty(cls[1])}", showarrow=False, font=dict(family=viz.MONO, size=11, color=cc[cls[1]]))
                        fd.update_layout(height=480, xaxis=dict(title=f"δ_R(C, M_{viz.pretty(cls[0])})", range=[0, lim]),
                                         yaxis=dict(title=f"δ_R(C, M_{viz.pretty(cls[1])})", range=[0, lim], scaleanchor="x"))
                        ev = ui.plot(fd, "rm_plane", select=True, name="riemann_distance_plane")
                        v = ui.on_click(ev, "rm_plane")
                        if v is not None and int(v) != tr:
                            st.session_state["rm_trial"] = int(v)
                            ui.rerun_local()
                        ui.figcap("6.6", "the dotted diagonal is the MDM decision boundary; × = trial lies on the wrong side (exploratory, means fit on all trials)")
                        ui.kicker(f"Trial {tr} covariance next to both class means")
                        hm = make_subplots(1, 3, subplot_titles=(f"trial {tr} · {viz.pretty(y[tr])}", f"mean {viz.pretty(cls[0])}", f"mean {viz.pretty(cls[1])}"), horizontal_spacing=0.04)
                        for k_, Mx in enumerate([Cv[tr], M[0], M[1]]):
                            hm.add_trace(go.Heatmap(z=Mx[::-1], x=names, y=names[::-1], colorscale=viz.DIVERGING, zmid=0, showscale=False), 1, k_ + 1)
                        hm.update_xaxes(showticklabels=False)
                        hm.update_yaxes(showticklabels=False)
                        hm.update_layout(height=260, margin=dict(t=30))
                        hm.update_annotations(font=dict(family=viz.MONO, size=10.5))
                        ui.plot(hm, "rm_heat")
                    elif view == "Class means":
                        from pyriemann.geometry.mean import mean_euclid
                        from pyriemann.geometry.distance import distance_riemann
                        ui.kicker("Riemannian class means and their log-difference")
                        from scipy.linalg import logm
                        diff = np.real(logm(np.linalg.solve(M[1], M[0]) + 0j)) if len(M) > 1 else M[0]
                        hm = make_subplots(1, 3, subplot_titles=(f"M {viz.pretty(cls[0])}", f"M {viz.pretty(cls[1])}", "log(M₂⁻¹M₁) — where they differ"), horizontal_spacing=0.05)
                        for k_, Mx in enumerate([M[0], M[1], diff]):
                            hm.add_trace(go.Heatmap(z=Mx[::-1], x=names, y=names[::-1], colorscale=viz.DIVERGING, zmid=0, showscale=False,
                                                    hovertemplate="%{x} × %{y}: %{z:.3g}<extra></extra>"), 1, k_ + 1)
                        hm.update_xaxes(showticklabels=False)
                        hm.update_yaxes(showticklabels=False)
                        hm.update_layout(height=330)
                        hm.update_annotations(font=dict(family=viz.MONO, size=10.5))
                        ui.plot(hm, "rm_means", name="riemann_means")
                        de = [distance_riemann(mean_euclid(Cv[y == c_]), M[k_]) for k_, c_ in enumerate(cls)]
                        ui.html(f'<div class="reading">The Euclidean average of covariance matrices is not the "centre" of the curved SPD manifold. Here the two means differ by '
                                f'<em>δ = {np.mean(de):.3f}</em>. The diagonal of the log-difference shows which electrodes change power most: '
                                f'<b>{", ".join(names[i] for i in np.argsort(-np.abs(np.diag(diff)))[:3])}</b>.</div>')
                        dvals = np.diag(diff)
                        ui.plot(viz.topo_fig(dvals, xy, names, height=280, title="diagonal of log(M₂⁻¹M₁) — power change map"), "rm_diag_topo")
                    else:
                        ui.kicker("Tangent-space vectors projected to 2-D (PCA)")
                        ft = go.Figure()
                        for c_ in cls:
                            idx = np.where(y == c_)[0]
                            ft.add_trace(go.Scatter(x=E[idx, 0], y=E[idx, 1], mode="markers", name=viz.pretty(c_), customdata=idx,
                                                    marker=dict(size=8, color=cc[c_], line=dict(width=0.8, color=viz.PAPER))))
                        ft.update_layout(height=470, xaxis_title="PC 1", yaxis_title="PC 2")
                        ev = ui.plot(ft, "rm_ts", select=True, name="tangent_space_map")
                        v = ui.on_click(ev, "rm_ts")
                        if v is not None:
                            st.session_state["rm_trial"] = int(v)
                        ui.figcap("6.7", "each SPD matrix is mapped to the flat tangent space at the grand mean, where ordinary linear classifiers work")
                with insp:
                    with st.container(key="inspector"):
                        ui.deck_header("Inspector", f"trial {tr}")
                        d0, d1 = D[tr, 0], D[tr, 1]
                        ok = cls[np.argmin(D[tr])] == y[tr]
                        ui.html(f'<div class="insp-big">{abs(d0 - d1):.2f}<small> margin</small></div>')
                        ui.html('<div class="kv">' + "".join(f'<div class="k">{k}</div><div class="v">{v}</div>' for k, v in [
                            ("true class", viz.pretty(y[tr])), (f"δ to {viz.pretty(cls[0])}", f"{d0:.3f}"), (f"δ to {viz.pretty(cls[1])}", f"{d1:.3f}"),
                            ("MDM says", viz.pretty(cls[np.argmin(D[tr])])), ("correct?", "yes" if ok else "no")]) + "</div>")
                        acc = np.mean(cls[np.argmin(D, 1)] == y)
                        ui.html(f'<div class="reading">On the curved manifold of covariance matrices, a trial is labelled by its nearest class mean. '
                                f'<em>{acc * 100:.0f}%</em> of trials sit on the right side. That figure is exploratory: the means saw every trial.</div>')
                        ui.html(ui.tag("affine-invariant metric", "t") + ui.tag(f"estimator {EST[est]}", ""))

            # ================================================================ xDAWN
            elif method == "xDAWN":
                from pyriemann.spatialfilters import Xdawn
                nf = max(1, ncomp // 2)

                def xd():
                    m = Xdawn(nfilter=nf, estimator=PYR_EST[est] if est != "empirical" else "scm").fit(X, y)
                    return m.filters_, m.patterns_
                try:
                    F, Pt = ui.memo("xdawn", key + str(nf), xd)
                except Exception:
                    m = Xdawn(nfilter=nf, estimator="lwf").fit(X, y)
                    F, Pt = m.filters_, m.patterns_
                nk = F.shape[0]
                comp_cls = [cls[k // nf] for k in range(nk)]
                snr = []
                for k in range(nk):
                    s = np.einsum("c,nct->nt", F[k], X)
                    ev_ = s[y == comp_cls[k]].mean(0)
                    snr.append(float(ev_.var() / (s[y == comp_cls[k]] - ev_).var()))
                st.session_state.setdefault("xd_comp", 0)
                comp = min(st.session_state["xd_comp"], nk - 1)
                best_ch = int(np.argmax(np.abs(Pt[comp])))
                with stage:
                    if not any(k in " ".join(map(str, cls)).lower() for k in ("target", "auditory", "visual")):
                        ui.note("xDAWN is built for event-related potentials (P300 and the like). Load <b>Synthetic · P300</b> from the data menu for the intended use.", "")
                    ui.kicker("Components ranked by evoked SNR — click one")
                    fb = go.Figure(go.Bar(x=[f"X{k + 1} · {viz.pretty(c_)}" for k, c_ in enumerate(comp_cls)], y=snr, customdata=list(range(nk)),
                                          marker=dict(color=[cc[c_] for c_ in comp_cls], line=dict(color=[viz.INK if k == comp else "rgba(0,0,0,0)" for k in range(nk)], width=2.5))))
                    fb.update_layout(height=210, yaxis_title="signal / noise", margin=dict(t=8))
                    ev = ui.plot(fb, "xd_bar", select=True, name="xdawn_snr")
                    v = ui.on_click(ev, "xd_bar")
                    if v is not None and int(v) != comp:
                        st.session_state["xd_comp"] = int(v)
                        ui.rerun_local()
                    a, b = st.columns(2)
                    with a:
                        ui.kicker("Filter")
                        ui.plot(viz.topo_fig(F[comp], xy, names, height=250), "xd_f", name="xdawn_filter")
                    with b:
                        ui.kicker("Pattern")
                        ui.plot(viz.topo_fig(Pt[comp], xy, names, height=250, highlight=names[best_ch]), "xd_p", name="xdawn_pattern")
                    ui.kicker(f"Enhanced response (X{comp + 1}) vs best single electrode ({names[best_ch]})")
                    s = np.einsum("c,nct->nt", F[comp], X)
                    fe = make_subplots(1, 2, subplot_titles=(f"raw {names[best_ch]}", f"xDAWN X{comp + 1}"))
                    for c_ in cls:
                        for k_, sig in enumerate([X[:, best_ch] * 1e6, s * 1e6]):
                            d = sig[y == c_]
                            mu, se = d.mean(0), d.std(0) / np.sqrt(len(d))
                            fe.add_trace(go.Scatter(x=np.r_[times, times[::-1]], y=np.r_[mu + 1.96 * se, (mu - 1.96 * se)[::-1]], fill="toself",
                                                    fillcolor=viz.rgba(cc[c_], 0.15), line_width=0, hoverinfo="skip", showlegend=False), 1, k_ + 1)
                            fe.add_trace(go.Scatter(x=times, y=mu, name=viz.pretty(c_), line=dict(color=cc[c_], width=2), showlegend=k_ == 0), 1, k_ + 1)
                    fe.update_layout(height=280)
                    fe.update_annotations(font=dict(family=viz.MONO, size=10.5))
                    ui.plot(fe, "xd_erp", name="xdawn_enhanced_erp")
                with insp:
                    with st.container(key="inspector"):
                        ui.deck_header("Inspector", f"X{comp + 1}")
                        raw_snr = (lambda d: d.mean(0).var() / (d - d.mean(0)).var())(X[y == comp_cls[comp], best_ch])
                        ui.html(f'<div class="insp-big">{snr[comp] / max(raw_snr, 1e-12):.1f}×<small> SNR gain</small></div>')
                        ui.html('<div class="kv">' + "".join(f'<div class="k">{k}</div><div class="v">{v}</div>' for k, v in [
                            ("enhances", viz.pretty(comp_cls[comp])), ("SNR xDAWN", f"{snr[comp]:.3f}"), (f"SNR {names[best_ch]}", f"{raw_snr:.3f}"),
                            ("peak electrode", names[best_ch])]) + "</div>")
                        ui.html('<div class="reading">xDAWN finds spatial filters that maximise the <em>evoked</em> (trial-averaged) signal relative to everything else. '
                                'A noisy single-trial ERP becomes readable.</div>')

            # ================================================================ PCA / Whitening
            elif method in ("PCA", "Whitening"):
                wv, V, Cm = ui.memo("pca", key, lambda: SP.pca_filters(X))
                ev_ratio = wv / wv.sum()
                st.session_state.setdefault("pc_comp", 0)
                comp = min(st.session_state["pc_comp"], len(wv) - 1)
                with stage:
                    if method == "PCA":
                        ui.kicker("Scree plot — click a component")
                        fb = go.Figure()
                        fb.add_trace(go.Bar(x=[f"PC{k + 1}" for k in range(len(wv))], y=ev_ratio * 100, customdata=list(range(len(wv))),
                                            marker=dict(color=[viz.VERM if k == comp else viz.INK for k in range(len(wv))])))
                        fb.add_trace(go.Scatter(x=[f"PC{k + 1}" for k in range(len(wv))], y=np.cumsum(ev_ratio) * 100, yaxis="y2", mode="lines+markers",
                                                line=dict(color=viz.TEAL), marker=dict(size=5), name="cumulative"))
                        fb.update_layout(height=250, yaxis_title="% variance", yaxis2=dict(overlaying="y", side="right", range=[0, 105], showgrid=False, title="cumulative %"),
                                         showlegend=False, margin=dict(t=8))
                        ev = ui.plot(fb, "pc_bar", select=True, name="pca_scree")
                        v = ui.on_click(ev, "pc_bar")
                        if v is not None and int(v) != comp:
                            st.session_state["pc_comp"] = int(v)
                            ui.rerun_local()
                        a, b = st.columns([1, 1.3])
                        with a:
                            ui.plot(viz.topo_fig(V[comp], xy, names, height=280, title=f"PC{comp + 1} loading"), "pc_topo", name="pca_topo")
                        with b:
                            s = np.einsum("c,nct->nt", V[comp], X)
                            fz = go.Figure()
                            for c_ in cls:
                                fz.add_trace(go.Scatter(x=times, y=s[y == c_].var(0) ** 0.5 * 1e6, name=viz.pretty(c_), line=dict(color=cc[c_], width=2)))
                            fz.update_layout(height=280, title="RMS over trials", xaxis_title="time (s)", yaxis_title="µV")
                            ui.plot(fz, "pc_ts")
                        ui.note("PCA is <b>unsupervised</b>: it maximises variance, not class difference. The biggest components are often background alpha or artifacts, not the ones that discriminate.", "")
                    else:
                        Wm = SP.whitening_matrix(Cm)
                        Cw = Wm @ Cm @ Wm.T
                        hm = make_subplots(1, 2, subplot_titles=("covariance before", "after whitening ≈ identity"), horizontal_spacing=0.06)
                        hm.add_trace(go.Heatmap(z=(Cm / np.abs(Cm).max())[::-1], x=names, y=names[::-1], colorscale=viz.DIVERGING, zmid=0, showscale=False), 1, 1)
                        hm.add_trace(go.Heatmap(z=Cw[::-1], x=names, y=names[::-1], colorscale=viz.DIVERGING, zmid=0, zmin=-1, zmax=1, showscale=False), 1, 2)
                        hm.update_xaxes(tickfont=dict(size=8))
                        hm.update_yaxes(tickfont=dict(size=8))
                        hm.update_layout(height=380)
                        hm.update_annotations(font=dict(family=viz.MONO, size=10.5))
                        ui.plot(hm, "wh_heat", name="whitening")
                        fe = go.Figure()
                        fe.add_trace(go.Scatter(y=wv / wv.max(), mode="lines+markers", name="before", line=dict(color=viz.INK)))
                        fe.add_trace(go.Scatter(y=np.ones_like(wv), mode="lines", name="after", line=dict(color=viz.VERM, dash="dash")))
                        fe.update_layout(height=220, yaxis_type="log", xaxis_title="eigenvalue index", yaxis_title="relative eigenvalue")
                        ui.plot(fe, "wh_eig")
                        ui.note("Whitening rotates and rescales so that every direction has unit variance and no correlations. CSP is exactly "
                                "<b>whitening + a rotation</b> that sorts the whitened directions by class variance ratio.", "t")
                with insp:
                    with st.container(key="inspector"):
                        ui.deck_header("Inspector", method)
                        k95 = int(np.searchsorted(np.cumsum(ev_ratio), 0.95) + 1)
                        ui.html(f'<div class="insp-big">{k95}<small> PCs → 95 %</small></div>')
                        ui.html('<div class="kv">' + "".join(f'<div class="k">{k}</div><div class="v">{v}</div>' for k, v in [
                            ("PC1 share", f"{ev_ratio[0] * 100:.1f}%"), ("condition no.", f"{wv[0] / max(wv[-1], 1e-30):.2g}"),
                            ("rank", int(np.sum(wv > wv[0] * 1e-10))), ("channels", len(wv))]) + "</div>")
                        if wv[-1] < wv[0] * 1e-8:
                            ui.note("The covariance is rank-deficient (average reference or interpolation removes one dimension). Use shrinkage before inverting.", "v")

            # ================================================================ CAR / Laplacian
            else:
                Xr, yr, _, _ = C.get_Xy(ep, band, window, pair)
                if pct < 100:
                    Xr, yr = Xr[keep], yr[keep]
                Xa = SP.pre_spatial(Xr, method, xyz)
                Lb, La = np.log(Xr.var(2)), np.log(Xa.var(2))
                if len(cls) == 2:
                    db = Lb[yr == cls[0]].mean(0) - Lb[yr == cls[1]].mean(0)
                    da = La[yr == cls[0]].mean(0) - La[yr == cls[1]].mean(0)
                else:
                    db, da = Lb.mean(0), La.mean(0)
                sb = np.array([SP.separation_auc(Lb[:, i], yr) for i in range(n_ch)])
                sa = np.array([SP.separation_auc(La[:, i], yr) for i in range(n_ch)])
                st.session_state.setdefault("ref_ch", "C3" if "C3" in names else names[0])
                chn = st.session_state["ref_ch"] if st.session_state["ref_ch"] in names else names[0]
                with stage:
                    ui.kicker(f"Log-power difference {viz.pretty(cls[0])} − {viz.pretty(cls[-1])}: before vs after {method} — click a sensor")
                    a, b = st.columns(2)
                    cm = max(np.abs(db).max(), np.abs(da).max())
                    with a:
                        ev1 = ui.plot(viz.topo_fig(db, xy, names, height=300, title="before", highlight=chn, cmax=np.abs(db).max()), "ref_b", select=True, name="reref_before")
                    with b:
                        ev2 = ui.plot(viz.topo_fig(da, xy, names, height=300, title=f"after {method}", highlight=chn, cmax=np.abs(da).max()), "ref_a", select=True, name="reref_after")
                    for evx, kx in ((ev1, "ref_b"), (ev2, "ref_a")):
                        v = ui.on_click(evx, kx)
                        if v and v != chn:
                            st.session_state["ref_ch"] = v
                            ui.rerun_local()
                    ci = names.index(chn)
                    a, b = st.columns([1.2, 1])
                    with a:
                        fv = go.Figure()
                        for k_, (Lx, lab) in enumerate([(Lb, "before"), (La, "after")]):
                            for c_ in cls:
                                fv.add_trace(go.Box(y=Lx[yr == c_, ci], x=[lab] * int(np.sum(yr == c_)), name=viz.pretty(c_), marker_color=cc[c_],
                                                    legendgroup=c_, showlegend=k_ == 0, boxpoints="all", jitter=0.4, pointpos=0))
                        fv.update_layout(height=280, boxmode="group", yaxis_title=f"log-power at {chn}", title=f"{chn}: separation {sb[ci]:.2f} → {sa[ci]:.2f}")
                        ui.plot(fv, "ref_box")
                    with b:
                        Rb = np.corrcoef(Xr.transpose(1, 0, 2).reshape(n_ch, -1))
                        Ra = np.corrcoef(Xa.transpose(1, 0, 2).reshape(n_ch, -1))
                        off = ~np.eye(n_ch, dtype=bool)
                        fh = go.Figure()
                        fh.add_trace(go.Histogram(x=np.abs(Rb[off]), name="before", marker_color=viz.MUTED, opacity=0.7, nbinsx=30))
                        fh.add_trace(go.Histogram(x=np.abs(Ra[off]), name="after", marker_color=viz.VERM, opacity=0.7, nbinsx=30))
                        fh.update_layout(height=280, barmode="overlay", xaxis_title="|channel correlation|", title="volume conduction")
                        ui.plot(fh, "ref_corr")
                with insp:
                    with st.container(key="inspector"):
                        ui.deck_header("Inspector", chn)
                        ui.html(f'<div class="insp-big">{sa.max():.2f}<small> best after</small></div>')
                        ui.html('<div class="kv">' + "".join(f'<div class="k">{k}</div><div class="v">{v}</div>' for k, v in [
                            ("best before", f"{names[int(np.argmax(sb))]} · {sb.max():.2f}"), ("best after", f"{names[int(np.argmax(sa))]} · {sa.max():.2f}"),
                            ("mean |r| before", f"{np.abs(Rb[off]).mean():.2f}"), ("mean |r| after", f"{np.abs(Ra[off]).mean():.2f}")]) + "</div>")
                        ui.html('<div class="reading">' + ("The common average reference removes what <em>all</em> electrodes share (reference noise, far-field activity)." if method == "CAR"
                                else "The surface Laplacian subtracts each electrode's <em>neighbours</em>. A spatial high-pass that sharpens focal sources like the hand area under C3/C4.") + "</div>")
                        ui.html(ui.tag("stateless · no fitting", "g") + ui.tag("applied directly on Apply", ""))

        stage_area()

        # ---------------------------------------------------------------- explain + export
        code = f"""import numpy as np
    from mne.decoding import CSP            # or: from spatial import CSP (this app's transparent version)
    X = epochs.copy().filter({band[0]}, {band[1]}).crop({window[0]}, {window[1]}).get_data()
    y = epochs.metadata['label'].to_numpy()
    csp = CSP(n_components={ncomp}, reg={'"ledoit_wolf"' if est == 'ledoit_wolf' else ('"oas"' if est == 'oas' else None)}, log=True)
    csp.fit(X, y)                           # exploratory only — refit inside each CV fold for evaluation!
    patterns = csp.patterns_                # interpret these, not csp.filters_
    """
        if method == "Riemann":
            code = f"""from pyriemann.estimation import Covariances
    from pyriemann.geometry.mean import mean_riemann
    from pyriemann.geometry.distance import distance_riemann
    C = Covariances('{PYR_EST[est]}').fit_transform(X)
    means = [mean_riemann(C[y == c]) for c in np.unique(y)]
    D = np.array([[distance_riemann(c, m) for m in means] for c in C])
    """
        ui.export_drawer("spatial", code, new_sp)

    with st.container(key="mode_rail"):
        mode = ui.seg("Mode", ["Concepts", "Your data"], default="Concepts", key="sp_mode") or "Concepts"
    if mode == "Concepts":
        recipe_panel(dict(sp), "sp_apply_c", editable=False)
        concepts_spatial.render()
    else:
        your_data()

    ui.explain("Spatial filtering in one minute",
               ["Every electrode hears many sources at once (<b>volume conduction</b>). A spatial filter <b>w</b> mixes the channels into one virtual channel <b>wᵀx</b>. "
                "<b>CSP</b> picks w so that its variance is as large as possible for one class while as small as possible for the other. That is a generalised eigenproblem, "
                "and the eigenvalue λ tells you how lopsided it is. <b>xDAWN</b> does the same for evoked responses. <b>Riemannian</b> methods skip filters and compare whole covariance matrices "
                "with a geometry that respects their curvature.",
                "Interpret <b>patterns</b>, not filters: a filter also has to cancel noise, so its weights can be large where there is no signal at all. "
                "Every fit on this page uses all trials, which is fine for <i>looking</i>, never for <i>evaluating</i>. Classification refits everything inside each training fold."],
               [("Why do CSP components come in pairs from both ends of the eigenvalue spectrum?", "λ close to 1 means 'big for class A, small for B'; close to 0 the opposite. Both ends are discriminative; the middle (λ ≈ 0.5) carries no class information."),
                ("Right-hand imagery causes ERD over C3. Which class does a C3-focused component 'favour'?", "Left hand. During right-hand imagery the C3 rhythm drops, so the component has more variance during the other class."),
                ("Why would CSP fitted on all trials give an over-optimistic accuracy?", "The filters were chosen using the test trials' labels, so the test data no longer tell you how a new trial would do. That is leakage."),
                ("What does shrinkage do to a covariance estimate and when do you need it?", "It pulls the estimate towards a scaled identity, trading a little bias for much less variance. You need it with few trials, many channels, or rank-deficient data."),
                ("Why is the Euclidean mean of covariance matrices a poor 'average'?", "SPD matrices live on a curved manifold. The arithmetic mean inflates the determinant (swelling effect). The Riemannian mean respects the geometry.")])
    downloads.downloads_panel(o, "sp")
