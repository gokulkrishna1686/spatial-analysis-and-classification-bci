"""Concept demos for the Spatial analysis page: tiny, fully interactive 2-D
models that show *why* and *how* spatial filters work. Computations live in toys.py."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import core
import toys as T
import ui
import viz

A_COL, B_COL = viz.TEAL, viz.VERM          # left hand, right hand
DEMOS = ["Why spatial filters?", "Spin a filter", "Whiten, then rotate", "Filters vs patterns", "Riemann in 2×2"]


def _notice(lines):
    ui.html('<div class="reading"><b>What to notice.</b> ' + " ".join(lines) + "</div>")


def _square(fig, lim, height=420):
    fig.update_xaxes(range=[-lim, lim], zeroline=True, zerolinecolor=viz.RULE)
    fig.update_yaxes(range=[-lim, lim], scaleanchor="x", scaleratio=1, zeroline=True, zerolinecolor=viz.RULE)
    fig.update_layout(height=height, margin=dict(l=50, r=10, t=30, b=45))
    return fig


def _cloud(fig, XA, XB, CA=None, CB=None, row=None, col=None, legend=True, size=5):
    kw = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Scatter(x=XA[:, 0], y=XA[:, 1], mode="markers", name="left hand", legendgroup="a", showlegend=legend,
                             marker=dict(size=size, color=A_COL, opacity=0.45)), **kw)
    fig.add_trace(go.Scatter(x=XB[:, 0], y=XB[:, 1], mode="markers", name="right hand", legendgroup="b", showlegend=legend,
                             marker=dict(size=size, color=B_COL, opacity=0.45)), **kw)
    for C, colr in ((CA, A_COL), (CB, B_COL)):
        if C is not None:
            ex, ey = T.ellipse(C)
            fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=colr, width=2.5), showlegend=False, hoverinfo="skip"), **kw)


# ---------------------------------------------------------------------------
@ui.guarded_fragment("Why spatial filters?")
def render_volume():
    ui.html('<div class="big-q">Every electrode hears <em>several</em> brain sources at once.</div>')
    spread = st.slider("How far each source spreads over the scalp (volume conduction)", 0.0, 1.0, 0.45, 0.05, key="vc_spread",
                       help="Real head tissue blurs every source over many centimetres of scalp.")
    names, a1, a2, t, c3, c4, s1, s2, r = T.volume_conduction(spread)
    xy = viz.project_2d(core.montage_xyz(names))
    a, b = st.columns([1.15, 1], gap="large")
    with a:
        fig = viz.topo_grid([a1, a2], xy, names, ["source under C3 (left hand area)", "source under C4 (right hand area)"],
                            cols=2, height=260, colorscale=viz.SEQ, cmax_shared=True)
        for tr in fig.data:
            if tr.type == "contour":
                tr.update(zmin=0, zmax=1)
        ui.plot(fig, "vc_topo")
        ui.figcap("C.1", "how strongly each electrode picks up each source (dark = strong)")
    with b:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                            subplot_titles=("the two brain sources (what we want)", "electrodes C3 and C4 (what we record)"))
        fig.add_trace(go.Scatter(x=t, y=s1, name="source 1", line=dict(color=A_COL, width=1.4)), 1, 1)
        fig.add_trace(go.Scatter(x=t, y=s2 - 4, name="source 2", line=dict(color=B_COL, width=1.4)), 1, 1)
        fig.add_trace(go.Scatter(x=t, y=c3, name="C3", line=dict(color=viz.INK, width=1.4)), 2, 1)
        fig.add_trace(go.Scatter(x=t, y=c4 - 4 * max(1, np.abs(c3).max() / 2.5), name="C4", line=dict(color=viz.MUTED, width=1.4)), 2, 1)
        fig.update_yaxes(showticklabels=False)
        fig.update_xaxes(title_text="time (s)", row=2, col=1)
        fig.update_layout(height=330, showlegend=False, margin=dict(l=20, r=10, t=30, b=40))
        ui.plot(fig, "vc_traces")
        ui.html(f'<div class="kv"><div class="k">correlation between C3 and C4</div><div class="v">{r:+.2f}</div>'
                f'<div class="k">the two sources are</div><div class="v">independent (r ≈ 0)</div></div>')
    _notice(["Move the slider right. The sources never change, but the electrode signals look more and more alike, and their correlation climbs.",
             "A single electrode is a <b>mixture</b>. A spatial filter combines electrodes (e.g. C3 minus its neighbours, or a CSP filter) to undo the mixing."])


# ---------------------------------------------------------------------------
@ui.guarded_fragment("Spin a filter")
def render_spin():
    ui.html('<div class="big-q">A spatial filter is a <em>direction</em>. CSP picks the direction where the two classes differ most in power.</div>')
    c = st.columns(3)
    erd = c[0].slider("ERD strength (class difference)", 0.0, 0.9, 0.6, 0.05, key="sp_erd", help="How much the mu rhythm drops on the side opposite the imagined hand.")
    mix = c[1].slider("Volume conduction between C3 and C4", 0.0, 0.9, 0.35, 0.05, key="sp_mix")
    shared = c[2].slider("Shared noise (seen by both electrodes)", 0.0, 2.0, 0.8, 0.1, key="sp_shared")
    XA, XB, CA, CB = T.two_class_2d(n=250, erd=erd, mix=mix, shared=shared, seed=1)
    lam_csp, _, ang = T.csp_2d(CA, CB)
    best_deg = float(np.rad2deg(ang[0]))
    st.session_state.setdefault("spin_theta", 20.0)
    if "spin_pending" in st.session_state:
        st.session_state["spin_theta"] = float(st.session_state.pop("spin_pending"))
    b1, b2 = st.columns([4, 1], vertical_alignment="bottom")
    theta_deg = b1.slider("Filter direction θ (degrees)", 0.0, 180.0, step=1.0, key="spin_theta",
                          help="w = (cos θ, sin θ). The filtered signal is w₁·C3 + w₂·C4.")
    if b2.button("Jump to CSP", key="spin_jump", use_container_width=True, help="Snap to the CSP solution (largest eigenvalue)."):
        st.session_state["spin_pending"] = float(round(best_deg))
        ui.rerun_local()
    th = np.deg2rad(theta_deg)
    thetas = np.linspace(0, np.pi, 181)
    vA, vB, lam = T.projected_variance(CA, CB, thetas)
    a, b = st.columns([1, 1.15], gap="large")
    with a:
        fig = go.Figure()
        _cloud(fig, XA, XB, CA, CB)
        lim = float(np.percentile(np.abs(np.r_[XA, XB]), 99)) * 1.15
        w = np.array([np.cos(th), np.sin(th)])
        fig.add_trace(go.Scatter(x=[-w[0] * lim, w[0] * lim], y=[-w[1] * lim, w[1] * lim], mode="lines", name="filter w",
                                 line=dict(color=viz.INK, width=3)))
        fig.add_annotation(x=w[0] * lim * 0.9, y=w[1] * lim * 0.9, text="w", showarrow=False, font=dict(size=16, color=viz.INK), xshift=10)
        fig.update_xaxes(title_text="C3 (µV)")
        fig.update_yaxes(title_text="C4 (µV)")
        ui.plot(_square(fig, lim), "spin_scatter")
        ui.figcap("C.2", "each dot is one time sample; ellipses = 2-σ covariance of each class")
    with b:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("power of the filtered signal per class", "CSP score λ = left / (left + right)"))
        deg = np.rad2deg(thetas)
        fig.add_trace(go.Scatter(x=deg, y=vA, name="left hand", mode="lines+markers", marker=dict(size=4, color=A_COL), line=dict(color=A_COL, width=2.5)), 1, 1)
        fig.add_trace(go.Scatter(x=deg, y=vB, name="right hand", mode="lines+markers", marker=dict(size=4, color=B_COL), line=dict(color=B_COL, width=2.5)), 1, 1)
        fig.add_trace(go.Scatter(x=deg, y=lam, name="λ", mode="lines+markers", marker=dict(size=4, color=viz.PALETTE[3]),
                                 line=dict(color=viz.PALETTE[3], width=2.5), showlegend=False), 2, 1)
        fig.add_hline(0.5, line=dict(color=viz.RULE, dash="dot"), row=2, col=1)
        for k, nm in ((0, "CSP filter 1"), (1, "CSP filter 2")):
            d = float(np.rad2deg(ang[k]))
            fig.add_trace(go.Scatter(x=[d], y=[lam_csp[k]], mode="markers+text", text=[nm], textposition="top center",
                                     marker=dict(size=11, color=viz.INK, symbol="diamond"), showlegend=False, hoverinfo="skip"), 2, 1)
        fig.add_vline(theta_deg, line=dict(color=viz.INK, width=1.5))
        fig.update_xaxes(title_text="filter direction θ (degrees), click a curve to set it", row=2, col=1)
        fig.update_yaxes(range=[0, 1], row=2, col=1)
        fig.update_layout(height=430, legend=dict(y=1.08), margin=dict(l=45, r=10, t=40, b=45))
        ev = ui.plot(fig, "spin_curve", select=True)
        v = ui.on_click(ev, "spin_curve", "x")
        if v is not None and abs(float(v) - theta_deg) > 0.5:
            st.session_state["spin_pending"] = float(round(float(v)))
            ui.rerun_local()
    i = int(np.argmin(np.abs(np.rad2deg(thetas) - theta_deg)))
    ratio = vA[i] / vB[i]
    ui.html(f'<div class="kv" style="max-width:560px"><div class="k">your filter at θ = {theta_deg:.0f}°</div><div class="v">λ = {lam[i]:.2f} · power ratio {max(ratio, 1 / ratio):.2f}×</div>'
            f'<div class="k">CSP optimum at θ = {best_deg:.0f}°</div><div class="v">λ = {lam_csp[0]:.2f}</div></div>')
    _notice(["Turn the filter: the two class powers rise and fall at different angles. CSP chooses the angle where one class is <b>much</b> stronger than the other (λ near 1),",
             "and the second filter where it is the other way round (λ near 0). Increase the shared noise: the best angle moves, because the filter learns to <b>subtract</b> what both electrodes have in common."])


# ---------------------------------------------------------------------------
@ui.guarded_fragment("Whiten, then rotate")
def render_whiten():
    ui.html('<div class="big-q">How CSP is computed: <em>whiten</em> the pooled data, then <em>rotate</em> to the axes where the classes differ.</div>')
    t = st.slider("Step: 0 = raw electrodes · 1 = whitened · 2 = rotated to the CSP axes", 0.0, 2.0, 0.0, 0.05, key="wr_t")
    XA, XB, CA, CB = T.two_class_2d(n=250, erd=0.6, mix=0.35, shared=0.8, seed=4)
    P, U, angle = T.whiten_rotate(CA, CB)
    M = T.whiten_rotate_transform(P, angle, t)
    YA, YB = XA @ M.T, XB @ M.T
    CA_t, CB_t = M @ CA @ M.T, M @ CB @ M.T
    a, b = st.columns([1.2, 1], gap="large")
    with a:
        fig = go.Figure()
        _cloud(fig, YA, YB, CA_t, CB_t)
        ex, ey = T.ellipse((CA_t + CB_t) / 2)
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", name="pooled (both classes)", line=dict(color=viz.INK, width=2, dash="dash")))
        lim = max(4.0, float(np.percentile(np.abs(np.r_[YA, YB]), 99)) * 1.2)
        names = ("C3", "C4") if t < 0.5 else (("whitened 1", "whitened 2") if t < 1.5 else ("CSP component 1", "CSP component 2"))
        fig.update_xaxes(title_text=names[0])
        fig.update_yaxes(title_text=names[1])
        ui.plot(_square(fig, lim, 440), "wr_scatter")
    with b:
        stage = "raw" if t < 0.5 else ("whitened" if t < 1.5 else "rotated")
        text = {
            "raw": "The two classes overlap and both ellipses are tilted the same way, because C3 and C4 share signal. Variance alone does not tell them apart well.",
            "whitened": "Whitening rescales the data so that the <b>pooled</b> covariance (dashed) becomes a circle: no direction is louder than another overall. "
                        "What is left are the <b>differences</b> between the classes: the two ellipses are now mirror images.",
            "rotated": "A rotation lines the axes up with the class ellipses. Along axis 1 the left-hand class is wide and the right-hand class narrow; along axis 2 the opposite. "
                       "These axes are the CSP filters, and the variance along them is the CSP feature.",
        }[stage]
        ui.html(f'<div class="card-h"><span>Stage: {stage}</span><span class="card-h-r">t = {t:.2f}</span></div><p>{text}</p>')
        vA, vB = np.diag(CA_t), np.diag(CB_t)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["axis 1", "axis 2"], y=vA, name="left hand", marker_color=A_COL))
        fig.add_trace(go.Bar(x=["axis 1", "axis 2"], y=vB, name="right hand", marker_color=B_COL))
        fig.update_layout(height=240, barmode="group", yaxis_title="variance", margin=dict(l=45, r=10, t=20, b=30))
        ui.plot(fig, "wr_bars")
        ui.figcap("C.3", "variance of each class along the current axes")
    _notice(["Slide from 0 to 2. At t = 1 the dashed pooled ellipse is a circle. At t = 2 the bars show one class big and the other small on each axis.",
             "This is exactly what the app's CSP does with 24 electrodes instead of 2."])


# ---------------------------------------------------------------------------
@ui.guarded_fragment("Filters vs patterns")
def render_haufe():
    ui.html('<div class="big-q">Why we look at <em>patterns</em>, not filters, to find where the signal comes from.</div>')
    noise = st.slider("Strength of the common noise on both channels", 0.0, 3.0, 1.5, 0.1, key="hf_noise")
    w, a_, X, s = T.haufe_demo(noise=noise, n=3000, seed=0)
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        n = 400
        tt = np.arange(n) / 250
        fig = go.Figure()
        off = 3 + 2 * noise
        fig.add_trace(go.Scatter(x=tt, y=s[:n] + off, name="true signal (only on channel 1)", line=dict(color=viz.TEAL, width=1.6)))
        fig.add_trace(go.Scatter(x=tt, y=X[0, :n], name="channel 1 = signal + noise", line=dict(color=viz.INK, width=1.2)))
        fig.add_trace(go.Scatter(x=tt, y=X[1, :n] - off, name="channel 2 = noise only", line=dict(color=viz.MUTED, width=1.2)))
        fig.add_trace(go.Scatter(x=tt, y=(w @ X[:, :n]) / w[0] - 2 * off, name="filter output w·x", line=dict(color=viz.VERM, width=1.6)))
        fig.update_yaxes(showticklabels=False)
        fig.update_layout(height=380, xaxis_title="time (s)", legend=dict(orientation="h", y=1.12), margin=dict(l=20, r=10, t=40, b=40))
        ui.plot(fig, "hf_traces")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["channel 1", "channel 2"], y=w, name="filter w", marker_color=viz.VERM))
        fig.add_trace(go.Bar(x=["channel 1", "channel 2"], y=a_, name="pattern a", marker_color=viz.TEAL))
        fig.add_hline(0, line=dict(color=viz.RULE))
        fig.update_layout(height=380, barmode="group", yaxis=dict(range=[-1.2, 1.2], title="weight (normalised)"), margin=dict(l=50, r=10, t=30, b=30))
        ui.plot(fig, "hf_bars")
    _notice(["The best filter gives channel 2 a big <b>negative</b> weight, although channel 2 contains no signal at all: it is used to <b>cancel the noise</b>.",
             "The pattern a = C·w / (wᵀ·C·w) puts the weight where the signal really is (channel 1). That is why the app always shows patterns on the head (Haufe et al., 2014)."])


# ---------------------------------------------------------------------------
@ui.guarded_fragment("Riemann in 2×2")
def render_riemann():
    ui.html('<div class="big-q">Covariance matrices are <em>shapes</em>. How should we average two shapes?</div>')
    c = st.columns(3)
    elong = c[0].slider("Elongation of each shape", 1.0, 8.0, 5.0, 0.5, key="rm_el")
    turn = c[1].slider("Angle between the shapes (degrees)", 0, 90, 90, 5, key="rm_turn")
    t = c[2].slider("Walk from A (0) to B (1)", 0.0, 1.0, 0.5, 0.05, key="rm_t")
    D = np.diag([elong, 1.0 / elong])
    A = D
    R = T.rot(np.deg2rad(turn))
    B = R @ D @ R.T
    ts = np.linspace(0, 1, 41)
    euc, geo = T.riemann_path(A, B, ts)
    i = int(round(t * 40))
    a, b = st.columns([1.1, 1], gap="large")
    with a:
        fig = go.Figure()
        for C, nm, col, dash in ((A, "A", viz.TEAL, "solid"), (B, "B", viz.VERM, "solid"),
                                 (euc[i], "Euclidean average", viz.MUTED, "dash"), (geo[i], "Riemannian (geodesic)", viz.PALETTE[3], "solid")):
            ex, ey = T.ellipse(C, k=1.0)
            fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", name=nm, line=dict(color=col, width=3 if nm[0] in "ER" else 2, dash=dash)))
        lim = np.sqrt(elong) * 1.15
        ui.plot(_square(fig, lim, 420), "rm_shapes")
    with b:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts, y=[np.linalg.det(m) for m in euc], name="Euclidean", line=dict(color=viz.MUTED, width=2.5, dash="dash")))
        fig.add_trace(go.Scatter(x=ts, y=[np.linalg.det(m) for m in geo], name="Riemannian", line=dict(color=viz.PALETTE[3], width=2.5)))
        fig.add_vline(t, line=dict(color=viz.INK, width=1.2))
        fig.update_layout(height=320, xaxis_title="position along the path", yaxis_title="determinant (≈ size of the shape)", margin=dict(l=55, r=10, t=20, b=45))
        ui.plot(fig, "rm_det")
        ui.html(f'<div class="kv"><div class="k">size at t = {t:.2f}, Euclidean</div><div class="v">{np.linalg.det(euc[i]):.2f}</div>'
                f'<div class="k">size at t = {t:.2f}, Riemannian</div><div class="v">{np.linalg.det(geo[i]):.2f}</div>'
                f'<div class="k">size of A and of B</div><div class="v">1.00</div></div>')
    _notice(["Two thin ellipses of size 1. Their ordinary (Euclidean) average is a fat ellipse that is much <b>bigger</b> than either: the swelling effect.",
             "The Riemannian average walks along the curved space of covariance matrices and keeps the size. MDM and tangent-space classifiers use this average."])


RENDER = {DEMOS[0]: render_volume, DEMOS[1]: render_spin, DEMOS[2]: render_whiten, DEMOS[3]: render_haufe, DEMOS[4]: render_riemann}


def render():
    demo = ui.seg("Concept", DEMOS, default=DEMOS[0], key="sp_concept") or DEMOS[0]
    RENDER[demo]()
