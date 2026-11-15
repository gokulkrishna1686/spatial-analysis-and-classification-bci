"""Visual system: palette, Plotly template, topomaps (2-D, clickable) and a 3-D head."""
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy.interpolate import RBFInterpolator

PAPER = "#FFFFFF"
CARD = "#FFFFFF"
INK = "#1B2130"
INK2 = "#485062"
MUTED = "#6B7383"
RULE = "#D5DAE1"
RULE2 = "#EEF0F3"
VERM = "#D6492A"
TEAL = "#12707A"
OCHRE = "#C08A1E"
INDIGO = "#5B6BC0"
GREEN = "#2F7D4F"
PALETTE = [TEAL, VERM, OCHRE, INDIGO, "#8A5A83", "#5E7F3A", MUTED]
KNOWN = {"left_hand": TEAL, "right_hand": VERM, "feet": OCHRE, "tongue": INDIGO,
         "nontarget": TEAL, "target": VERM, "auditory/left": TEAL, "auditory/right": INDIGO,
         "visual/left": VERM, "visual/right": OCHRE}

DIVERGING = [[0.0, "#0B4F57"], [0.18, TEAL], [0.38, "#A9CDCF"], [0.5, "#F6F6F4"],
             [0.62, "#EDB39F"], [0.82, VERM], [1.0, "#8E2A14"]]
SEQ = [[0.0, "#F6F7F9"], [0.35, "#E7C9A0"], [0.7, "#D6492A"], [1.0, "#5B1A0C"]]
SEQ_INK = [[0.0, "#F6F7F9"], [0.5, "#9DB0E6"], [1.0, "#2F5BD3"]]

MONO = "IBM Plex Sans, system-ui, sans-serif"   # chart text uses the UI sans-serif for readability
SANS = "IBM Plex Sans, system-ui, sans-serif"
SERIF = "Fraunces, Georgia, serif"


def class_color(label, classes=None):
    if label in KNOWN:
        return KNOWN[label]
    classes = list(classes or [])
    i = classes.index(label) if label in classes else abs(hash(label)) % len(PALETTE)
    return PALETTE[i % len(PALETTE)]


def class_colors(classes):
    out, used = {}, set()
    for i, c in enumerate(classes):
        col = KNOWN.get(c)
        if col is None or col in used:
            col = next(p for p in PALETTE if p not in used)
        out[c] = col
        used.add(col)
    return out


def pretty(label):
    return str(label).replace("_", " ")


def rgba(hex_color, a):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


_axis = dict(showgrid=True, gridcolor=RULE2, gridwidth=1, zeroline=False, showline=True, linecolor=RULE,
             linewidth=1, ticks="", tickfont=dict(family=SANS, size=12, color=MUTED),
             title=dict(font=dict(family=SANS, size=12.5, color=INK2)))
pio.templates["studio"] = go.layout.Template(layout=dict(
    font=dict(family=SANS, size=12.5, color=INK),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=PALETTE, xaxis=_axis, yaxis=_axis,
    margin=dict(l=52, r=18, t=40, b=44),
    title=dict(font=dict(family=SANS, size=14, color=INK), x=0.0, xanchor="left", y=0.98),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, bgcolor="rgba(0,0,0,0)",
                font=dict(family=SANS, size=12, color=INK2)),
    hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=RULE, font=dict(family=SANS, size=12, color=INK)),
    coloraxis=dict(colorbar=dict(outlinewidth=0, thickness=10, tickfont=dict(family=SANS, size=11))),
))
pio.templates.default = "studio"


# --------------------------------------------------------------------------
# Sensor positions
# --------------------------------------------------------------------------

def sensor_xyz(info):
    import mne
    picks = mne.pick_types(info, eeg=True)
    names = [info.ch_names[i] for i in picks]
    xyz = np.array([info["chs"][i]["loc"][:3] for i in picks], float)
    ok = np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 0)
    if ok.sum() < len(names):
        try:
            from core import montage_xyz
            xyz = montage_xyz(names)
            ok = np.ones(len(names), bool)
        except Exception:
            ang = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
            xyz = np.c_[0.07 * np.cos(ang), 0.07 * np.sin(ang), 0.05 * np.ones(len(names))]
    return names, xyz


def project_2d(xyz):
    n = np.linalg.norm(xyz, axis=1) + 1e-12
    theta = np.arccos(np.clip(xyz[:, 2] / n, -1, 1))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    r = theta / (np.pi / 2)
    if r.max() > 0.97:
        r = r / r.max() * 0.97
    return np.c_[r * np.cos(phi), r * np.sin(phi)]


def sensor_xy(info):
    names, xyz = sensor_xyz(info)
    return names, project_2d(xyz)


# --------------------------------------------------------------------------
# 2-D topomap
# --------------------------------------------------------------------------

def _grid(xy, values, res=64):
    v = np.asarray(values, float)
    scale = np.max(np.abs(v)) or 1.0
    g = np.linspace(-1.02, 1.02, res)
    gx, gy = np.meshgrid(g, g)
    pts = np.c_[gx.ravel(), gy.ravel()]
    try:
        f = RBFInterpolator(xy, v / scale, kernel="thin_plate_spline", smoothing=1e-3)
        z = f(pts).reshape(res, res) * scale
    except Exception:
        from scipy.interpolate import griddata
        z = griddata(xy, v, pts, method="nearest").reshape(res, res)
    z[np.hypot(gx, gy) > 1.0] = np.nan
    return g, z


def head_shapes(color="#7C8494", width=1.2):
    t = np.linspace(0, 2 * np.pi, 120)
    circle = "M " + " L ".join(f"{np.cos(a):.4f},{np.sin(a):.4f}" for a in t) + " Z"
    nose = "M -0.09,0.995 L 0,1.11 L 0.09,0.995"
    ear_l = "M -0.995,0.1 C -1.07,0.12 -1.09,-0.1 -0.995,-0.12"
    ear_r = "M 0.995,0.1 C 1.07,0.12 1.09,-0.1 0.995,-0.12"
    return [dict(type="path", path=p, line=dict(color=color, width=width), fillcolor="rgba(0,0,0,0)")
            for p in (circle, nose, ear_l, ear_r)]


def add_topomap(fig, values, xy, names, row=None, col=None, cmax=None, colorscale=None, show_names=False,
                sensor_size=5, sensor_colors=None, sensor_symbols=None, highlight=None, showscale=False,
                hover_unit="", contours=True, sensor_text=None):
    g, z = _grid(xy, values)
    cm = cmax or (np.nanmax(np.abs(values)) or 1.0)
    kw = dict(row=row, col=col) if row is not None else {}
    fig.add_trace(go.Contour(x=g, y=g, z=z, colorscale=colorscale or DIVERGING, zmin=-cm, zmax=cm,
                             contours=dict(coloring="heatmap", showlines=contours, start=-cm, end=cm, size=cm / 5),
                             line=dict(width=0.6, color="rgba(23,25,30,0.28)"), showscale=showscale,
                             hoverinfo="skip", connectgaps=False,
                             colorbar=dict(thickness=8, len=0.6, outlinewidth=0, tickfont=dict(family=SANS, size=9))),
                  **kw)
    cols = sensor_colors if sensor_colors is not None else [INK] * len(names)
    sizes = [sensor_size * (2.0 if highlight is not None and n in np.atleast_1d(highlight) else 1.0) for n in names]
    txt = sensor_text if sensor_text is not None else [f"{n}<br>{v:+.3g}{hover_unit}" for n, v in zip(names, values)]
    fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers+text" if show_names else "markers",
                             text=names if show_names else None, textposition="top center",
                             textfont=dict(family=SANS, size=8.5, color=INK2),
                             marker=dict(size=sizes, color=cols, symbol=sensor_symbols or "circle",
                                         line=dict(width=1, color=PAPER)),
                             customdata=names, hovertext=txt, hoverinfo="text", showlegend=False), **kw)
    for s in head_shapes():
        fig.add_shape(**s, **kw)
    ax = dict(visible=False, range=[-1.18, 1.18], showgrid=False, zeroline=False)
    if row is None:
        fig.update_xaxes(**ax)
        fig.update_yaxes(**ax, scaleanchor="x", scaleratio=1)
    else:
        fig.update_xaxes(**ax, row=row, col=col)
        fig.update_yaxes(**ax, row=row, col=col)
    return fig


def topo_fig(values, xy, names, title=None, height=300, **kw):
    fig = go.Figure()
    add_topomap(fig, values, xy, names, **kw)
    fig.update_layout(height=height, margin=dict(l=4, r=4, t=34 if title else 6, b=4), title=title,
                      dragmode=False)
    return fig


def topo_grid(list_values, xy, names, titles, cols=None, height=220, cmax_shared=False, **kw):
    from plotly.subplots import make_subplots
    n = len(list_values)
    cols = cols or n
    rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, horizontal_spacing=0.02, vertical_spacing=0.08)
    cm = max(np.nanmax(np.abs(v)) for v in list_values) if cmax_shared else None
    for i, v in enumerate(list_values):
        add_topomap(fig, v, xy, names, row=i // cols + 1, col=i % cols + 1, cmax=cm, sensor_size=3, **kw)
    for i in range(1, rows * cols + 1):
        fig.update_yaxes(scaleanchor=f"x{i if i > 1 else ''}", scaleratio=1, row=(i - 1) // cols + 1, col=(i - 1) % cols + 1)
    fig.update_layout(height=height * rows, margin=dict(l=4, r=4, t=30, b=4), dragmode=False)
    fig.update_annotations(font=dict(family=SANS, size=12, color=INK2))
    return fig


# --------------------------------------------------------------------------
# 3-D head
# --------------------------------------------------------------------------

def head3d(values, xyz, names, title=None, height=430, spin=True):
    u = xyz / (np.linalg.norm(xyz, axis=1, keepdims=True) + 1e-12)
    lat = np.linspace(-0.35, np.pi / 2, 34)
    lon = np.linspace(-np.pi, np.pi, 72)
    LA, LO = np.meshgrid(lat, lon)
    px, py, pz = (np.cos(LA) * np.cos(LO)).ravel(), (np.cos(LA) * np.sin(LO)).ravel(), np.sin(LA).ravel()
    pts = np.c_[px, py, pz]
    v = np.asarray(values, float)
    sc = np.max(np.abs(v)) or 1.0
    try:
        f = RBFInterpolator(u, v / sc, kernel="thin_plate_spline", smoothing=1e-2)
        inten = np.clip(f(pts), -1.2, 1.2) * sc
    except Exception:
        inten = np.zeros(len(pts))
    # triangulate the lat/lon grid
    nlo, nla = LA.shape
    I, J, K = [], [], []
    for i in range(nlo - 1):
        for j in range(nla - 1):
            a, b, c, d = i * nla + j, (i + 1) * nla + j, (i + 1) * nla + j + 1, i * nla + j + 1
            I += [a, a]; J += [b, c]; K += [c, d]
    stretch = np.array([0.92, 1.08, 1.0])
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=px * stretch[0], y=py * stretch[1], z=pz * stretch[2], i=I, j=J, k=K,
                            intensity=inten, colorscale=DIVERGING, cmin=-sc, cmax=sc, showscale=False,
                            lighting=dict(ambient=0.75, diffuse=0.5, specular=0.08, roughness=0.9),
                            hoverinfo="skip", flatshading=False))
    # nose
    fig.add_trace(go.Cone(x=[0], y=[1.12], z=[0.05], u=[0], v=[0.16], w=[0], sizemode="absolute", sizeref=0.14,
                          colorscale=[[0, RULE], [1, RULE]], showscale=False, hoverinfo="skip"))
    e = u * 1.015 * stretch
    fig.add_trace(go.Scatter3d(x=e[:, 0], y=e[:, 1], z=e[:, 2], mode="markers+text", text=names,
                               textfont=dict(family=SANS, size=9, color=INK), textposition="top center",
                               marker=dict(size=3.2, color=INK), hovertext=[f"{n}: {x:+.3g}" for n, x in zip(names, v)],
                               hoverinfo="text", showlegend=False))
    ax = dict(visible=False, showbackground=False)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=30 if title else 0, b=0), title=title,
                      scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="data",
                                 camera=dict(eye=dict(x=1.35, y=-1.35, z=1.1))))
    if spin:
        frames = []
        for k, a in enumerate(np.linspace(0, 2 * np.pi, 48)):
            frames.append(go.Frame(name=str(k), layout=dict(scene_camera=dict(eye=dict(x=1.9 * np.cos(a - np.pi / 4),
                                                                                    y=1.9 * np.sin(a - np.pi / 4), z=1.1)))))
        fig.frames = frames
        fig.update_layout(updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=0.02, xanchor="left",
                                            bgcolor=CARD, bordercolor=INK, font=dict(family=SANS, size=11),
                                            buttons=[dict(label="SPIN", method="animate",
                                                          args=[None, dict(frame=dict(duration=60, redraw=True),
                                                                           transition=dict(duration=0), fromcurrent=True,
                                                                           mode="immediate")]),
                                                     dict(label="STOP", method="animate",
                                                          args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])])
    return fig


def blank(text="", height=240):
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(family=SANS, size=13, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=height)
    return fig
