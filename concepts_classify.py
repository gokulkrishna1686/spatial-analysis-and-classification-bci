"""Concept demos for the Classification page: small interactive models that show
how a classifier decides, what chance level really is, over-/underfitting,
leakage and cross-validation. Computations live in toys.py."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import toys as T
import ui
import viz

C0, C1 = viz.TEAL, viz.VERM
DEMOS = ["Draw the line", "Chance is not 50%", "Too simple, too flexible", "Leakage in one picture", "What cross-validation does"]


def _notice(lines):
    ui.html('<div class="reading"><b>What to notice.</b> ' + " ".join(lines) + "</div>")


def _cm_fig(cm, labels=("left hand", "right hand"), height=240):
    tot = cm.sum(1, keepdims=True)
    txt = [[f"<b>{cm[i, j]}</b><br>{cm[i, j] / max(1, tot[i, 0]):.0%}" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=cm / np.maximum(1, tot), x=list(labels), y=list(labels), colorscale=viz.SEQ_INK, zmin=0, zmax=1,
                               showscale=False, text=txt, texttemplate="%{text}", textfont=dict(size=15)))
    fig.update_layout(height=height, xaxis_title="predicted", yaxis=dict(title="true", autorange="reversed"), margin=dict(l=90, r=10, t=10, b=45))
    return fig


# ---------------------------------------------------------------------------
@ui.guarded_fragment("Draw the line")
def render_lda():
    ui.html('<div class="big-q">A linear classifier turns every trial into one <em>score</em> and compares it with a <em>threshold</em>.</div>')
    c = st.columns(4)
    sep = c[0].slider("Distance between the classes", 0.0, 5.0, 1.25, 0.25, key="lda_sep")
    spread = c[1].slider("Spread within each class", 0.4, 2.5, 1.0, 0.1, key="lda_spread")
    n = c[2].slider("Trials per class", 20, 300, 120, 20, key="lda_n")
    thr = c[3].slider("Threshold on the score", -6.0, 6.0, 0.0, 0.25, key="lda_thr", help="0 = the LDA boundary. Move it to trade hits against false alarms.")
    X, y, w, s = T.lda_toy(sep, spread, n, seed=3)
    cm, tpr, fpr, acc = T.threshold_stats(s, y, thr)
    fr, tr, auc = T.roc(s, y)
    a, b, c3 = st.columns([1.15, 1.1, 0.9], gap="medium")
    with a:
        fig = go.Figure()
        for k, col, nm in ((0, C0, "left hand"), (1, C1, "right hand")):
            m = y == k
            fig.add_trace(go.Scatter(x=X[m, 0], y=X[m, 1], mode="markers", name=nm, marker=dict(size=6, color=col, opacity=0.6)))
        lim = float(np.percentile(np.abs(X), 99)) * 1.15
        # boundary: w·x - c = thr  ->  line perpendicular to w
        c0 = float(np.mean(X @ w) - np.mean(s))
        d = np.array([-w[1], w[0]]) / np.linalg.norm(w)
        p0 = w * (thr + c0) / (w @ w)
        fig.add_trace(go.Scatter(x=[p0[0] - d[0] * 3 * lim, p0[0] + d[0] * 3 * lim], y=[p0[1] - d[1] * 3 * lim, p0[1] + d[1] * 3 * lim],
                                 mode="lines", name="decision line", line=dict(color=viz.INK, width=3)))
        u = w / np.linalg.norm(w) * lim * 0.6
        fig.add_annotation(x=u[0], y=u[1], ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowwidth=2,
                           arrowcolor=viz.PALETTE[3], text="w", font=dict(color=viz.PALETTE[3], size=14))
        fig.update_xaxes(range=[-lim, lim], title_text="feature 1 (e.g. CSP 1)")
        fig.update_yaxes(range=[-lim, lim], scaleanchor="x", title_text="feature 2 (e.g. CSP 2)")
        fig.update_layout(height=400, margin=dict(l=50, r=10, t=30, b=45), legend=dict(y=1.08))
        ui.plot(fig, "lda_scatter")
        ui.figcap("C.1", "w = direction LDA projects onto; the line is where score = threshold")
    with b:
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.22, subplot_titles=("scores projected onto w", "ROC curve"))
        bins = np.linspace(s.min(), s.max(), 40)
        for k, col in ((0, C0), (1, C1)):
            fig.add_trace(go.Histogram(x=s[y == k], xbins=dict(start=bins[0], end=bins[-1], size=bins[1] - bins[0]), marker_color=col,
                                       opacity=0.6, showlegend=False), 1, 1)
        fig.add_vline(thr, line=dict(color=viz.INK, width=2.5), row=1, col=1)
        fig.add_trace(go.Scatter(x=fr, y=tr, mode="lines", line=dict(color=viz.PALETTE[3], width=2.5), showlegend=False), 2, 1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=viz.RULE, dash="dot"), showlegend=False), 2, 1)
        fig.add_trace(go.Scatter(x=[fpr], y=[tpr], mode="markers", marker=dict(size=13, color=viz.INK), showlegend=False), 2, 1)
        fig.update_layout(height=470, barmode="overlay", margin=dict(l=45, r=10, t=30, b=40))
        fig.update_xaxes(title_text="score", row=1, col=1)
        fig.update_xaxes(title_text="false alarm rate (left-hand trials called right hand)", range=[0, 1], row=2, col=1)
        fig.update_yaxes(title_text="hit rate", range=[0, 1.02], row=2, col=1)
        ui.plot(fig, "lda_hist")
    with c3:
        ui.kicker("At this threshold")
        ui.plot(_cm_fig(cm, ("left hand", "right hand")), "lda_cm")
        ui.html(f'<div class="kv"><div class="k">accuracy</div><div class="v">{acc:.1%}</div><div class="k">hit rate (right hand)</div><div class="v">{tpr:.1%}</div>'
                f'<div class="k">false alarms</div><div class="v">{fpr:.1%}</div><div class="k">area under ROC</div><div class="v">{auc:.2f}</div></div>')
    _notice(["Move the threshold: the black line slides across the cloud, the confusion matrix changes and the black dot runs along the ROC curve.",
             "The <b>AUC</b> does not depend on the threshold: it measures how well the scores separate the classes at all. Push the classes together and it drops towards 0.5."])


# ---------------------------------------------------------------------------
@ui.guarded_fragment("Chance is not 50%")
def render_chance():
    ui.html('<div class="big-q">With few test trials, a classifier that <em>guesses</em> can still score far above 1/K.</div>')
    c = st.columns(4)
    n = c[0].slider("Test trials", 10, 400, 40, 10, key="ch_n")
    k = c[1].select_slider("Classes K", [2, 3, 4, 5, 6], 2, key="ch_k")
    alpha = c[2].select_slider("Significance level α", [0.1, 0.05, 0.01, 0.001], 0.05, key="ch_alpha")
    mine = c[3].number_input("Your accuracy (%)", 0.0, 100.0, 62.5, 0.5, key="ch_mine")
    accs, pmf, thr, sim = T.binomial_null(n, k, alpha, n_sim=1000, seed=0)
    p = T.binomial_p(mine / 100, n, k)
    a, b = st.columns([2, 1], gap="large")
    with a:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=sim, histnorm="probability", xbins=dict(start=-0.5 / n, end=1 + 0.5 / n, size=1 / n),
                                   marker_color=viz.RULE, name="1,000 simulated guessers"))
        fig.add_trace(go.Scatter(x=accs, y=pmf, mode="lines+markers" if n <= 60 else "lines", name="theory (binomial)",
                                 line=dict(color=viz.INK, width=2), marker=dict(size=4)))
        for x, lab, col, dash in ((1 / k, f"chance 1/K = {1 / k:.0%}", viz.MUTED, "dot"), (thr, f"significance threshold {thr:.1%}", viz.OCHRE, "dash"),
                                  (mine / 100, f"your accuracy {mine:.1f}%", viz.VERM, "solid")):
            fig.add_vline(x, line=dict(color=col, width=2.5 if col == viz.VERM else 1.8, dash=dash))
            right = col != viz.MUTED
            fig.add_annotation(x=x, y=1.0, yref="paper", text=lab, showarrow=False, xanchor="left" if right else "right", xshift=4 if right else -4,
                               yshift=-4 - 20 * (col == viz.OCHRE), font=dict(color=col, size=12), bgcolor="rgba(255,255,255,0.85)")
        lo = max(0, 1 / k - 4 * np.sqrt((1 / k) * (1 - 1 / k) / n) - 0.05)
        fig.update_layout(height=380, xaxis=dict(title="accuracy of a random classifier", range=[lo, 1], tickformat=".0%"),
                          yaxis_title="probability", legend=dict(y=1.12), margin=dict(l=50, r=10, t=40, b=45), bargap=0.05)
        ui.plot(fig, "ch_hist")
    with b:
        sig = mine / 100 > thr
        ui.html(f'<span class="stamp {"ok" if sig else "bad"}">{"Significantly above chance" if sig else "Not significant"}</span>')
        ui.html(f'<div class="kv"><div class="k">chance level 1/K</div><div class="v">{1 / k:.1%}</div>'
                f'<div class="k">threshold (α = {alpha})</div><div class="v">{thr:.1%}</div>'
                f'<div class="k">P(guessing ≥ {mine:.1f}%)</div><div class="v">{p:.3g}</div>'
                f'<div class="k">guessers above threshold</div><div class="v">{(sim > thr).mean():.1%}</div></div>')
        ns = np.array([10, 20, 40, 80, 160, 320])
        th = [T.binomial_null(int(m), k, alpha, n_sim=10)[2] for m in ns]
        fig = go.Figure(go.Scatter(x=ns, y=th, mode="lines+markers", line=dict(color=viz.OCHRE, width=2.5)))
        fig.add_hline(1 / k, line=dict(color=viz.MUTED, dash="dot"))
        fig.update_layout(height=220, xaxis=dict(title="test trials", type="log", tickvals=list(ns), ticktext=[str(v) for v in ns]),
                          yaxis=dict(title="threshold", tickformat=".0%"), margin=dict(l=55, r=10, t=10, b=45))
        ui.plot(fig, "ch_thr")
    _notice([f"With {n} test trials, a pure guesser beats {thr:.0%} only {alpha * 100:g}% of the time, so you need more than that to claim anything.",
             "Add trials and the threshold slides down towards 1/K. That is why the app always reports the binomial threshold and a permutation test next to accuracy (Combrisson & Jerbi, 2015)."])


# ---------------------------------------------------------------------------
@ui.guarded_fragment("Too simple, too flexible")
def render_knn():
    ui.html('<div class="big-q">Flexible models can <em>memorise</em> training trials. Only held-out trials tell the truth.</div>')
    c = st.columns(3)
    k = c[0].slider("k nearest neighbours (small = flexible, large = simple)", 1, 99, 1, 2, key="kn_k")
    noise = c[1].slider("Noise in the data", 0.1, 0.6, 0.35, 0.05, key="kn_noise")
    n = c[2].slider("Trials", 60, 400, 200, 20, key="kn_n")
    X, y = T.moons_toy(n=n, noise=noise, seed=0)
    ks = np.arange(1, 100, 2)
    tr, te, (Xtr, Xte, ytr, yte) = T.knn_complexity_curve(X, y, ks, seed=0)
    from sklearn.neighbors import KNeighborsClassifier
    m = KNeighborsClassifier(n_neighbors=min(k, len(ytr))).fit(Xtr, ytr)
    a, b = st.columns([1.1, 1], gap="large")
    with a:
        g1, g2 = np.meshgrid(np.linspace(X[:, 0].min() - 0.4, X[:, 0].max() + 0.4, 150), np.linspace(X[:, 1].min() - 0.4, X[:, 1].max() + 0.4, 120))
        Z = m.predict_proba(np.c_[g1.ravel(), g2.ravel()])[:, 1].reshape(g1.shape)
        fig = go.Figure(go.Contour(x=g1[0], y=g2[:, 0], z=Z, showscale=False, zmin=0, zmax=1, contours=dict(start=0.5, end=0.5, size=0.5, coloring="heatmap"),
                                   colorscale=[[0, viz.rgba(C0, 0.35)], [0.5, "#F6F6F4"], [1, viz.rgba(C1, 0.35)]], line=dict(width=2, color=viz.INK), hoverinfo="skip"))
        for kk, col in ((0, C0), (1, C1)):
            fig.add_trace(go.Scatter(x=Xtr[ytr == kk, 0], y=Xtr[ytr == kk, 1], mode="markers", name=f"train · class {kk + 1}", marker=dict(size=7, color=col)))
            fig.add_trace(go.Scatter(x=Xte[yte == kk, 0], y=Xte[yte == kk, 1], mode="markers", name=f"held-out · class {kk + 1}",
                                     marker=dict(size=9, color=col, symbol="circle-open", line=dict(width=2))))
        fig.update_layout(height=420, margin=dict(l=40, r=10, t=30, b=40), legend=dict(y=1.12, font=dict(size=11)))
        ui.plot(fig, "kn_map")
    with b:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=tr, name="training trials", line=dict(color=viz.MUTED, width=2.5, dash="dash")))
        fig.add_trace(go.Scatter(x=ks, y=te, name="held-out trials", line=dict(color=viz.PALETTE[3], width=3)))
        fig.add_vline(k, line=dict(color=viz.INK, width=1.5))
        best = int(ks[np.argmax(te)])
        fig.add_annotation(x=best, y=float(te.max()), text=f"best held-out k = {best}", showarrow=True, ay=-30, font=dict(size=12))
        fig.update_layout(height=420, xaxis_title="k (← more flexible · simpler →)", yaxis=dict(title="accuracy", range=[0.4, 1.02], tickformat=".0%"),
                          legend=dict(y=1.1), margin=dict(l=55, r=10, t=30, b=45))
        ui.plot(fig, "kn_curve")
        i = int(np.argmin(np.abs(ks - k)))
        ui.html(f'<div class="kv"><div class="k">training accuracy at k = {k}</div><div class="v">{tr[i]:.0%}</div>'
                f'<div class="k">held-out accuracy</div><div class="v">{te[i]:.0%}</div>'
                f'<div class="k">gap</div><div class="v">{(tr[i] - te[i]) * 100:+.0f} points</div></div>')
    _notice(["At k = 1 the boundary wraps around every training dot: 100% on training trials, much less on new ones (overfitting).",
             "At very large k the boundary is too simple and both curves drop (underfitting). Pick complexity with held-out data, never with the test set (nested CV in Build)."])


# ---------------------------------------------------------------------------
@ui.guarded_fragment("Leakage in one picture")
def render_leak():
    ui.html('<div class="big-q">Choose features with <em>all</em> trials, then cross-validate, and pure noise looks like a brain signal.</div>')
    c = st.columns(4)
    n = c[0].slider("Trials", 20, 200, 60, 10, key="lk_n")
    p = c[1].select_slider("Noise features", [10, 50, 100, 250, 500, 1000, 2000], 500, key="lk_p")
    kb = c[2].slider("Features selected", 1, 50, 10, key="lk_k")
    reps = c[3].slider("Repeats", 3, 20, 8, key="lk_r")
    honest, leaky = T.leakage_toy(n, p, kb, reps, seed=0)
    a, b = st.columns([1, 1.3], gap="large")
    with a:
        fig = go.Figure(go.Bar(x=["selected inside each fold (honest)", "selected on all trials (leaky)"], y=[honest.mean(), leaky.mean()],
                               marker_color=[viz.TEAL, viz.VERM], text=[f"{honest.mean():.0%}", f"{leaky.mean():.0%}"], textposition="outside",
                               error_y=dict(type="data", array=[honest.std(), leaky.std()], color=viz.INK2)))
        fig.add_hline(0.5, line=dict(color=viz.INK, dash="dot"), annotation_text="chance", annotation_position="top left")
        fig.update_layout(height=380, yaxis=dict(range=[0, 1.05], title="cross-validated accuracy", tickformat=".0%"), margin=dict(l=55, r=10, t=30, b=40))
        ui.plot(fig, "lk_bars")
    with b:
        fig = go.Figure()
        for arr, nm, col in ((honest, "honest", viz.TEAL), (leaky, "leaky", viz.VERM)):
            fig.add_trace(go.Box(y=arr, name=nm, marker_color=col, boxpoints="all", jitter=0.5, pointpos=0))
        fig.add_hline(0.5, line=dict(color=viz.INK, dash="dot"))
        fig.update_layout(height=380, yaxis=dict(range=[0.2, 1.02], title="accuracy per repeat", tickformat=".0%"), showlegend=False,
                          margin=dict(l=55, r=10, t=30, b=40))
        ui.plot(fig, "lk_box")
    _notice([f"The data are random numbers with random labels: there is nothing to learn. The honest pipeline correctly stays near 50%; the leaky one claims about {leaky.mean():.0%}.",
             "Add more noise features and the lie gets bigger, because there are more lucky features to pick. In the app every fitted step sits inside the folds, so this cannot happen."])


# ---------------------------------------------------------------------------
@ui.guarded_fragment("What cross-validation does")
def render_cv():
    ui.html('<div class="big-q">Every trial is tested <em>exactly once</em>, by a model that never saw it.</div>')
    c = st.columns(3)
    k = c[0].slider("Number of folds k", 2, 10, 5, key="cvw_k")
    sep = c[1].slider("How separable the classes are", 0.0, 4.0, 1.5, 0.25, key="cvw_sep")
    step = c[2].slider("Show fold (1 … k)", 1, 10, 1, key="cvw_step", help="Step through the folds one by one.")
    step = min(step, k)
    X, y, _, _ = T.lda_toy(sep, 1.0, 20, seed=5)
    folds = T.cv_folds(X, y, k, seed=0)
    tr, te, acc = folds[step - 1]
    a, b = st.columns([1.4, 1], gap="large")
    with a:
        fig = go.Figure()
        for f in range(k):
            tr_f, te_f, _ = folds[f]
            colors = np.full(len(y), "#F0F2F5", dtype=object)
            colors[tr_f] = "#C3CBD6"
            colors[te_f] = viz.PALETTE[3] if f == step - 1 else "#93A6E0"
            fig.add_trace(go.Scatter(x=np.arange(len(y)), y=[f + 1] * len(y), mode="markers", showlegend=False, hoverinfo="skip",
                                     marker=dict(symbol="square", size=13, color=list(colors), opacity=1 if f < step else 0.25,
                                                 line=dict(width=1.5, color=[viz.INK if f == step - 1 and i in te_f else "rgba(0,0,0,0)" for i in range(len(y))]))))
        fig.update_layout(height=60 + 34 * k, xaxis=dict(title="trial", showgrid=False), yaxis=dict(title="fold", autorange="reversed", dtick=1, showgrid=False),
                          margin=dict(l=50, r=10, t=10, b=45))
        ui.plot(fig, "cvw_map")
        ui.html('<div class="legend-row"><span><i style="background:#C3CBD6"></i>training trials (model is fitted here)</span>'
                f'<span><i style="background:{viz.PALETTE[3]}"></i>test trials of the current fold</span><span><i style="background:#93A6E0"></i>already tested</span></div>')
    with b:
        accs = [f[2] for f in folds[:step]]
        fig = go.Figure(go.Bar(x=[f"fold {i + 1}" for i in range(k)], y=accs + [None] * (k - step),
                               marker_color=[viz.PALETTE[3] if i == step - 1 else "#93A6E0" for i in range(k)],
                               text=[f"{a_:.0%}" for a_ in accs] + [""] * (k - step), textposition="outside"))
        if step == k:
            fig.add_hline(float(np.mean(accs)), line=dict(color=viz.INK, width=2), annotation_text=f"mean {np.mean(accs):.0%} ± {np.std(accs):.0%}")
        fig.update_layout(height=320, yaxis=dict(range=[0, 1.1], title="test accuracy", tickformat=".0%"), margin=dict(l=55, r=10, t=30, b=40))
        ui.plot(fig, "cvw_bars")
    _notice([f"Fold {step} of {k}: the model is trained on the grey trials only and scored on the highlighted ones.",
             "Step through all folds: every trial is tested once, and the final number is the mean over folds. The spread between folds shows how stable the estimate is."])


RENDER = {DEMOS[0]: render_lda, DEMOS[1]: render_chance, DEMOS[2]: render_knn, DEMOS[3]: render_leak, DEMOS[4]: render_cv}


def render():
    demo = ui.seg("Concept", DEMOS, default=DEMOS[0], key="cl_concept") or DEMOS[0]
    RENDER[demo]()
