"""Classification (part 2 of 2 of our module)."""
import copy
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve

import classify as C
import concepts_classify
import downloads
import spatial as SP
import ui
import viz

o = ui.page_objects()
slot = ui.header_row("08", "Classification", "Can we read the <em>intention</em>?",
              "Build a pipeline, choose an honest evaluation protocol, place your bet, then find out whether the brain signal really carries the information, and how sure we can be.",
              flagship=True)

SCHEMES = {"kfold": "Stratified k-fold", "repeated": "Repeated k-fold", "loso_session": "Leave-one-session-out",
           "loso_subject": "Leave-one-subject-out", "cross_session": "Train session A → test session B"}
FIT_WHERE = {"csp": ("CSP", "fit: train fold"), "fbcsp": ("Filter-bank CSP", "fit: train fold"), "select": ("Select k-best", "fit: train fold"),
             "tangent_space": ("Tangent space", "fit: train fold"), "covariances": ("Covariances", "stateless"),
             "bandpower": ("Band power", "stateless"), "xdawn_cov": ("xDAWN covariances", "fit: train fold"),
             "features": ("Feature union", "fit: train fold"), "scaler": ("Standard scaler", "fit: train fold"), "clf": ("Classifier", "fit: train fold")}


CLF_DEFAULTS = {"lda": {"shrinkage": "auto"}, "lr": {"C": 1.0}, "svm": {"C": 1.0, "kernel": "rbf"}, "knn": {"k": 7},
                "rf": {"n_estimators": 200, "max_depth": 0}, "mlp": {"hidden": 32, "alpha": 1e-3}}


def norm_cfg(c, sp):
    """Fill in every default a widget would add, so 'unapplied changes' only appears after a real change."""
    c = copy.deepcopy(c)
    c.setdefault("band", sp["band"])
    c.setdefault("window", sp["window"])
    c.setdefault("n_components", sp["n_components"])
    c["clf_params"] = {**CLF_DEFAULTS.get(c.get("clf"), {}), **(c.get("clf_params") or {})}
    cv = dict(c.get("cv") or {})
    cv.setdefault("scheme", "kfold")
    if cv["scheme"] in ("kfold", "repeated"):
        cv.setdefault("k", 5)
    c["cv"] = cv
    for k, v in (("nested", False), ("n_perm", 100), ("trial_len", 4.0), ("pre", None), ("classes", None)):
        c.setdefault(k, v)
    if c.get("feature") == "fbcsp":
        c.setdefault("k_best", 10)
    c["band"], c["window"] = [float(b) for b in c["band"]], [float(w) for w in c["window"]]
    return c


def go_stage(name):
    """Switch the mission rail programmatically (a fresh widget key avoids
    mutating a widget's state after creation)."""
    st.session_state["mission_target"] = name
    st.session_state["mission_n"] = st.session_state.get("mission_n", 0) + 1
    st.rerun()


def pipe_strip(cfg, pipe, n_trials, n_ch):
    nested = cfg.get("nested") and cfg.get("clf") in C.PARAM_GRIDS
    blocks = [("in", "Epochs", f"{n_trials} × {n_ch} ch", "")]
    if cfg.get("pre"):
        blocks.append(("", cfg["pre"], "re-reference", '<div class="fit no">stateless</div>'))
    blocks.append(("", "Band-pass", f"{cfg['band'][0]:g}–{cfg['band'][1]:g} Hz", '<div class="fit no">stateless · per trial</div>'))
    blocks.append(("", "Crop", f"{cfg['window'][0]:g}–{cfg['window'][1]:g} s", '<div class="fit no">stateless</div>'))
    for name, step in pipe.steps:
        lab, where = FIT_WHERE.get(name, (name, "fit: train fold"))
        if name == "clf":
            lab = C.CLASSIFIERS.get(cfg.get("clf"), lab).split(" (")[0]
            if nested:
                where = "inner CV tunes · fit: train fold"
        if name == "csp":
            lab += f" · {step.n_components}"
        cls_ = "tr" if "fit" in where else "no"
        blocks.append(("", lab, step.__class__.__name__, f'<div class="fit {cls_}">{where}</div>'))
    blocks.append(("out", "Prediction", "test fold only", ""))
    out = '<div class="pipe">'
    for i, (kind, n, p, f) in enumerate(blocks):
        if i:
            out += '<div class="arr"></div>'
        out += f'<div class="blk {kind}"><div class="bn">{n}</div><div class="bp">{p}</div>{f}</div>'
    return out + "</div>"


def fold_map_svg(splits, meta, nested=False, seed=42, max_rows=20):
    n = len(meta)
    order = np.lexsort((meta["onset_s"].to_numpy(), meta["session"].to_numpy(), meta["subject"].to_numpy()))
    pos = np.empty(n, int)
    pos[order] = np.arange(n)
    W, rh, gap, left = 900, 14, 6, 64
    rows = splits[:max_rows]
    sub = 7 if nested else 0
    H = len(rows) * (rh + sub + gap) + 30
    cw = (W - left - 6) / n
    svg = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="IBM Plex Sans, sans-serif" font-size="11">']
    ss = meta[["subject", "session"]].to_numpy()[order]
    for i in range(1, n):
        if (ss[i] != ss[i - 1]).any():
            x = left + i * cw
            svg.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{H - 18}" stroke="#6B7383" stroke-width="1" stroke-dasharray="3 3"/>')
    starts = [0] + [i for i in range(1, n) if (ss[i] != ss[i - 1]).any()] + [n]
    for a, b in zip(starts[:-1], starts[1:]):
        svg.append(f'<text x="{left + (a + b) / 2 * cw:.1f}" y="{H - 4}" text-anchor="middle" fill="#6B6F76">S{ss[a][0]}·{ss[a][1]}</text>')
    rng = np.random.RandomState(seed)
    for r, (tr, te) in enumerate(rows):
        y0 = r * (rh + sub + gap)
        svg.append(f'<text x="0" y="{y0 + rh - 3}" fill="#485062">fold {r + 1}</text>')
        col = np.full(n, "#F0F2F5", dtype=object)
        col[pos[tr]] = "#C3CBD6"
        col[pos[te]] = "#2F5BD3"
        # merge runs of equal colour into rects
        i = 0
        while i < n:
            j = i
            while j + 1 < n and col[j + 1] == col[i]:
                j += 1
            svg.append(f'<rect x="{left + i * cw:.2f}" y="{y0}" width="{(j - i + 1) * cw + 0.3:.2f}" height="{rh}" fill="{col[i]}"/>')
            i = j + 1
        if nested:
            inner = np.zeros(n, bool)
            trp = rng.permutation(tr)
            inner[pos[trp[: len(trp) // 3]]] = True
            for k in range(n):
                if inner[k]:
                    svg.append(f'<rect x="{left + k * cw:.2f}" y="{y0 + rh + 1}" width="{cw + 0.3:.2f}" height="{sub - 2}" fill="#E8A33D"/>')
    if len(splits) > max_rows:
        svg.append(f'<text x="{W - 4}" y="{H - 4}" text-anchor="end" fill="#6B6F76">+{len(splits) - max_rows} more folds</text>')
    svg.append("</svg>")
    leg = ('<div class="legend-row"><span><i style="background:#C3CBD6"></i>train (everything is fitted here)</span>'
           '<span><i style="background:#2F5BD3"></i>test (only predicted)</span>'
           + ('<span><i style="background:#E8A33D"></i>inner validation (tuning, 1 of 3 inner folds)</span>' if nested else "")
           + '<span><i style="background:#F0F2F5;border:1px solid #D5DAE1"></i>unused</span><span>columns = trials in recording order · dashed = session/subject boundary</span></div>')
    return '<div class="foldmap">' + "".join(svg) + leg + "</div>"


def scoreboard(res):
    s = res["summary"]
    m = res["metric"]
    cells = [("accuracy" if m == "accuracy" else "balanced acc.", s[m], True, "{:.1%}"),
             ("balanced acc." if m == "accuracy" else "accuracy", s["balanced_accuracy" if m == "accuracy" else "accuracy"], False, "{:.1%}"),
             ("Cohen's κ", s["kappa"], False, "{:.2f}"), ("macro F1", s["f1"], False, "{:.2f}"),
             ("ROC-AUC", s["roc_auc"], False, "{:.2f}"), ("ITR bits/min", s["itr_bits_min"], False, "{:.1f}")]
    out = '<div class="score">'
    for k, (mu, sd), hero, fmt in cells:
        out += f'<div class="{"hero" if hero else ""}"><div class="k">{k}</div><div class="v">{fmt.format(mu)}</div><div class="s">± {fmt.format(sd)}</div></div>'
    return out + "</div>"


def verdict(res, bet):
    m = res["metric"]
    acc = res["summary"][m][0]
    p = res["p_value"]
    thr, ch = res["binom_threshold"], res["chance"]
    if np.isnan(p):
        ok = acc > thr
        stamp = ("Above chance", "ok") if ok else ("Not above chance", "bad")
    elif p < 0.05 and acc > thr:
        stamp = ("Significant", "ok")
    elif p < 0.05:
        stamp = ("Borderline", "meh")
    else:
        stamp = ("Not significant", "bad")
    ptxt = f"permutation <b>p = {p:.3f}</b> ({len(res['null'])} shuffles)" if not np.isnan(p) else "no permutation test run"
    txt = (f"Theoretical chance <b>{ch:.0%}</b> · binomial threshold <b>{thr:.1%}</b> for {res['n_test_total']} test trials (α = 0.05) · {ptxt}.")
    if stamp[1] != "ok":
        txt += " <b>Warning:</b> this result is not convincingly above chance. Do not claim the classifier works."
    b = ""
    if bet is not None:
        d = acc * 100 - bet
        b = f'<div class="bet">your bet {bet:.0f}% → reality {acc * 100:.1f}% ({d:+.1f} pts). {"Spot on." if abs(d) < 3 else ("Optimist!" if d < 0 else "You underestimated the brain.")}</div>'
    return f'<div class="verdict"><div class="stamp {stamp[1]}">{stamp[0]}</div><div class="txt">{txt}{b}</div></div>'


with ui.guard("Classification"):
    ep = o["epochs"]
    P = ui.params()
    seed = P["seed"]
    sf = ep.info["sfreq"]
    classes_all = ui.classes_of(ep)
    if "cl_cfg" not in st.session_state or st.session_state.get("cl_src") != ui.phash(P["classify"]):
        st.session_state["cl_cfg"] = copy.deepcopy(P["classify"])
        st.session_state["cl_src"] = ui.phash(P["classify"])
    cfg = st.session_state["cl_cfg"]
    cfg.setdefault("band", P["spatial"]["band"])
    cfg.setdefault("window", P["spatial"]["window"])
    cfg.setdefault("n_components", P["spatial"]["n_components"])
    FP = dict(P["features"], kinds=cfg.get("feature_kinds") or P["features"]["kinds"])   # feature families used by "Feature set"
    R = ui.S()["results"]
    res_store = R.get("clf")
    up_key = ui.phash([P[k] for k in ("data", "filter", "ica", "epochs", "seed")])
    cfg_key = ui.phash([up_key, cfg])
    fresh = res_store is not None and res_store["key"] == cfg_key

    tabs = ["Concepts", "Build", "Understand", "Compete", "Leakage guard", "Lab log"]
    done = {"Concepts": False, "Build": True, "Understand": res_store is not None, "Compete": R.get("bench") is not None,
            "Leakage guard": R.get("leak") is not None, "Lab log": bool(ui.S()["runs"])}
    nums = {"Concepts": "", "Build": "1.", "Understand": "2.", "Compete": "3.", "Leakage guard": "", "Lab log": ""}
    mt = st.session_state.get("mission_target", "Concepts")
    with st.container(key="mission_rail"):
        stage = ui.seg("Stage", tabs, default=mt, key=f"mission_{st.session_state.get('mission_n', 0)}",
                       format_func=lambda t: f"{nums[t]} {t}{'  ✓' if done[t] and t != 'Build' else ''}".strip()) or mt

    if stage == "Concepts":
        concepts_classify.render()

    # ============================================================ BUILD
    elif stage == "Build":
        ui.kicker("1 · Pick a recipe (or tune the cards below)")
        names_r = list(C.RECIPES) + ["Custom"]
        cur_rec = next((k for k, v in C.RECIPES.items() if v["feature"] == cfg.get("feature") and v["clf"] == cfg.get("clf")), "Custom")
        cols = st.columns(len(names_r))
        for col, rn in zip(cols, names_r):
            with col:
                with st.container(key=f"recipe_{'on_' if rn == cur_rec else ''}{names_r.index(rn)}"):
                    note = C.RECIPES[rn]["note"] if rn in C.RECIPES else "Any feature source × any classifier."
                    if st.button(rn, key=f"rbtn_{rn}", use_container_width=True, help=note):
                        if rn in C.RECIPES:
                            cfg["feature"], cfg["clf"] = C.RECIPES[rn]["feature"], C.RECIPES[rn]["clf"]
                            cfg["clf_params"] = {}
                        st.rerun()
        runslot = st.container()          # filled at the end, so it always reflects the cards below
        ui.kicker("2 · Fine-tune")
        c_pipe, c_data, c_prot = st.columns(3, gap="medium")
        with c_pipe:
            with st.container(key="deck_cl_pipe"):
                ui.deck_header("Pipeline", cur_rec)
                feats = list(C.FEATURE_SOURCES)
                cfg["feature"] = st.selectbox("Feature source", feats, index=feats.index(cfg.get("feature", "csp")) if cfg.get("feature") in feats else 0,
                                              format_func=C.FEATURE_SOURCES.get, help="How trials become numbers. Everything data-driven here is refit in every training fold.")
                clfs = list(C.CLASSIFIERS)
                if cfg["feature"] == "cov":
                    clfs = ["mdm", "lr", "svm", "lda"]
                cfg["clf"] = st.selectbox("Classifier", clfs, index=clfs.index(cfg.get("clf")) if cfg.get("clf") in clfs else 0,
                                          format_func=C.CLASSIFIERS.get, help="The decision rule on top of the features.")
                cp = cfg.setdefault("clf_params", {})
                c = cfg["clf"]
                if c == "lda":
                    cp["shrinkage"] = st.select_slider("LDA shrinkage", ["none", 0.01, 0.1, 0.3, 0.5, 0.9, "auto"], cp.get("shrinkage", "auto"), help="'auto' = Ledoit-Wolf optimal shrinkage of the class covariance.")
                elif c in ("lr", "svm"):
                    cp["C"] = st.select_slider("C (inverse regularisation)", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], float(cp.get("C", 1.0)), help="Small C = strong regularisation, simpler model.")
                    if c == "svm":
                        cp["kernel"] = st.selectbox("Kernel", ["rbf", "linear"], index=["rbf", "linear"].index(cp.get("kernel", "rbf")))
                elif c == "knn":
                    cp["k"] = st.slider("k neighbours", 1, 31, int(cp.get("k", 7)), 2)
                elif c == "rf":
                    cp["n_estimators"] = st.slider("Trees", 20, 500, int(cp.get("n_estimators", 200)), 20)
                    cp["max_depth"] = st.slider("Max depth (0 = unlimited)", 0, 20, int(cp.get("max_depth") or 0))
                elif c == "mlp":
                    cp["hidden"] = st.slider("Hidden units", 4, 128, int(cp.get("hidden", 32)), 4)
                    cp["alpha"] = st.select_slider("L2 penalty α", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], float(cp.get("alpha", 1e-3)))
                if cfg["feature"] in ("csp", "xdawn"):
                    cfg["n_components"] = st.slider("Spatial components", 2, min(12, len(ep.ch_names)), int(cfg.get("n_components", 6)), 1, help="CSP / xDAWN filters kept.")
                if cfg["feature"] == "features":
                    from features import FEATURE_INFO
                    fk = st.multiselect("Feature families", list(FEATURE_INFO), default=cfg.get("feature_kinds") or P["features"]["kinds"],
                                        help="  \n".join(f"{k}: {v}" for k, v in FEATURE_INFO.items())) or ["logvar"]
                    cfg["feature_kinds"] = fk
                if cfg["feature"] == "fbcsp":
                    cfg["k_best"] = st.slider("Features kept (MI selection)", 2, 28, int(cfg.get("k_best", 10)), help="Selection happens inside each training fold.")
        with c_data:
            with st.container(key="deck_cl_data"):
                ui.deck_header("Data", "stateless")
                cfg["band"] = list(st.slider("Band (Hz)", 1.0, float(min(60, sf / 2 - 1)), tuple(float(b) for b in cfg["band"]), 0.5, help="Stateless band-pass applied per trial."))
                t0, t1 = float(ep.times[0]), float(ep.times[-1])
                cfg["window"] = list(st.slider("Window (s)", t0, t1, (max(t0, float(cfg["window"][0])), min(t1, float(cfg["window"][1]))), 0.05, help="Part of the epoch used for training and testing."))
                if len(classes_all) > 2:
                    cfg["classes"] = st.multiselect("Classes", classes_all, default=cfg.get("classes") or classes_all, format_func=viz.pretty) or classes_all
                cfg["trial_len"] = st.number_input("Trial length for ITR (s)", 0.5, 20.0, float(cfg.get("trial_len", 4.0)), 0.5, help="Time per selection, including pauses. Used for bits/min.")
        with c_prot:
            with st.container(key="deck_cl_prot"):
                ui.deck_header("Protocol", "leak-free")
                sch = list(SCHEMES)
                cv = cfg.setdefault("cv", {"scheme": "kfold", "k": 5})
                cv["scheme"] = st.selectbox("Evaluation", sch, index=sch.index(cv.get("scheme", "kfold")), format_func=SCHEMES.get,
                                            help="Session/subject protocols test generalisation to new recordings, which is much harder than k-fold.")
                if cv["scheme"] in ("kfold", "repeated"):
                    cv["k"] = st.slider("Folds k", 2, 10, int(cv.get("k", 5)))
                if cv["scheme"] == "repeated":
                    cv["repeats"] = st.slider("Repeats", 2, 10, int(cv.get("repeats", 3)))
                if cv["scheme"] == "cross_session":
                    sess = sorted(ep.metadata.session.unique().tolist())
                    if len(sess) >= 2:
                        a_, b_ = st.columns(2)
                        cv["train"] = int(a_.selectbox("Train on session", sess, 0))
                        cv["test"] = int(b_.selectbox("Test on session", [s for s in sess if s != cv["train"]], 0))
                cfg["nested"] = st.toggle("Nested CV for hyper-parameters", bool(cfg.get("nested")), help="An inner 3-fold CV inside each training fold picks the hyper-parameters. The test fold is never used for tuning.",
                                          disabled=cfg["clf"] not in C.PARAM_GRIDS)
                cfg["n_perm"] = st.select_slider("Permutations", [0, 20, 50, 100, 200, 500], int(cfg.get("n_perm", 100)), help="Shuffle labels and rerun the whole CV this many times to get a p-value.")

        X, y, meta, times = C.get_Xy(ep, cfg["band"], cfg["window"], cfg.get("classes"), cfg.get("pre"))
        pipe = C.make_pipeline(cfg, sf, times, seed, FP)
        ui.kicker("3 · What will happen — every data-fitted block is refit inside each training fold")
        ui.html(pipe_strip(cfg, pipe, len(y), X.shape[1]))
        try:
            splits = C.make_splits(cfg["cv"], y, meta, seed)
            ui.kicker(f"Fold map · {SCHEMES[cfg['cv']['scheme']]} · {len(splits)} split(s)")
            ui.html(fold_map_svg(splits, meta, cfg.get("nested") and cfg["clf"] in C.PARAM_GRIDS, seed))
        except Exception as e:
            splits = None
            split_err = ui.friendly_message(e)
        _, cnt = np.unique(y, return_counts=True)

        with runslot:
            n_fit = (len(splits) if splits else 0) * (1 + cfg["n_perm"])
            title = cur_rec if cur_rec != "Custom" else f'{C.FEATURE_SOURCES[cfg["feature"]]} + {cfg["clf"].upper()}'
            ui.html(f'<div class="runbar"><span class="what">{title}</span>'
                    f'<span class="meta">{SCHEMES[cfg["cv"]["scheme"]]}{" · nested" if cfg.get("nested") else ""} · {cfg["n_perm"]} permutations · '
                    f'{len(y)} trials ({" / ".join(map(str, cnt))}) · chance {1 / len(cnt):.0%} · {n_fit} model fits</span></div>')
            with st.container(key="runbox"):
                if splits is None:
                    ui.note(f"<b>This protocol is not possible here:</b> {split_err} Pick another evaluation in the Protocol card.", "v")
                b1, b2 = st.columns([2.3, 1.4], vertical_alignment="center")
                with b1:
                    bet = st.slider("Place your bet: accuracy on unseen trials (%)", 0, 100, int(st.session_state.get("bet", 70)), key="bet",
                                    help="Guess before you run. You will see your guess next to the real result.")
                with b2:
                    go_ = st.button("Run leak-free evaluation", type="primary", use_container_width=True, disabled=splits is None, key="cl_run")
                if go_:
                    def job(cb):
                        r = C.run_classification(ep, cfg, seed, FP, progress=cb)
                        r["final"] = C.fit_final(r["pipe"], r["X"], r["y"])
                        return r
                    res = ui.run_with_progress("Evaluating", job)
                    R["clf"] = {"key": cfg_key, "res": res, "cfg": copy.deepcopy(cfg), "bet": bet, "time": dt.datetime.now().strftime("%H:%M:%S")}
                    R.pop("tr", None)
                    ui.S()["runs"].append({"time": R["clf"]["time"], "recipe": cur_rec, "feature": cfg["feature"], "clf": cfg["clf"],
                                           "protocol": SCHEMES[cfg["cv"]["scheme"]], "nested": cfg.get("nested", False),
                                           "metric": res["metric"], "score": res["summary"][res["metric"]][0], "std": res["summary"][res["metric"]][1],
                                           "kappa": res["summary"]["kappa"][0], "p": res["p_value"], "bet": bet, "cfg": json.dumps(cfg)})
                    go_stage("Understand")

    # ============================================================ UNDERSTAND
    elif stage == "Understand":
        if res_store is None:
            ui.note("Nothing evaluated yet. Go to <b>Build</b> and press <b>Run</b>.", "t")
            st.stop()
        res, rcfg = res_store["res"], res_store["cfg"]
        if not fresh:
            ui.note("Settings changed since this run. The numbers below belong to the previous configuration. Re-run in <b>Build</b>.", "")
        ui.html(scoreboard(res))
        ui.html(verdict(res, res_store.get("bet")))
        X, y, meta, times = res["X"], res["y"], res["meta"], res["times"]
        cls = np.array(res["classes"])
        cc = viz.class_colors(list(cls))
        oof = res["oof"]
        first = oof[oof.repeat == 0].sort_values("trial")
        st.write("")
        @ui.guarded_fragment("Results view")
        def results_views():
            """Clicks inside the result views rerun only this part of the page."""
            with st.container(key="mission_u"):
                uview = ui.seg("View", ["Chance", "Errors", "Over time", "Replay", "Playground", "Inside the model"], default="Chance", key="u_view") or "Chance"

            if uview == "Chance":
                a, b = st.columns([1.35, 1])
                m = res["metric"]
                with a:
                    ui.kicker("Null distribution — what shuffled labels achieve with the same pipeline and protocol")
                    fn = go.Figure()
                    if len(res["null"]):
                        fn.add_trace(go.Histogram(x=res["null"], nbinsx=30, marker_color=viz.RULE, marker_line=dict(color=viz.INK2, width=0.6), name="shuffled labels"))
                    obs = res["summary"][m][0]
                    for xv, lab, col, dash, yy in [(res["chance"], f"chance {res['chance']:.0%}", viz.MUTED, "dot", 1.13),
                                                   (res["binom_threshold"], f"binomial threshold {res['binom_threshold']:.1%}", viz.OCHRE, "dash", 1.04),
                                                   (obs, f"your pipeline {obs:.1%}", viz.VERM, "solid", 1.04)]:
                        fn.add_vline(xv, line=dict(color=col, width=2.5 if "pipeline" in lab else 1.6, dash=dash))
                        fn.add_annotation(x=xv, y=yy, yref="paper", text=lab, showarrow=False, xanchor="left" if "pipeline" in lab else "right",
                                          xshift=4 if "pipeline" in lab else -4, font=dict(family=viz.MONO, size=10, color=col))
                    fn.update_layout(height=340, xaxis_title=m.replace("_", " "), yaxis_title="count", xaxis_range=[0, 1.02], showlegend=False, margin=dict(t=52))
                    ui.plot(fn, "u_null", name="null_distribution")
                    ui.figcap("8.1", "the permutation test answers: how often does pure luck do at least this well?")
                with b:
                    ui.kicker("Per fold — click a dot")
                    fd = res["folds"]
                    ff = go.Figure()
                    ff.add_trace(go.Box(y=fd[m], boxpoints=False, marker_color=viz.INK2, name="", line=dict(width=1)))
                    ff.add_trace(go.Scatter(x=[""] * len(fd), y=fd[m], mode="markers", customdata=fd["fold"], marker=dict(size=11, color=viz.VERM, line=dict(width=1, color=viz.PAPER)),
                                            hovertemplate="fold %{customdata}: %{y:.3f}<extra></extra>"))
                    ff.add_hline(res["chance"], line=dict(color=viz.MUTED, dash="dot"))
                    ff.update_layout(height=330, yaxis=dict(range=[0, 1.02], title=m.replace("_", " ")), showlegend=False)
                    ev = ui.plot(ff, "u_folds", select=True)
                    v = ui.on_click(ev, "u_folds")
                    if v is not None:
                        st.session_state["u_fold"] = int(v)
                fsel = st.session_state.get("u_fold")
                a, b = st.columns([1, 1])
                with a:
                    ui.kicker("ROC — every fold (thin) and the selected fold (bold)" if len(cls) == 2 else "ROC (binary problems only)")
                    fr = go.Figure()
                    fr.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=viz.MUTED, dash="dot"), showlegend=False))
                    for i, (fpr, tpr) in enumerate(res["rocs"]):
                        bold = fsel == i + 1
                        fr.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=viz.VERM if bold else viz.rgba(viz.INK, 0.3), width=3 if bold else 1), showlegend=False))
                    fr.update_layout(height=300, xaxis_title="false positive rate", yaxis_title="true positive rate")
                    ui.plot(fr, "u_roc", name="roc")
                with b:
                    if fsel:
                        sub = oof[oof.fold == fsel]
                        ui.kicker(f"Fold {fsel} — {len(sub)} test trials, accuracy {np.mean(sub.y_true == sub.y_pred):.1%}")
                        cmf = confusion_matrix(sub.y_true, sub.y_pred, labels=cls)
                    else:
                        ui.kicker("All folds — confusion counts (click a fold dot to zoom in)")
                        cmf = res["confusion"]
                    fh = go.Figure(go.Heatmap(z=cmf, x=[viz.pretty(c_) for c_ in cls], y=[viz.pretty(c_) for c_ in cls], colorscale=viz.SEQ_INK, showscale=False,
                                              text=cmf, texttemplate="%{text}", textfont=dict(family=viz.MONO, size=16)))
                    fh.update_layout(height=300, xaxis_title="predicted", yaxis=dict(title="true", autorange="reversed"), margin=dict(l=100, r=10, t=10, b=50))
                    ui.plot(fh, "u_cmf")

            elif uview == "Errors":
                a, b = st.columns([1, 1.25])
                with a:
                    ui.kicker("Confusion matrix — click a cell to list its trials")
                    cm = res["confusion"]
                    K = len(cls)
                    gx, gy = np.meshgrid(range(K), range(K))
                    tot = cm.sum(1, keepdims=True)
                    fc = go.Figure()
                    for i in range(K):
                        for j in range(K):
                            frac = cm[i, j] / max(1, tot[i, 0])
                            col = viz.rgba(viz.TEAL if i == j else viz.VERM, 0.12 + 0.85 * frac)
                            fc.add_shape(type="rect", x0=j - 0.48, x1=j + 0.48, y0=i - 0.48, y1=i + 0.48, fillcolor=col, line=dict(color=viz.INK, width=1))
                    fc.add_trace(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode="markers+text", customdata=[f"{i},{j}" for i, j in zip(gy.ravel(), gx.ravel())],
                                            text=[f"<b>{cm[i, j]}</b><br>{cm[i, j] / max(1, tot[i, 0]):.0%}" for i, j in zip(gy.ravel(), gx.ravel())],
                                            textfont=dict(family=viz.MONO, size=15, color=viz.INK), marker=dict(size=70, symbol="square", opacity=0.01),
                                            hovertemplate="true %{y} → predicted %{x}<extra></extra>", showlegend=False))
                    fc.update_xaxes(tickvals=list(range(K)), ticktext=[viz.pretty(c_) for c_ in cls], title="predicted", showgrid=False, range=[-0.55, K - 0.45])
                    fc.update_yaxes(tickvals=list(range(K)), ticktext=[viz.pretty(c_) for c_ in cls], title="true", autorange="reversed", showgrid=False, scaleanchor="x")
                    fc.update_layout(height=360, dragmode=False, margin=dict(l=96, r=10, t=10, b=50))
                    ev = ui.plot(fc, "u_cm", select=True, name="confusion_matrix")
                    v = ui.on_click(ev, "u_cm")
                    if v:
                        st.session_state["u_cell"] = tuple(int(q) for q in v.split(","))
                    cell = st.session_state.get("u_cell", (0, 1) if K > 1 else (0, 0))
                    ti, pj = cell if max(cell) < K else (0, min(1, K - 1))
                with b:
                    sub = first[(first.y_true == cls[ti]) & (first.y_pred == cls[pj])].copy()
                    kind = "correct" if ti == pj else "misclassified"
                    ui.kicker(f"{len(sub)} {kind} trials · true {viz.pretty(cls[ti])} → predicted {viz.pretty(cls[pj])} — click a row")
                    sub["confidence"] = sub[f"p_{cls[pj]}"]
                    sub["session"] = meta.loc[sub.trial, "session"].to_numpy()
                    sub["onset_s"] = meta.loc[sub.trial, "onset_s"].round(1).to_numpy()
                    show = sub[["trial", "fold", "session", "onset_s", "confidence"]].sort_values("confidence", ascending=False)
                    evt = st.dataframe(show.style.format({"confidence": "{:.2f}", "onset_s": "{:.1f}"}).background_gradient(subset=["confidence"], cmap="Oranges"),
                                       hide_index=True, use_container_width=True, height=300, on_select="rerun", selection_mode="single-row", key="u_tab")
                    try:
                        rsel = evt.selection.rows
                    except Exception:
                        rsel = []
                    if rsel:
                        st.session_state["u_trial"] = int(show.iloc[rsel[0]].trial)
                ui.kicker("Trial barcode — every out-of-fold prediction in recording order (height = confidence in the true class) — click a bar")
                pt = np.array([first.loc[first.trial == t, f"p_{y[t]}"].values[0] for t in first.trial])
                order = np.lexsort((meta.onset_s.to_numpy()[first.trial], meta.session.to_numpy()[first.trial]))
                tr_ord = first.trial.to_numpy()[order]
                okk = (first.y_true.to_numpy() == first.y_pred.to_numpy())[order]
                fbar = go.Figure(go.Bar(x=list(range(len(tr_ord))), y=pt[order], customdata=tr_ord,
                                        marker=dict(color=[cc[y[t]] for t in tr_ord], line=dict(color=[viz.INK if not k else "rgba(0,0,0,0)" for k in okk], width=1.2),
                                                    opacity=[1.0 if k else 0.45 for k in okk]),
                                        hovertemplate="trial %{customdata}<br>P(true) = %{y:.2f}<extra></extra>"))
                fbar.add_hline(1 / len(cls), line=dict(color=viz.INK, width=1, dash="dot"))
                sess = meta.session.to_numpy()[tr_ord]
                for i in range(1, len(sess)):
                    if sess[i] != sess[i - 1]:
                        fbar.add_vline(i - 0.5, line=dict(color=viz.INK, width=1.5))
                fbar.update_layout(height=200, bargap=0.1, yaxis=dict(range=[0, 1], title="P(true)"), xaxis=dict(showticklabels=False, title="trial (recording order)"), margin=dict(t=6))
                ev = ui.plot(fbar, "u_bar", select=True, name="trial_barcode")
                v = ui.on_click(ev, "u_bar")
                if v is not None:
                    st.session_state["u_trial"] = int(v)
                ui.figcap("8.2", "faded bars with an outline = misclassified · dotted line = chance · vertical rule = session boundary. Drifts show up as runs of faded bars")
                tsel = st.session_state.get("u_trial", int(sub.trial.iloc[0]) if len(sub) else int(first.trial.iloc[0]))
                tsel = tsel if tsel < len(y) else 0
                row = first[first.trial == tsel].iloc[0]
                fold_i = int(row.fold) - 1
                tr_idx, te_idx = res["splits"][fold_i]
                a, b = st.columns([1.3, 1])
                with a:
                    names = [ep.ch_names[i] for i in range(X.shape[1])]
                    chs = [n for n in ("C3", "C4") if n in names] or names[:2]
                    fs_ = go.Figure()
                    for k_, ch in enumerate(chs):
                        ci = names.index(ch)
                        fs_.add_trace(go.Scatter(x=times, y=X[tsel, ci] * 1e6 - k_ * 6 * X[:, ci].std() * 1e6, name=ch, line=dict(width=1.2, color=[viz.TEAL, viz.VERM][k_ % 2])))
                    fs_.update_layout(height=280, title=f"trial {tsel} · true {viz.pretty(row.y_true)} · predicted {viz.pretty(row.y_pred)} · fold {row.fold}",
                                      xaxis_title="time (s)", yaxis=dict(showticklabels=False))
                    ui.plot(fs_, "u_tsig")
                with b:
                    feat = clone(res["pipe"]) if not hasattr(res["pipe"], "best_estimator_") else clone(res["pipe"].estimator)
                    try:
                        from sklearn.pipeline import Pipeline as SkP
                        head = SkP(feat.steps[:-1]) if len(feat.steps) > 1 else None
                        if head is not None:
                            head.fit(X[tr_idx], y[tr_idx])
                            Ftr, Fte = head.transform(X[tr_idx]), head.transform(X[[tsel]])
                            pca = PCA(2, random_state=seed).fit(Ftr)
                            A2, B2 = pca.transform(Ftr), pca.transform(Fte)
                            fe = go.Figure()
                            for c_ in cls:
                                mk = y[tr_idx] == c_
                                fe.add_trace(go.Scatter(x=A2[mk, 0], y=A2[mk, 1], mode="markers", name=f"train · {viz.pretty(c_)}", marker=dict(size=6, color=cc[c_], opacity=0.55)))
                            fe.add_trace(go.Scatter(x=B2[:, 0], y=B2[:, 1], mode="markers", name="this trial", marker=dict(size=20, symbol="star", color=cc[row.y_true], line=dict(width=2, color=viz.INK))))
                            fe.update_layout(height=280, title=f"where fold {row.fold}'s model saw it", xaxis_title="PC1 of features", yaxis_title="PC2")
                            ui.plot(fe, "u_tfeat")
                        else:
                            ui.plot(viz.blank("no feature stage (MDM)"), "u_tfeat_b")
                    except Exception as e:
                        ui.note(f"Feature view unavailable: {ui.friendly_message(e)}")

            elif uview in ("Over time", "Replay"):
                tr_res = R.get("tr")
                tcol = st.columns([1, 1, 1, 1.4])
                win = tcol[0].slider("Window (s)", 0.25, 2.0, 1.0, 0.25, help="Length of the sliding decoding window.")
                stp = tcol[1].slider("Step (s)", 0.1, 0.5, 0.25, 0.05, help="Shift between windows.")
                cap = tcol[2].slider("Folds used", 2, max(2, len(res["splits"])), min(5, len(res["splits"])), help="Fewer folds = faster.")
                with tcol[3]:
                    st.write("")
                    go_tr = st.button("Run time analysis (heavy)", type="primary", use_container_width=True, key="u_trrun")
                if go_tr:
                    Xf, yf, mf, tf = C.get_Xy(ep, rcfg["band"], (float(ep.times[0]), float(ep.times[-1])), rcfg.get("classes"), rcfg.get("pre"))
                    spl = res["splits"][:cap]
                    pipe = res["pipe"]

                    def job(cb):
                        a_ = C.time_resolved(pipe, Xf, yf, tf, spl, win, stp, progress=lambda f, m: cb(0.55 * f, m))
                        lc = C.learning_curve_manual(pipe, X, y, spl, seed=seed, progress=lambda f, m: cb(0.55 + 0.25 * f, m))
                        itr_rows = []
                        lens = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
                        w0 = max(float(rcfg["window"][0]), float(tf[0]))
                        for k_, L in enumerate(lens):
                            msk = (tf >= w0) & (tf <= w0 + L)
                            if msk.sum() < 10 or w0 + L > tf[-1] + 1e-9:
                                continue
                            accs = [accuracy_score(yf[te], clone(pipe).fit(Xf[tr][:, :, msk], yf[tr]).predict(Xf[te][:, :, msk])) for tr, te in spl]
                            overhead = max(0.0, float(rcfg.get("trial_len", 4.0)) - (rcfg["window"][1] - rcfg["window"][0]))
                            itr_rows.append({"L": L, "acc": np.mean(accs), "itr": C.itr_bits(np.mean(accs), len(np.unique(yf)), L + overhead)})
                            cb(0.8 + 0.2 * (k_ + 1) / len(lens), f"ITR window {L} s")
                        return {"tr": a_, "lc": lc, "itr": pd.DataFrame(itr_rows), "win": win, "key": res_store["key"]}
                    R["tr"] = tr_res = ui.run_with_progress("Time analysis", job)
                if not tr_res or tr_res.get("key") != res_store["key"]:
                    ui.note("Press <b>Run time analysis</b>. It re-runs the same leak-free CV in a sliding window across the trial, plus a learning curve and an ITR sweep.", "t")
                elif uview == "Over time":
                    trr = tr_res["tr"]
                    theo, thr = C.chance_levels(len(y), len(cls))
                    ft = go.Figure()
                    ft.add_hrect(y0=0, y1=thr, fillcolor=viz.RULE2, opacity=0.6, line_width=0)
                    ft.add_trace(go.Scatter(x=np.r_[trr["t"], trr["t"][::-1]], y=np.r_[trr["acc_mean"] + trr["acc_std"], (trr["acc_mean"] - trr["acc_std"])[::-1]],
                                            fill="toself", fillcolor=viz.rgba(viz.VERM, 0.18), line_width=0, hoverinfo="skip", showlegend=False))
                    ft.add_trace(go.Scatter(x=trr["t"], y=trr["acc_mean"], mode="lines+markers", line=dict(color=viz.VERM, width=2.5), marker=dict(size=6), name="CV accuracy"))
                    ft.add_vline(0, line=dict(color=viz.OCHRE, width=2), annotation_text="CUE", annotation_font=dict(family=viz.MONO, size=10, color=viz.OCHRE))
                    ft.add_hline(theo, line=dict(color=viz.MUTED, dash="dot"))
                    best = int(np.argmax(trr["acc_mean"]))
                    ft.add_annotation(x=trr["t"][best], y=trr["acc_mean"][best], text=f"peak {trr['acc_mean'][best]:.0%} @ {trr['t'][best]:.2f} s", ax=0, ay=-34,
                                      font=dict(family=viz.MONO, size=11), arrowcolor=viz.INK)
                    ft.update_layout(height=330, xaxis_title="window centre (s from cue)", yaxis=dict(range=[0.2 if len(cls) == 2 else 0, 1.02], title="accuracy"))
                    ui.plot(ft, "u_time", name="time_resolved_accuracy")
                    ui.figcap("8.3", f"grey band = below the binomial significance threshold ({thr:.0%}). When does the brain 'decide'?")
                    a, b = st.columns(2)
                    with a:
                        lc = tr_res["lc"].groupby("n_train").agg(test=("test_acc", "mean"), test_sd=("test_acc", "std"), train=("train_acc", "mean")).reset_index()
                        fl = go.Figure()
                        fl.add_trace(go.Scatter(x=lc.n_train, y=lc.train, name="train", line=dict(color=viz.MUTED, dash="dash")))
                        fl.add_trace(go.Scatter(x=lc.n_train, y=lc.test, name="test (held-out)", line=dict(color=viz.INK, width=2.5),
                                                error_y=dict(type="data", array=lc.test_sd.fillna(0), color=viz.INK, thickness=1)))
                        fl.update_layout(height=290, title="learning curve: how many trials do you need?", xaxis_title="training trials", yaxis=dict(range=[0.3, 1.02], title="accuracy"))
                        ui.plot(fl, "u_lc", name="learning_curve")
                    with b:
                        it = tr_res["itr"]
                        if len(it):
                            fi = make_subplots(specs=[[{"secondary_y": True}]])
                            fi.add_trace(go.Bar(x=it.L, y=it.itr, name="bits/min", marker_color=viz.INK, width=0.3), secondary_y=False)
                            fi.add_trace(go.Scatter(x=it.L, y=it.acc, name="accuracy", line=dict(color=viz.VERM, width=2.5), marker=dict(size=8)), secondary_y=True)
                            bi = int(it.itr.idxmax())
                            fi.add_annotation(x=it.L[bi], y=it.itr[bi], text=f"best {it.itr[bi]:.1f} bits/min", ax=0, ay=-28, font=dict(family=viz.MONO, size=11))
                            fi.update_layout(height=290, title="ITR frontier: speed vs accuracy", xaxis_title="decoding window (s)")
                            fi.update_yaxes(title_text="bits / min", secondary_y=False)
                            fi.update_yaxes(title_text="accuracy", range=[0.3, 1.02], secondary_y=True, showgrid=False)
                            ui.plot(fi, "u_itr", name="itr_frontier")
                        else:
                            ui.note("The epoch is too short for an ITR sweep.")
                else:
                    trr = tr_res["tr"]
                    if len(cls) != 2:
                        ui.note("The cursor replay is built for two-class problems.", "")
                    else:
                        ok_tr = np.where(~np.isnan(trr["probs"][:, 0, 0]))[0]
                        tsel = st.session_state.get("u_trial", int(ok_tr[0]))
                        if tsel not in ok_tr:
                            tsel = int(ok_tr[0])
                        c1_, c2_ = st.columns([1, 3])
                        with c1_:
                            tsel = st.selectbox("Trial", ok_tr.tolist(), index=ok_tr.tolist().index(tsel),
                                                format_func=lambda t: f"#{t} · {viz.pretty(y[t])}", help="Pick a trial (or click one in Errors).")
                            st.session_state["u_trial"] = tsel
                            pr = trr["probs"][tsel, :, 1]
                            wmask = (trr["t"] >= rcfg["window"][0]) & (trr["t"] <= rcfg["window"][1] + 0.5)
                            final = float(np.nanmean(pr[wmask])) if wmask.any() else float(pr[-1])
                            ui.html('<div class="kv">' + "".join(f'<div class="k">{k}</div><div class="v">{v}</div>' for k, v in [
                                ("true", viz.pretty(y[tsel])), ("decoded", viz.pretty(cls[int(final > 0.5)])), (f"mean P({viz.pretty(cls[1])})", f"{final:.2f}"),
                                ("windows", len(pr))]) + "</div>")
                            ui.note("Each window's probability comes from a model trained on the other folds. This is an honest offline simulation of online feedback.", "t")
                        with c2_:
                            data = {"t": [float(v) for v in trr["t"]], "p": [float(v) if np.isfinite(v) else 0.5 for v in pr], "classes": [str(c_) for c_ in cls],
                                    "colors": [cc[c_] for c_ in cls], "truth": str(y[tsel]), "trial": int(tsel), "cue": 0.0, "final": final}
                            htmlx = (Path(__file__).parent.parent / "assets" / "fx" / "replay.html").read_text(encoding="utf-8").replace("__DATA__", json.dumps(data))
                            components.html(htmlx, height=420)

            elif uview == "Playground":
                ui.html('<div class="big-q">Drag the sliders and watch the decision boundary <em>bend</em>. Too flexible and it memorises the training trials.</div>')
                tr_idx, te_idx = res["splits"][0]
                pipe0 = res["pipe"] if not hasattr(res["pipe"], "best_estimator_") else res["pipe"].estimator
                from sklearn.pipeline import Pipeline as SkP
                if len(pipe0.steps) < 2:
                    ui.note("This pipeline has no feature stage (MDM). Pick another recipe to use the playground.")
                    return
                head = SkP(clone(pipe0).steps[:-1]).fit(X[tr_idx], y[tr_idx])
                pca = PCA(2, random_state=seed).fit(head.transform(X[tr_idx]))
                A2, B2 = pca.transform(head.transform(X[tr_idx])), pca.transform(head.transform(X[te_idx]))
                sc = np.abs(A2).max(0) * 1.15
                pc = st.columns(5)
                kind = pc[0].selectbox("Classifier", ["svm", "knn", "lda", "lr", "rf", "mlp"], format_func=lambda k: C.CLASSIFIERS[k].split(" (")[0])
                Cv = pc[1].select_slider("C", [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], 1.0, help="SVM/LR regularisation.")
                gam = pc[2].select_slider("RBF γ", [0.01, 0.1, 0.5, 1.0, 5.0, 20.0, 100.0], 0.5, help="SVM kernel width: large γ = wiggly boundary.")
                kk = pc[3].slider("k (kNN)", 1, 41, 9, 2)
                dep = pc[4].slider("Depth (RF)", 1, 20, 4)
                from sklearn.svm import SVC
                mdl = {"svm": SVC(C=Cv, gamma=gam, probability=True, random_state=seed),
                       "knn": C.make_classifier("knn", {"k": kk}), "lda": C.make_classifier("lda", {}), "lr": C.make_classifier("lr", {"C": Cv}),
                       "rf": C.make_classifier("rf", {"max_depth": dep, "n_estimators": 150}, seed), "mlp": C.make_classifier("mlp", {"hidden": 64, "alpha": 1e-4}, seed)}[kind]
                mdl.fit(A2, y[tr_idx])
                g1, g2 = np.meshgrid(np.linspace(-sc[0], sc[0], 120), np.linspace(-sc[1], sc[1], 120))
                pz = mdl.predict_proba(np.c_[g1.ravel(), g2.ravel()])[:, -1].reshape(g1.shape)
                tra, tea = mdl.score(A2, y[tr_idx]), mdl.score(B2, y[te_idx])
                a, b = st.columns([2.2, 1])
                with a:
                    fp = go.Figure()
                    fp.add_trace(go.Contour(x=g1[0], y=g2[:, 0], z=pz, colorscale=[[0, viz.rgba(cc[cls[0]], 0.55)], [0.5, "#F3EFE6"], [1, viz.rgba(cc[cls[-1]], 0.55)]],
                                            zmin=0, zmax=1, showscale=False, contours=dict(start=0.5, end=0.5, size=0.5, coloring="heatmap", showlines=True),
                                            line=dict(width=2.5, color=viz.INK), hoverinfo="skip"))
                    for c_ in cls:
                        mk = y[tr_idx] == c_
                        fp.add_trace(go.Scatter(x=A2[mk, 0], y=A2[mk, 1], mode="markers", name=f"train {viz.pretty(c_)}", marker=dict(size=7, color=cc[c_], line=dict(width=0.6, color=viz.PAPER))))
                        mk = y[te_idx] == c_
                        fp.add_trace(go.Scatter(x=B2[mk, 0], y=B2[mk, 1], mode="markers", name=f"test {viz.pretty(c_)}", marker=dict(size=11, symbol="diamond-open", color=cc[c_], line=dict(width=2.2))))
                    fp.update_layout(height=470, xaxis=dict(range=[-sc[0], sc[0]], title="feature PC1 (fit on training fold)"), yaxis=dict(range=[-sc[1], sc[1]], title="feature PC2"))
                    ui.plot(fp, "u_play", name="decision_playground")
                with b:
                    gap_ = tra - tea
                    ui.html(f'<div class="insp-big">{tea:.0%}<small> test</small></div><div class="figcap">train {tra:.0%} · gap {gap_ * 100:+.0f} pts</div>')
                    lvl = min(1.0, max(0.0, gap_ / 0.3))
                    ui.html(f'<div class="kicker" style="margin-top:12px">Overfitting meter</div><div class="score-bar" style="height:14px"><span style="width:{lvl * 100:.0f}%;background:{viz.VERM if lvl > 0.5 else viz.OCHRE if lvl > 0.2 else viz.GREEN}"></span></div>')
                    ui.html('<div class="reading">' + ("Huge train/test gap: the boundary wraps around individual training trials (memorising)." if gap_ > 0.15 else
                             "Small gap: the model generalises. Can a simpler one do as well?" if gap_ < 0.05 else "Some overfitting. Try more regularisation.") + "</div>")
                    ui.note("Hollow diamonds are test trials of fold 1. The model never saw them. Only 2-D here, so absolute accuracy is lower than the full pipeline's.", "")

            else:
                ui.html(ui.tag("Deployment model · fitted on ALL trials", "solid") + ui.tag("for interpretation and export only · not an evaluation", "o"))
                fin = res["final"]
                fin = fin.best_estimator_ if hasattr(fin, "best_estimator_") else fin
                names, xy = viz.sensor_xy(ep.info)
                if "csp" in fin.named_steps:
                    csp = fin.named_steps["csp"]
                    k = csp.n_components
                    ui.kicker("CSP activation patterns used by the final model")
                    ui.plot(viz.topo_grid([csp.patterns_[i] for i in range(k)], xy, names,
                                          [f"C{i + 1} · λ={csp.eigenvalues_[i]:.2f} · {viz.pretty(csp.favours_[i])}" for i in range(k)], cols=min(6, k), height=200),
                            "u_pats", name="final_csp_patterns")
                clf = fin.named_steps.get("clf")
                coef = getattr(clf, "coef_", None)
                if coef is not None:
                    w = np.abs(np.atleast_2d(coef)).mean(0)
                    o_ = np.argsort(w)[::-1][:20]
                    ui.kicker("Feature weights (|coefficient| on standardised features)")
                    fw = go.Figure(go.Bar(x=[f"f{i + 1}" for i in o_], y=w[o_], marker_color=viz.INK))
                    fw.update_layout(height=260, yaxis_title="|w|")
                    ui.plot(fw, "u_coef", name="feature_importance")
                elif hasattr(clf, "feature_importances_"):
                    w = clf.feature_importances_
                    o_ = np.argsort(w)[::-1][:20]
                    fw = go.Figure(go.Bar(x=[f"f{i + 1}" for i in o_], y=w[o_], marker_color=viz.INK))
                    fw.update_layout(height=260, yaxis_title="importance", title="random-forest feature importance")
                    ui.plot(fw, "u_imp", name="feature_importance")
                elif clf.__class__.__name__ == "MDM":
                    ui.kicker("MDM class prototypes (Riemannian mean covariances)")
                    hm = make_subplots(1, len(clf.covmeans_), subplot_titles=[viz.pretty(c_) for c_ in clf.classes_])
                    for i, Mx in enumerate(clf.covmeans_):
                        hm.add_trace(go.Heatmap(z=Mx[::-1], colorscale=viz.DIVERGING, zmid=0, showscale=False), 1, i + 1)
                    hm.update_layout(height=300)
                    ui.plot(hm, "u_mdm", name="mdm_means")
                if "best_params" in res["folds"]:
                    ui.kicker("Hyper-parameters chosen by the inner CV in each outer fold")
                    st.dataframe(res["folds"][["fold", "best_params", res["metric"]]], hide_index=True, use_container_width=True)

        results_views()


    # ============================================================ COMPETE
    elif stage == "Compete":
        X, y, meta, times = C.get_Xy(ep, cfg["band"], cfg["window"], cfg.get("classes"), cfg.get("pre"))
        try:
            splits = C.make_splits(cfg["cv"], y, meta, seed)
        except Exception as e:
            ui.note(f"Protocol not possible: {ui.friendly_message(e)}", "v")
            st.stop()
        erp = any(k in " ".join(map(str, np.unique(y))).lower() for k in ("target", "auditory", "visual"))
        options = list(C.RECIPES) + ["Your design"]
        default = ["xDAWN + LR", "Tangent space + LR", "MDM", "Band power + SVM"] if erp else ["CSP + LDA", "FBCSP + LDA", "Tangent space + LR", "MDM", "Band power + SVM"]
        a, b = st.columns([3, 1])
        with a:
            chosen = st.multiselect("Contenders (4 or more)", options, default=default, help="All pipelines see exactly the same folds.")
        with b:
            st.write("")
            go_b = st.button("Start the race", type="primary", use_container_width=True, disabled=len(chosen) < 2)
        ui.html(f'<div class="figcap">protocol: <b>{SCHEMES[cfg["cv"]["scheme"]]}</b> · {len(splits)} folds · metric: {C.choose_metric(y).replace("_", " ")} · seed {seed}</div>')
        if len(chosen) < 4:
            ui.note("Pick at least four pipelines for a proper benchmark.", "")
        if go_b:
            metric = C.choose_metric(y)
            pipes = {}
            for n in chosen:
                c2 = copy.deepcopy(cfg)
                if n in C.RECIPES:
                    c2.update(feature=C.RECIPES[n]["feature"], clf=C.RECIPES[n]["clf"], clf_params={}, nested=False)
                pipes[n] = C.maybe_nested(C.make_pipeline(c2, sf, times, seed, FP), c2, seed)
            scores = {n: [] for n in chosen}
            race = st.empty()
            bar = st.progress(0.0, text="the race begins")
            for i, (tr, te) in enumerate(splits):
                for n, p in pipes.items():
                    m = clone(p).fit(X[tr], y[tr])
                    scores[n].append(C.score_metric(y[te], m.predict(X[te]), metric))
                means = {n: np.mean(v) for n, v in scores.items()}
                srt = sorted(means, key=means.get)
                fr = go.Figure(go.Bar(y=srt, x=[means[n] for n in srt], orientation="h", marker_color=[viz.VERM if n == srt[-1] else viz.INK for n in srt],
                                      text=[f"{means[n]:.1%}" for n in srt], textposition="outside", textfont=dict(family=viz.MONO, size=13)))
                fr.add_vline(1 / len(np.unique(y)), line=dict(color=viz.MUTED, dash="dot"))
                fr.update_layout(height=60 + 46 * len(srt), xaxis=dict(range=[0, 1.08], title=f"running mean after fold {i + 1}/{len(splits)}"), margin=dict(l=150, t=10))
                race.plotly_chart(ui.pin(fr), use_container_width=True, theme=None, key=f"race_{i}")
                bar.progress((i + 1) / len(splits), text=f"fold {i + 1}/{len(splits)}")
            bar.empty()
            race.empty()
            df = pd.DataFrame(scores)
            n_tr, n_te = np.mean([len(t) for t, _ in splits]), np.mean([len(t) for _, t in splits])
            R["bench"] = {"scores": df, "stats": C.benchmark_stats(df, n_tr, n_te), "metric": metric, "protocol": SCHEMES[cfg["cv"]["scheme"]],
                          "chance": 1 / len(np.unique(y)), "key": cfg_key}
        bench = R.get("bench")
        if bench:
            df, stt = bench["scores"], bench["stats"]
            means, sds = df.mean(), df.std(ddof=1)
            order = means.sort_values(ascending=False).index.tolist()
            ui.kicker(f"Leaderboard · {bench['protocol']} · {bench['metric'].replace('_', ' ')}")
            rows = ""
            for r_, n in enumerate(order):
                rows += (f'<tr><td class="rank">{r_ + 1}</td><td class="name">{n}</td><td>{means[n]:.1%}</td><td>± {sds[n]:.1%}</td>'
                         f'<td style="width:38%"><div class="bar"><span style="width:{means[n] * 100:.1f}%;animation-delay:{r_ * 0.12:.2f}s"></span>'
                         f'<em style="left:{bench["chance"] * 100:.1f}%"></em></div></td><td>{df[n].min():.0%}–{df[n].max():.0%}</td></tr>')
            ui.html(f'<table class="lb"><tr><th></th><th>pipeline</th><th>mean</th><th>sd</th><th>score (red tick = chance)</th><th>fold range</th></tr>{rows}</table>')
            a, b = st.columns([1.3, 1])
            with a:
                hl = st.session_state.get("bench_hl")
                fl = go.Figure()
                for n in order:
                    on = (hl is None) or hl == n
                    fl.add_trace(go.Scatter(x=[n] * len(df), y=df[n], mode="markers", customdata=[n] * len(df), name=n,
                                            marker=dict(size=10, color=viz.VERM if hl == n else viz.INK, opacity=1 if on else 0.25), showlegend=False))
                for i in range(len(df)):
                    fl.add_trace(go.Scatter(x=order, y=[df[n].iloc[i] for n in order], mode="lines", line=dict(color=viz.rgba(viz.INK, 0.12), width=1), hoverinfo="skip", showlegend=False))
                fl.add_trace(go.Scatter(x=order, y=[means[n] for n in order], mode="markers", marker=dict(symbol="line-ew-open", size=40, color=viz.VERM, line=dict(width=3)),
                                        error_y=dict(type="data", array=[sds[n] for n in order], color=viz.VERM, thickness=1.5, width=10), showlegend=False, hoverinfo="skip"))
                fl.add_hline(bench["chance"], line=dict(color=viz.MUTED, dash="dot"))
                fl.update_layout(height=360, yaxis=dict(range=[0, 1.02], title=bench["metric"].replace("_", " ")), title="per-fold scores (lines = same fold) — click a pipeline")
                ev = ui.plot(fl, "b_folds", select=True, name="benchmark_folds")
                v = ui.on_click(ev, "b_folds")
                if v:
                    st.session_state["bench_hl"] = None if hl == v else v
                    st.rerun()
            with b:
                pr = stt["pairs"]
                M = np.full((len(order), len(order)), np.nan)
                for _, r_ in pr.iterrows():
                    i, j = order.index(r_.A), order.index(r_.B)
                    M[max(i, j), min(i, j)] = r_.corrected_t_p_holm
                    M[min(i, j), max(i, j)] = r_.wilcoxon_p_holm
                txt = [["" if np.isnan(v_) else f"{v_:.3f}" for v_ in row] for row in M]
                fm = go.Figure(go.Heatmap(z=M, x=order, y=order, colorscale=[[0, viz.VERM], [0.05, "#EDB39F"], [0.0501, viz.RULE2], [1, viz.PAPER]], zmin=0, zmax=1,
                                          text=txt, texttemplate="%{text}", textfont=dict(family=viz.MONO, size=11), showscale=False, hoverinfo="skip"))
                fm.update_layout(height=360, title="Holm-corrected p-values", yaxis=dict(autorange="reversed"), xaxis=dict(tickangle=-30))
                ui.plot(fm, "b_stats", name="benchmark_stats")
                ui.html('<div class="figcap">lower triangle: Nadeau-Bengio corrected resampled t-test · upper: Wilcoxon signed-rank. Red = p &lt; 0.05</div>')
            fr_ = stt.get("friedman")
            msg = f"Friedman test across all pipelines: χ² = {fr_.statistic:.2f}, p = {fr_.pvalue:.4f}. " if fr_ is not None else ""
            ui.note(msg + f"Fold scores are not independent (training sets overlap), so ordinary tests are too optimistic. The corrected t-test accounts for that. "
                    f"With only {len(df)} folds Wilcoxon cannot go below p ≈ {2 / 2 ** len(df):.3f}: use repeated k-fold for more power.", "t")
            st.download_button("Benchmark scores · CSV", df.to_csv(index_label="fold"), "benchmark_scores.csv", key="b_csv")

    # ============================================================ LEAKAGE GUARD
    elif stage == "Leakage guard":
        X, y, meta, times = C.get_Xy(ep, cfg["band"], cfg["window"], cfg.get("classes"), cfg.get("pre"))
        pipe = C.make_pipeline(cfg, sf, times, seed, FP)
        ui.html('<div class="big-q">What is fitted <em>where</em>? The rule: anything that learns from data, labels or statistics, '
                'may only see the training fold.</div>')
        rows = [("Filtering (Setup)", "continuous recording", "stateless: same fixed filter for every sample, no labels", "safe"),
                ("ICA (Setup, optional)", "whole recording", "unsupervised: never sees labels; mild transductive use, standard practice", "caution" if P.get("ica") else "not used"),
                ("Epoching / baseline (Setup)", "each trial alone", "per-trial operation", "safe"),
                ("Band-pass + crop (this page)", "each trial alone", "stateless", "safe")]
        for name, step in pipe.steps:
            lab, where = FIT_WHERE.get(name, (name, "fit: train fold"))
            rows.append((f"{lab} ({step.__class__.__name__})", "training fold only", "refit from scratch in every fold (sklearn Pipeline + clone)", "safe"))
        if cfg.get("nested") and cfg["clf"] in C.PARAM_GRIDS:
            rows.append(("Hyper-parameter search", "inner folds of the training fold", f"GridSearchCV over {list(C.PARAM_GRIDS[cfg['clf']])}", "safe"))
        rows.append(("Permutation test", "same protocol, shuffled labels", "complete re-run of the whole pipeline per shuffle", "safe"))
        rows.append(("Spatial-analysis exploratory plots", "all trials", "for looking only, their numbers never enter the evaluation", "display only"))
        colmap = {"safe": "g", "caution": "o", "not used": "", "display only": "t"}
        ui.html('<table class="lb"><tr><th>step</th><th>sees</th><th>why it is leak-free</th><th>status</th></tr>' + "".join(
            f'<tr><td class="name">{a}</td><td>{b_}</td><td>{c_}</td><td><span class="tag {colmap[d]}">{d}</span></td></tr>'
            for a, b_, c_, d in rows) + "</table>")
        st.write("")
        ui.html('<div class="big-q">The cheat experiment: <em>shuffled labels</em>, nothing to learn. Watch what leakage does.</div>')
        a, b = st.columns([1, 2.2])
        with a:
            nsh = st.slider("Shuffles", 3, 30, 10, help="Independent random relabellings.")
            kb = st.slider("Features selected", 2, 40, 10, help="Select-k-best on band-power features (7 bands × channels).")
            go_l = st.button("Run the cheat experiment", type="primary", use_container_width=True)
            ui.note("<b>Honest:</b> feature selection inside each training fold.<br><b>Leaky:</b> feature selection on all trials, then CV. The test labels helped choose the features.", "v")
        with b:
            if go_l:
                spl = C.make_splits({"scheme": "kfold", "k": 5}, y, meta, seed)
                R["leak"] = ui.run_with_progress("Cheat experiment", lambda cb: C.leakage_demo(X, y, spl, sf, nsh, kb, seed, cb))
            lk = R.get("leak")
            if lk is not None:
                theo, thr = C.chance_levels(len(y), len(np.unique(y)))
                fl = go.Figure()
                for col_, nm, c_ in (("honest", "honest pipeline", viz.TEAL), ("leaky", "leaky pipeline", viz.VERM)):
                    fl.add_trace(go.Box(y=lk[col_], name=nm, marker_color=c_, boxpoints="all", jitter=0.4, pointpos=0, line=dict(width=1.5)))
                fl.add_hline(theo, line=dict(color=viz.INK, dash="dot"), annotation_text="chance", annotation_font=dict(family=viz.MONO, size=10))
                fl.add_hline(thr, line=dict(color=viz.OCHRE, dash="dash"), annotation_text="significance threshold", annotation_font=dict(family=viz.MONO, size=10, color=viz.OCHRE))
                fl.update_layout(height=360, yaxis=dict(range=[0.2, 1], title="CV accuracy on SHUFFLED labels"), showlegend=False)
                ui.plot(fl, "l_box", name="leakage_demo")
                ui.html(f'<div class="reading">With random labels the honest pipeline scores <em>{lk.honest.mean():.0%}</em>, i.e. chance, as it should. '
                        f'The leaky one "achieves" <em>{lk.leaky.mean():.0%}</em> on pure noise. '
                        f'That is the size of the lie leakage tells. Every number in Understand and Compete comes from the honest construction.</div>')
                ui.html(ui.tag("demonstration of what NOT to do", "v") + ui.tag("never used as a result", ""))

    # ============================================================ LAB LOG
    else:
        runs = ui.S()["runs"]
        if not runs:
            ui.note("No runs yet. Every evaluation you run is logged here.", "t")
        else:
            df = pd.DataFrame(runs)
            fl = go.Figure(go.Scatter(x=list(range(1, len(df) + 1)), y=df.score, error_y=dict(type="data", array=df["std"], color=viz.INK2),
                                      mode="markers+lines", marker=dict(size=11, color=viz.VERM), line=dict(color=viz.RULE),
                                      text=df.recipe + " · " + df.protocol, hovertemplate="%{text}<br>%{y:.1%}<extra></extra>"))
            fl.add_trace(go.Scatter(x=list(range(1, len(df) + 1)), y=df.bet / 100, mode="markers", marker=dict(symbol="x-thin", size=12, line=dict(width=2, color=viz.INK)), name="your bet",
                                    hoverinfo="skip"))
            fl.update_layout(height=280, xaxis_title="run #", yaxis=dict(range=[0, 1.02], title="score"), showlegend=False, title="your experiments (× = your bet)")
            ui.plot(fl, "log_plot", name="lab_log")
            show = df.drop(columns=["cfg"]).copy()
            show["score"] = show.score.map("{:.1%}".format)
            show["std"] = show["std"].map("{:.1%}".format)
            show["p"] = show.p.map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
            st.dataframe(show, hide_index=True, use_container_width=True)
            a, b = st.columns(2)
            with a:
                st.download_button("Lab log · CSV", df.to_csv(index=False), "lab_log.csv", use_container_width=True)
            with b:
                pick = st.selectbox("Restore a configuration", list(range(len(df))), format_func=lambda i: f"run {i + 1} · {df.recipe[i]} · {df.protocol[i]}")
                if st.button("Restore", use_container_width=True):
                    st.session_state["cl_cfg"] = json.loads(df.cfg[pick])
                    go_stage("Build")

    def _save_design():
        ui.apply("classify", cfg, "Classification design saved")
        st.session_state["cl_src"] = ui.phash(cfg)
    ui.apply_control(slot, "classify", norm_cfg(cfg, P["spatial"]), "Save design", key="cl_apply", on_apply=_save_design,
                     current=norm_cfg(P["classify"], P["spatial"]),
                     help="Stores these classification settings in the shared pipeline (used by the generated script and the report).")

    ui.explain("Honest classification",
               ["A classifier is only as good as its performance on trials it has <b>never seen</b>, not even indirectly. "
                "Cross-validation splits trials into folds; every data-driven step (CSP, scaling, feature selection, hyper-parameter tuning) is refit on the training part and only applied to the test part.",
                "Accuracy means nothing without <b>chance</b>: with few test trials a random classifier can easily reach 60%. The binomial threshold and the permutation test tell you whether your number is distinguishable from luck. "
                "ITR (Wolpaw) combines accuracy, number of classes and speed into bits per minute."],
               [("You fit CSP on all 160 trials and then cross-validate LDA. What is wrong?", "CSP used the labels of the test trials to choose its filters, so the test folds are no longer independent. That is leakage, and accuracy is inflated (see the cheat experiment)."),
                ("With 40 test trials and 2 classes, is 62.5 % accuracy significant?", "No. The binomial threshold at α = 0.05 for n = 40 is about 62.5–65 %. You need more trials or a higher score."),
                ("Why is leave-one-session-out usually lower than k-fold?", "k-fold mixes trials from the same session into train and test, so it ignores day-to-day changes (electrode shift, fatigue). Cross-session is closer to real use."),
                ("When should you use nested cross-validation?", "Whenever hyper-parameters are tuned. The inner CV chooses them using the training fold only, and the outer test fold stays untouched."),
                ("Two pipelines differ by 2 % across 5 folds. Can you claim one is better?", "Rarely. Fold scores are correlated, so use a corrected test (Nadeau-Bengio) and more folds or repeats. Small differences are usually noise.")])
    code = f"""import numpy as np
from sklearn.model_selection import StratifiedKFold
from classify import make_pipeline, get_Xy, evaluate, permutation_null, chance_levels  # embedded in the generated script
cfg = {json.dumps(cfg)}
X, y, meta, times = get_Xy(epochs, cfg['band'], cfg['window'], cfg.get('classes'), cfg.get('pre'))
pipe = make_pipeline(cfg, epochs.info['sfreq'], times, seed={seed})
splits = make_splits(cfg['cv'], y, meta, seed={seed})
res = evaluate(pipe, X, y, splits, cfg['trial_len'])          # every fold: clone(pipe).fit(X[train]) -> predict(X[test])
null = permutation_null(pipe, X, y, splits, cfg['n_perm'], seed={seed})
"""
    files = []
    if res_store:
        rr = res_store["res"]
        files = [("Per-fold metrics · CSV", rr["folds"].to_csv(index=False), "fold_metrics.csv", "text/csv"),
                 ("Predictions (out-of-fold) · CSV", rr["oof"].to_csv(index=False), "predictions.csv", "text/csv")]
    ui.export_drawer("classify", code, cfg, files)
    downloads.downloads_panel(o, "cl")
