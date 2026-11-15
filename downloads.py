"""Downloads panel shown at the bottom of the Spatial and Classification pages:
data, results, figures, parameters, standalone script, notebook, report, ZIP,
plus restoring a pipeline from parameters.json."""
import io
import json
import tempfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import streamlit as st

import export as X
import ui


def _build_files(o, exp_params, name, inc_raw, inc_csv, inc_figs):
    P = ui.params()
    R = ui.S()["results"]
    clf = R.get("clf")
    ep = o["epochs"]
    files = {f"{name}.py": X.make_script(exp_params), f"{name}.ipynb": X.make_notebook(exp_params),
             "parameters.json": json.dumps({k: v for k, v in exp_params.items() if k != "_name"}, indent=2, default=str)}
    tmp = Path(tempfile.mkdtemp())
    raw = o["raw"]
    if inc_raw:
        raw.save(tmp / "raw_clean_raw.fif", overwrite=True)
        files["data/raw_clean_raw.fif"] = (tmp / "raw_clean_raw.fif").read_bytes()
        try:
            mne.export.export_raw(tmp / "raw_clean.edf", raw.copy().pick("eeg"), fmt="edf", overwrite=True)
            files["data/raw_clean.edf"] = (tmp / "raw_clean.edf").read_bytes()
        except Exception as e:
            files["data/EDF_EXPORT_FAILED.txt"] = str(e)
    if inc_csv:
        d = raw.get_data(picks="eeg").T * 1e6
        files["data/raw_clean_uV.csv"] = "time_s," + ",".join(raw.copy().pick("eeg").ch_names) + "\n" + "\n".join(
            f"{t:.4f}," + ",".join(f"{v:.3f}" for v in row) for t, row in zip(raw.times, d))
    ep.save(tmp / "epochs-epo.fif", overwrite=True)
    files["data/epochs-epo.fif"] = (tmp / "epochs-epo.fif").read_bytes()
    buf = io.BytesIO()
    np.savez_compressed(buf, X=ep.get_data(picks="eeg"), y=ep.metadata.label.to_numpy().astype(str), ch_names=np.array(ep.ch_names),
                        sfreq=ep.info["sfreq"], times=ep.times, session=ep.metadata.session.to_numpy(), subject=ep.metadata.subject.to_numpy())
    files["data/epochs.npz"] = buf.getvalue()
    if clf:
        r = clf["res"]
        files["results/fold_metrics.csv"] = r["folds"].to_csv(index=False)
        files["results/predictions.csv"] = r["oof"].to_csv(index=False)
        files["results/metrics.json"] = json.dumps({"metric": r["metric"], "summary": r["summary"], "p_value": r["p_value"], "chance": r["chance"],
                                                    "binomial_threshold": r["binom_threshold"], "null": list(map(float, r["null"]))}, indent=2, default=float)
        import joblib
        mb = io.BytesIO()
        joblib.dump(r["final"], mb)
        files["results/model.joblib"] = mb.getvalue()
        try:
            from features import make_feature_union
            F = make_feature_union(P["features"]["kinds"], ep.info["sfreq"], r["times"], P["features"]).fit_transform(r["X"], r["y"])
            fb = io.BytesIO()
            np.save(fb, F)
            files["results/features.npy"] = fb.getvalue()
            files["results/features.csv"] = "label," + ",".join(f"f{i + 1}" for i in range(F.shape[1])) + "\n" + "\n".join(
                f"{lab}," + ",".join(f"{v:.6g}" for v in row) for lab, row in zip(r["y"], F))
        except Exception:
            pass
    if R.get("bench"):
        files["results/benchmark_scores.csv"] = R["bench"]["scores"].to_csv(index_label="fold")
        files["results/benchmark_stats.csv"] = R["bench"]["stats"]["pairs"].to_csv(index=False)
    if ui.S()["runs"]:
        files["results/lab_log.csv"] = pd.DataFrame(ui.S()["runs"]).to_csv(index=False)
    figs = ui.S()["figs"] if inc_figs else {}
    for k, f in figs.items():
        files[f"figures/{k}.html"] = f.to_html(include_plotlyjs="cdn", full_html=True)
    files["report.html"] = X.make_report(P, o["label"], len(ep), clf, R.get("bench"), figs)
    return files


def downloads_panel(o, where):
    """Everything downloadable, in one expander. `where` makes widget keys unique per page."""
    with st.expander("Downloads: data, model, script, notebook and report", icon=":material/download:"):
        P = ui.params()
        R = ui.S()["results"]
        clf = R.get("clf")
        exp_params = json.loads(json.dumps(P, default=str))
        if clf:
            exp_params["classify"] = clf["cfg"]
        c = st.columns([1, 1, 1, 1.4])
        inc_raw = c[0].toggle("Raw recording (FIF + EDF)", True, key=f"{where}_raw")
        inc_csv = c[1].toggle("Raw CSV (large)", False, key=f"{where}_csv")
        inc_figs = c[2].toggle("Figures (HTML)", True, key=f"{where}_figs")
        name = c[3].text_input("Script name", "bci_pipeline", key=f"{where}_name", label_visibility="collapsed")
        exp_params["_name"] = name
        rows = [("Reproduce", "standalone script (.py), notebook (.ipynb), parameters.json", True, ""),
                ("Data", "epochs (FIF, NPZ)" + (", raw (FIF, EDF)" if inc_raw else "") + (", raw CSV" if inc_csv else ""), True, ""),
                ("Results", "model, predictions, fold metrics, features", clf is not None, "run an evaluation in Classification first"),
                ("Benchmark", "scores and statistics", R.get("bench") is not None, "run Compete in Classification to include"),
                ("Report", "report.html: every step, metrics next to chance, limitations", True, "")]
        ready = '<span class="chip ok">Ready</span>'
        ui.html('<table class="lb"><tr><th>Bundle</th><th>Contains</th><th>Status</th></tr>' + "".join(
            '<tr><td class="name">' + a + "</td><td>" + b_ + "</td><td>" + (ready if ok else '<span class="chip">' + why + "</span>") + "</td></tr>"
            for a, b_, ok, why in rows) + "</table>")
        if st.button("Prepare files", type="primary", key=f"{where}_prep"):
            with st.spinner("Preparing files…"):
                st.session_state["export_files"] = _build_files(o, exp_params, name, inc_raw, inc_csv, inc_figs)
        files = st.session_state.get("export_files")
        if files:
            st.download_button(f"Download everything as ZIP ({len(files)} files)", X.zip_bytes(files), "bci_master_studio_export.zip",
                               type="primary", key=f"{where}_zip")
            groups = [("Reproduce", [k for k in files if "/" not in k and k != "report.html"]), ("Data", [k for k in files if k.startswith("data/")]),
                      ("Results", [k for k in files if k.startswith("results/")]), ("Report", ["report.html"])]
            cols = st.columns(4)
            for col, (title, keys) in zip(cols, groups):
                with col:
                    ui.deck_header(title, str(len(keys)))
                    for k in keys:
                        if k in files:
                            st.download_button(k.split("/")[-1], files[k], file_name=k.split("/")[-1], use_container_width=True, key=f"{where}_dl_{k}")
        st.divider()
        ui.kicker("Restore a pipeline from parameters.json")
        up = st.file_uploader("parameters.json", type=["json"], key=f"{where}_imp", label_visibility="collapsed")
        if up is not None and st.button("Restore pipeline", key=f"{where}_imp_go"):
            newp = X.params_from_json(up.getvalue().decode("utf-8"))
            ui.S()["history"].append(json.loads(json.dumps(P, default=str)))
            ui.S()["params"] = newp
            st.session_state.pop("cl_cfg", None)
            st.session_state.pop("export_files", None)
            st.toast("Pipeline restored from parameters.json")
            st.rerun()
