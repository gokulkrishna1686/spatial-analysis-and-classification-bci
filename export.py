"""Reproducibility exports: standalone script, Jupyter notebook, HTML report, ZIP.

The generated script embeds the app's own computational modules (core, spatial,
features, classify) verbatim, so it runs without the web app and produces the
same numbers for the same parameters and seed."""
import datetime as dt
import io
import json
import platform
import re
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
LIB_MODULES = ["core.py", "spatial.py", "features.py", "classify.py"]


def library_source():
    parts = []
    for m in LIB_MODULES:
        src = (ROOT / m).read_text(encoding="utf-8")
        src = "\n".join(l for l in src.splitlines() if "# @local-import" not in l)
        parts.append(f"# {'=' * 74}\n# ---- {m} (embedded from BCI Master Studio)\n# {'=' * 74}\n{src}\n")
    return "\n".join(parts)


def versions():
    import mne, sklearn, scipy, pyriemann
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "mne": mne.__version__,
            "scikit-learn": sklearn.__version__, "pyriemann": pyriemann.__version__}


MAIN = r'''

# ============================================================================
# ---- main: the pipeline exactly as configured in the app
# ============================================================================

def main(params=PARAMS, out_dir="bci_results"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    seed = params["seed"]
    print("1/4  loading data + filter + ICA + epochs ...")
    objs = build(params)
    epochs = objs["epochs"]
    print(f"     {objs['label']}: {len(epochs)} trials, {len(epochs.ch_names)} channels, {epochs.info['sfreq']:.0f} Hz")
    cfg = params["classify"]
    print(f"2/4  leak-free evaluation: {cfg['feature']} + {cfg['clf']} · {cfg['cv']['scheme']} · {cfg['n_perm']} permutations")
    res = run_classification(epochs, cfg, seed, params.get("features"))
    m = res["metric"]
    summary = {k: {"mean": v[0], "std": v[1]} for k, v in res["summary"].items()}
    out = {"metric": m, "summary": summary, "p_value": res["p_value"], "chance": res["chance"],
           "binomial_threshold": res["binom_threshold"], "n_test_total": res["n_test_total"],
           "folds": res["folds"].drop(columns=[c for c in ["best_params"] if c in res["folds"]]).to_dict(orient="list")}
    print("3/4  results")
    print(f"     {m}: {summary[m]['mean']:.4f} ± {summary[m]['std']:.4f}   chance {res['chance']:.3f}"
          f"   binomial threshold {res['binom_threshold']:.3f}   permutation p = {res['p_value']}")
    res["folds"].to_csv(os.path.join(out_dir, "fold_metrics.csv"), index=False)
    res["oof"].to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("4/4  fitting the deployment model on all trials (for export only)")
    try:
        import joblib
        joblib.dump(fit_final(res["pipe"], res["X"], res["y"]), os.path.join(out_dir, "model.joblib"))
    except Exception as e:
        print("     model not saved:", e)
    print("     written to", os.path.abspath(out_dir))
    print("METRICS_JSON " + json.dumps(out, default=float))
    return out


if __name__ == "__main__":
    main()
'''


def make_script(params):
    p = json.loads(json.dumps(params, default=str))
    if p["data"].get("source") == "upload":
        p["data"]["path"] = Path(p["data"]["path"]).name
    head = f'''#!/usr/bin/env python
"""BCI Master Studio · reproducible pipeline
Generated {dt.datetime.now():%Y-%m-%d %H:%M} — runs WITHOUT the web app.

    pip install mne scikit-learn pyriemann moabb scipy numpy pandas
    python {params.get('_name', 'bci_pipeline')}.py

Data source : {p['data'].get('source')}{'  (put the file next to this script)' if p['data'].get('source') == 'upload' else ''}
Seed        : {p['seed']}  (same seed + same library versions -> same numbers as in the app)
Library     : the app's own modules core / spatial / features / classify are embedded below.
"""
import json

PARAMS = json.loads(r"""{json.dumps(p, indent=2)}""")

'''
    return head + library_source() + MAIN


def make_notebook(params):
    import nbformat as nbf
    p = json.loads(json.dumps(params, default=str))
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell("# BCI Master Studio: reproducible pipeline\n"
                                 f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Runs without the web app. "
                                 "The library cell embeds the app's own processing code, so results match the app for the same seed."),
        nbf.v4.new_code_cell("import json\nPARAMS = json.loads(r'''" + json.dumps(p, indent=2) + "''')\nPARAMS['seed']"),
        nbf.v4.new_markdown_cell("## Library (embedded from the app)"),
        nbf.v4.new_code_cell(library_source()),
        nbf.v4.new_markdown_cell("## 1 · Data → filter → ICA → epochs"),
        nbf.v4.new_code_cell("objs = build(PARAMS)\nepochs = objs['epochs']\nprint(objs['label'])\nepochs"),
        nbf.v4.new_markdown_cell("## 2 · Leak-free classification\nEvery data-fitted step lives inside an sklearn Pipeline that is refit on each training fold."),
        nbf.v4.new_code_cell("res = run_classification(epochs, PARAMS['classify'], PARAMS['seed'], PARAMS.get('features'))\n"
                             "m = res['metric']\nprint(m, res['summary'][m], 'chance', res['chance'], 'binomial threshold', res['binom_threshold'], 'p', res['p_value'])\n"
                             "res['folds']"),
        nbf.v4.new_markdown_cell("## 3 · Confusion matrix and null distribution"),
        nbf.v4.new_code_cell("import matplotlib.pyplot as plt\nfig, ax = plt.subplots(1, 2, figsize=(10, 4))\n"
                             "ax[0].imshow(res['confusion'], cmap='Greys'); ax[0].set_xticks(range(len(res['classes']))); ax[0].set_xticklabels(res['classes'])\n"
                             "ax[0].set_yticks(range(len(res['classes']))); ax[0].set_yticklabels(res['classes']); ax[0].set_title('confusion (first repeat)')\n"
                             "if len(res['null']): ax[1].hist(res['null'], 30, color='0.8')\n"
                             "ax[1].axvline(res['summary'][m][0], color='#D6492A', lw=2); ax[1].axvline(res['binom_threshold'], ls='--', color='#C08A1E'); ax[1].set_title('null distribution')\nplt.show()"),
    ]
    nb["cells"] = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nbf.writes(nb)


def _tbl(d):
    return "<table>" + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in d.items()) + "</table>"


def make_report(params, label, n_trials, clf_store=None, bench=None, figs=None):
    figs = figs or {}
    first = True
    fig_html = []
    for name, fig in list(figs.items())[:24]:
        try:
            fig_html.append(f"<figure><figcaption>{name.replace('_', ' ')}</figcaption>"
                            + fig.to_html(full_html=False, include_plotlyjs="inline" if first else False, config={"displaylogo": False}) + "</figure>")
            first = False
        except Exception:
            pass
    steps = []
    for k in ("data", "filter", "ica", "epochs", "spatial", "features", "classify"):
        v = params.get(k)
        status = "applied" if k in params.get("applied", []) else ("default" if v else "not used")
        steps.append(f"<h3>{k} <span class='tag'>{status}</span></h3><pre>{json.dumps(v, indent=1, default=str)}</pre>")
    metrics = "<p>No classification has been run yet.</p>"
    if clf_store:
        r = clf_store["res"]
        m = r["metric"]
        rows = {k: f"{v[0]:.4f} ± {v[1]:.4f}" for k, v in r["summary"].items()}
        verdict = "significantly above chance" if (r["p_value"] < 0.05 and r["summary"][m][0] > r["binom_threshold"]) else "NOT convincingly above chance"
        metrics = (f"<p>Protocol: <b>{clf_store['cfg']['cv']}</b>, nested CV: {clf_store['cfg'].get('nested', False)}, "
                   f"{len(r['splits'])} splits, {r['n_test_total']} test trials.</p>" + _tbl(rows) +
                   f"<p>Theoretical chance {r['chance']:.3f} · binomial threshold (α = 0.05) {r['binom_threshold']:.3f} · "
                   f"permutation p = {r['p_value']:.4f} ({len(r['null'])} permutations) → <b>{verdict}</b>.</p>"
                   + r["folds"].drop(columns=[c for c in ["best_params"] if c in r["folds"]]).round(4).to_html(index=False))
    bench_html = ""
    if bench:
        df = bench["scores"]
        bench_html = (f"<h2>Benchmark</h2><p>{bench['protocol']} · metric {bench['metric']}</p>"
                      + df.agg(["mean", "std"]).T.sort_values("mean", ascending=False).round(4).to_html()
                      + bench["stats"]["pairs"].round(4).to_html(index=False))
    css = ("body{font-family:Georgia,serif;max-width:1000px;margin:40px auto;color:#17191E;background:#F5F2EA;padding:0 20px}"
           "h1{font-weight:400;font-size:2.4rem;border-bottom:2px solid #17191E}h2{font-weight:400;color:#D6492A;margin-top:2em}"
           "table{border-collapse:collapse;font-family:monospace;font-size:13px}td,th{border-bottom:1px solid #D8D2C4;padding:4px 10px;text-align:left}"
           "pre{background:#FBF9F4;border:1px solid #D8D2C4;padding:8px;font-size:12px;overflow:auto}"
           ".tag{font-family:monospace;font-size:11px;border:1px solid #6B6F76;padding:1px 6px;color:#6B6F76}figure{margin:20px 0}"
           "figcaption{font-family:monospace;font-size:12px;color:#6B6F76}")
    lim = ["Results are from offline cross-validation. Online closed-loop performance is usually lower.",
           "Unsupervised preprocessing (filtering, ICA) is applied to the whole recording before splitting. Labels are never used there, but it is mildly transductive.",
           "Fold scores are not independent; benchmark p-values use the Nadeau-Bengio correction and are still approximate.",
           "The deployment model is fitted on all trials for export/interpretation only; its training accuracy is not an evaluation.",
           "Synthetic data have known ground truth but are simpler than real EEG."]
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>BCI run report</title><style>{css}</style></head><body>"
            f"<h1>BCI Master Studio · run report</h1><p>{dt.datetime.now():%Y-%m-%d %H:%M} · seed {params['seed']}</p>"
            f"<h2>Dataset</h2><p>{label} · {n_trials} trials</p><h2>Pipeline steps</h2>{''.join(steps)}"
            f"<h2>Evaluation and metrics</h2>{metrics}{bench_html}<h2>Figures</h2>{''.join(fig_html) or '<p>none yet</p>'}"
            f"<h2>Limitations</h2><ul>{''.join(f'<li>{x}</li>' for x in lim)}</ul>"
            f"<h2>Software</h2>{_tbl(versions())}</body></html>")


def zip_bytes(files):
    """files: dict name -> bytes|str"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data if isinstance(data, (bytes, bytearray)) else str(data))
    return buf.getvalue()


def params_from_json(text):
    p = json.loads(text)
    if not isinstance(p, dict) or "data" not in p or "seed" not in p:
        raise ValueError("This JSON is not a BCI Master Studio parameter file (missing 'data' / 'seed').")
    from core import default_params
    base = default_params(p.get("seed", 42))
    base.update({k: v for k, v in p.items() if k in base})
    return base
