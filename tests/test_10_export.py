"""§10 Export: reproducibility of the generated script, notebook, report, params round-trip."""
import json
import re
import subprocess
import sys
import zipfile
import io

import nbformat

import classify as C
import core
import export as X


def test_generated_script_reproduces_app_metrics(tmp_path, params):
    p = json.loads(json.dumps(params))
    p["classify"]["n_perm"] = 5
    script = tmp_path / "pipe.py"
    script.write_text(X.make_script(p), encoding="utf-8")
    out = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("METRICS_JSON ")][0]
    m = json.loads(line[len("METRICS_JSON "):])
    app = C.run_classification(core.build(p)["epochs"], p["classify"], p["seed"], p["features"])
    for k in ("accuracy", "kappa", "f1"):
        assert abs(m["summary"][k]["mean"] - app["summary"][k][0]) < 1e-12, k
    assert abs(m["p_value"] - app["p_value"]) < 1e-12


def test_params_json_roundtrip(params):
    p = json.loads(json.dumps(params))
    p["filter"]["l_freq"] = 2.0
    back = X.params_from_json(json.dumps(p))
    assert back["filter"]["l_freq"] == 2.0 and back["seed"] == p["seed"]
    try:
        X.params_from_json('{"hello": 1}')
    except ValueError:
        pass
    else:
        raise AssertionError


def test_notebook_is_valid(params):
    nb = nbformat.reads(X.make_notebook(params), as_version=4)
    nbformat.validate(nb)
    assert any("run_classification" in c.source for c in nb.cells)


def test_report_and_zip(params):
    html = X.make_report(params, "test data", 10)
    assert "Limitations" in html and "seed" in html
    z = zipfile.ZipFile(io.BytesIO(X.zip_bytes({"a.txt": "x", "b/c.bin": b"\x00"})))
    assert sorted(z.namelist()) == ["a.txt", "b/c.bin"]
