"""Smoke test: every page opens without an exception or an error panel,
and the flagship flows (spatial views, a full classification run) work."""
import os

import pytest
from streamlit.testing.v1 import AppTest

os.environ["BCI_APPTEST"] = "1"
PAGES = ["pages/00_setup.py", "pages/01_spatial.py", "pages/02_classify.py"]


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    bad = [m.body for m in at.markdown if "Something went wrong" in m.body]
    assert not bad, bad[0][:300]


@pytest.mark.parametrize("page", PAGES)
def test_page_opens(page):
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()
    at.switch_page(page)
    at.run()
    _ok(at)


def test_spatial_concepts_views_and_methods():
    import concepts_spatial
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()
    at.switch_page("pages/01_spatial.py").run()
    for d in concepts_spatial.DEMOS:
        at.radio(key="sp_concept").set_value(d).run()
        _ok(at)
    at.radio(key="sp_mode").set_value("Your data").run()
    _ok(at)
    for v in ["Power maps", "Patterns", "Pull apart", "3-D head", "Beat CSP", "Stability", "Truth"]:
        at.radio(key="sp_view").set_value(v).run()
        _ok(at)
    for m in ["Riemann", "xDAWN", "PCA", "Whitening", "CAR", "Laplacian"]:
        at.radio(key="sp_method").set_value(m).run()
        _ok(at)


def test_classification_concepts_and_full_flow():
    import concepts_classify
    at = AppTest.from_file("app.py", default_timeout=600)
    at.run()
    at.switch_page("pages/02_classify.py").run()
    for d in concepts_classify.DEMOS:
        at.radio(key="cl_concept").set_value(d).run()
        _ok(at)
    at.radio(key="mission_0").set_value("Build").run()
    _ok(at)
    at.button(key="cl_run").click().run()
    _ok(at)
    assert at.session_state["P"]["results"]["clf"]["res"]["summary"]["accuracy"][0] > 0.6
    for v in ["Errors", "Playground", "Inside the model"]:
        at.radio(key="u_view").set_value(v).run()
        _ok(at)
