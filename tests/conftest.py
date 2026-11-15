import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("BCI_APPTEST", "1")

import core  # noqa: E402


@pytest.fixture(scope="session")
def params():
    p = core.default_params(42)
    p["data"]["trials_per_class"] = 30
    return p


@pytest.fixture(scope="session")
def objs(params):
    return core.build(params)


@pytest.fixture(scope="session")
def epochs(objs):
    return objs["epochs"]
