import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.data import DatasetSpec, make_synthetic_dataset  # noqa: E402


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    """A small, fast synthetic dataset shared across the test session."""
    path = tmp_path_factory.mktemp("data") / "bdd-test.lance"
    make_synthetic_dataset(path, DatasetSpec(n=1200, seed=7))
    return str(path)


@pytest.fixture
def fast_config():
    """A tiny-budget config so training tests finish in a couple of seconds."""
    from harness.config import load_config

    return load_config("base", {"budget": {"max_epochs": 8, "max_seconds": 30}})
