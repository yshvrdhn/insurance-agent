import pytest

from harness import IMMUTABLE_FILES, MODIFIABLE_FILES
from harness.loop import (REPO_ROOT, snapshot_immutable, verify_immutable)

torch = pytest.importorskip("torch")


def test_contract_files_exist():
    for rel in IMMUTABLE_FILES + MODIFIABLE_FILES:
        assert (REPO_ROOT / rel).exists(), f"missing contract file: {rel}"


def test_immutability_guard_passes_when_unchanged():
    snap = snapshot_immutable()
    verify_immutable(snap)  # no change -> no raise


def test_immutability_guard_detects_tamper():
    snap = snapshot_immutable()
    tampered = dict(snap)
    # pretend the evaluator was edited mid-iteration
    tampered["harness/evaluator.py"] = "deadbeef"
    # verify against the *real* current state, which differs from `tampered`
    with pytest.raises(RuntimeError, match="immutable files changed"):
        verify_immutable(tampered)


def test_evaluator_not_in_modifiable_set():
    assert "harness/evaluator.py" in IMMUTABLE_FILES
    assert "harness/evaluator.py" not in MODIFIABLE_FILES
    assert "harness/data.py" in IMMUTABLE_FILES


def test_loop_improves_over_baseline(tiny_dataset, tmp_path):
    from harness.config import load_config
    from harness.loop import run_loop

    # short, fast loop on the tiny fixture
    results = tmp_path / "results.tsv"
    out = run_loop(iters=3, dataset_path=tiny_dataset,
                   results_path=str(results), verbose=False)
    import harness.registry as registry

    rows = registry.read(str(results))
    assert len(rows) == 4  # baseline + 3 iters
    baseline = float(rows[0]["score"])
    assert out["best_score"] >= baseline  # never regresses below baseline
