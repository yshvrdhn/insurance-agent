import numpy as np

from harness.data import load_split
from harness.evaluator import MACRO_WEIGHT, evaluate, evaluator_fingerprint


class _FixedModel:
    """A model whose predictions we control, to check the scoring math."""

    def __init__(self, preds):
        self._preds = preds

    def predict(self, x):
        return self._preds[: len(x)]


def test_fingerprint_is_stable():
    assert evaluator_fingerprint() == evaluator_fingerprint()
    assert len(evaluator_fingerprint()) == 64


def test_perfect_model_scores_ceiling(tiny_dataset):
    x, y, _ = load_split(tiny_dataset, "test")
    res = evaluate(_FixedModel(y.copy()), tiny_dataset)
    assert res["overall_accuracy"] == 1.0
    assert res["worst_group"] == 1.0
    assert abs(res["score"] - (1.0 + MACRO_WEIGHT * 1.0)) < 1e-6


def test_score_is_worst_group_plus_macro(tiny_dataset):
    x, y, _ = load_split(tiny_dataset, "test")
    rng = np.random.default_rng(0)
    preds = y.copy()
    flip = rng.random(len(preds)) < 0.3  # corrupt ~30%
    preds[flip] = (preds[flip] + 1) % 3
    res = evaluate(_FixedModel(preds), tiny_dataset)
    expected = res["worst_group"] + MACRO_WEIGHT * res["macro_accuracy"]
    assert abs(res["score"] - round(expected, 4)) < 1e-3
    assert res["worst_group"] == min(res["per_weather"].values())


def test_confusion_matrix_sums_to_n(tiny_dataset):
    x, y, _ = load_split(tiny_dataset, "test")
    res = evaluate(_FixedModel(y.copy()), tiny_dataset)
    assert np.array(res["confusion"]).sum() == res["n"]
