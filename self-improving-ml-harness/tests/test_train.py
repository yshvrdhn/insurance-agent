import numpy as np
import pytest

torch = pytest.importorskip("torch")

from harness.evaluator import evaluate
from harness.mining import compute_sample_weights
from harness.train import train


def test_train_returns_usable_model(tiny_dataset, fast_config):
    model = train(fast_config, tiny_dataset)
    x = np.random.rand(5, 32, 32, 3).astype(np.float32)
    preds = model.predict(x)
    assert preds.shape == (5,)
    assert set(np.unique(preds)).issubset({0, 1, 2})


def test_training_beats_chance(tiny_dataset, fast_config):
    model = train(fast_config, tiny_dataset)
    res = evaluate(model, tiny_dataset)
    assert res["overall_accuracy"] > 0.5  # chance is ~0.33


def test_budget_caps_epochs(tiny_dataset):
    from harness.config import load_config

    cfg = load_config("base", {"budget": {"max_epochs": 2, "max_seconds": 60}})
    model = train(cfg, tiny_dataset)
    assert model.metrics["epochs_run"] <= 2


def test_sample_weights_shape_matches_train(tiny_dataset):
    cfg_on = {"mining": {"enabled": True, "strategy": "fog_boost", "boost": 4.0}}
    from harness.config import load_config

    w = compute_sample_weights(tiny_dataset, load_config("base", cfg_on))
    from harness.data import load_split

    xtr, _, _ = load_split(tiny_dataset, "train")
    assert w.shape == (len(xtr),)
    assert (w > 0).all()
    assert w.max() > 1.0  # some frames were upweighted
