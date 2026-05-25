"""The immutable scoring function.

**The agent must never edit this file.** ``harness/loop.py`` hashes it before
and after every iteration and refuses to record a score if it changed. It is the
single source of truth for "did that change actually help?".

The official score (see ``program.md``):

    score = worst_group_accuracy + 0.25 * macro_accuracy

where groups are the ``weather`` slices. Optimizing the worst slice keeps the
model honest about the conditions it is bad at, rather than hiding a blind spot
behind a good average.

This module also returns rich per-slice diagnostics (à la FiftyOne's slice-based
evaluation): per-weather and per-scene accuracy, a confusion matrix, and a
per-sample correctness vector that downstream mining can use. None of those feed
the scalar score; they exist so a human (or the agent) can *understand* a result.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from harness.data import WEATHER, class_names, load_split

MACRO_WEIGHT = 0.25
_THIS_FILE = Path(__file__).resolve()


def evaluator_fingerprint() -> str:
    """SHA-256 of this file's bytes. The loop uses it as a tamper check."""
    return hashlib.sha256(_THIS_FILE.read_bytes()).hexdigest()


def _per_group_accuracy(correct: np.ndarray, groups: np.ndarray) -> dict:
    out = {}
    for g in sorted(set(groups.tolist())):
        sel = groups == g
        out[str(g)] = float(correct[sel].mean()) if sel.any() else float("nan")
    return out


def evaluate(model, dataset_path, split: str = "test") -> dict:
    """Score ``model`` on the held-out ``split`` (default: ``test``).

    ``model`` is anything exposing ``predict(X) -> np.ndarray[int]`` over float
    images of shape ``(N, H, W, C)``. Returns the official ``score`` plus
    diagnostics. This is the *only* place the test split is read.
    """
    x, y, meta = load_split(dataset_path, split)
    pred = np.asarray(model.predict(x)).astype(np.int64)
    correct = (pred == y).astype(np.float64)

    overall = float(correct.mean())
    classes = class_names()
    per_class = {
        classes[c]: float(correct[y == c].mean()) if (y == c).any() else float("nan")
        for c in range(len(classes))
    }
    macro = float(np.nanmean(list(per_class.values())))

    per_weather = _per_group_accuracy(correct, meta["weather"])
    per_scene = _per_group_accuracy(correct, meta["scene"])

    # the score: worst weather slice, nudged by macro accuracy
    worst_group = float(min(per_weather.values()))
    worst_group_name = min(per_weather, key=per_weather.get)
    score = worst_group + MACRO_WEIGHT * macro

    n_cls = len(classes)
    confusion = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y, pred):
        confusion[t, p] += 1

    return {
        "score": round(score, 4),
        "worst_group": round(worst_group, 4),
        "worst_group_name": worst_group_name,
        "overall_accuracy": round(overall, 4),
        "macro_accuracy": round(macro, 4),
        "per_weather": {k: round(v, 4) for k, v in per_weather.items()},
        "per_scene": {k: round(v, 4) for k, v in per_scene.items()},
        "per_class": {k: round(v, 4) for k, v in per_class.items()},
        "confusion": confusion.tolist(),
        "n": int(len(y)),
        "split": split,
        "correct": correct.astype(bool).tolist(),
        "ids": meta["id"].tolist(),
    }
