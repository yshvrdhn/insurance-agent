import collections

import numpy as np

from harness.data import (IMG_C, IMG_H, IMG_W, WEATHER, DatasetSpec,
                          build_synthetic, load_split, make_synthetic_dataset)


def test_generation_is_deterministic():
    a = build_synthetic(DatasetSpec(n=200, seed=1))
    b = build_synthetic(DatasetSpec(n=200, seed=1))
    assert np.array_equal(a["image"], b["image"])
    assert np.array_equal(a["label"], b["label"])


def test_splits_disjoint_and_shaped(tiny_dataset):
    ids = {}
    for split in ("train", "val", "test"):
        x, y, meta = load_split(tiny_dataset, split)
        assert x.shape[1:] == (IMG_H, IMG_W, IMG_C)
        assert x.dtype == np.float32 and 0.0 <= x.min() and x.max() <= 1.0
        assert len(x) == len(y) == len(meta["weather"])
        ids[split] = set(meta["id"].tolist())
    # the test split must not leak into train/val
    assert ids["test"].isdisjoint(ids["train"])
    assert ids["test"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["val"])


def test_fog_is_minority_slice(tiny_dataset):
    _, _, meta = load_split(tiny_dataset, "train")
    counts = collections.Counter(meta["weather"])
    assert set(counts) <= set(WEATHER)
    # fog is the rarest weather by design
    assert counts["foggy"] == min(counts.values())


def test_brightness_encodes_timeofday(tiny_dataset):
    x, y, _ = load_split(tiny_dataset, "train")
    means = [x[y == c].mean() for c in range(3)]  # daytime, dawn, night
    assert means[0] > means[1] > means[2]
