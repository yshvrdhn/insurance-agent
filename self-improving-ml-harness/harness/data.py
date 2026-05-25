"""Dataset definition — **immutable**.

This module defines *the task*. Changing it would change what we are scoring,
so it lives in the immutable set (see ``program.md``).

We mirror BDD100K's structure (driving scenes tagged with ``weather``,
``timeofday`` and ``scene``) without the multi-hundred-megabyte download: a
deterministic generator paints small synthetic driving frames whose pixels
actually encode the labels. That keeps every tiny-tier notebook runnable on a
laptop and in CI with no GPU.

The classification target is ``timeofday`` (daytime / dawn / night). The
``weather`` attribute defines the evaluation *slices*. ``foggy`` is engineered to
be the hard slice: it is **rare** (see ``_WEATHER_P``) and its cues are degraded
— a large random brightness haze plus a washed-out, noisy colour tint — so its
class signal is weak and a budget baseline underfits it. Recovering that slice
(more capacity/epochs, regularization, and especially mining + up-weighting fog
frames) is what the harness optimizes. ``scene`` is an extra attribute available
for slicing and analysis.

Images are stored in a **Lance** dataset (columnar, zero-copy, versioned) with
one row per frame:

    id:int  image:fixed_size_list<uint8>[H*W*C]  label:int
    timeofday:str  weather:str  scene:str  split:str

Use :func:`load_split` to pull numpy arrays for training/eval, or
:func:`to_daft` for a distributed dataframe view.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa

IMG_H = 32
IMG_W = 32
IMG_C = 3
N_PIX = IMG_H * IMG_W * IMG_C

# label space (index == integer label)
TIMEOFDAY = ("daytime", "dawn", "night")
WEATHER = ("clear", "overcast", "rainy", "snowy", "foggy")
SCENE = ("highway", "city", "residential")

# Weather is deliberately imbalanced: fog is the rare, hard slice. A baseline
# trained on this distribution sees few fog frames and underfits them; the
# evaluator (worst-group score) punishes that, so the harness has to actively
# spend capacity / mined fog frames on the slice it would otherwise ignore.
_WEATHER_P = (0.30, 0.23, 0.20, 0.17, 0.10)  # clear, overcast, rainy, snowy, fog

# base luminance per time-of-day, in [0, 1]. Mean brightness is the *easy*,
# dominant cue with clean per-class gaps -- SGD on a small CNN latches onto it
# first and, on most weather, that is enough to score perfectly.
_BRIGHT = {"daytime": 0.62, "dawn": 0.44, "night": 0.26}

# A *weak* color tint per time-of-day (R, G, B multipliers). Because it is
# multiplicative, the channel ratios (dawn slightly warm, night slightly cool)
# are invariant to global brightness scaling -- so they survive fog haze. But
# the effect is deliberately small (~8%), so a budget-limited baseline that is
# already winning via brightness never bothers to learn it. Pushing training off
# the brightness cue (brightness-jitter augmentation, or mining fog frames) is
# what forces the model to invest in this robust signal.
_TINT = {
    "daytime": (1.00, 1.00, 1.00),
    "dawn": (1.09, 1.00, 0.92),
    "night": (0.92, 0.99, 1.09),
}

# per-weather effect:
#   (lum_offset, contrast_mult, texture_noise_std, haze_log_std,
#    tint_wash, color_noise_std)
# ``haze`` is a random per-image *multiplicative* brightness factor
# ``exp(N(0, haze))``; ``tint_wash`` scales how much of the time-of-day colour
# tint survives (1.0 = full, 0.0 = grey); ``color_noise`` is per-image per-channel
# multiplicative jitter that corrupts the colour ratio.
#
# On clear/overcast/etc, brightness is clean -> the slice is easy. Foggy scrambles
# brightness (large haze) AND washes most of the colour out AND adds colour noise,
# leaving only a weak, noisy residual signal. Combined with fog being a rare slice
# (``_WEATHER_P``), a budget baseline underfits fog badly. Extracting that weak
# signal -- more capacity/epochs, regularization, and especially mining + up-
# weighting fog frames -- is how the worst-group score climbs.
_WEATHER_FX = {
    "clear": (0.00, 1.00, 0.04, 0.05, 1.00, 0.02),
    "overcast": (-0.04, 0.90, 0.05, 0.07, 0.95, 0.03),
    "rainy": (-0.06, 0.82, 0.07, 0.09, 0.90, 0.04),
    "snowy": (0.05, 0.86, 0.07, 0.09, 0.90, 0.04),
    "foggy": (0.02, 0.60, 0.06, 0.55, 0.45, 0.10),
}

# scene -> (luminance offset, number of foreground "objects")
_SCENE_FX = {"highway": (0.02, 2), "city": (-0.03, 6), "residential": (0.00, 4)}


@dataclass(frozen=True)
class DatasetSpec:
    n: int = 1200
    seed: int = 7
    splits: tuple[float, float, float] = (0.7, 0.15, 0.15)  # train/val/test


def _render_one(rng: np.random.Generator, timeofday: str, weather: str,
                scene: str) -> np.ndarray:
    """Render a single HxWxC float image in [0, 1] from its attributes."""
    base = _BRIGHT[timeofday]
    w_off, contrast, noise_std, haze_log, tint_wash, color_noise = _WEATHER_FX[weather]
    s_off, n_obj = _SCENE_FX[scene]
    mean = float(np.clip(base + w_off + s_off, 0.06, 0.94))

    # vertical gradient: sky (top) brighter than road (bottom)
    rows = np.linspace(0.16, -0.16, IMG_H, dtype=np.float32)[:, None]
    img = mean + np.repeat(rows, IMG_W, axis=1)

    # scene-dependent foreground blocks (buildings / vehicles)
    for _ in range(n_obj):
        h = int(rng.integers(4, 12))
        w = int(rng.integers(4, 10))
        r0 = int(rng.integers(IMG_H - h))
        c0 = int(rng.integers(IMG_W - w))
        shade = float(rng.uniform(-0.25, 0.25))
        img[r0:r0 + h, c0:c0 + w] += shade

    # contrast around the local mean, then weather texture noise
    img = mean + (img - mean) * contrast
    img = img + rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)

    # broadcast to 3 channels and apply the time-of-day colour tint, washed out
    # by weather (fog -> mostly grey) and corrupted by per-channel colour noise
    tint = 1.0 + (np.asarray(_TINT[timeofday], dtype=np.float32) - 1.0) * tint_wash
    if color_noise > 0:
        tint = tint * np.exp(rng.normal(0.0, color_noise, size=IMG_C)).astype(np.float32)
    img3 = np.repeat(img[:, :, None], IMG_C, axis=2) * tint[None, None, :]

    # multiplicative haze: scrambles global brightness, preserves colour ratios
    if haze_log > 0:
        img3 = img3 * float(np.exp(rng.normal(0.0, haze_log)))

    return np.clip(img3, 0.0, 1.0)


def _assign_split(rng: np.random.Generator, n: int,
                  fracs: tuple[float, float, float]) -> np.ndarray:
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    splits = np.array(["train"] * n_tr + ["val"] * n_va +
                      ["test"] * (n - n_tr - n_va))
    rng.shuffle(splits)
    return splits


def build_synthetic(spec: DatasetSpec | None = None) -> dict:
    """Generate the synthetic dataset in memory.

    Returns a dict of equal-length arrays/lists keyed by column name. Splits
    are stratified-ish by random assignment with a fixed seed, so the test set
    is identical across runs — a precondition for an honest score.
    """
    spec = spec or DatasetSpec()
    rng = np.random.default_rng(spec.seed)

    timeofday = rng.choice(TIMEOFDAY, size=spec.n)
    weather = rng.choice(WEATHER, size=spec.n, p=_WEATHER_P)
    scene = rng.choice(SCENE, size=spec.n)
    splits = _assign_split(rng, spec.n, spec.splits)

    images = np.empty((spec.n, N_PIX), dtype=np.uint8)
    for i in range(spec.n):
        frame = _render_one(rng, timeofday[i], weather[i], scene[i])
        images[i] = (frame.reshape(-1) * 255.0).astype(np.uint8)

    labels = np.array([TIMEOFDAY.index(t) for t in timeofday], dtype=np.int16)
    return {
        "id": np.arange(spec.n, dtype=np.int64),
        "image": images,
        "label": labels,
        "timeofday": timeofday.astype(object),
        "weather": weather.astype(object),
        "scene": scene.astype(object),
        "split": splits.astype(object),
    }


def write_lance(data: dict, path: str | Path) -> Path:
    """Persist the in-memory dataset to a Lance dataset directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_type = pa.list_(pa.uint8(), N_PIX)  # fixed-size list per row
    table = pa.table(
        {
            "id": pa.array(data["id"]),
            "image": pa.array(list(data["image"]), type=image_type),
            "label": pa.array(data["label"].astype(np.int64)),
            "timeofday": pa.array(data["timeofday"], type=pa.string()),
            "weather": pa.array(data["weather"], type=pa.string()),
            "scene": pa.array(data["scene"], type=pa.string()),
            "split": pa.array(data["split"], type=pa.string()),
        }
    )
    lance.write_dataset(table, str(path), mode="overwrite")
    return path


def make_synthetic_dataset(path: str | Path, spec: DatasetSpec | None = None) -> Path:
    """Build and persist the synthetic dataset in one call."""
    return write_lance(build_synthetic(spec), path)


def _to_table(path: str | Path) -> pa.Table:
    return lance.dataset(str(path)).to_table()


def load_split(path: str | Path, split: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load one split as ``(X, y, meta)``.

    ``X`` is ``float32`` in [0, 1] with shape ``(N, H, W, C)``; ``y`` is int64
    of shape ``(N,)``; ``meta`` holds the ``weather``/``scene`` arrays for
    slice-based evaluation.
    """
    tbl = _to_table(path)
    df = tbl.to_pandas()
    df = df[df["split"] == split].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"split {split!r} is empty in {path}")
    x = np.stack(df["image"].to_numpy()).astype(np.float32) / 255.0
    x = x.reshape(-1, IMG_H, IMG_W, IMG_C)
    y = df["label"].to_numpy().astype(np.int64)
    meta = {
        "id": df["id"].to_numpy(),
        "weather": df["weather"].to_numpy(),
        "scene": df["scene"].to_numpy(),
        "timeofday": df["timeofday"].to_numpy(),
    }
    return x, y, meta


def to_daft(path: str | Path):
    """Return a Daft dataframe view of the dataset (lazy, distributed-ready)."""
    import daft

    return daft.read_lance(str(path))


def class_names() -> tuple[str, ...]:
    return TIMEOFDAY
