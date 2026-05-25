"""Hard-example mining — **agent-modifiable**.

The worst-group score is dragged down by the rare, washed-out *foggy* slice. The
model underfits it simply because it sees few fog frames. This module turns that
around: it embeds every training frame, indexes the embeddings in **LanceDB**,
and uses vector search to expand the set of fog-like frames, then hands the
trainer per-sample weights that upweight them.

The embedding is a tiny brightness-invariant colour signature -- exactly the cue
that survives fog -- so nearest neighbours of a fog frame are frames the model is
most likely to confuse with it. Upweighting that neighbourhood is a far more
surgical fix than blindly boosting every fog frame.

The agent may redesign the embedding, the neighbourhood size, or the weighting
scheme; the only contract is :func:`compute_sample_weights` returning a positive
weight per training row, in train-split order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from harness.data import load_split


def embed(x: np.ndarray) -> np.ndarray:
    """Brightness-invariant colour signature for each image ``(N, H, W, C)``.

    Per-channel means normalised to sum 1 (so global brightness cancels) plus the
    image's spatial contrast. This is the slice's surviving signal, which makes
    vector neighbours semantically "looks like the same time-of-day".
    """
    chan_mean = x.mean(axis=(1, 2))                      # (N, C)
    ratio = chan_mean / (chan_mean.sum(axis=1, keepdims=True) + 1e-6)
    contrast = x.std(axis=(1, 2, 3), keepdims=False)[:, None]
    feats = np.concatenate([ratio, contrast], axis=1).astype(np.float32)
    return feats


def _lancedb_neighbors(feats: np.ndarray, query_idx: np.ndarray,
                       k: int) -> set[int]:
    """Return the union of the k nearest neighbours of each queried row.

    Uses LanceDB vector search over an in-memory table. Falls back to a numpy
    brute-force search if LanceDB is unavailable, so the harness still runs.
    """
    try:
        import lancedb
        import pyarrow as pa

        dim = feats.shape[1]
        db = lancedb.connect("/tmp/.mlh_mining_lancedb")
        try:
            db.drop_table("frames")
        except Exception:
            pass
        tbl = db.create_table(
            "frames",
            data=[{"row": int(i), "vector": feats[i].tolist()}
                  for i in range(len(feats))],
        )
        found: set[int] = set()
        for qi in query_idx:
            res = (tbl.search(feats[qi].tolist())
                   .limit(k + 1)
                   .to_list())
            found.update(int(r["row"]) for r in res)
        return found
    except Exception:
        # numpy fallback: cosine-ish L2 nearest neighbours
        found = set()
        for qi in query_idx:
            d = ((feats - feats[qi]) ** 2).sum(axis=1)
            found.update(np.argsort(d)[:k + 1].tolist())
        return found


def compute_sample_weights(dataset_path: str | Path, config: dict) -> np.ndarray:
    """Per-train-row weights from the mining config block.

    Config (``configs/*.yaml`` -> ``mining:``)::

        mining:
          enabled: true
          strategy: fog_knn      # fog_knn | fog_boost | none
          boost: 6.0             # weight on mined fog/neighbour frames
          k: 8                   # neighbours per fog frame (fog_knn only)
    """
    x, y, meta = load_split(dataset_path, "train")
    n = len(x)
    weights = np.ones(n, dtype=np.float64)

    mining = config.get("mining", {})
    if not mining.get("enabled"):
        return weights

    strategy = mining.get("strategy", "fog_boost")
    boost = float(mining.get("boost", 5.0))
    fog_mask = meta["weather"] == "foggy"

    if strategy == "fog_boost":
        weights[fog_mask] = boost
    elif strategy == "fog_knn":
        feats = embed(x)
        fog_idx = np.where(fog_mask)[0]
        if len(fog_idx) > 0:
            neigh = _lancedb_neighbors(feats, fog_idx, int(mining.get("k", 8)))
            idx = np.fromiter(neigh, dtype=int)
            weights[idx] = boost
            weights[fog_mask] = boost  # always include the fog frames themselves
    elif strategy == "none":
        pass
    else:
        raise ValueError(f"unknown mining strategy: {strategy!r}")
    return weights
