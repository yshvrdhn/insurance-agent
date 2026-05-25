# Chapter 5 · Hard-example mining with LanceDB

> Companion notebook: [`notebooks/05_mining_lancedb.ipynb`](../notebooks/05_mining_lancedb.ipynb)

## The biggest lever

The worst slice is rare, faint-cued fog, and Chapter 4 showed generic knobs
won't fix it. The fix is to change *what the model trains on*: find the fog-like
frames, and make the trainer pay attention to them. That's hard-example mining,
and `harness/mining.py` (agent-modifiable) implements it with **LanceDB** vector
search.

## A cue-aware embedding

You can't mine by metadata alone — in the real BDD100K plenty of "clear" frames
are visually fog-like at dusk, and the `weather` tag is noisy. So we embed every
frame into the cue that actually matters:

```python
embed(x) = [R̄/Σ, Ḡ/Σ, B̄/Σ, contrast]   # brightness-invariant colour signature
```

Normalizing the channel means to sum 1 cancels global brightness — exactly the
thing fog corrupts — leaving the colour ratio that survives. Two frames that are
near neighbours in this space *look like the same time of day*, regardless of how
bright the haze made them.

## Vector search to expand the hard set

```python
mining:
  enabled: true
  strategy: fog_knn     # fog_boost | fog_knn | none
  boost: 6.0            # weight on mined frames
  k: 8                  # LanceDB neighbours per fog frame
```

- `fog_boost` simply up-weights the (few) frames tagged foggy.
- `fog_knn` indexes the embeddings in LanceDB and, for each fog frame, retrieves
  its `k` nearest neighbours — frames the model is most likely to confuse with
  fog — and up-weights that whole neighbourhood. It's a far more surgical fix
  than boosting the literal fog tag, because it captures *fog-like* frames the
  tag missed.

The up-weighting flows through to `train.py` as per-sample weights, which bias
the weighted sampler toward the mined frames.

## Before vs. after

The notebook trains with and without mining and plots the per-weather bars side
by side. The fog slice jumps (in our reference run, ≈0.64 → ≈0.82) while the
other slices hold. The official score climbs because its dominant term — the
*worst* group — is exactly the one mining targets.

## The trap: don't create a new worst slice

Over-boosting fog is a real failure mode. Push `boost` too high and the model
over-fits fog at the expense of, say, `rainy` — and now *rainy* is the worst
group and the score drops. This isn't a bug; it's the objective working as
intended. It's also the perfect motivation for the loop: you want a process that
tries a boost, measures the *worst* slice, and reverts if it shifted the
bottleneck. That process is Chapter 6.

## What you built

- A brightness-invariant embedding aimed at the surviving cue.
- LanceDB vector search to expand a hard slice beyond its noisy metadata tag.
- A measured, sometimes-double-edged improvement to the worst-group score — and
  a concrete reason the keep/revert loop exists.

Next: close the loop.
