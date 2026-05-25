# Chapter 4 · The training surface

> Companion notebook: [`notebooks/04_training_surface.ipynb`](../notebooks/04_training_surface.ipynb)

## What the agent is allowed to change

`harness/train.py` and `harness/model.py` are **agent-modifiable**. Together they
own everything about *how* the model learns:

- **architecture** (`model.py`): a small configurable CNN — `width`, `depth`,
  `dropout`. The only contract is `build_model(config, num_classes) -> nn.Module`
  mapping `(N, C, H, W)` floats in [0, 1] to logits.
- **optimization** (`train.py`): learning rate, weight decay, batch size, label
  smoothing, cosine schedule.
- **augmentation**: `hflip`, `brightness_jitter`, `contrast_jitter`,
  `per_image_norm`.
- **sample weights**: the hook that lets `mining.py` (Chapter 5) up-weight hard
  frames.

The one stable contract with the immutable evaluator: `train(...)` returns an
object exposing `predict(X) -> np.ndarray[int]`. Everything inside is fair game.

## The budget makes changes comparable

Every training run is capped by `configs/base.yaml: budget` — the first of
`max_epochs` or `max_seconds` to hit. This is what makes the loop honest about
*cost*. A change that only helps because it secretly trains 10× longer isn't an
improvement to the method; it's just more compute. By fixing the budget, the
score measures **quality per unit compute**. The notebook shows training
stopping exactly at `max_epochs`, and shows a wider model eating more of the
wall-clock budget for a given epoch count.

## Capacity and epochs help — but not the slice that matters

The notebook sweeps `width ∈ {8, 16, 32}` and `max_epochs ∈ {2, 6, 12}`. Bigger
and longer help the *overall* number a little. But watch the `foggy` column: it
barely moves. This is the central lesson of Part I — **generic knobs don't fix a
rare, degraded slice.** You can throw capacity at the problem and the worst-group
score, which is what we actually optimize, stays stubborn.

Augmentation tells the same story. `hflip` is free and label-preserving but
slice-neutral. `brightness_jitter` nudges robustness but can't conjure signal
that fog has washed out. The fog slice needs a *targeted* intervention.

## Why this matters for the loop

If every knob moved every slice, the loop would be boring — turn them all up and
go home. The interesting dynamics come from the fact that different changes have
different *incidence*. The harness has to discover that the fog problem is a
*data* problem (too few, too-faint fog examples), not a capacity problem — and
the tool for that is mining.

## What you built

- A tour of the modifiable surface and the `train` ↔ `evaluator` contract.
- Concrete evidence that capacity/epochs/augmentation improve the average while
  leaving the worst slice nearly fixed.
- The motivation for Chapter 5: to move fog, target fog.

Next: hard-example mining with LanceDB.
