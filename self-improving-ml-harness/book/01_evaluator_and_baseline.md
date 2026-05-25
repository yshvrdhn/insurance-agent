# Chapter 1 · The immutable evaluator & your first baseline

> Companion notebook: [`notebooks/01_fiftyone_evaluator.ipynb`](../notebooks/01_fiftyone_evaluator.ipynb)

## The one rule

Every self-improving system needs a number it is trying to move and a promise
that the number can't be gamed. In this book that number comes from
`harness/evaluator.py`, and the promise is simple: **the agent may never edit
it.** The evaluator is the ruler. If you let the thing being measured also hold
the ruler, you don't have a benchmark, you have wishful thinking.

So before we train anything, we fix the score:

```
score = worst_group_accuracy + 0.25 · macro_accuracy
```

The *groups* are weather slices — `clear`, `overcast`, `rainy`, `snowy`,
`foggy`. We optimize the **worst** slice deliberately. A driving-perception
model that is excellent at noon and blind in fog is not a good model; it is a
dangerous one with a good average. The `0.25 · macro_accuracy` term is a gentle
tie-breaker so the loop doesn't trash overall quality to nudge one slice.

## The task

We classify the **time of day** of a driving frame — `daytime`, `dawn`, or
`night`. The data mirrors [BDD100K](https://bdd-data.berkeley.edu/): each frame
carries `weather`, `timeofday`, and `scene` attributes. To keep every chapter
runnable on a laptop with no GPU and no multi-hundred-megabyte download, the
tiny tier is *synthetic* — a deterministic generator paints small driving frames
whose pixels actually encode the labels (Chapter 2 opens the generator up). The
schema, the attributes, and the contract are identical to the real thing;
Chapter 20 swaps in real BDD100K without changing a line of the harness.

## Train a baseline, then meet the worst slice

The notebook trains a small CNN under a fixed budget and scores it. The shape of
the result is always the same:

```
OFFICIAL SCORE : 0.88
worst slice    : foggy = 0.64
overall acc    : 0.93
```

The baseline is strong overall but **weakest on fog**. That is not an accident —
fog is rare in the data and its visual cues are washed out (Chapter 2 shows
exactly how). The entire rest of the book is about raising that 0.64 *without
touching the evaluator that reported it.*

## Slice-based thinking

Reporting a single accuracy number hides the blind spot that matters. The
evaluator instead returns a full per-slice breakdown — per weather, per scene,
per class — plus a confusion matrix and a per-sample correctness vector. This is
the same philosophy as [FiftyOne](https://docs.voxel51.com/)'s slice-based
evaluation, which we wire in for real in Chapter 12. The scalar `score` is what
the loop optimizes; the slices are how a human (or the agent) *understands* a
result and forms the next hypothesis.

## What you built

- A fixed task and a fixed, tamper-evident scorer.
- A baseline whose weakness (the foggy slice) is now quantified.
- The mental model for everything that follows: **propose → train → score →
  keep or revert**, judged by a ruler you are not allowed to move.

Next: open up the dataset and see *why* fog is hard.
