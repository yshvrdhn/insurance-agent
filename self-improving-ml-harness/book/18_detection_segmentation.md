# Chapter 18 · Beyond classification: detection & segmentation

> Companion notebook: [`notebooks/18_detection_segmentation.ipynb`](../notebooks/18_detection_segmentation.ipynb)
> Tier: `bdd-full` · **not executed in CI** (real BDD labels, GPU)

## The contract generalizes; the score and slices change

Everything in Parts I–II is task-agnostic: a fixed dataset, an immutable scorer,
a budgeted training surface, mining, and a keep/revert loop. To move from
time-of-day classification to BDD100K's real tasks — **2D detection** (cars,
pedestrians, signs), **drivable-area** and **lane** segmentation — you change two
things and *only* two:

1. the **evaluator's metric**, and
2. the **slice definitions**.

The loop, the budget, the mining flywheel, and the guardrails are untouched.

## Detection: worst-group mAP

```
score = worst_group_mAP + 0.25 · mAP_overall
```

Now the groups can be richer: weather × object-size (small/medium/large), or
per-class (pedestrians are the safety-critical worst slice). The mining
embedding becomes a crop embedding; LanceDB neighbours of a missed small-pedestrian-
in-rain are exactly the hard crops to up-weight. FiftyOne's detection
evaluation (per-class P/R, IoU sweeps, hardest images) plugs straight into the
Chapter 12/13 analysis layer.

## Segmentation: worst-group IoU

```
score = worst_group_mIoU + 0.25 · mIoU_overall
```

Drivable-area segmentation has a brutal worst slice — night + rain + reflective
road — that average mIoU hides. The worst-group objective forces the harness to
care about it, which is exactly the point: a lane model that fails on wet night
roads is the one that matters.

## What stays hard

The synthetic tiny tier can't stand in for real detection/segmentation — these
chapters require the real BDD100K download and a GPU. But the *shape* of the work
is identical to the tiny tier you already ran: form a worst-slice hypothesis,
mine the hard examples, train under budget, let the immutable evaluator decide.

## What you built

- A recipe for porting the harness to detection and segmentation by swapping only
  the metric and the slice definitions.
- Worst-group mAP / mIoU objectives that target the safety-critical slices.
- Confirmation that the contract — not the task — is the reusable core.
