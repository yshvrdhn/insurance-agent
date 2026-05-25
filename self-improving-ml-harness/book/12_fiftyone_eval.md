# Chapter 12 · Slice-based evaluation & mistakenness with FiftyOne

> Companion notebook: [`notebooks/12_fiftyone_eval.ipynb`](../notebooks/12_fiftyone_eval.ipynb)
> Tier: `bdd-small`+ · **runs here on the tiny tier** (FiftyOne headless; numpy fallback if absent)

## From numbers to looking at your data

The tiny-tier evaluator already returns per-slice accuracy and a confusion
matrix. [FiftyOne](https://docs.voxel51.com/) turns those numbers into something
you can *look at*: load the dataset, attach predictions, and explore the worst
slice frame-by-frame in a browser.

```python
import fiftyone as fo
ds = fo.Dataset.from_dir(...)            # frames + weather/timeofday tags
ds.set_values("pred", model_predictions)
results = ds.evaluate_classifications("pred", gt_field="timeofday", method="simple")
results.print_report()                   # per-class P/R/F1
session = fo.launch_app(ds.match_tags("foggy"))   # stare at the fog failures
```

## Mistakenness and hardness

FiftyOne computes two quantities that supercharge mining:

- **Mistakenness** — how likely a *label* is wrong, from the model's confidence
  vs. the annotation. On real BDD100K the `weather` tags are noisy; mistakenness
  surfaces mislabeled frames before they poison the fog slice.
- **Hardness** — how difficult a *sample* is for the model (low margin, high
  loss). The hardest fog frames are precisely the ones worth mining and
  up-weighting.

```python
import fiftyone.brain as fob
fob.compute_mistakenness(ds, "pred", label_field="timeofday")
fob.compute_hardness(ds, "pred")
hard_fog = ds.match_tags("foggy").sort_by("hardness", reverse=True)
```

## Closing the loop with FiftyOne

This is the scaled version of Chapter 5: instead of a hand-rolled colour
embedding, you mine with FiftyOne's hardness/mistakenness, export the hard frame
IDs, and feed them to `mining.py` as the up-weight set. The evaluator's scalar
score is unchanged — FiftyOne is an *analysis and mining* layer, not a second
scorer.

## What you built

- An interactive, visual read on the worst slice and its failure modes.
- Mistakenness to catch noisy labels and hardness to rank mining candidates.
- A bridge from FiftyOne's analysis back into the harness's mining surface.
