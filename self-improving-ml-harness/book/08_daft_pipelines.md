# Chapter 8 · From numpy to Daft pipelines

> Companion notebook: [`notebooks/08_daft_pipelines.ipynb`](../notebooks/08_daft_pipelines.ipynb)
> Tier: `bdd-small` · **not executed in CI** (laptop-scale, but heavier than the tiny tier)

## Why graduate off numpy

The tiny tier loads the whole dataset into a numpy array — fine for a few
thousand 32×32 frames, impossible for `bdd-small` (let alone `bdd-full`). The
fix is to stop materializing everything and start expressing the pipeline as a
**lazy dataframe** over the Lance dataset. [Daft](https://www.getdaft.io/) gives
us exactly that, and because `data.py` already stores frames in Lance, the
pipeline reads them natively.

## The same query, now streaming

```python
import daft
df = daft.read_lance("data/bdd-small.lance")
train = (df.where(df["split"] == "train")
           .with_column("img", df["image"].image.decode())
           .with_column("img", df["img"].image.resize(32, 32)))
```

The key shift: nothing runs until you call `.collect()` or iterate. Daft builds a
plan, pushes the `split` filter down into Lance (so you never read test rows
during training), and decodes/resizes images in parallel across cores — or
across a Ray cluster, with the *same code*, in Chapter 9.

## UDFs for the cue features

The Chapter 5 embedding becomes a Daft UDF, computed lazily and column-at-a-time:

```python
@daft.udf(return_dtype=daft.DataType.fixed_size_list(daft.DataType.float32(), 4))
def colour_signature(images): ...
df = df.with_column("embed", colour_signature(df["img"]))
```

This is how mining scales: instead of embedding everything in RAM, you stream
embeddings into a Lance column and let LanceDB index them on disk.

## What changes, what doesn't

- **Unchanged**: the contract. `evaluator.py`, the score, `program.md`, the
  keep/revert loop. The agent still edits the same modifiable set.
- **Changed**: the *implementation* of data loading and feature computation moves
  from eager numpy to a lazy, parallel Daft plan, and the trainer consumes Daft
  batches instead of a numpy array.

## What you built

- A streaming, parallel data pipeline that reads Lance and feeds the trainer
  without materializing the dataset.
- Cue features as Daft UDFs, ready to scale to the cluster.
- The realization that scaling is an *implementation* concern — the contract is
  invariant across tiers.
