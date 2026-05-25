# Chapter 13 · Embeddings, hardness & active mining at scale

> Companion notebook: [`notebooks/13_active_mining.ipynb`](../notebooks/13_active_mining.ipynb)
> Tier: `bdd-small`+ · **not executed in CI** (LanceDB ANN index + FiftyOne)

## Mining as a first-class data flywheel

On the tiny tier, mining was a one-shot up-weight. At scale it becomes a
*flywheel*: every iteration's mistakes feed the next iteration's training
distribution. The pieces are now real infrastructure:

1. **Embed at scale.** Replace the 4-dim colour signature with a learned
   embedding — the penultimate layer of the current model — computed as a Daft
   UDF and written to a Lance `embedding` column.
2. **Index with LanceDB.** Build an approximate-nearest-neighbour (IVF/PQ) index
   so neighbour queries over millions of frames are milliseconds, not a scan.

   ```python
   tbl.create_index(num_partitions=256, num_sub_vectors=16)  # IVF-PQ
   neighbours = tbl.search(query_vec).limit(50).to_list()
   ```
3. **Rank by hardness.** Use Chapter 12's hardness/mistakenness to choose *which*
   frames to expand, not just the metadata tag.
4. **Up-weight & retrain.** Feed the mined neighbourhood to `mining.py` →
   `train.py` sample weights, exactly as on the tiny tier.

## Active learning, not just re-weighting

With real BDD100K you also have *unlabeled* frames. The same index powers active
learning: embed the unlabeled pool, retrieve the neighbours of your hard fog
frames, and send *those* for annotation. You spend the labeling budget where the
worst slice needs it most — the data-centric mirror of the compute budget.

## Guarding against feedback collapse

A flywheel can spin out: relentlessly mining fog can make the training set so
fog-heavy that another slice collapses (the Chapter 5 trap, at scale). Two
safeguards:

- the **worst-group objective** itself punishes over-correction — if rainy
  collapses, the score drops and the loop reverts;
- cap the mined fraction per iteration so the distribution drifts gradually.

## What you built

- A learned-embedding mining pipeline indexed with LanceDB ANN.
- Hardness-ranked, not tag-based, selection of frames to expand.
- An active-learning loop that targets the annotation budget at the worst slice,
  with safeguards against feedback collapse.
