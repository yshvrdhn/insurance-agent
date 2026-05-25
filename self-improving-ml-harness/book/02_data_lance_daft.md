# Chapter 2 · The dataset: BDD-style frames in Lance & Daft

> Companion notebook: [`notebooks/02_data_lance_daft.ipynb`](../notebooks/02_data_lance_daft.ipynb)

## Why Lance, why Daft

Two infrastructure choices shape everything downstream:

- **[Lance](https://lancedb.github.io/lance/)** is the storage format. It's
  columnar, zero-copy, versioned, and designed for ML payloads — images,
  embeddings, tensors — not just scalars. One row per frame:

  ```
  id:int  image:fixed_size_list<uint8>[H·W·C]  label:int
  timeofday:str  weather:str  scene:str  split:str
  ```

- **[Daft](https://www.getdaft.io/)** is the dataframe. It's lazy and
  distributed, so the exact query you write against the tiny tier runs unchanged
  against `bdd-full` on a cluster (Chapter 8). It reads Lance natively.

`harness/data.py` is in the **immutable set**. The dataset *is* the task —
quietly changing it would change what we're scoring — so it's protected by the
same hash guard as the evaluator.

## The split contract

`load_split` returns `(X, y, meta)` for `train`, `val`, or `test`. Splits are
assigned once from a fixed seed, so the **test set is byte-for-byte identical
across every run** — a precondition for an honest score. Training code is only
ever allowed to read `train` and `val`; the `test` split belongs to the
evaluator alone.

## Why fog is the hard slice

The notebook visualizes the cue structure, and it's worth internalizing because
it explains every result in the book:

1. **Brightness is the easy, dominant cue.** Daytime frames are bright, night
   frames dark, dawn in between. A small CNN latches onto mean brightness first,
   and on most weather that's enough to score nearly perfectly.

2. **Fog scrambles brightness.** Foggy frames get a large *random multiplicative*
   haze — the whole frame is scaled up or down by an unpredictable factor. You
   can't average that away: the per-image mean brightness becomes uninformative.

3. **A weak colour tint survives.** Dawn is subtly warm, night subtly cool, as a
   *multiplicative* tint — so the channel *ratios* are invariant to the haze. But
   the effect is small (~9%) and, in fog, washed toward grey and corrupted by
   colour noise. It's a real signal, just a faint one.

4. **Fog is rare.** It's the minority weather class, so a baseline sees few fog
   frames and has little incentive to learn the faint cue.

Put together: the baseline reads brightness, wins on common weather, and
underfits the rare, brightness-corrupted fog slice. Recovering fog means pushing
the model to invest in the weak colour cue — via more capacity, more epochs,
regularization, and above all **mining and up-weighting fog frames** (Chapter 5).

## What you built

- A feel for the Lance schema and how to query attributes with Daft.
- The class- and weather-balance of the data, including fog's scarcity.
- A precise, mechanistic understanding of *why* the worst slice is worst — which
  is what lets you form good hypotheses instead of flailing at hyperparameters.

Next: open the evaluator itself and the guard that keeps it honest.
