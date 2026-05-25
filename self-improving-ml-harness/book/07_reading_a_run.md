# Chapter 7 · Reading a run

> Companion notebook: [`notebooks/07_reading_a_run.ipynb`](../notebooks/07_reading_a_run.ipynb)

## `results.tsv` is the memory

The loop appends one row per attempt to `results.tsv`:

| column | meaning |
|--------|---------|
| `iter` | iteration index (0 = baseline) |
| `score` | this attempt's official score |
| `best_so_far` | best score through this iteration |
| `kept` | was the change kept? |
| `worst_group`, `worst_group_name` | the slice that capped the score |
| `overall_accuracy`, `macro_accuracy` | diagnostics |
| `seconds` | training time (the budget spent) |
| `rationale` | one line: *why* this change was tried |

It's a plain TSV on purpose: greppable, diffable, and loadable into pandas or
Daft in one line. It is both the agent's working memory (step 1 of every
iteration is "read the history") and your audit trail afterwards.

## Reading the story

The notebook loads the TSV and reconstructs the narrative: which changes were
kept, how `best_so_far` climbed from baseline, and — crucially — how the
`worst_group` accuracy tracked toward the overall accuracy as the loop closed
the fog gap. When the two lines converge, the model is no longer lopsided; the
worst slice is no longer dramatically worse than the average.

A healthy run shows:

- a monotone `best_so_far` (the keep/revert invariant),
- a `worst_group` line trending up faster than `overall_accuracy` (you're fixing
  the blind spot, not just padding the average),
- and usually a moment where `worst_group_name` *changes* — the bottleneck moved
  from fog to another slice — which is your cue that the easy fog wins are spent.

## Using it to drive the next decision

This is the feedback the agent (or you) acts on:

- Worst slice still fog, with headroom? → mine harder, add capacity.
- Worst slice moved to rainy after a big fog boost? → you over-corrected; dial
  the boost back (this is the Chapter 5 trap, now visible in the data).
- `best_so_far` flat for several iters? → the curated moves are exhausted; it's
  time for a genuinely new hypothesis (Part III's LLM proposer is good at this).

## Part I, complete

You now have the whole contract working end to end on the tiny tier:

- a **fixed task** in Lance (`data.py`),
- an **immutable, tamper-evident scorer** (`evaluator.py`),
- a **budgeted training surface** (`train.py`, `model.py`),
- **LanceDB mining** to target the worst slice (`mining.py`),
- a **keep/revert loop** with an immutability guard (`loop.py`),
- and a readable **audit trail** (`results.tsv`).

Part II keeps this exact contract and scales the implementation onto real
distributed infrastructure — Daft pipelines, Ray Train, Ray Tune, and FiftyOne —
on the larger `bdd-small` tier.
