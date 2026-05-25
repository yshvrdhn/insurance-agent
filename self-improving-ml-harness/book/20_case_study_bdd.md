# Chapter 20 · Running it for real on BDD100K: a case study

> Companion notebook: [`notebooks/20_case_study_bdd.ipynb`](../notebooks/20_case_study_bdd.ipynb)
> Tier: `bdd-full` · **not executed in CI** (real download, GPU cluster, API key)

## Swapping the synthetic tier for the real thing

The whole book has run on a synthetic dataset that shares BDD100K's schema and
attributes. The payoff is that going real touches *only the ingest path*:

1. Register at [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu/) and
   download images + labels.
2. Implement the loader stubbed in `scripts/download_bdd.py --real`: read each
   frame and the `attributes` block of the label JSON (`weather`, `timeofday`,
   `scene`), and call `harness.data.write_lance(...)` with the *same schema*.
3. Everything else — evaluator, training surface, mining, loop, guardrails — is
   unchanged.

## The end-to-end run

```bash
python scripts/download_bdd.py --real --src /data/bdd100k --tier full
# point the loop at a Claude proposer (Ch. 14) on a Ray cluster (Ch. 9)
python -m harness.loop --iters 200 --dataset data/bdd-full.lance
```

A realistic narrative from `results.tsv`:

- **iters 0–5**: baseline; worst slice is night-rain detection. Agent reads the
  FiftyOne hardness report.
- **iters 6–40**: mining flywheel — embed, LanceDB-neighbour the hard night-rain
  crops, up-weight. Worst-group mAP climbs; rationales name the slice each time.
- **iter ~45**: over-mining shifts the bottleneck to small-pedestrians-in-fog;
  the loop reverts the last boost and the agent rebalances.
- **iters 46–120**: capacity + schedule tuning via Ray Tune within budget.
- **stop**: marginal score per GPU-hour falls below the `program.md` threshold.

## What "self-improving" actually delivered

Not a magic model — a **legible, bounded research process**: every gain traceable
to a kept change with a rationale, every regression reverted, the scorer provably
untouched, the spend capped. That is the real product of this book — not the
final number, but a trustworthy loop that produced it.

## Where to go next

- Port to detection/segmentation (Chapter 18) on your own fleet data.
- Replace the colour embedding with a foundation-model embedding for mining.
- Let the agent edit `model.py` architectures directly, with the Chapter 16
  review checklist as the safety net.

The contract is the reusable part. Point it at a new task, a new dataset, a new
worst slice — and turn the loop.
