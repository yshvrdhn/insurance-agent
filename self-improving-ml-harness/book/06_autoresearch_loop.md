# Chapter 6 · The keep/revert loop

> Companion notebook: [`notebooks/06_autoresearch_loop.ipynb`](../notebooks/06_autoresearch_loop.ipynb)

## The whole system, in one function

`harness/loop.py` is the AutoResearch loop:

```
read history → propose a change → train under budget → score with the
immutable evaluator → keep if it beats the best, else revert → log to results.tsv
```

Run it:

```bash
bash scripts/run_loop.sh          # or: python -m harness.loop --iters 8
```

A reference run reads like a lab notebook:

```
[iter 0] baseline score=0.883 worst=0.644(foggy)
[iter 1] KEEP   score=0.972 (best=0.972) :: upweight fog frames 5x
[iter 3] KEEP   score=1.018 (best=1.018) :: more capacity for the weak fog signal
[iter 4] revert score=0.951 (best=1.018) :: more epochs
[iter 5] KEEP   score=1.019 (best=1.019) :: weight decay to denoise fog frames
```

The best score only ever moves up, because a change is kept only when it beats
the incumbent. Reverted iterations aren't wasted — they're recorded as the dead
ends the agent explored, which is exactly the memory the *next* proposal reads.

## The pluggable proposer

In a real run an LLM edits `train.py` / `mining.py` / `configs` between
iterations (Part III). So the loop is runnable and testable on its own, the
default `SearchProposer` stands in for the agent: a greedy walk over a curated
space of moves (mining, capacity, schedule, regularization). Swap in your own —
including one backed by Claude — via the `proposer` argument:

```python
run_loop(iters=20, proposer=MyClaudeProposer())
```

The proposer sees the full `results.tsv` history and the current best config,
and returns `(override, rationale)`. That's the entire interface the agent
needs.

## The immutability guard

Before recording *any* score, the loop re-hashes every file in
`harness.IMMUTABLE_FILES` and compares against a snapshot taken at the start of
the iteration:

```python
snap = snapshot_immutable()
result = evaluate(...)          # train + score
verify_immutable(snap)          # raises if evaluator/data/loop changed
```

The notebook demonstrates the rejection: tamper with the recorded hash of
`evaluator.py` and `verify_immutable` raises. In a real agent run this is the
seatbelt — if the model ever "improves" by editing the scorer or the task, the
iteration is voided instead of celebrated.

## What you built

- A working autonomous loop that improves the worst-group score and never
  regresses below its own best.
- A clean seam (`proposer`) to drop an LLM into the decision role.
- A demonstrated, enforced immutability guarantee.

Next: how to read the trail it leaves behind.
