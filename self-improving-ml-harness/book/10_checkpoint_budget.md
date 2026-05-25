# Chapter 10 · Checkpointing, fault tolerance & the budget at scale

> Companion notebook: [`notebooks/10_checkpoint_budget.ipynb`](../notebooks/10_checkpoint_budget.ipynb)
> Tier: `bdd-small`+ · **runs here on the tiny tier** (Ray Train checkpointing, local mode)

## Why a self-improving system must checkpoint

An autonomous loop runs unattended for hours or days. Spot instances get
reclaimed, nodes die, a budget timer fires mid-epoch. The loop's promise — "score
whatever checkpoint exists at the budget" (`program.md`) — only holds if there
*is* a checkpoint. So checkpointing isn't a nicety here; it's what makes the
budget rule enforceable.

## Ray Train checkpoints

```python
from ray.train import Checkpoint, report
# inside the train loop, each epoch:
report({"val_acc": val_acc}, checkpoint=Checkpoint.from_directory(ckpt_dir))
```

Ray persists checkpoints to durable storage (S3/GCS/NFS) and, on worker failure,
restarts from the latest one. The loop then loads the best checkpoint to hand to
the evaluator — the same `TrainedModel.save()` / load path you saw on the tiny
tier, just backed by cluster storage.

## The budget as a hard deadline

At scale the budget is a wall-clock deadline, not an epoch count. The trainer
checks the clock each step and, when the deadline passes, stops and reports the
best checkpoint so far. This makes two failure modes safe:

- a change that converges slowly → scored at its budget-truncated checkpoint
  (and will lose to a faster one, correctly),
- a change that diverges → the best-val checkpoint (often an early epoch) is what
  gets scored, not the blown-up final weights.

## Idempotent iterations

Because every iteration writes to its own run directory and the *decision* is
derived purely from `results.tsv` + the immutable evaluator, a crashed iteration
can be retried without corrupting history. The loop appends; it never mutates a
prior row. Reproducibility (fixed seeds, fixed test split) means a retry yields
the same score.

## What you built

- Durable checkpointing wired into the Ray Train step.
- A budget that behaves as a hard deadline with safe truncation semantics.
- Idempotent, crash-safe iterations — a prerequisite for unattended runs.
