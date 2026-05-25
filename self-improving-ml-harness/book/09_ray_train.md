# Chapter 9 · Distributed training with Ray Train

> Companion notebook: [`notebooks/09_ray_train.ipynb`](../notebooks/09_ray_train.ipynb)
> Tier: `bdd-small`+ · **not executed in CI** (needs a Ray runtime / multiple workers)

## The training surface, distributed

`harness/train.py`'s single-process loop is the right thing on the tiny tier and
the wrong thing on `bdd-full`. [Ray Train](https://docs.ray.io/en/latest/train/train.html)
wraps the *same* training function and runs it across N workers with data-parallel
gradients, without rewriting the model or the loss.

```python
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

def train_func(cfg):
    # the body is harness/train.py's per-worker step, with:
    #   model = ray.train.torch.prepare_model(build_model(cfg, n))
    #   loader = ray.train.torch.prepare_data_loader(loader)
    ...

trainer = TorchTrainer(
    train_func, train_loop_config=cfg,
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
)
result = trainer.fit()
```

`prepare_model` / `prepare_data_loader` handle DistributedDataParallel and the
distributed sampler; Ray handles process launch, rendezvous, and gradient
all-reduce.

## The budget becomes a cluster budget

On the tiny tier the budget is `max_epochs` / `max_seconds` on one core. At scale
it generalizes to wall-clock across workers, and the *cost* of a proposed change
now includes how well it parallelizes. A change that improves the score but
halves throughput may not be worth it under a fixed cluster-hour budget — a
trade-off the loop can be taught to weigh (Chapter 19).

## Keeping the score honest at scale

Two things stay sacred:

- **Evaluation stays single-source.** Sharding the *test* split across workers and
  averaging is a subtle way to change the metric (per-shard worst-group ≠ global
  worst-group). The evaluator runs on the full, unsharded test split, exactly as
  on the tiny tier.
- **The immutability guard runs on the driver**, hashing the protected files once
  per iteration regardless of worker count.

## What you built

- The tiny-tier training step, lifted to data-parallel Ray Train workers.
- An understanding of how the compute budget generalizes to cluster wall-clock.
- The guardrails that keep the score identical whether you train on 1 core or 64.
