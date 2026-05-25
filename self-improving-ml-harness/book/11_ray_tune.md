# Chapter 11 · Hyperparameter search with Ray Tune

> Companion notebook: [`notebooks/11_ray_tune.ipynb`](../notebooks/11_ray_tune.ipynb)
> Tier: `bdd-small`+ · **not executed in CI** (needs Ray Tune)

## Two kinds of search

The keep/revert loop is a *qualitative* search — it changes the method (add
mining, change the architecture). [Ray Tune](https://docs.ray.io/en/latest/tune/index.html)
is a *quantitative* search — given a method, find good continuous
hyperparameters. They compose: the agent proposes a method, Tune dials it in,
the evaluator scores the result.

`configs/search.yaml` already declares the space:

```yaml
search:
  algorithm: asha
  num_samples: 16
  metric: score          # the evaluator's official score
  mode: max
  space:
    train.lr:           { dist: loguniform, low: 1.0e-4, high: 5.0e-3 }
    train.weight_decay: { dist: loguniform, low: 1.0e-6, high: 1.0e-3 }
    model.width:        { dist: choice, values: [16, 24, 32, 48] }
    mining.boost:       { dist: uniform, low: 2.0, high: 9.0 }
```

## ASHA, PBT, BOHB

- **ASHA** (Asynchronous Successive Halving) — start many trials cheap, kill the
  laggards early, give survivors more budget. Best default for our fixed-budget
  setting: it spends compute where the worst-group score is already promising.
- **PBT** (Population Based Training) — evolve a population, periodically copying
  the weights of winners and perturbing their hyperparameters. Good for schedules
  (lr, `mining.boost`) that should *change* during training.
- **BOHB** — Bayesian optimization over the halving bracket; sample-efficient
  when each trial is expensive.

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
tuner = tune.Tuner(
    trainable,                       # trains + scores with the immutable evaluator
    param_space=space,
    tune_config=tune.TuneConfig(metric="score", mode="max",
                                scheduler=ASHAScheduler(), num_samples=16),
)
best = tuner.fit().get_best_result()
```

## Tune optimizes the *real* score

The trial function trains under budget and returns `evaluator.evaluate(...)['score']`
— the same worst-group-plus-macro objective, computed by the same immutable code.
Tune never sees the test split directly; it only sees the scalar the evaluator
returns. This keeps automated HPO inside the same anti-reward-hacking contract as
everything else.

## What you built

- A search space declared in config and three schedulers to explore it.
- The composition of qualitative (loop) and quantitative (Tune) search.
- HPO that optimizes the official worst-group score without breaching the
  evaluation contract.
