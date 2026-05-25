# Chapter 17 · Multi-objective & Pareto fronts

> Companion notebook: [`notebooks/17_multi_objective.ipynb`](../notebooks/17_multi_objective.ipynb)
> Tier: `bdd-small`+ · **runs here on the tiny tier** (score-vs-latency Pareto front)

## One scalar isn't the whole story

The official score collapses everything into one number, which is what makes the
keep/revert decision crisp. But a real perception model is judged on more than
worst-group accuracy: **latency**, **model size**, **energy**, **calibration**.
A change that wins 0.01 of worst-group accuracy by doubling inference latency may
be a bad trade on a car.

## Pareto thinking

Instead of a single best, you track a **Pareto front**: the set of configs where
you can't improve one objective without hurting another. Ray Tune supports
multi-objective search directly:

```python
tune.TuneConfig(metric=["score", "latency_ms"], mode=["max", "min"])
# inspect the non-dominated set
front = [r for r in results if is_pareto(r, results)]
```

The agent's job shifts from "maximize the scalar" to "push the front out" — find
configs that are better on worst-group *at the same latency*, or as fast *at the
same accuracy*.

## Keeping the contract

The immutable evaluator still owns the accuracy term; the new objectives
(latency, size) are *measured*, not scored by a mutable function, so they're just
as tamper-resistant — you measure latency on fixed hardware with a fixed
protocol, the analogue of the fixed test split. `program.md` is where you'd
declare the trade-off policy (e.g., "never accept >50 ms p99") so the loop's
keep/revert rule can encode it.

## Scalarization vs. constraints

Two ways to make a multi-objective problem decidable again:

- **Constraint**: keep the single worst-group score, but reject any config that
  violates a latency/size budget. Simple, and it keeps Part I's machinery intact.
- **Scalarization**: combine objectives with weights (as the score already does
  with worst-group + macro). Flexible, but the weights are a value judgment that
  belongs in `program.md`, not in the agent's discretion.

## What you built

- A move from single-score to Pareto-front optimization.
- Multi-objective search in Ray Tune over accuracy and deployment costs.
- Two principled ways (constraint, scalarization) to keep the keep/revert
  decision well-defined.
