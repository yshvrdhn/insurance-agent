# Chapter 15 · Prompting the harness: program.md & results.tsv as memory

> Companion notebook: [`notebooks/15_prompting_the_harness.ipynb`](../notebooks/15_prompting_the_harness.ipynb)
> Tier: any · **not executed in CI**

## The context window is the harness's working memory

A self-improving loop is only as good as what the agent sees each iteration.
Three artifacts make up that context, and each plays a distinct role:

- **`program.md`** — the immutable direction. It states the objective (the exact
  score formula), the contract (what's modifiable, what's forbidden), the budget,
  and the guardrails. It goes in the *system prompt* and is prompt-cached: it
  never changes during a run.
- **`results.tsv`** — the episodic memory. Every past attempt, its score, whether
  it was kept, the worst slice, and the one-line rationale. This is what lets the
  agent avoid re-proposing a dead end and notice when the bottleneck slice moved.
- **the modifiable source** — `train.py` / `mining.py` / `model.py` / `configs`,
  so the agent can write a *diff*, not vague advice.

## Compressing history as the run grows

After 100 iterations `results.tsv` is too long to drop in verbatim every time.
Strategies that work:

- **Keep the Pareto frontier**, summarize the rest. Always show the best result,
  the current best, and the last few attempts in full; compress the middle into
  "tried X family, best was Y."
- **Cluster by hypothesis.** "Mining experiments: 12 tried, best boost=6 → 1.02.
  Capacity: 8 tried, plateaued at 1.01." This is the agent's own lab-notebook
  summary, regenerated periodically.

## Writing a good rationale

The `rationale` field isn't decoration — it's the breadcrumb the *next* iteration
reads. A good one names the hypothesis and the slice: *"foggy still worst at
0.73; expand fog mining to k=12 neighbours."* A bad one ("tune things") gives the
next proposal nothing to build on. When the agent writes the rationale, you get a
legible research narrative for free.

## What you built

- A clear mapping of the three context artifacts to their roles (direction,
  memory, editable surface).
- History-compression strategies that keep long runs inside the context budget.
- The discipline of slice-named rationales that make the run self-documenting.
