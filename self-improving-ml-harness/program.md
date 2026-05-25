# program.md — agent direction

> This file is **human-authored** and **immutable during a run**. It is the
> only place where the goal of the loop is stated. The harness reads it once at
> startup. The agent may *read* it as often as it likes but must never edit it.

## Objective

Maximize the **official score** produced by `harness/evaluator.py` on the
held-out `test` split of the active dataset tier, **without ever editing the
evaluator**.

The official score is:

```
score = worst_group_accuracy            # the lowest per-(weather) slice accuracy
        + 0.25 * macro_accuracy         # tie-break toward overall quality
```

We optimize the **worst slice** on purpose: a self-driving perception model
that is great in clear daytime but blind in fog is not safe. The `0.25`
coefficient keeps the loop from sacrificing too much average accuracy to nudge
one slice.

## The contract

The agent may edit **only** the files in the *modifiable set*:

- `harness/train.py` — the training surface (model, optimizer, schedule, aug)
- `harness/model.py` — model architectures
- `harness/mining.py` — hard-example mining strategy
- `configs/*.yaml` — hyperparameters and search spaces

The agent may **never** edit:

- `program.md` (this file)
- `harness/evaluator.py` (the immutable scorer)
- `harness/data.py` (the dataset definition — changing it would change the task)
- `harness/loop.py`, `harness/config.py`, `harness/registry.py` (the loop plumbing)
- `tests/` (the contract tests, including the evaluator-integrity test)

`harness/loop.py` enforces this by hashing the immutable set before and after
each iteration and refusing to record any score if a protected file changed.

## The compute budget

Each training iteration runs under a fixed budget (see `configs/base.yaml`:
`budget`). A change that needs more compute than the budget allows is not a
valid improvement — the loop kills training at the budget and scores whatever
checkpoint exists. Trading the whole budget for a marginal gain is discouraged;
prefer changes that improve the score *per unit compute*.

## How an iteration works

1. Read `results.tsv` — the full history of past attempts and their scores.
2. Form a hypothesis ("the foggy slice is weak and rare — mine fog-like frames
   and upweight them so the model stops underfitting them").
3. Edit one file in the modifiable set.
4. Run `harness/train.py` under the budget.
5. Score with `harness/evaluator.py`.
6. If the new score beats the best-so-far, **keep** the change (commit).
   Otherwise **revert** it.
7. Append the attempt to `results.tsv` with a one-line rationale.

## Guardrails

- Reproducibility: every iteration sets the global seed from `configs/base.yaml`.
- No test-set peeking: `evaluator.py` is the only code allowed to read the
  `test` split. Training code that reads `test` is a contract violation.
- One change at a time: keep iterations small so the keep/revert signal is clean.
- Honesty: if a change cannot be evaluated (crash, timeout), record it as a
  failure rather than dropping it.
