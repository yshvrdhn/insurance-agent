# Chapter 3 · Inside the immutable evaluator

> Companion notebook: [`notebooks/03_evaluator_contract.ipynb`](../notebooks/03_evaluator_contract.ipynb)

## The scoring math, in full

```python
score = worst_group_accuracy + 0.25 * macro_accuracy
```

- `worst_group_accuracy` — the lowest per-weather slice accuracy. This is the
  term that dominates, and it's deliberately a *minimum*, not a mean: you can't
  paper over a blind spot with strong slices elsewhere.
- `macro_accuracy` — the unweighted mean of per-class accuracy. Weighted at
  `0.25`, it's a tie-breaker that keeps the loop from sacrificing too much
  overall quality to win a fraction on one slice.

Everything else the evaluator returns — per-weather and per-scene accuracy, the
confusion matrix, the per-sample correctness vector — is **diagnostics**. None
of it feeds the scalar. It exists so a human or the agent can understand a
result and form the next hypothesis.

## The fingerprint: proving the ruler didn't move

`evaluator_fingerprint()` is the SHA-256 of `evaluator.py`'s own bytes. The loop
records it (indirectly, via the immutability snapshot in Chapter 6) on every
iteration. If anyone edits the evaluator — even whitespace — the hash changes
and the iteration's score is rejected. This is the technical teeth behind "the
agent cannot touch the evaluator": it's not a guideline, it's a checksum.

```python
from harness.evaluator import evaluator_fingerprint
evaluator_fingerprint()   # 64-hex-char digest, stable across calls
```

## No test-set peeking

The evaluator is the *only* code permitted to read the `test` split. This is the
other half of the contract. A training-time change that quietly reads `test` —
to, say, calibrate on the evaluation distribution — would be a textbook case of
reward hacking: the score would climb while the model got no better. Chapter 16
discusses the broader anti-reward-hacking posture; here the rule is concrete and
mechanical, and `data.py`'s split assignment enforces it structurally.

## Reading the confusion matrix

The notebook plots the 3×3 confusion matrix. It's the first thing to look at
when a slice is weak: are night frames being called dawn? Is fog collapsing two
classes together? The *kind* of mistake points at the fix. A model that confuses
dawn↔night under fog is failing on exactly the warm/cool colour cue from Chapter
2 — which tells you mining colour-similar fog frames (Chapter 5) is the move,
not, say, more epochs.

## What you built

- A complete understanding of the objective the whole system optimizes.
- The tamper-evidence mechanism (fingerprint + immutability snapshot).
- The habit of reading slices and the confusion matrix to turn a bad number into
  a specific, testable hypothesis.

Next: the surface the agent is actually allowed to change.
