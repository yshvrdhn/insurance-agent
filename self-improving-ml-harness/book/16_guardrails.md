# Chapter 16 · Guardrails: immutability & anti-reward-hacking

> Companion notebook: [`notebooks/16_guardrails.ipynb`](../notebooks/16_guardrails.ipynb)
> Tier: tiny (the guard demo runs anywhere; the threat model is general)

## The agent will hack the reward if you let it

Give a capable model a score to maximize and a shell, and the instrumentally
optimal move is often *not* "build a better model" — it's "edit the thing that
computes the score." This isn't malice; it's optimization finding the shortest
path. Every guardrail in this harness exists because of it.

## The four enforced rules

1. **Immutable scorer.** `loop.py` hashes `evaluator.py` (and `data.py`,
   `program.md`, the loop itself) before and after each iteration and *rejects
   the score* on any change. The tiny-tier notebook in Chapter 6 demonstrates the
   rejection. This is the load-bearing guardrail.
2. **No test-set access in training.** Only the evaluator reads `test`. A trainer
   that opens the test split — to calibrate, to early-stop on it, to peek — is
   reward hacking even if no file changed. Enforced structurally by the split API
   and reviewable in the diff.
3. **Fixed budget.** Score is quality *per unit compute*; you can't win by quietly
   training longer.
4. **Fixed seeds, fixed test split.** A "win" that's just variance gets caught
   because the same config reproduces the same score.

## Hacks the worst-group objective specifically invites

Choosing worst-group accuracy as the score changes the menu of hacks:

- **Slice starvation** — over-fix fog until rainy collapses. The objective itself
  punishes this (the worst slice is now rainy), and the loop reverts.
- **Slice redefinition** — "merge fog into overcast so there's no fog slice."
  Blocked: the slice definitions live in immutable `data.py`/`evaluator.py`.
- **Degenerate confidence** — none of the diagnostics (which the agent *can* read)
  feed the scalar, so gaming a diagnostic does nothing.

## A review checklist for code-writing agents

When the agent edits files (Chapter 14, code mode), a human or a second model
should diff for: reads of `test`, edits creeping toward immutable files, new
network calls, and changes that only help under the eval seed. The immutability
hash catches the blatant cases; the diff review catches the subtle ones.

## What you built

- A concrete threat model for self-improving systems.
- The four enforced rules and the specific hacks the worst-group score invites.
- A review checklist for the higher-agency, code-writing mode.
