# Chapter 14 · The agent in the loop: wiring Claude to propose changes

> Companion notebook: [`notebooks/14_agent_in_the_loop.ipynb`](../notebooks/14_agent_in_the_loop.ipynb)
> Tier: any · **runs here with a mock proposer** (swap in `ClaudeProposer` + an API key for the real agent)

## Replacing the stand-in proposer

Part I's `SearchProposer` walked a *curated* list of moves. The point of the
whole exercise is to replace that fixed list with an LLM that reads the results
and *writes new code*. The loop already has the seam — `proposer.propose(history,
best_config) -> (override, rationale)`. A Claude-backed proposer fills it in.

```python
import anthropic

class ClaudeProposer:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def propose(self, history, best_config):
        msg = self.client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            system=open("program.md").read(),          # the immutable direction
            messages=[{"role": "user", "content": format_history(history)}],
            tools=[PROPOSE_CHANGE_TOOL],                # structured output
        )
        change = parse_tool_use(msg)
        return change["override"], change["rationale"]
```

Use **prompt caching** on the large, stable parts (`program.md`, the source of
the modifiable files) so each iteration only pays for the short, changing history
— this is the dominant cost lever in a long run.

## Two modes of agency

- **Config-space agent** (shown above): the agent returns a config override,
  same as the search proposer. Safe, bounded, easy to validate.
- **Code-writing agent**: the agent edits `train.py` / `mining.py` / `model.py`
  directly (via the Claude Agent SDK or Claude Code), then the loop trains and
  scores. More powerful, and exactly why the immutability guard exists — the
  agent has a shell, so the *only* thing stopping it from editing the evaluator
  is the hash check in `loop.py`.

## The decision stays with the harness, not the model

Crucially, the LLM **proposes**; the **harness decides**. Claude never reports the
score — `evaluator.py` does, and `loop.py` does the keep/revert. The model's job
is hypotheses and code; the system's job is honest measurement. This division is
what makes the loop trustworthy even as the proposer gets cleverer.

## What you built

- A Claude-backed proposer that drops into the existing loop seam.
- The distinction between a bounded config-space agent and a code-writing agent.
- The principle that the agent proposes while the immutable harness measures and
  decides.
