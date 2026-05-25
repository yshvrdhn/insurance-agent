# CLAUDE.md — working notes for Agentic Flywheel

Context for resuming work on this project. Read this first.

## What this is

**Agentic Flywheel** — a hands-on book + runnable codebase on *MLOps for the age
of agents*. An LLM-driven AutoResearch loop wraps a computer-vision workload:
**propose a change → train under a fixed budget → score with an immutable
evaluator → keep if it beats the best, else revert → log**. Generalizes
Karpathy's AutoResearch from LLM training to a vision system with real data infra
(Lance, LanceDB, Daft, Ray, FiftyOne).

This directory (`self-improving-ml-harness/` inside the `insurance-agent` repo)
is the self-contained project. It is destined for the standalone repo
**github.com/yshvrdhn/agentic-flywheel** (the distribution is named
`agentic-flywheel`; the importable Python package stays `harness`).

## The contract (read before editing anything)

Three-file separation, enforced by a hash guard in `harness/loop.py`:

| Role | Files |
|------|-------|
| **Immutable** (never edit during a run) | `program.md`, `harness/evaluator.py`, `harness/data.py`, `harness/loop.py`, `harness/config.py`, `harness/registry.py` |
| **Agent-modifiable** | `harness/train.py`, `harness/model.py`, `harness/mining.py`, `configs/*.yaml` |

`harness/loop.py` SHA-256s the immutable set before/after each iteration and
rejects the score if anything protected changed. The lists live in
`harness/__init__.py` (`IMMUTABLE_FILES`, `MODIFIABLE_FILES`).

The official score (in `evaluator.py`, do not change):
```
score = worst_group_accuracy + 0.25 * macro_accuracy   # groups = weather slices
```

## The task & the engineered difficulty (important)

Synthetic, BDD-shaped data (`harness/data.py`) — no download, no GPU. Classify
**time-of-day** (daytime/dawn/night); **weather** defines the eval slices.

`foggy` is deliberately the **hard, rare worst slice**, by design in `data.py`:
- **rare**: `_WEATHER_P` makes fog ~10% of frames;
- **brightness scrambled**: fog applies a large random *multiplicative* haze
  `exp(N(0, 0.55))`, so mean-brightness (the easy cue) is useless on fog;
- **colour washed out + noisy**: fog halves the time-of-day colour tint and adds
  colour noise, leaving only a weak residual signal.

So a budget baseline underfits fog. The levers that recover it (and that the loop
discovers): **fog mining + up-weighting** (biggest), capacity, epochs,
regularization. Over-boosting fog shifts the worst slice to another weather — the
intended "watch the worst slice, one change at a time" lesson.

⚠️ If you retune `data.py`, the difficulty is sensitive: a too-clean colour cue
makes the task trivially 100% (BatchNorm makes the CNN brightness-invariant for
free). Verify with the real CNN, not a linear model. Reference baseline ≈ **0.88**
(fog ≈ 0.64); a good loop run reaches ≈ **1.02**.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install -e ".[dev,torch]"          # tiny tier (Ch. 1-7)
uv pip install -e ".[ray,fiftyone]"       # scaling tier (Ch. 8-20)
python scripts/download_bdd.py --tier tiny   # synthesizes data/bdd-tiny.lance
```

## Run

```bash
pytest                                    # 17 tests (unit + contract + integration)
python -m harness.loop --iters 8          # the autonomous loop
bash scripts/run_loop.sh
python notebooks/build_notebooks.py       # build+execute Ch. 1-7 notebooks
python notebooks/build_scaling_notebooks.py   # build+execute Ch. 8-20 notebooks
```

Notebooks are generated from the build scripts (single source of truth) and
executed in place via `nbclient`. Re-run a single one: `... build_notebooks.py 05`.

## Layout

```
program.md              # the agent's immutable direction file
harness/                # data, evaluator, train, model, mining, loop, config, registry
configs/                # base.yaml (+ search.yaml for Ray Tune)
scripts/                # download_bdd.py (synthetic + real-ingest stub), run_loop.sh
notebooks/              # 01-07 tiny tier, 08-20 scaling tier (+ 2 build_*.py)
book/                   # 20 chapters + README.md (TOC); book/README.md maps ch->notebook
tests/                  # conftest + test_{data,evaluator,train,loop}.py
data/                   # gitignored (only data/.gitkeep tracked)
```

## Gotchas hit during the build (don't re-discover these)

- **Lance Python bindings are `pylance`** (imports as `lance`), NOT the PyPI
  package `lance`. **Daft is `daft`** (0.7.x), NOT `getdaft` (old 0.5.x) — having
  both installed collides. pyproject pins `pylance` and `daft`.
- **Ray Tune** runs each trial in its own working dir → relative dataset paths
  break. Always pass an **absolute** dataset path into trials.
- **Ray Train** `result.metrics` can be `None`; access it defensively.
- **FiftyOne** notebook guards its import and falls back to a numpy per-slice
  report, so Ch. 12 runs even without FiftyOne. It needs its bundled Mongo.
- Notebook figures (`notebooks/*.png`) and `notebooks/results.tsv` are build
  artifacts → gitignored (the images are embedded in the executed `.ipynb`).

## Status (complete)

- Harness, configs, scripts, tests (17 passing), `program.md` — done.
- All **20 notebooks** authored and **executed cleanly** (0 cell errors). Ch.
  8-13 use the real stack in local mode; Ch. 14-20 use clearly-labelled mock/toy
  stand-ins for the API-key/GPU/cluster pieces.
- All 20 book chapters + TOC written; chapter "Tier" notes reflect what executes.

Work happens on branch `claude/self-improving-ml-harness-VmFqR` (PR #1 on
`insurance-agent`). To populate the standalone `agentic-flywheel` repo, lift this
subdirectory out and push it (see the project README / chat history).

## Conventions

- Keep edits minimal and within the modifiable set unless explicitly extending
  the framework. Never edit `evaluator.py`/`data.py` to move the score.
- Training reads only `train`/`val`; the `test` split belongs to the evaluator.
- Everything is seeded for reproducibility (`configs/base.yaml: seed`).
