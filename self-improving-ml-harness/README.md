# Self-Improving ML Harness

A hands-on book on **MLOps for the age of agents** — building self-improving ML
systems with an LLM-driven harness around a real computer-vision workload.

The book walks through building an autonomous loop that proposes changes to a
vision pipeline, trains under a fixed compute budget, evaluates against an
immutable scoring function, and keeps or reverts each change. This is
[Karpathy's AutoResearch](https://github.com/karpathy/autoresearch) pattern,
generalized beyond LLM training to a full vision system with proper data
infrastructure.

> **Runs out of the box.** Chapters 1–7 (the tiny tier) execute end-to-end on a
> **synthetic, BDD-shaped dataset** that the repo generates locally — no GPU and
> no multi-hundred-megabyte download. The synthetic frames carry the same
> `weather` / `timeofday` / `scene` schema as real BDD100K, so the later chapters
> swap in the real dataset without changing the harness.

## The stack

| Layer            | Tool        | What it does                                   |
|------------------|-------------|------------------------------------------------|
| Data pipeline    | Daft        | Distributed multimodal dataframes              |
| Storage          | Lance       | Columnar format for datasets, embeddings, tensors |
| Vector search    | LanceDB     | Hard-example mining via similarity             |
| Training         | Ray Train   | Distributed model training                     |
| Hyperparameter   | Ray Tune    | ASHA / PBT / BOHB search                       |
| Evaluation       | FiftyOne    | Slice-based eval, mistakenness, hardness       |
| Agent            | Claude / Codex CLI | The decision layer                      |

The **tiny tier** uses the lightweight subset of this stack (Lance, LanceDB,
Daft, a CPU Torch model) so it runs anywhere. Ray, FiftyOne, and the cluster
pieces come online in Parts II–III.

## The three-file contract

Following AutoResearch, the harness enforces a strict separation:

```
program.md            ← human-authored direction (immutable during a run)
harness/evaluator.py  ← immutable scoring function (agent CANNOT touch)
harness/data.py       ← the task definition (immutable)
harness/train.py      ← agent-modifiable training surface
harness/model.py      ← agent-modifiable architectures
harness/mining.py     ← agent-modifiable hard-example mining
configs/              ← agent-modifiable config files
```

`harness/loop.py` hashes the immutable set before and after each iteration and
refuses to record a score if a protected file changed — you cannot win by
editing the ruler.

## The dataset

[BDD100K](https://bdd-data.berkeley.edu/) — diverse driving scenes from Berkeley
DeepDrive, tagged with weather, time-of-day, and scene. Three tiers:

- `bdd-tiny` (synthetic, ~3K frames) — runs anywhere, Chapters 1–7
- `bdd-small` (~12K frames) — laptop OK, Chapters 8–13
- `bdd-full` — Ray cluster / GPUs, Chapters 14–20

```bash
python scripts/download_bdd.py --tier tiny     # synthesizes the tiny tier
python scripts/download_bdd.py --real --src /path/to/bdd100k --tier full  # real
```

The task: classify **time-of-day**. Slices are **weather**, and `foggy` is the
engineered worst slice — rare and visually degraded — that the harness must learn
to recover.

## Quickstart

```bash
# one-time setup
uv pip install -e ".[dev,torch]"
python scripts/download_bdd.py --tier tiny

# build + run the Chapter 1–7 notebooks (executes them end to end)
python notebooks/build_notebooks.py

# run the autonomous keep/revert loop
bash scripts/run_loop.sh            # or: python -m harness.loop --iters 8

# tests
pytest
```

A loop run reads like a lab notebook:

```
[iter 0] baseline score=0.883 worst=0.644(foggy)
[iter 1] KEEP   score=0.972 :: upweight fog frames 5x
[iter 3] KEEP   score=1.018 :: more capacity for the weak fog signal
[iter 5] KEEP   score=1.019 :: weight decay to denoise fog frames
```

## Repo layout

```
.
├── book/             # 20 written chapters (markdown) + TOC
├── notebooks/        # executable companions (Ch. 1–7 run in CI)
├── harness/          # the library (the three-file contract)
├── configs/          # base + search configs (agent-modifiable)
├── scripts/          # download_bdd.py, run_loop.sh
├── tests/            # unit + contract + integration tests
├── data/             # gitignored; BDD subsets land here
└── program.md        # the agent's direction file
```

## Chapter index

See [book/README.md](book/README.md) for the full 20-chapter TOC and the
chapter-to-notebook mapping.

## License

Code: MIT. The BDD100K dataset has its own research-use license; see the BDD
site for terms.
