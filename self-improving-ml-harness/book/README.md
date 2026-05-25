# Self-Improving ML Harness — the book

MLOps for the age of agents. We build an autonomous loop that proposes changes
to a computer-vision pipeline, trains under a fixed compute budget, scores
against an immutable evaluator, and keeps or reverts each change — Karpathy's
[AutoResearch](https://github.com/karpathy/autoresearch) pattern, generalized
from LLM training to a full vision system with real data infrastructure.

Every chapter has an executable companion notebook in [`../notebooks/`](../notebooks).
The **tiny tier (Ch. 1–7) runs anywhere** — no GPU, no download — on a synthetic
BDD-shaped dataset that ships with the repo. Later chapters scale the *same
contract* onto real distributed infra.

## How to read this book

```bash
uv pip install -e ".[dev,torch]"      # tiny tier needs torch (CPU is fine)
python scripts/download_bdd.py --tier tiny
python notebooks/build_notebooks.py   # build + run the Ch. 1–7 notebooks
bash scripts/run_loop.sh              # watch the loop improve the score
```

## The three-file contract

```
program.md            ← human-authored direction (immutable during a run)
harness/evaluator.py  ← immutable scoring function (the agent CANNOT touch)
harness/data.py       ← the task definition (immutable)
harness/train.py      ← agent-modifiable training surface
harness/model.py      ← agent-modifiable architectures
harness/mining.py     ← agent-modifiable hard-example mining
configs/*.yaml        ← agent-modifiable hyperparameters
```

`harness/loop.py` hashes the immutable set before and after every iteration and
refuses to record a score if anything protected changed. You cannot win by
editing the ruler.

## Table of contents

### Part I — The contract (`bdd-tiny`, runs anywhere)

| Ch | Title | Notebook | Stack |
|----|-------|----------|-------|
| 1  | [The immutable evaluator & your first baseline](01_evaluator_and_baseline.md) | `01_fiftyone_evaluator` | Lance, Torch |
| 2  | [The dataset: BDD-style frames in Lance & Daft](02_data_lance_daft.md) | `02_data_lance_daft` | Lance, Daft |
| 3  | [Inside the immutable evaluator](03_evaluator_contract.md) | `03_evaluator_contract` | — |
| 4  | [The training surface](04_training_surface.md) | `04_training_surface` | Torch |
| 5  | [Hard-example mining with LanceDB](05_mining_lancedb.md) | `05_mining_lancedb` | LanceDB |
| 6  | [The keep/revert loop](06_autoresearch_loop.md) | `06_autoresearch_loop` | — |
| 7  | [Reading a run](07_reading_a_run.md) | `07_reading_a_run` | pandas |

### Part II — Scaling out (`bdd-small`, laptop OK)

| Ch | Title | Notebook | Stack |
|----|-------|----------|-------|
| 8  | [From numpy to Daft pipelines](08_daft_pipelines.md) | `08_daft_pipelines` | Daft |
| 9  | [Distributed training with Ray Train](09_ray_train.md) | `09_ray_train` | Ray Train |
| 10 | [Checkpointing, fault tolerance & the budget at scale](10_checkpoint_budget.md) | `10_checkpoint_budget` | Ray Train |
| 11 | [Hyperparameter search with Ray Tune](11_ray_tune.md) | `11_ray_tune` | Ray Tune |
| 12 | [Slice-based evaluation & mistakenness with FiftyOne](12_fiftyone_eval.md) | `12_fiftyone_eval` | FiftyOne |
| 13 | [Embeddings, hardness & active mining at scale](13_active_mining.md) | `13_active_mining` | LanceDB, FiftyOne |

### Part III — The full system (`bdd-full`, cluster / GPU)

| Ch | Title | Notebook | Stack |
|----|-------|----------|-------|
| 14 | [The agent in the loop: wiring Claude to propose changes](14_agent_in_the_loop.md) | `14_agent_in_the_loop` | Claude API |
| 15 | [Prompting the harness: program.md & results.tsv as memory](15_prompting_the_harness.md) | `15_prompting_the_harness` | Claude API |
| 16 | [Guardrails: immutability & anti-reward-hacking](16_guardrails.md) | `16_guardrails` | — |
| 17 | [Multi-objective & Pareto fronts](17_multi_objective.md) | `17_multi_objective` | Ray Tune |
| 18 | [Beyond classification: detection & segmentation](18_detection_segmentation.md) | `18_detection_segmentation` | Torch, FiftyOne |
| 19 | [Cost, compute budgets & scheduling on a cluster](19_cost_and_scheduling.md) | `19_cost_and_scheduling` | Ray |
| 20 | [Running it for real on BDD100K: a case study](20_case_study_bdd.md) | `20_case_study_bdd` | everything |

> **Runnable scope.** Chapters 1–7 execute end-to-end on the synthetic tiny tier
> in this repo (and in CI). Chapters 8–20 describe the same contract on real
> distributed infrastructure (Ray, FiftyOne, a GPU/cluster, the real BDD100K
> download); their notebooks are authored against that infra and are *not*
> executed in CI. Where a concept can be shown on the tiny tier, the chapter does
> so first.
