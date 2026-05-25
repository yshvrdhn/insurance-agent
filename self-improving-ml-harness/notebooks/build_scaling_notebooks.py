#!/usr/bin/env python
"""Generate and execute the scaling-tier companion notebooks (Ch. 8-20).

These chapters target real distributed infra (Ray, FiftyOne, a cluster, the real
BDD download, the Claude API). So the *book* is honest and every notebook still
runs top-to-bottom, each notebook demonstrates its concept on the synthetic tiny
tier and on this machine:

* heavy/optional libraries (Ray, FiftyOne) are guarded -- if absent, the cell
  prints a clear note instead of crashing;
* the Claude-API / GPU-detection / cluster pieces use clearly-labelled mock or
  toy stand-ins so the mechanics execute without a key, a GPU, or a cluster.

    python notebooks/build_scaling_notebooks.py            # build + execute all
    python notebooks/build_scaling_notebooks.py --no-exec  # build only
    python notebooks/build_scaling_notebooks.py 11         # just one
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def md(text: str):
    return ("md", text.strip("\n"))


def code(text: str):
    return ("code", text.strip("\n"))


BOOT = code(
    """
import sys, os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd().parent))
import matplotlib; matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from harness.data import make_synthetic_dataset, DatasetSpec, load_split, class_names

DATA = Path("../data/bdd-tiny.lance")
if not DATA.exists():
    make_synthetic_dataset(DATA, DatasetSpec(n=3000, seed=7))
print("dataset:", DATA, "| NOTE: these chapters scale to bdd-small/full; here we")
print("demonstrate the mechanics on the tiny tier so they run with no GPU/cluster.")
"""
)

NOTEBOOKS: dict[str, list] = {}

# --------------------------------------------------------------------------- #
NOTEBOOKS["08_daft_pipelines"] = [
    md(
        """
# 8 · From numpy to Daft pipelines

The tiny tier loads everything into a numpy array. That doesn't scale. Here we
express the data pipeline as a lazy **Daft** dataframe over the Lance dataset —
the same code runs on a cluster against `bdd-full` (Chapter 9).
"""
    ),
    BOOT,
    md("### Read Lance into Daft and push the split filter down"),
    code(
        """
import daft, lance
tbl = lance.dataset(str(DATA)).to_table().drop(["image"])   # metadata columns
df = daft.from_arrow(tbl)
print("schema:"); print(df.schema())
train = df.where(df["split"] == "train")
print("train rows:", train.count_rows())
(train.groupby("weather").agg(daft.col("id").count().alias("n"))
      .sort("weather").show())
"""
    ),
    md("### Stream cue features into the dataframe and aggregate per slice"),
    code(
        """
# In production the embedding is a Daft UDF over the image column (sketch below);
# here we compute it with the harness helper and attach it as columns.
from harness.mining import embed
x, y, meta = load_split(DATA, "train")
f = embed(x)   # (N, 4): R,G,B colour ratio + contrast
feat = daft.from_pydict({
    "weather": meta["weather"], "r": f[:, 0], "g": f[:, 1], "b": f[:, 2],
})
(feat.groupby("weather")
     .agg(daft.col("r").mean().alias("R"), daft.col("b").mean().alias("B"))
     .sort("weather").show())
print("Fog frames sit apart in colour-ratio space -> that's what mining exploits.")
"""
    ),
    md(
        """
```python
# production: decode + featurize as a streamed UDF, never materializing the set
@daft.udf(return_dtype=daft.DataType.fixed_size_list(daft.DataType.float32(), 4))
def colour_signature(images): ...
df = daft.read_lance("data/bdd-full.lance").with_column("embed", colour_signature(df["image"]))
```
The query plan, the filter pushdown, and the per-slice aggregation are identical
whether the data is 3K rows on a laptop or 100K on a Ray cluster.
"""
    ),
]

# --------------------------------------------------------------------------- #
NOTEBOOKS["09_ray_train"] = [
    md(
        """
# 9 · Distributed training with Ray Train

`harness/train.py`'s single-process loop is right for the tiny tier and wrong for
`bdd-full`. **Ray Train** runs the *same* training function across N data-parallel
workers. Below we run it locally with 2 CPU workers; on a cluster you change only
`num_workers` / `use_gpu`.
"""
    ),
    BOOT,
    code(
        """
try:
    import ray
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer
    HAVE_RAY = True
except Exception as e:
    HAVE_RAY = False; print("Ray not installed -> showing the code only:", e)
"""
    ),
    md("### Wrap the per-worker training step and fit across 2 workers"),
    code(
        """
def train_func(cfg):
    import numpy as np, torch, torch.nn.functional as F
    import ray.train
    from harness.data import load_split, class_names
    from harness.model import build_model
    rank = ray.train.get_context().get_world_rank()
    world = ray.train.get_context().get_world_size()
    x, y, _ = load_split(cfg["data"], "train")
    x, y = x[rank::world], y[rank::world]      # shard across workers
    model = ray.train.torch.prepare_model(build_model(cfg, len(class_names())))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    xt = torch.from_numpy(x.transpose(0, 3, 1, 2)).float()
    yt = torch.from_numpy(y).long()
    for epoch in range(cfg["epochs"]):
        opt.zero_grad(); loss = F.cross_entropy(model(xt), yt); loss.backward(); opt.step()
        ray.train.report({"loss": float(loss)})

if HAVE_RAY:
    if not ray.is_initialized():
        ray.init(num_cpus=4, logging_level="ERROR", include_dashboard=False)
    trainer = TorchTrainer(
        train_func,
        train_loop_config={"data": str(DATA.resolve()), "lr": 1e-3, "epochs": 3,
                           "model": {"name": "tiny_cnn", "width": 16, "depth": 2}},
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
    )
    result = trainer.fit()
    m = result.metrics or {}
    loss = m.get("loss")
    if loss is None and getattr(result, "metrics_dataframe", None) is not None \\
            and "loss" in result.metrics_dataframe:
        loss = float(result.metrics_dataframe["loss"].iloc[-1])
    print("Ray Train ran across 2 data-parallel workers.")
    print("final loss:", round(loss, 4) if loss is not None else "(metrics not surfaced)")
else:
    print("(install ray[train] to run this)")
"""
    ),
    md(
        """
Two workers each trained on their shard with synchronized gradients
(`prepare_model` wraps DistributedDataParallel). The evaluator still scores the
**full, unsharded** test split on the driver — sharding evaluation would quietly
change the worst-group metric.
"""
    ),
]

# --------------------------------------------------------------------------- #
NOTEBOOKS["10_checkpoint_budget"] = [
    md(
        """
# 10 · Checkpointing, fault tolerance & the budget at scale

An unattended loop must survive preemption. Ray Train persists a checkpoint each
epoch and restarts from it on failure; the budget becomes a hard deadline that
scores the *best checkpoint so far*. We demo checkpoint → restore locally.
"""
    ),
    BOOT,
    code(
        """
try:
    import ray, tempfile, os, torch
    from ray.train import ScalingConfig, Checkpoint
    from ray.train.torch import TorchTrainer
    HAVE_RAY = True
except Exception as e:
    HAVE_RAY = False; print("Ray not installed -> code only:", e)
"""
    ),
    code(
        """
def train_func(cfg):
    import tempfile, os, torch, torch.nn.functional as F, ray.train
    from harness.data import load_split, class_names
    from harness.model import build_model
    x, y, _ = load_split(cfg["data"], "train")
    model = build_model(cfg, len(class_names()))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.from_numpy(x.transpose(0, 3, 1, 2)).float(); yt = torch.from_numpy(y).long()
    best = 1e9
    for epoch in range(cfg["epochs"]):
        opt.zero_grad(); loss = F.cross_entropy(model(xt), yt); loss.backward(); opt.step()
        with tempfile.TemporaryDirectory() as d:
            torch.save(model.state_dict(), os.path.join(d, "m.pt"))
            ray.train.report({"loss": float(loss)},
                             checkpoint=Checkpoint.from_directory(d))

if HAVE_RAY:
    if not ray.is_initialized():
        ray.init(num_cpus=4, logging_level="ERROR", include_dashboard=False)
    trainer = TorchTrainer(
        train_func,
        train_loop_config={"data": str(DATA.resolve()), "epochs": 4,
                           "model": {"name": "tiny_cnn", "width": 16, "depth": 2}},
        scaling_config=ScalingConfig(num_workers=1))
    result = trainer.fit()
    print("best checkpoint:", result.checkpoint is not None,
          "| final loss:", round(result.metrics["loss"], 4))
    print("on preemption Ray restarts from this checkpoint; the budget deadline")
    print("scores whatever checkpoint exists -> safe truncation, no lost run.")
else:
    print("(install ray[train] to run this)")
"""
    ),
    md(
        """
Because each iteration writes to its own run dir and the *decision* derives only
from `results.tsv` + the immutable evaluator, a crashed iteration can be retried
without corrupting history — and fixed seeds make the retry reproducible.
"""
    ),
]

# --------------------------------------------------------------------------- #
NOTEBOOKS["11_ray_tune"] = [
    md(
        """
# 11 · Hyperparameter search with Ray Tune

The loop is a *qualitative* search (change the method); **Ray Tune** is a
*quantitative* one (dial in the hyperparameters). Each trial trains under budget
and returns the **evaluator's official score** — Tune never touches the test
split directly. We run a small ASHA search.
"""
    ),
    BOOT,
    code(
        """
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    HAVE_RAY = True
except Exception as e:
    HAVE_RAY = False; print("Ray Tune not installed -> code only:", e)
"""
    ),
    code(
        """
def trainable(params):
    # Ray Tune runs each trial in its own working dir, so the dataset path must
    # be absolute (passed in via params["data"]).
    from harness.config import load_config
    from harness.train import train
    from harness.evaluator import evaluate
    from harness.mining import compute_sample_weights
    p = params["data"]
    cfg = load_config("base", {
        "dataset": {"path": p},
        "train": {"lr": params["lr"]},
        "model": {"width": params["width"]},
        "mining": {"enabled": True, "strategy": "fog_boost", "boost": params["boost"]},
        "budget": {"max_epochs": 8, "max_seconds": 60},
    })
    w = compute_sample_weights(p, cfg)
    m = train(cfg, p, sample_weights=w)
    return {"score": evaluate(m, p)["score"]}

if HAVE_RAY:
    if not ray.is_initialized():
        ray.init(num_cpus=4, logging_level="ERROR", include_dashboard=False)
    space = {"data": str(DATA.resolve()),
             "lr": tune.loguniform(3e-4, 3e-3),
             "width": tune.choice([16, 24, 32]),
             "boost": tune.uniform(3.0, 8.0)}
    tuner = tune.Tuner(
        tune.with_resources(trainable, {"cpu": 2}),
        param_space=space,
        tune_config=tune.TuneConfig(metric="score", mode="max",
                                    scheduler=ASHAScheduler(), num_samples=6))
    res = tuner.fit()
    best = res.get_best_result(metric="score", mode="max")
    print("best score:", round(best.metrics["score"], 4), "| config:", best.config)
else:
    print("(install ray[tune] to run this)")
"""
    ),
    md(
        """
ASHA started 6 trials cheap and gave budget to the promising ones. The best
config is a candidate the *loop* can adopt — qualitative and quantitative search
compose, both judged by the same immutable score.
"""
    ),
]

# --------------------------------------------------------------------------- #
NOTEBOOKS["12_fiftyone_eval"] = [
    md(
        """
# 12 · Slice-based evaluation with FiftyOne

The evaluator already returns per-slice numbers; **FiftyOne** turns them into a
dataset you can explore and adds *mistakenness* and *hardness*. We build a
FiftyOne dataset from the test split and evaluate it; if FiftyOne isn't
installed, we fall back to the same per-slice report in numpy so the cell still
runs.
"""
    ),
    BOOT,
    code(
        """
from harness.config import load_config
from harness.train import train
model = train(load_config("base", {"budget": {"max_epochs": 10}}), DATA)
xte, yte, mte = load_split(DATA, "test")
pred = model.predict(xte)
names = list(class_names())
"""
    ),
    code(
        """
try:
    import fiftyone as fo
    from PIL import Image
    import tempfile, os
    d = tempfile.mkdtemp()
    ds = fo.Dataset(name="bdd_eval", overwrite=True)
    samples = []
    for i in range(len(xte)):
        p = os.path.join(d, f"{i}.png")
        Image.fromarray((xte[i] * 255).astype("uint8")).save(p)
        s = fo.Sample(filepath=p)
        s["weather"] = mte["weather"][i]
        s["gt"] = fo.Classification(label=names[yte[i]])
        s["pred"] = fo.Classification(label=names[pred[i]])
        samples.append(s)
    ds.add_samples(samples)
    results = ds.evaluate_classifications("pred", gt_field="gt", method="simple")
    print("FiftyOne per-class report:"); results.print_report()
    for w in sorted(set(mte["weather"])):
        v = ds.match(fo.ViewField("weather") == w)
        acc = v.match(fo.ViewField("pred.label") == fo.ViewField("gt.label")).count() / v.count()
        print(f"  {w:9s} acc={acc:.3f}")
    print("(fo.launch_app(ds.match_tags('foggy')) opens the worst slice in a browser)")
except Exception as e:
    print("FiftyOne unavailable -> numpy per-slice fallback:", type(e).__name__)
    correct = (pred == yte)
    for w in sorted(set(mte["weather"])):
        sel = mte["weather"] == w
        print(f"  {w:9s} acc={correct[sel].mean():.3f}")
"""
    ),
    md(
        """
FiftyOne's value is *looking* at the fog failures and computing **mistakenness**
(likely-wrong labels) and **hardness** (low-margin samples) — the ranking that
feeds mining at scale (Chapter 13). The scalar score is unchanged; FiftyOne is an
analysis layer, not a second scorer.
"""
    ),
]

# --------------------------------------------------------------------------- #
NOTEBOOKS["13_active_mining"] = [
    md(
        """
# 13 · Embeddings, hardness & active mining at scale

Mining becomes a flywheel: embed frames, index them in **LanceDB**, retrieve the
hard slice's neighbourhood by ANN search, rank by **hardness**, up-weight, retrain.
We run the full cycle on the tiny tier.
"""
    ),
    BOOT,
    code(
        """
from harness.config import load_config
from harness.train import train
from harness.evaluator import evaluate
from harness.mining import embed
import numpy as np

# 1) a trained model gives us per-frame hardness (low margin = hard)
base = train(load_config("base", {"budget": {"max_epochs": 10}}), DATA)
xtr, ytr, mtr = load_split(DATA, "train")
proba = base.predict_proba(xtr)
margin = np.sort(proba, axis=1)[:, -1] - np.sort(proba, axis=1)[:, -2]
print("hardest frames (lowest margin) by weather:")
order = np.argsort(margin)[:50]
import collections; print(" ", dict(collections.Counter(mtr["weather"][order])))
"""
    ),
    code(
        """
# 2) index embeddings in LanceDB and expand the fog set by ANN neighbours
import lancedb
feats = embed(xtr).astype("float32")
db = lancedb.connect("/tmp/.mlh_ch13")
try: db.drop_table("f")
except Exception: pass
tbl = db.create_table("f", data=[{"row": i, "weather": str(mtr["weather"][i]),
                                  "vector": feats[i].tolist()} for i in range(len(feats))])
fog = np.where(mtr["weather"] == "foggy")[0]
mined = set(fog.tolist())
for qi in fog[:30]:
    for r in tbl.search(feats[qi].tolist()).limit(6).to_list():
        mined.add(int(r["row"]))
print(f"fog frames: {len(fog)} -> mined neighbourhood: {len(mined)} (LanceDB ANN)")
"""
    ),
    code(
        """
# 3) up-weight the mined set and retrain; compare the fog slice
w = np.ones(len(xtr)); w[np.fromiter(mined, int)] = 6.0
cfg = load_config("base", {"budget": {"max_epochs": 10}})
mined_model = train(cfg, DATA, sample_weights=w)
b, m = evaluate(base, DATA), evaluate(mined_model, DATA)
print(f"baseline  fog={b['per_weather']['foggy']:.3f}  score={b['score']:.3f}")
print(f"+mining   fog={m['per_weather']['foggy']:.3f}  score={m['score']:.3f}")
"""
    ),
    md(
        """
With *unlabeled* frames the same index powers **active learning**: retrieve the
neighbours of hard fog frames and send those for annotation — spend the labeling
budget where the worst slice needs it. Cap the mined fraction per round so the
distribution drifts gradually and another slice doesn't collapse.
"""
    ),
]

# --------------------------------------------------------------------------- #
NOTEBOOKS["14_agent_in_the_loop"] = [
    md(
        """
# 14 · The agent in the loop: wiring Claude to propose changes

The loop has a seam: `proposer.propose(history, best_config) -> (override,
rationale)`. The real proposer is Claude (code below). With no API key here, we
run a **MockProposer** that reads the history heuristically — same interface,
same loop — so the mechanics execute end to end.
"""
    ),
    BOOT,
    md(
        """
```python
# the real thing (needs ANTHROPIC_API_KEY):
import anthropic
class ClaudeProposer:
    def __init__(self): self.client = anthropic.Anthropic()
    def propose(self, history, best_config):
        msg = self.client.messages.create(
            model="claude-opus-4-7", max_tokens=1024,
            system=open("../program.md").read(),          # prompt-cache this
            messages=[{"role": "user", "content": format_history(history)}],
            tools=[PROPOSE_CHANGE_TOOL])
        change = parse_tool_use(msg)
        return change["override"], change["rationale"]
```
"""
    ),
    code(
        """
# a stand-in that emulates an LLM reading results.tsv and picking the next move
class MockProposer:
    PLAYBOOK = [
        ({"mining": {"enabled": True, "strategy": "fog_boost", "boost": 5.0}},
         "worst slice is foggy & rare -> upweight fog 5x"),
        ({"mining": {"enabled": True, "strategy": "fog_knn", "boost": 6.0, "k": 8}},
         "expand fog set via LanceDB neighbours"),
        ({"model": {"width": 32}}, "more capacity for the weak fog cue"),
        ({"train": {"weight_decay": 1e-4}}, "denoise high-variance fog frames"),
    ]
    def __init__(self): self.i = 0
    def propose(self, history, best_config):
        # a real agent would reason over `history`; we step the playbook
        move = self.PLAYBOOK[min(self.i, len(self.PLAYBOOK) - 1)]; self.i += 1
        return move

from harness.loop import run_loop
out = run_loop(iters=4, dataset_path=str(DATA), results_path="results.tsv",
               proposer=MockProposer(), verbose=True)
print("\\nbest score:", round(out["best_score"], 4))
"""
    ),
    md(
        """
The LLM **proposes**; the **harness decides**. Claude never reports the score —
`evaluator.py` does and `loop.py` does the keep/revert. Swap `MockProposer` for
`ClaudeProposer` and set `ANTHROPIC_API_KEY` to put a real model in the seat.
"""
    ),
]

# --------------------------------------------------------------------------- #
NOTEBOOKS["15_prompting_the_harness"] = [
    md(
        """
# 15 · Prompting the harness: program.md & results.tsv as memory

The agent's context each iteration is three artifacts: **program.md** (immutable
direction, system prompt, prompt-cached), **results.tsv** (episodic memory), and
the **modifiable source**. We assemble the actual prompt — no API call needed.
"""
    ),
    BOOT,
    code(
        """
from harness.loop import run_loop
run_loop(iters=4, dataset_path=str(DATA), results_path="results.tsv", verbose=False)

system_prompt = Path("../program.md").read_text()
print("SYSTEM PROMPT: program.md  (", len(system_prompt), "chars, prompt-cached )\\n")
print(system_prompt[:380], "...\\n")
"""
    ),
    code(
        """
import csv
rows = list(csv.DictReader(open("results.tsv"), delimiter="\\t"))
def format_history(rows, last=6):
    best = max(rows, key=lambda r: float(r["score"]))
    lines = [f"best so far: score={best['score']} via {best['action']}"]
    lines.append("recent attempts (compress older ones at scale):")
    for r in rows[-last:]:
        lines.append(f"  iter {r['iter']}: score={r['score']} kept={r['kept']} "
                     f"worst={r['worst_group_name']}({r['worst_group']}) :: {r['rationale']}")
    return "\\n".join(lines)
print("USER MESSAGE (episodic memory from results.tsv):\\n")
print(format_history(rows))
"""
    ),
    code(
        """
PROPOSE_CHANGE_TOOL = {
    "name": "propose_change",
    "description": "Propose one config override to try next, with a rationale.",
    "input_schema": {"type": "object", "properties": {
        "override": {"type": "object"},
        "rationale": {"type": "string"}}, "required": ["override", "rationale"]}}
print("TOOL the model calls to return a structured proposal:")
import json; print(json.dumps(PROPOSE_CHANGE_TOOL, indent=2)[:400], "...")
"""
    ),
    md(
        """
A good `rationale` names the hypothesis and the slice — it's the breadcrumb the
next iteration reads. As history grows, compress the middle (keep the Pareto
frontier + the last few attempts in full) to stay inside the context budget.
"""
    ),
]

# --------------------------------------------------------------------------- #
NOTEBOOKS["16_guardrails"] = [
    md(
        """
# 16 · Guardrails: immutability & anti-reward-hacking

Give a capable agent a score and a shell and the shortest path is often to edit
the scorer. Every guardrail exists for that. We exercise the load-bearing one:
the immutability hash guard.
"""
    ),
    BOOT,
    code(
        """
from harness.loop import snapshot_immutable, verify_immutable
from harness import IMMUTABLE_FILES, MODIFIABLE_FILES
print("immutable (agent CANNOT touch):"); [print("  ", f) for f in IMMUTABLE_FILES]
print("modifiable (agent CAN edit):");    [print("  ", f) for f in MODIFIABLE_FILES]

snap = snapshot_immutable()
verify_immutable(snap)                      # clean iteration -> passes
print("\\nclean iteration: guard passes")
"""
    ),
    code(
        """
# simulate the agent editing the evaluator mid-iteration
tampered = dict(snap); tampered["harness/evaluator.py"] = "0" * 64
try:
    verify_immutable(tampered)
except RuntimeError as e:
    print("REJECTED:", e)
print("\\nThe score is thrown out -> you cannot win by editing the ruler.")
"""
    ),
    md(
        """
The other enforced rules: only the evaluator reads `test` (no peeking), a fixed
**budget** (no winning by training longer), and fixed seeds + a fixed test split
(a 'win' that's just variance reproduces away). The worst-group objective also
self-punishes *slice starvation* — over-fixing fog until another slice collapses
drops the score, so the loop reverts it.
"""
    ),
]

# --------------------------------------------------------------------------- #
NOTEBOOKS["17_multi_objective"] = [
    md(
        """
# 17 · Multi-objective & Pareto fronts

A real model is judged on more than accuracy — **latency** and **size** matter.
We sweep model width, measure worst-group score *and* an inference-latency proxy,
and find the **Pareto front**.
"""
    ),
    BOOT,
    code(
        """
import time
from harness.config import load_config
from harness.train import train
from harness.evaluator import evaluate
xte, _, _ = load_split(DATA, "test")
results = []
for width in [8, 12, 16, 24, 32]:
    cfg = load_config("base", {"model": {"width": width}, "budget": {"max_epochs": 10}})
    m = train(cfg, DATA)
    t = time.time(); [m.predict(xte) for _ in range(3)]; latency = (time.time() - t) / 3 * 1000
    r = evaluate(m, DATA)
    results.append({"width": width, "score": r["score"], "latency_ms": round(latency, 1)})
    print(f"width={width:2d}  score={r['score']:.3f}  latency={latency:6.1f} ms/batch")
"""
    ),
    code(
        """
def pareto(rows):
    front = []
    for r in rows:
        dominated = any((o["score"] >= r["score"] and o["latency_ms"] <= r["latency_ms"]
                         and o != r) for o in rows)
        if not dominated: front.append(r)
    return front
front = pareto(results)
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.scatter([r["latency_ms"] for r in results], [r["score"] for r in results], label="all")
fx = sorted(front, key=lambda r: r["latency_ms"])
ax.plot([r["latency_ms"] for r in fx], [r["score"] for r in fx], "-o", color="#c0392b", label="Pareto front")
for r in results: ax.annotate(f"w{r['width']}", (r["latency_ms"], r["score"]))
ax.set_xlabel("latency (ms/batch)"); ax.set_ylabel("worst-group score"); ax.legend()
fig.tight_layout(); fig.savefig("fig_17_pareto.png", dpi=90); plt.close(fig)
from IPython.display import Image; Image("fig_17_pareto.png")
"""
    ),
    md(
        """
The front is the set where you can't improve score without paying latency. The
agent's job shifts from "maximize the scalar" to "push the front out". Encode the
trade-off policy (e.g. "never accept >X ms p99") in `program.md` so the
keep/revert rule stays well-defined.
"""
    ),
]

# --------------------------------------------------------------------------- #
NOTEBOOKS["18_detection_segmentation"] = [
    md(
        """
# 18 · Beyond classification: detection & segmentation

The contract is task-agnostic — only the **metric** and the **slice definitions**
change. Real detection/segmentation needs the BDD labels and a GPU; here we show
the metric swap with a tiny **worst-group IoU** computed on toy masks.
"""
    ),
    BOOT,
    code(
        """
import numpy as np
rng = np.random.default_rng(0)
# toy: 200 frames, per-frame drivable-area mask IoU, tagged by weather
weather = rng.choice(["clear", "rainy", "night_rain"], size=200, p=[0.5, 0.3, 0.2])
quality = {"clear": 0.85, "rainy": 0.7, "night_rain": 0.5}   # night-rain is worst
iou = np.clip([rng.normal(quality[w], 0.08) for w in weather], 0, 1)

def worst_group_miou(iou, groups, macro_w=0.25):
    per = {g: float(iou[groups == g].mean()) for g in sorted(set(groups))}
    worst = min(per.values()); macro = float(np.mean(list(per.values())))
    return {"score": round(worst + macro_w * macro, 4), "per_group": {k: round(v,3) for k,v in per.items()},
            "worst": min(per, key=per.get)}
print(worst_group_miou(iou, weather))
"""
    ),
    md(
        """
Swap `worst_group_accuracy` for `worst_group_mIoU` (segmentation) or
`worst_group_mAP` (detection) and the rest of the harness — budget, mining, loop,
guardrails — is unchanged. The safety-critical worst slice (here night+rain) is
exactly what average mIoU would hide, which is why the objective targets it.
"""
    ),
]

# --------------------------------------------------------------------------- #
NOTEBOOKS["19_cost_and_scheduling"] = [
    md(
        """
# 19 · Cost, compute budgets & scheduling on a cluster

At scale the budget is money. We run the loop, read the `seconds` column, apply a
GPU-hour cost model, and find the point of diminishing returns — the information
that tells you when to stop paying.
"""
    ),
    BOOT,
    code(
        """
from harness.loop import run_loop
run_loop(iters=8, dataset_path=str(DATA), results_path="results.tsv", verbose=False)
import pandas as pd
df = pd.read_csv("results.tsv", sep="\\t")
GPU_USD_PER_HR = 2.50
df["cum_seconds"] = df["seconds"].cumsum()
df["cum_usd"] = df["cum_seconds"] / 3600 * GPU_USD_PER_HR
df[["iter", "score", "best_so_far", "seconds", "cum_usd"]].round(3)
"""
    ),
    code(
        """
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(df["cum_usd"], df["best_so_far"], "-o", color="#2c3e50")
ax.set_xlabel("cumulative spend (USD, modeled)"); ax.set_ylabel("best score")
ax.set_title("Diminishing returns: where to stop the run")
fig.tight_layout(); fig.savefig("fig_19_cost.png", dpi=90); plt.close(fig)
gain = df["best_so_far"].iloc[-1] - df["best_so_far"].iloc[0]
print(f"score gain {gain:.3f} for ${df['cum_usd'].iloc[-1]:.4f} (modeled)")
print("policy: stop when marginal score / $ falls below a program.md threshold")
from IPython.display import Image; Image("fig_19_cost.png")
"""
    ),
    md(
        """
A run budget belongs in `program.md` so it's part of the immutable direction.
Ray's scheduler enforces it as infra: spot instances for fault-tolerant trials
(you checkpoint, Ch. 10), resource-aware placement, and early-stopping doomed
trials (ASHA) as cost control.
"""
    ),
]

# --------------------------------------------------------------------------- #
NOTEBOOKS["20_case_study_bdd"] = [
    md(
        """
# 20 · Running it for real on BDD100K: a case study

Going from synthetic to real BDD100K touches **only the ingest path**
(`scripts/download_bdd.py --real`); the harness is unchanged. Here we run the
full loop on the tiny tier as a faithful stand-in and produce the run report you'd
read after a real run.
"""
    ),
    BOOT,
    code(
        """
from harness.loop import run_loop
out = run_loop(iters=10, dataset_path=str(DATA), results_path="results.tsv", verbose=False)
import pandas as pd
df = pd.read_csv("results.tsv", sep="\\t")
kept = df[df["kept"].astype(str) == "True"]
print("KEPT CHANGES (the research narrative):")
for _, r in kept.iterrows():
    print(f"  iter {int(r['iter']):2d}: score {r['score']:.3f}  :: {r['rationale']}")
print(f"\\nbaseline {df.iloc[0]['score']:.3f} -> best {df['best_so_far'].max():.3f}")
"""
    ),
    code(
        """
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
a.plot(df["iter"], df["best_so_far"], "-o", color="#27ae60"); a.set_title("best score")
a.set_xlabel("iteration")
a.scatter(df["iter"], df["score"], c=["#27ae60" if k=="True" else "#c0392b" for k in df["kept"].astype(str)], zorder=3)
b.plot(df["iter"], df["worst_group"], "-o", color="#8e44ad", label="worst-group")
b.plot(df["iter"], df["overall_accuracy"], "--", color="#7f8c8d", label="overall")
b.set_title("worst slice closes the gap"); b.set_xlabel("iteration"); b.legend()
fig.tight_layout(); fig.savefig("fig_20_case.png", dpi=90); plt.close(fig)
from IPython.display import Image; Image("fig_20_case.png")
"""
    ),
    md(
        """
The product isn't a magic model — it's a **legible, bounded research process**:
every gain traced to a kept change with a rationale, every regression reverted,
the scorer provably untouched, the spend capped. Point the same contract at a new
task, dataset, or worst slice, and turn the loop.

**To run it for real:** register at bdd-data.berkeley.edu, implement the loader in
`scripts/download_bdd.py --real`, drive the loop with `ClaudeProposer` (Ch. 14) on
a Ray cluster (Ch. 9). Nothing else changes.
"""
    ),
]


def build_one(name: str, cells: list, execute: bool) -> bool:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(s) if k == "md" else new_code_cell(s)
                for k, s in cells]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    out = HERE / f"{name}.ipynb"
    ok = True
    if execute:
        from nbclient import NotebookClient
        try:
            NotebookClient(nb, timeout=900, kernel_name="python3",
                           resources={"metadata": {"path": str(HERE)}}).execute()
            print(f"  executed {name}")
        except Exception as e:
            ok = False
            print(f"  !! FAILED {name}: {type(e).__name__}: {str(e)[:160]}")
    nbformat.write(nb, out)
    print(f"  wrote {out.relative_to(REPO)}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default=None)
    ap.add_argument("--no-exec", action="store_true")
    args = ap.parse_args()
    failed = []
    for name, cells in NOTEBOOKS.items():
        if args.which and not name.startswith(args.which):
            continue
        print(f"building {name} ...")
        if not build_one(name, cells, execute=not args.no_exec):
            failed.append(name)
    print("\\nSUMMARY:", "all OK" if not failed else f"FAILED: {failed}")


if __name__ == "__main__":
    main()
