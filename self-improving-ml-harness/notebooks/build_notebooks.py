#!/usr/bin/env python
"""Generate the tiny-tier companion notebooks (Ch. 1-7) and execute them.

Each notebook is defined here as a list of (kind, source) cells so the book's
executable companions are reproducible: re-running this script rebuilds every
notebook from a single source of truth and re-executes it end to end.

    python notebooks/build_notebooks.py            # build + execute all
    python notebooks/build_notebooks.py --no-exec  # build only (fast)
    python notebooks/build_notebooks.py 05         # just one notebook

Notebooks 8-20 (the small/full scaling tiers) are written as prose chapters in
book/ but are not executed here -- they need Ray / FiftyOne / a cluster.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def md(text: str):
    return ("md", text.strip("\n"))


def code(text: str):
    return ("code", text.strip("\n"))


# A bootstrap cell every notebook runs first: put the repo on the path, use a
# headless matplotlib backend, and build a small dataset if one isn't present.
BOOT = code(
    """
import sys, os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd().parent))   # make `harness` importable
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

from harness.data import make_synthetic_dataset, DatasetSpec, load_split, class_names

DATA = Path("../data/bdd-tiny.lance")
if not DATA.exists():
    print("building a small bdd-tiny ...")
    make_synthetic_dataset(DATA, DatasetSpec(n=3000, seed=7))
print("dataset:", DATA)
"""
)


NOTEBOOKS: dict[str, list] = {}

# --------------------------------------------------------------------------- #
# NB01 — the evaluator & the baseline (slice-based eval, à la FiftyOne)
# --------------------------------------------------------------------------- #
NOTEBOOKS["01_fiftyone_evaluator"] = [
    md(
        """
# 1 · The immutable evaluator & your first baseline

This is the heart of the whole book: a **scoring function the agent can never
touch**. Everything the harness does is judged by it.

We:
1. load the BDD-style dataset (synthetic, but same schema/attributes as BDD100K),
2. train a quick baseline CNN,
3. score it with `harness.evaluator` — and discover the **worst slice**.

The score is `worst_group_accuracy + 0.25 * macro_accuracy`, where groups are
the `weather` slices. We optimize the *worst* slice on purpose: a perception
model that is blind in fog is not safe, no matter how good its average is.
"""
    ),
    BOOT,
    md("### A look at the data — driving frames tagged with weather / timeofday"),
    code(
        """
x, y, meta = load_split(DATA, "train")
names = class_names()
fig, axes = plt.subplots(2, 5, figsize=(11, 4.6))
for ax, i in zip(axes.ravel(), np.random.default_rng(0).choice(len(x), 10, replace=False)):
    ax.imshow(np.clip(x[i], 0, 1)); ax.axis("off")
    ax.set_title(f"{names[y[i]]}\\n{meta['weather'][i]}", fontsize=8)
fig.suptitle("bdd-tiny: time-of-day is the label, weather defines the slices")
fig.tight_layout(); fig.savefig("fig_01_frames.png", dpi=90); plt.close(fig)
from IPython.display import Image; Image("fig_01_frames.png")
"""
    ),
    md("### Train a quick baseline under the compute budget"),
    code(
        """
from harness.config import load_config
from harness.train import train

cfg = load_config("base", {"budget": {"max_epochs": 12, "max_seconds": 60}})
model = train(cfg, DATA)
print("validation accuracy:", model.metrics["val_acc"],
      "| epochs:", model.metrics["epochs_run"],
      "| seconds:", model.metrics["seconds"])
"""
    ),
    md("### Score it with the immutable evaluator"),
    code(
        """
from harness.evaluator import evaluate
res = evaluate(model, DATA)
print("OFFICIAL SCORE :", res["score"])
print("worst slice    :", res["worst_group_name"], "=", res["worst_group"])
print("overall acc    :", res["overall_accuracy"])
print("macro acc      :", res["macro_accuracy"])
res["per_weather"]
"""
    ),
    md("### The worst slice, visualized"),
    code(
        """
pw = res["per_weather"]
order = sorted(pw, key=pw.get)
fig, ax = plt.subplots(figsize=(6, 3.2))
bars = ax.bar(order, [pw[w] for w in order],
              color=["#c0392b" if w == res["worst_group_name"] else "#2980b9" for w in order])
ax.set_ylim(0, 1); ax.set_ylabel("accuracy"); ax.set_title("Per-weather accuracy (worst slice in red)")
ax.axhline(res["overall_accuracy"], ls="--", c="gray", label="overall")
ax.legend(); fig.tight_layout(); fig.savefig("fig_01_slices.png", dpi=90); plt.close(fig)
from IPython.display import Image; Image("fig_01_slices.png")
"""
    ),
    md(
        """
**Read the result.** The baseline is strong on the common, well-lit weather but
weakest on **fog** — fog is rare in the data and its cues are washed out. The
whole job of the harness, in the chapters that follow, is to raise that worst
slice *without ever editing this evaluator*.
"""
    ),
]

# --------------------------------------------------------------------------- #
# NB02 — the dataset in Lance & Daft
# --------------------------------------------------------------------------- #
NOTEBOOKS["02_data_lance_daft"] = [
    md(
        """
# 2 · The dataset: BDD-style frames in Lance, queried with Daft

The dataset lives in **Lance** — a columnar, zero-copy, versioned format built
for ML. Each row is one frame: an image plus its `weather` / `timeofday` /
`scene` attributes and its `split`. We query it with **Daft**, a distributed
multimodal dataframe, exactly as we would the real BDD100K.
"""
    ),
    BOOT,
    md("### The Lance dataset and its schema"),
    code(
        """
import lance
ds = lance.dataset(str(DATA))
print("rows:", ds.count_rows())
print(ds.schema)
"""
    ),
    md("### Query attributes with Daft"),
    code(
        """
import daft
# read straight from the Lance table into a Daft dataframe
df = daft.from_arrow(lance.dataset(str(DATA)).to_table().drop(["image"]))
counts = (df.groupby("split", "weather").agg(daft.col("id").count().alias("n"))
            .sort(["split", "weather"]))
counts.show(20)
"""
    ),
    md("### Class balance and the rare fog slice"),
    code(
        """
import collections
_, y, meta = load_split(DATA, "train")
wc = collections.Counter(meta["weather"])
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
a.bar(list(class_names()), np.bincount(y)); a.set_title("time-of-day (label) balance")
ws = sorted(wc, key=wc.get)
b.bar(ws, [wc[w] for w in ws], color="#16a085"); b.set_title("weather balance — fog is rarest")
fig.tight_layout(); fig.savefig("fig_02_balance.png", dpi=90); plt.close(fig)
from IPython.display import Image; Image("fig_02_balance.png")
"""
    ),
    md("### Why fog is hard: brightness is corrupted, only weak colour survives"),
    code(
        """
x, y, meta = load_split(DATA, "train")
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
for ax, cls in zip(axes, range(3)):
    for w, c in [("clear", "#2980b9"), ("foggy", "#c0392b")]:
        sel = (y == cls) & (meta["weather"] == w)
        ax.hist(x[sel].mean(axis=(1, 2, 3)), bins=20, alpha=0.6, label=w, color=c)
    ax.set_title(class_names()[cls]); ax.set_xlabel("mean brightness")
axes[0].legend()
fig.suptitle("Per-image brightness: clean & separable for clear, scrambled for fog")
fig.tight_layout(); fig.savefig("fig_02_bright.png", dpi=90); plt.close(fig)
from IPython.display import Image; Image("fig_02_bright.png")
"""
    ),
    md(
        """
Brightness cleanly separates time-of-day under **clear** weather, but under
**fog** the haze scrambles it — the histograms overlap. The only surviving cue
is a faint colour tint, which the next chapters teach the model to use.
"""
    ),
]

# --------------------------------------------------------------------------- #
# NB03 — the evaluator deep-dive
# --------------------------------------------------------------------------- #
NOTEBOOKS["03_evaluator_contract"] = [
    md(
        """
# 3 · Inside the immutable evaluator

The evaluator is the contract. This chapter opens it up:

* the **official score** and why it is worst-group + a macro tie-breaker,
* the **fingerprint** the loop uses to prove the evaluator wasn't tampered with,
* the diagnostics it returns (per-slice accuracy, confusion matrix) that *inform*
  the agent but never feed the scalar score.
"""
    ),
    BOOT,
    code(
        """
from harness.config import load_config
from harness.train import train
from harness.evaluator import evaluate, evaluator_fingerprint, MACRO_WEIGHT
model = train(load_config("base"), DATA)
res = evaluate(model, DATA)
print("score formula: worst_group + %.2f * macro" % MACRO_WEIGHT)
print("  %.4f = %.4f + %.2f * %.4f" % (res["score"], res["worst_group"], MACRO_WEIGHT, res["macro_accuracy"]))
"""
    ),
    md("### The tamper-proof fingerprint"),
    code(
        """
fp = evaluator_fingerprint()
print("evaluator sha256:", fp)
# the loop records this every iteration; if evaluator.py changes, the hash changes
assert fp == evaluator_fingerprint()
print("stable across calls:", True)
"""
    ),
    md("### Confusion matrix — *which* mistakes is the model making?"),
    code(
        """
import numpy as np
cm = np.array(res["confusion"]); names = list(class_names())
fig, ax = plt.subplots(figsize=(4.2, 3.8))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_xticklabels(names); ax.set_yticks(range(3)); ax.set_yticklabels(names)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black")
fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig("fig_03_cm.png", dpi=90); plt.close(fig)
from IPython.display import Image; Image("fig_03_cm.png")
"""
    ),
    md(
        """
**The rule.** `evaluator.py` and `data.py` are in the *immutable set*. The agent
may read these diagnostics to form a hypothesis, but the moment training code
reads the `test` split — or anything edits the evaluator — the run is void. The
loop enforces this with the fingerprint above (Chapter 6).
"""
    ),
]

# --------------------------------------------------------------------------- #
# NB04 — the training surface
# --------------------------------------------------------------------------- #
NOTEBOOKS["04_training_surface"] = [
    md(
        """
# 4 · The training surface

`harness/train.py` and `harness/model.py` are **agent-modifiable**. This chapter
tours the knobs the agent turns: model capacity, the optimizer/schedule,
augmentation, and the **compute budget** that makes every change comparable.
"""
    ),
    BOOT,
    md("### Capacity vs. budget — bigger isn't free"),
    code(
        """
import time
from harness.config import load_config
from harness.train import train
from harness.evaluator import evaluate

rows = []
for width in [8, 16, 32]:
    cfg = load_config("base", {"model": {"width": width}, "budget": {"max_epochs": 12}})
    t = time.time(); m = train(cfg, DATA); dt = time.time() - t
    r = evaluate(m, DATA)
    rows.append((width, r["score"], r["worst_group"], r["overall_accuracy"], round(dt, 1)))
    print(f"width={width:2d}  score={r['score']:.3f}  fog={r['per_weather']['foggy']:.3f}  overall={r['overall_accuracy']:.3f}  {dt:.1f}s")
"""
    ),
    md("### The budget caps training — a change that needs more compute isn't valid"),
    code(
        """
for me in [2, 6, 12]:
    cfg = load_config("base", {"budget": {"max_epochs": me, "max_seconds": 60}})
    m = train(cfg, DATA)
    print(f"max_epochs={me:2d} -> ran {m.metrics['epochs_run']} epochs, val={m.metrics['val_acc']:.3f}, {m.metrics['seconds']:.1f}s")
"""
    ),
    md("### Augmentation knobs (generic robustness)"),
    code(
        """
for aug in [{}, {"hflip": True}, {"brightness_jitter": 0.3}]:
    cfg = load_config("base", {"augment": aug})
    r = evaluate(train(cfg, DATA), DATA)
    print(f"aug={str(aug):28s} score={r['score']:.3f} fog={r['per_weather']['foggy']:.3f}")
"""
    ),
    md(
        """
Capacity and epochs help a little, but notice the **foggy** slice barely moves —
generic knobs don't fix a rare, degraded slice. That takes targeted
**hard-example mining**, which is Chapter 5.
"""
    ),
]

# --------------------------------------------------------------------------- #
# NB05 — mining with LanceDB
# --------------------------------------------------------------------------- #
NOTEBOOKS["05_mining_lancedb"] = [
    md(
        """
# 5 · Hard-example mining with LanceDB

The worst slice is rare *foggy* frames. Generic training underfits them. Here we
fix it surgically: embed every frame, index the embeddings in **LanceDB**, use
vector search to gather the fog-like neighbourhood, and **up-weight** it during
training. This is the single biggest lever on the worst-group score.
"""
    ),
    BOOT,
    md("### A brightness-invariant colour embedding"),
    code(
        """
from harness.mining import embed
x, y, meta = load_split(DATA, "train")
feats = embed(x)
print("embedding shape:", feats.shape, "(R,G,B colour ratio + contrast)")
"""
    ),
    md("### Vector search: nearest neighbours of a fog frame are fog-like frames"),
    code(
        """
from harness.mining import compute_sample_weights
from harness.config import load_config
cfg = load_config("base", {"mining": {"enabled": True, "strategy": "fog_knn", "boost": 6.0, "k": 8}})
w = compute_sample_weights(DATA, cfg)
print("rows up-weighted:", int((w > 1).sum()), "of", len(w))
print("fog frames in train:", int((meta['weather'] == 'foggy').sum()))
print("-> mining expanded the fog set via LanceDB neighbours")
"""
    ),
    md("### Before vs. after: train with and without fog mining"),
    code(
        """
from harness.train import train
from harness.evaluator import evaluate
base = evaluate(train(load_config("base"), DATA), DATA)
mined = evaluate(train(cfg, DATA, sample_weights=w), DATA)
print(f"baseline : score={base['score']:.3f}  fog={base['per_weather']['foggy']:.3f}")
print(f"+mining  : score={mined['score']:.3f}  fog={mined['per_weather']['foggy']:.3f}")
"""
    ),
    code(
        """
ws = sorted(base["per_weather"], key=base["per_weather"].get)
xpos = np.arange(len(ws)); fig, ax = plt.subplots(figsize=(6.5, 3.3))
ax.bar(xpos - 0.2, [base["per_weather"][k] for k in ws], 0.4, label="baseline", color="#95a5a6")
ax.bar(xpos + 0.2, [mined["per_weather"][k] for k in ws], 0.4, label="+ fog mining", color="#27ae60")
ax.set_xticks(xpos); ax.set_xticklabels(ws); ax.set_ylim(0, 1); ax.set_ylabel("accuracy")
ax.set_title("Mining lifts the fog slice"); ax.legend()
fig.tight_layout(); fig.savefig("fig_05_mining.png", dpi=90); plt.close(fig)
from IPython.display import Image; Image("fig_05_mining.png")
"""
    ),
    md(
        """
Mining moves the slice that matters. But be careful: over-boosting fog can starve
another slice and make *it* the new worst group — which is exactly the kind of
trade-off the keep/revert loop is built to catch (Chapter 6).
"""
    ),
]

# --------------------------------------------------------------------------- #
# NB06 — the keep/revert loop
# --------------------------------------------------------------------------- #
NOTEBOOKS["06_autoresearch_loop"] = [
    md(
        """
# 6 · The keep/revert loop

Now we close the loop. `harness/loop.py` proposes a change, trains under budget,
scores with the immutable evaluator, and **keeps the change only if the score
improved** — otherwise it reverts. Every attempt, kept or not, is appended to
`results.tsv`. The built-in `SearchProposer` stands in for the LLM agent.
"""
    ),
    BOOT,
    code(
        """
from harness.loop import run_loop
out = run_loop(iters=8, dataset_path=str(DATA), results_path="results.tsv", verbose=True)
print("\\nbest score:", round(out["best_score"], 4))
"""
    ),
    md("### The score trajectory — and the immutability guard that protects it"),
    code(
        """
import csv
rows = list(csv.DictReader(open("results.tsv"), delimiter="\\t"))
its = [int(r["iter"]) for r in rows]
score = [float(r["score"]) for r in rows]
best = [float(r["best_so_far"]) for r in rows]
kept = [r["kept"] == "True" for r in rows]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(its, best, "-o", color="#27ae60", label="best so far")
ax.scatter([i for i, k in zip(its, kept) if k], [s for s, k in zip(score, kept) if k],
           color="#27ae60", zorder=3, label="kept")
ax.scatter([i for i, k in zip(its, kept) if not k], [s for s, k in zip(score, kept) if not k],
           color="#c0392b", marker="x", zorder=3, label="reverted")
ax.set_xlabel("iteration"); ax.set_ylabel("official score"); ax.set_title("AutoResearch loop")
ax.legend(); fig.tight_layout(); fig.savefig("fig_06_loop.png", dpi=90); plt.close(fig)
from IPython.display import Image; Image("fig_06_loop.png")
"""
    ),
    md("### The guard in action — you cannot win by editing the ruler"),
    code(
        """
from harness.loop import snapshot_immutable, verify_immutable
snap = snapshot_immutable()
verify_immutable(snap)                      # clean: passes
tampered = dict(snap); tampered["harness/evaluator.py"] = "0" * 64
try:
    verify_immutable(tampered)
except RuntimeError as e:
    print("REJECTED:", e)
"""
    ),
    md(
        """
The green line never goes down: a change is only kept when it beats the best.
The red X's are reverted experiments — dead ends the agent tried and abandoned.
And if any protected file changes mid-iteration, the score is thrown out.
"""
    ),
]

# --------------------------------------------------------------------------- #
# NB07 — reading a run
# --------------------------------------------------------------------------- #
NOTEBOOKS["07_reading_a_run"] = [
    md(
        """
# 7 · Reading a run

`results.tsv` is the loop's memory and your audit trail. This chapter shows how
to read one: which changes were kept, how the worst slice moved, and how to spot
the moment a different slice became the bottleneck.
"""
    ),
    BOOT,
    code(
        """
from harness.loop import run_loop
run_loop(iters=8, dataset_path=str(DATA), results_path="results.tsv", verbose=False)
import pandas as pd
df = pd.read_csv("results.tsv", sep="\\t")
df[["iter", "score", "best_so_far", "kept", "worst_group", "worst_group_name", "rationale"]]
"""
    ),
    md("### The story of the run"),
    code(
        """
kept = df[df["kept"].astype(str) == "True"]
print("kept changes:")
for _, r in kept.iterrows():
    print(f"  iter {int(r['iter'])}: score {r['score']:.3f}  ({r['rationale']})")
print(f"\\nbaseline {df.iloc[0]['score']:.3f} -> best {df['best_so_far'].max():.3f}")
print("worst slice was:", df["worst_group_name"].value_counts().to_dict())
"""
    ),
    code(
        """
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(df["iter"], df["worst_group"], "-o", label="worst-group acc", color="#8e44ad")
ax.plot(df["iter"], df["overall_accuracy"], "--", label="overall acc", color="#7f8c8d")
ax.set_xlabel("iteration"); ax.set_ylabel("accuracy"); ax.set_title("What the loop optimized")
ax.legend(); fig.tight_layout(); fig.savefig("fig_07_run.png", dpi=90); plt.close(fig)
from IPython.display import Image; Image("fig_07_run.png")
"""
    ),
    md(
        """
That's the tiny tier end to end: a fixed task, an immutable scorer, a budgeted
training surface, mining, and a loop that keeps only what helps. From here the
book scales the *same contract* to bigger data and real distributed infra —
Daft pipelines, Ray Train, Ray Tune, FiftyOne, and finally an LLM in the loop
(Chapters 8-20).
"""
    ),
]


def build_one(name: str, cells: list, execute: bool) -> None:
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(src) if kind == "md" else new_code_cell(src)
        for kind, src in cells
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    out = HERE / f"{name}.ipynb"
    if execute:
        from nbclient import NotebookClient

        client = NotebookClient(nb, timeout=600, kernel_name="python3",
                                resources={"metadata": {"path": str(HERE)}})
        client.execute()
        print(f"  executed {name} ({len(nb.cells)} cells)")
    nbformat.write(nb, out)
    print(f"  wrote {out.relative_to(REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?", default=None,
                    help="build only the notebook whose name starts with this")
    ap.add_argument("--no-exec", action="store_true")
    args = ap.parse_args()
    for name, cells in NOTEBOOKS.items():
        if args.which and not name.startswith(args.which):
            continue
        print(f"building {name} ...")
        build_one(name, cells, execute=not args.no_exec)


if __name__ == "__main__":
    main()
