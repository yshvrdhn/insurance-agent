"""The AutoResearch loop — **immutable plumbing**.

Implements the keep/revert loop from ``program.md``:

    read history -> propose a change -> train under budget -> score with the
    immutable evaluator -> keep if it beats the best, else revert -> log.

Two things make the loop trustworthy:

* **Immutability guard.** Before recording any score it re-hashes every file in
  :data:`harness.IMMUTABLE_FILES`. If the evaluator (or the task definition, or
  this loop) changed during the iteration, the score is rejected -- you cannot
  "win" by editing the ruler.
* **A pluggable proposer.** In a real run an LLM edits ``train.py`` /
  ``mining.py`` / ``configs`` between iterations. So the loop is runnable and
  testable on its own, the default :class:`SearchProposer` stands in for the
  agent: it walks a curated space of config moves (mining, capacity, schedule,
  regularization) with a greedy keep/revert policy. Swap in your own proposer --
  including one backed by Claude -- via the ``proposer`` argument.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
from pathlib import Path

from harness import IMMUTABLE_FILES
from harness import registry
from harness.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_immutable() -> dict[str, str]:
    """Hash every protected file. Call before an iteration."""
    snap = {}
    for rel in IMMUTABLE_FILES:
        p = REPO_ROOT / rel
        snap[rel] = _hash_file(p) if p.exists() else "<missing>"
    return snap


def verify_immutable(snapshot: dict[str, str]) -> None:
    """Raise if any protected file changed since ``snapshot`` was taken."""
    current = snapshot_immutable()
    changed = [k for k in snapshot if current.get(k) != snapshot[k]]
    if changed:
        raise RuntimeError(
            "immutable files changed during the iteration; score rejected: "
            + ", ".join(changed)
        )


class SearchProposer:
    """A stand-in for the LLM agent: a greedy walk over config moves.

    Each move is ``(override_dict, rationale)``; ``override_dict`` is deep-merged
    onto the current best config. The list is ordered roughly by expected payoff
    so the demo loop tells a coherent story, but a ``seed`` shuffles ties.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.moves = [
            ({"mining": {"enabled": True, "strategy": "fog_boost", "boost": 5.0}},
             "Fog is the worst, rarest slice -> upweight fog frames 5x."),
            ({"mining": {"enabled": True, "strategy": "fog_knn", "boost": 6.0, "k": 8}},
             "Expand fog upweighting to LanceDB nearest neighbours (fog-like frames)."),
            ({"model": {"width": 32}},
             "Give the net more capacity to fit the weak fog signal."),
            ({"budget": {"max_epochs": 24}},
             "Spend more of the budget on epochs to extract the weak fog cue."),
            ({"train": {"weight_decay": 1e-4}},
             "Add weight decay to denoise the high-variance fog frames."),
            ({"train": {"label_smoothing": 0.05}},
             "Label smoothing to stop overconfidence on noisy fog labels."),
            ({"augment": {"hflip": True}},
             "Horizontal flip: cheap, label-preserving augmentation."),
            ({"train": {"cosine_lr": True}},
             "Cosine LR decay for a cleaner final convergence."),
            ({"model": {"dropout": 0.1}},
             "Light dropout for regularization."),
        ]
        self.i = 0

    def propose(self, history: list[dict], best_config: dict):
        if self.i >= len(self.moves):
            # once the curated moves are exhausted, jitter the fog boost
            boost = round(self.rng.uniform(3.0, 9.0), 1)
            return ({"mining": {"enabled": True, "strategy": "fog_knn",
                                "boost": boost, "k": 8}},
                    f"Explore fog-knn boost={boost}.")
        move = self.moves[self.i]
        self.i += 1
        return move


def _evaluate_config(config: dict, dataset_path: str):
    """Train under budget and score. Imports are local so the loop module stays
    importable without the [torch] extra."""
    from harness.evaluator import evaluate
    from harness.mining import compute_sample_weights
    from harness.train import train

    weights = compute_sample_weights(dataset_path, config)
    model = train(config, dataset_path, sample_weights=weights)
    result = evaluate(model, dataset_path)
    result["seconds"] = model.metrics.get("seconds", 0.0)
    return result


def run_loop(iters: int = 8, dataset_path: str = "data/bdd-tiny.lance",
             results_path: str = "results.tsv", base_config: str = "base",
             proposer: SearchProposer | None = None, verbose: bool = True) -> dict:
    """Run ``iters`` keep/revert iterations. Returns the best config + score."""
    proposer = proposer or SearchProposer()
    best_config = load_config(base_config)
    Path(results_path).unlink(missing_ok=True)

    # iteration 0: the baseline
    snap = snapshot_immutable()
    base_result = _evaluate_config(best_config, dataset_path)
    verify_immutable(snap)
    best_score = base_result["score"]
    registry.append(results_path, registry.Attempt(
        iter=0, action="baseline", score=best_score, best_so_far=best_score,
        kept=True, worst_group=base_result["worst_group"],
        worst_group_name=base_result["worst_group_name"],
        overall_accuracy=base_result["overall_accuracy"],
        macro_accuracy=base_result["macro_accuracy"],
        seconds=base_result["seconds"], rationale="baseline config"))
    if verbose:
        print(f"[iter 0] baseline score={best_score:.3f} "
              f"worst={base_result['worst_group']:.3f}"
              f"({base_result['worst_group_name']})")

    from harness.config import _deep_merge

    for it in range(1, iters + 1):
        history = registry.read(results_path)
        override, rationale = proposer.propose(history, best_config)
        candidate = _deep_merge(best_config, override)

        snap = snapshot_immutable()
        result = _evaluate_config(candidate, dataset_path)
        verify_immutable(snap)

        kept = result["score"] > best_score
        if kept:
            best_config = candidate
            best_score = result["score"]
        registry.append(results_path, registry.Attempt(
            iter=it, action=str(override), score=result["score"],
            best_so_far=best_score, kept=kept,
            worst_group=result["worst_group"],
            worst_group_name=result["worst_group_name"],
            overall_accuracy=result["overall_accuracy"],
            macro_accuracy=result["macro_accuracy"],
            seconds=result["seconds"], rationale=rationale))
        if verbose:
            flag = "KEEP" if kept else "revert"
            print(f"[iter {it}] {flag} score={result['score']:.3f} "
                  f"(best={best_score:.3f}) worst={result['worst_group']:.3f}"
                  f"({result['worst_group_name']}) :: {rationale}")

    return {"best_config": best_config, "best_score": best_score}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AutoResearch loop.")
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--dataset", default="data/bdd-tiny.lance")
    parser.add_argument("--results", default="results.tsv")
    parser.add_argument("--base-config", default="base")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = run_loop(iters=args.iters, dataset_path=args.dataset,
                   results_path=args.results, base_config=args.base_config,
                   proposer=SearchProposer(seed=args.seed))
    print(f"\nbest score: {out['best_score']:.4f}")
    print(f"results written to: {args.results}")


if __name__ == "__main__":
    main()
