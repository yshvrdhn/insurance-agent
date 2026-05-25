#!/usr/bin/env bash
# Run the autonomous AutoResearch loop on the tiny tier.
#
#   bash scripts/run_loop.sh            # 8 iterations on bdd-tiny
#   ITERS=20 TIER=small bash scripts/run_loop.sh
#
# This uses the built-in SearchProposer (a stand-in for the LLM agent). To drive
# the loop with Claude/Codex instead, pass your own proposer to
# harness.loop.run_loop (see book/14_agent_in_the_loop.md).
set -euo pipefail

cd "$(dirname "$0")/.."

ITERS="${ITERS:-8}"
TIER="${TIER:-tiny}"
DATASET="data/bdd-${TIER}.lance"
PY="${PYTHON:-python}"

if [ ! -d "$DATASET" ]; then
  echo "dataset $DATASET not found; building it..."
  "$PY" scripts/download_bdd.py --tier "$TIER"
fi

echo "running $ITERS iterations on $DATASET"
"$PY" -m harness.loop --iters "$ITERS" --dataset "$DATASET" --results results.tsv

echo
echo "results.tsv:"
column -t -s $'\t' results.tsv
