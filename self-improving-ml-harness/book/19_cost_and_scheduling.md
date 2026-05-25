# Chapter 19 · Cost, compute budgets & scheduling on a cluster

> Companion notebook: [`notebooks/19_cost_and_scheduling.ipynb`](../notebooks/19_cost_and_scheduling.ipynb)
> Tier: `bdd-full` · **runs here on the tiny tier** (cost model over a real loop run)

## The budget is the whole game

On the tiny tier the budget is a few seconds of CPU. On a cluster it's real money
— GPU-hours, sometimes thousands of iterations. A self-improving loop that
ignores cost will happily spend $10k to find a 0.005 improvement. So the budget
graduates from a training cap to a **portfolio allocation problem**.

## Three budget layers

1. **Per-iteration budget** — wall-clock per training run (Chapter 10). Caps the
   cost of any single experiment.
2. **Per-iteration *value*** — score gained per GPU-hour. The loop should prefer
   cheap experiments with good expected value; ASHA (Chapter 11) already does
   this within a search, and the agent proposer can be prompted to weigh it.
3. **Run budget** — total spend before the loop stops. When the marginal score
   per dollar drops below a threshold, the run is done. This belongs in
   `program.md` so it's part of the immutable direction.

## Scheduling on Ray

Ray's scheduler and autoscaler turn the allocation policy into infrastructure:

- **Spot instances** for fault-tolerant trials (you checkpoint, Chapter 10), with
  on-demand fallback for the driver/loop.
- **Resource-aware placement** — small proposer-evaluation trials on cheap nodes,
  full training on GPU nodes.
- **Early stopping as cost control** — killing a doomed trial isn't just faster,
  it's cheaper, and ASHA makes it automatic.

## Reporting cost alongside score

Add `seconds` (already in `results.tsv`) and a derived `$ / Δscore` to the run
summary. A good run report shows not just "best score 1.05" but "best score 1.05
at $42, with diminishing returns after iteration 30" — the information a human
needs to decide whether to keep paying.

## What you built

- A three-layer budget model: per-iteration, value-per-dollar, and total run.
- Ray scheduling patterns (spot, resource-aware placement, early stop) that
  enforce the budget as infrastructure.
- Cost-aware run reporting so the loop's spend is legible and bounded.
