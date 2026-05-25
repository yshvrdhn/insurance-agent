"""The results ledger — **immutable plumbing**.

Every attempt the loop makes is appended to ``results.tsv`` as one row. This is
the memory the agent reads at the start of each iteration (``program.md`` step 1)
and the audit trail a human reads afterwards. It is intentionally a plain TSV:
greppable, diffable, and trivially loaded into pandas/Daft.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

FIELDS = [
    "iter",
    "timestamp",
    "action",
    "score",
    "best_so_far",
    "kept",
    "worst_group",
    "worst_group_name",
    "overall_accuracy",
    "macro_accuracy",
    "seconds",
    "rationale",
]


@dataclass
class Attempt:
    iter: int
    action: str
    score: float
    best_so_far: float
    kept: bool
    worst_group: float
    worst_group_name: str
    overall_accuracy: float
    macro_accuracy: float
    seconds: float
    rationale: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def as_row(self) -> dict:
        d = asdict(self)
        return {k: d[k] for k in FIELDS}


def append(path: str | Path, attempt: Attempt) -> None:
    path = Path(path)
    new = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t")
        if new:
            writer.writeheader()
        writer.writerow(attempt.as_row())


def read(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def best_score(path: str | Path) -> float:
    rows = read(path)
    if not rows:
        return float("-inf")
    return max(float(r["score"]) for r in rows)
