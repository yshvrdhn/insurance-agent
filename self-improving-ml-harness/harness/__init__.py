"""Self-Improving ML Harness.

An LLM-driven AutoResearch loop around a small computer-vision workload.

The package is split along the three-file contract described in ``program.md``:

* **immutable**     : :mod:`harness.data`, :mod:`harness.evaluator`,
                      :mod:`harness.loop`, :mod:`harness.config`,
                      :mod:`harness.registry`
* **agent-modifiable**: :mod:`harness.train`, :mod:`harness.model`,
                      :mod:`harness.mining`, ``configs/*.yaml``

Importing this package is cheap; heavy optional deps (torch, ray, fiftyone)
are imported lazily inside the modules that need them.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Files the loop must protect. Paths are relative to the repo root.
IMMUTABLE_FILES = (
    "program.md",
    "harness/evaluator.py",
    "harness/data.py",
    "harness/loop.py",
    "harness/config.py",
    "harness/registry.py",
)

MODIFIABLE_FILES = (
    "harness/train.py",
    "harness/model.py",
    "harness/mining.py",
)

__all__ = ["__version__", "IMMUTABLE_FILES", "MODIFIABLE_FILES"]
