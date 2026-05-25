"""Config loading and merging.

**Immutable plumbing.** The *values* live in ``configs/*.yaml`` (which the agent
may edit); this module just loads and merges them. Keeping the loader immutable
means the agent cannot change what a config *means*, only what it *says*.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(name: str = "base", overrides: dict | None = None) -> dict:
    """Load ``configs/<name>.yaml``, following an optional ``extends:`` chain.

    A config may declare ``extends: base`` to inherit from another file; the
    parent is loaded first and the child merged on top. ``overrides`` (a plain
    dict, e.g. a single hyperparameter the loop is trying) is merged last.
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open() as fh:
        cfg = yaml.safe_load(fh) or {}

    parent_name = cfg.pop("extends", None)
    if parent_name:
        parent = load_config(parent_name)
        cfg = _deep_merge(parent, cfg)

    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def flatten(cfg: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config to ``{"a.b.c": value}`` — handy for results.tsv."""
    out: dict[str, Any] = {}
    for key, val in cfg.items():
        full = f"{prefix}{key}"
        if isinstance(val, dict):
            out.update(flatten(val, prefix=f"{full}."))
        else:
            out[full] = val
    return out
