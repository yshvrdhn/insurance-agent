#!/usr/bin/env python
"""Fetch (or synthesize) a BDD100K subset into a Lance dataset.

The book is built around BDD100K, but the real dataset needs a (free) Berkeley
DeepDrive registration and a multi-hundred-megabyte download, which makes the
notebooks impossible to run in CI or on a fresh laptop. So by default this
script *synthesizes* a BDD-shaped dataset with the same schema and the same
``weather`` / ``timeofday`` / ``scene`` attributes (see ``harness/data.py``).
Everything in the book runs identically on it.

When you are ready for the real thing, register at https://bdd-data.berkeley.edu,
download the images + labels, and re-run with ``--real --src /path/to/bdd`` (the
real-ingest path is left as a clearly marked TODO so you can wire in your own
loader without touching the synthetic contract).

Usage::

    python scripts/download_bdd.py --tier tiny
    python scripts/download_bdd.py --tier small --out data/bdd-small.lance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# make `harness` importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.data import DatasetSpec, make_synthetic_dataset  # noqa: E402

# rows per tier (keyframes). tiny runs anywhere; small is laptop-OK; full wants
# a cluster but is still generated synthetically here for reproducibility.
TIER_SIZES = {"tiny": 3000, "small": 12000, "full": 60000}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=sorted(TIER_SIZES), default="tiny")
    ap.add_argument("--out", default=None, help="output .lance path")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--real", action="store_true",
                    help="ingest a real BDD100K download instead of synthesizing")
    ap.add_argument("--src", default=None, help="path to a real BDD100K download")
    args = ap.parse_args()

    out = Path(args.out or f"data/bdd-{args.tier}.lance")

    if args.real:
        raise SystemExit(
            "Real BDD ingest is not wired up in this template. Point --src at "
            "your BDD100K download and implement the loader here: read the image "
            "frames + the `attributes` field of the label JSON (weather, "
            "timeofday, scene), then call harness.data.write_lance(...) with the "
            "same schema. Until then, drop --real to use the synthetic dataset."
        )

    n = TIER_SIZES[args.tier]
    print(f"synthesizing bdd-{args.tier}: {n} frames -> {out}")
    make_synthetic_dataset(out, DatasetSpec(n=n, seed=args.seed))
    print(f"done. load splits with: harness.data.load_split({str(out)!r}, 'train')")


if __name__ == "__main__":
    main()
