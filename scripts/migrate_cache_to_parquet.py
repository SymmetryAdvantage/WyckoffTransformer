#!/usr/bin/env python
"""Rewrite a dataset cache's ``data.pkl.gz`` as per-split Parquet files.

Why, and what the format is, is in ``wyckoff_transformer/dataset_cache.py``.
Nothing has to be migrated to keep working -- every reader falls back to the
pickle -- but a cache that has been converted loads a couple of orders of
magnitude faster and stops being a channel for arbitrary code.

    python scripts/migrate_cache_to_parquet.py lemat_bulk_fmax1_stress
    python scripts/migrate_cache_to_parquet.py --all

The pickle is **kept**, because a run in flight on another machine may still be
reading it and because nothing else in this repository can rebuild it.  Delete
it by hand once the caches on every machine that shares the store have been
converted; until then it is simply ignored.

Memory: one split is held decoded at a time, which is ~20 GB for LeMat-Bulk's
training split.  ``--splits`` converts fewer at once on a smaller machine.
"""
from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path

from wyckoff_transformer.dataset_cache import (
    LEGACY_CACHE_NAME,
    legacy_path,
    load_legacy_cache,
    provenance,
    resolve_cache,
    save_split,
    split_path,
)
from wyckoff_transformer.paths import cache_root

logger = logging.getLogger("migrate_cache_to_parquet")


def convert(cache: Path, splits=None, force: bool = False) -> int:
    """Convert one cache.  Returns how many splits were written."""
    cache = resolve_cache(cache)
    if not legacy_path(cache).is_file():
        logger.info("%s has no %s; nothing to convert", cache, LEGACY_CACHE_NAME)
        return 0
    started = time.time()
    # Once, not once per split: the pickle is a single stream, so reading it
    # three times would cost three times as much and save no memory.
    frames = load_legacy_cache(cache)
    logger.info("Read %s in %.0f s: %s", legacy_path(cache).name, time.time() - started,
                {split: len(frame) for split, frame in frames.items()})
    wanted = tuple(splits) if splits else tuple(frames)
    missing = [split for split in wanted if split not in frames]
    if missing:
        raise KeyError(f"{legacy_path(cache)} has no split(s) {missing}; "
                       f"it holds {sorted(frames)}")

    written = 0
    for split in wanted:
        target = split_path(cache, split)
        if target.is_file() and not force:
            logger.info("%s exists; skipped (--force to rewrite)", target)
            continue
        started = time.time()
        # The options the pickle was built with are not in it and not anywhere
        # else, so the record says what this conversion knows and no more.
        save_split(frames.pop(split), cache, split, build=provenance(
            "migrate_cache_to_parquet", converted_from=LEGACY_CACHE_NAME,
            original_build="unrecorded"))
        gc.collect()
        logger.info("Wrote %s in %.0f s", target.name, time.time() - started)
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="*", help="Dataset names, or cache directories.")
    parser.add_argument("--all", action="store_true",
                        help=f"Every cache under the cache root that still has a {LEGACY_CACHE_NAME}.")
    parser.add_argument("--splits", nargs="+", default=None,
                        help="Convert only these splits. Defaults to every split present.")
    parser.add_argument("--force", action="store_true",
                        help="Rewrite a split that has already been converted.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.all == bool(args.dataset):
        parser.error("Name the datasets to convert, or pass --all -- not both, not neither.")

    if args.all:
        caches = sorted(path.parent for path in cache_root().glob(f"*/{LEGACY_CACHE_NAME}"))
    else:
        caches = [Path(name) if "/" in name else cache_root() / name for name in args.dataset]

    total = 0
    for cache in caches:
        logger.info("=== %s", cache)
        total += convert(cache, args.splits, args.force)
    print(f"Converted {total} split(s). The pickles are kept; delete them once every "
          f"machine sharing this store has been converted.")


if __name__ == "__main__":
    main()
