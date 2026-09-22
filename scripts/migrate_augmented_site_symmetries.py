#!/usr/bin/env python
"""Add ``site_symmetries_augmented`` to a cached Wyckoff dataset, in place.

Why this exists is in ``docs/wyckoff_augmentation_audit.md``: an equivalent
description of a gene has to carry the oriented site-symmetry symbol alongside
the enumeration index, because in 26 orthorhombic space groups a relabelling
changes the symbol.  Caches written before that was understood carry only
``sites_enumeration_augmented``, and every fingerprint taken from one is wrong in
those space groups.

**Nothing expensive is recomputed.**  The augmentation is a pure function of
``(spacegroup_number, wyckoff_letters)``, and both are already columns, so this
never parses a CIF, never runs pyxtal symmetry detection, and never touches a
structure.  It rewrites two columns.

    python scripts/migrate_augmented_site_symmetries.py cache/lemat_bulk_fmax1_stress

``--output`` writes elsewhere instead of in place; ``--check`` reports what would
change and writes nothing.  The old files are kept as
``<split>.parquet.pre-augmentation-fix`` unless ``--no-backup``.

Rebuild afterwards, in this order:

1. the fingerprint set, ``gene_fingerprints.pkl.gz`` (deleted here so it cannot
   be read stale);
2. the tensor key table, ``gene_keys.npz`` (likewise);
3. the tokenised tensors under ``tensors/``, **only if you intend to train**;
   an existing checkpoint is unaffected as an artifact.

Memory: the whole frame is held, which is ~25 GB for LeMat-Bulk.  Use a machine
that has it.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from wyckoff_transformer.data import (
    AUGMENTED_SITE_SYMMETRIES,
    AUGMENTED_SITES_ENUMERATION,
    augmented_sites,
)
from wyckoff_transformer.dataset_cache import (
    build_info, load_cache, provenance, resolve_cache, save_split, split_path)
from wyckoff_transformer.preprocess_wychoffs import get_augmentation_dict
from wyckoff_transformer.tokenization import load_wyckoff_mappings

logger = logging.getLogger(__name__)

#: Caches downstream of the augmentation, which become wrong the moment it changes.
DERIVED = ("gene_fingerprints.pkl.gz", "gene_keys.npz")


def migrate_frame(frame: pd.DataFrame, augmentation, enum_by_ss, ss_from_letter):
    """Rewrite one split's augmentation columns.  Returns how many rows moved."""
    symmetries, enumerations, changed = [], [], 0
    previous = (frame[AUGMENTED_SITES_ENUMERATION].values
                if AUGMENTED_SITES_ENUMERATION in frame.columns else None)
    for position, (space_group, letters) in enumerate(
            zip(frame["spacegroup_number"].values, frame["wyckoff_letters"].values)):
        symmetry, enumeration = augmented_sites(
            int(space_group), list(letters), enum_by_ss, augmentation, ss_from_letter)
        symmetries.append(symmetry)
        enumerations.append(enumeration)
        if previous is not None and set(map(tuple, previous[position])) != set(enumeration):
            changed += 1
    frame[AUGMENTED_SITE_SYMMETRIES] = pd.Series(symmetries, index=frame.index, dtype=object)
    frame[AUGMENTED_SITES_ENUMERATION] = pd.Series(enumerations, index=frame.index, dtype=object)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cache", type=Path, help="A dataset cache directory.")
    parser.add_argument("--output", type=Path, default=None, help="Write here instead of in place.")
    parser.add_argument("--check", action="store_true", help="Report and write nothing.")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    cache = resolve_cache(args.cache)
    logger.info("Reading %s", cache)
    frames = load_cache(cache)
    # Read before anything is renamed or overwritten: the new record carries the
    # old one, so a migrated split still says how its rows were built.
    previous = {split: build_info(cache, split) for split in frames}

    augmentation = get_augmentation_dict()
    mappings = load_wyckoff_mappings()
    enum_by_ss = mappings.enum_from_ss_letter
    ss_from_letter = mappings.ss_from_letter

    total = moved = 0
    for split, frame in frames.items():
        if "wyckoff_letters" not in frame.columns:
            raise KeyError(
                f"Split {split!r} has no 'wyckoff_letters', so the augmentation cannot be "
                "recomputed from it. This cache has to be rebuilt from structures.")
        changed = migrate_frame(frame, augmentation, enum_by_ss, ss_from_letter)
        logger.info("Split %s: %d rows, %d whose enumeration variants changed",
                    split, len(frame), changed)
        total += len(frame)
        moved += changed

    print(f"{total} rows; {moved} ({moved / total:.3%}) had their variant set change")
    if args.check:
        print("--check: nothing written")
        return

    target = resolve_cache(args.output) if args.output else cache
    if target == cache and not args.no_backup:
        for split in frames:
            current = split_path(cache, split)
            backup = current.with_suffix(current.suffix + ".pre-augmentation-fix")
            if current.is_file() and not backup.exists():
                current.rename(backup)
                logger.info("Kept the old split as %s", backup)
    for split, frame in frames.items():
        save_split(frame, target, split, build=provenance(
            "migrate_augmented_site_symmetries",
            rewrote=[AUGMENTED_SITE_SYMMETRIES, AUGMENTED_SITES_ENUMERATION],
            supersedes=previous[split]))
    logger.info("Wrote %s", target)

    for name in DERIVED:
        for candidate in target.glob(name.replace(".", "*.", 1)):
            candidate.unlink()
            logger.info("Removed %s, which was derived from the old augmentation", candidate)
    print("Rebuild the fingerprint set and the key table before scoring anything.")


if __name__ == "__main__":
    main()
