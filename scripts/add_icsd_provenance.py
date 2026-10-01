#!/usr/bin/env python3
"""Derive a dataset cache that adds an ICSD-provenance column to an existing one.

    python scripts/add_icsd_provenance.py lemat_bulk_fmax1_stress lemat_bulk_fmax1_stress_icsd

Copies the parent's Wyckoff records split by split and adds ``icsd_backed``: 1.0
for a Materials Project row whose ``theoretical`` flag is False in
``data/mp_provenance.csv.gz`` (an ICSD-descended entry), 0.0 for everything else --
theoretical MP rows, MP ids the API no longer resolves, OQMD and Alexandria. That
is the reading `formula_energy.dataset._kinds` uses, so the flag means the same
thing at the gene level as at the formula level.

No pyxtal, no symmetry: the rows and their genes are the parent's, untouched. The
new cache has no tensors; tokenise it with a tokeniser that carries the column
(``yamls/tokenisers/der_tokenizer_v1_icsd.yaml``).
"""
import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from wyckoff_transformer.dataset_cache import cache_exists, load_cache, provenance, save_cache
from wyckoff_transformer.dataset_manifest import load_manifest
from wyckoff_transformer.paths import cache_root

logger = logging.getLogger("add_icsd_provenance")

DEFAULT_PROVENANCE = Path(__file__).resolve().parent.parent / "data" / "mp_provenance.csv.gz"
COLUMN = "icsd_backed"


def icsd_flags(immutable_ids: pd.Index, provenance_table: pd.DataFrame) -> np.ndarray:
    """1.0 where MP says the entry is not theoretical, else 0.0."""
    theoretical = pd.Series(immutable_ids, index=immutable_ids).map(provenance_table["theoretical"])
    return theoretical.eq(False).to_numpy(dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("parent", help="Existing dataset cache to derive from")
    parser.add_argument("target", help="Name of the new dataset cache")
    parser.add_argument("--provenance-csv", type=Path, default=DEFAULT_PROVENANCE,
                        help="material_id, theoretical, n_icsd from Materials Project")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    source = cache_root() / args.parent
    target = cache_root() / args.target
    if cache_exists(target):
        raise FileExistsError(f"{target} already holds a cache; refusing to overwrite it")

    table = pd.read_csv(args.provenance_csv, index_col="material_id")
    frames = load_cache(source)
    for split, frame in frames.items():
        frame[COLUMN] = icsd_flags(frame.index, table)
        logger.info("%s: %d rows, %d ICSD-backed (%.2f%%)", split, len(frame),
                    int(frame[COLUMN].sum()), 100 * frame[COLUMN].mean())

    fields = load_manifest(args.parent).fields_record()
    fields[COLUMN] = load_manifest(args.target).fields_record()[COLUMN]
    target.mkdir(parents=True, exist_ok=True)
    save_cache(frames, target, provenance(
        "scripts/add_icsd_provenance.py", parent=args.parent,
        provenance_csv=str(args.provenance_csv), manifest=args.target, fields=fields))
    shutil.copy2(source / "split_ids.json", target / "split_ids.json")
    logger.info("Wrote %s", target)


if __name__ == "__main__":
    main()
