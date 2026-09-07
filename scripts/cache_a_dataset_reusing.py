#!/usr/bin/env python3
"""Cache a dataset that is a superset of one already cached, reusing its symmetry records.

`scripts/cache_a_dataset.py` runs pyxtal over every structure. When the new dataset differs
from an existing cache only by a relaxed filter -- as `lemat_bulk_fmax1` differs from
`lemat_bulk_ehull` only in cutting max_force at 1 eV/A instead of 0.02 -- that repeats
several million symmetry determinations to get the same answer. `structure_to_sites` is a
pure function of the structure and the tolerances, so the overlap can simply be copied:
here that is 4.2M of 4.7M rows, leaving ~0.5M to compute.

Two differences from the plain path, both deliberate:

The CSVs are streamed in chunks rather than read whole. `read_MP` parses every CIF in a
split into a pymatgen Structure and holds them all at once, which for a few million rows
is tens of gigabytes of objects that are discarded a moment later.

There is no ``--max-wp``. Truncating to the first N Wyckoff positions would apply only to
the freshly computed records -- the reused ones come from a cache built by some other run,
whose ``max_wp`` this script cannot know -- so a single cache would mix truncated and
untruncated structures, and truncation silently changes a composition. ``--max-sites``
drops over-long structures instead, which is well defined either way. Use
``scripts/cache_a_dataset.py --max-wp`` to build a truncated cache from scratch.

A structure pyxtal cannot handle is dropped with a count, not raised. `structure_to_sites`
raises RuntimeError after thirty tolerance retries, which through `Pool.map` takes the
whole run down -- an unaffordable failure mode several hours in, for a row the dataset can
do without.
"""
import argparse
import gzip
import logging
import pickle
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import pandas as pd

from wyckoff_transformer.data import (
    LEGACY_SCALAR_COLUMNS, get_composition_from_symmetry_sites, read_cif, structure_to_sites)
from wyckoff_transformer.preprocess_wychoffs import get_augmentation_dict
from wyckoff_transformer.tokenization import load_wyckoff_mappings

logger = logging.getLogger("cache_a_dataset_reusing")

REPO = Path(__file__).resolve().parent.parent
#: Everything `compute_symmetry_sites` derives from the structure alone, and therefore
#: everything that may be copied from another cache built at the same tolerances.
SYMMETRY_COLUMNS = (
    "site_symmetries", "elements", "multiplicity", "wyckoff_letters",
    "sites_enumeration", "dof", "spacegroup_number", "sites_enumeration_augmented",
    "composition",
)


def _cif_to_sites(cif: str, **kwargs) -> Optional[dict]:
    """Symmetry-site record for one CIF, or None if it cannot be determined."""
    try:
        record = structure_to_sites(read_cif(cif), **kwargs)
    except Exception:
        logger.debug("failed to derive symmetry sites", exc_info=True)
        return None
    record["composition"] = get_composition_from_symmetry_sites(record)
    return record


def load_reuse_frame(paths) -> pd.DataFrame:
    """One frame of symmetry records, indexed by structure id, from existing caches."""
    frames = []
    for path in paths:
        logger.info("reading %s", path)
        with gzip.open(path, "rb") as handle:
            cached = pickle.load(handle)
        for split, frame in cached.items():
            missing = set(SYMMETRY_COLUMNS) - set(frame.columns)
            if missing:
                raise KeyError(f"{path}:{split} has no {sorted(missing)}")
            frames.append(frame[list(SYMMETRY_COLUMNS)])
            logger.info("  %s: %d rows", split, len(frame))
    reuse = pd.concat(frames)
    reuse = reuse[~reuse.index.duplicated(keep="first")]
    logger.info("reusable symmetry records: %d", len(reuse))
    return reuse


def verify_reuse(csv_path: Path, reuse: pd.DataFrame, to_sites, sample: int):
    """Recompute a sample of the reusable records and insist they match.

    The whole shortcut rests on `structure_to_sites` being a pure function of the
    structure and the tolerances, so the one thing that can silently ruin the cache is a
    tolerance or an ordering that differs from whatever built the source. Both caches on
    disk turned out to be sorted by Wyckoff letter, which is not this function's default;
    finding that out from a mismatch here is much cheaper than finding it out from a
    model that trained on two conventions at once.
    """
    header = pd.read_csv(csv_path, index_col=0, nrows=sample * 40)
    common = header.index.intersection(reuse.index)[:sample]
    if not len(common):
        logger.warning("no reusable rows among the first rows of %s; skipping the check",
                       csv_path.name)
        return
    mismatched = []
    for structure_id in common:
        fresh = to_sites(header.loc[structure_id, "cif"])
        if fresh is None:
            continue
        for column in SYMMETRY_COLUMNS:
            left, right = fresh[column], reuse.loc[structure_id, column]
            same = list(left) == list(right) if isinstance(left, list) else left == right
            if not same:
                mismatched.append((structure_id, column, left, right))
    if mismatched:
        for row in mismatched[:5]:
            logger.error("reuse mismatch: %s %s: %r != %r", *row)
        raise ValueError(
            f"{len(mismatched)} field(s) of {len(common)} sampled records differ from a "
            "fresh computation. The reused cache was built with different tolerances or a "
            "different site order; recompute instead of reusing.")
    logger.info("verified %d reused records against a fresh computation", len(common))


def cache_split(
    csv_path: Path,
    reuse: pd.DataFrame,
    scalar_columns,
    n_jobs: Optional[int],
    chunk_size: int,
    symmetry_precision: float,
    symmetry_a_tol: float,
    sort_by_letter: bool,
    max_sites: Optional[int],
    verify: int = 0,
) -> pd.DataFrame:
    """Symmetry records plus scalar labels for one split."""
    to_sites = partial(
        _cif_to_sites,
        wychoffs_enumerated_by_ss=load_wyckoff_mappings().enum_from_ss_letter,
        wychoffs_augmentation=get_augmentation_dict(),
        tol=symmetry_precision,
        a_tol=symmetry_a_tol,
        max_wp=None,
        sort_by_letter=sort_by_letter,
    )
    if verify:
        verify_reuse(csv_path, reuse, to_sites, verify)
    blocks = []
    reused = computed = failed = dropped = 0
    with Pool(n_jobs) as pool:
        for index, chunk in enumerate(pd.read_csv(csv_path, index_col=0, chunksize=chunk_size)):
            missing = [column for column in scalar_columns if column not in chunk.columns]
            if missing:
                raise KeyError(f"{csv_path} has no {missing}; it holds {sorted(chunk.columns)}")
            known = chunk.index.intersection(reuse.index)
            fresh = chunk.index.difference(reuse.index)
            parts = []
            if len(known):
                parts.append(reuse.loc[known])
                reused += len(known)
            if len(fresh):
                records = pool.map(to_sites, chunk.loc[fresh, "cif"], chunksize=64)
                usable = [i for i, record in enumerate(records) if record is not None]
                failed += len(records) - len(usable)
                computed += len(usable)
                if usable:
                    parts.append(pd.DataFrame.from_records(
                        [records[i] for i in usable],
                        index=fresh[usable])[list(SYMMETRY_COLUMNS)])
            if not parts:
                continue
            block = pd.concat(parts)
            if max_sites is not None:
                too_long = block["site_symmetries"].str.len() > max_sites
                dropped += int(too_long.sum())
                block = block[~too_long]
                if block.empty:
                    continue
            for column in scalar_columns:
                block[column] = chunk.loc[block.index, column]
            blocks.append(block)
            logger.info(
                "%s chunk %3d: reused %d, computed %d, failed %d, over-long %d",
                csv_path.name, index, reused, computed, failed, dropped)
    if dropped:
        logger.warning("%s: dropped %d structures with more than %d Wyckoff sites",
                       csv_path.name, dropped, max_sites)
    if failed:
        logger.warning("%s: dropped %d structures pyxtal could not handle", csv_path.name, failed)
    result = pd.concat(blocks)
    # Rows come out grouped reused-then-fresh within each chunk rather than in CSV order.
    # That is deterministic given the same inputs, and nothing downstream reads position:
    # the split is decided by id before this runs, the labels travel on the row, and the
    # loader batches by a sampled permutation. Left alone so this script and the cache it
    # already produced stay in exact correspondence.
    logger.info("%s: %d rows", csv_path.name, len(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="Name under data/ and cache/.")
    parser.add_argument("--reuse", type=Path, nargs="+", required=True,
                        help="data.pkl.gz files whose symmetry records may be copied. They "
                             "must have been built at the same tolerances.")
    parser.add_argument("--scalar-columns", nargs="*", default=None,
                        help="Per-structure scalar columns to carry from the CSVs into the "
                             "cache. Defaults to whichever of the legacy names are present.")
    parser.add_argument("--observed-gene-minimum-target", action="store_true",
                             help="Add gene_min_formation_energy_per_atom from the lowest "
                                  "formation_energy_per_atom observed for each augmented Wyckoff "
                                  "gene across all splits.")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--symmetry-precision", type=float, default=0.1)
    parser.add_argument("--symmetry-a-tol", type=float, default=5.0)
    parser.add_argument("--no-sort-by-letter", dest="sort_by_letter", action="store_false",
                        help="Keep pyxtal's own site order. The default sorts by Wyckoff "
                             "letter, which is what every cache on disk did and therefore "
                             "what a reused record has to be compared against.")
    parser.set_defaults(sort_by_letter=True)
    parser.add_argument("--max-sites", type=int, default=None,
                        help="Drop structures with more Wyckoff sites than this. Not a "
                             "truncation: cutting sites off a structure silently changes "
                             "its composition, and its energy labels then describe a "
                             "compound that is not in the row. Every sequence tensor is "
                             "padded to the longest structure in the dataset, so a handful "
                             "of very long ones cost width on every row -- 0.1%% of "
                             "LeMat-Bulk runs past 61 sites and would take the padded "
                             "width from 62 to 361.")
    parser.add_argument("--verify-reuse", type=int, default=200,
                        help="Recompute this many reusable records per split and abort if "
                             "any differs. 0 skips the check.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data_dir = REPO / "data" / args.dataset
    reuse = load_reuse_frame(args.reuse)

    result = {}
    for split in ("train", "val", "test"):
        csv_path = data_dir / f"{split}.csv.gz"
        if not csv_path.exists():
            logger.warning("%s not found; skipping", csv_path)
            continue
        columns = args.scalar_columns
        if columns is None:
            header = pd.read_csv(csv_path, nrows=0)
            columns = [name for name in LEGACY_SCALAR_COLUMNS if name in header.columns]
            logger.info("carrying scalar columns %s", columns)
        result[split] = cache_split(
            csv_path, reuse, columns, args.n_jobs, args.chunk_size,
            args.symmetry_precision, args.symmetry_a_tol, args.sort_by_letter,
            args.max_sites, verify=args.verify_reuse)

    if args.observed_gene_minimum_target:
        from wyckoff_transformer.gene_energy import add_observed_gene_minimum

        add_observed_gene_minimum(result)

    out = REPO / "cache" / args.dataset / "data.pkl.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wb") as handle:
        pickle.dump(result, handle)
    logger.info("wrote %s (%.1f MB): %s", out, out.stat().st_size / 1e6,
                {name: len(frame) for name, frame in result.items()})


if __name__ == "__main__":
    main()
