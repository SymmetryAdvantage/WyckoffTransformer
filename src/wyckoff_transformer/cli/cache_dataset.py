#!/usr/bin/env python3
"""Build a dataset cache from split CSVs: ``wyformer-cache-dataset <dataset>``.

Reads ``data/<dataset>/{train,val,test}.csv[.gz]``, runs pyxtal over every
structure, and writes the Wyckoff records to ``cache/<dataset>/`` as one Parquet
file per split (``wyckoff_transformer.dataset_cache``).

Four things it does differently from the two scripts it replaces
(``scripts/cache_a_dataset.py`` and ``scripts/cache_a_dataset_reusing.py``):

**It never truncates a gene.**  ``--max-wp`` is gone, and with it the
``max_wp`` argument of ``structure_to_sites``.  Truncating to the first N
Wyckoff positions silently changes a structure's composition while its energy
labels stay attached, so the row then describes a compound that is not in it.
``--max-sites`` **drops** an over-long structure instead, which is well defined.

**Labels are carried automatically.**  Every column of the split CSV that holds
a number, a boolean or a string comes through to the cache; there is no list to
keep in step with the data.  ``--scalar-columns`` narrows that to a chosen few,
and a column named there and missing is an error.  See :func:`columns_to_carry`.

**It reads the CSVs in chunks.**  The old from-scratch path parsed every CIF of
a split into a pymatgen ``Structure`` and held them all, which for LeMat-Bulk is
tens of gigabytes discarded a moment later.

**There is no reuse.**  Copying symmetry records from a cache built at the same
tolerances was worth about six hours on LeMat-Bulk, and cost a subtlety per
option -- a truncation that could only apply to the freshly computed rows, a
reused row whose provenance no longer said what it had been built with.  From
scratch, once, is what this does.

A structure pyxtal cannot handle is dropped with a count, not raised:
``structure_to_sites`` raises after thirty tolerance retries, which through
``Pool.map`` takes the whole run down -- an unaffordable failure mode several
hours in, for a row the dataset can do without.
"""
from __future__ import annotations

import argparse
import logging
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from pandas.api import types as pdtypes

from wyckoff_transformer.data import (
    filter_noisy_pymatgen_warnings,
    get_composition_from_symmetry_sites,
    read_cif,
    structure_to_sites,
)
from wyckoff_transformer.dataset_cache import (
    SPLITS, dataset_cache_dir, provenance, save_cache)
from wyckoff_transformer.paths import data_path
from wyckoff_transformer.preprocess_wychoffs import get_augmentation_dict
from wyckoff_transformer.tokenization import load_wyckoff_mappings

logger = logging.getLogger("cache_dataset")

#: The column holding the structure.  Read, never carried.
STRUCTURE_COLUMN = "cif"

#: What the symmetry record itself contributes.  A CSV column of the same name
#: is a different thing -- ``mp_20``'s ``elements`` is the formula's element
#: list, not the per-site one -- and would overwrite it, so it is skipped.
DERIVED_COLUMNS = frozenset({
    "site_symmetries", "elements", "multiplicity", "wyckoff_letters",
    "sites_enumeration", "dof", "spacegroup_number", "composition",
    "site_symmetries_augmented", "sites_enumeration_augmented",
})

#: Rows read, symmetrised and appended at a time.
DEFAULT_CHUNK_SIZE = 200_000


def split_csv(dataset_dir: Path, split: str) -> Optional[Path]:
    """The CSV holding *split*, gzipped or not, or ``None`` if there is none."""
    for name in (f"{split}.csv.gz", f"{split}.csv"):
        candidate = dataset_dir / name
        if candidate.is_file():
            return candidate
    return None


def columns_to_carry(
    header: pd.DataFrame,
    requested: Optional[Sequence[str]] = None,
    where: str = "",
) -> list[str]:
    """Which CSV columns travel into the cache alongside the symmetry record.

    Without *requested*, every column that holds a number, a boolean or a string
    -- which is what a per-structure label is -- except the structure itself and
    anything the symmetry record already provides. Enumerating them by hand is
    what this replaces: a conditioning label left off that list was built and
    then silently never reached a tensor.

    With *requested*, exactly those, and a missing one is an error: the
    difference between "this dataset happens to have band gaps" and "this run is
    conditioned on max_force".

    Args:
        header: The split's columns and dtypes -- ``read_csv(..., nrows=0)`` is
            enough, and cheap.
        requested: Carry only these.
        where: Named in the messages, so a refusal says which split.

    Raises:
        KeyError: If a requested column is absent, is the structure, or is one
            the symmetry record provides.
    """
    available = list(header.columns)
    if requested is not None:
        missing = [name for name in requested if name not in available]
        if missing:
            raise KeyError(f"{where} has no column(s) {missing}; it holds {available}")
        clashing = [name for name in requested
                    if name == STRUCTURE_COLUMN or name in DERIVED_COLUMNS]
        if clashing:
            raise KeyError(
                f"{where}: {clashing} cannot be carried -- the symmetry record "
                f"provides {sorted(DERIVED_COLUMNS)} and {STRUCTURE_COLUMN!r} is the "
                "structure itself.")
        return list(requested)

    carried, skipped = [], []
    for name in available:
        if name == STRUCTURE_COLUMN:
            continue
        if name in DERIVED_COLUMNS:
            skipped.append(f"{name} (the symmetry record provides it)")
            continue
        dtype = header[name].dtype
        if (pdtypes.is_numeric_dtype(dtype) or pdtypes.is_bool_dtype(dtype)
                or pdtypes.is_string_dtype(dtype)):
            carried.append(name)
        else:
            skipped.append(f"{name} ({dtype})")
    if skipped:
        logger.warning("%s: not carrying %s", where, ", ".join(skipped))
    logger.info("%s: carrying %s", where, ", ".join(carried) or "no labels")
    return carried


def _cif_to_sites(cif: str, **kwargs) -> Optional[dict]:
    """Symmetry-site record for one CIF, or None if it cannot be determined."""
    try:
        record = structure_to_sites(read_cif(cif), **kwargs)
    except Exception:
        logger.debug("failed to derive symmetry sites", exc_info=True)
        return None
    record["composition"] = get_composition_from_symmetry_sites(record)
    return record


def cache_split(
    csv_path: Path,
    scalar_columns: Sequence[str],
    n_jobs: Optional[int],
    chunk_size: int,
    symmetry_precision: float,
    symmetry_a_tol: float,
    sort_by_letter: bool,
    max_sites: Optional[int],
) -> pd.DataFrame:
    """Symmetry records plus carried labels for one split."""
    to_sites = partial(
        _cif_to_sites,
        wychoffs_enumerated_by_ss=load_wyckoff_mappings().enum_from_ss_letter,
        wychoffs_augmentation=get_augmentation_dict(),
        tol=symmetry_precision,
        a_tol=symmetry_a_tol,
        sort_by_letter=sort_by_letter,
    )
    blocks = []
    computed = failed = dropped = 0
    with Pool(n_jobs) as pool:
        for index, chunk in enumerate(
                pd.read_csv(csv_path, index_col=0, chunksize=chunk_size)):
            records = pool.map(to_sites, chunk[STRUCTURE_COLUMN], chunksize=64)
            usable = [position for position, record in enumerate(records)
                      if record is not None]
            failed += len(records) - len(usable)
            computed += len(usable)
            if not usable:
                continue
            block = pd.DataFrame.from_records(
                [records[position] for position in usable],
                index=chunk.index[usable])
            if max_sites is not None:
                too_long = block["site_symmetries"].str.len() > max_sites
                dropped += int(too_long.sum())
                block = block[~too_long]
                if block.empty:
                    continue
            for column in scalar_columns:
                block[column] = chunk.loc[block.index, column]
            blocks.append(block)
            logger.info("%s chunk %3d: computed %d, failed %d, over-long %d",
                        csv_path.name, index, computed, failed, dropped)
    if dropped:
        logger.warning("%s: dropped %d structures with more than %d Wyckoff sites",
                       csv_path.name, dropped, max_sites)
    if failed:
        logger.warning("%s: dropped %d structures pyxtal could not handle", csv_path.name, failed)
    if not blocks:
        raise ValueError(f"{csv_path}: no structure survived symmetrisation")
    result = pd.concat(blocks)
    logger.info("%s: %d rows", csv_path.name, len(result))
    return result


def cache_dataset(
    dataset: str,
    max_sites: Optional[int] = None,
    scalar_columns: Optional[Sequence[str]] = None,
    n_jobs: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    symmetry_precision: float = 0.1,
    symmetry_a_tol: float = 5.0,
    sort_by_letter: bool = True,
    observed_gene_minimum_target: bool = False,
) -> dict[str, pd.DataFrame]:
    """Symmetrise every split of *dataset* and write the cache.  Returns the frames."""
    dataset_dir = data_path(dataset)
    frames = {}
    for split in SPLITS:
        csv_path = split_csv(dataset_dir, split)
        if csv_path is None:
            logger.warning("no %s split in %s; skipping", split, dataset_dir)
            continue
        carried = columns_to_carry(
            pd.read_csv(csv_path, nrows=0, index_col=0),
            scalar_columns,
            where=str(csv_path),
        )
        frames[split] = cache_split(
            csv_path, carried, n_jobs, chunk_size, symmetry_precision,
            symmetry_a_tol, sort_by_letter, max_sites)
    if not frames:
        raise FileNotFoundError(
            f"{dataset_dir} holds no {'/'.join(SPLITS)} CSV to cache.")

    if observed_gene_minimum_target:
        from wyckoff_transformer.gene_energy import add_observed_gene_minimum  # noqa: PLC0415

        add_observed_gene_minimum(frames)

    build = provenance(
        "wyformer-cache-dataset",
        max_sites=max_sites,
        symmetry_precision=symmetry_precision,
        symmetry_a_tol=symmetry_a_tol,
        sort_by_letter=sort_by_letter,
        scalar_columns=list(scalar_columns) if scalar_columns is not None else None,
        # Which splits, not just whether: the minimum is taken over every split
        # present at cache time, so a cache built while a split was missing
        # carries a different target under the same column name.
        observed_gene_minimum_over=sorted(frames) if observed_gene_minimum_target else None,
    )
    save_cache(frames, dataset_cache_dir(dataset), build)
    return frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="Name under data/, and the cache to write under cache/.")
    parser.add_argument("--max-sites", type=int, default=None,
                        help="Drop structures with more Wyckoff sites than this. Not a "
                             "truncation: cutting sites off a structure silently changes "
                             "its composition, and its energy labels then describe a "
                             "compound that is not in the row. Every sequence tensor is "
                             "padded to the longest structure in the dataset, so a handful "
                             "of very long ones cost width on every row -- 0.1%% of "
                             "LeMat-Bulk runs past 61 sites and would take the padded "
                             "width from 62 to 361.")
    parser.add_argument("--scalar-columns", nargs="*", default=None,
                        help="Carry only these per-structure columns from the CSVs. The "
                             "default carries every numeric, boolean and string column, "
                             "so a conditioning label cannot be left out by omission. A "
                             "column named here and missing is an error.")
    parser.add_argument("--observed-gene-minimum-target", action="store_true",
                        help="Add gene_min_formation_energy_per_atom from the lowest "
                             "formation_energy_per_atom observed for each augmented "
                             "Wyckoff gene across all splits.")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Worker processes for symmetry determination. One per core "
                             "by default.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="CSV rows held at a time.")
    parser.add_argument("--symmetry-precision", type=float, default=0.1,
                        help="Passed to pyxtal.from_seed as tol.")
    parser.add_argument("--symmetry-a-tol", type=float, default=5.0,
                        help="Passed to pyxtal.from_seed as a_tol.")
    parser.add_argument("--no-sort-by-letter", dest="sort_by_letter", action="store_false",
                        help="Keep pyxtal's own site order. The default sorts each "
                             "structure's sites by Wyckoff letter, which is what every "
                             "cache now on disk did and therefore what makes a new one "
                             "comparable with them.")
    parser.set_defaults(sort_by_letter=True)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    # Before any Pool exists, so its workers inherit the filter across fork and
    # do not each repeat what is already known. Said once here because the
    # alternative -- saying it half a million times -- is what this replaces.
    filter_noisy_pymatgen_warnings()
    logger.info(
        "Suppressed for this build: pymatgen's CIF coordinate-rounding note "
        "(read_cif) and its missing-Pauling-electronegativity note for the "
        "noble gases (both expected, neither actionable)")
    frames = cache_dataset(
        args.dataset,
        max_sites=args.max_sites,
        scalar_columns=args.scalar_columns,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk_size,
        symmetry_precision=args.symmetry_precision,
        symmetry_a_tol=args.symmetry_a_tol,
        sort_by_letter=args.sort_by_letter,
        observed_gene_minimum_target=args.observed_gene_minimum_target,
    )
    print(f"Cached {dataset_cache_dir(args.dataset)}: "
          f"{ {split: len(frame) for split, frame in frames.items()} }")


if __name__ == "__main__":
    main()
