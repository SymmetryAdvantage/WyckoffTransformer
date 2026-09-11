#!/usr/bin/env python3
"""Report, for every LeMat-Bulk variant on disk, how it was force-filtered and whether it
carries the recovered Materials Project forces.

Several LeMat variants coexist in ``data/`` and ``cache/``, built months apart under
different rules, and nothing in a split CSV or a cache says which rules those were. Two
differences change what a model trained on the variant has seen:

``max_force`` cut
    The legacy ``scripts/pipeline_lemat_20wyckoffs.py`` cut at 0.02 eV/A, which is a
    provenance filter: it keeps 95.6% of Alexandria against 35.5% of the ICSD-backed
    Materials Project rows. ``scripts/build_lemat_bulk_fmax.py`` cuts at 1.0 instead.
    Inferred here as the smallest standard threshold no row exceeds.

recovered forces
    30,679 rows -- all Materials Project -- have an empty ``forces`` array in the archive
    and were long either dropped (a ``max_force <= X`` cut drops NaN for every X) or kept
    with an imputed value. ``scripts/recover_mp_forces.py`` reads the true values from MP's
    task documents. Detected here by comparing each variant's ``max_force`` on those rows
    against the recovered values: equal means recovered, one repeated value means imputed,
    absent means the row was filtered out.

See ``docs/lemat_bulk_pipeline.md``. Run from a checkout with the stores attached::

    python scripts/audit_lemat_variants.py
    python scripts/audit_lemat_variants.py --variant data/lemat_bulk_fmax1_stress
"""
from __future__ import annotations

import argparse
import gzip
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from wyckoff_transformer.paths import cache_root, data_root

logger = logging.getLogger("audit_lemat_variants")

#: Thresholds the two builders use, smallest first.
KNOWN_CUTS = (0.02, 0.1, 1.0)
#: Columns worth reading out of a split CSV; the CIFs are most of the file.
COLUMNS = ("immutable_id", "max_force", "max_force_missing", "stress_hydrostatic")


def archive_forces() -> pd.Series:
    """``max_force`` as the archive reports it, by structure id, NaN where it is empty.

    The fallback for a cache built without a ``max_force`` scalar column: the cut it was
    built under is still legible from the archive values of the rows that survived it.
    """
    path = data_root() / "lemat-bulk" / "convergence_labels.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    table = pd.read_parquet(path, columns=["immutable_id", "max_force", "convergence_source"])
    # The recovered rows are empty *in the archive*, which is what a past cut saw.
    table.loc[table["convergence_source"] != "archive", "max_force"] = np.nan
    return table.set_index("immutable_id")["max_force"]


def recovered_forces() -> pd.Series:
    """True ``max_force`` of the rows whose archived forces are empty, by structure id."""
    path = cache_root() / "mp_forces_recovery" / "runs" / "full" / "results.parquet"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run scripts/recover_mp_forces.py all --run full")
    table = pd.read_parquet(path, columns=["immutable_id", "group", "status", "max_abs_force"])
    table = table[(table["group"] == "missing") & (table["status"] == "exact_forces_stress")]
    return table.set_index("immutable_id")["max_abs_force"]


def load_splits(directory: Path) -> dict[str, pd.DataFrame]:
    frames = {}
    for split in ("train", "val", "test"):
        path = directory / f"{split}.csv.gz"
        if path.exists():
            frames[split] = pd.read_csv(path, usecols=lambda c: c in COLUMNS)
    return frames


def load_cache(path: Path) -> dict[str, pd.DataFrame]:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def describe(name: str, frames: dict[str, pd.DataFrame], truth: pd.Series,
             archive: pd.Series) -> dict:
    """One row of the report for one variant."""
    record = {"variant": name, "rows": sum(len(f) for f in frames.values()),
              "splits": " ".join(f"{k} {len(v)}" for k, v in frames.items())}
    # Caches are indexed by structure id; split CSVs carry it as a column.
    joined = pd.concat([f.reset_index().set_index(
        "immutable_id" if "immutable_id" in f.reset_index().columns else "index")
        for f in frames.values()])
    record["stress labels"] = "stress_hydrostatic" in joined.columns
    stored = "max_force" in joined.columns
    if not stored:
        if archive.empty:
            record["max_force cut"] = "unknown: no max_force column"
            record["recovered forces"] = "unknown"
            return record
        joined = joined.assign(max_force=archive.reindex(joined.index))

    force = joined["max_force"]
    cut = next((f"<= {c}" for c in KNOWN_CUTS if force.max() <= c + 1e-9),
               f"none below {force.max():.4g}")
    record["max_force cut"] = cut if stored else f"{cut} (from the archive, not stored)"

    present = truth.index.intersection(joined.index)
    if len(present) == 0:
        record["recovered forces"] = "n/a: holds none of the formerly-empty rows"
    elif not stored:
        record["recovered forces"] = f"no such column; holds {len(present)} formerly-empty rows"
    else:
        seen = joined.loc[present, "max_force"]
        if np.allclose(seen, truth.loc[present], rtol=1e-6, atol=1e-9):
            record["recovered forces"] = f"yes: {len(present)} rows"
        elif seen.round(8).nunique() == 1:
            record["recovered forces"] = f"NO: {len(present)} rows imputed at {seen.iloc[0]:.8g}"
        else:
            matching = int(np.isclose(seen, truth.loc[present], rtol=1e-6, atol=1e-9).sum())
            record["recovered forces"] = f"partial: {matching} of {len(present)} rows"
    if "max_force_missing" in joined.columns:
        record["recovered forces"] += f"; flag on {int(joined['max_force_missing'].sum())}"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", action="append", default=None,
                        help="Audit only these, as 'data/<name>' or 'cache/<name>'. "
                             "Repeatable; default is every LeMat variant found.")
    parser.add_argument("--output", type=Path, default=None, help="Also write the report as CSV.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    truth = recovered_forces()
    archive = archive_forces()
    logger.info("%d recovered rows, median max_force %.4f", len(truth), truth.median())

    if args.variant:
        wanted = [(v.split("/", 1)[0], v.split("/", 1)[1]) for v in args.variant]
    else:
        wanted = ([("data", p.name) for p in sorted(data_root().glob("lemat_bulk*")) if p.is_dir()]
                  + [("cache", p.name) for p in sorted(cache_root().glob("lemat_bulk*")) if p.is_dir()])

    records = []
    for store, name in wanted:
        if store == "data":
            frames = load_splits(data_root() / name)
            if not frames:
                logger.warning("data/%s holds no split CSVs; skipped", name)
                continue
        else:
            path = cache_root() / name / "data.pkl.gz"
            if not path.exists():
                logger.warning("%s is missing; skipped", path)
                continue
            frames = load_cache(path)
        records.append(describe(f"{store}/{name}", frames, truth, archive))
        logger.info("%s: %s", records[-1]["variant"], records[-1]["recovered forces"])
        del frames

    report = pd.DataFrame(records)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print()
    print(report.to_string(index=False))
    if args.output is not None:
        report.to_csv(args.output, index=False)
        logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
