#!/usr/bin/env python3
"""Build a LeMat-Bulk split carrying dirty-data labels and formation energy.

The dataset this replaces cut ``max_force <= 0.02``, which
``wyckoff_transformer.formula_energy.dataset`` shows is a provenance filter wearing a
convergence costume: it keeps 95.6% of Alexandria rows but only 35.5% of the ICSD-backed
ones, because Materials Project reports forces from a different protocol whose median is
0.028 -- above the cut. Cutting at 1 eV/A instead excludes only the genuinely pathological
relaxations (0.6% of the archive) and lets the model see the structures the old cut threw
away, on the understanding that it is *told* how converged each row is rather than being
asked to pretend they are all equal.

Three conditioning labels come out, all per structure, all in physical units:

``energy_above_hull``
    ``e_hull`` from ``formula_energy.hull_table``, eV/atom, against the archive's own
    phase diagram. Clipped at zero -- pymatgen's ``get_e_above_hull`` is already
    non-negative up to float noise, and one row lands at -2.7e-15.

``delta_e_polymorph``
    How far this structure sits above the best polymorph of the same *reduced* formula
    present in this dataset, eV/atom. Absolute ``e_hull`` cannot express "ground state of
    its formula": a row at 0.15 may be the best structure anyone has computed for that
    composition or a polymorph 0.1 above it, and those want different generations.
    Computed from ``energy_corrected / n_atoms`` rather than ``e_form`` because the two
    give identical differences within a composition -- the elemental reference cancels --
    and the per-atom energy is defined for every row, including the 11% where the hull
    construction returned nothing.

``max_force``
    eV/A, the largest force component on any atom. The convergence covariate: at
    generation time, asking for ``max_force = 0`` asks for the relaxed mode of the
    distribution rather than its unconverged tail. Note that *exactly* zero -- 27.4% of
    Alexandria and 39.3% of OQMD rows -- carries no information about relaxation. Where
    every site sits on a Wyckoff position with no free coordinate, the site symmetry
    leaves no invariant vector, so the force vanishes identically at any geometry and
    VASP's symmetrisation writes it out as 0.0 before a single ionic step has run. In a
    sample of 150 such rows, 150 had zero free positional parameters, against 0 of 150
    rows above 5 meV/A; 96% of them still report a stress above 1 kB, so what they do
    not report is any statement about the cell. Conditioning at 0 selects structures
    whose Wyckoff positions are fully determined, not structures that relaxed well.

``max_force_missing``
    1.0 where the archive reports no forces at all, 0.0 elsewhere. 30,679 rows -- 22.1%
    of Materials Project, and nothing from Alexandria or OQMD -- carry an empty ``forces``
    array (and an empty ``stress_tensor``), because the MP task chosen to represent the
    bulk material did not report them. A ``max_force <= X`` filter drops those rows for
    any X, since NaN fails every comparison: the same accidental provenance filter as the
    0.02 cut, one layer down, and it survived loosening the cut to 1. They are kept here,
    with ``max_force`` imputed as the median over the rows of *their own source database*
    that do have forces (0.0417 for MP, against 0.0034 for Alexandria) and the indicator
    saying the value is imputed. Imputing zero instead would have filed them under the
    symmetry-locked mode above, which is precisely what they are not known to be.

``formation_energy_per_atom``
    The PBE formation energy from the same phase-diagram calculation that supplies
    ``e_hull``. The cache derives an augmentation-invariant observed minimum over
    all rows sharing a Wyckoff gene and uses that derived column as the critic target.

The minimum behind ``delta_e_polymorph`` is taken over the whole filtered dataset, not the
train split alone. A train-only minimum makes the label mean something different on either
side of the split: 62.9% of held-out rows share a reduced formula with train, and a
quarter of those sit *below* the train minimum, which would hand the validation loss
negative offsets that ``log1p`` cannot represent. The label is a model input rather than a
target, so a composition's floor crossing the split leaks nothing about the Wyckoff gene
that is actually being predicted.

The held-out ids are inherited verbatim from ``cache/lemat_bulk_ehull`` where the file
written by ``--split-ids`` is available. This dataset is a superset of that one, so a
fresh random split would scatter the old held-out rows into training and pull old training
rows into validation -- either direction quietly invalidates a comparison against the
already-trained e_hull-only baseline.
"""
import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("build_lemat_bulk_fmax")

REPO = Path(__file__).resolve().parent.parent
#: ``max_force`` and the CIFs. The energy CSV has a ``max_force`` column too, and it is
#: NaN in every one of its rows.
DEFAULT_STRUCTURE_CSV = REPO / "data" / "lemat-bulk" / "lemat_pbe.csv.gz"
#: Written by ``scripts/build_lemat_bulk_fmax.py --labels-only`` (or the scratch probe):
#: one row per structure with the reduced formula, cell size and energies, no CIF.
DEFAULT_LABELS = REPO / "data" / "lemat-bulk" / "labels.parquet"
DEFAULT_SPLIT_IDS = REPO / "cache" / "lemat_bulk_ehull" / "split_ids.json"

#: Formation energies outside this window are corrupt rather than exotic -- the archive
#: runs to -37.9 and +650.7 eV/atom. It matters more here than in a mean-fitting model:
#: ``delta_e_polymorph`` takes a minimum, so one corrupt row poisons the label of every
#: other polymorph of its composition, and a spuriously low energy also *defines* the
#: hull, so the row arrives labelled perfectly stable.
MAX_ABS_E_FORM = 5.0

#: Columns written to the split CSVs. ``read_MP`` takes column 0 as the index and needs
#: ``cif``; ``compute_symmetry_sites`` carries the rest through to the cache.
OUTPUT_COLUMNS = [
    "immutable_id",
    "cif",
    "energy_above_hull",
    "delta_e_polymorph",
    "max_force",
    "max_force_missing",
    "formation_energy_per_atom",
]


def _source(ids: pd.Series) -> pd.Series:
    """Source database of each row, read off the ``immutable_id`` prefix.

    The three archives converged their relaxations to different criteria, so "how
    converged is a row that reports no forces" has a different answer in each.
    """
    text = ids.astype(str)
    return pd.Series(
        np.where(text.str.startswith("mp-"), "mp",
                 np.where(text.str.startswith("agm"), "alexandria", "oqmd")),
        index=ids.index)


def build_labels(structure_csv: Path, energy_csv: Path, out: Path) -> pd.DataFrame:
    """Join forces onto energies and add the reduced-formula key and per-atom energy."""
    sys.path.insert(0, str(REPO / "src"))
    from wyckoff_transformer.formula_energy.dataset import _keys_and_sizes

    logger.info("reading %s", structure_csv)
    force = pd.read_csv(
        structure_csv,
        usecols=["immutable_id", "full_formula", "chemsys", "energy_corrected", "max_force"])
    logger.info("reading %s", energy_csv)
    energy = pd.read_csv(energy_csv, usecols=["immutable_id", "e_form", "e_hull"])
    table = force.merge(energy, on="immutable_id", how="inner")
    del force, energy
    keys, sizes = _keys_and_sizes(table["full_formula"])
    table["formula"] = keys
    table["n_atoms"] = sizes
    table["energy_per_atom"] = table["energy_corrected"] / table["n_atoms"]
    table.drop(columns=["energy_corrected", "full_formula"], inplace=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    logger.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return table


def label_rows(labels: pd.DataFrame, max_force: float) -> pd.DataFrame:
    """Filter to the trainable rows and attach the conditioning labels."""
    kept = labels
    logger.info("rows                      : %9d", len(kept))
    # NaN fails every comparison, so `max_force <= X` would silently take the rows with
    # no forces reported out with the unconverged ones. They are a different thing and
    # are kept, flagged by `max_force_missing`.
    kept = kept[kept["max_force"].isna() | (kept["max_force"] <= max_force)]
    logger.info("max_force <= %-12g: %9d", max_force, len(kept))
    logger.info("  of which force is absent: %9d", int(kept["max_force"].isna().sum()))
    kept = kept[kept["e_hull"].notna()]
    logger.info("  & e_hull is not NaN     : %9d", len(kept))
    kept = kept[kept["e_form"].abs() <= MAX_ABS_E_FORM]
    logger.info("  & |e_form| <= %-10g: %9d", MAX_ABS_E_FORM, len(kept))
    # Exactly one row of the archive -- an Rb structure -- has no immutable_id. Everything
    # downstream is keyed by it, so it cannot be carried; drop it here rather than losing
    # it silently to an index lookup that quietly misses NaN.
    kept = kept[kept["immutable_id"].notna()]
    logger.info("  & immutable_id present  : %9d", len(kept))

    kept = kept.copy()
    # get_e_above_hull is non-negative by construction; the clip is against float noise,
    # not against the physics. log1p rejects the -2.7e-15 row otherwise.
    kept["energy_above_hull"] = kept["e_hull"].clip(lower=0.0)
    kept["max_force_missing"] = kept["max_force"].isna().astype(float)
    # Per source, because the archives differ by an order of magnitude in residual force
    # and every row missing forces is from Materials Project, the loosest of the three.
    # `median` skips NaN; a source with no measured row at all falls back to the global
    # median, which cannot happen with the current archive but costs one call to rule out.
    by_source = kept.groupby(_source(kept["immutable_id"]), sort=False)["max_force"]
    kept["max_force"] = kept["max_force"].fillna(by_source.transform("median")).fillna(
        kept["max_force"].median())
    # ``e_form`` is the formation energy per atom against the same elemental
    # references that define ``e_hull``. Preserve it so cache construction can
    # assign every equivalent Wyckoff gene its observed minimum.
    kept["formation_energy_per_atom"] = kept["e_form"]
    floor = kept.groupby("formula", sort=False)["energy_per_atom"].transform("min")
    kept["delta_e_polymorph"] = (kept["energy_per_atom"] - floor).clip(lower=0.0)
    return kept


def assign_split(ids: pd.Index, split_ids: Path, seed: int, val_size: int, test_size: int):
    """val/test inherited from the previous dataset where possible, random otherwise."""
    if split_ids.exists():
        payload = json.loads(split_ids.read_text())
        held = {}
        for name in ("val", "test"):
            wanted = pd.Index(payload[name])
            present = wanted.intersection(ids)
            logger.info(
                "%s: %d of %d ids from %s survive this dataset's filters",
                name, len(present), len(wanted), split_ids)
            held[name] = present
        overlap = held["val"].intersection(held["test"])
        if len(overlap):
            raise ValueError(f"{len(overlap)} ids are in both the inherited val and test sets")
        return held["val"], held["test"]

    logger.warning("%s is missing; falling back to a fresh random split", split_ids)
    rng = np.random.default_rng(seed)
    shuffled = ids.to_numpy().copy()
    rng.shuffle(shuffled)
    return pd.Index(shuffled[:val_size]), pd.Index(shuffled[val_size:val_size + test_size])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="lemat_bulk_fmax1",
                        help="Dataset name; the splits are written to data/<name>/.")
    parser.add_argument("--max-force", type=float, default=1.0,
                        help="Keep structures whose largest force component is at most this, eV/A.")
    parser.add_argument("--structure-csv", type=Path, default=DEFAULT_STRUCTURE_CSV)
    parser.add_argument("--energy-csv", type=Path,
                        default=REPO / "data" / "lemat-bulk" / "lemat_pbe_ehull.csv.gz")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS,
                        help="Cached label table; rebuilt from the CSVs when absent.")
    parser.add_argument("--rebuild-labels", action="store_true")
    parser.add_argument("--split-ids", type=Path, default=DEFAULT_SPLIT_IDS,
                        help="JSON of {split: [immutable_id]} to inherit val/test from.")
    parser.add_argument("--seed", type=int, default=20260906,
                        help="Only used when --split-ids is missing.")
    parser.add_argument("--val-size", type=int, default=100000)
    parser.add_argument("--test-size", type=int, default=100000)
    parser.add_argument("--chunk-size", type=int, default=250000,
                        help="Rows of the structure CSV held in memory at once. The CIFs are "
                             "most of the 4.3 GB, so they are streamed rather than joined.")
    parser.add_argument("--compresslevel", type=int, default=1,
                        help="gzip level for the output. 1 costs ~15%% more disk than 9 and "
                             "writes several times faster; these files are read once.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.rebuild_labels or not args.labels.exists():
        labels = build_labels(args.structure_csv, args.energy_csv, args.labels)
    else:
        logger.info("reading cached labels %s", args.labels)
        labels = pd.read_parquet(args.labels)

    kept = label_rows(labels, args.max_force)
    del labels
    kept = kept.set_index("immutable_id")
    if not kept.index.is_unique:
        raise ValueError("immutable_id is not unique in the label table")
    logger.info("distinct reduced formulas : %9d", kept["formula"].nunique())
    logger.info("delta_e_polymorph == 0    : %8.2f%%", 100 * float((kept["delta_e_polymorph"] == 0).mean()))

    val_ids, test_ids = assign_split(
        kept.index, args.split_ids, args.seed, args.val_size, args.test_size)
    split_of = pd.Series("train", index=kept.index)
    split_of[val_ids] = "val"
    split_of[test_ids] = "test"
    logger.info("split sizes: %s", split_of.value_counts().to_dict())

    out_dir = REPO / "data" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        name: gzip.open(out_dir / f"{name}.csv.gz", "wt", newline="",
                        compresslevel=args.compresslevel)
        for name in ("train", "val", "test")
    }
    written = dict.fromkeys(handles, 0)
    try:
        for name, handle in handles.items():
            handle.write(",".join(OUTPUT_COLUMNS) + "\n")
        reader = pd.read_csv(
            args.structure_csv, usecols=["immutable_id", "cif"], chunksize=args.chunk_size)
        for index, chunk in enumerate(reader):
            chunk = chunk.set_index("immutable_id")
            chunk = chunk[chunk.index.isin(kept.index)]
            if chunk.empty:
                continue
            block = chunk.join(kept[
                [
                    "energy_above_hull",
                    "delta_e_polymorph",
                    "max_force",
                    "max_force_missing",
                    "formation_energy_per_atom",
                ]
            ])
            block = block.reset_index()[OUTPUT_COLUMNS]
            for name, handle in handles.items():
                rows = block[split_of.loc[block["immutable_id"]].to_numpy() == name]
                if rows.empty:
                    continue
                rows.to_csv(handle, header=False, index=False)
                written[name] += len(rows)
            if index % 4 == 0:
                logger.info("chunk %3d: written %s", index, written)
    finally:
        for handle in handles.values():
            handle.close()

    logger.info("final row counts: %s", written)
    expected = split_of.value_counts().to_dict()
    if written != expected:
        raise ValueError(f"wrote {written}, expected {expected}: the structure CSV is "
                         "missing rows the label table has")
    for name in handles:
        path = out_dir / f"{name}.csv.gz"
        logger.info("%s: %.1f MB", path, path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
