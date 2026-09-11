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
    1.0 where no forces are known, 0.0 elsewhere. 30,679 rows -- 22.1% of Materials
    Project, and nothing from Alexandria or OQMD -- carry an empty ``forces`` array and an
    empty ``stress_tensor`` in the archive. They were never absent at the source: LeMat
    read the task document's top-level ``output.forces``, which the 2013-2017 legacy tasks
    leave empty, while the last ionic step of the same calculation keeps both.
    ``scripts/recover_mp_forces.py`` reads them back from MP's public S3 task documents
    for 30,676 of the rows, matched to LeMat's geometry within 1e-5 A and its energy
    within 1e-5 eV, and reproduces the archived arrays bit for bit on 800 control rows
    that were never missing. ``--recovered-forces`` (the default) uses them; three rows
    stay ambiguous.

    What is still missing is imputed as the median over *its own source database*, with
    the indicator saying so. Imputing zero would have filed those rows under the
    symmetry-locked mode above. The imputation is the older dataset's fallback, and it was
    wrong for the rows it covered: the median ``max_force`` of the recovered rows is
    0.088, twice the 0.0417 imputed, and 356 of them exceed the 1 eV/A cut. A
    ``max_force <= X`` filter on unrecovered rows would drop them for any X, since NaN
    fails every comparison, so they are kept rather than filtered.

``stress_hydrostatic``, ``stress_von_mises``, ``stress_missing``
    The residual stress, kBar, in the archive's VASP sign (positive: the cell is compressed
    and wants to expand): its hydrostatic part ``tr(sigma)/3`` and the von Mises equivalent
    of its deviator. Residual stress, not force, sets how much energy an unfinished
    relaxation leaves on the table -- a gradient-matched ORB estimate over 1,004 rows put
    98% of it in the cell block, rank-correlated 0.81 with stress against 0.31 with force
    -- and it is informative on the symmetry-locked rows where ``max_force`` is
    identically zero. It is not purely a convergence label, though: OQMD's hydrostatic
    stress is one-signed (97% positive above 5 kBar, median +7.5 against MP's +0.02) and
    scales with pseudopotential hardness (median +62 kBar with F, +3 to +4 with K, Cs, I,
    Br) rather than with the residual force: the relaxation ended at the minimum of a stale
    basis or of different settings than the calculation reporting the energy (for
    Alexandria, LeMat's row is a separate calculation at the path's final geometry). The
    label is therefore provenance-laden, but not spurious -- it comes from the same
    calculation as the energy, so the energy it implies is real on the label's own surface.
    ``stress_missing`` marks the same rows as ``max_force_missing``.

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

from wyckoff_transformer.paths import cache_root, data_root

logger = logging.getLogger("build_lemat_bulk_fmax")

REPO = Path(__file__).resolve().parent.parent
#: ``max_force`` and the CIFs. The energy CSV has a ``max_force`` column too, and it is
#: NaN in every one of its rows.
DEFAULT_STRUCTURE_CSV = data_root() / "lemat-bulk" / "lemat_pbe.csv.gz"
#: Written by ``scripts/build_lemat_bulk_fmax.py --labels-only`` (or the scratch probe):
#: one row per structure with the reduced formula, cell size and energies, no CIF.
DEFAULT_LABELS = data_root() / "lemat-bulk" / "labels.parquet"
DEFAULT_SPLIT_IDS = cache_root() / "lemat_bulk_ehull" / "split_ids.json"
#: LeMat-Bulk as downloaded: per-atom ``forces`` (eV/A) and ``stress_tensor`` (kBar, VASP
#: sign), which the structure CSV only summarises as ``max_force``.
DEFAULT_RAW = data_root() / "lemat-bulk" / "raw" / "data.parquet"
#: Written by ``scripts/recover_mp_forces.py``: forces and stress for the MP rows whose
#: archived arrays are empty, read from the last ionic step of the very task LeMat took the
#: energy from.
DEFAULT_RECOVERED = cache_root() / "mp_forces_recovery" / "runs" / "full" / "results.parquet"
DEFAULT_CONVERGENCE_LABELS = data_root() / "lemat-bulk" / "convergence_labels.parquet"

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
    "stress_hydrostatic",
    "stress_von_mises",
    "stress_missing",
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


def _max_abs_component(column) -> np.ndarray:
    """Largest |component| of a ``list<list<double>>`` arrow column per row; NaN when empty."""
    import pyarrow.compute as pc

    array = column.combine_chunks()
    inner = array.flatten()
    row_of_inner = pc.list_parent_indices(array).to_numpy()
    values = inner.flatten().fill_null(np.nan).to_numpy(zero_copy_only=False)
    row_of_value = row_of_inner[pc.list_parent_indices(inner).to_numpy()]
    out = np.full(len(array), -np.inf)
    finite = np.isfinite(values)
    np.maximum.at(out, row_of_value[finite], np.abs(values[finite]))
    out[out == -np.inf] = np.nan
    return out


def _stress_invariants(stress: np.ndarray) -> dict:
    """Hydrostatic part and von Mises equivalent of ``(n, 3, 3)`` stresses, kBar, VASP sign."""
    stress = 0.5 * (stress + np.transpose(stress, (0, 2, 1)))
    hydrostatic = np.trace(stress, axis1=1, axis2=2) / 3.0
    deviator = stress - hydrostatic[:, None, None] * np.eye(3)
    return {"stress_hydrostatic": hydrostatic,
            "stress_von_mises": np.sqrt(1.5 * np.einsum("nij,nij->n", deviator, deviator))}


def build_convergence_labels(raw: Path, recovered: Path | None, out: Path) -> pd.DataFrame:
    """Per-row ``max_force`` and stress invariants, with the archive's gaps filled from MP.

    ``convergence_source`` says where each row's numbers came from: ``archive``,
    ``mp_task_doc`` (recovered), or ``missing``.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    logger.info("reading forces and stress from %s", raw)
    parquet = pq.ParquetFile(raw)
    parts = []
    for group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(group, columns=["immutable_id", "forces", "stress_tensor"])
        column = table.column("stress_tensor").combine_chunks()
        present = pc.list_value_length(column).fill_null(0).to_numpy() == 3
        stress = np.full((table.num_rows, 3, 3), np.nan)
        stress[present] = column.flatten().flatten().to_numpy(zero_copy_only=False).reshape(-1, 3, 3)
        parts.append(pd.DataFrame({
            "immutable_id": table.column("immutable_id").to_pandas(),
            "max_force": _max_abs_component(table.column("forces")),
            **_stress_invariants(stress),
        }))
    frame = pd.concat(parts, ignore_index=True).dropna(subset=["immutable_id"])
    frame = frame.set_index("immutable_id")
    if not frame.index.is_unique:
        raise ValueError(f"immutable_id is not unique in {raw}")
    frame["convergence_source"] = np.where(frame["max_force"].notna(), "archive", "missing")

    if recovered is not None:
        docs = pd.read_parquet(recovered, columns=[
            "immutable_id", "group", "status", "forces", "stress", "control_dF", "control_dS"])
        control = docs[docs["group"] != "missing"]
        logger.info("recovery control: %d rows, max |dF| %.3g, max |dS| %.3g against the archive",
                    len(control), control["control_dF"].abs().max(), control["control_dS"].abs().max())
        docs = docs[(docs["group"] == "missing") & (docs["status"] == "exact_forces_stress")]
        docs = docs.set_index("immutable_id")
        gaps = docs.index[frame.loc[docs.index, "max_force"].isna()]
        if len(gaps) != len(docs):
            raise ValueError(f"{len(docs) - len(gaps)} recovered rows already have archived forces")
        forces = docs.loc[gaps, "forces"].map(lambda f: float(np.abs(np.stack(f)).max()))
        stress = np.stack([np.stack(s).astype(float) for s in docs.loc[gaps, "stress"]])
        frame.loc[gaps, "max_force"] = forces.to_numpy()
        for name, values in _stress_invariants(stress).items():
            frame.loc[gaps, name] = values
        frame.loc[gaps, "convergence_source"] = "mp_task_doc"
    logger.info("convergence sources: %s", frame["convergence_source"].value_counts().to_dict())
    frame = frame.reset_index()
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    logger.info("wrote %s", out)
    return frame


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


def attach_convergence_labels(labels: pd.DataFrame, convergence: pd.DataFrame) -> pd.DataFrame:
    """Take ``max_force`` from the convergence table, and add the stress invariants.

    Where the structure CSV already has ``max_force`` the two are the same arrays read
    twice, so they must agree; that doubles as a check on the join.
    """
    merged = labels[["immutable_id", "max_force"]].merge(
        convergence, on="immutable_id", how="left", suffixes=("", "_convergence"),
        validate="one_to_one")
    # The one row without an immutable_id cannot be joined; label_rows drops it anyway.
    known = (merged["max_force"].notna() & merged["immutable_id"].notna()).to_numpy()
    if not np.allclose(merged.loc[known, "max_force"], merged.loc[known, "max_force_convergence"],
                       rtol=1e-6, atol=1e-9):
        raise ValueError("the structure CSV's max_force disagrees with the raw archive's forces")
    labels = labels.copy()
    labels["max_force"] = merged["max_force_convergence"].to_numpy()
    for column in ("stress_hydrostatic", "stress_von_mises", "convergence_source"):
        labels[column] = merged[column].to_numpy()
    return labels


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
    # Forces and stress are absent from exactly the same rows, and are filled the same way.
    kept["stress_missing"] = kept["stress_hydrostatic"].isna().astype(float)
    logger.info("  of which stress is absent: %8d", int(kept["stress_missing"].sum()))
    for column in ("stress_hydrostatic", "stress_von_mises"):
        by_source = kept.groupby(_source(kept["immutable_id"]), sort=False)[column]
        kept[column] = kept[column].fillna(by_source.transform("median")).fillna(
            kept[column].median())
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
                        default=data_root() / "lemat-bulk" / "lemat_pbe_ehull.csv.gz")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS,
                        help="Cached label table; rebuilt from the CSVs when absent.")
    parser.add_argument("--rebuild-labels", action="store_true")
    parser.add_argument("--raw-parquet", type=Path, default=DEFAULT_RAW,
                        help="LeMat-Bulk parquet carrying the per-atom forces and stress tensors.")
    parser.add_argument("--recovered-forces", type=lambda s: None if s.lower() == "none" else Path(s),
                        default=DEFAULT_RECOVERED,
                        help="scripts/recover_mp_forces.py results, filling the MP rows whose "
                             "archived forces and stress are empty. 'none' keeps them missing "
                             "(imputed and flagged), which is how lemat_bulk_fmax1 was built.")
    parser.add_argument("--convergence-labels", type=Path, default=DEFAULT_CONVERGENCE_LABELS,
                        help="Cached per-row max_force and stress table; rebuilt when absent.")
    parser.add_argument("--rebuild-convergence-labels", action="store_true")
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

    convergence_path = args.convergence_labels
    if args.recovered_forces is None:
        # A table built without the recovered rows must not be read back as one built with them.
        convergence_path = convergence_path.with_name(convergence_path.stem + "_archive_only.parquet")
    if args.rebuild_convergence_labels or not convergence_path.exists():
        convergence = build_convergence_labels(args.raw_parquet, args.recovered_forces, convergence_path)
    else:
        logger.info("reading cached convergence labels %s", convergence_path)
        convergence = pd.read_parquet(convergence_path)
    labels = attach_convergence_labels(labels, convergence)
    del convergence

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

    out_dir = data_root() / args.name
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
                    "stress_hydrostatic",
                    "stress_von_mises",
                    "stress_missing",
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
