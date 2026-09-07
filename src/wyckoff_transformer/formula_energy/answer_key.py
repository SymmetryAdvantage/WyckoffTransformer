"""The answer key: which formulas a shallower archive would have been wrong about.

Screening claims to find formulas whose floor is lower than anyone currently
knows. That claim is normally unfalsifiable without running the search. This
module makes it falsifiable using the archive's own layers.

Hold out one search process. Rebuild the convex hull from what remains -- the
*shallow world*, Materials Project and OQMD, no Alexandria. Train on shallow
labels alone. Then ask which formulas the full archive later put below that
shallow hull. Those are real discoveries relative to the shallow world's state of
knowledge, and a screener that could not have predicted them would not have
predicted the ones still outstanding either.

This is Matbench Discovery's temporal logic without needing dates, and it is
generous with positives: Alexandria lowered the minimum by more than 50 meV/atom
for 36.3% of the formulas Materials Project and OQMD had already computed.

The hull has to be genuinely recomputed, not filtered. ``e_hull`` in the archive
is measured against the deep hull, and it is clipped at zero, so a structure that
sits *below* the shallow hull -- exactly the case of interest -- cannot be
recognised by subsetting rows.
:mod:`wyckoff_transformer.formula_energy.hull_table` rebuilds phase diagrams from
a reference set, which is what :func:`compute_shallow_energies` drives.

The two worlds also have to be put on one energy scale before they can be
compared. A formation energy is measured against elemental references, and those
are the lowest elemental entries *in that world*: withholding Alexandria raises
25 of them, by up to 36 meV/atom for bromine. Left uncorrected the comparison
mixes two scales, and it shows -- the raw ``drop`` has a slightly negative median,
which is impossible when one archive contains the other.
:func:`elemental_references` and :func:`frame_shift` translate the deep minimum
into the shallow frame; the shift is constant per composition, so it moves a
formula's whole set of polymorphs together and cannot change which one is lowest.

The key covers formulas the shallow world already knew about. A formula with no
shallow entry at all is a harder question -- pure extrapolation, with no
observable answer either way -- and is left to the screening run rather than
smuggled into the evaluation.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from wyckoff_transformer.formula_energy import dataset as ds
from wyckoff_transformer.formula_energy import hull_table as ht

logger = logging.getLogger(__name__)

#: Everything except Alexandria. MP is ICSD-seeded plus targeted studies, OQMD is
#: a prototype library; Alexandria is the mass-substitution campaign whose
#: findings become the answer key.
SHALLOW_SOURCES: Sequence[str] = ("mp_icsd", "mp_theoretical", "oqmd")

DEFAULT_DIR = Path("data/formula_energy")
#: The shallow subset, in the shape :mod:`hull_table` consumes.
SHALLOW_ROWS = DEFAULT_DIR / "shallow_rows.csv.gz"
#: The same rows with ``e_form`` and ``e_hull`` recomputed against themselves.
SHALLOW_ENERGIES = DEFAULT_DIR / "shallow_pbe_ehull.csv.gz"
SHALLOW_TABLE = DEFAULT_DIR / "shallow_table.parquet"
ANSWER_KEY = DEFAULT_DIR / "answer_key.parquet"

#: Columns the phase diagram needs. ``energy_corrected`` is the total energy it
#: is built from, and is not carried in the formula table.
SHALLOW_COLUMNS = ht.LIGHT_COLUMNS


def extract_shallow_rows(
    energy_csv: Path = ds.DEFAULT_ENERGY_CSV,
    provenance_csv: Path = ds.DEFAULT_PROVENANCE,
    out_csv: Path = SHALLOW_ROWS,
    sources: Sequence[str] = SHALLOW_SOURCES,
) -> Path:
    """Write the rows belonging to the held-in search processes.

    Returns:
        ``out_csv``.
    """
    frame = pd.read_csv(energy_csv, usecols=list(SHALLOW_COLUMNS), low_memory=False)
    provenance = pd.read_csv(provenance_csv, index_col="material_id")
    kinds = ds._kinds(frame["immutable_id"], provenance)
    frame = frame[kinds.isin(sources).to_numpy()]
    frame = frame.dropna(subset=["full_formula", "chemsys", "energy_corrected"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    logger.info("wrote %d shallow rows to %s", len(frame), out_csv)
    return out_csv


def compute_shallow_energies(
    rows_csv: Path = SHALLOW_ROWS,
    out_csv: Path = SHALLOW_ENERGIES,
    workers: int = 16,
) -> Path:
    """Rebuild formation energies and hull distances against the shallow set alone.

    The shallow set is its own reference, so ``e_hull`` here is a distance within
    the shallow world; what the key compares against it is the *deep* minimum,
    translated into the shallow frame by :func:`frame_shift`.

    Args:
        rows_csv: The shallow rows, from :func:`extract_shallow_rows`.
        out_csv: Destination, with ``e_form`` and ``e_hull`` appended.
        workers: Processes for the phase diagrams.

    Returns:
        *out_csv*.
    """
    return ht.annotate_csv(rows_csv, out_csv, workers=workers)


def elemental_references(
    energy_csv: Path = ds.DEFAULT_ENERGY_CSV,
    ids: Optional[set] = None,
) -> pd.Series:
    """Lowest total energy per atom among the unary entries, per element.

    This is what ``pymatgen`` uses as ``el_refs`` when it builds a phase diagram,
    and therefore what every formation energy in that world is measured against.

    Args:
        energy_csv: The rows to take references from.
        ids: Restrict to these ``immutable_id``s -- pass the shallow row ids to
            get the shallow world's references from the same file.
    """
    frame = pd.read_csv(
        energy_csv, usecols=["immutable_id", "full_formula", "chemsys", "energy_corrected"],
        low_memory=False,
    )
    frame["energy_corrected"] = pd.to_numeric(frame["energy_corrected"], errors="coerce")
    frame = frame.dropna(subset=["full_formula", "chemsys", "energy_corrected"])
    if ids is not None:
        frame = frame[frame["immutable_id"].isin(ids)]
    unary = frame[~frame["chemsys"].astype(str).str.contains("-")]
    atoms = unary["full_formula"].map(lambda text: sum(ds.parse_full_formula(text).values()))
    return unary.assign(per_atom=unary["energy_corrected"] / atoms).groupby("chemsys")["per_atom"].min()


def frame_shift(formulas: Sequence[str], delta: pd.Series) -> np.ndarray:
    """``sum_i f_i * (mu_i_deep - mu_i_shallow)``, the translation onto the shallow scale.

    Adding this to a deep formation energy gives the same structure's formation
    energy as the shallow world would have computed it.
    """
    shifts = np.zeros(len(formulas))
    lookup = delta.to_dict()
    for row, formula in enumerate(formulas):
        counts = ds.parse_formula(formula)
        total = sum(counts.values())
        shifts[row] = sum(count / total * lookup.get(symbol, 0.0) for symbol, count in counts.items())
    return shifts


def build_answer_key(
    deep: pd.DataFrame,
    shallow: pd.DataFrame,
    tolerance: float = 1e-6,
    delta: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Join the two worlds and mark what the shallow one was wrong about.

    Args:
        deep: Formula table over every source.
        shallow: Formula table over :data:`SHALLOW_SOURCES`, with ``e_form`` and
            ``e_hull`` recomputed against the shallow reference.
        tolerance: Slack on the comparison, eV/atom.
        delta: Per-element ``mu_deep - mu_shallow``, from
            :func:`elemental_references`. Without it the two worlds' formation
            energies are compared on different scales.

    Returns:
        One row per formula the shallow world knew about, carrying the shallow
        state of knowledge, the deep outcome, and ``discovered``.
    """
    shared = shallow.index.intersection(deep.index)
    if not len(shared):
        raise ValueError("The two worlds share no formulas; were they built from the same source?")

    shift = np.zeros(len(shared)) if delta is None else frame_shift(shared, delta)
    key = pd.DataFrame({
        "shallow_e_form_min": shallow.loc[shared, "e_form_min"],
        "shallow_e_hull": shallow.loc[shared, "e_hull_at_composition"],
        "shallow_n_rows": shallow.loc[shared, "n_rows"],
        "deep_e_form_min": deep.loc[shared, "e_form_min"].to_numpy() + shift,
        "deep_n_rows": deep.loc[shared, "n_rows"],
        "reference_shift": shift,
        "split": shallow.loc[shared, "split"],
    })
    # How much the withheld process lowered the floor, and whether it lowered it
    # far enough to change the shallow world's mind about stability.
    key["drop"] = key["shallow_e_form_min"] - key["deep_e_form_min"]
    key["discovered"] = key["deep_e_form_min"] < key["shallow_e_hull"] - tolerance
    # How far the shallow world's own best structure sits above the shallow hull:
    # the room a discovery has to occupy. Zero when the formula already defines
    # the hull, in which case nothing can be found below it.
    key["headroom"] = key["shallow_e_form_min"] - key["shallow_e_hull"]
    return key


def describe(key: pd.DataFrame) -> dict:
    """The numbers that say whether the evaluation has anything to detect."""
    return {
        "formulas": len(key),
        "discovered": int(key["discovered"].sum()),
        "discovered_rate": float(key["discovered"].mean()),
        "drop_gt_10meV": float((key["drop"] > 0.01).mean()),
        "drop_gt_50meV": float((key["drop"] > 0.05).mean()),
        "drop_median": float(key["drop"].median()),
        "drop_mean": float(key["drop"].mean()),
        # A negative drop is impossible once both worlds are on one scale: the
        # deep archive contains the shallow one, so its minimum cannot be higher.
        "drop_negative": float((key["drop"] < -1e-9).mean()),
        **{
            f"discovered_rate_{split}": float(key.loc[key["split"] == split, "discovered"].mean())
            for split in ("train", "val", "test")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-csv", type=Path, default=ds.DEFAULT_ENERGY_CSV)
    parser.add_argument("--provenance", type=Path, default=ds.DEFAULT_PROVENANCE)
    parser.add_argument("--deep-table", type=Path, default=ds.DEFAULT_TABLE)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-hull", action="store_true",
                        help="reuse an existing shallow energy CSV instead of recomputing it")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if not args.skip_hull:
        extract_shallow_rows(args.energy_csv, args.provenance)
        compute_shallow_energies(workers=args.workers)

    shallow = ds.build(energy_csv=SHALLOW_ENERGIES, provenance_csv=args.provenance)
    shallow.to_parquet(SHALLOW_TABLE)
    deep = pd.read_parquet(args.deep_table)

    shallow_ids = set(pd.read_csv(SHALLOW_ROWS, usecols=["immutable_id"])["immutable_id"])
    delta = (elemental_references(args.energy_csv)
             - elemental_references(args.energy_csv, ids=shallow_ids)).dropna()
    logger.info("%d elemental references moved by more than 1 meV/atom (worst %.4f)",
                int((delta.abs() > 1e-3).sum()), delta.abs().max())

    key = build_answer_key(deep, shallow, delta=delta)
    key.to_parquet(ANSWER_KEY)
    width = max(len(name) for name in describe(key))
    for name, value in describe(key).items():
        print(f"  {name:<{width}}  {value:,.4f}" if isinstance(value, float) else f"  {name:<{width}}  {value:,}")
    print(f"wrote {SHALLOW_TABLE} and {ANSWER_KEY}")


if __name__ == "__main__":
    main()
