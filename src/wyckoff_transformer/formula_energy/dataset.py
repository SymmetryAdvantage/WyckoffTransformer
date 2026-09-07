"""Build the formula-level table: one row per reduced formula, keyed to its floor.

The input is a LeMat-Bulk energy CSV -- ``immutable_id, full_formula, chemsys,
energy_corrected, max_force, e_form, e_hull`` -- not the tokenised cache. Three
reasons. A composition model needs no Wyckoff sets, so the tokenisation is dead
weight. The cache is filtered for converged forces and tokenisable Wyckoff sets
in a way that costs most of the Materials Project rows (31,794 against the CSV's
138,931), and MP rows are the only ones carrying experimental provenance, which
is the scarce side of this problem. And ``e_form`` is in the CSV but not the
cache, while the cache's ``energy_above_hull`` is clipped at zero and so cannot
be inverted back to an absolute energy.

``max_force`` is joined from ``lemat_pbe.csv.gz``, where it actually has values
-- the column exists in the energy CSV but is NaN in every row of it.

``e_form - e_hull`` is the hull's formation energy at that composition. That is
not a coincidence to be checked at runtime but the definition of ``e_hull``, and
it means the energy a structure has to beat is available per row with no phase
diagram: :func:`build_formula_table` carries it through as
``e_hull_at_composition`` and asserts it is constant within a formula.

Aggregation is over the *reduced* formula, so ``BaTiO3`` and ``Ba2Ti2O6`` are one
compound with two polymorphs rather than two compounds. Splits are assigned by
hashing that key, so no formula can appear on both sides -- the shipped cache
split is row-random, and 62.9% of its test rows share a reduced formula with
train, which would let a formula-level model read its answers out of the
training set.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from wyckoff_transformer.csp import parse_formula, reduce_formula

logger = logging.getLogger(__name__)

#: The energy table produced by :mod:`wyckoff_transformer.formula_energy.hull_table`.
DEFAULT_ENERGY_CSV = Path("data/lemat-bulk/lemat_pbe_ehull.csv.gz")
#: Written by ``scripts/pull_mp_provenance.py``.
DEFAULT_PROVENANCE = Path("data/mp_provenance.csv.gz")
#: ``max_force`` lives here and is entirely NaN in the energy CSV, which is why
#: ``scripts/pipeline_lemat_20wyckoffs.py`` merges the two on ``immutable_id``.
DEFAULT_FORCE_CSV = Path("data/lemat-bulk/lemat_pbe.csv.gz")
#: Where :func:`build_formula_table` caches its result.
DEFAULT_TABLE = Path("data/formula_energy/formula_table.parquet")

#: Columns read from the energy CSV. Omitting ``cif`` is what keeps the read to
#: minutes rather than tens of minutes -- it is most of the 1.14 GB.
ENERGY_COLUMNS = ("immutable_id", "full_formula", "chemsys", "e_form", "e_hull")

#: No force filter by default, against the precedent of
#: ``scripts/pipeline_lemat_20wyckoffs.py``, which cuts at 0.02. Measured on this
#: data, that cut is a provenance filter wearing a convergence costume: it keeps
#: 95.6% of Alexandria rows but only 35.5% of the ICSD-backed ones, because
#: Materials Project reports forces from a different protocol and its median is
#: 0.028 -- above the cut. Since the ICSD flag is the scarcest and most
#: informative channel we have, that trade is bad on its own.
#:
#: It is also unnecessary. Differencing each row against its own formula's median
#: energy, a large force goes with a *higher* energy, not a lower one -- mean
#: +0.34 eV/atom in the ``>0.5`` bucket against +0.02 below 0.02, and the share of
#: anomalously low rows falls from 12.7% to 3.1% as force rises. An unconverged
#: relaxation has not reached the minimum yet, so it cannot corrupt one. The cut
#: would discard 134k formulas outright and change the label of 2.8% of the rest.
#: Corrupt energies are caught by :data:`DEFAULT_MAX_ABS_E_FORM` instead, and
#: ``max_force`` survives as a covariate on the excess-scale head, where a
#: poorly converged bound can be learned to be a loose one.
DEFAULT_MAX_FORCE = None

#: Formation energies outside this window are corrupt, not exotic: the file runs
#: to -37.9 and +650.7 eV/atom, where real formation energies sit inside a few
#: eV. Wren applies the same cleaning ("Formation Energy less than 5 eV per
#: atom") before training, and it matters more here, because a minimum takes the
#: single most negative value rather than averaging it away.
DEFAULT_MAX_ABS_E_FORM = 5.0

#: The four provenance channels, in the order they appear in the feature vector.
#: They are different search processes, not different amounts of one process: MP
#: is ICSD-seeded plus targeted studies, OQMD is a prototype library, Alexandria
#: is mass substitution. Collapsing them into a scalar n(X) throws away the
#: distinction that predicts how loose the bound is.
KINDS = ("mp_icsd", "mp_theoretical", "oqmd", "agm")


def parse_full_formula(full_formula: str) -> Counter:
    """``"Li12 As4 H64 S16 O32"`` into a Counter of element symbol to count.

    LeMat-Bulk's ``full_formula`` is ``pymatgen``'s ``Composition.formula``, which
    separates elements with spaces. :func:`wyckoff_transformer.csp.parse_formula`
    rejects those, since a gap between tokens is how it catches a typo, so the
    spaces are stripped before it sees the string.
    """
    return parse_formula(full_formula.replace(" ", ""))


def formula_key(counts: Counter) -> str:
    """Canonical key for a composition: reduced, alphabetical, explicit counts.

    ``As1H16Li3O8S4``. Explicit ``1``s and a fixed element order make the string
    a bijection with the reduced composition, so it can be used as a groupby key
    and parsed back without a lookup table.
    """
    return "".join(f"{symbol}{count}" for symbol, count in sorted(reduce_formula(counts).items()))


def _keys_and_sizes(full_formulas: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Reduced-formula key and cell size for each row, parsing each string once.

    ``full_formula`` repeats across polymorphs and across cell multiples, so
    factorising first turns 5.3M parses into one per distinct string.
    """
    codes, uniques = pd.factorize(full_formulas, sort=False)
    keys = np.empty(len(uniques), dtype=object)
    sizes = np.empty(len(uniques), dtype=np.int64)
    # reduce_formula announces every reduction, which is useful when a person
    # typed a pre-multiplied formula at the CSP CLI and is noise here: cell
    # multiples are what an archive of polymorphs is made of, and there are two
    # million of them.
    csp_logger = logging.getLogger("wyckoff_transformer.csp")
    previous = csp_logger.level
    csp_logger.setLevel(logging.WARNING)
    try:
        for index, text in enumerate(uniques):
            counts = parse_full_formula(text)
            keys[index] = formula_key(counts)
            sizes[index] = sum(counts.values())
    finally:
        csp_logger.setLevel(previous)
    return keys[codes], sizes[codes]


def _kinds(immutable_ids: pd.Series, provenance: pd.DataFrame) -> pd.Series:
    """Label each row with the search process that produced it.

    The id prefix names the source database. Within Materials Project the
    ``theoretical`` flag splits ICSD-descended entries from computed ones, and
    that split carries information a row count does not: paired within formula,
    an ICSD-backed entry is the archive minimum 56.9% of the time against 11.4%
    for a theoretical MP entry in the same formula.
    """
    theoretical = immutable_ids.map(provenance["theoretical"])
    return pd.Series(
        np.select(
            [
                theoretical.eq(False).to_numpy(),
                theoretical.eq(True).to_numpy(),
                # An mp- id the API no longer resolves -- about a thousand of
                # them -- is unknown, not Alexandria. Calling it theoretical is
                # the conservative reading: it withholds a claim of experimental
                # backing rather than inventing a provenance the row does not
                # have, and it keeps these rows out of the g_C population.
                immutable_ids.str.startswith("mp-", na=False).to_numpy(),
                immutable_ids.str.startswith("oqmd", na=False).to_numpy(),
            ],
            ["mp_icsd", "mp_theoretical", "mp_theoretical", "oqmd"],
            default="agm",
        ),
        index=immutable_ids.index,
        dtype="object",
    )


def load_rows(
    energy_csv: Path = DEFAULT_ENERGY_CSV,
    provenance_csv: Path = DEFAULT_PROVENANCE,
    force_csv: Optional[Path] = DEFAULT_FORCE_CSV,
    max_force: Optional[float] = DEFAULT_MAX_FORCE,
    max_abs_e_form: Optional[float] = DEFAULT_MAX_ABS_E_FORM,
    sources: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Read the energy CSV, drop what cannot be used, and label the provenance.

    Args:
        energy_csv: LeMat-Bulk energies with ``e_form`` and ``e_hull``.
        provenance_csv: ``material_id, theoretical, n_icsd`` from Materials Project.
        force_csv: Where to take ``max_force`` from, or ``None`` to do without it
            and lose both the convergence filter and the force covariates.
        max_force: Drop rows above this force, or ``None`` to keep all.
        max_abs_e_form: Drop rows outside this formation-energy window, or
            ``None`` to keep all. See :data:`DEFAULT_MAX_ABS_E_FORM`.
        sources: Keep only these :data:`KINDS`. ``("mp_icsd", "mp_theoretical",
            "oqmd")`` builds the shallow world used as the screening answer key.

    Returns:
        One row per usable structure, with ``formula``, ``cell_size``, ``kind``
        and ``max_force`` added.
    """
    frame = pd.read_csv(energy_csv, usecols=list(ENERGY_COLUMNS), low_memory=False)
    logger.info("read %d rows from %s", len(frame), energy_csv)

    # The CSV was written next to a cif column, and a handful of rows carry a
    # non-numeric string in the numeric fields. Coercing rather than trusting the
    # inferred dtype keeps one malformed row from turning a whole column to
    # object and the comparison below into a TypeError.
    for column in ("e_form", "e_hull"):
        coerced = pd.to_numeric(frame[column], errors="coerce")
        unparseable = int(coerced.isna().sum() - frame[column].isna().sum())
        if unparseable:
            logger.warning("%d rows have an unreadable %s", unparseable, column)
        frame[column] = coerced

    # e_form is NaN wherever no phase diagram could be built for the row --
    # Yb, the polonium-and-beyond elements, and chemical systems of ten or more
    # elements. Without a formation energy the row cannot bound anything.
    frame = frame[frame["e_form"].notna() & frame["e_hull"].notna()]
    if max_abs_e_form is not None:
        corrupt = int((frame["e_form"].abs() > max_abs_e_form).sum())
        if corrupt:
            logger.info("dropping %d rows with |e_form| > %g", corrupt, max_abs_e_form)
        frame = frame[frame["e_form"].abs() <= max_abs_e_form]

    if force_csv is None:
        frame["max_force"] = np.nan
    else:
        forces = pd.read_csv(force_csv, usecols=["immutable_id", "max_force"], low_memory=False)
        forces = forces.set_index("immutable_id")["max_force"]
        frame["max_force"] = pd.to_numeric(frame["immutable_id"].map(forces), errors="coerce").to_numpy()
        if max_force is not None:
            keep = frame["max_force"].notna() & (frame["max_force"] <= max_force)
            logger.info("force filter keeps %d of %d rows", int(keep.sum()), len(frame))
            frame = frame[keep]

    provenance = pd.read_csv(provenance_csv, index_col="material_id")
    frame["kind"] = _kinds(frame["immutable_id"], provenance).to_numpy()
    frame["n_icsd"] = frame["immutable_id"].map(provenance["n_icsd"]).fillna(0).astype(np.int32)
    if sources is not None:
        frame = frame[frame["kind"].isin(sources)]

    frame["formula"], frame["cell_size"] = _keys_and_sizes(frame["full_formula"])
    logger.info("%d usable rows, %d distinct formulas", len(frame), frame["formula"].nunique())
    return frame.drop(columns=["full_formula"])


def build_formula_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Collapse structures to one row per reduced formula.

    The label is ``e_form_min``: the lowest formation energy anyone has recorded
    for this composition, which is an upper bound on ``f*(X)``, tight to an
    unknown degree. Everything else in the table is either the chemistry the
    location head is allowed to see, or evidence about how loose that bound is.

    Args:
        rows: Output of :func:`load_rows`.

    Returns:
        A frame indexed by reduced formula.
    """
    if rows.empty:
        raise ValueError("No rows to aggregate")

    grouped = rows.groupby("formula", sort=False)
    table = pd.DataFrame({
        "e_form_min": grouped["e_form"].min(),
        "n_rows": grouped.size(),
        "n_cell_sizes": grouped["cell_size"].nunique(),
        "n_icsd": grouped["n_icsd"].sum(),
        "max_force_min": grouped["max_force"].min(),
        "max_force_median": grouped["max_force"].median(),
        "chemsys": grouped["chemsys"].first(),
    })

    # e_form - e_hull is the hull's formation energy at this composition. It is a
    # property of the composition, so every row of a formula must agree; a
    # disagreement means the rows were keyed wrongly or came from two hulls.
    hull = rows["e_form"] - rows["e_hull"]
    spread = hull.groupby(rows["formula"], sort=False).agg(lambda values: values.max() - values.min())
    if spread.max() > 1e-6:
        offenders = spread.nlargest(3)
        raise ValueError(
            f"e_form - e_hull is not constant within a formula (max spread {spread.max():.3g}); "
            f"worst: {offenders.to_dict()}"
        )
    table["e_hull_at_composition"] = hull.groupby(rows["formula"], sort=False).first()

    counts = (
        rows.groupby(["formula", "kind"], sort=False).size().unstack("kind", fill_value=0)
        .reindex(columns=list(KINDS), fill_value=0)
    )
    table = table.join(counts.add_prefix("n_"))

    # Which process produced the minimum, and how far the best experimentally
    # backed entry sits above it. Together these are the direct test of "the
    # observed structure is the ground state": the assumption holds exactly when
    # icsd_excess is zero.
    argmin_kind = rows.loc[grouped["e_form"].idxmin(), "kind"]
    argmin_kind.index = grouped["e_form"].idxmin().index
    table["argmin_kind"] = argmin_kind
    icsd = rows[rows["kind"] == "mp_icsd"]
    table["e_form_min_icsd"] = icsd.groupby("formula", sort=False)["e_form"].min()
    table["has_icsd"] = table["e_form_min_icsd"].notna()
    table["icsd_excess"] = table["e_form_min_icsd"] - table["e_form_min"]

    elements = table.index.to_series().map(lambda key: tuple(sorted(parse_formula(key))))
    table["n_elements"] = elements.str.len().astype(np.int16)
    return table


def assign_split(
    keys: Iterable[str],
    val_permille: int = 50,
    test_permille: int = 50,
    salt: str = "",
) -> pd.Series:
    """Deterministically split by hashing the grouping key.

    Hashing rather than sampling means the split survives a rebuild, a change of
    row filter and a change of source set without being stored anywhere: a
    formula lands in the same fold in the shallow world as in the deep one, which
    is what makes the two comparable.

    Args:
        keys: The grouping key per row -- the reduced formula, or the chemical
            system to hold out whole systems instead.
        val_permille: Parts per thousand assigned to validation.
        test_permille: Parts per thousand assigned to test.
        salt: Change to draw a different split.

    Returns:
        A Series of ``"train"``, ``"val"`` or ``"test"``.
    """
    if val_permille < 0 or test_permille < 0 or val_permille + test_permille >= 1000:
        raise ValueError(f"Nonsensical split sizes: {val_permille=}, {test_permille=}")
    keys = pd.Index(keys)
    buckets = np.fromiter(
        (
            int.from_bytes(hashlib.blake2b((salt + key).encode(), digest_size=8).digest(), "big") % 1000
            for key in keys
        ),
        dtype=np.int64,
        count=len(keys),
    )
    return pd.Series(
        np.select(
            [buckets < val_permille, buckets < val_permille + test_permille],
            ["val", "test"],
            default="train",
        ),
        index=keys,
        dtype="object",
    )


def build(
    energy_csv: Path = DEFAULT_ENERGY_CSV,
    provenance_csv: Path = DEFAULT_PROVENANCE,
    force_csv: Optional[Path] = DEFAULT_FORCE_CSV,
    max_force: Optional[float] = DEFAULT_MAX_FORCE,
    max_abs_e_form: Optional[float] = DEFAULT_MAX_ABS_E_FORM,
    sources: Optional[Sequence[str]] = None,
    split_by: str = "formula",
    salt: str = "",
) -> pd.DataFrame:
    """Read, aggregate and split in one call. See the functions it composes."""
    table = build_formula_table(
        load_rows(energy_csv, provenance_csv, force_csv, max_force, max_abs_e_form, sources)
    )
    # The archive defines the neighbourhood densities, so this is self-referential
    # and has to come after the table exists.
    from wyckoff_transformer.formula_energy.features import SystemDensity  # noqa: PLC0415
    table = SystemDensity.from_table(table).attach(table)
    if split_by == "formula":
        table["split"] = assign_split(table.index, salt=salt).to_numpy()
    elif split_by == "chemsys":
        by_system = assign_split(pd.unique(table["chemsys"]), salt=salt)
        table["split"] = table["chemsys"].map(by_system).to_numpy()
    else:
        raise ValueError(f"Unknown split_by: {split_by!r}, expected 'formula' or 'chemsys'")
    return table


def describe(table: pd.DataFrame) -> Dict[str, float]:
    """The numbers worth printing after a build, as a flat dict."""
    icsd = table[table["has_icsd"]]
    return {
        "formulas": len(table),
        "structures": int(table["n_rows"].sum()),
        "share_n_1": float((table["n_rows"] == 1).mean()),
        "formulas_with_icsd": len(icsd),
        "share_with_icsd": float(table["has_icsd"].mean()),
        "icsd_is_argmin": float((icsd["icsd_excess"] <= 1e-9).mean()) if len(icsd) else math.nan,
        "icsd_beaten_by_50meV": float((icsd["icsd_excess"] > 0.05).mean()) if len(icsd) else math.nan,
        "icsd_excess_mean": float(icsd["icsd_excess"].mean()) if len(icsd) else math.nan,
        "hull_defining": float((table["e_form_min"] <= table["e_hull_at_composition"] + 1e-6).mean()),
        **{f"share_{split}": float((table["split"] == split).mean()) for split in ("train", "val", "test")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-csv", type=Path, default=DEFAULT_ENERGY_CSV)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--force-csv", type=Path, default=DEFAULT_FORCE_CSV,
                        help="where max_force comes from; the energy CSV has it as all-NaN")
    parser.add_argument("--max-force", type=float, default=-1.0 if DEFAULT_MAX_FORCE is None else DEFAULT_MAX_FORCE,
                        help="drop rows above this force; negative keeps all (the default)")
    parser.add_argument("--max-abs-e-form", type=float, default=DEFAULT_MAX_ABS_E_FORM,
                        help="drop rows outside this formation-energy window; negative keeps all")
    parser.add_argument("--sources", nargs="+", choices=KINDS, default=None,
                        help=f"keep only these; {' '.join(KINDS[:3])} is the shallow world")
    parser.add_argument("--split-by", choices=("formula", "chemsys"), default="formula")
    parser.add_argument("--salt", default="")
    parser.add_argument("--check", action="store_true", help="build and report, write nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    table = build(
        args.energy_csv, args.provenance, args.force_csv,
        None if args.max_force < 0 else args.max_force,
        None if args.max_abs_e_form < 0 else args.max_abs_e_form,
        args.sources, args.split_by, args.salt,
    )
    width = max(len(name) for name in describe(table))
    for name, value in describe(table).items():
        print(f"  {name:<{width}}  {value:,.4f}" if isinstance(value, float) else f"  {name:<{width}}  {value:,}")
    if not args.check:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
