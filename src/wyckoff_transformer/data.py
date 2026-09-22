"""Helpers for reading crystal datasets and deriving symmetry-site records."""

from collections import Counter
from functools import lru_cache, partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence, Tuple
import logging
import warnings

import numpy as np
import pandas as pd
from pymatgen.core import Element, Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SymmetryUndeterminedError
from pyxtal import pyxtal

from wyckoff_transformer.preprocess_wychoffs import get_augmentation_dict
from wyckoff_transformer.tokenization import load_wyckoff_mappings


logger = logging.getLogger(__name__)


def read_cif(cif: str) -> Structure:
    """Read a CIF string into a pymatgen structure."""
    return CifParser.from_str(cif).parse_structures(primitive=False)[0]


#: Written alongside ``sites_enumeration_augmented`` and aligned with it index for
#: index.  See :func:`augmented_sites` for why it has to exist.
AUGMENTED_SITE_SYMMETRIES = "site_symmetries_augmented"
AUGMENTED_SITES_ENUMERATION = "sites_enumeration_augmented"


@lru_cache(maxsize=1)
def _ss_from_letter() -> dict:
    from wyckoff_transformer.tokenization import load_wyckoff_mappings  # noqa: PLC0415

    return load_wyckoff_mappings().ss_from_letter


def augmented_sites(
    space_group: int,
    wyckoff_letters: Sequence[str],
    wychoffs_enumerated_by_ss: dict,
    wychoffs_augmentation: dict,
    ss_from_letter: Optional[dict] = None,
) -> Tuple[Tuple[tuple, ...], Tuple[tuple, ...]]:
    """The equivalent descriptions of one gene, as *paired* symmetry and enumeration.

    A Wyckoff position is identified by its space group, its oriented
    site-symmetry symbol and its index *within that symbol*.  The relabellings in
    ``wychoffs_augmentation`` permute letters, and in 26 orthorhombic space groups
    a relabelling sends a letter to one with a **different** symbol -- ``2..`` to
    ``.2.`` -- because the normaliser permutes the crystal axes.  International
    Tables makes the same point with ``I222``.

    So an augmentation cannot be expressed as a new enumeration vector alone: the
    symbol has to travel with it, or the pair ``(old symbol, new index)`` names a
    third position that is neither the original nor its image.  That is what this
    function exists to prevent, and
    ``docs/wyckoff_augmentation_audit.md`` records what it cost to find.

    Returns:
        ``(site_symmetries, sites_enumeration)``, each a tuple of per-variant
        tuples, aligned index for index.  Variants are deduplicated **as pairs**
        -- two relabellings that agree on the indices but not on the symbols are
        genuinely different descriptions -- and sorted, so the order does not
        depend on the iteration order of the frozenset the relabellings are held
        in and a cache is reproducible.
    """
    if ss_from_letter is None:
        ss_from_letter = _ss_from_letter()
    symbols = ss_from_letter[space_group]
    enumerations = wychoffs_enumerated_by_ss[space_group]
    variants = {
        (
            tuple(symbols[augmentator[letter]] for letter in wyckoff_letters),
            tuple(enumerations[augmentator[letter]] for letter in wyckoff_letters),
        )
        for augmentator in wychoffs_augmentation[space_group]
    }
    ordered = sorted(variants)
    return tuple(pair[0] for pair in ordered), tuple(pair[1] for pair in ordered)


def pyxtal_notation_to_sites(
    pyxtal_record: dict,
    wychoffs_enumerated_by_ss: dict,
    ss_from_letter: dict,
    wychoffs_augmentation: dict = None,
) -> dict:
    site_symmetries = []
    elements = []
    sites_enumeration = []
    multiplicity = []
    wyckoff_letters = []
    for this_element, sites in zip(pyxtal_record["species"], pyxtal_record["sites"]):
        true_element = Element(this_element)
        for this_site in sites:
            elements.append(true_element)
            letter = this_site[-1]
            wyckoff_letters.append(letter)
            multiplicity.append(int(this_site[:-1]))
            sites_enumeration.append(
                wychoffs_enumerated_by_ss[pyxtal_record["group"]][letter]
            )
            ss = ss_from_letter[pyxtal_record["group"]][letter]
            site_symmetries.append(ss)

    sites_dict = {
        "site_symmetries": site_symmetries,
        "elements": elements,
        "multiplicity": multiplicity,
        "wyckoff_letters": wyckoff_letters,
        "sites_enumeration": sites_enumeration,
        "spacegroup_number": pyxtal_record["group"],
    }
    if wychoffs_augmentation is not None:
        symmetries, enumerations = augmented_sites(
            pyxtal_record["group"], sites_dict["wyckoff_letters"],
            wychoffs_enumerated_by_ss, wychoffs_augmentation, ss_from_letter)
        sites_dict[AUGMENTED_SITE_SYMMETRIES] = symmetries
        sites_dict[AUGMENTED_SITES_ENUMERATION] = enumerations
    return sites_dict


def kick_pyxtal_until_it_works(
    structure: Structure,
    tol: float = 0.1,
    a_tol: float = 5.0,
    attempts: int = 30,
) -> pyxtal:
    """Retry pyxtal conversion across a range of tolerances."""
    n_down_multipliers = attempts // 2
    tolerances = np.empty(attempts)
    tolerances[::2] = np.logspace(0, 2, attempts - n_down_multipliers)
    tolerances[1::2] = np.logspace(-0.01, -6, n_down_multipliers)

    for attempt, tolerance in enumerate(tolerances):
        try:
            pyxtal_structure = pyxtal()
            pyxtal_structure.from_seed(structure, tol=tol * tolerance, a_tol=a_tol)
            return pyxtal_structure
        except (AttributeError, SymmetryUndeterminedError):
            logger.exception(
                "Attempt %i failed to convert structure %s to symmetry sites with tolerance %f.",
                attempt,
                structure,
                tol,
            )
    raise RuntimeError("Failed to make pyxtal work.")


def structure_to_sites(
    structure: Structure,
    wychoffs_enumerated_by_ss: dict,
    wychoffs_augmentation: Optional[dict] = None,
    tol: float = 0.1,
    a_tol: float = 5.0,
    sort_by_letter: bool = False,
) -> dict:
    """Convert a structure to a symmetry-site record.

    Every site of the structure is in the record. There is deliberately no way to
    truncate one: keeping the first N Wyckoff positions bounds the sequence length by
    silently changing the composition, so the row's energy labels come to describe a
    compound that is not in it. `wyformer-cache-dataset --max-sites` drops an over-long
    structure instead.

    `sort_by_letter` orders the sites by Wyckoff letter rather than however pyxtal
    returned them. It is what every cache now on disk was built with, and therefore what
    makes a new one comparable with them; it defaults to off so that the callers that
    symmetrise a generated structure keep pyxtal's order.
    """
    pyxtal_structure = kick_pyxtal_until_it_works(structure, tol=tol, a_tol=a_tol)
    if len(pyxtal_structure.atom_sites) == 0:
        raise ValueError("pyxtal failed to convert the structure to symmetry sites.")

    if sort_by_letter:
        atom_sites = sorted(pyxtal_structure.atom_sites, key=lambda x: x.wp.letter)
    else:
        atom_sites = list(pyxtal_structure.atom_sites)
    elements = []
    wyckoffs = []
    site_symmetries = []
    for site in atom_sites:
        site.wp.get_site_symmetry()
        wyckoffs.append(site.wp)
        elements.append(Element(site.specie))
        site_symmetries.append(site.wp.site_symm)
    site_enumeration = [
        wychoffs_enumerated_by_ss[pyxtal_structure.group.number][wp.letter]
        for wp in wyckoffs
    ]
    multiplicity = [wp.multiplicity for wp in wyckoffs]
    dof = [wp.get_dof() for wp in wyckoffs]

    sites_dict = {
        "site_symmetries": site_symmetries,
        "elements": elements,
        "multiplicity": multiplicity,
        "wyckoff_letters": [wp.letter for wp in wyckoffs],
        "sites_enumeration": site_enumeration,
        "dof": dof,
        "spacegroup_number": pyxtal_structure.group.number,
    }
    if wychoffs_augmentation is not None:
        symmetries, enumerations = augmented_sites(
            pyxtal_structure.group.number, sites_dict["wyckoff_letters"],
            wychoffs_enumerated_by_ss, wychoffs_augmentation)
        sites_dict[AUGMENTED_SITE_SYMMETRIES] = symmetries
        sites_dict[AUGMENTED_SITES_ENUMERATION] = enumerations
    return sites_dict


def read_MP(
    MP_csv: Path | str,
    n_jobs: Optional[int] = None,
    drop_na: bool = False,
) -> pd.DataFrame:
    """Read a Materials Project CSV into a dataframe with parsed structures."""
    try:
        MP_df = pd.read_csv(MP_csv, index_col=0)
    except FileNotFoundError:
        try:
            MP_df = pd.read_csv(f"{MP_csv}.csv", index_col=0)
        except FileNotFoundError:
            MP_df = pd.read_csv(f"{MP_csv}.csv.gz", index_col=0)

    if drop_na:
        print("Dropping NaN values in 'cif' column.")
        print(f"Initial number of rows: {len(MP_df)}")
        MP_df.dropna(subset=["cif"], inplace=True)
        print(f"Number of rows after dropping NaN values: {len(MP_df)}")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"No Pauling electronegativity for \w+. "
                "Setting to NaN. This has no physical meaning, and is "
                "mainly done to avoid errors caused by the code expecting a float."
            ),
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=(
                r"Issues encountered while parsing CIF: \d+"
                " fractional coordinates rounded to ideal"
                " values to avoid issues with finite precision."
            ),
            category=UserWarning,
        )
        print("Suppressed warnings: CIF rounding & Pauling electronegativity")
        with Pool(n_jobs) as pool:
            MP_df["structure"] = pool.map(read_cif, MP_df["cif"])
    MP_df.drop(columns=["cif"], inplace=True)
    return MP_df


def get_composition_from_symmetry_sites(record: pd.Series) -> dict:
    """Return an element-count dictionary from a symmetry-site record."""
    result = Counter()
    try:
        for element, multiplicity in zip(record["elements"], record["multiplicity"]):
            result[element] += multiplicity
    except TypeError:
        return None
    return result


def get_composition(structure: Structure) -> dict[Element, float]:
    """Return an element-count dictionary from a structure."""
    str_dict = structure.composition.get_el_amt_dict()
    return {Element(k): v for k, v in str_dict.items()}


#: Scalar columns carried from a source CSV into the cache without being asked for. These
#: predate `scalar_columns` and stay for the datasets already built on them; anything new
#: should be named explicitly instead of added here.
LEGACY_SCALAR_COLUMNS = (
    "formation_energy_per_atom",
    "energy_above_hull",
    "band_gap",
    "log_klat",
    "klat",
)


def compute_symmetry_sites(
    datasets_pd: dict[str, pd.DataFrame],
    n_jobs: Optional[int] = None,
    symmetry_precision: float = 0.1,
    symmetry_a_tol: float = 5.0,
    scalar_columns: Optional[Sequence[str]] = None,
    sort_by_letter: bool = False,
) -> dict[str, pd.DataFrame]:
    """Compute symmetry-site records for one or more structure datasets.

    `scalar_columns` names per-structure scalars to carry through to the cache alongside
    the symmetry records -- conditioning labels, mostly. They are copied verbatim, so a
    column must already hold what the model should see. Unlike LEGACY_SCALAR_COLUMNS,
    which are copied when present and skipped when not, a column named here and missing
    is an error: it is the difference between "this dataset happens to have band gaps"
    and "this run is conditioned on max_force".
    """
    wychoffs_enumerated_by_ss = load_wyckoff_mappings().enum_from_ss_letter

    structure_to_sites_with_args = partial(
        structure_to_sites,
        wychoffs_enumerated_by_ss=wychoffs_enumerated_by_ss,
        wychoffs_augmentation=get_augmentation_dict(),
        tol=symmetry_precision,
        a_tol=symmetry_a_tol,
        sort_by_letter=sort_by_letter,
    )
    result = {}
    for dataset_name, dataset in datasets_pd.items():
        with Pool(n_jobs) as pool:
            symmetry_list = pool.map(structure_to_sites_with_args, dataset.loc[:, "structure"])
        symmetry_dataset = pd.DataFrame.from_records(symmetry_list).set_index(dataset.index)
        symmetry_dataset["composition"] = symmetry_dataset.apply(
            get_composition_from_symmetry_sites, axis=1
        )
        for column in LEGACY_SCALAR_COLUMNS:
            if column in dataset.columns:
                symmetry_dataset[column] = dataset[column]
        for column in scalar_columns or ():
            if column not in dataset.columns:
                raise KeyError(
                    f"Split {dataset_name!r} has no column {column!r}; it holds "
                    f"{sorted(dataset.columns)}")
            symmetry_dataset[column] = dataset[column]
        result[dataset_name] = symmetry_dataset
    return result
