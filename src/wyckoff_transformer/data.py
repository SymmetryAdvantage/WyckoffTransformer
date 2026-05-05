"""Helpers for reading crystal datasets and deriving symmetry-site records."""

from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional
import logging
import warnings

import numpy as np
import pandas as pd
from pymatgen.core import Element, Structure
from pymatgen.io.cif import CifParser
from pyxtal import pyxtal

from wyckoff_transformer.preprocess_wychoffs import get_augmentation_dict
from wyckoff_transformer.tokenization import load_wyckoff_mappings


logger = logging.getLogger(__name__)


def read_cif(cif: str) -> Structure:
    """Read a CIF string into a pymatgen structure."""
    return CifParser.from_str(cif).parse_structures(primitive=False)[0]


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
        augmented_enumeration = [
            [
                wychoffs_enumerated_by_ss[pyxtal_record["group"]][augmentator[letter]]
                for letter in sites_dict["wyckoff_letters"]
            ]
            for augmentator in wychoffs_augmentation[pyxtal_record["group"]]
        ]
        sites_dict["sites_enumeration_augmented"] = frozenset(
            map(tuple, augmented_enumeration)
        )
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
        except AttributeError:
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
    max_wp: Optional[int] = None,
) -> dict:
    """Convert a structure to a symmetry-site record."""
    pyxtal_structure = kick_pyxtal_until_it_works(structure, tol=tol, a_tol=a_tol)
    if len(pyxtal_structure.atom_sites) == 0:
        raise ValueError("pyxtal failed to convert the structure to symmetry sites.")

    if max_wp is None:
        atom_sites = pyxtal_structure.atom_sites
    else:
        atom_sites = sorted(pyxtal_structure.atom_sites, key=lambda x: x.wp.letter)[:max_wp]
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
        augmented_enumeration = [
            [
                wychoffs_enumerated_by_ss[pyxtal_structure.group.number][augmentator[letter]]
                for letter in sites_dict["wyckoff_letters"]
            ]
            for augmentator in wychoffs_augmentation[pyxtal_structure.group.number]
        ]
        sites_dict["sites_enumeration_augmented"] = frozenset(
            map(tuple, augmented_enumeration)
        )
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


def compute_symmetry_sites(
    datasets_pd: dict[str, pd.DataFrame],
    n_jobs: Optional[int] = None,
    symmetry_precision: float = 0.1,
    symmetry_a_tol: float = 5.0,
    max_wp: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    """Compute symmetry-site records for one or more structure datasets."""
    wychoffs_enumerated_by_ss = load_wyckoff_mappings().enum_from_ss_letter

    structure_to_sites_with_args = partial(
        structure_to_sites,
        wychoffs_enumerated_by_ss=wychoffs_enumerated_by_ss,
        wychoffs_augmentation=get_augmentation_dict(),
        tol=symmetry_precision,
        a_tol=symmetry_a_tol,
        max_wp=max_wp,
    )
    result = {}
    for dataset_name, dataset in datasets_pd.items():
        with Pool(n_jobs) as pool:
            symmetry_list = pool.map(structure_to_sites_with_args, dataset.loc[:, "structure"])
        symmetry_dataset = pd.DataFrame.from_records(symmetry_list).set_index(dataset.index)
        symmetry_dataset["composition"] = symmetry_dataset.apply(
            get_composition_from_symmetry_sites, axis=1
        )
        if "formation_energy_per_atom" in dataset.columns:
            symmetry_dataset["formation_energy_per_atom"] = dataset["formation_energy_per_atom"]
        if "energy_above_hull" in dataset.columns:
            symmetry_dataset["energy_above_hull"] = dataset["energy_above_hull"]
        if "band_gap" in dataset.columns:
            symmetry_dataset["band_gap"] = dataset["band_gap"]
        if "log_klat" in dataset.columns:
            symmetry_dataset["log_klat"] = dataset["log_klat"]
        if "klat" in dataset.columns:
            symmetry_dataset["klat"] = dataset["klat"]
        result[dataset_name] = symmetry_dataset
    return result


def read_all_MP_csv(
    mp_path: Path = Path(__file__).resolve().parents[2] / "data" / "mp_20",
    n_jobs: Optional[int] = None,
    symmetry_precision: float = 0.1,
    symmetry_a_tol: float = 5.0,
    max_wp: Optional[int] = None,
) -> tuple[dict[str, pd.DataFrame], int]:
    """Read all split CSVs for a dataset and convert them to symmetry-site records."""
    datasets_pd = {}
    for dataset_name in ("train", "test", "val"):
        print(f"Reading dataset {dataset_name}...")
        try:
            datasets_pd[dataset_name] = read_MP(mp_path / f"{dataset_name}")
        except FileNotFoundError:
            logger.warning("Dataset %s not found.", dataset_name)
    print("Computing symmetry sites...")
    symmetry_datasets = compute_symmetry_sites(
        datasets_pd,
        n_jobs=n_jobs,
        symmetry_precision=symmetry_precision,
        symmetry_a_tol=symmetry_a_tol,
        max_wp=max_wp,
    )
    return symmetry_datasets