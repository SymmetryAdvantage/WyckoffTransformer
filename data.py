"""
This module provides functions for reading the structures and extracting the symmetry information.
"""
from typing import Optional
from functools import partial
from collections import Counter
from pathlib import Path
import pickle
import warnings
import logging
from multiprocessing import Pool
import pandas as pd
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure, Element
from pyxtal import pyxtal

from preprocess_wychoffs import get_augmentation_dict

logger = logging.getLogger(__name__)

def read_cif(cif: str) -> Structure:
    """
    Reads a CIF file and returns the structure.

    Args:
        cif (str): The CIF file content.

    Returns:
        Structure: The structure.
    """
    return CifParser.from_str(cif).parse_structures(primitive=False)[0]

def pyxtal_notation_to_sites(
    pyxtal_record: dict,
    wychoffs_enumerated_by_ss: dict,
    ss_from_letter: dict,
    wychoffs_augmentation: dict = None) -> dict:

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
            sites_enumeration.append(wychoffs_enumerated_by_ss[pyxtal_record['group']][letter])
            ss = ss_from_letter[pyxtal_record['group']][letter]
            site_symmetries.append(ss)

    sites_dict = {
        "site_symmetries": site_symmetries,
        "elements": elements,
        "multiplicity": multiplicity,
        "wyckoff_letters": wyckoff_letters,
        "sites_enumeration": sites_enumeration,
        "spacegroup_number": pyxtal_record['group']
    }
    if wychoffs_augmentation is not None:
        augmented_enumeration = [
            [wychoffs_enumerated_by_ss[pyxtal_record['group']][augmentator[letter]] for
              letter in sites_dict["wyckoff_letters"]]
                for augmentator in wychoffs_augmentation[pyxtal_record['group']]
        ]
        sites_dict["sites_enumeration_augmented"] = frozenset(map(tuple, augmented_enumeration))
    return sites_dict


def structure_to_sites(
    structure: Structure,
    wychoffs_enumerated_by_ss: dict,
    wychoffs_augmentation: dict = None,
    tol: float = 0.1,
    return_none_on_exception: bool = False) -> dict:
    """
    Converts a pymatgen structure to a dictionary of symmetry sites.

    Args:
        structure (Structure): The pymatgen structure.
        wychoffs_enumerated_by_ss (dict)
        wychoffs_augmentation (dict, optional)
        tol (float, optional): The tolerance passed to pyxtal().from_seed.
            Defaults to 0.1, as in Materials Project.

    Returns:
        dict
    """
    pyxtal_structure = pyxtal()
    try:
        pyxtal_structure.from_seed(structure, tol=tol)
    except Exception:
        if return_none_on_exception:
            return None
        raise

    elements = [Element(site.specie) for site in pyxtal_structure.atom_sites]
    # electronegativity = [element.X for element in elements]
    wyckoffs = [site.wp for site in pyxtal_structure.atom_sites]
    for wp in wyckoffs:
        wp.get_site_symmetry()
    site_symmetries = [wp.site_symm for wp in wyckoffs]
    site_enumeration = [wychoffs_enumerated_by_ss[pyxtal_structure.group.number][wp.letter] for wp in wyckoffs]
    multiplicity = [wp.multiplicity for wp in wyckoffs]
    dof = [wp.get_dof() for wp in wyckoffs]

    # order = np.lexsort((site_enumeration, multiplicity, electronegativity))
    #sites_dict = {
    #    "site_symmetries": [site_symmetries[i] for i in order],
    #    "elements": [elements[i] for i in order],
    #    "multiplicity": [multiplicity[i] for i in order],
    #    "wyckoff_letters": [wyckoffs[i].letter for i in order],
    #    "sites_enumeration": [site_enumeration[i] for i in order],
    #    "dof": [dof[i] for i in order],
    #    "spacegroup_number": int(pyxtal_structure.group.number)
    #}
    sites_dict = {
        "site_symmetries": site_symmetries,
        "elements": elements,
        "multiplicity": multiplicity,
        "wyckoff_letters": [wp.letter for wp in wyckoffs],
        "sites_enumeration": site_enumeration,
        "dof": dof,
        "spacegroup_number": pyxtal_structure.group.number
    }
    if wychoffs_augmentation is not None:
        augmented_enumeration = [
            [wychoffs_enumerated_by_ss[pyxtal_structure.group.number][augmentator[letter]] for
              letter in sites_dict["wyckoff_letters"]]
                for augmentator in wychoffs_augmentation[pyxtal_structure.group.number]
        ]
        sites_dict["sites_enumeration_augmented"] = frozenset(map(tuple, augmented_enumeration))
    return sites_dict


def read_MP(MP_csv: Path|str, n_jobs: Optional[int] = None):
    """
    Reads a Materials Project CSV file and returns a DataFrame with structures.

    Args:
        MP_csv (Path|str): The path to the Materials Project CSV file.

    Returns:
        pd.DataFrame: The DataFrame with structures.
    """
    MP_df = pd.read_csv(MP_csv, index_col=0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Issues encountered while parsing CIF: \d+"
                " fractional coordinates rounded to ideal"
                " values to avoid issues with finite precision.",
            category=UserWarning,
            module="pymatgen.io.cif"
        )
        print("Suppressed CIF rounding warnings.")
        with Pool(processes=n_jobs) as pool:
            MP_df["structure"] = pool.map(read_cif, MP_df["cif"])
    MP_df.drop(columns=["cif"], inplace=True)
    return MP_df


def get_composition_from_symmetry_sites(record: pd.Series) -> dict:
    """
    Returns the composition of a record as a dictionary.

    Returns:
        dict: A dictionary with the elements as keys and the number of atoms as values.
    """
    result = Counter()
    for element, multiplicity in zip(record["elements"], record["multiplicity"]):
        result[element] += multiplicity
    return result


def get_composition(structure: Structure) -> dict[Element, float]:
    """
    Returns the composition of a structure as a dictionary.

    Args:
        structure (Structure): The pymatgen structure.

    Returns:
        dict: A dictionary with the elements as keys and the number of atoms as values.
    """
    str_dict = structure.composition.get_el_amt_dict()
    return {Element(k): v for k, v in str_dict.items()}


def compute_symmetry_sites(
    datasets_pd: dict[str, pd.DataFrame],
    wychoffs_enumerated_by_ss_file: Path = Path(__file__).parent.resolve() / "cache" / "wychoffs_enumerated_by_ss.pkl.gz",
    n_jobs: Optional[int] = None) -> tuple[dict[str, pd.DataFrame], int]:

    with open(wychoffs_enumerated_by_ss_file, "rb") as f:
        wychoffs_enumerated_by_ss = pickle.load(f)[0]
    
    structure_to_sites_with_args = partial(
                structure_to_sites,
                wychoffs_enumerated_by_ss=wychoffs_enumerated_by_ss,
                wychoffs_augmentation=get_augmentation_dict(),
                return_none_on_exception=False
    )
    result = {}
    for dataset_name, dataset in datasets_pd.items():
        with Pool(processes=n_jobs) as p:
            symmetry_list = p.map(structure_to_sites_with_args, dataset['structure'])
            symmetry_dataset = pd.DataFrame.from_records(symmetry_list).set_index(dataset.index)
        symmetry_dataset["composition"] = symmetry_dataset.apply(get_composition_from_symmetry_sites, axis=1)
        if "formation_energy_per_atom" in dataset.columns:
            symmetry_dataset['formation_energy_per_atom'] = dataset['formation_energy_per_atom']
        if "band_gap" in dataset.columns:
            symmetry_dataset['band_gap'] = dataset['band_gap']
        result[dataset_name] = symmetry_dataset
    return result


def read_all_MP_csv(
    mp_path: Path = Path(__file__).parent.resolve() / "cdvae"/"data"/"mp_20",
    wychoffs_enumerated_by_ss_file: Path = Path(__file__).parent.resolve() / "cache" / "wychoffs_enumerated_by_ss.pkl.gz",
    file_format: str = "csv",
    n_jobs: Optional[int] = None) -> tuple[dict[str, pd.DataFrame], int]:
    """
    Reads all Materials Project CSV files and returns a dictionary of DataFrames.

    Args:
        mp_path (Path, optional): The path to the Materials Project CSV files. Defaults to "cdvae/data/mp_20".
        wychoffs_enumerated_by_ss_file (Path, optional): The path to the Wyckoff positions enumerated by space group file. Defaults to "wychoffs_enumerated_by_ss.pkl.gz".
        file_format (str, optional): The file format. Defaults to "csv". Can be archived csv openable by pandas.

    Returns:
        dict: A dictionary with the following keys:
            - train: DataFrame with training data
            - test: DataFrame with testing data
            - val: DataFrame with validation data
    """
    datasets_pd = {}
    for dataset_name in ("train", "test", "val"):
        try:
            datasets_pd[dataset_name] = read_MP(mp_path / f"{dataset_name}.{file_format}")
        except FileNotFoundError:
            logger.warning("Dataset %s not found.", dataset_name)
    symmetry_datasets = compute_symmetry_sites(datasets_pd, wychoffs_enumerated_by_ss_file, n_jobs=n_jobs)
    return symmetry_datasets

