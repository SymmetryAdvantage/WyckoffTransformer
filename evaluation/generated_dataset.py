from enum import Enum
from typing import Optional, List, Tuple, Iterable
from collections.abc import Sequence
from itertools import repeat
import warnings
import gzip
from multiprocessing import Pool
import pickle
from copy import deepcopy
from functools import partial
from pathlib import Path
from ast import literal_eval
import json
from operator import attrgetter
from pymatgen.core import Composition, DummySpecies, Element, Structure
from pymatgen.io.cif import CifParser
from omegaconf import OmegaConf
import monty.json
import numpy as np
import torch
import pandas as pd
from .novelty import record_to_augmented_fingerprint
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from scripts.data import read_cif, compute_symmetry_sites, read_MP, pyxtal_notation_to_sites
from scripts.preprocess_wychoffs import get_augmentation_dict
from wyckoff_transformer.evaluation import wycryst_to_pyxtal_dict
from wyckoff_transformer.tokenization import get_wp_index

from .DiffCSP_to_sites import load_diffcsp_dataset, record_to_pyxtal
from .cdvae_metrics import (
    Crystal, structure_validity, timed_smact_validity_from_record, prop_model_eval)


StructureStorage = Enum("StructureStorage", [
    "DiffCSP_pt",
    "CrystalFormer",
    "NongWei",
    "Raymond",
    "CDVAE_csv_cif",
    "PymatgenJson",
    "EhullCSVCIF",
    "FlowMM",
    "DFT_CSV_CIF",
    "SymmCD_csv",
    "NongWei_CHGNet_csv",
    "old_atomated_csv",
    "atomated_csv",
    "invalid_cifs",
    "Pickle",
    "csv_cif_with_missing_values"
    ])

WyckoffStorage = Enum("WyckoffStorage", [
    "pyxtal_json",
    "WyCryst_csv",
    "WTCache",
    "letters_json",
    "site_symmetry_json"
    ])

DATASET_TO_CDVAE = {
    "mp_20_biternary": "mp20",
    "mpts_52": "mp20",
    "mp_20": "mp20",
    "mp_2022": "mp20",
    "mp_20_h": "mp20",
    "perov_5": "perovskite",
    "carbon_24": "carbon"}

DATA_KEYS = frozenset(("structures", "wyckoffs", "e_hull", "index_preserved"))

def load_all_from_config(
    datasets: Optional[List[Tuple[str]]] = None,
    dataset_name: str = "mp_20",
    config_path: Path = Path(__file__).parent.parent / "generated" / "datasets.yaml"):

    config = OmegaConf.load(config_path)
    if datasets is None:
        available_dataset_signatures = []
        def traverse_config(
            this_config: OmegaConf,
            transformations: List[str]) -> None:
            if DATA_KEYS.intersection(this_config.keys()):
                available_dataset_signatures.append(transformations)
            for new_transformation, new_config in this_config.items():
                if new_transformation in DATA_KEYS:
                    continue
                traverse_config(new_config, transformations + [new_transformation])
        traverse_config(config[dataset_name], [])
    else:
        available_dataset_signatures = datasets
    available_dataset_signatures = list(map(tuple, available_dataset_signatures))
    load_from_this_dataset = partial(GeneratedDataset.from_cache, dataset=dataset_name)
    with Pool(10) as pool:
        datasest_list = pool.map(load_from_this_dataset, available_dataset_signatures)
    datasets = dict(zip(available_dataset_signatures, datasest_list))

    if dataset_name == "mp_20":
        if ("WyckoffTransformer", "CrySPR", "CHGNet_fix_release") in datasets:
            datasets[('WyckoffTransformer', 'CrySPR', 'CHGNet_fix_release')].data["corrected_chgnet_ehull"] = \
            datasets[('WyckoffTransformer', 'CrySPR', 'CHGNet_fix')].data["corrected_chgnet_ehull"] - \
                datasets[('WyckoffTransformer', 'CrySPR', 'CHGNet_fix')].data["energy_per_atom"] + \
                datasets[('WyckoffTransformer', 'CrySPR', 'CHGNet_fix_release')].data["energy_per_atom"]
        if ("WyckoffTransformer", "CrySPR", "CHGNet_free") in datasets:
            datasets[('WyckoffTransformer', 'CrySPR', 'CHGNet_free')].data["corrected_chgnet_ehull"] = \
            datasets[('WyckoffTransformer', 'CrySPR', 'CHGNet_fix')].data["corrected_chgnet_ehull"] - \
                datasets[('WyckoffTransformer', 'CrySPR', 'CHGNet_fix')].data["energy_per_atom"] + \
                datasets[('WyckoffTransformer', 'CrySPR', 'CHGNet_free')].data["energy_per_atom"]
    return datasets


def load_pymatgen_json(path: Path):
    with gzip.open(path, "rb") as f:
        data = json.load(f)
    return monty.json.MontyDecoder().process_decoded(data)


def load_crystalformer(path: Path):
    dataset = pd.read_csv(path)
    decoder = monty.json.MontyDecoder()
    structures = dataset.cif.map(lambda s: decoder.process_decoded(literal_eval(s)))
    structures.name = "structure"
    return structures


def load_flowmm(path:Path|str):
    if isinstance(path, str):
        path = Path(path)
    index = []
    structures = []
    decoder = monty.json.MontyDecoder()
    for file_path in path.rglob("*.json"):
        with open(file_path, "rt", encoding="ascii") as f:
            this_structure = decoder.decode(f.read())
            if any(isinstance(e, DummySpecies) for e in this_structure.composition.elements):
                continue
            index.append(int(file_path.stem))
            structures.append(this_structure)
    return pd.Series(data=structures, index=index, name="structure")


def load_NongWei(path: Path):
    # There are two possible directory structures:
    # 0-499 & 500-999 / 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ...
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ...
    # We handle them all by using rglob
    try:
        data = pd.read_csv(path/"id_with_energy_per_atom.csv", index_col="id",
                usecols=["id", "energy_per_atom"])
    except FileNotFoundError:
        try:
            data = pd.read_csv(path/"WyCryst_mp20_result.csv", index_col=0)
        except FileNotFoundError:
            try:
                data = pd.read_csv(path/"WyckoffTransformer_mpts52_result.csv.gz", index_col="folder_ind")
            except FileNotFoundError:
                data = pd.DataFrame()
    data['structure'] = pd.Series(dtype=object, index=data.index)
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
        cif_file_paths = path.rglob("**/min_e_strc.cif")
        structures_index = []
        structures = []
        for this_cif_path in cif_file_paths:
            with open(this_cif_path, "rt", encoding="ascii") as f:
                structures_index.append(int(this_cif_path.parent.stem))
                structures.append(read_cif(f.read()))
    structures_pd = pd.Series(data=structures, index=structures_index)
    data["structure"] = structures_pd
    data.dropna(axis=0, subset=["structure"], inplace=True)
    return data


def read_cif_or_none(cif: str) -> Structure|None:
    try:
        return CifParser.from_str(cif).parse_structures(primitive=False)[0]
    except ValueError:
        return None

def load_NongWei_CHGNet_csv(path: Path) -> (pd.DataFrame, int):
    # TODO how to accout for NaNs during metric computation?
    data = pd.read_csv(path, index_col="id")
    initial_size = len(data)
    print(f"Read {len(data)} CIFs")
    data.dropna(inplace=True)
    # cif_generated is the input cif to CHGNet
    # cif is the output cif from CHGNet
    # ehull_refs_to_conventional_vc-relax is the ehull from CHGNet after relaxation
    # ...no-relax is the ehull from CHGNet before relaxation
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
        data["structure"] = data.cif.apply(read_cif_or_none)
    data["corrected_chgnet_ehull"] = data["ehull_refs_to_conventional_vc-relax"]
    data.dropna(inplace=True)
    print(f"Valid records: {len(data)}")
    return data, initial_size


def load_Raymond(path: Path):
    index_csv_path = list(path.glob("*.csv"))
    if len(index_csv_path) == 0:
        index_csv_path = list(path.glob("*.csv.gz"))
    if len(index_csv_path) == 0:
        raise ValueError("No index CSV found")
    if len(index_csv_path) > 1:
        raise ValueError("Multiple index CSVs found, please specify the correct one")
    index_csv_path = index_csv_path[0]
    data = pd.read_csv(index_csv_path, index_col=0)
    data['structure'] = pd.Series(dtype=object, index=data.index)
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
        cif_file_paths = path.rglob("**/*relax_postitions_and_cell*.cif")
        structures_index = []
        structures = []
        for this_cif_path in cif_file_paths:
            with open(this_cif_path, "rt", encoding="ascii") as f:
                structures_index.append(int(this_cif_path.parent.stem))
                structures.append(read_cif(f.read()))
    structures_pd = pd.Series(data=structures, index=structures_index)
    data["structure"] = structures_pd
    data.dropna(axis=0, subset=["structure"], inplace=True)
    return data


def load_WyCryst_csv(path: Path) -> pd.Series:
    wycryst_data_raw = pd.read_csv(path, index_col=0, converters=dict(zip(
                                    ("reconstructed_ratio1", "reconstructed_wyckoff",
                                     "str_wyckoff", "ter_sys"),
                                    repeat(literal_eval, 4)
                            )))
    wycryst_data = wycryst_data_raw.apply(wycryst_to_pyxtal_dict, axis=1).dropna()
    return pd.DataFrame.from_records(wycryst_data.tolist(), index=wycryst_data.index)


def load_ehull_csv_cif(path: Path):
    data = pd.read_csv(path, index_col=0)
    data["structure"] = data.relxed_structure.apply(read_cif)
    return data


def load_old_atomated_csv(path: Path) -> pd.DataFrame:
    structure_from_dict_str = lambda x: Structure.from_dict(literal_eval(x))
    return pd.read_csv(path, index_col="material_id", converters={"structure": structure_from_dict_str})

def load_atomated_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col="material_id", converters={
        "structure": partial(Structure.from_str, fmt="json")})


def load_invalid_cifs(path: Path):
    data = pd.read_csv(path, index_col=0)
    data["structure"] = data.cif.apply(read_cif_or_none)
    data.dropna(inplace=True)
    return data

def to_json(obj) -> str:
    """
    Convert an object to a JSON string, parsable by monty.json.MontyDecoder.
    """
    encoder = monty.json.MontyEncoder()
    if isinstance(obj, str):
        return obj
    if isinstance(obj, frozenset):
        obj = tuple(obj)
    return encoder.encode(obj)

def save_as_json_csv(path: Path, data: pd.DataFrame) -> None:
    """
    Save a DataFrame to csv with JSON-encoded columns.
    """
    data.map(to_json).to_csv(path, index_label="material_id")

class LetterDictToSitesConverter:
    def __init__(self, 
        wyckoffs_db_file = Path(__file__).parent.parent / "cache" / "wychoffs_enumerated_by_ss.pkl.gz",
        multiplicity_engineer_file = Path(__file__).parent.parent / "cache" / "engineers" / "multiplicity.pkl.gz"):
            with gzip.open(wyckoffs_db_file, "rb") as f:
                self.wychoffs_enumerated_by_ss, _, self.ss_from_letter = pickle.load(f)
            with gzip.open(multiplicity_engineer_file, "rb") as f:
                self.multiplicity_engineer = pickle.load(f)
            self.augmentation_dict = get_augmentation_dict()
            self.wp_index = get_wp_index()

    def __call__(self, letter_dict: dict) -> dict:
        """
        Args:
            letter_dict: A dictionary containing the following keys:
                "spacegroup_number": int
                "wyckoff_sites": List[Tuple[str, str]]
                    [(element, letter), ...] OR [(element, multiplicity and letter), ...] e.g.
                    ["Ba", "a"] OR ["Ba", "2a"]
        Returns:
            sites_dict: A dictionary containing the following keys:
                "site_symmetries": List[str]
                "elements": List[Element]
                "multiplicity": List[int]
                "wyckoff_letters": List[str]
                "sites_enumeration": List[int]
                "spacegroup_number": int
                "sites_enumeration_augmented": frozenset
        Raises:
            KeyError: If the Wyckoff letter is not found in the database
        """
        site_symmetries = []
        elements = []
        sites_enumeration = []
        multiplicity = []
        wyckoff_letters = []
        space_group = letter_dict["spacegroup_number"]
        available_sites = deepcopy(self.wp_index[space_group])
        for this_element, raw_letter in letter_dict["wyckoff_sites"]:
            true_element = Element(this_element)
            letter = raw_letter[-1]
            elements.append(true_element)
            wyckoff_letters.append(letter)
            ss = self.ss_from_letter[space_group][letter]
            # Will raise KeyError if the letter is not found in the SG
            # or a 0-DoF site is repeated
            this_multiplicity, this_dof = available_sites[ss][letter]
            if this_dof == 0:
                del available_sites[ss][letter]
            sites_enumeration.append(self.wychoffs_enumerated_by_ss[space_group][letter])
            site_symmetries.append(ss)
            multiplicity.append(self.multiplicity_engineer.db.loc[
                space_group, ss, sites_enumeration[-1]])
            if this_multiplicity != multiplicity[-1]:
                raise RuntimeError("Multiplicity does not match for get_wp_index and multiplicity engineer")
            if len(raw_letter) > 1:
                declared_multiplicity = int(raw_letter[:-1])
                if declared_multiplicity != multiplicity[-1]:
                    raise ValueError(f"Declared multiplicity {declared_multiplicity} does not match "
                                     f"engineered multiplicity {multiplicity[-1]}")

        sites_dict = {
            "site_symmetries": site_symmetries,
            "elements": elements,
            "multiplicity": multiplicity,
            "wyckoff_letters": wyckoff_letters,
            "sites_enumeration": sites_enumeration,
            "spacegroup_number": space_group
        }

        augmented_enumeration = [
            [self.wychoffs_enumerated_by_ss[space_group][augmentator[letter]] for
            letter in sites_dict["wyckoff_letters"]]
                for augmentator in self.augmentation_dict[space_group]
        ]
        sites_dict["sites_enumeration_augmented"] = frozenset(map(tuple, augmented_enumeration))
        return sites_dict


class SiteSymmetryToRecordConverter:
    def __init__(self, 
        wyckoffs_db_file = Path(__file__).parent.parent / "cache" / "wychoffs_enumerated_by_ss.pkl.gz",
        multiplicity_engineer_file = Path(__file__).parent.parent / "cache" / "engineers" / "multiplicity.pkl.gz"):
            with gzip.open(wyckoffs_db_file, "rb") as f:
                self.wychoffs_enumerated_by_ss, self.letter_from_ss_enum, _ = pickle.load(f)
            self.letter_to_record_converter = LetterDictToSitesConverter(
                wyckoffs_db_file, multiplicity_engineer_file)


    def __call__(self, ss_dict: dict) -> dict:
        """
        Intended to be used with record_to_pyxtal to get all the variables
        Args:
            ss_dict: A dictionary containing the following
                spacegroup_number: int
                wyckoff_sites: List[Tuple[str, str, int]]
                    [(element, site_symmetry, enumeration), ...]
                    ["Na", "m", 0]
        Returns:
            record: A dictionary containing the following keys:
                wyckoff_letters: List[str]
                multiplicity: List[int]
                elements: List[Element]
                sites_enumeration_augmented: frozenset
        """
        lettered_sites = []
        space_group = ss_dict["spacegroup_number"]
        for raw_element, site_symmetry, enumeration in ss_dict["wyckoff_sites"]:
            letter = self.letter_from_ss_enum[space_group][site_symmetry][enumeration]
            lettered_sites.append((raw_element, letter))

        return self.letter_to_record_converter(
            {"spacegroup_number": space_group, "wyckoff_sites": lettered_sites})


def read_json(path: Path):
    if path.suffix == ".json":
        with open(path, "rt", encoding="ascii") as f:
            return json.load(f)
    elif path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown file type {path.suffix}")


def load_site_symmetry_json(path: Path):
    data = read_json(path)
    converted = SiteSymmetryToRecordConverter()
    converted_data = []
    for record in data:
        try:
            converted_data.append(converted(record))
        except (KeyError, ValueError):
            # KeyError = Invalid Wyckoff representation
            # ValueError = Invalid Element
            continue
    print(f"Valid records: {len(converted_data)} / {len(data)} = {len(converted_data) / len(data):.2%}")
    return pd.DataFrame.from_records(converted_data, index=range(len(converted_data)))


def load_letters_json(path: Path):
    data = read_json(path)
    converter = LetterDictToSitesConverter()
    converted_data = []
    for letter_dict in data:
        try:
            converted_data.append(converter(letter_dict))
        except (KeyError, ValueError):
            # KeyError = Invalid Wyckoff representation
            # ValueError = Invalid Element
            continue
    print(f"Valid records: {len(converted_data)} / {len(data)} = {len(converted_data) / len(data):.2%}")
    return pd.DataFrame.from_records(converted_data, index=range(len(converted_data)))


def load_csv_cif_dft(path: Path, index_prefix: str):
    data = pd.read_csv(path, index_col=0)
    data.index.set_names("dft_index", inplace=True)
    needed_indices = [x.startswith(index_prefix+"-") for x in data.index]
    data = data[needed_indices]
    data['true_index'] = [int(x.split("-")[1]) for x in data.index]
    data.set_index("true_index", inplace=True)
    data['structure'] = data.cif.apply(read_cif)
    return data


def load_SymmCD_csv(path: Path):
    data = pd.read_csv(path, header=None, names=["cif"])
    print(f"Read {len(data)} CIFs")
    data["structure"] = data.cif.apply(read_cif)
    return data

def load_pickle(path: Path):
    if path.suffix == ".pkl":
        opener = open
    elif path.suffix == ".gz":
        opener = gzip.open
    else:
        raise ValueError(f"Unknown file type {path.suffix}")
    with opener(path, "rb") as f:
        return pickle.load(f)

class GeneratedDataset():
    @classmethod
    def from_cache(
        cls,
        transformations: Iterable[str],
        dataset: str = "mp_20",
        cache_path: Path = Path(__file__).parent.parent / "cache"):

        cache_location = cache_path.joinpath(
            dataset, "analysis_datasets", *transformations).with_suffix(".pkl.gz")
        with gzip.open(cache_location, "rb") as f:
            return pickle.load(f)


    @classmethod
    def from_transformations(
        cls,
        transformations: Sequence[str],
        dataset: str = "mp_20",
        config_path: Path = Path(__file__).parent.parent / "generated" / "datasets.yaml",
        root_path: Path = Path(__file__).parent.parent / "generated",
        cache_path: Path = Path(__file__).parent.parent / "cache",
        sort: bool = True):

        result = cls(dataset, cache_path.joinpath(dataset, "analysis_datasets", *transformations).with_suffix(".pkl.gz"))
        data_config = OmegaConf.load(config_path)[dataset]

        for level in transformations:
            data_config = data_config[level]
        if "structures" in data_config:
            result.load_structures(
                root_path / data_config.structures.path,
                data_config.structures.storage_type,
                data_config.structures.get("storage_key", None),
                sort=sort
                )
        if "wyckoffs" in data_config:
            result.load_wyckoffs(
                root_path / data_config.wyckoffs.path,
                data_config.wyckoffs.storage_type, data_config.wyckoffs.get("cache_key", None))
        if "e_hull" in data_config:
            result.load_corrected_chgnet_ehull(root_path / data_config.e_hull)

        return result

    def __init__(self,
            dataset_name: str,
            cache_location: Path):

        self.dataset_name = dataset_name
        self.cache_location = cache_location
        self.cdvae_dataset = DATASET_TO_CDVAE[dataset_name]
        self.data = pd.DataFrame()
        self.unfiltered_size = None

    def load_structures(
        self,
        path: Path|str,
        storage_type: StructureStorage|str,
        storage_key: Optional[str] = None,
        sort=True):
        # If structures are defined, wyckoffs must be obtained with pyxtal.from_seed(structures)
        # If structures are not defined, wyckoffs can be read externally
        if "wyckoffs" in self.data.columns:
            raise ValueError("Wyckoff positions are already defined")
        if isinstance(path, str):
            path = Path(path)
        if isinstance(storage_type, str):
            storage_type = StructureStorage[storage_type]
        if storage_type == StructureStorage.DiffCSP_pt:
            self.data["structure"] = load_diffcsp_dataset(path)
        elif storage_type == StructureStorage.CrystalFormer:
            self.data["structure"] = load_crystalformer(path)
        elif storage_type == StructureStorage.PymatgenJson:
            self.data["structure"] = load_pymatgen_json(path)
        elif storage_type == StructureStorage.FlowMM:
            self.data["structure"] = load_flowmm(path)
        elif storage_type == StructureStorage.EhullCSVCIF:
            self.data = load_ehull_csv_cif(path)
        elif storage_type == StructureStorage.NongWei:
            self.data = load_NongWei(path)
        elif storage_type == StructureStorage.Raymond:
            self.data = load_Raymond(path)
        elif storage_type == StructureStorage.CDVAE_csv_cif:
            self.data = read_MP(path)
        elif storage_type == StructureStorage.csv_cif_with_missing_values:
            self.data = read_MP(path, drop_na=True)
        elif storage_type == StructureStorage.DFT_CSV_CIF:
            self.data = load_csv_cif_dft(path, storage_key)
        elif storage_type == StructureStorage.SymmCD_csv:
            self.data = load_SymmCD_csv(path)
        elif storage_type == StructureStorage.NongWei_CHGNet_csv:
            self.data, self.unfiltered_size = load_NongWei_CHGNet_csv(path)
        elif storage_type == StructureStorage.old_atomated_csv:
            self.data = load_old_atomated_csv(path)
        elif storage_type == StructureStorage.invalid_cifs:
            self.data = load_invalid_cifs(path)
        elif storage_type == StructureStorage.atomated_csv:
            self.data = load_atomated_csv(path)
        elif storage_type == StructureStorage.Pickle:
            self.data = load_pickle(path)
        else:
            raise ValueError("Unknown storage type")
        if self.data.index.duplicated().any():
            raise ValueError("Duplicate indices in the dataset")
        if sort:
            self.data.sort_index(inplace=True)
        self.data["density"] = self.data["structure"].map(attrgetter("density"))
        self.structures_file = path
        if self.unfiltered_size is None:
            self.unfiltered_size = len(self.data)

    def load_corrected_chgnet_ehull(self, path: Path|str, index_json: Optional[Path] = None):
        e_hull_data = pd.read_csv(path, index_col=False)
        print(e_hull_data.head())
        if index_json is not None:
            with open(index_json, "rt", encoding="ascii") as f:
                index = json.load(f)
            e_hull_data.index = np.array(index)[e_hull_data.index]
            print(e_hull_data.head())
        else:
            if "folder_ind" in e_hull_data.columns:
                e_hull_data.set_index("folder_ind", inplace=True)
            else:
                e_hull_data.set_index(e_hull_data.columns[0], inplace=True)
        # Verify that the data matches
        e_hull_data = e_hull_data.reindex(self.data.index, copy=False)
        if "structure" in self.data.columns:
            for e_hull_formula, structure in zip(e_hull_data["formula"], self.data["structure"]):
                if isinstance(e_hull_formula, float):
                    continue
                if Composition(e_hull_formula).reduced_composition != structure.composition.reduced_composition:
                   raise ValueError(f"Formula mismatch between {e_hull_formula} and {structure.composition}")
        self.data["corrected_chgnet_ehull"] = e_hull_data["corrected_chgnet_ehull"]

    def load_wyckoffs(self, path: Path|str,
        storage_type: WyckoffStorage|str,
        cache_key: Optional[str] = None):

        if isinstance(storage_type, str):
            storage_type = WyckoffStorage[storage_type]
        if storage_type == WyckoffStorage.pyxtal_json:
            wcykoffs = pd.read_json(path)
        elif storage_type == WyckoffStorage.WTCache:
            with gzip.open(path, "rb") as f:
                wcykoffs = pickle.load(f)[cache_key]
        elif storage_type == WyckoffStorage.WyCryst_csv:
            wcykoffs = load_WyCryst_csv(path)
        elif storage_type == WyckoffStorage.letters_json:
            wcykoffs = load_letters_json(path)
        elif storage_type == WyckoffStorage.site_symmetry_json:
            wcykoffs = load_site_symmetry_json(path)
        else:
            raise ValueError(f"Unknown storage type {storage_type}")
        if len(self.data) == 0:
            self.data = wcykoffs
        else:
            self.data.loc[:, wcykoffs.columns] = wcykoffs
        if "site_symmetries" not in self.data.columns:
            # We have read the pyXtal format
            wyckoffs_db_file = Path(__file__).parent.parent / "cache" / "wychoffs_enumerated_by_ss.pkl.gz"
            with gzip.open(wyckoffs_db_file, "rb") as f:
                wychoffs_enumerated_by_ss, _, ss_from_letter = pickle.load(f)
            augmentation_dict = get_augmentation_dict()
            wyckoff_converted = partial(
                pyxtal_notation_to_sites,
                wychoffs_enumerated_by_ss=wychoffs_enumerated_by_ss,
                ss_from_letter=ss_from_letter,
                wychoffs_augmentation=augmentation_dict
            )
            proper_wyckoffs_series = self.data.apply(wyckoff_converted, axis=1)
            proper_wyckoffs = pd.DataFrame.from_records(
                proper_wyckoffs_series.tolist(), index=proper_wyckoffs_series.index)
            self.data.loc[:, proper_wyckoffs.columns] = proper_wyckoffs
        if "numIons" not in self.data.columns:
            self.convert_wyckoffs_to_pyxtal()
        self.wyckoffs_file = path


    def compute_wyckoffs(self, n_jobs: Optional[int] = None):
        wyckoffs = compute_symmetry_sites({"_": self.data}, n_jobs=n_jobs)["_"]
        self.data.loc[:, wyckoffs.columns] = wyckoffs
        # TODO fix data / lib
        self.data.dropna(axis=0, subset=wyckoffs.columns, inplace=True)

    def convert_wyckoffs_to_pyxtal(self):
        pyxtal_series = self.data.apply(record_to_pyxtal, axis=1)
        in_pyxtal_format = pd.DataFrame.from_records(pyxtal_series.tolist(), index=pyxtal_series.index)
        self.data.loc[:, in_pyxtal_format.columns] = in_pyxtal_format

    def compute_cdvae_crystals(self, n_jobs: Optional[int] = None):
        with Pool(n_jobs) as pool:
            self.data["cdvae_crystal"] = pool.map(Crystal.from_pymatgen, self.data["structure"])

    def compute_wyckoff_fingerprints(self):
        self.data["fingerprint"] = self.data.apply(record_to_augmented_fingerprint, axis=1)

    def dump_to_cache(self, path: Optional[Path] = None):
        if path is None:
            path = self.cache_location
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump(self, f)

    def compute_smact_validity(self):
        self.data["smact_validity"] = self.data.apply(timed_smact_validity_from_record, axis=1)

    def compute_naive_validity(self):
        if "smact_validity" not in self.data.columns:
            self.compute_smact_validity()
        self.data["structural_validity"] = self.data["structure"].apply(structure_validity)
        self.data["naive_validity"] = self.data["structural_validity"] & self.data["smact_validity"]

    def compute_cdvae_e(self,
        sample_size: Optional[int] = None,
        device: torch.device = torch.device("cpu")):

        sample_rows = self.data.index[:sample_size]
        sample = self.data.loc[sample_rows, "cdvae_crystal"].map(attrgetter("dict"))
        try:
            energies = prop_model_eval(self.cdvae_dataset, sample, device=device)
        except IndexError:
            # Rare atom types
            energies = np.nan
        self.data["cdvae_e"] = pd.Series()
        self.data.loc[sample_rows, "cdvae_e"] = energies
