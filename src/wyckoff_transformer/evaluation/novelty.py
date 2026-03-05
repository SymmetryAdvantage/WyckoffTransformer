from typing import Dict
from math import gcd
from collections import defaultdict, Counter
from pandas import Series
import pandas as pd
from joblib import Parallel, delayed
from pymatgen.analysis.structure_matcher import StructureMatcher


def record_to_augmented_fingerprint(row: Dict|Series) -> tuple:
    """
    Computes a fingerprint taking into account equivalent Wyckoff position enumeration.
    Args:
        row contains the Wyckoff information:
        - spacegroup_number
        - elements
        - site_symmetries
        - sites_enumeration_augmented
    Returns:
        frozenset of all possible Wyckoff representations of the structure.
    """
    return (
        row["spacegroup_number"],
        frozenset(            
            map(lambda enumertaion:
                frozenset(Counter(
                    map(
                        tuple,
                        zip(row["elements"], row["site_symmetries"], enumertaion)
                    )
                ).items()), row["sites_enumeration_augmented"]
            )
        )
    )


def record_to_anonymous_fingerprint(row: Dict|Series) -> tuple:
    """
    Computes a fingerprint taking into account equivalent Wyckoff position enumeration.
    Args:
        row contains the Wyckoff information:
        - spacegroup_number
        - site_symmetries
        - sites_enumeration_augmented
    Returns:
        frozenset of all possible Wyckoff representations of the structure, without taking elements into account.
    """
    return (
        row["spacegroup_number"],
        frozenset(            
            map(lambda enumertaion:
                frozenset(Counter(
                    map(
                        tuple,
                        zip(row["site_symmetries"], enumertaion)
                    )
                ).items()), row["sites_enumeration_augmented"]
            )
        )
    )


def count_and_freeze(data):
    return frozenset(Counter(data).items())

def record_to_relaxed_AFLOW_fingerprint(row: Dict|Series) -> tuple:
    sites = frozenset(            
            map(lambda enumertaion:
                frozenset(Counter(
                    map(
                        tuple,
                        zip(row["site_symmetries"], enumertaion)
                    )
                ).items()), row["sites_enumeration_augmented"]
            )
        )
    element_counts = defaultdict(int)
    for element, multiplicity in zip(row["elements"], row["multiplicity"]):
        element_counts[element] += multiplicity
    stochio_gcd = gcd(*element_counts.values())
    simplified_stochio = [multiplicity // stochio_gcd for multiplicity in element_counts.values()]
    return (
        row["spacegroup_number"],
        count_and_freeze(simplified_stochio),
        sites
    )

def record_to_strict_AFLOW_fingerprint(row: Dict|Series) -> tuple:
    """
    Computes a fingerprint taking into account equivalent Wyckoff position enumeration.
    Fingerprint doesn't contain chemical elements, but keeps track which Wyckoff positions have
    the same elements.
    Args:
        row contains the Wyckoff information:
        - spacegroup_number
        - elements
        - site_symmetries
        - sites_enumeration_augmented
    Returns:
        frozenset of all possible Wyckoff representations of the structure.
    """
    all_variants = []
    for enumeration in row["sites_enumeration_augmented"]:
        per_element_wyckoffs = defaultdict(list)
        for element, site_symmetry, site_enumeration in zip(row["elements"], row["site_symmetries"], enumeration):
            per_element_wyckoffs[element].append((site_symmetry, site_enumeration))
        all_variants.append(count_and_freeze(map(count_and_freeze, per_element_wyckoffs.values())))
    return (
        row["spacegroup_number"],
        frozenset(all_variants)
    )


def filter_by_unique_structure(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filters a dataset for unique structures. First compares fingerprints,
    then uses StructureMatcher for fine comparison.
    """
    if 'structure' not in data:
        return data.drop_duplicates('fingerprint')
    present = defaultdict(list)
    unique_indices = []
    for index, row in data.iterrows():
        if row.fingerprint not in present:
            present[row.fingerprint].append(row.structure)
            unique_indices.append(index)
        else:
            for present_structure in present[row.fingerprint]:
                if StructureMatcher().fit(row.structure, present_structure):
                    break
            else:
                present[row.fingerprint].append(row.structure)
                unique_indices.append(index)
    return data.loc[unique_indices]


def get_reduced_composition(structure):
    """
    Returns a frozenset of the reduced composition of the structure.
    """
    return frozenset(map(tuple, structure.composition.reduced_composition.as_dict().items()))

def filter_by_unique_structure_reduced_comp_index(data: pd.DataFrame) -> pd.DataFrame:
    present = defaultdict(list)
    unique_indices = []
    for index, structure in data.structure.items():
        # Strutures with different reduced compositions can't match
        reduced_composition = get_reduced_composition(structure)
        if reduced_composition not in present:
            unique_indices.append(index)
        else:
            for present_structure in present[reduced_composition]:
                if StructureMatcher().fit(structure, present_structure):
                    break
            else:
                unique_indices.append(index)
        present[reduced_composition].append(structure)
    return data.loc[unique_indices]


class NoveltyFilter():
    """
    Uses fingerprints and StructureMatcher to filter for novel structures.
    """
    def __init__(self,
                 reference_dataset: pd.DataFrame,
                 reference_index_type: str = 'fingerprint',
                 n_jobs: int = 1):
        """
        Args:
            reference_dataset: The dataset to use as reference for novelty detection
            must have columns:
                'fingerprint' with a hashable fingerprint
                'structure' with a Structure for fine comparison
        """
        if n_jobs > 1:
            raise NotImplementedError("The naive n_jobs implementation is actually rather slow")
        self.reference_index_type = reference_index_type
        reference_dict = defaultdict(list)
        for _, record in reference_dataset.iterrows():
            reference_dict[self.get_reference_index(record)].append(record)
        self.reference_dict = dict(zip(reference_dict.keys(), map(tuple, reference_dict.values())))
        self.matcher = StructureMatcher()
        self.n_jobs = n_jobs


    def get_reference_index(self, record: pd.Series):
        if self.reference_index_type == 'fingerprint':
            return record.fingerprint
        elif self.reference_index_type == 'reduced_composition':
            if 'structure' in record:
                return get_reduced_composition(record.structure)
            else:
                raise NotImplementedError(
                    "Reduced composition index requires 'structure' column in the record.")
        else:
            raise ValueError(f"Unknown reference index type: {self.reference_index_type}")
        

    def is_novel(self, record: pd.Series) -> bool:
        """
        Args:
            record: The record to check for novelty
            columns:
                'fingerprint' with a hashable fingerprint
                'structure' with a Structure for fine comparison.
                   if not present, only fingerprints will be compared
        """
        reference_index = self.get_reference_index(record)
        if reference_index in self.reference_dict:
            if 'structure' not in record:
                return False
            for reference_record in self.reference_dict[reference_index]:
                if self.matcher.fit(record.structure, reference_record.structure):
                    return False
        return True


    def get_novel(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            dataset: The dataset to filter for novelty
            must have column 'fingerprint'.
            If 'structure' is present, it will be used for fine comparison,
            if not, structures with same fingerprints will be considered same
        """
        if self.n_jobs == 1:
            return dataset.loc[dataset.apply(self.is_novel, axis=1)]
        else:
            novel_records = Parallel(n_jobs=self.n_jobs)(
                delayed(self.is_novel)(record) for _, record in dataset.iterrows())
            return dataset.loc[novel_records]
