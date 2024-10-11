from typing import Dict
from collections import defaultdict, Counter
from pandas import Series
import pandas as pd
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


class NoveltyFilter():
    def __init__(self, reference_dataset: pd.DataFrame):
        """
        Args:
            reference_dataset: The dataset to use as reference for novelty detection
            must have columns:
                'fingerprint' with a hashable fingerprint
                'structure' with a Structure for fine comparison
        """
        reference_dict = defaultdict(list)
        for _, record in reference_dataset.iterrows():
            reference_dict[record.fingerprint].append(record)
        self.reference_dict = dict(zip(reference_dict.keys(), map(tuple, reference_dict.values())))
        self.matcher = StructureMatcher()
    
    def is_novel(self, record: pd.Series) -> bool:
        """
        Args:
            record: The record to check for novelty
            must have columns:
                'fingerprint' with a hashable fingerprint
                'structure' with a Structure for fine comparison
        """
        if record.fingerprint in self.reference_dict:
            if 'structure' not in record:
                return False
            for reference_record in self.reference_dict[record.fingerprint]:
                if self.matcher.fit(record.structure, reference_record.structure):
                    return False
        return True

        
    def get_novel(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            dataset: The dataset to filter for novelty
            must have column 'fingerprint'.
            If 'structure' is present, it will be used for fine comparison,
            if not structures with same fingerprints will be same
        """
        return dataset.loc[dataset.apply(self.is_novel, axis=1)]