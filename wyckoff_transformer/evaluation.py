from typing import Dict, Iterable, List, Optional, Tuple
from itertools import chain, product
from collections import defaultdict, Counter
from functools import partial
from math import gcd
import numpy as np
from scipy.stats import kstest, chi2_contingency
import pandas as pd
from pyxtal.symmetry import Group
from pymatgen.core import Element
from wrapt_timeout_decorator import timeout
import smact
import smact.screening
import wandb
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

def count_unique(x: Iterable) -> int:
    return len(frozenset(x))


class DoFCounter():
    def __init__(self):
        dof_from_letter = defaultdict(dict)
        for group_number in range(1, 231):
            group = Group(group_number)
            for wp in group.Wyckoff_positions:
                wp.get_site_symmetry()
                dof_from_letter[group_number][wp.letter] = wp.get_dof()
        self.dof_from_letter = dof_from_letter

    def get_total_dof(self, record: Dict) -> int:
        """
        Computes the total number of degrees of freedom in a structure.
        Args:
            record: dictionary with a structure in pyxtal.from_random arguments format.
        """
        return sum(map(lambda site: self.dof_from_letter[record['group']][site.lstrip('1234567890')],
            chain(*record['sites'])))


# Function from CDVAE
# https://github.com/txie-93/cdvae/blob/f857f598d6f6cca5dc1ea0582d228f12dcc2c2ea/scripts/eval_utils.py
def smact_validity(comp, count,
                   use_pauling_test=True,
                   include_alloys=True):
    elem_symbols = tuple(comp)
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    for ox_states in product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = smact.screening.pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append(
                        tuple([elem_symbols, ox_states, ratio]))
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    if len(compositions) > 0:
        return True
    else:
        return False


def smact_validity_optimised(
                   elem_symbols, count,
                   use_pauling_test=True,
                   include_alloys=True,
                   apply_gcd=False):
    element_set = frozenset(elem_symbols)
    if len(element_set) == 1:
        return True
    if include_alloys and element_set.issubset(smact.metals):
        return True
    
    space = smact.element_dictionary(elem_symbols)
    electronegs = [e.pauling_eneg for e in space.values()]
    ox_combos = [e.oxidation_states for e in space.values()]

    if apply_gcd:
        gcd_count = gcd(*count)
        count = tuple(c // gcd_count for c in count)
    threshold = max(count)
    stoichs = [(c,) for c in count]
    for ox_states in product(*ox_combos):
        cn_r = smact.neutral_ratios_iter(ox_states, stoichs=stoichs, threshold=threshold)
        if any(True for _ in cn_r):
            if not use_pauling_test:
                return True
            try:
                if smact.screening.pauling_test(ox_states, electronegs, symbols=elem_symbols):
                    return True
            except TypeError:
                # if no electronegativity data, assume it is okay
                return True
    return False


def smact_validity_from_record(record: Dict, apply_gcd: bool=True) -> bool:
    """
    Computes the SMACT validity of a record in pyxtal.from_random arguments format.
    """
    return smact_validity_optimised(record['species'], record['numIons'], apply_gcd=apply_gcd)


@timeout(15)
def timed_smact_validity_from_record(record: Dict, apply_gcd: bool=True) -> bool:
    """
    Computes the SMACT validity of a record in pyxtal.from_random arguments format. If the computation
    takes longer than 15 seconds, returns False.
    """
    try:
        return smact_validity_optimised(record['species'], record['numIons'], apply_gcd=apply_gcd)
    except TimeoutError:
        logger.debug("SMAC-T validitiy timed out, returning False")
        return False


def record_to_augmented_fingerprints(row):
    """
    Computes a fingerprint for each possible Wyckoff position enumeration.
    """
    spacegroup_number = row["spacegroup_number"]
    def get_augmentation_fingerprint(augmentation):
        site_symmetries = frozenset(map(tuple, zip(row["elements"], row["site_symmetries"], augmentation)))
        return (spacegroup_number, site_symmetries)
    return frozenset(map(get_augmentation_fingerprint, row["sites_enumeration_augmented"]))


def generated_to_fingerprint(wy_dict, letter_to_ss, letter_to_enum):
    elements = []
    site_symmetries = []
    site_enumerations = []
    for specie, sites in zip(wy_dict["species"], wy_dict["sites"]):
        element = Element(specie)
        for site in sites:
            elements.append(element)
            site_symmetries.append(letter_to_ss[wy_dict["group"]][site[-1:]])
            site_enumerations.append(letter_to_enum[wy_dict["group"]][site[-1:]])
    return (
        wy_dict["group"],
        frozenset(map(tuple, zip(elements, site_symmetries, site_enumerations))),
    )


def wycryst_to_pyxtal_dict(record):
    species = []
    all_sites = []
    numIons = []
    for reported_count, (element, sites) in zip(record["reconstructed_ratio1"], record["reconstructed_wyckoff"].items()):
        counted_ions = 0
        this_sites = []
        for site in sites:
            this_sites.append(site)
            counted_ions += int(site[:-1])
        if counted_ions != reported_count:
            logging.warning("Reported count %f does not match sum of site counts %i for"
                " record %i element %s. Elements: %s",
                reported_count, counted_ions, record.name, element, record['reconstructed_wyckoff'].keys())
            return None
        species.append(element)
        all_sites.append(this_sites)
        numIons.append(counted_ions)
    return {
        "species": species,
        "sites": all_sites,
        "numIons": numIons,
        "group": record.reconstructed_sg
    }


class StatisticalEvaluator():
    def __init__(self,
                 test_dataset: pd.DataFrame,
                 train_dataset: Optional[pd.DataFrame] = None):
        self.test_dataset = test_dataset
        self.dof_counter = None
        if train_dataset is not None:
            self.train_fingerprints = frozenset(chain.from_iterable(train_dataset.apply(record_to_augmented_fingerprints, axis=1)))
            with open(Path(__file__).parent.parent.resolve() / "cache" / "wychoffs_enumerated_by_ss.pkl.gz", "rb") as f:
                self.letter_to_enum, _ , self.letter_to_ss = pickle.load(f)[:3]
            self.generated_to_fingerprint = partial(
                generated_to_fingerprint, letter_to_ss=self.letter_to_ss, letter_to_enum=self.letter_to_enum)
        self.test_novelty = None
        self.test_sg_counts = self.test_dataset['spacegroup_number'].value_counts()

    def get_test_novelty(self):
        # Just a single augmentation variant per structure
        if self.test_novelty is None:
            test_fp = self.test_dataset.apply(record_to_augmented_fingerprints, axis=1).map(lambda x: next(iter(x)))
            self.test_novelty = sum((fp not in self.train_fingerprints for fp in test_fp)) / len(test_fp)
        return self.test_novelty

    def get_num_sites_ks(self, generated_structures: Iterable[Dict], return_counts:bool=False) -> float:
        """
        Computes the Kolmogorov-Smirnov statistic between the numbers of sites in the 
        generated structures and the test dataset. Note that Kolmorogov-Smirnov statistic doesn't
        a priory depend on the number examples, but p-value does.
        Args:
            generated_structures: Iterable of dictionaries with the generated structures in
                pyxtal.from_random arguments format.
        """
        test_num_sites = self.test_dataset['site_symmetries'].map(len)
        generated_num_sites = [sum(map(len, wygene['sites'])) for wygene in generated_structures]
        if return_counts:
            return kstest(test_num_sites, generated_num_sites), generated_num_sites
        else:
            return kstest(test_num_sites, generated_num_sites)


    def get_num_elements_ks(self, generated_structures: Iterable[Dict], return_counts:bool=False) -> float:
        """
        Computes the Kolmogorov-Smirnov statistic between the numbers of elements in the 
        generated structures and the test dataset. Note that Kolmorogov-Smirnov statistic doesn't
        a priory depend on the number examples, but p-value does.
        Args:
            generated_structures: Iterable of dictionaries with the generated structures in
                pyxtal.from_random arguments format.
        """
        test_num_elements = self.test_dataset['elements'].map(count_unique)
        generated_num_elements = [len(frozenset(record["species"])) for record in generated_structures]
        if return_counts:
            return kstest(test_num_elements, generated_num_elements), generated_num_elements
        else:
            return kstest(test_num_elements, generated_num_elements)

    def get_dof_ks(self, generated_structures: Iterable[Dict]) -> float:
        """
        Computes the Kolmogorov-Smirnov statistic between the degrees of freedom in the 
        generated structures and the test dataset. Note that Kolmorogov-Smirnov statistic doesn't
        a priory depend on the number examples, but p-value does.
        Args:
            generated_structures: Iterable of dictionaries with the generated structures in
                pyxtal.from_random arguments format.
        """
        if self.dof_counter is None:
            self.dof_counter = DoFCounter()
        test_dof = self.test_dataset['dof'].apply(sum)
        generated_dof = [self.dof_counter.get_total_dof(record) for record in generated_structures]
        return kstest(test_dof, generated_dof)
    
    def get_sg_chi2(self,
        generated_structures: Iterable[Dict],
        sample_size: Optional[int|str] = None,
        return_counts: bool = False) -> float|Tuple[float, np.ndarray]:
        """
        Computes the chi-squared statistic between the space group numbers in the
        generated structures and the test dataset.
        Args:
            generated_structures: Iterable of dictionaries with the generated structures in
                pyxtal.from_random arguments format.
            sample_size: If int, the number of samples to use from the generated dataset.
                if "test", use the same number as in the test dataset. NOTE: it takes first
                N samples without shuffling in the interest of reproducibility.
        """
        if sample_size is None:
            sample = generated_structures
        elif isinstance(sample_size, int):
            sample = generated_structures[:sample_size]
        elif sample_size == "test":
            sample = generated_structures[:len(self.test_dataset)]
        else:
            raise ValueError(f"Unknown sample_size {sample_size}")
        generated_sg = [record['group'] for record in sample]
        generated_sg_counts = pd.Series(generated_sg).value_counts()
        if not generated_sg_counts.index.isin(self.test_sg_counts.index).all():
            logger.warning("Generated dataset has extra space groups compared to test")
        generated_sg_counts = generated_sg_counts.reindex_like(self.test_sg_counts).fillna(0)
        chi2 = chi2_contingency(np.vstack([self.test_sg_counts, generated_sg_counts]))
        if return_counts:
            return chi2, generated_sg_counts
        else:
            return chi2


    def get_elements_chi2(self,
        generated_structures: Iterable[Dict],
        sample_size: Optional[int|str] = None,
        return_counts: bool = False) -> float|Tuple[float, np.ndarray]:
        """
        Computes the chi-squared statistic between the element counts in the
        generated structures and the test dataset.
        Args:
            generated_structures: Iterable of dictionaries with the generated structures in
                pyxtal.from_random arguments format.
            sample_size: If int, the number of samples to use from the generated dataset.
                if "test", use the same number as in the test dataset. NOTE: it takes first
                N samples without shuffling in the interest of reproducibility.
        """
        if sample_size is None:
            sample = generated_structures
        elif isinstance(sample_size, int):
            sample = generated_structures[:sample_size]
        elif sample_size == "test":
            sample = generated_structures[:len(self.test_dataset)]
        else:
            raise ValueError(f"Unknown sample_size {sample_size}")
        generated_element_counts = Counter()
        for record in sample:
            for element, multiplicity in zip(record['species'], record['numIons']):
                generated_element_counts[Element(element)] += multiplicity
        test_element_counts = self.test_dataset['composition'].apply(Counter).sum()
        test_element_counts = pd.Series(test_element_counts)
        generated_counts = pd.Series(generated_element_counts)
        if not generated_counts.index.isin(test_element_counts.index).all():
            logger.warning("Generated dataset has extra elements compared to test")
        generated_counts = generated_counts.reindex_like(test_element_counts).fillna(0)
        
        chi2 = chi2_contingency(np.vstack([test_element_counts, generated_counts]))
        if return_counts:
            return chi2, generated_counts, test_element_counts
        else:
            return chi2
        

    def count_novel(self, generated_structures: Iterable[Dict]) -> float:
        generated_fp = list(map(self.generated_to_fingerprint, generated_structures))
        return sum((fp not in self.train_fingerprints for fp in generated_fp)) / len(generated_fp)

    
    def get_novel(self, generated_structures: Iterable[Dict], return_index=False) -> List[Dict]:
        generated_fp = list(map(self.generated_to_fingerprint, generated_structures))
        if return_index:
            res = [(record, i) for i, (record, fp) in enumerate(zip(generated_structures, generated_fp)) if fp not in self.train_fingerprints]
            return list(zip(*res))
        else:
            return [record for record, fp in zip(generated_structures, generated_fp) if fp not in self.train_fingerprints]


    def get_novel_dataframe(self, structures: pd.DataFrame) -> pd.DataFrame:
        generated_fp = structures.apply(record_to_augmented_fingerprints, axis=1)
        return structures[~generated_fp.isin(self.train_fingerprints)]


def smac_validity_from_counter(counter: Dict[Element, float], apply_gcd=True) -> bool:
    res = smact_validity_optimised(tuple((elem.symbol for elem in counter.keys())), tuple(map(int, counter.values())), apply_gcd=apply_gcd)
    if res is None:
        return False
    return res


def ks_to_dict(ks_statistic) -> Dict[str, float]:
    return {"statistic": ks_statistic.statistic, "pvalue": ks_statistic.pvalue}


def evaluate_and_log(
    generated_wp: List[Dict],
    generation_name: str,
    attempted_n_structures: int,
    evaluator: StatisticalEvaluator):

    wp_formal_validity = len(generated_wp) / attempted_n_structures
    wandb.run.summary["formal_validity"][generation_name] = wp_formal_validity
    print(f"Wyckchoffs' formal validity: {wp_formal_validity}")

    print(f"Generated dataset size: {len(generated_wp)}")
    if len(generated_wp) == 0:
        print("No structures generated, skipping evaluation")
        return
    
    ks_num_sites, generated_num_sites = evaluator.get_num_sites_ks(generated_wp, return_counts=True)
    num_sites_bins = np.arange(0, 21)
    num_sites_hist = np.histogram(generated_num_sites, bins=num_sites_bins)
    wandb.run.summary["num_sites"][generation_name] = {"hist": wandb.Histogram(np_histogram=num_sites_hist)}
    
    ks_num_elements, generated_num_elements = evaluator.get_num_elements_ks(generated_wp, return_counts=True)
    num_elements_bins = np.arange(1, 10)
    num_elements_hist = np.histogram(generated_num_elements, bins=num_elements_bins)
    wandb.run.summary.update({"num_elements": {generation_name: {"hist": wandb.Histogram(np_histogram=num_elements_hist)}}})
    
    ks_dof = evaluator.get_dof_ks(generated_wp)
    wandb.run.summary["wp"][generation_name] = {"ks_num_sites_vs_test": ks_to_dict(ks_num_sites)}
    wandb.run.summary["wp"][generation_name]["ks_num_elements_vs_test"] = ks_to_dict(ks_num_elements)
    wandb.run.summary["wp"][generation_name]["ks_dof_vs_test"] = ks_to_dict(ks_dof)
    print("Kolmogorov-Smirnov statistics:")
    print(f"Number of sites: {ks_num_sites}")
    print(f"Number of elements: {ks_num_elements}")
    print(f"Degrees of freedom: {ks_dof}")
    wandb.run.summary["wp"][generation_name]["generated_actual_size"] = len(generated_wp)
    
    smac_validity_count = sum(map(timed_smact_validity_from_record, generated_wp))
    smac_validity_fraction = smac_validity_count / len(generated_wp)
    print(f"SMAC-T validity: fraction {smac_validity_fraction}; count {smac_validity_count}")
    wandb.run.summary["smact_validity"][generation_name] = smac_validity_fraction


def main():
    import gzip, json
    for run in ("je9sllnx", "aik4ie80", "uoz22ycs", "d9b2y4ke"):
        with gzip.open(f"runs/{run}/gnerated_wp_no_calibration.json.gz", "rb") as f:
            generated_structures = json.load(f)
        # print("Testing naive and optimised SMACT validity functions")
        optimised_validity = list(map(partial(smact_validity_from_record, apply_gcd=False), generated_structures))
        naive_validity = list(map(smact_validity, (record['species'] for record in generated_structures), (record['numIons'] for record in generated_structures)))
        assert naive_validity == optimised_validity
        print(f"Run {run} non-GCD validity is {sum(optimised_validity) / len(generated_structures)}")
        gcd_validity_generated = list(map(partial(smact_validity_from_record, apply_gcd=True), generated_structures))
        print(f"Run {run} GCD validity is {sum(gcd_validity_generated) / len(generated_structures)}")
    with gzip.open("cache/mp_20_biternary/data.pkl.gz", "rb") as f:
        dataset_pd = pd.read_pickle(f)
    test_dataset = dataset_pd['test']
    # Compute SMACT for the test dataset
    test_smact_valid = test_dataset.composition.map(partial(smac_validity_from_counter, apply_gcd=False))
    test_smact_valid_naive = test_dataset.composition.map(lambda counter: smact_validity(tuple((elem.symbol for elem in counter.keys())), tuple(map(int, counter.values()))))
    assert test_smact_valid_naive.equals(test_smact_valid)
    print("Asserts passed")
    print(f"Test dataset non-GCD validity is {test_smact_valid.sum() / len(test_smact_valid)}")
    gcd_validity_test = test_dataset.composition.map(partial(smac_validity_from_counter, apply_gcd=True))
    print(f"Test dataset GCD validity is {gcd_validity_test.sum() / len(gcd_validity_test)}")


if __name__ == "__main__":
    main()    
