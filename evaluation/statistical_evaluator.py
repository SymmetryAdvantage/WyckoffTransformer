from typing import Optional, Tuple, Dict, List
from collections import Counter
import logging
from scipy.stats import kstest, wasserstein_distance, chisquare
from operator import attrgetter
import pandas as pd
import numpy as np
import random
from pymatgen.core import Element

from wyckoff_transformer.evaluation import DoFCounter, count_unique
from .cdvae_metrics import COV_Cutoffs, compute_cov, Crystal
from .novelty import NoveltyFilter

logger = logging.getLogger(__name__)

def compute_count_with_aggregation(
    data: pd.Series,
    aggregation_threshold: float,
    aggregation_label) -> pd.Series:

    representative = (data/data.sum() >= aggregation_threshold)
    print(f"Will use {representative.sum()} with frequency >= {aggregation_threshold}"
            f" out of {len(data)} present in the test dataset, rest are aggregated")
    aggregated_count = data[~representative].sum()
    filtered_data = data.loc[representative].copy()
    filtered_data.loc[aggregation_label] = aggregated_count
    return filtered_data

def normalise_counter(counter: Counter) -> Counter:
    total = sum(counter.values())
    return Counter({k: v/total for k, v in counter.items()})

class StatisticalEvaluator():
    def __init__(self,
                 test_dataset: pd.DataFrame,
                 cdvae_eval_model_name: str = "mp20"):
        """
        Args:
            test_dataset: DataFrame with the test dataset.
            sg_inclusion_threshold: Minimum frequency of space group in the test dataset to be
                included in the evaluation. Space groups with lower frequency are ignored.
        """
        self.test_dataset = test_dataset
        self.test_sg_frequency = self.test_dataset['spacegroup_number'].value_counts() / len(self.test_dataset)
        #if sg_aggregation_threshold is not None:
        #    self.test_sg_counts = compute_count_with_aggregation(self.test_sg_counts, sg_aggregation_threshold, 0)

        self.test_num_sites = self.test_dataset['site_symmetries'].map(len)
        self.test_num_elements = self.test_dataset['elements'].map(count_unique)
        self.test_num_atoms = self.test_dataset['structure'].map(len)
        self.dof_counter = DoFCounter()
        self.test_dof = self.test_dataset['dof'].apply(sum)

        self.test_element_frequency = pd.Series(
            self.test_dataset['composition'].apply(normalise_counter).sum())
        self.test_element_frequency /= self.test_element_frequency.sum()
        #if element_aggregation_threshold is not None:
        #    self.test_element_counts = compute_count_with_aggregation(
        #        self.test_element_counts, element_aggregation_threshold, 0)
        if "density" in self.test_dataset.columns:
            self.test_density = self.test_dataset['density']
        else:
            self.test_density = self.test_dataset.structure.map(attrgetter('density'))
        self.cdvae_eval_model_name = cdvae_eval_model_name

    def get_num_sites_ks(self, generated_structures: pd.DataFrame) -> float:
        generated_num_sites = generated_structures.sites.map(lambda x: sum(map(len, x)))
        return kstest(self.test_num_sites, generated_num_sites)

    def get_num_sites_emd(self,generated_structures: pd.DataFrame, return_counts:bool=False):
        """
        Computes the EMD between the numbers of sites in the generated structures and the test dataset.
        Args:
            generated_structures
        Returns:
            EMD value
        """
        generated_num_sites = generated_structures.sites.map(lambda x: sum(map(len, x)))

        if return_counts:
            return wasserstein_distance(self.test_num_sites, generated_num_sites), generated_num_sites
        else:
            return wasserstein_distance(self.test_num_sites, generated_num_sites)


    def get_num_atoms_emd(self, generated_structures: pd.DataFrame) -> float:
        """
        Computes the EMD between the numbers of atoms in the generated structures and the test dataset.
        Args:
            generated_structures
        Returns:
            EMD value
        """
        generated_num_atoms = generated_structures.structure.map(len)
        return wasserstein_distance(self.test_num_atoms, generated_num_atoms)


    def get_num_elements_ks(self, generated_structures: pd.DataFrame, return_counts:bool=False):
        """
        Computes the Kolmogorov-Smirnov statistic between the numbers of elements in the 
        generated structures and the test dataset. Note that Kolmorogov-Smirnov statistic doesn't
        a priory depend on the number examples, but p-value does.
        Args:
            generated_structures: Iterable of dictionaries with the generated structures in
                pyxtal.from_random arguments format.
        """
        
        generated_num_elements = generated_structures.species.map(lambda x: len(frozenset(x)))

        if return_counts:
            return kstest(self.test_num_elements, generated_num_elements), generated_num_elements
        else:
            return kstest(self.test_num_elements, generated_num_elements)

    def get_num_elements_emd(self, generated_structures: pd.DataFrame) -> float:
        """
        Computes the EMD between the numbers of elements in the generated structures and the test dataset.
        Args:
            generated_structures
        Returns:
            EMD value
        """
        generated_num_elements = generated_structures.species.map(lambda x: len(frozenset(x)))
        return wasserstein_distance(self.test_num_elements, generated_num_elements)

    def get_dof_ks(self, generated_structures: pd.DataFrame):
        """
        Computes the Kolmogorov-Smirnov statistic between the degrees of freedom in the 
        generated structures and the test dataset. Note that Kolmorogov-Smirnov statistic doesn't
        a priory depend on the number examples, but p-value does.
        Args:
            generated_structures: Iterable of dictionaries with the generated structures in
                pyxtal.from_random arguments format.
        """ 
        if "dof" in generated_structures.columns:
            dofs = generated_structures.dof.apply(sum)
        else:
            dofs = generated_structures.apply(self.dof_counter.get_total_dof, axis=1)
        return kstest(self.test_dof, dofs)

    def get_dof_emd(self, generated_structures: pd.DataFrame) -> float:
        """
        Computes the EMD between the degrees of freedom in the generated structures and the test dataset.
        Args:
            generated_structures
        Returns:
            EMD value
        """
        if "dof" in generated_structures.columns:
            dofs = generated_structures.dof.apply(sum)
        else:
            dofs = generated_structures.apply(self.dof_counter.get_total_dof, axis=1)
        return wasserstein_distance(self.test_dof, dofs)
    
    def get_density_emd(self, generated_structures: pd.DataFrame) -> float:
        if "density" in generated_structures.columns:
            generated_density = generated_structures.density
        else:
            generated_density = generated_structures.structure.map(attrgetter('density'))
        return wasserstein_distance(self.test_density, generated_density)
    
    def get_sg_chi2(self,
        generated_structures: pd.DataFrame,
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
            sample = generated_structures.iloc[:sample_size]
        elif sample_size == "test":
            sample = generated_structures.iloc[:len(self.test_dataset)]
        else:
            raise ValueError(f"Unknown sample_size {sample_size}")
        generated_sg_counts = sample.spacegroup_number.value_counts()
        #sg_to_be_aggregated = generated_sg_counts.index[~generated_sg_counts.index.isin(self.test_sg_counts.index)]
        
        filtered_generated_sg_counts = generated_sg_counts.reindex_like(self.test_sg_frequency).fillna(0)
        #filtered_generated_sg_counts.loc[0] = generated_sg_counts.loc[sg_to_be_aggregated].sum()

        #contingency_table = pd.concat([self.test_sg_counts, filtered_generated_sg_counts], axis=1)
        #chi2 = chi2_contingency(contingency_table).statistic
        filtered_generated_sg_counts /= filtered_generated_sg_counts.sum()
        chi2 = chisquare(filtered_generated_sg_counts, f_exp=self.test_sg_frequency).statistic
        
        if return_counts:
            return chi2, filtered_generated_sg_counts
        else:
            return chi2

    def get_cdvae_e_emd(self, generated_structures: pd.DataFrame) -> float:
        """
        Computes the EMD between the CDVAE energies in the generated structures and the test dataset.
        Only structurally valid structures are considered - the ones with interatomic distances >0.5A.
        Without this restriction the results are dominated by outliers.
        Args:
            generated_structures
        Returns:
            EMD value
        """
        return wasserstein_distance(
            self.test_dataset.loc[self.test_dataset.structural_validity, 'cdvae_e'],
            generated_structures.loc[generated_structures.structural_validity, 'cdvae_e']
        )

    def get_coverage(
        self,
        generated_crystals: List[Crystal],
        target_sample_size: Optional[int|str] = None,
        random_seed: Optional[int] = 42) -> Dict[str, float]:
        """
        Computes the COV metrics from CDVAE paper. They depend on the number of samples
        in the test and generated datasets.

        Args:
            generated_structures: an iterable of CDVAE Crystal objects.
        Returns:
            Dictionary with the COV metrics.
        """
        valid_crystals = [crystal for crystal in generated_crystals if crystal.comp_fp is not None]
        if len(valid_crystals) < len(generated_crystals):
            logger.warning("Ignoring %i generated samples without composition fingerprints.",
                len(generated_crystals) - len(valid_crystals))
        if target_sample_size is not None:
            if target_sample_size < len(valid_crystals):
                logger.info("Using only %i samples out of %i generated samples.",
                    target_sample_size, len(valid_crystals))
                sampler = random.Random(random_seed)
                valid_crystals = sampler.sample(valid_crystals, target_sample_size)
            elif target_sample_size > len(valid_crystals):
                logger.warning("Requested sample size %i is larger than the number of generated samples %i.",
                               target_sample_size, len(valid_crystals))
                sampler = np.random.default_rng(random_seed)
                valid_crystals = sampler.choice(valid_crystals, target_sample_size, replace=True)

        cutoff_dict = COV_Cutoffs[self.cdvae_eval_model_name]
        cov_metrics_dict = compute_cov(
            self.test_dataset.cdvae_crystal, valid_crystals,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    def get_elements_chi2(self,
        generated_structures: pd.DataFrame,
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
        
        if "composition" in sample.columns:
            generated_element_counts = pd.Series(
                sample['composition'].apply(normalise_counter).sum())
        else:
            generated_element_counts = Counter()
            for _, record in sample.iterrows():
                total_elements = sum(record['numIons'])
                for element, multiplicity in zip(record['species'], record['numIons']):
                    generated_element_counts[Element(element)] += multiplicity / total_elements
            generated_element_counts = pd.Series(generated_element_counts)
        #elements_to_be_aggregated = generated_counts.index[~generated_counts.index.isin(self.test_element_counts.index)]
        filtered_generated_counts = generated_element_counts.reindex_like(self.test_element_frequency).fillna(0)
        filtered_generated_counts /= filtered_generated_counts.sum()
        #filtered_generated_counts[0] = generated_counts.loc[elements_to_be_aggregated].sum()
        
        chi2 = chisquare(filtered_generated_counts, f_exp=self.test_element_frequency).statistic
        if return_counts:
            return chi2, filtered_generated_counts
        else:
            return chi2

    def compute_cdvae_metrics(self, dataset: pd.DataFrame,
                              novelty_filter: Optional[NoveltyFilter] = None,
                              sample_size_for_precision: int = 10000,
                              random_seed: int = 42,
                              compute_novelty = False) -> Dict[str, float]:
        """
        Computes the "CDVAE metrics" for a given dataset, from
        Xie et al. "Crystal Diffusion Variational AutoEncoder" [ICLR 2022].
        Optionally applies a novelty filter to the dataset to prevent rewarding the models 
        for generating non-novel structures, see
        "Wyckoff transformer: Generation of symmetric crystals" [ICML 2025] for details.
        Args:
            dataset (pd.DataFrame): DataFrame containing the dataset to evaluate.
            sample_size_for_precision (int): Number of samples to use for COV-P calculation.
                It's the only metric which expetected value depends on the sample size, so we must ensure that
                the same sample size is used for all datasets.
            novelty_filter (Optional[NoveltyFilter]): If provided, applies the novelty filter to the
                dataset before computing the metrics.
        """
        result = {}
        logger.info("Computing CDVAE metrics...")
        logger.info("Dataset size: %d", len(dataset))
        if novelty_filter is not None:
            intial_size = len(dataset)
            dataset = novelty_filter.get_novel(dataset)
            logger.info("Dataset size after novelty filtering: %d", len(dataset))
            if compute_novelty:                
                result["Novelty"] = 100 * len(dataset) / intial_size
        elif compute_novelty:
            raise ValueError("Novelty filter is not provided, but compute_novelty is True.")
        # The original CDVAE paper uses 10k structures for validity and coverage metrics
        # Validity is a binary per-sample metric, so its expected value does not depend on the sample size
        result["Compositional"] = 100*dataset.smact_validity.mean()
        result["Structural"] = 100*dataset.structural_validity.mean()
        cov_metrics = self.get_coverage(dataset.cdvae_crystal)
        result["Recall"] = 100*cov_metrics["cov_recall"]
        if sample_size_for_precision == len(dataset):
            result["Precision"] = 100*cov_metrics["cov_precision"]
        else:
            precision_samples = dataset.sample(
                sample_size_for_precision, random_state=random_seed, replace=False)
            cov_metrics_sampled = self.get_coverage(
                precision_samples.cdvae_crystal)
            result["Precision"] = 100*cov_metrics_sampled["cov_precision"]
        valid = dataset.loc[dataset.naive_validity]
        logger.info("Valid samples for EMD: %d", len(valid))
        # The original CDVAE paper uses 1k structures for property metrics
        # But EMD does not depend on the sample size, so we can use the whole valid set
        result["EMD_rho"] = self.get_density_emd(valid)
        result["EMD_E"] = self.get_cdvae_e_emd(valid)
        result["EMD_Elements"] = self.get_num_elements_emd(valid)
        return result