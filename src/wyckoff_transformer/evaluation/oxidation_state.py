"""Oxidation-state guessing used by the charge-neutrality validity check.

Vendored from LeMat-GenBench's ``utils/oxidation_state.py`` (Apache-2.0),
https://github.com/LeMaterial/lemat-genbench, so that structure validity can be
computed without importing that package.  The logic is reproduced as-is --
including its quirks -- because our numbers have to match the benchmark's, not
improve on them; the only edits are the logger and the data-file lookup, which
now resolves inside this package.

``compositional_oxi_state_guesses`` is itself adapted from pymatgen's
``Composition._get_oxi_state_guesses``, re-ranked by oxidation-state priors
derived from ICSD entries in LeMat-Bulk.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import lru_cache
from itertools import combinations_with_replacement, product
from pathlib import Path

import numpy as np
from pymatgen.core.periodic_table import Element, Species
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import logging

logger = logging.getLogger(__name__)

#: Vendored alongside this module; see the LeMat-Bulk paper for provenance.
DATA_DIR = Path(__file__).resolve().parent / "data"
OXI_PROBS_FILE = DATA_DIR / "lemat_icsd_oxi_dict_probs.json"
OXI_STATE_MAPPING_FILE = DATA_DIR / "lemat_icsd_oxi_state_mapping.json"


@lru_cache(maxsize=None)
def _load_json(path: Path) -> dict:
    """Read and cache one of the vendored JSON tables."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def electronegativity_correlation(
        elements: list[str],
        oxidation_states: list[int | float]
        ) -> float:
    """
    Calculate correlation between oxidation states and electronegativity.
    
    Args:
        elements: List of element symbols
        oxidation_states: List of oxidation state values (averaged per element)
        TODO this should probably be scaled to follow the number of elements with that 
        oxidation state.
        
    Returns:
        Pearson correlation coefficient between oxidation states and electronegativity.
        Returns NaN if correlation cannot be calculated.
    """

    en_vals = []
    for el in elements: 
        try:
            en_vals.append(Element(el).X)
        except (KeyError, AttributeError):
            # Use Pauling scale default if not available, or raise error
            logger.warning(f"No electronegativity data for element {el}")
            return np.nan  # Can't calculate correlation without complete data

    if len(en_vals) != len(oxidation_states):
        logger.error("Mismatch in array lengths for correlation calculation")
        return np.nan
    else:
        corr = np.corrcoef(oxidation_states, en_vals)[0,1]
        return corr


def compositional_oxi_state_guesses(
    comp,
    all_oxi_states: bool,
    max_sites: int | None,
    oxi_states_override: dict[str, list] | None,
    target_charge: float,
) -> tuple[tuple, tuple, tuple]:
    """Utility operation for guessing oxidation states. 
    Adapted from the _get_oxi_state_guesses function from Pymatgen.core.Composition

    See `oxi_state_guesses` for full details. This operation does the
    calculation of the most likely oxidation states

    Args:
        comp: A Pymatgen composition object.
        oxi_states_override (dict): dict of str->list to override an element's common oxidation states, e.g.
            {"V": [2,3,4,5]}.
        target_charge (float): the desired total charge on the structure. Default is 0 signifying charge balance.
        all_oxi_states (bool): if True, all oxidation states of an element, even rare ones, are used in the search
            for guesses. However, the full oxidation state list is *very* inclusive and can produce nonsensical
            results. If False, the icsd_oxidation_states list is used when present, or the common_oxidation_states
            is used when icsd_oxidation_states is not present. These oxidation states lists comprise more
            commonly occurring oxidation states and results in more reliable guesses, albeit at the cost of
            missing some uncommon situations. The default is False.
        max_sites (int): if possible, will reduce Compositions to at most
            this many sites to speed up oxidation state guesses. If the
            composition cannot be reduced to this many sites a ValueError
            will be raised. Set to -1 to just reduce fully. If set to a
            number less than -1, the formula will be fully reduced but a
            ValueError will be thrown if the number of atoms in the reduced
            formula is greater than abs(max_sites).

    Returns:
        list[dict]: Each dict maps the element symbol to a list of
            oxidation states for each site of that element. For example, Fe3O4 could
            return a list of [2,2,2,3,3,3] for the oxidation states of the 6 Fe sites.
            If the composition is not charge balanced, an empty list is returned.
    """
    # Reduce Composition if necessary
    if max_sites and max_sites < 0:
        comp = comp.reduced_composition

        if max_sites < -1 and comp.num_atoms > abs(max_sites):
            raise ValueError(
                f"Composition {comp} cannot accommodate max_sites setting!"
            )

    elif max_sites and comp.num_atoms > max_sites:
        reduced_comp, reduced_factor = comp.get_reduced_composition_and_factor()
        if reduced_factor > 1:
            reduced_comp *= max(1, int(max_sites / reduced_comp.num_atoms))
            comp = reduced_comp  # as close to max_sites as possible
        if comp.num_atoms > max_sites:
            raise ValueError(
                f"Composition {comp} cannot accommodate max_sites setting!"
            )

    # Prior probabilities of oxidation states, used to rank solutions.
    loaded_dict = _load_json(OXI_PROBS_FILE)
    type(comp).oxi_prob = loaded_dict
    oxi_states_override = oxi_states_override or {}
    # Assert Composition only has integer amounts
    if not all(amt == int(amt) for amt in comp.values()):
        raise ValueError(
            "Charge balance analysis requires integer values in Composition!"
        )

    # For each element, determine all possible sum of oxidations
    # (taking into account nsites for that particular element)
    el_amt = comp.get_el_amt_dict()
    n_sites = int(sum(el_amt.values()))
    elements = list(el_amt)
    el_sums: list = []  # matrix: dim1= el_idx, dim2=possible sums
    el_sum_scores: defaultdict = defaultdict(set)  # dict of el_idx, sum -> score
    el_best_oxid_combo: dict = {}  # dict of el_idx, sum -> oxid combo with best score

    for idx, el in enumerate(elements):
        el_sum_scores[idx] = {}
        el_best_oxid_combo[idx] = {}
        el_sums.append([])
        if oxi_states_override.get(el):
            oxids: list | tuple = oxi_states_override[el]
        elif all_oxi_states:
            oxids = Element(el).oxidation_states
        else:
            oxids = (
                Element(el).icsd_oxidation_states or Element(el).common_oxidation_states
            )

        # Get all possible combinations of oxidation states
        # and sum each combination
        for oxid_combo in combinations_with_replacement(oxids, int(el_amt[el])):
            # check to make sure none of the oxidation states deviate by more than 1 
            if max(oxid_combo) - min(oxid_combo) <= 1: 
                # List this sum as a possible option
                oxid_sum = sum(oxid_combo)
                if oxid_sum not in el_sums[idx]:
                    el_sums[idx].append(oxid_sum)

                # Determine how probable is this combo?
                if not all_oxi_states:
                    scores = []
                    for o in oxid_combo:
                        scores.append(type(comp).oxi_prob[str(Species(el, o))])
                    score = math.prod(scores)

                    # If it is the most probable combo for a certain sum,
                    # store the combination
                    if oxid_sum not in el_sum_scores[idx] or score > el_sum_scores[idx].get(
                        oxid_sum, 0):
                        
                        el_sum_scores[idx][oxid_sum] = score
                        el_best_oxid_combo[idx][oxid_sum] = oxid_combo
                            
            else:
                pass
    
    # Determine which combination of oxidation states for each element
    # is the most probable

    el_sums = [[x for x in sublist if x != 0] for sublist in el_sums]
    
    all_sols = []  # will contain all solutions
    all_oxid_combo = []  # will contain the best combination of oxidation states for each site
    all_scores = []  # will contain a score for each solution
    scores = []
    for x in product(*el_sums):
        # Each x is a trial of one possible oxidation sum for each element
        if sum(x) == target_charge:  # charge balance condition
            el_sum_sol = dict(zip(elements, x, strict=True))  # element->oxid_sum
            # Normalize oxid_sum by amount to get avg oxid state
            sol = {el: v / el_amt[el] for el, v in el_sum_sol.items()}
            # Add the solution to the list of solutions

                        
            all_sols.append(sol)
            
            if not all_oxi_states:
                # Determine the score for this solution
                scores = []
                for idx, v in enumerate(x):
                    scores.append(el_sum_scores[idx][v])
                # the score is geometric mean of the scores for all the elements in the compostion 
                all_scores.append(math.prod(scores)**(1/n_sites))
                # Collect the combination of oxidation states for each site
                all_oxid_combo.append(
                    {
                        e: el_best_oxid_combo[idx][v]
                        for idx, (e, v) in enumerate(zip(elements, x, strict=True))
                    }
                )
            else:
                all_scores.append(electronegativity_correlation(elements=list(sol.keys()), oxidation_states=list(sol.values())))


    # Sort the solutions from highest to lowest score
    if all_scores:
        if all_oxi_states:
            # For correlation: more negative is better (ascending sort)
            sorted_data = sorted(
                zip(all_scores, all_sols),
                key=lambda x: x[0]  # Sort by score
            )
            all_scores, all_sols = zip(*sorted_data)
            all_oxid_combo = all_sols
            return (
                tuple(all_sols),
                tuple(all_oxid_combo),
                tuple(all_scores),
            )

        else:
            # For probabilities: higher is better (descending sort)
            sorted_data = sorted(
                zip(all_scores, all_sols, all_oxid_combo),
                key=lambda x: x[0],
                reverse=True
            )
            all_scores, all_sols, all_oxid_combo = zip(*sorted_data)


            return (
                tuple(all_sols),
                tuple(all_oxid_combo),
                tuple(all_scores),
            )
    else:
        return (
            tuple(all_sols),
            tuple(all_oxid_combo),
            tuple(all_scores),
            )


def get_inequivalent_site_info(structure):
    """Gets the symmetrically inequivalent sites as found by the
    SpacegroupAnalyzer class from Pymatgen.

    Parameters
    ----------
    structure : pymatgen.core.structure.Structure
        The Pymatgen structure of interest.

    Returns
    -------
    dict
        A dictionary containing three lists, one of the inequivalent sites, one
        for the atom types they correspond to and the last for the multiplicity.
    """

    # Get the symmetrically inequivalent indexes
    inequivalent_sites = (
        SpacegroupAnalyzer(structure).get_symmetrized_structure().equivalent_indices
    )

    # Equivalent indexes must all share the same atom type
    multiplicities = [len(xx) for xx in inequivalent_sites]
    inequivalent_sites = [xx[0] for xx in inequivalent_sites]
    species = [str(structure[xx].specie) for xx in inequivalent_sites]

    return {
        "sites": inequivalent_sites,
        "species": species,
        "multiplicities": multiplicities,
    }


