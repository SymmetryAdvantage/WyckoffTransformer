from typing import Dict, Iterable, Set, FrozenSet, Optional, List, Tuple
from copy import deepcopy
import logging
from itertools import chain
from operator import attrgetter, itemgetter
from functools import partial
from collections import defaultdict
from enum import Enum
from pathlib import Path
import gzip
import pickle
from pandas import DataFrame
import torch
import omegaconf


from pyxtal.symmetry import Group

ServiceToken = Enum('ServiceToken', ['PAD', 'STOP', 'MASK'])

class EnumeratingTokeniser(dict):
    @classmethod
    def from_token_set(cls,
        all_tokens: Set|FrozenSet,
        max_tokens: Optional[int] = None):
        for special_token in ServiceToken:
            if special_token.name in all_tokens:
                raise ValueError(f"Special token {special_token.name} is in the dataset")
        instance = cls()
        instance.update({token: idx for idx, token in enumerate(
            chain(all_tokens, map(attrgetter('name'), ServiceToken)))})
        instance.stop_token = instance[ServiceToken.STOP.name]
        instance.pad_token = instance[ServiceToken.PAD.name]
        instance.mask_token = instance[ServiceToken.MASK.name]
        # Theoretically, we can check it in the beginnig, but
        # the performance hit is negligible
        if max_tokens is not None and len(instance) > max_tokens:
            raise ValueError(f"Too many tokens: {len(instance)}. Remember "
            f"that we also added {len(ServiceToken)} service tokens")
        instance.to_token = [token for token, idx in sorted(instance.items(), key=itemgetter(1))]
        return instance


    def tokenise_sequence(self,
                          sequence: Iterable,
                          original_max_len: int,
                          **tensor_args) -> torch.Tensor:
        tokenised_sequence = [self[token] for token in sequence]
        padding = [self.pad_token] * (original_max_len - len(tokenised_sequence))
        return torch.tensor(tokenised_sequence + [self.stop_token] + padding, **tensor_args)
    

    def tokenise_single(self, token, **tensor_args) -> torch.Tensor:
        return torch.tensor(self[token], **tensor_args)


def tokenise_dataset(datasets_pd: Dict[str, DataFrame],
                     config: omegaconf.OmegaConf) -> \
                        Tuple[Dict[str, Dict[str, torch.Tensor|List[List[torch.Tensor]]]], Dict[str, EnumeratingTokeniser]]:
    tokenisers = {}
    dtype = getattr(torch, config.dtype)
    max_tokens = torch.iinfo(dtype).max
    for token_field in config.token_fields.pure_categorical:
        all_tokens = frozenset(chain.from_iterable(chain.from_iterable(map(itemgetter(token_field), datasets_pd.values()))))
        tokenisers[token_field] = EnumeratingTokeniser.from_token_set(all_tokens, max_tokens)

    for sequence_field in config.sequence_fields.pure_categorical:
        all_tokens = frozenset(chain.from_iterable(map(lambda df: frozenset(df[sequence_field].tolist()), datasets_pd.values())))
        tokenisers[sequence_field] = EnumeratingTokeniser.from_token_set(all_tokens, max_tokens)

    # We don't check consistency among the fields here
    # The value is for the original sequences, withot service tokens
    original_max_len = max(map(len, chain.from_iterable(
        map(itemgetter(config.token_fields.pure_categorical[0]),
            datasets_pd.values()))))
    

    
    tensors = defaultdict(dict)
    for dataset_name, dataset in datasets_pd.items():
        for field in config.token_fields.pure_categorical:
            tensors[dataset_name][field] = torch.stack(
                dataset[field].map(partial(
                    tokenisers[field].tokenise_sequence,
                    original_max_len=original_max_len,
                    dtype=dtype)).to_list())
        for field in config.sequence_fields.pure_categorical:
            tensors[dataset_name][field] = torch.stack(
                dataset[field].map(partial(
                    tokenisers[field].tokenise_single,
                    dtype=dtype)).to_list())
        # Conuter fields are processed into two tensors: tokenised values, and the counts
        for field, tokeniser_filed in config.sequence_fields.counters:
            tensors[dataset_name][f"{field}_tokens"] = torch.stack(
                    dataset[field].map(lambda dict_:

                        
                        
                    dtype=dtype)).to_list())
            
        for field in config.augmented_token_fields:
            augmented_field = f"{field}_augmented"
            tensors[dataset_name][augmented_field] = dataset[augmented_field].map(lambda variants:
                    [tokenisers[field].tokenise_sequence(
                        variant, original_max_len=original_max_len, dtype=dtype)
                        for variant in variants]).to_list()
        # Assuming all the fields have the same length
        tensors[dataset_name]["pure_sequence_length"] = torch.tensor(
            dataset[config.token_fields.pure_categorical[0]].map(len).to_list(), dtype=dtype)
    return tensors, tokenisers


def load_tensors_and_tokenisers(
    dataset: str,
    config_name: str,
    cache_path: Path = Path(__file__).parent.parent.resolve() / "cache"):
    
    this_cache_path = cache_path / dataset
    with gzip.open(this_cache_path / 'tensors' / f'{config_name}.pkl.gz', "rb") as f:
        tensors = pickle.load(f)
    with gzip.open(this_cache_path / 'tokenisers' / f'{config_name}.pkl.gz', "rb") as f:
        tokenisers = pickle.load(f)
    return tensors, tokenisers


def get_wp_index() -> dict:
    wp_index = dict()
    for group_number in range(1, 231):
        group = Group(group_number)
        wp_index[group_number] = defaultdict(dict)
        for wp in group.Wyckoff_positions:
            wp.get_site_symmetry()
            wp_index[group_number][wp.site_symm][wp.letter] = (wp.multiplicity, wp.get_dof())
    return wp_index


def get_letter_from_ss_enum_idx(
    enum_tokeniser: EnumeratingTokeniser) -> dict:
    """
    Processes the real-space index of Wyckhoff letters by space group, site symmetry, and enumeration
    into a dict indexed by space group, site symmetry, and enumeration TOKEN to make the generation
    a tiny-tiny little bit faster.
    """
    preprocessed_wyckhoffs_cache_path = Path(__file__).parent.parent.resolve() / "cache" / "wychoffs_enumerated_by_ss.pkl.gz"
    with open(preprocessed_wyckhoffs_cache_path, "rb") as f:
        letter_from_ss_enum = pickle.load(f)[1]
    letter_from_ss_enum_idx = defaultdict(dict)
    for space_group, ss_enum_dict in letter_from_ss_enum.items():
        for ss, enum_dict in ss_enum_dict.items():
            letter_from_ss_enum_idx[space_group][ss] = dict()
            for enum, letter in enum_dict.items():
                letter_from_ss_enum_idx[space_group][ss][enum_tokeniser[enum]] = letter
    return letter_from_ss_enum_idx  


pyxtal_cascade_order = ("elements", "site_symmetries", "sites_enumeration")

def tensor_to_pyxtal(
    space_group_tensor: torch.Tensor,
    wp_tensor: torch.Tensor,
    tokenisers: Dict[str, EnumeratingTokeniser],
    cascade_order: Tuple[str, ...],
    letter_from_ss_enum_idx,
    wp_index,
    enforced_min_elements: Optional[int] = None,
    enforced_max_elements: Optional[int] = None) -> Optional[dict]:
    """
    This function has a lot of expectations.
    1. The cascade order is ("elements", "site_symmetries", "sites_enumeration")
    2. Those exact fields are also present in the tokenisers
    3. "spacegroup_number" is also in the tokenisers
    Args:
        space_group_tensor: The tensor with the space group
        wp_tensor: The tensor with the Wyckoff positions
        tokenisers: The tokenisers for the fields
        letter_from_ss_enum_idx: A dict with the Wyckoff positions
            indexed by space group, site symmetry, and enumeration TOKEN
        enforced_min_elements: The minimum number of elements in the structure
        enforced_max_elements: The maximum number of elements in the structure

    """
    cascade_permutation = [cascade_order.index(field) for field in pyxtal_cascade_order]
    cononical_wp_tensor = wp_tensor[:, cascade_permutation]

    stop_tokens = torch.tensor([tokenisers[field].stop_token for field in pyxtal_cascade_order], device=wp_tensor.device)
    mask_tokens = torch.tensor([tokenisers[field].mask_token for field in pyxtal_cascade_order], device=wp_tensor.device)
    pad_tokens = torch.tensor([tokenisers[field].pad_token for field in pyxtal_cascade_order], device=wp_tensor.device)

    space_group_real = tokenisers["spacegroup_number"].to_token[space_group_tensor.item()]
    pyxtal_args = defaultdict(lambda: [0, []])
    available_sites = deepcopy(wp_index[space_group_real])
    for this_token_cascade in cononical_wp_tensor:
        # Valid stop. In principle, tokens should only be either with
        # all stops, or all not stops, so an argumnet can be made for
        # invalidating the structure.
        if (this_token_cascade == stop_tokens).any():
            break
        if (this_token_cascade == pad_tokens).any():
            logging.info("PAD token in generated sequence")
            return None
        if (this_token_cascade == mask_tokens).any():
            logging.info("MASK token in generated sequence")
            return None
        element_idx, ss_idx, enum_idx = this_token_cascade.tolist()
        ss = tokenisers["site_symmetries"].to_token[ss_idx]
        try:
            wp_letter = letter_from_ss_enum_idx[space_group_real][ss][enum_idx]
        except KeyError:
            logging.info("Invalid combination: space group %i, site symmetry %s, enum token %i", space_group_real,
                         ss, enum_idx)
            return None
        try:
            our_site = available_sites[ss][wp_letter]
        except KeyError:
            logging.info("Repeated special WP: %i, %s, %s", space_group_real, ss, wp_letter)
            return None
        element = tokenisers["elements"].to_token[element_idx]
        pyxtal_args[element][0] += our_site[0]
        pyxtal_args[element][1].append(str(our_site[0]) + wp_letter)
        if our_site[1] == 0: # The position is special
            del available_sites[ss][wp_letter]
    if enforced_min_elements is not None and len(pyxtal_args.keys()) < enforced_min_elements:
        logging.info("Not enough elements")
        return None
    if enforced_max_elements is not None and len(pyxtal_args.keys()) > enforced_max_elements:
        logging.info("Too many elements")
        return None
    if len(pyxtal_args) == 0:
        logging.info("No structure generated, STOP in the first token")
        return None
    return {
            "group": space_group_real,
            "sites": [x[1] for x in pyxtal_args.values()],
            "species": list(map(str, pyxtal_args.keys())),
            "numIons": [x[0] for x in pyxtal_args.values()]
        }
