from typing import Dict, Iterable, Set, FrozenSet, Optional, List, Tuple
from copy import deepcopy
import logging
from itertools import chain
from operator import attrgetter, itemgetter
from functools import partial
from collections import defaultdict, UserDict
from enum import Enum
from pathlib import Path
import gzip
import pickle
import numpy as np
from pandas import DataFrame, Series, MultiIndex
import torch
import omegaconf
from pandarallel import pandarallel
from pyxtal.symmetry import Group

from .pyxtal_fix import SS_CORRECTIONS

# Order is important here, as we can use it to sort the tokens
ServiceToken = Enum('ServiceToken', ['MASK', 'STOP', 'PAD'])
logger = logging.getLogger(__name__)

class TupleDict(UserDict):
    def __getitem__(self, key):
        if isinstance(key, tuple):
            return super().__getitem__(key)
        return super().__getitem__(tuple(key))


class SpaceGroupEncoder(dict):
    """
    Encodes the spacegroup number as a one-hot tensor via
    get_spg_symmetry_object().to_matrix_representation_spg()
    Removes constants among the present groups.
    """
    @classmethod
    def from_sg_set(cls, all_space_groups: Set[int]|FrozenSet[int]):
        symbols = ("P", "A", "C", "I", "R", "F")
        symbols_one_hot_matrix = np.eye(len(symbols))
        symbol_to_one_hot = dict(zip(symbols, symbols_one_hot_matrix))
        all_spgs_raw = dict()
        for group_number in all_space_groups:
            group = Group(group_number)
            all_spgs_raw[group_number] = np.concatenate(
                [group.get_spg_symmetry_object().to_matrix_representation_spg().ravel(),
                 symbol_to_one_hot[group.symbol[0]]])
        all_spgs_sum = sum(all_spgs_raw.values())
        varying_indices = ~((all_spgs_sum == 0) | (all_spgs_sum == len(all_spgs_raw)))
        logger.info("Space group one-hot encoding: %i groups, %i varying elements", len(all_space_groups), varying_indices.sum())
        instance = cls()
        # Numpy array is unhashable, so we convert it to tuple for compatibility reasons
        instance.np_dict = dict()
        instance.to_token = TupleDict()
        for group_number, spg in all_spgs_raw.items():
            instance.np_dict[group_number] = spg[varying_indices]
            instance[group_number] = tuple(spg[varying_indices])
            instance.to_token[tuple(spg[varying_indices])] = group_number
        return instance


    def encode_spacegroups(self, space_groups: Iterable[int], **tensor_args) -> torch.Tensor:
        """
        Returns a one-hot encoded vector for each space group.
        Args:
            space_groups [batch_size]: An iterable of space group numbers.
        Returns:
            [batch_size, n_features]: A tensor with one-hot encoded space groups.
            n_fetures depends on the number of varying elements in the space groups present
            in constructor, so it might be different for different datsets.
        """
        return torch.stack([torch.from_numpy(self.np_dict[sg]) for sg in space_groups]).to(**tensor_args)


class EnumeratingTokeniser(dict):
    @classmethod
    def from_token_set(cls,
        all_tokens: Set|FrozenSet,
        max_tokens: Optional[int] = None,
        include_stop: bool = True):
        for special_token in ServiceToken:
            if special_token.name in all_tokens:
                raise ValueError(f"Special token {special_token.name} is in the dataset")
        instance = cls()
        instance.update({token: idx for idx, token in enumerate(
            chain(sorted(all_tokens), map(attrgetter('name'), ServiceToken)))})
        instance.stop_token = instance[ServiceToken.STOP.name]
        instance.pad_token = instance[ServiceToken.PAD.name]
        instance.mask_token = instance[ServiceToken.MASK.name]
        instance.include_stop = include_stop
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
        if self.include_stop:
            padding = [self.stop_token] + padding
        return torch.tensor(tokenised_sequence + padding, **tensor_args)
    

    def tokenise_single(self, token, **tensor_args) -> torch.Tensor:
        return torch.tensor(self[token], **tensor_args)


class FeatureEngineer():
    def __init__(self,
            data: Dict[Tuple, int]|Series,
            inputs: Optional[Tuple] = None,
            name: Optional[str] = None,
            stop_token: Optional[int] = None,
            pad_token: Optional[int] = None,
            mask_token: Optional[int] = None,
            include_stop: bool = True):
        if isinstance(data, Series):
            if inputs is not None or name is not None:
                raise ValueError("If data is a DataFrame, inputs and name should be None")
            self.db = data
        else:
            index = MultiIndex.from_tuples(data.keys(), names=inputs)
            self.db = Series(data=data.values(), index=index, name=name)
        self.inputs = self.db.index.names
        self.stop_token = stop_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.default_value = 0
        self.include_stop = include_stop

    def get_feature_tensor_from_series(
        self,
        record: Series,
        original_max_len: int,
        **tensor_args) -> torch.Tensor:

        indexed_record = record.loc[self.db.index.names]
        # No need to write the general solution, for now we only need multiplicity
        # WARNING(kazeevn): only one structure is supported:
        # the first input is sequence-level, the next two are token-level    
        this_db = self.db.loc[indexed_record.iloc[0]]
        # Beautiful, but slow
        res = this_db.loc[map(tuple, zip(*indexed_record.iloc[1:]))].to_list()
        # Since in our infinite wisdom we decided to compute multiplicity
        # two times, we might as well just check
        if self.db.name in record.index:
            if record[self.db.name] != res:
                logger.error("Record")
                logger.error(record)
                logger.error("Mismatch in %s", self.db.name)
                logger.error(record[self.db.name])
                logger.error(res)
                raise ValueError("Mismatch")
        padding = [self.pad_token] * (original_max_len - len(res))
        if self.include_stop:
            padding = [self.stop_token] + padding
        return torch.tensor(res + padding, **tensor_args)
    
    def get_feature_from_token_batch(
        self,
        level_0: torch.Tensor,
        levels_plus: List[torch.Tensor]):
        """
        Every tensor has shape [batch_size]
        """
        return self.db.reindex(map(tuple, zip(level_0, *levels_plus)), fill_value=self.default_value).values


class DummyItemGetter():
    def __getitem__(self, key):
        return key


class PassThroughTokeniser():
    def __init__(self, min_value:int, max_value:int,
        stop_token: Optional[int] = None,
        pad_token: Optional[int] = None,
        mask_token: Optional[int] = None):
        """
        min_value and max_value should include the service tokens
        """

        self.min_value = min_value
        self.max_value = max_value
        self.stop_token = stop_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.to_token = DummyItemGetter()
    
    def __len__(self):
        return self.max_value - self.min_value + 1

    def __getitem__(self, token):
        return token


def tokenise_engineer(
    engineer: FeatureEngineer,
    tokenisers: EnumeratingTokeniser):
    
    include_stop_all = []
    for field in engineer.db.index.names:
        if hasattr(tokenisers[field], "include_stop"):
            include_stop_all.append(tokenisers[field].include_stop)

    if all(include_stop_all):
        include_stop = True
    elif not any(include_stop_all):
        include_stop = False
    else:
        raise ValueError("Inconsistent include_stop")
    tokenised_data = {}
    for index, value in engineer.db.items():
        try:
            new_index = tuple((tokenisers[field][this_index] for field, this_index in zip(engineer.db.index.names, index)))
        except KeyError:
            continue
        tokenised_data[new_index] = value
    return FeatureEngineer(tokenised_data, engineer.db.index.names, name=engineer.db.name,
        stop_token=engineer.stop_token, pad_token=engineer.pad_token, mask_token=engineer.mask_token,
        include_stop=include_stop)


def argsort_multiple(*tensors: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Argsorts the tensors along the dim dimension. The order is determined by
    the first tensor, in case of a tie the second tensor is used, and so on.
    Args:
        tensors: The tensors to argsort
        dim: The dimension to argsort along
    Returns:
        A tensor with the indices of the sorted tensors
    """
    if len(tensors) == 1:
        return torch.argsort(tensors[0], dim=dim)
    elif len(tensors) == 2:
        if tensors[0].dtype != torch.uint8 or tensors[1].dtype != torch.uint8:
            # Will need to check for overflow
            raise NotImplementedError("Only uint8 tensors are supported")
        megaindex = tensors[0].type(torch.int64) * tensors[1].max().type(torch.int64) + tensors[1].type(torch.int64)
        return torch.argsort(megaindex)
    else:
        raise NotImplementedError("Only one or two tensors are supported")


def tokenise_dataset(datasets_pd: Dict[str, DataFrame],
                     config: omegaconf.OmegaConf,
                     tokenizer_path: Optional[Path|str] = None,
                     n_jobs: int = None) -> \
                        Tuple[Dict[str, Dict[str, torch.Tensor|List[List[torch.Tensor]]]], Dict[str, EnumeratingTokeniser]]:
    dtype = getattr(torch, config.dtype)
    include_stop = config.get("include_stop", True)
    pandarallel.initialize()
    if tokenizer_path is None:
        tokenisers = {}
        max_tokens = torch.iinfo(dtype).max
        for token_field in config.token_fields.pure_categorical:
            all_tokens = frozenset(chain.from_iterable(chain.from_iterable(map(itemgetter(token_field), datasets_pd.values()))))
            tokenisers[token_field] = EnumeratingTokeniser.from_token_set(all_tokens, max_tokens, include_stop=include_stop)

        if "pure_categorical" in config.sequence_fields:
            # Cell variable sequence_field defined in loopPylintW0640:cell-var-from-loop
            for sequence_field in config.sequence_fields.pure_categorical:
                all_tokens = frozenset(chain.from_iterable(map(lambda df: frozenset(df[sequence_field].tolist()), datasets_pd.values())))
                tokenisers[sequence_field] = EnumeratingTokeniser.from_token_set(all_tokens, max_tokens)
        
        if "space_group" in config.sequence_fields:
            for sequence_field in config.sequence_fields.space_group:
                all_space_groups = frozenset(chain.from_iterable(map(itemgetter(sequence_field), datasets_pd.values())))
                tokenisers[sequence_field] = SpaceGroupEncoder.from_sg_set(all_space_groups)
    else:
        with gzip.open(tokenizer_path, "rb") as f:
            tokenisers = pickle.load(f)

    raw_engineers = {}
    token_engineers = dict()
    if "engineered" in config.token_fields:
        for engineered_field_name, engineered_field_definiton in config.token_fields.engineered.items():
            if engineered_field_definiton.type != "map":
                raise ValueError("Only map engineered_field fields are supported")
            if len(engineered_field_definiton.inputs) != 3:
                raise NotImplementedError("Only 3 inputs are supported")
            with gzip.open(Path(__file__).parent.parent.resolve() / "cache" / "engineers" / f"{engineered_field_name}.pkl.gz", "rb") as f:
                raw_engineer = pickle.load(f)
            raw_engineers[engineered_field_name] = raw_engineer
            # Now we need to convert the token values to token indices
            # And adjust include_stop
            token_engineers[engineered_field_name] = tokenise_engineer(raw_engineer, tokenisers)
            raw_engineers[engineered_field_name].include_stop = token_engineers[engineered_field_name].include_stop
            # The values haven't changed, only the keys, so we can reuse the stop, pad, and mask tokens
            tokenisers[engineered_field_name] = PassThroughTokeniser(
                min(token_engineers[engineered_field_name].db.min().min(), raw_engineer.stop_token, raw_engineer.pad_token, raw_engineer.mask_token),
                max(token_engineers[engineered_field_name].db.max().max(), raw_engineer.stop_token, raw_engineer.pad_token, raw_engineer.mask_token),
                stop_token = raw_engineer.stop_token,
                pad_token = raw_engineer.pad_token,
                mask_token = raw_engineer.mask_token)

    # We don't check consistency among the fields here
    # The value is for the original sequences, withot service tokens
    original_max_len = max(map(len, chain.from_iterable(
        map(itemgetter(config.token_fields.pure_categorical[0]),
            datasets_pd.values()))))
  
    tensors = defaultdict(dict)
    for dataset_name, dataset in datasets_pd.items():
        dataset = dataset[dataset['spacegroup_number'].isin(tokenisers['spacegroup_number'])]

        for field in config.token_fields.pure_categorical:
            tensors[dataset_name][field] = torch.stack(
                dataset[field].map(partial(
                    tokenisers[field].tokenise_sequence,
                    original_max_len=original_max_len,
                    dtype=dtype)).to_list())

        if "engineered" in config.token_fields:
            for field in config.token_fields.engineered:
                compute_feature_function = partial(
                    raw_engineers[field].get_feature_tensor_from_series,
                    original_max_len=original_max_len,
                    dtype=dtype)
                tensor_list = dataset.parallel_apply(
                    compute_feature_function, axis=1).to_list()
                tensors[dataset_name][field] = torch.stack(tensor_list)
                logger.debug("Engineered field %s shape %s", field, tensors[dataset_name][field].shape)
   
        if "pure_categorical" in config.sequence_fields:
            for field in config.sequence_fields.pure_categorical:
                tensors[dataset_name][field] = torch.stack(
                    dataset[field].map(partial(
                        tokenisers[field].tokenise_single,
                        dtype=dtype)).to_list())

        if "space_group" in config.sequence_fields:
            for field in config.sequence_fields.space_group:
                tensors[dataset_name][field] = tokenisers[field].encode_spacegroups(dataset[field], dtype=dtype)

        if "no_processing" in config.sequence_fields:
            for field in config.sequence_fields.no_processing:
                tensors[dataset_name][field] = torch.Tensor(dataset[field].array)

        if "counters" in config.sequence_fields:
            # Counter fields are processed into two tensors: tokenised values, and the counts
            # WARNING Cell variable tokeniser_filed defined in loopPylintW0640:cell-var-from-loop
            for field, tokeniser_field in config.sequence_fields.counters.items():
                tensors[dataset_name][f"{field}_tokens"] = \
                        dataset[field].map(lambda dict_:
                            torch.stack([tokenisers[tokeniser_field].tokenise_single(key, dtype=dtype)
                                for key in dict_.keys()])).to_list()
                tensors[dataset_name][f"{field}_counts"] = \
                        dataset[field].map(lambda dict_:
                            torch.tensor(tuple(dict_.values()), dtype=dtype)).to_list()

        if "augmented_token_fields" in config:
            # WARNING Cell variable field defined in loopPylintW0640:cell-var-from-loop
            for field in config.augmented_token_fields:
                augmented_field = f"{field}_augmented"
                tensors[dataset_name][augmented_field] = dataset[augmented_field].map(lambda variants:
                        [tokenisers[field].tokenise_sequence(
                            variant, original_max_len=original_max_len, dtype=dtype)
                            for variant in variants]).to_list()
        # We can have long sequences, but still a limited number of tokens
        if "pure_sequence_length_dtype" in config:
            pure_sequence_length_dtype = getattr(torch, config.pure_sequence_length_dtype)
        else:
            pure_sequence_length_dtype = dtype
        # Assuming all the fields have the same length
        tensors[dataset_name]["pure_sequence_length"] = torch.tensor(
            dataset[config.token_fields.pure_categorical[0]].map(len).to_list(),
            dtype=pure_sequence_length_dtype)

        if "token_sort" in config:
            if "augmented_token_fields" in config:
                raise NotImplementedError("Token sort is not implemented for augmented fields")
            key_tensors = [tensors[dataset_name][field] for field in config.token_sort]
            order = argsort_multiple(*key_tensors, dim=1)
            logger.debug("Order tensor")
            logger.debug(order[:5])
            fields_to_sort = list(config.token_fields.pure_categorical)
            fields_to_sort.extend(config.token_fields.get("engineered", []))
            for field in fields_to_sort:
                logger.debug("Sorting tensor %s", field)
                logger.debug(tensors[dataset_name][field][:5])
                tensors[dataset_name][field] = tensors[dataset_name][field].gather(1, order)
                logger.debug("Sorted tensor %s", field)
                logger.debug(tensors[dataset_name][field][:5])
    return tensors, tokenisers, token_engineers


def load_tensors_and_tokenisers(
    dataset: str,
    config_name: str,
    cache_path: Path = Path(__file__).parent.parent.resolve() / "cache"):

    this_cache_path = cache_path / dataset
    with gzip.open(this_cache_path / 'tensors' / f'{config_name}.pkl.gz', "rb") as f:
        tensors = pickle.load(f)
    with gzip.open(this_cache_path / 'tokenisers' / f'{config_name}.pkl.gz', "rb") as f:
        tokenisers = pickle.load(f)
        token_engineers = pickle.load(f)
    return tensors, tokenisers, token_engineers


def get_wp_index() -> dict:
    wp_index = dict()
    for group_number in range(1, 231):
        group = Group(group_number)
        wp_index[group_number] = defaultdict(dict)
        for wp in group.Wyckoff_positions:
            wp.get_site_symmetry()
            try:
                site_symm = SS_CORRECTIONS[group_number][wp.letter]
            except KeyError:
                site_symm = wp.site_symm
            wp_index[group_number][site_symm][wp.letter] = (wp.multiplicity, wp.get_dof())
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
                if enum in enum_tokeniser:
                    letter_from_ss_enum_idx[space_group][ss][enum_tokeniser[enum]] = letter
    return letter_from_ss_enum_idx  


def tensor_to_pyxtal(
    space_group_tensor: torch.Tensor,
    wp_tensor: torch.Tensor,
    tokenisers: Dict[str, EnumeratingTokeniser],
    cascade_order: Tuple[str, ...],
    letter_from_ss_enum_idx,
    ss_from_letter,
    wp_index,
    enforced_min_elements: Optional[int] = None,
    enforced_max_elements: Optional[int] = None) -> Optional[dict]:
    """
    The functio supports two cascade modes:
        (elements, site_symmetries, sites_enumeration)
        (elements, wyckoff_letters)
    Args:
        space_group_tensor: The tensor with the space group
        wp_tensor: The tensor with the Wyckoff positions
        tokenisers: The tokenisers for the fields
        letter_from_ss_enum_idx: A dict with the Wyckoff positions
            indexed by space group, site symmetry, and enumeration TOKEN
        enforced_min_elements: The minimum number of elements in the structure
        enforced_max_elements: The maximum number of elements in the structure

    """
    ss_pyxtal_cascde_order = ("elements", "site_symmetries", "sites_enumeration")
    letters_pyxtal_cascade_order = ("elements", "wyckoff_letters")
    if set(cascade_order) == set(ss_pyxtal_cascde_order):
        pyxtal_cascade_order = ss_pyxtal_cascde_order
        ss_mode = True
    elif set(cascade_order) == set(letters_pyxtal_cascade_order):
        pyxtal_cascade_order = letters_pyxtal_cascade_order
        ss_mode = False
    else:
        raise NotImplementedError("Unsupported cascade")

    cascade_permutation = [cascade_order.index(field) for field in pyxtal_cascade_order]
    cononical_wp_tensor = wp_tensor[:, cascade_permutation]

    stop_tokens = torch.tensor([tokenisers[field].stop_token for field in pyxtal_cascade_order], device=wp_tensor.device)
    mask_tokens = torch.tensor([tokenisers[field].mask_token for field in pyxtal_cascade_order], device=wp_tensor.device)
    pad_tokens = torch.tensor([tokenisers[field].pad_token for field in pyxtal_cascade_order], device=wp_tensor.device)

    if space_group_tensor.size() == (1,):
        space_group_input = space_group_tensor.item()
    else:
        space_group_input = space_group_tensor.numpy()
    space_group_real = tokenisers["spacegroup_number"].to_token[space_group_input]
    pyxtal_args = defaultdict(lambda: [0, []])
    available_sites = deepcopy(wp_index[space_group_real])
    for this_token_cascade in cononical_wp_tensor:
        # Valid stop. In principle, tokens should only be either with
        # all stops, or all not stops, so an argumnet can be made for
        # invalidating the structure.
        if (this_token_cascade == stop_tokens).any():
            break
        if (this_token_cascade == pad_tokens).any():
            logger.info("PAD token in generated sequence")
            return
        if (this_token_cascade == mask_tokens).any():
            logger.info("MASK token in generated sequence")
            return
        if ss_mode:
            element_idx, ss_idx, enum_idx = this_token_cascade.tolist()
            ss = tokenisers["site_symmetries"].to_token[ss_idx]
            try:
                wp_letter = letter_from_ss_enum_idx[space_group_real][ss][enum_idx]
            except KeyError:
                logger.info("Invalid combination: space group %i, site symmetry %s, enum token %i", space_group_real,
                            ss, enum_idx)
                return
        else:
            element_idx, wp_letter_idx = this_token_cascade.tolist()
            wp_letter = tokenisers["wyckoff_letters"].to_token[wp_letter_idx]
            try:
                ss = ss_from_letter[space_group_real][wp_letter]
            except KeyError:
                logger.info("Invalid combination: space group %i, wp letter %s", space_group_real, wp_letter)
                return
        try:
            our_site = available_sites[ss][wp_letter]
        except KeyError:
            logger.info("Repeated special WP: %i, %s, %s", space_group_real, ss, wp_letter)
            return
        element = tokenisers["elements"].to_token[element_idx]
        pyxtal_args[element][0] += our_site[0]
        pyxtal_args[element][1].append(str(our_site[0]) + wp_letter)
        if our_site[1] == 0: # The position is special
            del available_sites[ss][wp_letter]
    if enforced_min_elements is not None and len(pyxtal_args.keys()) < enforced_min_elements:
        logger.info("Not enough elements")
        return
    if enforced_max_elements is not None and len(pyxtal_args.keys()) > enforced_max_elements:
        logger.info("Too many elements")
        return
    if len(pyxtal_args) == 0:
        logger.info("No structure generated, STOP in the first token")
        return
    return {
            "group": space_group_real,
            "sites": [x[1] for x in pyxtal_args.values()],
            "species": list(map(str, pyxtal_args.keys())),
            "numIons": [x[0] for x in pyxtal_args.values()]
        }
