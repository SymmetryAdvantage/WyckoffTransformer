from typing import Dict, Iterable, NamedTuple, Set, FrozenSet, Optional, List, Tuple
import json
import gzip
import logging
import pickle
from itertools import chain
from operator import attrgetter, itemgetter
from collections import defaultdict, UserDict
from enum import Enum
from pathlib import Path
import numpy as np
from pandas import DataFrame
import torch
from pyxtal.symmetry import Group
from omegaconf import OmegaConf, DictConfig

from wyckoff_transformer.wyckoff_processor import (
    FeatureEngineer,
    WyckoffProcessor,
)

WYCKOFF_MAPPINGS_FILENAME = "wyckoffs_enumerated_by_ss.json"
_PACKAGE_MAPPINGS_PATH = Path(__file__).parent / WYCKOFF_MAPPINGS_FILENAME


class WyckoffMappings(NamedTuple):
    """Wyckoff position mappings for all 230 3-D space groups.

    Attributes:
        enum_from_ss_letter: sg -> letter -> enumeration index within the site symmetry.
        letter_from_ss_enum: sg -> site_symmetry -> enumeration index -> Wyckoff letter.
        ss_from_letter:      sg -> letter -> site symmetry string.
    """
    enum_from_ss_letter: dict
    letter_from_ss_enum: dict
    ss_from_letter: dict


def load_wyckoff_mappings(path: Optional[Path] = None) -> WyckoffMappings:
    """Load Wyckoff position mappings from a model directory or package data.

    Args:
        path: Directory containing wyckoffs_enumerated_by_ss.json.
              If None, loads from the installed package data.
    Returns:
        WyckoffMappings named tuple with fields
        enum_from_ss_letter, letter_from_ss_enum, ss_from_letter.
    """
    json_file = (Path(path) / WYCKOFF_MAPPINGS_FILENAME) if path is not None else _PACKAGE_MAPPINGS_PATH
    with open(json_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return WyckoffMappings(
        enum_from_ss_letter={int(sg): v for sg, v in raw["enum_from_ss_letter"].items()},
        letter_from_ss_enum={
            int(sg): {ss: {int(e): letter for e, letter in ed.items()} for ss, ed in sd.items()}
            for sg, sd in raw["letter_from_ss_enum"].items()
        },
        ss_from_letter={int(sg): v for sg, v in raw["ss_from_letter"].items()},
    )


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
    get_spg_symmetry_object().to_matrix_representation()
    Removes constants among the present groups.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.np_dict = {
            group_number: np.array(encoded_group)
            for group_number, encoded_group in self.items()
        }
        self.to_token = TupleDict(
            {tuple(encoded_group): group_number for group_number, encoded_group in self.items()}
        )

    @classmethod
    def from_sg_set(cls, all_space_groups: Set[int]|FrozenSet[int]):
        symbols = ("P", "A", "B", "C", "I", "R", "F")
        symbols_one_hot_matrix = np.eye(len(symbols))
        symbol_to_one_hot = dict(zip(symbols, symbols_one_hot_matrix))
        all_spgs_raw = dict()
        for group_number in all_space_groups:
            group = Group(group_number)
            all_spgs_raw[group_number] = np.concatenate(
                [group.get_spg_symmetry_object().to_matrix_representation().ravel(),
                 symbol_to_one_hot[group.symbol[0]]])
        all_spgs_sum = sum(all_spgs_raw.values())
        varying_indices = ~((all_spgs_sum == 0) | (all_spgs_sum == len(all_spgs_raw)))
        logger.info("Space group one-hot encoding: %i groups, %i varying elements", len(all_space_groups), varying_indices.sum())
        # Numpy array is unhashable, so we convert it to tuple for compatibility reasons
        encoded_mapping = {}
        for group_number, spg in all_spgs_raw.items():
            encoded_mapping[group_number] = tuple(spg[varying_indices])
        instance = cls(encoded_mapping)

        if len(set(instance.values())) != len(all_spgs_raw):
            raise ValueError("Space group encoding is not unique")
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
    def __init__(self, *args, stop_token=None, pad_token=None, mask_token=None,
                 include_stop=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_token: Optional[int] = stop_token
        self.pad_token: Optional[int] = pad_token
        self.mask_token: Optional[int] = mask_token
        self.include_stop: bool = include_stop
        self.to_token: List = [token for token, idx in sorted(self.items(), key=itemgetter(1))]

    @classmethod
    def from_token_set(cls,
        all_tokens: Set|FrozenSet,
        max_tokens: Optional[int] = None,
        include_stop: bool = True):
        for special_token in ServiceToken:
            if special_token.name in all_tokens:
                raise ValueError(f"Special token {special_token.name} is in the dataset")
        token_map = {token: idx for idx, token in enumerate(
            chain(sorted(all_tokens), map(attrgetter('name'), ServiceToken)))}
        instance = cls(
            token_map,
            stop_token=token_map[ServiceToken.STOP.name],
            pad_token=token_map[ServiceToken.PAD.name],
            mask_token=token_map[ServiceToken.MASK.name],
            include_stop=include_stop,
        )
        # Theoretically, we can check it in the beginnig, but
        # the performance hit is negligible
        if max_tokens is not None and len(instance) > max_tokens:
            raise ValueError(f"Too many tokens: {len(instance)}. Remember "
            f"that we also added {len(ServiceToken)} service tokens")
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

    def get_letter_from_ss_enum_idx(self, path: Optional[Path] = None) -> dict:
        """
        Processes the real-space index of Wyckhoff letters by space group, site symmetry,
        and enumeration into a dict indexed by space group, site symmetry, and
        enumeration TOKEN to make generation a little faster.

        Args:
            path: Directory containing wyckoffs_enumerated_by_ss.json.
                  If None, loads from the installed package data.
        """
        letter_from_ss_enum = load_wyckoff_mappings(path).letter_from_ss_enum
        letter_from_ss_enum_idx = defaultdict(dict)
        for space_group, ss_enum_dict in letter_from_ss_enum.items():
            for ss, enum_dict in ss_enum_dict.items():
                letter_from_ss_enum_idx[space_group][ss] = dict()
                for enum, letter in enum_dict.items():
                    if enum in self:
                        letter_from_ss_enum_idx[space_group][ss][self[enum]] = letter
        return letter_from_ss_enum_idx


class DummyItemGetter():
    def __getitem__(self, key):
        return key


class PassThroughTokeniser():
    def __init__(self,
        values_count: int,
        stop_token: Optional[int] = None,
        pad_token: Optional[int] = None,
        mask_token: Optional[int] = None):
        # Values count includes the service tokens

        self.values_count = values_count
        self.stop_token = stop_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.to_token = DummyItemGetter()
    
    def __len__(self):
        return self.values_count

    def __getitem__(self, token):
        return token


TokeniserType = EnumeratingTokeniser | SpaceGroupEncoder | PassThroughTokeniser


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
        default_value=engineer.default_value, include_stop=include_stop)


def tokenise_dataset(datasets_pd: Dict[str, DataFrame],
                     config: DictConfig,
                     tokenizer_path: Optional[Path|str] = None,
                     n_jobs: Optional[int] = None) -> \
                        Tuple[Dict[str, Dict[str, torch.Tensor|List[List[torch.Tensor]]]],
                              Dict[str, EnumeratingTokeniser]]:
    """
    Tokenises the dataset according to the config. If tokenizer_path is provided, it loads the tokenisers from the path instead of creating new ones.
    Args:
        datasets_pd: A dict with the dataset name as key and the dataset as a pandas DataFrame as value. We must pass the all the data to ensure that
            every possible token is included in the tokeniser, even if it is not present in a specific split.
        config: The config for the tokenisation, see the yamls/tokenisers folder for examples.
        tokenizer_path: The path to the tokenisers. If None, new tokenisers are created.
        n_jobs: The number of jobs to use for parallel processing. If None, it uses the default number of physical cores.
    Returns:
        A tuple with the tokenised tensors and the tokenisers.
            The tokenised tensors are a dict with the dataset name as key and a dict with the field name as key and the tokenised tensor as value.
            The tokenisers are a dict with the field name as key and the tokeniser as value.
    """
    processor = WyckoffProcessor.from_config(config, tokenizer_path=tokenizer_path)
    return processor.tokenise_dataset(datasets_pd=datasets_pd, n_jobs=n_jobs)


def load_tensors_and_tokenisers(
    dataset: str,
    config_name: str,
    use_cached_tensors: bool = True,
    cache_path: Path = Path(__file__).resolve().parents[2] / "cache",
    tokenizer_path: Optional[Path] = None):

    this_cache_path = cache_path / dataset
    if use_cached_tensors:
        processor = WyckoffProcessor.from_pretrained(this_cache_path / "tokenisers" / f"{config_name}.json")
        tokenisers = processor.tokenisers
        token_engineers = processor.token_engineers
        try:
            tensors = torch.load(this_cache_path / 'tensors' / f'{config_name}.pt', weights_only=False)
        except FileNotFoundError:
            logger.warning("Tensors not found at %s", this_cache_path / 'tensors' / f'{config_name}.pt')
            raise
        return tensors, tokenisers, token_engineers
    else:
        cache_path = Path(__file__).resolve().parents[2] / "cache" / dataset
        with gzip.open(cache_path / 'data.pkl.gz', "rb") as f:
            datasets_pd = pickle.load(f)
        return tokenise_dataset(
            datasets_pd=datasets_pd,
            config=OmegaConf.load(
                Path(__file__).resolve().parents[2] / 'yamls' / 'tokenisers' / f'{config_name}.yaml'),
            tokenizer_path=tokenizer_path,
        )


def get_wp_index() -> dict:
    wp_index = dict()
    for group_number in range(1, 231):
        group = Group(group_number)
        wp_index[group_number] = defaultdict(dict)
        for wp in group.Wyckoff_positions:
            wp.get_site_symmetry()
            site_symm = wp.site_symm
            wp_index[group_number][site_symm][wp.letter] = (wp.multiplicity, wp.get_dof())
    return wp_index