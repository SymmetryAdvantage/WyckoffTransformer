from typing import Any, Dict, Iterable, NamedTuple, Set, FrozenSet, Optional, List, Tuple
import json
import logging
import shutil
from itertools import chain
from operator import attrgetter, itemgetter
from collections import defaultdict, UserDict
from enum import Enum
from pathlib import Path
import numpy as np
from pandas import DataFrame
import torch
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors_file
from pandarallel import pandarallel
from pyxtal.symmetry import Group
from omegaconf import OmegaConf, DictConfig

from wyckoff_transformer.dataset_cache import load_cache
from wyckoff_transformer.dataset_manifest import warn_if_obsolete_dataset
from wyckoff_transformer.paths import cache_root
from wyckoff_transformer.wyckoff_processor import (
    ENGINEERS_DIR,
    MODEL_ENGINEERS_DIRNAME,
    FeatureEngineer,
    WyckoffProcessor,
    argsort_multiple,
)

#: Key a tokeniser config sets to say it must not be used for new work.
#:
#: Its value is the reason, in prose, and the reason is what a caller is shown --
#: a bare flag would leave whoever hits it guessing whether the config is merely
#: old or actively wrong.  Nothing is deleted when a config becomes obsolete:
#: models were trained with it and have to keep loading, so inference warns and
#: only training refuses.
OBSOLETE_KEY = "obsolete"


class ObsoleteTokeniserError(ValueError):
    """Training was asked for a tokeniser configuration that must not be used."""


def obsolete_reason(config) -> Optional[str]:
    """Why this tokeniser config is obsolete, or None if it is not."""
    if config is None:
        return None
    getter = getattr(config, "get", None)
    if getter is None:
        return None
    reason = config.get(OBSOLETE_KEY)
    if reason is None or reason is False:
        return None
    return str(reason)


def warn_if_obsolete(config, context: str = "") -> Optional[str]:
    """Log why an obsolete tokeniser is obsolete, and carry on.

    For the paths that have to keep working: loading a checkpoint for inference,
    re-scoring an old run, reading a cache that was tokenised with it.  Refusing
    there would make previously trained models unloadable, which is a worse
    outcome than a noisy one.
    """
    reason = obsolete_reason(config)
    if reason is None:
        return None
    name = config.get("name", "<unnamed>")
    logger.warning(
        "Tokeniser configuration %r is obsolete%s: %s", name,
        f" ({context})" if context else "", reason)
    return reason


def refuse_if_obsolete(config, context: str = "") -> None:
    """Raise rather than start training on an obsolete tokeniser configuration."""
    reason = obsolete_reason(config)
    if reason is None:
        return
    name = config.get("name", "<unnamed>")
    raise ObsoleteTokeniserError(
        f"Tokeniser configuration {name!r} is obsolete and must not be trained on"
        f"{f' ({context})' if context else ''}: {reason}"
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


def save_package_data(model_dir: Path) -> List[Path]:
    """Copy the Wyckoff mappings and every engineer into a model directory.

    A model is only meaningful against the exact tables it was trained with. With its
    own copies -- the mappings next to its wyckoff_processor.json, the engineers and
    frozen tables under ``engineers/`` -- it keeps working whatever the installed
    package holds later. Returns the files written.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    written = [Path(shutil.copy2(_PACKAGE_MAPPINGS_PATH, model_dir / WYCKOFF_MAPPINGS_FILENAME))]
    engineers_dir = model_dir / MODEL_ENGINEERS_DIRNAME
    engineers_dir.mkdir(exist_ok=True)
    for source in sorted(ENGINEERS_DIR.glob("*.json")):
        written.append(Path(shutil.copy2(source, engineers_dir / source.name)))
    return written


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

    def __missing__(self, key):
        # A saved tokeniser holds enum tokens -- pymatgen's Element -- by their value,
        # because that is what they serialise to; the data still holds the enum members.
        if isinstance(key, Enum) and key.value in self:
            return self[key.value]
        raise KeyError(key)

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
TENSOR_CACHE_SUFFIX = ".safetensors"
_TENSOR_CACHE_FORMAT_KEY = "wyckoff_transformer_tensor_cache_format"
_TENSOR_CACHE_FORMAT_VERSION = "2"
_TENSOR_CACHE_STRUCTURE_KEY = "wyckoff_transformer_tensor_cache_structure"


def _store_cache_tensor(flat_tensors: dict[str, torch.Tensor], tensor: torch.Tensor) -> str:
    tensor_key = f"tensor_{len(flat_tensors)}"
    flat_tensors[tensor_key] = tensor.detach().cpu().contiguous()
    return tensor_key


def _serialise_tensor_list(node: list[torch.Tensor], flat_tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    if not node:
        return {"type": "tensor_list", "storage": "empty"}

    tensors = [tensor.detach().cpu().contiguous() for tensor in node]
    dims = {tensor.dim() for tensor in tensors}
    if len(dims) != 1:
        raise TypeError("Tensor cache lists must contain tensors with the same rank")

    ndim = dims.pop()
    shapes = [tuple(tensor.shape) for tensor in tensors]
    if len(set(shapes)) == 1:
        data_key = _store_cache_tensor(flat_tensors, torch.stack(tensors, dim=0))
        return {"type": "tensor_list", "storage": "stack", "data": data_key}

    if ndim == 0:
        raise TypeError("Tensor cache cannot serialise mixed scalar tensor shapes")

    tail_shapes = {tuple(tensor.shape[1:]) for tensor in tensors}
    if len(tail_shapes) != 1:
        raise TypeError("Tensor cache lists must contain tensors with matching trailing dimensions")

    lengths = torch.tensor([tensor.shape[0] for tensor in tensors], dtype=torch.int64)
    data_key = _store_cache_tensor(flat_tensors, torch.cat(tensors, dim=0))
    lengths_key = _store_cache_tensor(flat_tensors, lengths)
    return {
        "type": "tensor_list",
        "storage": "concat",
        "data": data_key,
        "lengths": lengths_key,
    }


def _serialise_tensor_tree(node: Any, flat_tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    if isinstance(node, torch.Tensor):
        return {"type": "tensor", "key": _store_cache_tensor(flat_tensors, node)}

    if isinstance(node, dict):
        if not all(isinstance(key, str) for key in node):
            raise TypeError("Tensor cache dictionaries must use string keys")
        return {
            "type": "dict",
            "items": {
                key: _serialise_tensor_tree(value, flat_tensors)
                for key, value in node.items()
            },
        }

    if isinstance(node, list):
        if not node:
            return {"type": "list", "storage": "empty"}

        if all(isinstance(item, torch.Tensor) for item in node):
            return _serialise_tensor_list(node, flat_tensors)

        if all(isinstance(item, list) for item in node):
            lengths_key = _store_cache_tensor(
                flat_tensors,
                torch.tensor([len(item) for item in node], dtype=torch.int64),
            )
            flattened_items = list(chain.from_iterable(node))
            return {
                "type": "list",
                "storage": "nested",
                "lengths": lengths_key,
                "items": _serialise_tensor_tree(flattened_items, flat_tensors),
            }

        return {
            "type": "list",
            "storage": "items",
            "items": [_serialise_tensor_tree(item, flat_tensors) for item in node],
        }

    raise TypeError(f"Unsupported tensor cache value type: {type(node)!r}")


def _deserialise_tensor_tree(structure: dict[str, Any], flat_tensors: dict[str, torch.Tensor]) -> Any:
    node_type = structure["type"]

    if node_type == "tensor":
        return flat_tensors[structure["key"]]

    if node_type == "dict":
        return {
            key: _deserialise_tensor_tree(value, flat_tensors)
            for key, value in structure["items"].items()
        }

    if node_type == "tensor_list":
        storage = structure["storage"]
        if storage == "empty":
            return []

        data = flat_tensors[structure["data"]]
        if storage == "stack":
            return list(data.unbind(dim=0))

        if storage == "concat":
            lengths = flat_tensors[structure["lengths"]].tolist()
            return list(torch.split(data, lengths, dim=0))

        raise ValueError(f"Unsupported tensor cache tensor-list storage type: {storage}")

    if node_type == "list":
        storage = structure["storage"]
        if storage == "empty":
            return []

        if storage == "nested":
            lengths = flat_tensors[structure["lengths"]].tolist()
            flat_items = _deserialise_tensor_tree(structure["items"], flat_tensors)
            output = []
            offset = 0
            for length in lengths:
                output.append(flat_items[offset:offset + length])
                offset += length
            return output

        if storage == "items":
            return [
                _deserialise_tensor_tree(item, flat_tensors)
                for item in structure["items"]
            ]

        raise ValueError(f"Unsupported tensor cache list storage type: {storage}")

    raise ValueError(f"Unsupported tensor cache node type: {node_type}")


def save_tensor_cache(tensors: dict[str, Any], path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    flat_tensors: dict[str, torch.Tensor] = {}
    structure = _serialise_tensor_tree(tensors, flat_tensors)
    save_safetensors_file(
        flat_tensors,
        str(destination),
        metadata={
            _TENSOR_CACHE_FORMAT_KEY: _TENSOR_CACHE_FORMAT_VERSION,
            _TENSOR_CACHE_STRUCTURE_KEY: json.dumps(structure, separators=(",", ":")),
        },
    )
    return destination


def load_tensor_cache(path: Path | str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    source = Path(path)
    if source.suffix != TENSOR_CACHE_SUFFIX:
        raise ValueError(f"Unsupported tensor cache extension: {source.suffix}")

    device = str(map_location)
    with safe_open(str(source), framework="pt", device=device) as handle:
        metadata = handle.metadata()
        if metadata is None:
            raise ValueError(f"Tensor cache {source} is missing metadata")
        if metadata.get(_TENSOR_CACHE_FORMAT_KEY) != _TENSOR_CACHE_FORMAT_VERSION:
            raise ValueError(f"Tensor cache {source} has unsupported metadata format")
        try:
            structure = json.loads(metadata[_TENSOR_CACHE_STRUCTURE_KEY])
        except KeyError as exc:
            raise ValueError(f"Tensor cache {source} is missing structure metadata") from exc
        flat_tensors = {key: handle.get_tensor(key) for key in handle.keys()}

    return _deserialise_tensor_tree(structure, flat_tensors)


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
                     config: Optional[DictConfig],
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
            Ignored when tokenizer_path is given: the saved processor carries its own.
        tokenizer_path: The path to the tokenisers. If None, new tokenisers are created.
            If given, the saved tokenisers are used, with the engineers saved next to them.
        n_jobs: The number of jobs to use for parallel processing. If None, it uses the default number of physical cores.
    Returns:
        A tuple with the tokenised tensors and the tokenisers.
            The tokenised tensors are a dict with the dataset name as key and a dict with the field name as key and the tokenised tensor as value.
            The tokenisers are a dict with the field name as key and the tokeniser as value.
    """
    processor = WyckoffProcessor.from_config(config, tokenizer_path=tokenizer_path)
    # tokenizer_path again: without it the processor would build fresh tokenisers from
    # this data and read the package's engineers, whatever it was loaded from
    return processor.tokenise_dataset(
        datasets_pd=datasets_pd, tokenizer_path=tokenizer_path, n_jobs=n_jobs)


def load_tensors_and_tokenisers(
    dataset: str,
    config_name: str,
    use_cached_tensors: bool = True,
    cache_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None):

    if cache_path is None:
        cache_path = cache_root()
    this_cache_path = cache_path / dataset
    warn_if_obsolete_dataset(dataset, "loading its tensors")
    if use_cached_tensors:
        processor = WyckoffProcessor.from_pretrained(this_cache_path / "tokenisers" / f"{config_name}.json")
        tokenisers = processor.tokenisers
        token_engineers = processor.token_engineers
        safetensors_path = this_cache_path / 'tensors' / f'{config_name}{TENSOR_CACHE_SUFFIX}'
        try:
            tensors = load_tensor_cache(safetensors_path)
        except FileNotFoundError:
            logger.warning("Tensors not found at %s", safetensors_path)
            raise
        return tensors, tokenisers, token_engineers
    else:
        datasets_pd = load_cache(this_cache_path)
        # A saved processor carries its own config; the repository's YAML, which may
        # have changed or gone since, is only for building new tokenisers.
        config = None if tokenizer_path is not None else OmegaConf.load(
            Path(__file__).resolve().parents[2] / 'yamls' / 'tokenisers' / f'{config_name}.yaml')
        return tokenise_dataset(
            datasets_pd=datasets_pd,
            config=config,
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