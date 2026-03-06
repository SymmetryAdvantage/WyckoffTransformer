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
from pandarallel import pandarallel
from pyxtal.symmetry import Group, site_symmetry
from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel


# Order is important here, as we can use it to sort the tokens
ServiceToken = Enum('ServiceToken', ['MASK', 'STOP', 'PAD'])
logger = logging.getLogger(__name__)

generation_modes = Enum('GenerationModes', ["SiteSymmetry", "WyckoffLetters", "HarmonicCluster"])

JsonPrimitive = str | int | float | bool | None
JsonContainer = Dict[str, object] | List[object]


class _SerialisedTokeniser(BaseModel):
    kind: str
    payload: Dict[str, object]


class _SerialisedFeatureEngineer(BaseModel):
    name: Optional[str]
    inputs: List[Optional[str]]
    stop_token: object = None
    pad_token: object = None
    mask_token: object = None
    default_value: object = 0
    include_stop: bool = True
    entries: List[Tuple[List[object], object]]


class _WyckoffProcessorState(BaseModel):
    config: Dict[str, object]
    tokenisers: Dict[str, _SerialisedTokeniser]
    token_engineers: Dict[str, _SerialisedFeatureEngineer]

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
        
        def get_transferable_spg_matrix(group: Group) -> np.ndarray:
            """
            Build a translation-aware space-group symmetry matrix.
            
            Necessary to avoid
            https://github.com/MaterSim/PyXtal/issues/330

            pyxtal's Group.get_spg_symmetry_object() currently constructs
            site_symmetry without parse_trans=True, which collapses screw/glide
            information and causes collisions for distinct space groups.
            """
            wp = group.get_wyckoff_position(0)
            ops = wp.get_euclidean_ops() if 143 <= group.number <= 194 else wp.ops
            bravais = group.symbol[0]
            if bravais in ("A", "B", "C", "I"):
                ops = ops[: int(len(ops) / 2)]
            elif bravais == "R":
                ops = ops[: int(len(ops) / 3)]
            elif bravais == "F":
                ops = ops[: int(len(ops) / 4)]

            return site_symmetry(
                ops, group.lattice_type, bravais, group.number, wp_id=0, parse_trans=True
            ).to_matrix_representation().ravel()

        all_spgs_raw = dict()
        for group_number in all_space_groups:
            group = Group(group_number)
            all_spgs_raw[group_number] = np.concatenate(
                [get_transferable_spg_matrix(group),
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

    def get_letter_from_ss_enum_idx(self) -> dict:
        """
        Processes the real-space index of Wyckhoff letters by space group, site symmetry,
        and enumeration into a dict indexed by space group, site symmetry, and
        enumeration TOKEN to make generation a little faster.
        """
        preprocessed_wyckhoffs_cache_path = Path(__file__).resolve().parents[2] / "cache" / "wychoffs_enumerated_by_ss.pkl.gz"
        with gzip.open(preprocessed_wyckhoffs_cache_path, "rb") as f:
            letter_from_ss_enum = pickle.load(f)[1]
        letter_from_ss_enum_idx = defaultdict(dict)
        for space_group, ss_enum_dict in letter_from_ss_enum.items():
            for ss, enum_dict in ss_enum_dict.items():
                letter_from_ss_enum_idx[space_group][ss] = dict()
                for enum, letter in enum_dict.items():
                    if enum in self:
                        letter_from_ss_enum_idx[space_group][ss][self[enum]] = letter
        return letter_from_ss_enum_idx


class FeatureEngineer():
    @staticmethod
    def _lexsort_db(db: Series) -> Series:
        if isinstance(db.index, MultiIndex):
            return db.sort_index()
        return db

    def __init__(self,
            data: Dict[Tuple, int]|Series,
            inputs: Optional[Tuple] = None,
            name: Optional[str] = None,
            stop_token: Optional[int] = None,
            pad_token: Optional[int] = None,
            mask_token: Optional[int] = None,
            default_value = 0,
            include_stop: bool = True):
        if isinstance(data, Series):
            if inputs is not None or name is not None:
                raise ValueError("If data is a DataFrame, inputs and name should be None")
            self.db = data
        else:
            index = MultiIndex.from_tuples(data.keys(), names=inputs)
            self.db = Series(data=data.values(), index=index, name=name)
        # Ensure MultiIndex lookups are lexsorted once at construction time.
        # This avoids repeated runtime warnings and slower indexing paths.
        self.db = self._lexsort_db(self.db)
        self.inputs = self.db.index.names
        self.stop_token = stop_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.default_value = default_value
        self.include_stop = include_stop
        if self.db.dtype == 'O':
            all_feature_shapes = self.db.apply(np.shape)
            unqiue_shapes = all_feature_shapes.unique()
            if len(unqiue_shapes) != 1:
                raise ValueError("Inconsistent shapes")
            self.feature_shape = unqiue_shapes[0]
        # Scalar
        else:
            self.feature_shape = tuple()


    def pad_and_stop(self,
        sequence,
        original_max_len: int,
        **tensor_args) -> torch.Tensor:

        padding = [self.pad_token] * (original_max_len - len(sequence))
        if self.include_stop:
            padding = [self.stop_token] + padding
        padded_sequence = sequence + padding
        if self.feature_shape:
            padded_sequence = np.array(padded_sequence)
        return torch.tensor(padded_sequence, **tensor_args)


    def get_feature_tensor_from_series(
        self,
        record: Series,
        original_max_len: int,
        **tensor_args) -> torch.Tensor:

        indexed_record = record.loc[self.db.index.names]
        logger.debug("Indexed record")
        logger.debug(indexed_record)
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
        return self.pad_and_stop(res, original_max_len, **tensor_args)


    def get_feature_from_augmented_series(
        self,
        record: Series,
        augmented_field_orginal_name: str,
        original_max_len: int,
        **tensor_args) -> torch.Tensor:
        """
        Args:
            record: The record to process
            augmented_field_orginal_name: The original name of the field containing the augmented variants
                e. g. "sites_enumeration"
            original_max_len: The maximum length of the sequence
            **tensor_args: Additional arguments for the torch.tensor
        Returns:
            A list of tensors with the augmented features processed by the engineer
        """
        # We need to unravel the augmented field
        augmented_field = f"{augmented_field_orginal_name}_augmented"
        augmentation_variants = record.at[augmented_field]
        # WARNING(kazeevn): only one structure is supported:
        # The first input is sequence-level, the next two are token-level
        this_db = self.db.xs(record.at[self.db.index.names[0]], level=0)
        assert len(self.db.index.names) == 3
        assert self.db.index.names[2] == augmented_field_orginal_name
        queries = [list(map(tuple, zip(record.at[self.db.index.names[1]], variant))) for variant in augmentation_variants]
        padding_function = partial(
            self.pad_and_stop,
            original_max_len=original_max_len,
            **tensor_args)
        return [padding_function(this_db.loc[query].to_list()) for query in queries]


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
    if len(tensors) == 2:
        if tensors[0].dtype != torch.uint8 or tensors[1].dtype != torch.uint8:
            # Will need to check for overflow
            raise NotImplementedError("Only uint8 tensors are supported")
        # Use max + 1 as radix to avoid collisions:
        # (a, b) -> a * radix + b
        radix = tensors[1].max().type(torch.int64) + 1
        megaindex = tensors[0].type(torch.int64) * radix + tensors[1].type(torch.int64)
        return torch.argsort(megaindex, dim=dim)

    raise NotImplementedError("Only one or two tensors are supported")


class WyckoffProcessor:
    """
    Master object encapsulating tokeniser/token-engineer lifecycle.
    """

    def __init__(
        self,
        config: Optional[DictConfig | Dict[str, object]] = None,
        tokenisers: Optional[Dict[str, TokeniserType]] = None,
        token_engineers: Optional[Dict[str, FeatureEngineer]] = None,
    ):
        if config is None:
            config = {}
        self.config = OmegaConf.create(config)
        self.tokenisers: Dict[str, TokeniserType] = tokenisers or {}
        self.token_engineers: Dict[str, FeatureEngineer] = token_engineers or {}

    @staticmethod
    def _normalise_path(path: Path | str) -> Path:
        path = Path(path)
        if path.suffix == ".json":
            return path
        return path / "wyckoff_processor.json"

    @staticmethod
    def _to_jsonable(value: object) -> JsonPrimitive | JsonContainer:
        if isinstance(value, np.ndarray):
            return {"__ndarray__": value.tolist()}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, tuple):
            return [WyckoffProcessor._to_jsonable(v) for v in value]
        if isinstance(value, list):
            return [WyckoffProcessor._to_jsonable(v) for v in value]
        if isinstance(value, dict):
            return {k: WyckoffProcessor._to_jsonable(v) for k, v in value.items()}
        return value

    @staticmethod
    def _from_jsonable(value: object) -> object:
        if isinstance(value, dict) and "__ndarray__" in value:
            return np.array(value["__ndarray__"])
        if isinstance(value, list):
            return [WyckoffProcessor._from_jsonable(v) for v in value]
        if isinstance(value, dict):
            return {k: WyckoffProcessor._from_jsonable(v) for k, v in value.items()}
        return value

    @staticmethod
    def _restore_hashable(value: object) -> object:
        if isinstance(value, list):
            return tuple(WyckoffProcessor._restore_hashable(v) for v in value)
        if isinstance(value, dict):
            return {k: WyckoffProcessor._restore_hashable(v) for k, v in value.items()}
        return value

    @staticmethod
    def _serialise_tokeniser(tokeniser: TokeniserType) -> _SerialisedTokeniser:
        if isinstance(tokeniser, EnumeratingTokeniser):
            mapping = [
                [
                    WyckoffProcessor._to_jsonable(k),
                    WyckoffProcessor._to_jsonable(v),
                ]
                for k, v in tokeniser.items()
            ]
            return _SerialisedTokeniser(
                kind="enumerating",
                payload={
                    "mapping": mapping,
                    "stop_token": WyckoffProcessor._to_jsonable(tokeniser.stop_token),
                    "pad_token": WyckoffProcessor._to_jsonable(tokeniser.pad_token),
                    "mask_token": WyckoffProcessor._to_jsonable(tokeniser.mask_token),
                    "include_stop": tokeniser.include_stop,
                },
            )
        if isinstance(tokeniser, SpaceGroupEncoder):
            return _SerialisedTokeniser(
                kind="space_group",
                payload={
                    "mapping": [
                        [
                            WyckoffProcessor._to_jsonable(k),
                            WyckoffProcessor._to_jsonable(list(v)),
                        ]
                        for k, v in tokeniser.items()
                    ],
                },
            )
        if isinstance(tokeniser, PassThroughTokeniser):
            return _SerialisedTokeniser(
                kind="pass_through",
                payload={
                    "values_count": WyckoffProcessor._to_jsonable(tokeniser.values_count),
                    "stop_token": WyckoffProcessor._to_jsonable(tokeniser.stop_token),
                    "pad_token": WyckoffProcessor._to_jsonable(tokeniser.pad_token),
                    "mask_token": WyckoffProcessor._to_jsonable(tokeniser.mask_token),
                },
            )
        raise TypeError(f"Unsupported tokeniser type: {type(tokeniser)}")

    @staticmethod
    def _deserialise_tokeniser(serialised: _SerialisedTokeniser) -> TokeniserType:
        payload = serialised.payload
        if serialised.kind == "enumerating":
            return EnumeratingTokeniser(
                {WyckoffProcessor._restore_hashable(k): v for k, v in payload["mapping"]},
                stop_token=payload["stop_token"],
                pad_token=payload["pad_token"],
                mask_token=payload["mask_token"],
                include_stop=payload["include_stop"],
            )
        if serialised.kind == "space_group":
            return SpaceGroupEncoder({group_number: tuple(spg) for group_number, spg in payload["mapping"]})
        if serialised.kind == "pass_through":
            return PassThroughTokeniser(
                values_count=payload["values_count"],
                stop_token=payload["stop_token"],
                pad_token=payload["pad_token"],
                mask_token=payload["mask_token"],
            )
        raise ValueError(f"Unknown tokeniser kind: {serialised.kind}")

    @staticmethod
    def _serialise_feature_engineer(engineer: FeatureEngineer) -> _SerialisedFeatureEngineer:
        entries = []
        for index, value in engineer.db.items():
            if isinstance(index, tuple):
                index_items = [WyckoffProcessor._to_jsonable(v) for v in index]
            else:
                index_items = [WyckoffProcessor._to_jsonable(index)]
            entries.append((index_items, WyckoffProcessor._to_jsonable(value)))
        return _SerialisedFeatureEngineer(
            name=engineer.db.name,
            inputs=list(engineer.db.index.names),
            stop_token=WyckoffProcessor._to_jsonable(engineer.stop_token),
            pad_token=WyckoffProcessor._to_jsonable(engineer.pad_token),
            mask_token=WyckoffProcessor._to_jsonable(engineer.mask_token),
            default_value=WyckoffProcessor._to_jsonable(engineer.default_value),
            include_stop=engineer.include_stop,
            entries=entries,
        )

    @staticmethod
    def _deserialise_feature_engineer(serialised: _SerialisedFeatureEngineer) -> FeatureEngineer:
        data = {
            tuple(WyckoffProcessor._restore_hashable(v) for v in index): WyckoffProcessor._from_jsonable(value)
            for index, value in serialised.entries
        }
        return FeatureEngineer(
            data=data,
            inputs=tuple(serialised.inputs),
            name=serialised.name,
            stop_token=WyckoffProcessor._from_jsonable(serialised.stop_token),
            pad_token=WyckoffProcessor._from_jsonable(serialised.pad_token),
            mask_token=WyckoffProcessor._from_jsonable(serialised.mask_token),
            default_value=WyckoffProcessor._from_jsonable(serialised.default_value),
            include_stop=serialised.include_stop,
        )

    @classmethod
    def from_config(
        cls,
        config: DictConfig | Dict[str, object],
        tokenizer_path: Optional[Path | str] = None,
    ):
        if tokenizer_path is not None:
            return cls.from_pretrained(tokenizer_path)
        return cls(config=config)

    def save_pretrained(self, path: Path | str) -> Path:
        destination = self._normalise_path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        config_payload = OmegaConf.to_container(self.config, resolve=True)
        state = _WyckoffProcessorState(
            config=config_payload,
            tokenisers={
                key: self._serialise_tokeniser(tokeniser)
                for key, tokeniser in self.tokenisers.items()
            },
            token_engineers={
                key: self._serialise_feature_engineer(engineer)
                for key, engineer in self.token_engineers.items()
            },
        )
        destination.write_text(state.model_dump_json(indent=2), encoding="ascii")
        return destination

    @classmethod
    def from_pretrained(cls, path: Path | str):
        source = cls._normalise_path(path)
        state = _WyckoffProcessorState.model_validate_json(source.read_text(encoding="ascii"))
        return cls(
            config=OmegaConf.create(state.config),
            tokenisers={
                key: cls._deserialise_tokeniser(tokeniser)
                for key, tokeniser in state.tokenisers.items()
            },
            token_engineers={
                key: cls._deserialise_feature_engineer(engineer)
                for key, engineer in state.token_engineers.items()
            },
        )

    def tokenise_dataset(
        self,
        datasets_pd: Dict[str, DataFrame],
        tokenizer_path: Optional[Path | str] = None,
        n_jobs: Optional[int] = None,
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor | List[List[torch.Tensor]]]], Dict[str, TokeniserType], Dict[str, FeatureEngineer]]:
        config = self.config
        dtype = getattr(torch, config.dtype)
        include_stop = config.get("include_stop", True)
        if n_jobs is not None:
            pandarallel.initialize(nb_workers=n_jobs)
        else:
            # Preserve the NB_PHYSICAL_CORES default
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
            preloaded_processor = WyckoffProcessor.from_pretrained(tokenizer_path)
            tokenisers = preloaded_processor.tokenisers
        raw_engineers = {}
        token_engineers = {}
        if "engineered" in config.token_fields:
            for engineered_field_name, engineered_field_definiton in config.token_fields.engineered.items():
                if engineered_field_definiton.type != "map":
                    raise ValueError("Only map engineered_field fields are supported")
                if len(engineered_field_definiton.inputs) != 3:
                    raise NotImplementedError("Only 3 inputs are supported")
                with gzip.open(Path(__file__).resolve().parents[2] / "cache" / "engineers" / f"{engineered_field_name}.pkl.gz", "rb") as f:
                    raw_engineer = pickle.load(f)
                raw_engineers[engineered_field_name] = raw_engineer
                # Now we need to convert the token values to token indices
                # And adjust include_stop
                token_engineers[engineered_field_name] = tokenise_engineer(raw_engineer, tokenisers)
                raw_engineers[engineered_field_name].include_stop = token_engineers[engineered_field_name].include_stop
                # The values haven't changed, only the keys, so we can reuse the stop, pad, and mask tokens
                if token_engineers[engineered_field_name].db.dtype == 'O':
                    try:
                        value_counts = token_engineers[engineered_field_name].db.nunique()
                    except TypeError: # unhashable type: 'numpy.ndarray', etc.
                        value_counts = len(token_engineers[engineered_field_name].db)
                else:
                    if token_engineers[engineered_field_name].db.min() != 0:
                        logger.warning("The minimum value in the engineered field %s is not 0", engineered_field_name)
                        # We still set the count to max, potentially inluding some never used values
                        # +1 for token #0
                    value_counts = token_engineers[engineered_field_name].db.max() + 1

                tokenisers[engineered_field_name] = PassThroughTokeniser(
                    values_count=value_counts + 3, # For service tokens
                    stop_token=raw_engineer.stop_token,
                    pad_token=raw_engineer.pad_token,
                    mask_token=raw_engineer.mask_token)

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
                for field, field_config in config.token_fields.engineered.items():
                    logger.debug("Processing engineered field %s", field)
                    if "dtype" in field_config:
                        field_dtype = getattr(torch, field_config.dtype)
                    else:
                        field_dtype = dtype
                    compute_feature_function = partial(
                        raw_engineers[field].get_feature_tensor_from_series,
                        original_max_len=original_max_len,
                        dtype=field_dtype)
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
                    try:
                        tensors[dataset_name][field] = torch.Tensor(dataset[field].array)
                    except KeyError as e:
                        logger.warning("Field %s not found in the dataset", field)
                        logger.warning(e)


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

            for field, field_config in config.token_fields.get("augmented_engineered", {}).items():
                augmented_field = f"{field}_augmented"
                if "dtype" in config.token_fields.engineered[field]:
                    field_dtype = getattr(torch, config.token_fields.engineered[field].dtype)
                else:
                    field_dtype = dtype
                tensors[dataset_name][augmented_field] = dataset.parallel_apply(
                    partial(
                        raw_engineers[field].get_feature_from_augmented_series,
                        augmented_field_orginal_name=field_config.augmented_input,
                        original_max_len=original_max_len,
                        dtype=field_dtype), axis=1).to_list()

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

        self.tokenisers = tokenisers
        self.token_engineers = token_engineers
        return tensors, tokenisers, token_engineers

    def tensor_to_pyxtal(
        self,
        space_group_tensor: torch.Tensor,
        wp_tensor: torch.Tensor,
        cascade_order: Tuple[str, ...],
        letter_from_ss_enum_idx,
        ss_from_letter,
        wp_index,
        enforced_min_elements: Optional[int] = None,
        enforced_max_elements: Optional[int] = None,
    ) -> Optional[dict]:
        if not self.tokenisers:
            raise ValueError("Tokenisers are not initialised")
        ss_pyxtal_cascde_order = ("elements", "site_symmetries", "sites_enumeration")
        letters_pyxtal_cascade_order = ("elements", "wyckoff_letters")
        harmonic_pyxtal_cascade_order = ("elements", "site_symmetries", "harmonic_cluster")
        if set(cascade_order) == set(ss_pyxtal_cascde_order):
            pyxtal_cascade_order = ss_pyxtal_cascde_order
            mode = generation_modes.SiteSymmetry
        elif set(cascade_order) == set(letters_pyxtal_cascade_order):
            pyxtal_cascade_order = letters_pyxtal_cascade_order
            mode = generation_modes.WyckoffLetters
        elif set(cascade_order) == set(harmonic_pyxtal_cascade_order):
            pyxtal_cascade_order = harmonic_pyxtal_cascade_order
            mode = generation_modes.HarmonicCluster
        else:
            raise NotImplementedError("Unsupported cascade")

        cascade_permutation = [cascade_order.index(field) for field in pyxtal_cascade_order]
        cononical_wp_tensor = wp_tensor[:, cascade_permutation]

        stop_tokens = torch.tensor([self.tokenisers[field].stop_token for field in pyxtal_cascade_order], device=wp_tensor.device)
        mask_tokens = torch.tensor([self.tokenisers[field].mask_token for field in pyxtal_cascade_order], device=wp_tensor.device)
        pad_tokens = torch.tensor([self.tokenisers[field].pad_token for field in pyxtal_cascade_order], device=wp_tensor.device)

        if space_group_tensor.size() == (1,):
            space_group_input = space_group_tensor.item()
        else:
            space_group_input = space_group_tensor.numpy()
        space_group_real = self.tokenisers["spacegroup_number"].to_token[space_group_input]
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
            if mode == generation_modes.SiteSymmetry:
                element_idx, ss_idx, enum_idx = this_token_cascade.tolist()
                ss = self.tokenisers["site_symmetries"].to_token[ss_idx]
                try:
                    wp_letter = letter_from_ss_enum_idx[space_group_real][ss][enum_idx]
                except KeyError:
                    logger.info("Invalid combination: space group %i, site symmetry %s, enum token %i", space_group_real,
                                ss, enum_idx)
                    return
            elif mode == generation_modes.WyckoffLetters:
                element_idx, wp_letter_idx = this_token_cascade.tolist()
                wp_letter = self.tokenisers["wyckoff_letters"].to_token[wp_letter_idx]
                try:
                    ss = ss_from_letter[space_group_real][wp_letter]
                except KeyError:
                    logger.info("Invalid combination: space group %i, wp letter %s", space_group_real, wp_letter)
                    return
            elif mode == generation_modes.HarmonicCluster:
                element_idx, ss_idx, cluster_idx = this_token_cascade.tolist()
                ss = self.tokenisers["site_symmetries"].to_token[ss_idx]
                try:
                    # WARNING tuple might fail for some SG encodings
                    enum = self.token_engineers["sites_enumeration"].db.loc[tuple(space_group_input), ss_idx, cluster_idx]
                except KeyError:
                    logger.info("Invalid combination: space group %i, site symmetry %s, cluster token %i", space_group_real,
                                ss, cluster_idx)
                    return
                wp_letter = letter_from_ss_enum_idx[space_group_real][ss][enum]
            else:
                raise NotImplementedError("Unsupported cascade")
            try:
                our_site = available_sites[ss][wp_letter]
            except KeyError:
                logger.info("Repeated special WP: %i, %s, %s", space_group_real, ss, wp_letter)
                return
            element = self.tokenisers["elements"].to_token[element_idx]
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