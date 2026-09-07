"""Shared conversion of Wyckoff records into scalar-prediction tensors."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch

logger = logging.getLogger(__name__)


def filter_supported_tokens(df: pd.DataFrame, trainer) -> Tuple[pd.DataFrame, List]:
    """Split records into vocabulary-supported rows and rows a model cannot read."""
    token_config = trainer.tokeniser_config
    pure_fields = list(token_config.token_fields.pure_categorical)
    augmented_fields = list(token_config.get("augmented_token_fields", []))
    space_group_fields = list(token_config.sequence_fields.get("space_group", []))

    supported_indices = []
    dropped = []
    for idx, row in df.iterrows():
        unsupported = False
        for field in pure_fields:
            sequence = row[field]
            if sequence is None:
                unsupported = True
                break
            if any(token not in trainer.tokenisers[field] for token in sequence):
                logger.warning(
                    "Dropping structure %s: field '%s' contains tokens outside the vocabulary.",
                    idx,
                    field,
                )
                unsupported = True
                break
        if unsupported:
            dropped.append(idx)
            continue
        for field in space_group_fields:
            space_group = row[field]
            if space_group not in trainer.tokenisers[field]:
                logger.warning(
                    "Dropping structure %s: space group '%s' not in the vocabulary.",
                    idx,
                    space_group,
                )
                unsupported = True
                break
        if unsupported:
            dropped.append(idx)
            continue
        for field in augmented_fields:
            variants = row.get(f"{field}_augmented", [])
            for variant in variants:
                if any(token not in trainer.tokenisers[field] for token in variant):
                    logger.warning(
                        "Dropping structure %s: augmented field '%s' contains tokens outside "
                        "the vocabulary.",
                        idx,
                        field,
                    )
                    unsupported = True
                    break
            if unsupported:
                break
        if unsupported:
            dropped.append(idx)
        else:
            supported_indices.append(idx)
    if dropped:
        logger.warning("Dropped %d structures with unsupported symmetry tokens.", len(dropped))
    if not supported_indices:
        raise ValueError("All structures were dropped due to unsupported tokens.")
    return df.loc[supported_indices], dropped


def _get_dtype(dtype_name: str) -> torch.dtype:
    try:
        return getattr(torch, dtype_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported dtype '{dtype_name}' in tokeniser config.") from exc


def build_tokenised_prediction_tensors(
    df: pd.DataFrame,
    trainer,
) -> Dict[str, object]:
    """Tokenise symmetry records into the input shape ``predict_scalars`` expects."""
    if trainer.tokeniser_config is None:
        raise ValueError("Trainer does not expose tokeniser configuration.")
    token_config = trainer.tokeniser_config
    dtype = _get_dtype(token_config.dtype)
    pure_fields: List[str] = list(token_config.token_fields.pure_categorical)
    max_len = int(df[pure_fields[0]].map(len).max())
    data_dict: Dict[str, object] = {}

    for field in pure_fields:
        sequences = [
            trainer.tokenisers[field].tokenise_sequence(
                seq, original_max_len=max_len, dtype=dtype)
            for seq in df[field]
        ]
        data_dict[field] = torch.stack(sequences)

    engineered_fields = token_config.token_fields.get("engineered", {})
    for field_name, field_cfg in engineered_fields.items():
        engineer = trainer.token_engineers[field_name]
        if hasattr(field_cfg, "get"):
            dtype_name = field_cfg.get("dtype", token_config.dtype)
        else:
            dtype_name = token_config.dtype
        field_dtype = _get_dtype(dtype_name)

        def compute_engineered_tensor(row: pd.Series) -> torch.Tensor:
            try:
                return engineer.get_feature_tensor_from_series(
                    row, original_max_len=max_len, dtype=field_dtype)
            except KeyError:
                fallback_values = row.get(engineer.db.name)
                if fallback_values is None:
                    sequence_length = len(row[pure_fields[0]]) if pure_fields else 0
                    fallback_values = [engineer.default_value] * sequence_length
                else:
                    fallback_values = list(fallback_values)
                return engineer.pad_and_stop(
                    fallback_values,
                    original_max_len=max_len,
                    dtype=field_dtype)

        tensors = df.apply(compute_engineered_tensor, axis=1).to_list()
        data_dict[field_name] = torch.stack(tensors)

    space_group_fields: Iterable[str] = token_config.sequence_fields.get("space_group", [])
    for field in space_group_fields:
        data_dict[field] = trainer.tokenisers[field].encode_spacegroups(
            df[field],
            dtype=dtype,
        )

    if "counters" in token_config.sequence_fields:
        for field, tokeniser_field in token_config.sequence_fields.counters.items():
            tokenised_values = []
            counts = []
            for composition in df[field]:
                if not composition:
                    tokenised_values.append(torch.empty(0, dtype=dtype))
                    counts.append(torch.empty(0, dtype=dtype))
                    continue
                value_tokens = [
                    trainer.tokenisers[tokeniser_field].tokenise_single(element, dtype=dtype)
                    for element in composition.keys()
                ]
                tokenised_values.append(torch.stack(value_tokens))
                counts.append(torch.tensor(tuple(composition.values()), dtype=dtype))
            data_dict[f"{field}_tokens"] = tokenised_values
            data_dict[f"{field}_counts"] = counts

    if "augmented_token_fields" in token_config:
        for field in token_config.augmented_token_fields:
            augmented_column = f"{field}_augmented"
            if augmented_column in df.columns:
                augmented_source = df[augmented_column].to_list()
            else:
                augmented_source = [[] for _ in range(len(df))]
            augmented_variants: List[List[torch.Tensor]] = []
            for idx, variants in enumerate(augmented_source):
                use_variants = variants if variants else [df[field].iloc[idx]]
                augmented_variants.append([
                    trainer.tokenisers[field].tokenise_sequence(
                        variant,
                        original_max_len=max_len,
                        dtype=dtype,
                    )
                    for variant in use_variants
                ])
            data_dict[augmented_column] = augmented_variants

    length_dtype = _get_dtype(token_config.get("pure_sequence_length_dtype", token_config.dtype))
    data_dict["pure_sequence_length"] = torch.tensor(
        df[pure_fields[0]].map(len).to_list(),
        dtype=length_dtype,
    )

    start_field = trainer.start_name
    if start_field not in data_dict:
        if start_field in df.columns:
            data_dict[start_field] = trainer.tokenisers[start_field].encode_spacegroups(
                df[start_field],
                dtype=dtype,
            )
        else:
            raise ValueError(f"Start field '{start_field}' is missing from the tokenised data.")
    return data_dict
