#!/usr/bin/env python3
"""Slice a cached dataset to retain only structures with energy_above_hull <= cutoff.

This creates a new dataset cache directory with sliced .safetensors, Wyckoff
records and copied tokenisers, preserving the exact vocabulary, site order, and token
mappings of the source dataset in ~1 minute without re-running pyxtal or tokenisation.
"""
import argparse
import json
import logging
import shutil
import time
from pathlib import Path
import torch

from wyckoff_transformer.dataset_cache import (
    cache_exists, load_cache, provenance, save_cache)
from wyckoff_transformer.dataset_manifest import (
    ManifestNotFound, load_manifest, refuse_if_obsolete_dataset)
from wyckoff_transformer.paths import cache_root
from wyckoff_transformer.tokenization import load_tensor_cache, save_tensor_cache

logger = logging.getLogger("slice_dataset_by_ehull")


def slice_safetensors(
    source_path: Path,
    target_path: Path,
    ehull_cutoff: float,
) -> dict[str, torch.Tensor]:
    """Load safetensors, slice every field by energy_above_hull <= cutoff, and save."""
    logger.info("Loading tensors from %s ...", source_path)
    tensors = load_tensor_cache(source_path)
    filtered_tensors = {}
    masks = {}

    for split, split_dict in tensors.items():
        if "energy_above_hull" not in split_dict:
            raise KeyError(f"Split '{split}' has no 'energy_above_hull' tensor")
        
        ehull = split_dict["energy_above_hull"]
        mask = ehull <= ehull_cutoff
        masks[split] = mask
        mask_list = mask.tolist()
        
        filtered_split = {}
        for key, val in split_dict.items():
            if isinstance(val, torch.Tensor):
                filtered_split[key] = val[mask]
            elif isinstance(val, list):
                # e.g. sites_enumeration_augmented, composition_tokens, composition_counts
                filtered_split[key] = [item for item, m in zip(val, mask_list) if m]
            else:
                raise TypeError(f"Unexpected type {type(val)} for key '{key}' in split '{split}'")
        
        filtered_tensors[split] = filtered_split
        logger.info(
            "  %s: %d -> %d rows (%.2f%%)",
            split,
            len(mask_list),
            mask.sum().item(),
            100.0 * mask.sum().item() / len(mask_list),
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving sliced tensors to %s ...", target_path)
    save_tensor_cache(filtered_tensors, target_path)
    logger.info("Saved %s (%.1f MB)", target_path.name, target_path.stat().st_size / 1e6)
    return masks


def slice_dataframe_cache(
    source_path: Path,
    target_path: Path,
    ehull_cutoff: float,
) -> dict:
    """Load the Wyckoff records, filter each split DataFrame, and save."""
    logger.info("Loading DataFrame cache from %s ...", source_path)
    data_pd = load_cache(source_path)

    filtered_pd = {}
    for split, df in data_pd.items():
        if "energy_above_hull" not in df.columns:
            raise KeyError(f"DataFrame split '{split}' has no 'energy_above_hull' column")
        filtered_df = df[df["energy_above_hull"] <= ehull_cutoff].copy()
        filtered_pd[split] = filtered_df
        logger.info(
            "  DataFrame %s: %d -> %d rows",
            split,
            len(df),
            len(filtered_df),
        )

    # A slice only drops rows, so every column keeps the meaning the source's manifest
    # gives it; recorded so the slice's own manifest can be checked against it.
    source_name = Path(source_path).name
    present = set().union(*(df.columns for df in filtered_pd.values()))
    try:
        declared = load_manifest(source_name).fields_record()
    except ManifestNotFound:
        declared = None  # only reachable with --allow-obsolete-dataset
    fields = None if declared is None else {
        name: record for name, record in declared.items() if name in present}
    logger.info("Saving filtered DataFrame cache to %s ...", target_path)
    save_cache(filtered_pd, target_path, provenance(
        "slice_dataset_by_ehull", sliced_from=source_name,
        ehull_cutoff=ehull_cutoff, manifest=Path(target_path).name, fields=fields))
    return filtered_pd


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source-dataset", type=str, default="lemat_bulk_fmax1_stress",
                        help="Name of source dataset directory under cache/")
    parser.add_argument("--target-dataset", type=str, default="lemat_bulk_fmax1_stress_ehull01",
                        help="Name of target dataset directory under cache/")
    parser.add_argument("--tokeniser-name", type=str, default="lemat_bulk_fmax1_sg_multiplicity",
                        help="Name of tokeniser (stem of .safetensors and .json)")
    parser.add_argument("--ehull-cutoff", type=float, default=0.1,
                        help="Upper bound for energy_above_hull (inclusive, in eV/atom)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Cache root directory (default: cache_root())")
    parser.add_argument("--allow-obsolete-dataset", action="store_true",
                        help="Slice a source dataset yamls/datasets/ marks obsolete.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    refuse_if_obsolete_dataset(args.source_dataset, "slicing",
                               allow=args.allow_obsolete_dataset)
    try:
        load_manifest(args.target_dataset)
    except ManifestNotFound:
        logger.warning(
            "%s has no manifest in yamls/datasets/, so nothing will train on the slice "
            "until one is added -- with 'parent: %s'.", args.target_dataset,
            args.source_dataset)

    root = args.cache_dir if args.cache_dir is not None else cache_root()
    source_dir = root / args.source_dataset
    target_dir = root / args.target_dataset

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset cache not found at {source_dir}")

    t0 = time.time()
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Safetensors
    source_safetensors = source_dir / "tensors" / f"{args.tokeniser_name}.safetensors"
    target_safetensors = target_dir / "tensors" / f"{args.tokeniser_name}.safetensors"
    if not source_safetensors.exists():
        raise FileNotFoundError(f"Source safetensors not found at {source_safetensors}")
    masks = slice_safetensors(source_safetensors, target_safetensors, args.ehull_cutoff)

    # 2. Tokeniser JSON
    source_tokeniser_json = source_dir / "tokenisers" / f"{args.tokeniser_name}.json"
    target_tokeniser_json = target_dir / "tokenisers" / f"{args.tokeniser_name}.json"
    if source_tokeniser_json.exists():
        target_tokeniser_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_tokeniser_json, target_tokeniser_json)
        logger.info("Copied tokeniser JSON to %s", target_tokeniser_json)

    # 3. The Wyckoff records
    if cache_exists(source_dir):
        filtered_pd = slice_dataframe_cache(source_dir, target_dir, args.ehull_cutoff)
        
        # 4. split_ids.json (for val and test)
        split_ids = {split: df.index.tolist() for split, df in filtered_pd.items() if split in ("val", "test")}
        target_split_ids = target_dir / "split_ids.json"
        with open(target_split_ids, "w") as f:
            json.dump(split_ids, f)
        logger.info("Wrote %s with %s", target_split_ids.name, {k: len(v) for k, v in split_ids.items()})

    logger.info("Finished slicing dataset in %.1f seconds.", time.time() - t0)


if __name__ == "__main__":
    main()
