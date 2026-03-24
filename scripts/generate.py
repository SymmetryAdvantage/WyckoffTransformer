import argparse
import collections
import gzip
import json
import pickle
import logging
import numbers
import time
from collections import Counter
from pathlib import Path
from typing import Tuple

import torch
# torch.set_float32_matmul_precision('high')
import wandb
from omegaconf import OmegaConf

from wyckoff_transformer.trainer import WyckoffTrainer


def _resolve_sg_cache_path(dataset_name: str) -> Path:
    """Resolve dataset path inside cache directory."""
    cache_root = Path(__file__).parent.parent.resolve() / "cache"
    candidates = [dataset_name]
    if "-" in dataset_name:
        candidates.append(dataset_name.replace("-", "_"))
    for candidate in candidates:
        candidate_path = cache_root / candidate
        if candidate_path.exists():
            return candidate_path
    raise FileNotFoundError(f"Dataset '{dataset_name}' not found under {cache_root}")


def _select_tensor_and_tokeniser(cache_path: Path) -> Tuple[Path, Path]:
    """Select matching tensor/tokeniser files."""
    tensor_dir = cache_path / "tensors"
    tokeniser_dir = cache_path / "tokenisers"
    if not tensor_dir.exists() or not tokeniser_dir.exists():
        raise FileNotFoundError(f"Cache path {cache_path} lacks 'tensors' or 'tokenisers' directories.")
    for tensor_path in sorted(tensor_dir.glob("*.pt")):
        tokeniser_path = tokeniser_dir / f"{tensor_path.stem}.pkl.gz"
        if tokeniser_path.exists():
            return tensor_path, tokeniser_path
    raise FileNotFoundError(f"No matching tensor/tokeniser pair found in {cache_path}.")


def _decode_space_groups(
    start_tensor: torch.Tensor,
    source_tokeniser,
) -> Counter:
    """Decode space group identifiers from stored start tensors."""
    counts: Counter = Counter()
    # EnumeratingTokeniser case: 1D tensor with token indices
    if start_tensor.dim() == 1:
        indices = start_tensor.flatten().tolist()
        for idx in indices:
            counts[source_tokeniser.to_token[idx]] += 1
        return counts

    if not hasattr(source_tokeniser, "encode_spacegroups"):
        raise ValueError("Unsupported start tensor structure for provided tokeniser.")

    dtype = start_tensor.dtype
    sg_numbers = list(source_tokeniser.keys())
    encoded_reference = source_tokeniser.encode_spacegroups(sg_numbers, dtype=dtype, device="cpu").cpu()
    reference_map = {tuple(row.tolist()): sg for row, sg in zip(encoded_reference, sg_numbers)}
    for row in start_tensor.cpu():
        key = tuple(row.tolist())
        try:
            counts[reference_map[key]] += 1
        except KeyError as exc:
            raise ValueError("Encountered unknown space group encoding in cached tensors.") from exc
    return counts


def prepare_start_tensor_from_cache(
    trainer: WyckoffTrainer,
    dataset_name: str,
    n_samples: int,
) -> torch.Tensor:
    """Prepare start tensor sampled from cached dataset distribution."""
    cache_path = _resolve_sg_cache_path(dataset_name)
    tensor_path, tokeniser_path = _select_tensor_and_tokeniser(cache_path)

    try:
        cached_tensors = torch.load(tensor_path, map_location="cpu")
    except pickle.UnpicklingError as exc:
        if "collections.defaultdict" not in str(exc):
            raise
        try:
            torch.serialization.add_safe_globals([collections.defaultdict])  # type: ignore[attr-defined]
        except AttributeError:
            pass
        cached_tensors = torch.load(tensor_path, map_location="cpu", weights_only=False)
    with gzip.open(tokeniser_path, "rb") as f:
        cached_tokenisers = pickle.load(f)
        _ = pickle.load(f)  # token engineers, unused here

    start_field = trainer.start_name
    if start_field not in cached_tokenisers:
        raise ValueError(f"Start field '{start_field}' missing in cached tokenisers for dataset '{dataset_name}'.")
    source_tokeniser = cached_tokenisers[start_field]

    space_group_counts: Counter = Counter()
    for split_name in ("train", "val", "test"):
        split = cached_tensors.get(split_name)
        if not split:
            continue
        split_tensor = split.get(start_field)
        if split_tensor is None:
            continue
        space_group_counts.update(_decode_space_groups(split_tensor, source_tokeniser))

    if not space_group_counts:
        raise ValueError(f"No space group data found for start field '{start_field}' in dataset '{dataset_name}'.")

    target_tokeniser = trainer.tokenisers[start_field]
    filtered_counts = Counter({
        sg: count for sg, count in space_group_counts.items()
        if isinstance(sg, numbers.Integral) and sg in target_tokeniser
    })
    if not filtered_counts:
        raise ValueError(
            "None of the space groups from the requested distribution are available in the target tokeniser.")

    total = sum(filtered_counts.values())
    weights = torch.tensor(
        [filtered_counts[sg] / total for sg in filtered_counts],
        dtype=torch.float32,
        device=trainer.device,
    )
    sg_values = torch.tensor(list(filtered_counts.keys()), dtype=torch.long, device=trainer.device)
    sampled_indices = torch.multinomial(weights, n_samples, replacement=True)
    sampled_sgs = sg_values[sampled_indices].tolist()

    start_dtype = trainer.train_dataset.start_tokens.dtype
    if trainer.model.start_type == "categorial":
        token_ids = torch.tensor(
            [target_tokeniser[sg] for sg in sampled_sgs],
            dtype=start_dtype,
            device=trainer.device,
        )
        return token_ids
    if trainer.model.start_type == "one_hot":
        start_tensor = target_tokeniser.encode_spacegroups(
            sampled_sgs,
            dtype=start_dtype,
            device=trainer.device,
        )
        return start_tensor
    raise ValueError(f"Unsupported start type '{trainer.model.start_type}' for custom sg distribution.")


def main():
    parser = argparse.ArgumentParser(description="Generate structures using a Wyckoff transformer.")
    parser.add_argument("output", type=Path, help="The output file.")
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--wandb-run", type=str, help="The W&B run to use for the model.")
    model_source.add_argument("--model-path", type=Path,
           help="The path to the model directory. Should contain best_model_params.pt, "
               "wyckoff_processor.json, config.yaml")
    model_source.add_argument("--hf-model", type=str,
           help="HuggingFace repo ID to load the model from, e.g. 'username/model-name'.")
    parser.add_argument("--use-cached-tensors", action="store_true",
           help="Load cached tensors and datasets as before. By default generation does not require datasets and "
               "samples start tokens from the saved space-group distribution.")
    parser.add_argument("--initial-n-samples", type=int, help="The number of samples to try"
        " before filtering out the invalid ones.", default=1100)
    parser.add_argument("--firm-n-samples", type=int, help="The number of samples after generation, "
        "subsampling the valid ones if nesessary.", default=1000)
    parser.add_argument("--update-wandb", action="store_true", help="Update the W&B run with the "
        "generated structures and quality metrics.")
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"), help="The device to use.")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate the generator.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    parser.add_argument("--csx", action="store_true",
                        help="Enable Chemical System eXploration (CSX).")
    parser.add_argument("--required-elements", "--r", type=str,
                        help="Required elements for CSX mode (e.g., 'Li-S'). Must be provided if --csx is used.")
    parser.add_argument("--allowed-elements", "--a", type=str, default="all",
                        help="Allowed elements for CSX mode: 'all', 'fix', or a custom set (e.g., 'Li-S-P-O').")
    parser.add_argument("--sg-dist", type=str, default=None,
                        help="Override the initial space group distribution using tensors cached under cache/<dataset>.")
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    if args.output.suffixes != [".json", ".gz"]:
        raise ValueError("Output file must be a .json.gz file.")
    if args.csx and not args.required_elements:
        parser.error("--required-elements is required when --csx is enabled.")
    if args.update_wandb and not args.wandb_run:
        parser.error("--update-wandb requires --wandb-run.")
    if args.wandb_run:
        if args.update_wandb:
            wandb_run = wandb.init(project="WyckoffTransformer", id=args.wandb_run, resume=True)
        else:
            wandb_run = wandb.Api().run(f"WyckoffTransformer/{args.wandb_run}")
        config = OmegaConf.create(dict(wandb_run.config))
        run_path = Path(__file__).parent.parent / "runs" / args.wandb_run
    elif args.model_path:
        run_path = args.model_path
        config = OmegaConf.load(run_path / "config.yaml")

    generation_start_time = time.time()
    if args.hf_model:
        trainer = WyckoffTrainer.from_huggingface(
            args.hf_model,
            device=args.device,
        )
    else:
        trainer = WyckoffTrainer.from_config(
            config,
            device=args.device,
            use_cached_tensors=args.use_cached_tensors,
            run_path=run_path,
            load_datasets=args.use_cached_tensors or args.calibrate)
        trainer.model.load_state_dict(torch.load(trainer.run_path / "best_model_params.pt", weights_only=True))
    start_tensor_override = None
    if args.sg_dist is not None:
        start_tensor_override = prepare_start_tensor_from_cache(
            trainer=trainer,
            dataset_name=args.sg_dist,
            n_samples=args.initial_n_samples,
        )

    if args.csx:
        print("--- Running in Chemical System eXploration (CSX) mode ---")
        generated_wp = trainer.generate_csx_structures(
            n_structures=args.initial_n_samples,
            calibrate=args.calibrate,
            required_element_set=args.required_elements,
            allowed_element_set=args.allowed_elements,
            start_tensor=start_tensor_override,
        )
    else:
        print("--- Running in Default Generation mode ---")
        generated_wp = trainer.generate_structures(
            args.initial_n_samples,
            args.calibrate,
            start_tensor=start_tensor_override,
        )

    generation_end_time = time.time()
    print(f"Generation in total took {generation_end_time - generation_start_time} seconds")
    # print(f"Tensor generation took {tensor_generated_time - generation_start_time} seconds")
    # print(f"Detokenizing took {generation_end_time - tensor_generated_time} seconds")
    # wp_formal_validity = len(generated_wp) / generation_size
    # print(f"Wyckchoffs formal validity: {wp_formal_validity}")
    if args.firm_n_samples is not None:
        if len(generated_wp) >= args.firm_n_samples:
            generated_wp = generated_wp[:args.firm_n_samples]
        else:
            raise ValueError("Not enough valid structures to subsample.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt") as f:
        json.dump(generated_wp, f)


if __name__ == "__main__":
    main()
