import argparse
import gzip
import json
import logging
import numbers
import time
from collections import Counter
from pathlib import Path
from typing import Tuple

import torch
import wandb
from omegaconf import OmegaConf

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT, wandb_run_path
from wyckoff_transformer.chemical_system import (
    chemical_system_vector,
    parse_chemical_system,
)
from wyckoff_transformer.cli import describe_condition, resolve_condition_values
from wyckoff_transformer.system_prior import SystemDraws, SystemSpaceGroupPrior
from wyckoff_transformer.tokenization import TENSOR_CACHE_SUFFIX, load_tensor_cache
from wyckoff_transformer.trainer import WyckoffTrainer, load_model_weights
from wyckoff_transformer.wyckoff_processor import WyckoffProcessor


def _resolve_sg_cache_path(dataset_name: str, cache_root: Path | None = None) -> Path:
    """Resolve dataset path inside cache directory."""
    if cache_root is None:
        cache_root = Path.cwd() / "cache"
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
    for tensor_path in sorted(tensor_dir.glob(f"*{TENSOR_CACHE_SUFFIX}")):
        tokeniser_path = tokeniser_dir / f"{tensor_path.stem}.json"
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
    cache_root: Path | None = None,
) -> torch.Tensor:
    """Prepare start tensor sampled from cached dataset distribution."""
    cache_path = _resolve_sg_cache_path(dataset_name, cache_root=cache_root)
    tensor_path, tokeniser_path = _select_tensor_and_tokeniser(cache_path)

    cached_tensors = load_tensor_cache(tensor_path, map_location="cpu")
    cached_tokenisers = WyckoffProcessor.from_pretrained(tokeniser_path).tokenisers

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


def chemical_system_vector_for_generation(
    system: str,
    elements_tokeniser,
    n_rows: int,
    device: torch.device,
) -> torch.Tensor:
    """One chemical-system row, repeated over the batch.

    Every structure in a run is asked for the same system, so the block is built once
    and expanded rather than assembled per row.
    """
    _, tokens = parse_chemical_system(system, elements_tokeniser)
    row = chemical_system_vector(tokens, len(elements_tokeniser), device=device)
    return row.unsqueeze(0).expand(n_rows, -1).contiguous()


def prepare_start_tensor_for_space_groups(
    trainer: WyckoffTrainer,
    space_groups: list[int],
    n_samples: int,
) -> torch.Tensor:
    """Start tokens drawn uniformly from the space groups the caller named.

    The space group is this architecture's start token, so it is a conditioning input
    like any other -- the difference is only that it enters as the sequence's first
    element rather than through AdaLN. Naming one pins every generated structure to it;
    naming several spreads the batch evenly over them, which is what a chemical-system
    sweep over candidate settings wants. Saying nothing leaves the choice to the
    training distribution, which is the default and the right thing when the question
    is "what does this system crystallise as".
    """
    tokeniser = trainer.tokenisers[trainer.start_name]
    unknown = [sg for sg in space_groups if sg not in tokeniser]
    if unknown:
        raise ValueError(
            f"Space groups {unknown} are not in the model's vocabulary; it was trained on "
            f"{len(tokeniser)} of them.")
    picks = [space_groups[index] for index in
             torch.randint(0, len(space_groups), (n_samples,)).tolist()]
    if trainer.model.start_type == "categorial":
        return torch.tensor([tokeniser[sg] for sg in picks],
                            dtype=torch.int64, device=trainer.device)
    if trainer.model.start_type == "one_hot":
        return tokeniser.encode_spacegroups(
            picks, dtype=torch.float32, device=trainer.device)
    raise ValueError(f"Unsupported start type '{trainer.model.start_type}'.")


def plan_output_path(output: Path) -> Path:
    """`out.json.gz` -> `out.plan.json`, so a run keeps what it asked for next to what it got."""
    name = output.name
    for suffix in (".json.gz", ".json"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return output.with_name(f"{name}.plan.json")


def build_draws(args, vocabulary: list, n_structures: int) -> SystemDraws:
    """The per-structure (chemical system, space group) requests, drawn or read.

    A plan read from disk fixes the batch size to its own: it names a system and a
    space group for each structure, so asking for a different number of structures
    would mean either dropping requests or inventing them.
    """
    if args.system_plan is not None:
        with open(args.system_plan, "rt") as plan_file:
            manifest = json.load(plan_file)
        draws = SystemDraws.from_manifest(manifest, vocabulary)
        if args.initial_n_samples != len(draws):
            print(f"--- Plan holds {len(draws)} structures; generating that many rather "
                  f"than --initial-n-samples {args.initial_n_samples} ---")
        return draws
    prior = SystemSpaceGroupPrior.load(args.system_prior)
    if list(prior.element_symbols) != vocabulary:
        raise ValueError(
            f"The prior at {args.system_prior} was built over {prior.n_elements} element "
            f"tokens and the model knows {len(vocabulary)}; they have to come from the "
            "same dataset, or a system would decode into different elements.")
    return prior.sample(
        n_structures,
        required=args.required_elements,
        allowed=args.allowed_elements,
        novel_fraction=args.novel_fraction,
        rng=args.sampler_seed)


def keep_required_elements(generated_wp: list, required: str | None) -> list:
    """Drop structures that left a required element out, and say how many.

    Nothing forces a required element in when the batch is drawn from a plan, and
    that is deliberate: the mask is the row's own chemical system, and the argmax
    forcing that `required_element_set` applies would place an element at whatever
    Wyckoff position happened to be next rather than where the model wanted it.
    Rejection after the fact costs a few percent of the batch and distorts nothing --
    it conditions on the event instead of editing the sample.
    """
    if not required:
        return generated_wp
    wanted = {part for part in required.split("-") if part}
    kept = [structure for structure in generated_wp
            if wanted.issubset(set(structure["species"]))]
    if len(kept) != len(generated_wp):
        print(f"--- {len(kept)} of {len(generated_wp)} generated structures contain "
              f"{'-'.join(sorted(wanted))}; the rest answered with a subsystem ---")
    return kept


def main():
    parser = argparse.ArgumentParser(description="Generate structures using a Wyckoff transformer.")
    parser.add_argument("output", type=Path, help="The output file.")
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--wandb-run", type=str, help="The W&B run to use for the model.")
    parser.add_argument("--wandb-entity", type=str, default=WANDB_ENTITY,
                        help="W&B entity holding --wandb-run. Pinned by default so a run is "
                             "looked up where it was logged, not under the shell's default.")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT, help="W&B project")
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
    parser.add_argument("--required-elements", "--r", type=str,
                        help="Required elements for constrained generation (e.g., 'Li-S'). "
                             "All listed elements will appear in every generated structure.")
    parser.add_argument("--allowed-elements", "--a", type=str, default=None,
                        help="Allowed elements for constrained generation: 'fix' (restrict to --required-elements), "
                             "or a custom set (e.g., 'Li-S-P-O'). Defaults to all elements when omitted.")
    parser.add_argument("--chemical-system", type=str, default=None, metavar="Ba-Ti-O",
                        help="Generate within this set of elements, for a model trained "
                             "with chemical_system_conditioning. The set is both an input "
                             "to the model and a mask on the sampler, so every structure "
                             "is built from these elements and nothing else. Use "
                             "--allowed-elements instead to mask an unconditioned model.")
    parser.add_argument("--system-prior", type=Path, default=None, metavar="PRIOR.npz",
                        help="Draw a chemical system and a space group per structure from "
                             "this prior, inside --required-elements and --allowed-elements, "
                             "instead of naming one system for the whole batch. Built by "
                             "`wyformer-system-prior build`; see "
                             "docs/chemical_system_sampler.md.")
    parser.add_argument("--system-plan", type=Path, default=None, metavar="PLAN.json",
                        help="A plan already drawn by `wyformer-system-prior sample`. Fixes "
                             "the batch size to the plan's, and makes the campaign exactly "
                             "reproducible without carrying the prior around.")
    parser.add_argument("--novel-fraction", type=float, default=None,
                        help="With --system-prior: the share of structures asked for a "
                             "chemical system the training data does not contain. Defaults "
                             "to the novelty rate measured when the prior was built. Pass "
                             "--system-plan for the rest of the sampler's knobs.")
    parser.add_argument("--sampler-seed", type=int, default=None,
                        help="Seed for --system-prior, so the same plan can be redrawn.")
    parser.add_argument("--space-group", type=str, default=None, metavar="N[,M...]",
                        help="Generate only in these space groups, by number, spread "
                             "evenly over them. The space group is this model's start "
                             "token, so it is a conditioning input; omit it to sample "
                             "from the training distribution.")
    parser.add_argument("--sg-dist", type=str, default=None,
                        help="Override the initial space group distribution using tensors cached under cache/<dataset>.")
    parser.add_argument("--condition", action="append", metavar="NAME=VALUE", default=None,
                        help="Value of one conditioning feature to use for every generated "
                             "structure, e.g. --condition energy_above_hull=0. Repeat once "
                             "per feature; a model conditioned on several channels needs "
                             "all of them. Omit to sample the conditioning from the training "
                             "distribution, which needs --use-cached-tensors.")
    parser.add_argument("--condition-value", type=float, default=None,
                        help="Shorthand for --condition <the one feature>=VALUE. Refused for "
                             "a model with more than one conditioning feature, where a single "
                             "number does not say what it applies to.")
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    if args.output.suffixes != [".json", ".gz"]:
        raise ValueError("Output file must be a .json.gz file.")
    if args.update_wandb and not args.wandb_run:
        parser.error("--update-wandb requires --wandb-run.")
    if args.space_group is not None and args.sg_dist is not None:
        parser.error("--space-group and --sg-dist both set the start tokens; use one.")
    if args.system_prior is not None and args.system_plan is not None:
        parser.error("--system-prior draws a plan and --system-plan reads one; use one.")
    plan_requested = args.system_prior is not None or args.system_plan is not None
    if plan_requested:
        # A plan supplies the conditioning vector and the start token of every row, so
        # anything else that sets either of them is saying something the plan already says.
        for flag, value in (("--chemical-system", args.chemical_system),
                            ("--space-group", args.space_group),
                            ("--sg-dist", args.sg_dist)):
            if value is not None:
                parser.error(
                    f"{flag} and the system plan both set what each structure is asked "
                    "for; use one.")
    space_groups = None
    if args.space_group is not None:
        try:
            space_groups = [int(part) for part in args.space_group.split(",") if part.strip()]
        except ValueError:
            parser.error(f"--space-group takes space group numbers, got {args.space_group!r}")
        if not space_groups:
            parser.error("--space-group is empty")

    generation_start_time = time.time()
    if args.hf_model:
        trainer = WyckoffTrainer.from_huggingface(
            args.hf_model,
            device=args.device,
        )
    else:
        if args.wandb_run:
            if args.update_wandb:
                wandb_run = wandb.init(
                    entity=args.wandb_entity, project=args.wandb_project,
                    id=args.wandb_run, resume=True)
            else:
                # A two-part path resolves against the account default entity, which is not
                # necessarily the one the run was logged under.
                wandb_run = wandb.Api().run(
                    wandb_run_path(args.wandb_run, args.wandb_entity, args.wandb_project))
            config = OmegaConf.create(dict(wandb_run.config))
            run_path = Path.cwd() / "runs" / args.wandb_run
        elif args.model_path:
            run_path = args.model_path
            config = OmegaConf.load(run_path / "config.yaml")
        else:
            raise ValueError("Model source not specified.")
        trainer = WyckoffTrainer.from_config(
            config,
            device=args.device,
            use_cached_tensors=args.use_cached_tensors,
            run_path=run_path,
            load_datasets=args.use_cached_tensors or args.calibrate)
        load_model_weights(
            trainer.model, trainer.run_path / "best_model_params.pt", args.device)
    start_tensor_override = None
    if args.sg_dist is not None:
        start_tensor_override = prepare_start_tensor_from_cache(
            trainer=trainer,
            dataset_name=args.sg_dist,
            n_samples=args.initial_n_samples,
        )
    elif space_groups is not None:
        try:
            start_tensor_override = prepare_start_tensor_for_space_groups(
                trainer=trainer,
                space_groups=space_groups,
                n_samples=args.initial_n_samples,
            )
        except ValueError as error:
            parser.error(str(error))
        print(f"--- Space group fixed to {args.space_group} ---")

    cond = None
    try:
        condition_values = resolve_condition_values(
            trainer, args.condition, args.condition_value)
    except ValueError as error:
        parser.error(str(error))
    if condition_values is not None:
        try:
            cond = trainer.build_condition_from_values(
                condition_values, args.initial_n_samples, device=args.device)
        except ValueError as error:
            parser.error(str(error))
        print(f"--- Conditioning on {describe_condition(condition_values)} ---")
    elif trainer.condition_features:
        print(f"--- Conditional model ({', '.join(trainer.condition_features)}); "
              "sampling condition from training data ---")

    system_cond = None
    allowed_elements = args.allowed_elements
    if args.chemical_system is not None:
        if not trainer.chemical_system_conditioning:
            parser.error(
                "--chemical-system needs a model trained with "
                "chemical_system_conditioning; this one is not. To restrict the sampler "
                "without conditioning on the set, pass --allowed-elements.")
        elements_tokeniser = trainer.tokenisers["elements"]
        try:
            symbols, _ = parse_chemical_system(args.chemical_system, elements_tokeniser)
            system_cond = chemical_system_vector_for_generation(
                args.chemical_system, elements_tokeniser, args.initial_n_samples, args.device)
        except (KeyError, ValueError) as error:
            parser.error(str(error))
        # The mask is what makes the set a guarantee rather than a preference: the
        # conditioning says which elements the structure is made of, and the model is
        # free to disagree. Unless the caller has narrowed or widened the pool by hand,
        # the sampler sees exactly the system asked for.
        if allowed_elements is None:
            allowed_elements = "-".join(symbols)
        print(f"--- Conditioning on the chemical system {'-'.join(symbols)}; "
              f"the sampler is restricted to {allowed_elements} ---")
    elif trainer.chemical_system_conditioning and not plan_requested:
        print("--- Chemical-system-conditioned model; sampling the system from the "
              "training distribution (pass --chemical-system to ask for one) ---")

    element_mask = None
    draws = None
    n_structures = args.initial_n_samples
    if plan_requested:
        if not trainer.chemical_system_conditioning:
            parser.error(
                "A system plan needs a model trained with chemical_system_conditioning; "
                "this one is not. Use --allowed-elements to mask an unconditioned model.")
        elements_tokeniser = trainer.tokenisers["elements"]
        vocabulary = [str(symbol) for symbol in elements_tokeniser.to_token]
        try:
            draws = build_draws(args, vocabulary, n_structures)
        except (KeyError, ValueError) as error:
            parser.error(str(error))
        n_structures = len(draws)
        system_cond = draws.conditioning_block(len(elements_tokeniser), device=args.device)
        start_tensor_override = draws.start_tensor(
            trainer.tokenisers[trainer.start_name], trainer.model.start_type,
            device=args.device)
        # Per row, and the row's own system rather than the palette: a single allowed set
        # for the batch would let every structure back into the whole palette and undo the
        # sampling that chose the systems.
        element_mask = draws.element_mask(
            len(elements_tokeniser), stop_token=elements_tokeniser.stop_token,
            device=args.device)
        allowed_elements = None
        source = "Plan" if args.system_plan is not None else "Sampled plan"
        print(f"--- {source}: {draws.summary(top=5)} ---")
        plan_path = plan_output_path(args.output)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(plan_path, "wt") as plan_file:
            json.dump(draws.manifest(), plan_file, indent=1)
        print(f"--- Plan written to {plan_path} ---")

    use_element_constraints = (not plan_requested) and (
        args.required_elements is not None or allowed_elements is not None)
    if plan_requested:
        print("--- Running in per-row chemical system generation mode ---")
    elif use_element_constraints:
        print("--- Running in constrained element generation mode ---")
    else:
        print("--- Running in default generation mode ---")
    generated_wp = trainer.generate_structures(
        n_structures=n_structures,
        calibrate=args.calibrate,
        start_tensor=start_tensor_override,
        cond=cond,
        composition_cond=system_cond,
        required_element_set=args.required_elements if use_element_constraints else None,
        allowed_element_set=allowed_elements if allowed_elements is not None else "all",
        allowed_element_mask=element_mask,
    )

    generation_end_time = time.time()
    print(f"Generation in total took {generation_end_time - generation_start_time} seconds")
    if draws is not None:
        generated_wp = keep_required_elements(generated_wp, draws.query.get("required"))
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
