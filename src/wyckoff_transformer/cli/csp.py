"""``wyformer-csp``: predict structures for a given composition.

Takes a formula rather than a sample count, decodes Wyckoff genes that have
exactly that composition, and ranks them by a censored regressor's estimate of
``min(E | gene)``.  The output is the same list of pyxtal dicts
``wyformer-generate`` writes, so it feeds ``wyformer-cryspr`` unchanged.

    wyformer-csp out.json.gz --model-path runs/abc123 \\
        --regressor-path runs/def456 --formula BaTiO3 --z 1 --condition-value 0

Space groups are enumerated rather than sampled: every group that can express
the composition at the requested number of formula units is decoded from, and
the candidates are pooled and ranked together.  That is the CSP question --
which of all the settings this formula could adopt has the lowest attainable
energy -- and it is why the space group is not a sampling decision here.
"""
import argparse
import gzip
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from omegaconf import OmegaConf

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT, wandb_run_path
from wyckoff_transformer.composition import composition_vector, describe
from wyckoff_transformer.csp import (
    CompositionTarget,
    ConstrainedDecoder,
    candidates_to_pyxtal,
    rank_by_predicted_minimum,
)
from wyckoff_transformer.trainer import WyckoffTrainer, load_model_weights

logger = logging.getLogger(__name__)


def load_trainer(
    device: torch.device,
    model_path: Optional[Path] = None,
    wandb_run: Optional[str] = None,
    hf_model: Optional[str] = None,
    wandb_entity: str = WANDB_ENTITY,
    wandb_project: str = WANDB_PROJECT,
    load_datasets: bool = False,
) -> WyckoffTrainer:
    """Load a trained model from any of the three sources the other CLIs accept."""
    import wandb  # noqa: PLC0415

    if hf_model:
        return WyckoffTrainer.from_huggingface(hf_model, device=device)
    if wandb_run:
        # A two-part path resolves against the account default entity, which is not
        # necessarily the one the run was logged under.
        run = wandb.Api().run(wandb_run_path(wandb_run, wandb_entity, wandb_project))
        config = OmegaConf.create(dict(run.config))
        run_path = Path.cwd() / "runs" / wandb_run
    elif model_path:
        run_path = model_path
        config = OmegaConf.load(run_path / "config.yaml")
    else:
        raise ValueError("No model source given")
    trainer = WyckoffTrainer.from_config(
        config, device=device, use_cached_tensors=load_datasets, run_path=run_path,
        load_datasets=load_datasets)
    load_model_weights(trainer.model, trainer.run_path / "best_model_params.pt", device)
    return trainer


def candidate_space_groups(trainer: WyckoffTrainer,
                           only: Optional[List[int]] = None) -> List[int]:
    """Space groups the backbone knows, optionally narrowed by the caller."""
    known = sorted(sg for sg in trainer.tokenisers[trainer.start_name].keys()
                   if isinstance(sg, int))
    if only is None:
        return known
    missing = sorted(set(only) - set(known))
    if missing:
        raise ValueError(f"Space groups {missing} are not in the model's vocabulary")
    return sorted(only)


def start_tensor_for(trainer: WyckoffTrainer, sg_number: int,
                     device: torch.device) -> torch.Tensor:
    """The one-row start tensor encoding a space group, in the model's convention."""
    tokeniser = trainer.tokenisers[trainer.start_name]
    if trainer.model.start_type == "categorial":
        return torch.tensor([tokeniser[sg_number]], dtype=torch.int64, device=device)
    if trainer.model.start_type == "one_hot":
        return tokeniser.encode_spacegroups([sg_number], dtype=torch.float32, device=device)
    raise ValueError(f"Unsupported start type {trainer.model.start_type!r}")


def build_condition_vector(
    backbone: WyckoffTrainer,
    target: CompositionTarget,
    scalar_cond: Optional[torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Assemble what the backbone's AdaLN expects, for one target composition.

    Mirrors `WyckoffTrainer.build_cond`, which does the same job from a dataset:
    the scalar first, then the composition. A backbone that was not trained with
    composition conditioning gets only the scalar, and the formula reaches it as
    a decoding constraint alone.
    """
    parts = []
    if backbone.condition_feature is not None:
        if scalar_cond is None:
            raise ValueError(
                f"The backbone is conditioned on {backbone.condition_feature!r}; "
                "pass --condition-value.")
        parts.append(scalar_cond)
    if backbone.composition_conditioning:
        parts.append(composition_vector(
            target.element_tokens, target.counts, backbone.n_elements,
            device=device).unsqueeze(0))
        logger.info("Conditioning on composition %s: %s", target,
                    ", ".join(describe(parts[-1][0], backbone.tokenisers["elements"])))
    if not parts:
        return None
    total = torch.cat(parts, dim=-1)
    declared = getattr(backbone.model, "condition_dim", None)
    if declared is not None and total.shape[-1] != declared:
        raise ValueError(
            f"Built a {total.shape[-1]}-wide conditioning vector for a model expecting "
            f"{declared}.")
    return total


def run_csp(
    backbone: WyckoffTrainer,
    regressor: Optional[WyckoffTrainer],
    formula: str,
    z_values: Optional[List[int]],
    space_groups: List[int],
    max_z: int,
    n_candidates: int,
    strategy: str,
    beam_width: Optional[int],
    temperature: float,
    condition_value: Optional[float],
    device: torch.device,
    seed: Optional[int] = None,
    deduplicate: bool = True,
) -> List[dict]:
    """Decode and rank candidates for one formula, over every space group and z.

    `z_values` of None enumerates the formula-unit counts each space group can
    actually hold, up to `max_z`; a list restricts that enumeration rather than
    overriding it, so an impossible request is dropped rather than attempted.

    Sampling repeats genes wherever a space group has few legal ones -- at z=1 in a
    high-symmetry group there may be only a handful -- and a repeat costs a
    relaxation without buying a structure, so identical genes are collapsed unless
    `deduplicate` is False. Two genes are the same when they occupy the same
    Wyckoff positions with the same elements; the decoding order is not part of
    the gene.
    """
    decoder = ConstrainedDecoder(
        model=backbone.model,
        cascade_order=backbone.cascade_order,
        cascade_is_target=backbone.cascade_is_target,
        tokenisers=backbone.tokenisers,
        token_engineers=backbone.token_engineers,
        masks=backbone.masks_dict,
        stops=backbone.stops_dict,
        max_sequence_len=backbone.max_sequence_length,
        device=device)

    scalar_cond = None
    if condition_value is not None:
        scalar_cond = backbone.transform_condition(
            torch.full((1, 1), condition_value, dtype=torch.float32, device=device))

    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    scored: List[tuple] = []
    seen: set = set()
    # Atoms per formula unit, in the order CompositionTarget puts its columns.
    unit = CompositionTarget.for_formula(formula, 1, backbone.tokenisers["elements"])
    targets: Dict[int, CompositionTarget] = {}
    conds: Dict[int, Optional[torch.Tensor]] = {}
    for sg_number in space_groups:
        start = start_tensor_for(backbone, sg_number, device)
        combinatorics = decoder.combinatorics(sg_number, start)
        # z is part of what CSP predicts, not an input, so it is enumerated over what
        # this space group can actually hold rather than guessed by the caller.
        allowed = combinatorics.feasible_z(
            unit.counts, backbone.max_sequence_length, max_z)
        if z_values is not None:
            requested = set(z_values)
            skipped = sorted(requested - set(allowed))
            allowed = [z for z in allowed if z in requested]
            if skipped:
                logger.debug("Space group %d cannot hold %s at z in %s",
                             sg_number, formula, skipped)
        if not allowed:
            continue
        logger.info("Space group %3d: z in %s", sg_number, allowed)
        for z in allowed:
            if z not in targets:
                targets[z] = CompositionTarget.for_formula(
                    formula, z, backbone.tokenisers["elements"])
                # The composition vector carries the cell size, which z sets.
                conds[z] = build_condition_vector(
                    backbone, targets[z], scalar_cond, device)
            target, cond = targets[z], conds[z]
            candidates = decoder.decode(
                start=start, sg_number=sg_number, target=target,
                n_candidates=n_candidates, strategy=strategy, beam_width=beam_width,
                temperature=temperature, cond=cond, generator=generator)
            if not candidates:
                continue
            if regressor is not None:
                candidates = rank_by_predicted_minimum(
                    candidates, regressor, start, combinatorics)
            logger.info("  z=%d: %d candidates%s", z,
                        len(candidates),
                        "" if regressor is None else
                        f", best predicted {candidates[0].predicted_energy:.4f} eV/atom")
            for candidate, structure in zip(
                    candidates, candidates_to_pyxtal(candidates, backbone, start, combinatorics)):
                if structure is None:
                    logger.warning("Space group %d produced a gene the processor rejected",
                                   sg_number)
                    continue
                fingerprint = (sg_number, tuple(sorted(candidate.rows)))
                if deduplicate:
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                structure = dict(structure)
                structure["csp"] = {
                    "formula": formula, "z": z, "space_group": sg_number,
                    "log_prob": candidate.log_prob, "n_sites": candidate.n_sites,
                    "predicted_energy": candidate.predicted_energy,
                }
                scored.append((candidate, structure))

    # Rank across space groups too: which setting this formula adopts is the question.
    if regressor is not None:
        scored.sort(key=lambda item: item[0].predicted_energy)
    else:
        scored.sort(key=lambda item: item[0].normalised_log_prob(), reverse=True)
    return [structure for _, structure in scored]


def main():
    parser = argparse.ArgumentParser(
        description="Predict crystal structures for a given composition.")
    parser.add_argument("output", type=Path, help="Output .json.gz file.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wandb-run", type=str, help="W&B run holding the backbone.")
    source.add_argument("--model-path", type=Path,
                        help="Directory with best_model_params.pt, wyckoff_processor.json, config.yaml")
    source.add_argument("--hf-model", type=str, help="HuggingFace repo ID of the backbone.")
    parser.add_argument("--wandb-entity", type=str, default=WANDB_ENTITY)
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT)

    regressor_source = parser.add_mutually_exclusive_group()
    regressor_source.add_argument(
        "--regressor-path", type=Path,
        help="Directory of the energy regressor. Train it with scalar_loss=censored "
             "(yamls/models/base_sg_energy_censored.yaml): an MSE regressor predicts the "
             "mean energy of a gene's structures, not the minimum, and ranking by it "
             "favours genes with little positional freedom.")
    regressor_source.add_argument("--regressor-wandb-run", type=str,
                                  help="W&B run holding the energy regressor.")

    parser.add_argument("--formula", type=str, required=True,
                        help="Reduced formula, e.g. BaTiO3.")
    parser.add_argument("--z", type=int, nargs="+", default=None,
                        help="Formula units per conventional cell. By default every value "
                             "each space group can actually hold, up to --max-z, is tried "
                             "and the results pooled: z is part of what CSP predicts, not "
                             "something the caller knows, and it is not free either -- "
                             "rocksalt is z=4 in Fm-3m and no z at all in Pnma. Give values "
                             "here only to narrow that enumeration.")
    parser.add_argument("--max-z", type=int, default=8,
                        help="Largest number of formula units to consider.")
    parser.add_argument("--space-groups", type=int, nargs="+", default=None,
                        help="Restrict to these space groups. Defaults to every group the "
                             "model knows that can express the composition.")
    parser.add_argument("--n-candidates", type=int, default=32,
                        help="Candidates per (space group, z).")
    parser.add_argument("--strategy", choices=("sample", "beam"), default="sample",
                        help="'sample' draws from the composition-masked distribution and keeps "
                             "the candidates diverse, which is what reranking wants. 'beam' "
                             "returns the modal genes, which share prefixes and spread the "
                             "relaxation budget less well.")
    parser.add_argument("--beam-width", type=int, default=None,
                        help="Paths kept per cascade field under --strategy beam.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--condition-value", type=float, default=None,
                        help="Conditioning value for a conditional backbone: 0 for "
                             "Delta_E_polymorph, or the formation energy that puts the "
                             "composition on the hull.")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="Keep repeated genes. By default identical genes are collapsed, "
                             "since a repeat costs a relaxation and buys no structure.")
    parser.add_argument("--top", type=int, default=None,
                        help="Keep only this many candidates overall, best first.")
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    if args.output.suffixes != [".json", ".gz"]:
        parser.error("Output file must be a .json.gz file.")

    started = time.time()
    backbone = load_trainer(
        device=args.device, model_path=args.model_path, wandb_run=args.wandb_run,
        hf_model=args.hf_model, wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project)
    if args.condition_value is None and backbone.condition_feature is not None:
        parser.error(
            f"The backbone is conditioned on {backbone.condition_feature!r}; pass "
            "--condition-value (0 for Delta_E_polymorph).")
    if backbone.composition_conditioning:
        print("--- Backbone is conditioned on the composition; the formula is an input, "
              "not only a decoding constraint ---")
    else:
        print("--- Backbone is not composition-conditioned; the formula enters as a "
              "decoding constraint only ---")

    regressor = None
    if args.regressor_path or args.regressor_wandb_run:
        regressor = load_trainer(
            device=args.device, model_path=args.regressor_path,
            wandb_run=args.regressor_wandb_run, wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project)
        if getattr(regressor, "scalar_loss", "mse") != "censored":
            logger.warning(
                "The regressor was fitted with scalar_loss=%r, so it estimates the mean "
                "energy of each gene's structures rather than the minimum. Ranking will "
                "lean towards genes with few degrees of freedom.",
                getattr(regressor, "scalar_loss", "mse"))
    else:
        logger.warning("No regressor given; candidates are ordered by model likelihood alone.")

    structures = run_csp(
        backbone=backbone, regressor=regressor, formula=args.formula, z_values=args.z,
        space_groups=candidate_space_groups(backbone, args.space_groups), max_z=args.max_z,
        n_candidates=args.n_candidates, strategy=args.strategy, beam_width=args.beam_width,
        temperature=args.temperature, condition_value=args.condition_value,
        device=args.device, seed=args.seed, deduplicate=not args.keep_duplicates)
    if args.top is not None:
        structures = structures[:args.top]

    print(f"{len(structures)} candidates for {args.formula} in "
          f"{time.time() - started:.1f} s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt") as handle:
        json.dump(structures, handle)


if __name__ == "__main__":
    main()
