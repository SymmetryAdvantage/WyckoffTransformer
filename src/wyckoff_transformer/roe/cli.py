"""``wyformer-roe``: run one of the rules of engagement end to end.

    wyformer-roe list
    wyformer-roe run fire-control --model-path runs/<run> --output-dir out/ \\
        --regressor-path runs/<gene-energy-run> -- --devices cuda:0
    wyformer-roe report out/

Everything after a bare ``--`` is handed to ``wyformer-protocol`` untouched, so
the reconstruction keeps every flag it has -- the MLIP, the devices, the trial
schedule, the timeouts -- without this CLI having to mirror any of them.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT
from wyckoff_transformer.cli import parse_condition_assignments
from wyckoff_transformer.evaluation.protocol import (
    DEFAULT_REFERENCE_CACHE,
    DEFAULT_REFERENCE_SPLITS,
)
from wyckoff_transformer.roe import builtin
from wyckoff_transformer.roe.cohort import MANIFEST_FILE
from wyckoff_transformer.roe.plan import RULES_OF_ENGAGEMENT, Engagement, resolve
from wyckoff_transformer.roe.report import engagement_report, write_report

logger = logging.getLogger(__name__)

DEFAULT_N_GENES = 1000


def split_protocol_argv(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Everything before the first bare ``--`` is ours; the rest is the protocol's."""
    argv = list(argv)
    if "--" not in argv:
        return argv, []
    cut = argv.index("--")
    return argv[:cut], argv[cut + 1:]


def _list(args: argparse.Namespace) -> None:
    width = max(len(name) for name in RULES_OF_ENGAGEMENT)
    for roe in RULES_OF_ENGAGEMENT.values():
        chain = " -> ".join(("generate", *roe.filters, "reconstruct"))
        if roe.needs_sampler:
            chain = "sample system -> " + chain
        print(f"{roe.name:<{width}}  {roe.summary}")
        print(f"{'':<{width}}  {chain}")
        if args.verbose:
            print(f"{'':<{width}}  {roe.rationale}\n")


def _build_source(args, roe):
    if args.genes is not None:
        return builtin.GeneFileSource(args.genes), None
    from wyckoff_transformer.cli.csp import load_trainer

    trainer = load_trainer(
        device=args.device,
        model_path=args.model_path,
        wandb_run=args.wandb_run,
        hf_model=args.hf_model,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        load_datasets=False,
    )
    if hasattr(trainer.model, "_orig_mod"):
        trainer.model = trainer.model._orig_mod
    condition = parse_condition_assignments(args.condition or []) or None
    space_groups = None
    if args.space_group:
        space_groups = [int(part) for part in args.space_group.split(",") if part.strip()]
    source = builtin.WyFormerGeneSource(
        trainer,
        condition_values=condition,
        chemical_system=args.chemical_system,
        space_groups=space_groups,
        required_elements=args.required_elements,
        allowed_elements=args.allowed_elements,
        temperature=args.temperature,
        oversample=args.oversample,
        allow_fewer=args.allow_fewer,
        device=args.device,
        model_id=str(args.model_path or args.wandb_run or args.hf_model),
    )
    return source, trainer


def _build_sampler(args, trainer):
    if args.system_plan is not None:
        if trainer is None:
            raise SystemExit(
                "--system-plan decodes element tokens through the model's vocabulary, "
                "so it needs a model rather than --genes.")
        vocabulary = [str(symbol) for symbol in trainer.tokenisers["elements"].to_token]
        return builtin.PlanFileSampler(args.system_plan, vocabulary)
    if args.system_prior is not None:
        return builtin.SystemPriorSampler(
            args.system_prior,
            required=args.required,
            allowed=args.allowed,
            novel_fraction=args.novel_fraction,
            system_temperature=args.system_temperature,
            sg_temperature=args.sg_temperature,
            min_arity=args.min_arity,
            max_arity=args.max_arity,
            seed=args.sampler_seed,
        )
    return None


def _build_screen(args):
    return builtin.NoveltyUniquenessScreen(
        reference_cache=args.reference_cache,
        reference_splits=tuple(s.strip() for s in args.reference_splits.split(",")),
        fingerprint_cache=args.reference_fingerprint_cache,
        key_table_path=args.key_table,
        backend=args.screen_backend,
        uniqueness=not args.no_uniqueness,
        novelty=not args.no_novelty,
    )


def _build_energy(args):
    from wyckoff_transformer.cli.csp import load_trainer
    from wyckoff_transformer.cli.gene_screen import load_reference

    if args.regressor_path is None and args.regressor_wandb_run is None:
        raise SystemExit(
            "This mode ranges before it fires, so it needs an energy predictor: pass "
            "--regressor-path or --regressor-wandb-run.")
    regressor = load_trainer(
        device=args.device,
        model_path=args.regressor_path,
        wandb_run=args.regressor_wandb_run,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
    )
    return builtin.PredictedHullFilter(
        regressor,
        load_reference(args.hull_reference),
        margin=args.hull_margin,
        select=args.energy_select,
        top_k=args.energy_top,
        on_missing_hull=args.on_missing_hull,
        augmentation_samples=args.augmentation_samples,
        regressor_id=str(args.regressor_path or args.regressor_wandb_run),
        reference_id=str(args.hull_reference),
    )


def _run(args: argparse.Namespace, protocol_argv: list[str]) -> None:
    roe = resolve(args.mode)
    source, trainer = _build_source(args, roe)
    sampler = _build_sampler(args, trainer)

    filters = {}
    if "screen" in roe.filters:
        filters["screen"] = _build_screen(args)
    if "energy" in roe.filters:
        filters["energy"] = _build_energy(args)

    reconstructor = None
    if not args.no_reconstruct:
        if args.reconstructor == "cryspr":
            reconstructor = builtin.CrySPRReconstructor(protocol_argv)
        else:
            reconstructor = builtin.DiffCSPReconstructor()

    engagement = Engagement(
        roe=roe, source=source, filters=filters, sampler=sampler,
        reconstructor=reconstructor)

    cohort = engagement.run(
        args.n_genes, args.output_dir,
        target_engaged=args.target_engaged,
        max_rounds=args.max_rounds,
        max_sampled=args.max_sampled,
    )
    report = engagement_report(cohort, roe, args.output_dir)
    path = write_report(report, args.output_dir)
    print(json.dumps(
        {"rules_of_engagement": roe.name, **report["cohort"],
         "cost": report["cost"], "trials_per_hit": report.get("trials_per_hit")},
        indent=2))
    print(f"\nFull report: {path}")


def _report(args: argparse.Namespace) -> None:
    """Rebuild the report from what is already on disk.

    A reconstruction that ran on another machine, or was resumed, leaves the
    protocol outputs and the cohort in the same directory and nothing joining
    them; this is that join.
    """
    import gzip

    import pandas as pd

    from wyckoff_transformer.roe.cohort import COHORT_FILE, Cohort

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise SystemExit(f"No {MANIFEST_FILE} in {output_dir}; this is not a mode's output directory.")
    with open(manifest_path, "rt", encoding="utf-8") as handle:
        manifest = json.load(handle)
    table = pd.read_csv(output_dir / COHORT_FILE, index_col=0)
    with gzip.open(output_dir / "engaged_genes.json.gz", "rt", encoding="utf-8") as handle:
        engaged = json.load(handle)

    cohort = Cohort(genes=[{} for _ in range(manifest["sampled"])], table=table)
    cohort.provenance = manifest.get("provenance", {})
    if len(engaged) != int(table["kept"].sum()):
        logger.warning(
            "The gene file holds %d genes and the cohort marks %d as engaged; the "
            "report is built from the cohort.", len(engaged), int(table["kept"].sum()))
    roe = resolve(manifest["provenance"].get("rules_of_engagement", args.mode))
    report = engagement_report(cohort, roe, output_dir)
    report["stages"] = manifest.get("stages", [])
    path = write_report(report, output_dir)
    print(f"Report written to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wyformer-roe",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    listing = subparsers.add_parser("list", help="The modes, what they run, and why.")
    listing.add_argument("--verbose", "-v", action="store_true", help="Include the rationale.")
    listing.set_defaults(func=_list)

    run = subparsers.add_parser(
        "run", help="Draw a cohort, filter it, and reconstruct what survives.")
    run.add_argument("mode", choices=sorted(RULES_OF_ENGAGEMENT))
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--n-genes", type=int, default=DEFAULT_N_GENES,
                     help="Genes to draw in the first round.")
    run.add_argument("--target-engaged", type=int, default=None,
                     help="Keep drawing until this many genes survive every filter, "
                          "then truncate to it. This is what makes two modes "
                          "comparable: the reconstruction is the expensive part, so "
                          "the budget rather than the draw is what to hold fixed.")
    run.add_argument("--max-rounds", type=int, default=12,
                     help="Top-up rounds before giving up on --target-engaged.")
    run.add_argument("--max-sampled", type=int, default=None,
                     help="Genes drawn before giving up. Defaults to 50x the target.")
    run.add_argument("--device", type=torch.device, default=torch.device("cpu"))

    source = run.add_argument_group("gene source")
    which = source.add_mutually_exclusive_group(required=True)
    which.add_argument("--model-path", type=Path)
    which.add_argument("--wandb-run", type=str)
    which.add_argument("--hf-model", type=str)
    which.add_argument("--genes", type=Path,
                       help="Re-run the mode's filters over a cohort already drawn.")
    source.add_argument("--wandb-entity", type=str, default=WANDB_ENTITY)
    source.add_argument("--wandb-project", type=str, default=WANDB_PROJECT)
    source.add_argument("--condition", action="append", metavar="NAME=VALUE")
    source.add_argument("--temperature", type=float, default=1.0)
    source.add_argument("--chemical-system", type=str, default=None, metavar="Ba-Ti-O",
                        help="One system for the whole cohort. Naming one is not a "
                             "torpedo run: a run samples the systems it aims at.")
    source.add_argument("--space-group", type=str, default=None, metavar="N[,M...]")
    source.add_argument("--required-elements", type=str, default=None)
    source.add_argument("--allowed-elements", type=str, default=None)
    source.add_argument("--oversample", type=float, default=builtin.DEFAULT_OVERSAMPLE)
    source.add_argument("--allow-fewer", action="store_true",
                        help="Accept a short cohort rather than raising.")

    sampler = run.add_argument_group("chemical-system sampler (torpedo-run)")
    sampler.add_argument("--system-prior", type=Path, default=None)
    sampler.add_argument("--system-plan", type=Path, default=None)
    sampler.add_argument("--required", "-r", type=str, default=None)
    sampler.add_argument("--allowed", "-a", type=str, default=None)
    sampler.add_argument("--novel-fraction", type=float, default=None)
    sampler.add_argument("--system-temperature", type=float, default=1.0)
    sampler.add_argument("--sg-temperature", type=float, default=1.0)
    sampler.add_argument("--min-arity", type=int, default=None)
    sampler.add_argument("--max-arity", type=int, default=None)
    sampler.add_argument("--sampler-seed", type=int, default=None)

    screen = run.add_argument_group("screen slot")
    screen.add_argument("--reference-cache", type=Path, default=DEFAULT_REFERENCE_CACHE)
    screen.add_argument("--reference-splits", type=str,
                        default=",".join(DEFAULT_REFERENCE_SPLITS))
    screen.add_argument("--reference-fingerprint-cache", type=Path, default=None,
                        help="For --screen-backend python.")
    screen.add_argument("--screen-backend",
                        choices=builtin.NoveltyUniquenessScreen.BACKENDS, default="tensor",
                        help="'tensor' uses the 128-bit gene keys and a sorted table; "
                             "'python' uses the nested-frozenset fingerprint set. They "
                             "are pinned equivalent by tests/test_gene_hash.py, and the "
                             "tensor one needs two orders of magnitude less memory.")
    screen.add_argument("--key-table", type=Path, default=None,
                        help="For --screen-backend tensor. Built beside the reference "
                             "on first use if absent.")
    screen.add_argument("--no-uniqueness", action="store_true")
    screen.add_argument("--no-novelty", action="store_true")

    energy = run.add_argument_group("energy slot")
    energy.add_argument("--regressor-path", type=Path, default=None)
    energy.add_argument("--regressor-wandb-run", type=str, default=None)
    energy.add_argument("--hull-reference", type=Path,
                        default=Path("data/lemat-bulk/lemat_pbe_ehull.csv.gz"))
    energy.add_argument("--hull-margin", type=float, default=0.0,
                        help="Keep genes whose predicted e_hull is at most this, in "
                             "eV/atom. 0 is 'predicted on or below the hull'.")
    energy.add_argument("--energy-select",
                        choices=builtin.PredictedHullFilter.SELECTORS, default="threshold",
                        help="'threshold' keeps every gene at or below --hull-margin; "
                             "'top' keeps the --energy-top lowest. Use 'top' when the "
                             "reconstruction budget is what is fixed: at a threshold of "
                             "0 only a few percent of novel genes get through, so "
                             "filling a budget from it takes tens of thousands of draws.")
    energy.add_argument("--energy-top", type=int, default=None,
                        help="The budget, for --energy-select top.")
    energy.add_argument("--on-missing-hull", choices=("keep", "drop"), default="keep",
                        help="What to do with a gene whose composition the reference "
                             "hull does not cover. Keeping is the default because "
                             "dropping makes the filter select against novel chemistry.")
    energy.add_argument("--augmentation-samples", type=int, default=1)

    yard = run.add_argument_group("reconstruction")
    yard.add_argument("--reconstructor", choices=("cryspr", "diffcsp++"), default="cryspr")
    yard.add_argument("--no-reconstruct", action="store_true",
                      help="Stop at the filtered gene file. What a run whose "
                           "reconstruction happens elsewhere wants.")
    run.set_defaults(func=_run)

    report = subparsers.add_parser(
        "report", help="Rebuild the report from a directory a run left behind.")
    report.add_argument("output_dir", type=Path)
    report.add_argument("--mode", choices=sorted(RULES_OF_ENGAGEMENT), default=None,
                        help="Only needed when the manifest does not name one.")
    report.set_defaults(func=_report)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    ours, protocol_argv = split_protocol_argv(
        sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(ours)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.func is _run:
        _run(args, protocol_argv)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
