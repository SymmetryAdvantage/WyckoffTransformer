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
        # With a model, the plan is checked against its vocabulary; re-filtering a
        # pool (--genes) decodes it with the vocabulary the plan itself carries.
        vocabulary = None if trainer is None else [
            str(symbol) for symbol in trainer.tokenisers["elements"].to_token]
        return builtin.PlanFileSampler(args.system_plan, vocabulary)
    if args.system_prior is not None and args.closure:
        targets = [t.strip() for t in args.targets.split(",")] if args.targets else None
        return builtin.SubsystemClosureSampler(
            args.system_prior,
            targets=targets,
            n_targets=None if targets else args.n_targets,
            target_arity=args.target_arity,
            min_arity=args.closure_min_arity,
            target_share=args.target_share,
            required=args.required,
            allowed=args.allowed,
            novel_fraction=args.novel_fraction,
            system_temperature=args.system_temperature,
            sg_temperature=args.sg_temperature,
            seed=args.sampler_seed,
        )
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
    from wyckoff_transformer.cli.gene_screen import check_regressor_reference

    check_regressor_reference(regressor, args.hull_reference, args.allow_incompatible_energy)
    correction = known = None
    if args.residuals is not None:
        from wyckoff_transformer.gene_energy_residuals import (
            KNOWN_GENES_FILE,
            KnownGeneEnergies,
            load_correction,
        )

        known = KnownGeneEnergies.load(args.residuals / KNOWN_GENES_FILE)
        correction = load_correction(
            args.residuals, kappa=args.residual_kappa, regressor=regressor,
            allow_incompatible_energy=args.allow_incompatible_energy)
    elif args.energy_basis == "corrected":
        raise SystemExit("--energy-basis corrected needs --residuals")
    # No margin means "rank everything" under `rank` and "on or below the hull"
    # under `threshold`.
    margin = args.hull_margin
    if margin is None and args.energy_select != "rank":
        margin = 0.0
    return builtin.PredictedHullFilter(
        regressor,
        load_reference(args.hull_reference),
        margin=margin,
        select=args.energy_select,
        top_k=args.energy_top,
        on_missing_hull=args.on_missing_hull,
        augmentation_samples=args.augmentation_samples,
        regressor_id=str(args.regressor_path or args.regressor_wandb_run),
        reference_id=str(args.hull_reference),
        hull=args.energy_hull,
        energy=args.energy_basis,
        correction=correction,
        known_genes=known,
        residuals_id=str(args.residuals) if args.residuals is not None else None,
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


#: What ``draw`` writes: the pool, and the plan it was drawn against.
POOL_GENES_FILE = "wyckoff_genes.json.gz"
POOL_PLAN_FILE = "system_plan.json"
POOL_MANIFEST_FILE = "pool_manifest.json"


def _draw(args: argparse.Namespace) -> None:
    """Draw one pool that several arms then select from with ``--genes``.

    What makes the arms of a torpedo run paired: they aim at the same targets,
    see the same draws, and differ only in what they select.
    """
    import gzip
    import time

    source, trainer = _build_source(args, None)
    sampler = _build_sampler(args, trainer)
    if sampler is None and getattr(source, "requires_plan", False):
        raise SystemExit("This checkpoint generates into a planned system: pass --system-prior.")
    started = time.time()
    plan = sampler.plan(source.attempts(args.n_genes)) if sampler is not None else None
    genes = source.draw(args.n_genes, plan)
    seconds = time.time() - started

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / POOL_GENES_FILE, "wt", encoding="utf-8") as handle:
        json.dump(genes, handle)
    if plan is not None:
        with open(out / POOL_PLAN_FILE, "wt", encoding="utf-8") as handle:
            json.dump(plan.manifest(), handle, indent=1)
        print(plan.summary())
    with open(out / POOL_MANIFEST_FILE, "wt", encoding="utf-8") as handle:
        json.dump({
            "genes": len(genes), "asked": args.n_genes, "seconds": round(seconds, 1),
            "source": source.describe(),
            "sampler": sampler.describe() if sampler is not None else None,
        }, handle, indent=1, default=str)
    print(f"{len(genes)} genes in {seconds:.0f} s: {out / POOL_GENES_FILE}")


def _flatten(prefix: str, value, out: dict) -> dict:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}/{key}", item, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = value
    return out


def _upload(args: argparse.Namespace) -> None:
    """One W&B run holding a campaign: the pool, the residuals, and every arm.

    Nothing under ``runs/`` is the only copy of a result: each arm's report goes
    into the run summary under the arm's name, and every input and output worth
    keeping into one artifact.
    """
    import wandb

    from wyckoff_transformer.cli.protocol_wandb import add_protocol_outputs
    from wyckoff_transformer.gene_energy_residuals import (
        RESIDUALS_FILE,
        VALIDATION_FILE,
    )
    from wyckoff_transformer.paths import wandb_dir
    from wyckoff_transformer.roe.cohort import COHORT_FILE, ENGAGED_GENES_FILE
    from wyckoff_transformer.roe.report import REPORT_FILE

    root = Path(args.root)
    config = {"root": str(root), **dict(item.split("=", 1) for item in args.config or [])}
    run = wandb.init(dir=wandb_dir(), entity=args.wandb_entity, project=args.wandb_project,
                     name=args.name, id=args.run_id, resume="allow", job_type="roe",
                     config=config)
    try:
        summary = {}
        artifact = wandb.Artifact(args.artifact or args.name, type="roe", metadata=config)
        for sub, names in (("pool", (POOL_GENES_FILE, POOL_PLAN_FILE, POOL_MANIFEST_FILE)),
                           ("residuals", (RESIDUALS_FILE, VALIDATION_FILE))):
            for name in names:
                path = root / sub / name
                if path.is_file():
                    artifact.add_file(str(path), name=f"{sub}/{name}")
        validation = root / "residuals" / VALIDATION_FILE
        if validation.is_file():
            _flatten("residuals", json.loads(validation.read_text()), summary)
        for arm in args.arms:
            arm_dir = root / arm
            for name in (MANIFEST_FILE, COHORT_FILE, ENGAGED_GENES_FILE, REPORT_FILE,
                         "funnel_from_reconstructor.json"):
                if (arm_dir / name).is_file():
                    artifact.add_file(str(arm_dir / name), name=f"{arm}/{name}")
            add_protocol_outputs(artifact, arm_dir / "protocol", prefix=f"{arm}/protocol/")
            report = arm_dir / REPORT_FILE
            if report.is_file():
                _flatten(arm, json.loads(report.read_text()), summary)
        run.summary.update(summary)
        run.log_artifact(artifact)
    finally:
        run.finish()
    print(f"Logged {len(summary)} summary values and artifact {args.artifact or args.name}")


def _add_source_arguments(parser: argparse.ArgumentParser, allow_genes: bool) -> None:
    source = parser.add_argument_group("gene source")
    which = source.add_mutually_exclusive_group(required=True)
    which.add_argument("--model-path", type=Path)
    which.add_argument("--wandb-run", type=str)
    which.add_argument("--hf-model", type=str)
    if allow_genes:
        which.add_argument("--genes", type=Path,
                           help="Re-run the mode's filters over a cohort already drawn.")
    else:
        parser.set_defaults(genes=None)
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


def _add_sampler_arguments(parser: argparse.ArgumentParser) -> None:
    sampler = parser.add_argument_group("chemical-system sampler (torpedo-run)")
    sampler.add_argument("--system-prior", type=Path, default=None,
                         help="A chemical-system run keeps its own, as system_prior.npz "
                              "in the run directory.")
    sampler.add_argument("--system-plan", type=Path, default=None)
    sampler.add_argument("--required", "-r", type=str, default=None)
    sampler.add_argument("--allowed", "-a", type=str, default=None)
    sampler.add_argument("--novel-fraction", type=float, default=None)
    sampler.add_argument("--system-temperature", type=float, default=1.0)
    sampler.add_argument("--sg-temperature", type=float, default=1.0)
    sampler.add_argument("--min-arity", type=int, default=None)
    sampler.add_argument("--max-arity", type=int, default=None)
    sampler.add_argument("--sampler-seed", type=int, default=None)
    sampler.add_argument("--closure", action="store_true",
                         help="Aim at target systems and request every subsystem of "
                              "each (down to --closure-min-arity) as well.")
    sampler.add_argument("--targets", type=str, default=None, metavar="A-B-C,D-E-F",
                         help="With --closure: name the targets instead of drawing them.")
    sampler.add_argument("--n-targets", type=int, default=50,
                         help="With --closure: how many targets to draw from the prior.")
    sampler.add_argument("--target-arity", type=int, default=3)
    sampler.add_argument("--closure-min-arity", type=int, default=2)
    sampler.add_argument("--target-share", type=float, default=0.5,
                         help="Share of each target's rows spent on the target itself; "
                              "the rest is split evenly over its subsystems.")


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

    _add_source_arguments(run, allow_genes=True)
    _add_sampler_arguments(run)

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
    energy.add_argument("--hull-margin", type=float, default=None,
                        help="Keep genes whose predicted e_hull is at most this, in "
                             "eV/atom. Defaults to 0, 'predicted on or below the hull', "
                             "except under --energy-select rank, where it defaults to "
                             "no ceiling.")
    energy.add_argument("--energy-select",
                        choices=builtin.PredictedHullFilter.SELECTORS, default="threshold",
                        help="'threshold' keeps every gene at or below --hull-margin; "
                             "'top' keeps the --energy-top lowest. Use 'top' when the "
                             "reconstruction budget is what is fixed: at a threshold of "
                             "0 only a few percent of novel genes get through, so "
                             "filling a budget from it takes tens of thousands of draws. "
                             "'rank' leaves the cut to --target-engaged, taken after "
                             "every filter: the one to use when a screen runs after "
                             "the energy slot.")
    energy.add_argument("--energy-hull", choices=builtin.PredictedHullFilter.HULLS,
                        default="reference",
                        help="'reference': each gene against the DFT hull alone. "
                             "'joint': the candidates join the hull, and each is scored "
                             "against the hull of the reference and every other one.")
    energy.add_argument("--energy-basis", choices=builtin.PredictedHullFilter.ENERGIES,
                        default="raw",
                        help="'corrected' uses DFT for genes the archive holds and "
                             "subtracts the regressor's local residual from the rest; "
                             "needs --residuals.")
    energy.add_argument("--residuals", type=Path, default=None,
                        help="Directory written by wyckoff_transformer.gene_energy_residuals. "
                             "Given, the corrected variants are written as columns even "
                             "when the selection uses the raw energy.")
    energy.add_argument("--allow-incompatible-energy", action="store_true",
                        help="Range with a regressor whose target is not the hull "
                             "reference's formation energy, or residuals measured on another "
                             "regressor's target. The differences are logged.")
    energy.add_argument("--residual-kappa", type=float, default=None,
                        help="Force the shrinkage; default is the validated one, or no "
                             "correction if validation found it does not help.")
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

    draw = subparsers.add_parser(
        "draw", help="Draw one pool, and the plan it was aimed with, for several arms.")
    draw.add_argument("--output-dir", type=Path, required=True)
    draw.add_argument("--n-genes", type=int, required=True)
    draw.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    _add_source_arguments(draw, allow_genes=False)
    _add_sampler_arguments(draw)
    draw.set_defaults(func=_draw)

    upload = subparsers.add_parser(
        "upload", help="Log a campaign's pool, residuals and arms to one W&B run.")
    upload.add_argument("root", type=Path)
    upload.add_argument("--name", required=True, help="W&B run name.")
    upload.add_argument("--run-id", default=None, help="Resume this W&B run id.")
    upload.add_argument("--artifact", default=None, help="Defaults to --name.")
    upload.add_argument("--arms", nargs="+", required=True)
    upload.add_argument("--config", action="append", metavar="KEY=VALUE")
    upload.add_argument("--wandb-entity", type=str, default=WANDB_ENTITY)
    upload.add_argument("--wandb-project", type=str, default=WANDB_PROJECT)
    upload.set_defaults(func=_upload)
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
