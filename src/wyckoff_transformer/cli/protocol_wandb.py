"""``wyformer-protocol-wandb``: run the de novo ranking protocol on a W&B run.

Takes a run id, generates a fresh gene cohort from its checkpoint exactly as
``wyformer-generate`` does with no conditioning, runs the full
screen -> generate -> relax -> score cascade of
:mod:`wyckoff_transformer.cli.protocol`, and then writes the funnel metrics into
the run's summary and every protocol output into one versioned artifact.

    uv run wyformer-protocol-wandb <run-id> \\
        --output-dir generated/<run-id>/protocol \\
        --condition energy_above_hull=0 \\
        --devices cuda:0,cuda:1 --workers-per-device 2

The cohort is *generated*, not read: 1000 genes by default (``--n-genes``),
sampled from the run's saved space-group distribution.  The gene file is written
into ``--output-dir`` and shipped in the artifact, so the exact set a run was
scored on stays recoverable.  A conditional run needs its target passed with
``--condition NAME=VALUE`` (datasets are not loaded, so it cannot be sampled
from training data); an unconditional run takes neither flag.

Every key of ``funnel.json`` is flattened into ``run.summary`` under a
``protocol/`` prefix.  ``screen.json``, the generated draws in
``pyxtal.extxyz``, the per-trial ``pyxtal.csv`` and ``relaxations.csv``,
``structures.csv``, ``funnel.json``,
``manifest.json``, the generated gene file and ``cifs/`` go into an artifact
named ``protocol_<run-id>`` of type ``protocol_eval``.  ``--no-upload`` runs
everything and skips only the write-back.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Optional

import torch

from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT, wandb_run_path
from wyckoff_transformer.cli import describe_condition, resolve_condition_values
from wyckoff_transformer.cli import protocol as protocol_cli
from wyckoff_transformer.cli.csp import load_trainer
from wyckoff_transformer.evaluation.hull_mlips import DEFAULT_HULL_MLIP, HULL_MLIPS
from wyckoff_transformer.evaluation.protocol import (
    DEFAULT_REFERENCE_CACHE,
    DEFAULT_REFERENCE_SPLITS,
    DEFAULT_TRIAL_SCHEDULE,
)

logger = logging.getLogger(__name__)

#: Files ``load_trainer`` and dataset-free generation need under ``runs/<id>/``.
#: Downloaded from the run when absent.
REQUIRED_RUN_FILES = (
    "best_model_params.pt",
    "wyckoff_processor.json",
    "spacegroup_distribution.json",
)

GENES_FILE = "wyckoff_genes.json.gz"
ARTIFACT_TYPE = "protocol_eval"
SUMMARY_PREFIX = "protocol/"


def ensure_run_files(run, run_dir: Path) -> None:
    """Make sure every file in :data:`REQUIRED_RUN_FILES` sits in *run_dir*.

    ``load_trainer`` reads the checkpoint, processor and space-group
    distribution straight off disk under ``runs/<id>/`` -- the sibling CLIs
    assume they are already there.  Here the only input is a run id, so the
    missing ones are pulled from the run.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_RUN_FILES:
        target = run_dir / name
        if target.is_file():
            continue
        try:
            logger.info("Downloading %s -> %s", name, target)
            run.file(name).download(root=str(run_dir), replace=True)
        except Exception as exc:  # noqa: BLE001 - surface a usable message
            for artifact in reversed(list(run.logged_artifacts())):
                if name not in {artifact_file.name for artifact_file in artifact.files()}:
                    continue
                logger.info(
                    "Downloading %s from artifact %s -> %s",
                    name,
                    artifact.name,
                    target,
                )
                artifact.download(root=str(run_dir))
                break
            else:
                raise FileNotFoundError(
                    f"Run {run.id} has no {name!r} to download ({exc}). Put the "
                    f"model files in {run_dir}/ by hand and re-run."
                ) from exc


def generate_genes(
    run_id: str,
    entity: str,
    project: str,
    n_genes: int,
    oversample: float,
    device: torch.device,
    output_path: Path,
    condition: Optional[list] = None,
    condition_value: Optional[float] = None,
) -> int:
    """Generate a gene cohort from the run's checkpoint and write it to disk.

    Mirrors ``wyformer-generate`` with no element constraints: start tokens are
    sampled from the run's saved space-group distribution, and the formally
    valid genes are truncated to *n_genes*.

    A conditional run needs its target passed explicitly via *condition*
    (``["energy_above_hull=0"]``) or *condition_value*: datasets are not loaded
    here, so the conditioning cannot be sampled from the training distribution
    the way ``wyformer-generate --use-cached-tensors`` does.

    Returns:
        The number of genes written (always *n_genes* on success).
    """
    import wandb  # noqa: PLC0415

    run = wandb.Api().run(wandb_run_path(run_id, entity, project))
    ensure_run_files(run, Path.cwd() / "runs" / run_id)

    trainer = load_trainer(
        device=device,
        wandb_run=run_id,
        wandb_entity=entity,
        wandb_project=project,
        load_datasets=False,
    )
    attempted = max(n_genes + 1, int(round(n_genes * oversample)))

    condition_values = resolve_condition_values(trainer, condition, condition_value)
    cond = None
    if condition_values is not None:
        cond = trainer.build_condition_from_values(
            condition_values, attempted, device=device
        )
        logger.info("Conditioning generation on %s", describe_condition(condition_values))
    elif trainer.condition_features:
        raise ValueError(
            f"Run {run_id} conditions on {list(trainer.condition_features)}; pass "
            "--condition NAME=VALUE (e.g. --condition energy_above_hull=0). Datasets "
            "are not loaded here, so the conditioning cannot be sampled from training data."
        )

    logger.info("Generating %d genes (%d attempted) from run %s", n_genes, attempted, run_id)
    generated = trainer.generate_structures(
        n_structures=attempted, calibrate=False, cond=cond
    )
    if len(generated) < n_genes:
        raise ValueError(
            f"Only {len(generated)} of {attempted} generated genes are formally "
            f"valid; need {n_genes}. Raise --oversample."
        )
    generated = generated[:n_genes]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        json.dump(generated, handle)
    logger.info("Wrote %d genes -> %s", len(generated), output_path)
    return len(generated)


def flatten_funnel(funnel: dict) -> dict:
    """Numeric leaves of ``funnel.json``, keyed for ``run.summary``.

    The funnel is already a flat dict of scalars; this just drops the ``None``
    entries a partial run leaves and prefixes the rest.
    """
    return {
        f"{SUMMARY_PREFIX}{key}": value
        for key, value in funnel.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def build_stage_args(args, gene_file: Path) -> Namespace:
    """An argparse-style object the :mod:`~wyckoff_transformer.cli.protocol` stages accept."""
    return Namespace(
        input=gene_file,
        output_dir=args.output_dir,
        mlip=args.mlip,
        cores=args.cores,
        devices=args.devices,
        workers_per_device=args.workers_per_device,
        n_trials=args.n_trials,
        pyxtal_cores=args.pyxtal_cores,
        pyxtal_timeout=args.pyxtal_timeout,
        fmax=args.fmax,
        relax_timeout=args.relax_timeout,
        release_symmetry=args.release_symmetry,
        rattle=args.rattle,
        limit=args.limit,
        resume=args.resume,
        reference_cache=args.reference_cache,
        reference_splits=args.reference_splits,
        reference_fingerprint_cache=args.reference_fingerprint_cache,
        lemat_cif_csv=args.lemat_cif_csv,
        # Read by the stages when they configure logging in their pool workers.
        debug=args.debug,
    )


def upload(args, gene_file: Path, funnel: dict) -> None:
    """Write the funnel into the run's summary and the outputs into one artifact."""
    import wandb  # noqa: PLC0415

    out = args.output_dir
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        id=args.wandb_run,
        resume="must",
    )
    summary = flatten_funnel(funnel)
    try:
        run.summary.update(summary)

        manifest_path = out / protocol_cli.MANIFEST_FILE
        metadata = {}
        if manifest_path.is_file():
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact = wandb.Artifact(
            name=f"protocol_{args.wandb_run}",
            type=ARTIFACT_TYPE,
            metadata=metadata,
        )
        for name in (
            GENES_FILE,
            protocol_cli.SCREEN_FILE,
            # The draws themselves, not just their outcome: with these a single
            # trial's relaxation can be repeated exactly, which the kept CIF
            # and the per-trial row alone do not allow.
            protocol_cli.PYXTAL_FILE,
            protocol_cli.PYXTAL_TRIALS_FILE,
            protocol_cli.RELAXATIONS_FILE,
            protocol_cli.STRUCTURES_FILE,
            protocol_cli.FUNNEL_FILE,
            protocol_cli.MANIFEST_FILE,
        ):
            path = out / name
            if path.is_file():
                artifact.add_file(str(path), name=name)
        cif_dir = out / protocol_cli.CIF_DIR
        if cif_dir.is_dir():
            artifact.add_dir(str(cif_dir), name=protocol_cli.CIF_DIR)
        run.log_artifact(artifact)
    finally:
        run.finish()
    logger.info("Logged %d summary metrics and artifact protocol_%s",
                len(summary), args.wandb_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wyformer-protocol-wandb",
        description=(
            "Generate a gene cohort from a W&B run, rank it with the de novo "
            "protocol, and log the funnel metrics and outputs back to the run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("wandb_run", type=str, help="W&B run id to evaluate.")
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory for the gene file and every stage output.",
    )
    parser.add_argument("--wandb-entity", type=str, default=WANDB_ENTITY,
                        help="Entity the run was logged under.")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--no-upload", dest="upload", action="store_false",
                        help="Run everything but skip the write-back to W&B.")

    gen = parser.add_argument_group("generation")
    gen.add_argument("--n-genes", type=int, default=1000,
                     help="Genes to generate and score. The default is the MetaSUN cohort size.")
    gen.add_argument("--oversample", type=float, default=1.15,
                     help="Generate this multiple of --n-genes, then keep the first n valid.")
    gen.add_argument("--gen-device", type=torch.device, default=torch.device("cpu"),
                     help="Device for generation. Relaxation hardware is set separately below.")
    gen.add_argument("--skip-generate", action="store_true",
                     help="Reuse an existing gene file in --output-dir instead of generating.")
    gen.add_argument("--condition", action="append", metavar="NAME=VALUE", default=None,
                     help="Conditioning target for every generated gene, e.g. "
                          "--condition energy_above_hull=0. Repeat once per feature. "
                          "Required for a conditional run: datasets are not loaded here, "
                          "so the conditioning cannot be sampled from training data.")
    gen.add_argument("--condition-value", type=float, default=None,
                     help="Shorthand for --condition <the one feature>=VALUE, for a "
                          "single-channel conditional model.")

    pyxtal = parser.add_argument_group("PyXtal generation")
    pyxtal.add_argument("--pyxtal-cores", type=int, default=None,
                        help="CPU processes drawing PyXtal structures. Defaults to every core.")
    pyxtal.add_argument("--pyxtal-timeout", type=float, default=300.0,
                        help="Seconds one PyXtal draw may take before it is abandoned.")

    hardware = parser.add_argument_group("relaxation hardware")
    hardware.add_argument("--cores", type=int, default=None,
                          help="Run relaxation on CPU with this many workers.")
    hardware.add_argument("--devices", type=str, default=None,
                          help="Run relaxation on GPU: e.g. 'cuda:0,cuda:1'.")
    hardware.add_argument("--workers-per-device", type=int, default=1)

    relax = parser.add_argument_group("relaxation")
    relax.add_argument("--mlip", type=str, default=DEFAULT_HULL_MLIP, choices=sorted(HULL_MLIPS))
    relax.add_argument("--n-trials", type=str, default=DEFAULT_TRIAL_SCHEDULE)
    relax.add_argument("--fmax", type=float, default=0.05)
    relax.add_argument("--relax-timeout", type=float, default=1800.0,
                       help="Seconds one trial's four-stage relaxation may take.")
    relax.add_argument("--release-symmetry", action=argparse.BooleanOptionalAction, default=True)
    relax.add_argument("--rattle", action=argparse.BooleanOptionalAction, default=True)
    relax.add_argument("--limit", type=int, default=None, help="Only relax genes below this index.")
    relax.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                       help="Keep the trials the generate and relax logs already recorded "
                            "and do only the rest. --no-resume starts both from scratch.")

    reference = parser.add_argument_group("references")
    reference.add_argument("--reference-cache", type=Path, default=DEFAULT_REFERENCE_CACHE)
    reference.add_argument("--reference-splits", type=str, default=",".join(DEFAULT_REFERENCE_SPLITS))
    reference.add_argument(
        "--reference-fingerprint-cache", type=Path,
        default=Path("cache/lemat_bulk_ehull/gene_fingerprints.pkl.gz"),
    )
    reference.add_argument(
        "--lemat-cif-csv", type=Path, default=Path("data/lemat-bulk/lemat_pbe.csv.gz"),
    )

    parser.add_argument("--debug", action="store_true", help="DEBUG-level logging.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gene_file = args.output_dir / GENES_FILE

    if args.skip_generate:
        if not gene_file.is_file():
            raise FileNotFoundError(f"--skip-generate but no gene file at {gene_file}")
        logger.info("Reusing gene file %s", gene_file)
    else:
        generate_genes(
            run_id=args.wandb_run,
            entity=args.wandb_entity,
            project=args.wandb_project,
            n_genes=args.n_genes,
            oversample=args.oversample,
            device=args.gen_device,
            output_path=gene_file,
            condition=args.condition,
            condition_value=args.condition_value,
        )

    stage_args = build_stage_args(args, gene_file)
    for stage in protocol_cli.STAGES:
        protocol_cli.run_stage(stage, stage_args)

    funnel = json.loads(
        (args.output_dir / protocol_cli.FUNNEL_FILE).read_text(encoding="utf-8")
    )
    if args.upload:
        upload(args, gene_file, funnel)
    else:
        logger.info("--no-upload: skipping W&B write-back")
    print(json.dumps(flatten_funnel(funnel), indent=2))


if __name__ == "__main__":
    main()
