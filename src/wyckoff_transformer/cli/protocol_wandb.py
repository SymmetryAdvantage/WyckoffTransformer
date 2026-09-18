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
from training data); an unconditional run takes neither flag. Conditions
in ``{energy_above_hull, delta_e_polymorph, max_force}`` default to 0 if required
by the model and omitted.

Every key of ``funnel.json`` is flattened into ``run.summary`` under a
``protocol/`` prefix.  ``screen.json``, the generated draws in
``pyxtal.extxyz``, the per-trial ``pyxtal.csv`` and ``relaxations.csv``,
``structures.csv``, ``funnel.json``,
``manifest.json``, the generated gene file and ``cifs/`` go into an artifact
named ``protocol_<run-id>`` of type ``protocol_eval``.  ``--no-upload`` runs
everything and skips only the write-back.

To re-score a run whose relaxations are already done -- e.g. after a change to
how novelty is judged -- pass ``--from-artifact --stages score``: the previous
``protocol_<run-id>`` artifact is downloaded into ``--output-dir`` and only the
score stage runs, then the refreshed ``funnel.json`` and ``structures.csv`` go
back as a new artifact version and ``run.summary`` is overwritten.  After a
change of *reference* it is ``--stages screen,score``: gene novelty is judged in
the screen, and ``score`` refuses a screen judged against another reference.  A
re-score that fails uploads nothing.
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

from wyckoff_transformer.paths import runs_root, wandb_dir
from wyckoff_transformer.system_prior import SYSTEM_PRIOR_FILE_NAME
from wyckoff_transformer.tokenization import WYCKOFF_MAPPINGS_FILENAME
from wyckoff_transformer.wyckoff_processor import MODEL_ENGINEERS_DIRNAME
from wyckoff_transformer import WANDB_ENTITY, WANDB_PROJECT, wandb_run_path
from wyckoff_transformer.cli import (
    DEFAULT_CONDITION_TARGETS,
    DEFAULT_SWEEP_FEATURE,
    describe_condition,
    resolve_condition_values,
)
from wyckoff_transformer.cli import protocol as protocol_cli
from wyckoff_transformer.cryspr.basin_hopping import (
    BASINHOP_STDEV,
    BASINHOP_STRAIN_STDEV,
    DEFAULT_STEPS as BASINHOP_STEPS,
    DEFAULT_TEMPERATURE_EV_PER_ATOM as BASINHOP_TEMPERATURE,
)
from wyckoff_transformer.cryspr.generator import DEFAULT_PYXTAL_TOL_FACTOR, PRERELAX_FMAX
from wyckoff_transformer.cryspr.mlips import prerelax_mlip_names
from wyckoff_transformer.cryspr.prescreen import DEDUP_ENERGY_TOL_EV_PER_ATOM
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
    WYCKOFF_MAPPINGS_FILENAME,
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
    ensure_run_engineers(run, run_dir)


def ensure_system_prior(run, run_dir: Path) -> Optional[Path]:
    """The run's own ``system_prior.npz``, fetched from W&B if it is not already local.

    Optional, unlike :data:`REQUIRED_RUN_FILES`: only a `chemical_system_conditioning` run
    has one, and runs trained before `WyckoffTrainer.save_system_prior` existed carry none
    at all. Hence None rather than an exception -- the caller falls back to the cache and
    says what to do about it.

    The run's files are tried first and the artifacts second, the same order and for the
    same reason as `ensure_run_files`: walking every artifact of a chained run means
    listing hundreds of checkpoints.
    """
    target = run_dir / SYSTEM_PRIOR_FILE_NAME
    if target.is_file():
        return target
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("Downloading %s -> %s", SYSTEM_PRIOR_FILE_NAME, target)
        run.file(SYSTEM_PRIOR_FILE_NAME).download(root=str(run_dir), replace=True)
        return target
    except Exception as exc:  # noqa: BLE001 - absence is one of the expected answers
        logger.info("Run %s has no %s among its files (%s); trying its artifacts",
                    run.id, SYSTEM_PRIOR_FILE_NAME, exc)
    for artifact in reversed(list(run.logged_artifacts())):
        if SYSTEM_PRIOR_FILE_NAME not in {file.name for file in artifact.files()}:
            continue
        logger.info("Downloading %s from artifact %s -> %s",
                    SYSTEM_PRIOR_FILE_NAME, artifact.name, run_dir)
        artifact.download(root=str(run_dir))
        if target.is_file():
            return target
    return None


def ensure_run_engineers(run, run_dir: Path) -> None:
    """Fetch the run's own ``engineers/`` from its processors artifact, if it has one.

    Runs trained before models carried their engineers have none; loading those falls
    back to the package's, which is what they were trained with unless it has changed.
    """
    if (run_dir / MODEL_ENGINEERS_DIRNAME).is_dir():
        return
    prefix = f"{MODEL_ENGINEERS_DIRNAME}/"
    for artifact in run.logged_artifacts():
        if artifact.type != "processors":
            continue
        if any(artifact_file.name.startswith(prefix) for artifact_file in artifact.files()):
            logger.info("Downloading %s from artifact %s -> %s", prefix, artifact.name, run_dir)
            artifact.download(root=str(run_dir))
            return
    logger.warning(
        "Run %s has no %s; it predates models carrying their engineers, so the "
        "package's are used", run.id, prefix)


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
    system_prior: Optional[Path] = None,
    temperature: float = 1.0,
    manifest_path: Optional[Path] = None,
    allow_fewer: bool = False,
) -> int:
    """Generate a gene cohort from the run's checkpoint and write it to disk.

    Mirrors ``wyformer-generate`` with no element constraints: start tokens are
    sampled from the run's saved space-group distribution, and the formally
    valid genes are truncated to *n_genes*.

    *temperature* rescales every generated cascade field's logits; the start
    token is drawn from the run's saved space-group distribution regardless, so
    the space-group marginal is the same at every temperature.

    *manifest_path* receives what only this step knows: the temperature the
    cohort was drawn at, and how much of the raw draw was formally valid.  The
    kept cohort is truncated to *n_genes*, so the rejected fraction is
    unrecoverable from the gene file afterwards -- and it moves with the
    temperature, which is exactly what a sweep needs to be able to see.

    A conditional run uses the targets passed via *condition*
    (``["energy_above_hull=0"]``) or *condition_value*. Any unspecified
    features among ``energy_above_hull``, ``delta_e_polymorph``, and
    ``max_force`` default to 0.0. Other condition features must be passed
    explicitly because datasets are not loaded here.

    Returns:
        The number of genes written (always *n_genes* on success, or fewer if *allow_fewer* is True).
    """
    import wandb  # noqa: PLC0415

    run = wandb.Api().run(wandb_run_path(run_id, entity, project))
    ensure_run_files(run, runs_root() / run_id)

    trainer = load_trainer(
        device=device,
        wandb_run=run_id,
        wandb_entity=entity,
        wandb_project=project,
        load_datasets=False,
    )
    if hasattr(trainer.model, "_orig_mod"):
        trainer.model = trainer.model._orig_mod

    attempted = max(n_genes + 1, int(round(n_genes * oversample)))

    if condition_value is not None and condition is None:
        if (
            trainer.condition_features
            and len(trainer.condition_features) > 1
            and DEFAULT_SWEEP_FEATURE in trainer.condition_features
        ):
            condition = [f"{DEFAULT_SWEEP_FEATURE}={condition_value}"]
            condition_value = None

    condition_values = resolve_condition_values(trainer, condition, condition_value)
    if trainer.condition_features:
        if condition_values is None:
            condition_values = {}
        for feature in trainer.condition_features:
            if feature not in condition_values and feature in DEFAULT_CONDITION_TARGETS:
                condition_values[feature] = DEFAULT_CONDITION_TARGETS[feature]

        missing = [f for f in trainer.condition_features if f not in condition_values]
        if missing:
            raise ValueError(
                f"Run {run_id} conditions on {list(trainer.condition_features)}; pass "
                f"--condition NAME=VALUE for {missing} (e.g. --condition {missing[0]}=0). "
                "Datasets are not loaded here, so the conditioning cannot be sampled from training data."
            )
        ordered = {f: condition_values[f] for f in trainer.condition_features if f in condition_values}
        ordered.update({k: v for k, v in condition_values.items() if k not in ordered})
        condition_values = ordered

    cond = None
    if condition_values:
        cond = trainer.build_condition_from_values(
            condition_values, attempted, device=device
        )
        logger.info("Conditioning generation on %s", describe_condition(condition_values))

    start_tensor = None
    composition_cond = None
    element_mask = None
    if getattr(trainer, "chemical_system_conditioning", None) is True:
        from wyckoff_transformer.paths import cache_root
        from wyckoff_transformer.system_prior import SystemSpaceGroupPrior

        dataset_name = run.config.get("dataset")
        prior_path = system_prior
        if prior_path is None:
            # The run's own, which is the only one guaranteed to share the checkpoint's
            # element tokens. Runs trained before it was saved have none, so the cache the
            # run trained on is the fallback -- and that only works on a machine that has it.
            prior_path = ensure_system_prior(run, runs_root() / run_id)
        if prior_path is None and dataset_name:
            candidate = cache_root() / dataset_name / SYSTEM_PRIOR_FILE_NAME
            if candidate.is_file():
                prior_path = candidate
        if prior_path is None or not Path(prior_path).is_file():
            raise ValueError(
                f"Run {run_id} uses chemical_system_conditioning, but carries no "
                f"{SYSTEM_PRIOR_FILE_NAME} and none is cached for {dataset_name!r}. Build "
                f"one from the tensor cache the run trained on -- `wyformer-system-prior "
                f"build {dataset_name}` -- and pass it as --system-prior."
            )
        logger.info("Sampling chemical systems and space groups from %s", prior_path)
        prior = SystemSpaceGroupPrior.load(prior_path)
        elements_tokeniser = trainer.tokenisers["elements"]
        # The same guard `wyformer-generate` applies, and it matters more here: the prior
        # may have come from the cache rather than from the run, and a prior built over a
        # different vocabulary would decode every system into the wrong elements silently.
        vocabulary = [str(symbol) for symbol in elements_tokeniser.to_token]
        if list(prior.element_symbols) != vocabulary:
            raise ValueError(
                f"The prior at {prior_path} was built over {prior.n_elements} element tokens "
                f"and run {run_id} knows {len(vocabulary)}; they have to come from the same "
                "dataset, or a system would decode into different elements.")
        draws = prior.sample(attempted, required=None, allowed=None)
        composition_cond = draws.conditioning_block(len(elements_tokeniser), device=device)
        start_tensor = draws.start_tensor(
            trainer.tokenisers[trainer.start_name], trainer.model.start_type, device=device
        )
        element_mask = draws.element_mask(
            len(elements_tokeniser), stop_token=elements_tokeniser.stop_token, device=device
        )

    logger.info("Generating %d genes (%d attempted) from run %s at T=%g",
                n_genes, attempted, run_id, temperature)
    generated = trainer.generate_structures(
        n_structures=attempted,
        calibrate=False,
        cond=cond,
        composition_cond=composition_cond,
        start_tensor=start_tensor,
        allowed_element_mask=element_mask,
        temperature=temperature,
    )
    if len(generated) < n_genes:
        if allow_fewer and len(generated) > 0:
            logger.warning(
                "Only %d of %d generated genes are formally valid; proceeding with %d "
                "genes (--allow-fewer).",
                len(generated), attempted, len(generated),
            )
        else:
            raise ValueError(
                f"Only {len(generated)} of {attempted} generated genes are formally "
                f"valid; need {n_genes}. Raise --oversample."
            )
    if manifest_path is not None:
        from wyckoff_transformer.cli.protocol import update_manifest  # noqa: PLC0415

        update_manifest(manifest_path, {
            "sampling_temperature": temperature,
            "generation_attempted": attempted,
            "generation_formally_valid": len(generated),
            "formal_gene_validity": round(len(generated) / attempted, 4),
        })
    generated = generated[:n_genes]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        json.dump(generated, handle)
    logger.info("Wrote %d genes -> %s", len(generated), output_path)
    return len(generated)


def download_protocol_artifact(
    run_id: str, entity: str, project: str, output_dir: Path, version: str = "latest"
) -> str:
    """Pull a run's ``protocol_<id>`` artifact into *output_dir*.

    Used by ``--from-artifact`` to re-score an already-relaxed run: the stage
    outputs (``screen.json``, ``structures.csv``, ``cifs/``, the gene file) are
    read straight back, so no cohort is generated and nothing is relaxed.

    Returns:
        The concrete artifact version that was downloaded (e.g. ``"v1"``).
    """
    import wandb  # noqa: PLC0415

    name = f"protocol_{run_id}"
    artifact = wandb.Api().artifact(
        f"{entity}/{project}/{name}:{version}", type=ARTIFACT_TYPE
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact.download(root=str(output_dir))
    logger.info("Downloaded %s:%s -> %s", name, artifact.version, output_dir)
    return artifact.version


def flatten_funnel(funnel: dict, prefix: str = SUMMARY_PREFIX) -> dict:
    """Numeric leaves of ``funnel.json``, keyed hierarchically for ``run.summary``.

    Traverses hierarchical sections (e.g. ``protocol/gene/``,
    ``protocol/fixed_symmetry/``, ``protocol/free/``) and flattens them into
    summary keys, dropping ``None`` and boolean entries.
    """
    out = {}
    for key, value in funnel.items():
        if isinstance(value, dict):
            sub = flatten_funnel(value, prefix=f"{prefix}{key}/")
            out.update(sub)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[f"{prefix}{key}"] = value
    return out


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
        trial_multiplier=args.trial_multiplier,
        pyxtal_cores=args.pyxtal_cores,
        pyxtal_timeout=args.pyxtal_timeout,
        pyxtal_tol_factor=args.pyxtal_tol_factor,
        fmax=args.fmax,
        relax_timeout=args.relax_timeout,
        release_symmetry=args.release_symmetry,
        rattle=args.rattle,
        relax_from=args.relax_from,
        prerelax_mlip=args.prerelax_mlip,
        prerelax_fmax=args.prerelax_fmax,
        prerelax_max_expansion=args.prerelax_max_expansion,
        prescreen_mlip=args.prescreen_mlip,
        prescreen_fmax=args.prescreen_fmax,
        prescreen_dedup=args.prescreen_dedup,
        prescreen_energy_tol=args.prescreen_energy_tol,
        prescreen_release_symmetry=args.prescreen_release_symmetry,
        prescreen_rattle=args.prescreen_rattle,
        prescreen_select=args.prescreen_select,
        basinhop_mlip=args.basinhop_mlip,
        basinhop_steps=args.basinhop_steps,
        basinhop_temperature=args.basinhop_temperature,
        basinhop_stdev=args.basinhop_stdev,
        basinhop_strain_stdev=args.basinhop_strain_stdev,
        # The optional stages' own arguments, so that --stages can name them.
        template_index=args.template_index,
        template_candidates=args.template_candidates,
        limit=args.limit,
        resume=args.resume,
        retry_failed=args.retry_failed,
        allow_incomplete=args.allow_incomplete,
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
        dir=wandb_dir(),
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
        if args.from_artifact is not None:
            metadata["rescored_from"] = f"protocol_{args.wandb_run}:{args.from_artifact}"
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
            # Only present in a wide-then-narrow run; add_file is guarded on
            # is_file() below, so a single-stage run simply ships neither.
            protocol_cli.PRESCREEN_TRIALS_FILE,
            protocol_cli.PRESCREEN_SELECTION_FILE,
            protocol_cli.STRUCTURES_FILE,
            protocol_cli.STRUCTURES_FIXED_FILE,
            protocol_cli.FUNNEL_FILE,
            protocol_cli.MANIFEST_FILE,
        ):
            path = out / name
            if path.is_file():
                artifact.add_file(str(path), name=name)
        cif_dir = out / protocol_cli.CIF_DIR
        if cif_dir.is_dir():
            artifact.add_dir(str(cif_dir), name=protocol_cli.CIF_DIR)
        cif_fixed_dir = out / protocol_cli.CIF_FIXED_DIR
        if cif_fixed_dir.is_dir():
            artifact.add_dir(str(cif_fixed_dir), name=protocol_cli.CIF_FIXED_DIR)
        run.log_artifact(artifact)
    finally:
        run.finish()

    try:
        api_run = wandb.Api().run(wandb_run_path(args.wandb_run, args.wandb_entity, args.wandb_project))
        for k in list(api_run.summary.keys()):
            if k.startswith("protocol/") and not k.startswith(("protocol/gene/", "protocol/fixed_symmetry/", "protocol/free/")):
                del api_run.summary[k]
        for k, v in summary.items():
            api_run.summary[k] = v
        api_run.summary.update()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to sync summary via wandb Api: %s", exc)

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
    parser.add_argument(
        "--stages", type=str, default=",".join(protocol_cli.STAGES),
        help=(
            "Comma-separated subset of %s to run, in this order. The optional "
            "stages %s are accepted but not in the default: 'template' adds one "
            "template-matched start per gene, 'prescreen' narrows a widened draw "
            "down to the schedule's usual count on a cheap potential."
            % (",".join(protocol_cli.STAGES), ",".join(protocol_cli.OPTIONAL_STAGES))
        ),
    )
    parser.add_argument(
        "--from-artifact", nargs="?", const="latest", default=None, metavar="VERSION",
        help="Re-score an already-relaxed run: download its protocol_<id> "
             "artifact (this VERSION, or the latest) into --output-dir and run "
             "only the stages named by --stages (use --stages score, or "
             "--stages screen,score when the reference has changed). Implies "
             "--skip-generate; nothing is generated or relaxed.",
    )

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
                          "Features energy_above_hull, delta_e_polymorph, and max_force "
                          "default to 0 if not specified.")
    gen.add_argument("--condition-value", type=float, default=None,
                     help="Shorthand for --condition <the one feature>=VALUE (or "
                          "energy_above_hull=VALUE on multi-channel models).")
    gen.add_argument("--system-prior", type=Path, default=None,
                     help="Path to a system_prior.npz. For chemical_system_conditioning models, "
                          "defaults to the one the run carries in its W&B files, and failing "
                          "that to cache/<dataset>/system_prior.npz.")
    gen.add_argument("--temperature", type=float, default=1.0,
                     help="Softmax temperature for every generated cascade field. Below 1 "
                          "sharpens the sampler, above 1 flattens it. Recorded in "
                          "manifest.json as sampling_temperature.")
    gen.add_argument("--allow-fewer", action="store_true",
                     help="Keep whatever valid genes were generated if fewer than --n-genes (e.g. for checkpoints early in training).")

    pyxtal = parser.add_argument_group("PyXtal generation")
    pyxtal.add_argument("--pyxtal-cores", type=int, default=None,
                        help="CPU processes drawing PyXtal structures. Defaults to every core.")
    pyxtal.add_argument("--pyxtal-timeout", type=float, default=300.0,
                        help="Seconds one PyXtal draw may take before it is abandoned.")
    pyxtal.add_argument("--pyxtal-tol-factor", type=float, default=DEFAULT_PYXTAL_TOL_FACTOR,
                        help="Scale on PyXtal's inter-atomic distance floor, 0.5*(r_a+r_b) "
                             "times this. Lower is more permissive; 1.3 is what every "
                             "published number was measured with.")

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
    relax.add_argument("--relax-timeout", type=float, default=300.0,
                       help="Seconds one trial's four-stage relaxation may take.")
    relax.add_argument("--release-symmetry", action=argparse.BooleanOptionalAction, default=True)
    relax.add_argument("--rattle", action=argparse.BooleanOptionalAction, default=True)
    relax.add_argument(
        "--relax-from", type=str, default="pyxtal",
        choices=sorted(protocol_cli.RELAX_SOURCES),
        help="Starting structures: what 'generate' wrote, or what 'prescreen' selected.",
    )
    relax.add_argument(
        "--prerelax-mlip", type=str, default=None, choices=prerelax_mlip_names(),
        help="Relax every trial on this cheap potential before --mlip. Off by default.",
    )
    relax.add_argument(
        "--prerelax-fmax", type=float, default=PRERELAX_FMAX,
        help="Force convergence of the pre-relaxation, eV/A.",
    )
    relax.add_argument(
        "--prerelax-max-expansion", type=float, default=None,
        help="Fall back to the raw draw if the pre-relaxation grew the cell by "
             "more than this factor. Unset by default.",
    )
    relax.add_argument(
        "--trial-multiplier", type=int, default=1,
        help="Draw this multiple of the schedule's trials. Pair 10 with --stages "
             "...,prescreen,relax and --relax-from prescreen.",
    )

    prescreen = parser.add_argument_group("wide-then-narrow (stage: prescreen)")
    prescreen.add_argument(
        "--prescreen-mlip", type=str, default=None, choices=prerelax_mlip_names(),
        help="Potential the pre-screen relaxes and ranks with. Defaults to nep89.",
    )
    prescreen.add_argument(
        "--prescreen-fmax", type=float, default=PRERELAX_FMAX,
        help="Force convergence of the pre-screen relaxation, eV/A.",
    )
    prescreen.add_argument(
        "--prescreen-dedup", type=str, default="matcher", choices=("matcher", "energy"),
        help="How two pre-relaxed draws of one gene are judged the same structure.",
    )
    prescreen.add_argument(
        "--prescreen-energy-tol", type=float, default=DEDUP_ENERGY_TOL_EV_PER_ATOM,
        help="Energy gap, eV/atom, above which the matcher is not called.",
    )
    prescreen.add_argument(
        "--prescreen-release-symmetry", action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the unconstrained stage in the pre-screen. With "
             "--prescreen-rattle and --prescreen-select 1 this is the "
             "NEP89-first arm.",
    )
    prescreen.add_argument(
        "--prescreen-rattle", action=argparse.BooleanOptionalAction, default=False,
        help="Run the rattle stage in the pre-screen.",
    )
    prescreen.add_argument(
        "--prescreen-select", type=str, default="dof",
        help="Structures per gene handed to the scoring potential: 'dof' for the "
             "trial schedule's allotment, or an integer.",
    )

    basinhop = parser.add_argument_group("basin hopping (stage: basinhop)")
    basinhop.add_argument(
        "--basinhop-mlip", type=str, default=None, choices=prerelax_mlip_names(),
        help="Potential the walk relaxes and ranks with.",
    )
    basinhop.add_argument("--basinhop-steps", type=int, default=BASINHOP_STEPS,
                          help="Hops per starting draw.")
    basinhop.add_argument("--basinhop-temperature", type=float, default=BASINHOP_TEMPERATURE,
                          help="Metropolis temperature, eV/atom. 0 is downhill only.")
    basinhop.add_argument("--basinhop-stdev", type=float, default=BASINHOP_STDEV,
                          help="Displacement drawn per hop, A, before projection.")
    basinhop.add_argument("--basinhop-strain-stdev", type=float, default=BASINHOP_STRAIN_STDEV,
                          help="Cell strain drawn per hop, before projection.")

    template = parser.add_argument_group("template starts (stage: template)")
    template.add_argument(
        "--template-index", type=Path, default=None,
        help="Parquet of LeMat-Bulk keyed by anonymous Wyckoff fingerprint.",
    )
    template.add_argument(
        "--template-candidates", type=int, default=4,
        help="Templates carried out of the index per gene, closest formula first.",
    )
    relax.add_argument("--limit", type=int, default=None, help="Only relax genes below this index.")
    relax.add_argument("--retry-failed", action="store_true",
                       help="On --resume, re-run failed trials too. A killed-worker "
                            "or failed-GPU row is re-run regardless.")
    relax.add_argument("--allow-incomplete", action="store_true",
                       help="Score although trials are still unanswered because a GPU "
                            "failed or a worker was killed. Refused by default.")
    relax.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                       help="Keep the trials the generate and relax logs already recorded "
                            "and do only the rest. --no-resume starts both from scratch.")

    reference = parser.add_argument_group("references")
    reference.add_argument("--reference-cache", type=Path, default=DEFAULT_REFERENCE_CACHE)
    reference.add_argument("--reference-splits", type=str, default=",".join(DEFAULT_REFERENCE_SPLITS))
    reference.add_argument(
        "--reference-fingerprint-cache", type=Path, default=None,
        help="Defaults to gene_fingerprints.pkl.gz beside --reference-cache.",
    )
    reference.add_argument(
        "--lemat-cif-csv", type=Path, default=Path("data/lemat-bulk/lemat_pbe.csv.gz"),
    )

    parser.add_argument("--debug", action="store_true", help="DEBUG-level logging.")
    return parser


def refuse_to_resample_under_resume(args, gene_file: Path) -> None:
    """Refuse to sample a new gene file over one whose stage logs would be resumed.

    Sampling writes over *gene_file*, and the stages then resume their logs by
    ``(gene index, trial)`` alone, so the old draws and relaxations would be
    scored as the new genes' -- which is what ``protocol_ehull5x-20260904-213346``
    v1 and v2 were.  :func:`protocol.claim_lineage` would refuse at the generate
    stage, but only after the old gene file, the one those logs *can* be resumed
    with, had already been replaced; hence this check, before sampling.

    Raises:
        protocol.StaleOutputError: With ``--resume`` and any stage log present.
    """
    if not args.resume:
        return
    present = [
        name for name in protocol_cli.TRIAL_LOGS
        if protocol_cli.read_rows(args.output_dir / name).shape[0]
    ]
    if not present:
        return
    raise protocol_cli.StaleOutputError(
        f"{args.output_dir} already holds {', '.join(present)}, built from "
        f"{gene_file}; sampling a new gene file would resume them as the new "
        "genes' trials. Pass --skip-generate to resume the existing cohort, or "
        "--no-resume to sample a new one and start every stage over."
    )


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gene_file = args.output_dir / GENES_FILE

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    known = protocol_cli.STAGES + protocol_cli.OPTIONAL_STAGES
    unknown = [s for s in stages if s not in known]
    if unknown:
        raise SystemExit(f"--stages: {unknown} not in {known}")

    if args.from_artifact is not None:
        download_protocol_artifact(
            args.wandb_run, args.wandb_entity, args.wandb_project,
            args.output_dir, version=args.from_artifact,
        )

    if args.skip_generate or args.from_artifact is not None:
        if not gene_file.is_file():
            raise FileNotFoundError(
                f"no gene file at {gene_file} (--skip-generate/--from-artifact)"
            )
        logger.info("Reusing gene file %s", gene_file)
    else:
        refuse_to_resample_under_resume(args, gene_file)
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
            system_prior=args.system_prior,
            temperature=args.temperature,
            manifest_path=args.output_dir / protocol_cli.MANIFEST_FILE,
            allow_fewer=args.allow_fewer,
        )

    stage_args = build_stage_args(args, gene_file)
    stage_exc = None
    try:
        for stage in stages:
            protocol_cli.run_stage(stage, stage_args)
    except (protocol_cli.IncompleteStageError, protocol_cli.StaleOutputError):
        # Nothing is wrong with the run that a partial report would describe:
        # its trials are waiting for --resume, or its outputs are waiting to be
        # rebuilt from one cohort, and uploading now would publish a new
        # artifact version -- and overwrite the run's summary -- for a cohort
        # that is not finished.
        raise
    except Exception as exc:
        if args.from_artifact is not None:
            # The artifact already holds a complete funnel. A partial one built
            # from the re-screen would publish a new version, and overwrite the
            # run's gene metrics, for a re-score that did not happen.
            raise
        stage_exc = exc

    if stage_exc is not None:
        screen_path = args.output_dir / protocol_cli.SCREEN_FILE
        if screen_path.is_file():
            import pandas as pd  # noqa: PLC0415

            try:
                screen = protocol_cli.read_screen(screen_path)
                partial_funnel = protocol_cli.funnel(screen, pd.DataFrame())
                funnel_path = args.output_dir / protocol_cli.FUNNEL_FILE
                funnel_path.write_text(
                    json.dumps(partial_funnel, indent=2) + "\n", encoding="utf-8"
                )
                if args.upload:
                    try:
                        upload(args, gene_file, partial_funnel)
                    except Exception as upload_exc:  # noqa: BLE001
                        logger.warning("Failed to upload partial metrics to W&B: %s", upload_exc)
                else:
                    logger.info("--no-upload: skipping W&B write-back")
                print(json.dumps(flatten_funnel(partial_funnel), indent=2))
            except Exception as report_exc:  # noqa: BLE001
                logger.warning("Failed to report partial metrics: %s", report_exc)
        raise stage_exc

    # Only the score stage writes funnel.json, and --stages need not include it:
    # a run that is only drawing or only pre-screening -- because the next half
    # wants different hardware -- has nothing to report or upload yet, and
    # reading the funnel anyway turns a finished stage into a crash.
    funnel_path = args.output_dir / protocol_cli.FUNNEL_FILE
    if not funnel_path.is_file():
        logger.info(
            "Stages %s produced no %s; nothing to report or upload. "
            "Run --stages score to score what they left.",
            ",".join(stages), protocol_cli.FUNNEL_FILE,
        )
        return

    funnel = json.loads(funnel_path.read_text(encoding="utf-8"))
    if args.upload:
        upload(args, gene_file, funnel)
    else:
        logger.info("--no-upload: skipping W&B write-back")
    print(json.dumps(flatten_funnel(funnel), indent=2))


if __name__ == "__main__":
    main()
