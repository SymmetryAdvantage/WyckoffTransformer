#!/usr/bin/env python3
"""Relax one Matbench MLIP arm from a protocol artifact's raw PyXtal draws.

Run one model per environment. Dependencies of leaderboard MLIPs conflict, so
``--calculator-factory`` can name a factory in that model's own environment.
The factory takes ``device=...`` and returns an ASE calculator. Registered
WyFormer MLIPs need no factory. The output is resumable by (gene, trial).
Use ``--workers`` for concurrent trials of the same arm; the coordinator
alone writes its ledger and W&B snapshots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import logging
import math
import multiprocessing
import os
import shutil
import signal
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ase.io import iread

from wyckoff_transformer.cryspr.generator import (
    KEPT_CIF_SUFFIX,
    KEPT_FIXED_CIF_SUFFIX,
    _trial_seed,
    relax_trial,
)
from wyckoff_transformer.cryspr.mlips import MLIP_REGISTRY, build_prerelax_calculator
from wyckoff_transformer.cryspr.relaxer import _print_logm_warning_once
from wyckoff_transformer.evaluation.hull_mlips import HULL_MLIPS

LOG = logging.getLogger("mlip_bias_relax")
FIELDS = (
    "gene", "trial", "status", "formula", "n_atoms", "energy_ev",
    "energy_ev_per_atom", "fixed_energy_ev", "fixed_energy_ev_per_atom",
    "cif", "fixed_cif", "seconds", "error",
)
DEFAULT_INPUT_ARTIFACT = (
    "symmetry-advantage/WyckoffTransformer/"
    "protocol_ehull_adamw_wsd_5x_cfg-20260924-223159.cfg-grid-e0p05-w5:v0"
)
ARTIFACT_TYPE = "mlip_bias_relax"


@contextmanager
def trial_timeout(seconds: float):
    """Apply the effective per-trial wall-clock timeout on Linux."""
    if seconds <= 0:
        raise ValueError("Relaxation timeout must be positive")

    def timed_out(signum, frame):
        raise TimeoutError(f"Relaxation exceeded {seconds:g} seconds")

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_calculator(name: str, device: str, factory: str | None):
    if factory:
        module_name, sep, function_name = factory.partition(":")
        if not sep or not module_name or not function_name:
            raise ValueError("--calculator-factory must be MODULE:FUNCTION")
        function = getattr(importlib.import_module(module_name), function_name)
        return function(device=device)
    return build_prerelax_calculator(name, device=device)


def load_rows(path: Path) -> dict[tuple[int, int], dict]:
    if not path.exists():
        return {}
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if rows and tuple(rows[0]) != FIELDS:
        raise ValueError(f"Unexpected columns in {path}")
    result = {}
    for row in rows:
        key = int(row["gene"]), int(row["trial"])
        if key in result:
            raise ValueError(f"Duplicate trial {key} in {path}")
        result[key] = row
    return result


def is_oom_or_timeout(row: dict) -> bool:
    """Recognise Torch, cuEquivariance and Warp allocation failures."""
    error = row["error"].lower()
    return row["status"] == "failed" and any(
        text in error for text in (
            "timeouterror", "out of memory", "outofmemoryerror",
            "cudaerrormemoryallocation", "failed to allocate",
        )
    )


def apply_historical_timeout(rows_path: Path, rows: dict, seconds: float) -> int:
    """Discard earlier successes exceeding the effective timeout."""
    changed = 0
    for row in rows.values():
        if row["status"] == "ok" and float(row["seconds"]) > seconds:
            row["status"] = "failed"
            for field in ("energy_ev", "energy_ev_per_atom", "fixed_energy_ev",
                          "fixed_energy_ev_per_atom", "cif", "fixed_cif"):
                row[field] = ""
            row["error"] = f"TimeoutError: trial exceeded {seconds:g} seconds"
            changed += 1
    if changed:
        replacement = rows_path.with_suffix(".csv.tmp")
        with replacement.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows.values())
        os.replace(replacement, rows_path)
    return changed


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def prepare_recovery(output: Path, rows: dict, seconds: float, workers: int) -> dict:
    """Save the retry set so recovery resumes without repeating finished attempts."""
    path = output / "recovery_settings.json"
    saved = json.loads(path.read_text()) if path.exists() else {"history": []}
    history = saved["history"]
    if not history or history[-1]["relax_timeout_s"] != seconds:
        history.append({
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "relax_timeout_s": seconds, "workers": workers,
            "original_failures": [row.copy() for row in rows.values() if is_oom_or_timeout(row)],
            "completed": [],
        })
    attempt = history[-1]
    finished = {tuple(key) for key in attempt["completed"]}
    for original in attempt["original_failures"]:
        key = int(original["gene"]), int(original["trial"])
        # A flushed replacement may precede the completion marker on a crash.
        if key in rows and rows[key] != original and key not in finished:
            attempt["completed"].append(list(key))
    write_json(path, saved)
    return saved


def resolve_relax_timeout(output: Path, source_timeout: float,
                          override: float | None, rows: dict) -> float:
    """Remember timeout overrides across resumes without changing trial identity."""
    settings_path = output / "relaxation_settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else None
    previous = float(settings["relax_timeout_s"]) if settings else source_timeout
    effective = previous if override is None else override
    if not math.isfinite(effective) or effective <= 0:
        raise ValueError("Relaxation timeout must be finite and positive")
    if settings is None or effective != previous:
        history = settings.get("history", []) if settings else []
        history.append({
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "previous_timeout_s": previous,
            "relax_timeout_s": effective,
            "origin": "command_line" if override is not None else "input_manifest",
            "completed_trials_at_change": len(rows),
            "last_recorded_trial": list(next(reversed(rows))) if rows else None,
        })
        write_json(settings_path, {
            "format": 1, "source_timeout_s": source_timeout,
            "relax_timeout_s": effective, "history": history,
        })
    return effective


def wandb_identity(manifest: dict) -> tuple[str, str]:
    payload = json.dumps(manifest, sort_keys=True).encode()
    digest = hashlib.sha256(payload).hexdigest()
    return digest[:8], f"mlip-bias-{digest[:16]}"


def record_execution(output: Path, workers: int, device: str, rows: dict) -> None:
    """Keep execution changes outside the scientific trial/run identity."""
    path = output / "execution_settings.json"
    saved = json.loads(path.read_text()) if path.exists() else {"history": []}
    current = {"workers": workers, "device": device,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "cpu_affinity": sorted(os.sched_getaffinity(0)),
               "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}
    if saved.get("current") != current:
        saved["history"].append({"date_utc": datetime.now(timezone.utc).isoformat(),
                                 "completed_trials_at_change": len(rows), **current})
        saved["current"] = current
        write_json(path, saved)


def restore_from_wandb(output: Path, artifact_name: str, *, entity: str, project: str) -> bool:
    """Restore a complete snapshot when the local trial log is absent."""
    import wandb

    try:
        artifact = wandb.Api().artifact(
            f"{entity}/{project}/{artifact_name}:latest", type=ARTIFACT_TYPE
        )
    except wandb.errors.CommError as exc:
        if "not found" in str(exc).lower() or "does not exist" in str(exc).lower():
            return False
        raise
    artifact.download(root=str(output))
    LOG.info("Restored %s into %s", artifact.name, output)
    return True


def snapshot_to_wandb(run, output: Path, artifact_name: str, rows: dict) -> None:
    """Mirror the trial ledger and every kept CIF, including losing trials."""
    import wandb

    metadata = {"n_trials": len(rows), "mlip": json.loads(
        (output / "study_manifest.json").read_text()
    )["mlip"]}
    settings_path = output / "relaxation_settings.json"
    if settings_path.is_file():
        metadata["relax_timeout_s"] = json.loads(settings_path.read_text())["relax_timeout_s"]
    artifact = wandb.Artifact(artifact_name, type=ARTIFACT_TYPE, metadata=metadata)
    for name in ("study_manifest.json", "trials.csv", "selected_free.json",
                 "selected_fixed_symmetry.json", "backend_transition.json",
                 "relaxation_settings.json", "execution_settings.json", "recovery_settings.json",
                 "failure_pass_settings.json", "before_failure_pass.csv"):
        path = output / name
        if path.is_file():
            artifact.add_file(str(path), name=name)
    paths = set()
    for row in rows.values():
        for field in ("cif", "fixed_cif"):
            if row[field]:
                paths.add(row[field])
    for relative in sorted(paths):
        path = output / relative
        if not path.is_file():
            raise FileNotFoundError(f"Cannot checkpoint missing {path}")
        artifact.add_file(str(path), name=relative)
    failure_settings = output / "failure_pass_settings.json"
    if failure_settings.is_file():
        # Intermediate cells and optimizer logs explain a failed recovery.
        for row in json.loads(failure_settings.read_text())["original_failures"]:
            directory = output / "trials" / str(row["gene"]) / f"trial-{row['trial']}"
            for path in sorted(directory.glob("*")):
                relative = str(path.relative_to(output))
                if path.is_file() and path.suffix in (".cif", ".log") and relative not in paths:
                    artifact.add_file(str(path), name=relative)
    for directory in ("cifs", "cifs_fixed_symmetry"):
        path = output / directory
        if path.is_dir():
            artifact.add_dir(str(path), name=directory)
    diagnostics = output / "failure_pass_diagnostics"
    if diagnostics.is_dir():
        artifact.add_dir(str(diagnostics), name="failure_pass_diagnostics")
    run.log_artifact(artifact).wait()
    run.summary["completed_trials"] = len(rows)
    run.summary["successful_trials"] = sum(r["status"] == "ok" for r in rows.values())
    LOG.info("Uploaded %s (%d trials) to W&B", artifact_name, len(rows))


def select_structures(output: Path, rows: dict[tuple[int, int], dict]) -> None:
    """Choose each gene's lowest energy trial on this arm's own potential."""
    for track, energy_key, source_key, directory in (
        ("free", "energy_ev_per_atom", "cif", "cifs"),
        ("fixed_symmetry", "fixed_energy_ev_per_atom", "fixed_cif", "cifs_fixed_symmetry"),
    ):
        chosen = {}
        for (gene, trial), row in rows.items():
            raw = row[energy_key]
            source = output / row[source_key] if row[source_key] else None
            if not raw:
                continue
            if source is None or not source.is_file():
                raise FileNotFoundError(f"Trial {(gene, trial)} has energy but no {source_key}: {source}")
            energy = float(raw)
            candidate = (energy, trial, source)
            if gene not in chosen or candidate[:2] < chosen[gene][:2]:
                chosen[gene] = candidate
        target_dir = output / directory
        target_dir.mkdir(exist_ok=True)
        for old in target_dir.glob("*.cif"):
            if old.stem.isdecimal() and int(old.stem) not in chosen:
                old.unlink()
        for gene, (_, _, source) in chosen.items():
            shutil.copyfile(source, target_dir / f"{gene}.cif")
        write_json(output / f"selected_{track}.json", {
            str(gene): {"trial": trial, "energy_ev_per_atom": energy,
                        "cif": f"{directory}/{gene}.cif"}
            for gene, (energy, trial, _) in sorted(chosen.items())
        })
        LOG.info("Selected %d %s structures", len(chosen), track)


def execute_trial(atoms, *, calculator, output: Path, mlip: str, device: str,
                  fmax: float, relax_timeout: float) -> dict:
    """Run one uniquely named trial; the coordinator owns the ledger and W&B."""
    gene, trial = int(atoms.info["gene"]), int(atoms.info["trial"])
    started = time.monotonic()
    formula = atoms.get_chemical_formula(mode="metal")
    row = dict.fromkeys(FIELDS, "")
    row.update(gene=gene, trial=trial, status="failed", formula=formula,
               n_atoms=len(atoms))
    trial_dir = output / "trials" / str(gene) / f"trial-{trial}"
    clear_cuda_cache = False
    try:
        with trial_timeout(relax_timeout):
            relaxed, energy, (fixed, fixed_energy) = relax_trial(
                atoms_in=atoms, calculator=calculator, trial_dir=trial_dir,
                label=f"{mlip} gene {gene} trial {trial}",
                release_symmetry=True, rattle=True,
                seed=_trial_seed(gene, trial), fmax=fmax,
            )
        if relaxed is not None:
            row.update(status="ok", energy_ev=energy,
                       energy_ev_per_atom=energy / len(relaxed),
                       cif=str((trial_dir / f"{formula}{KEPT_CIF_SUFFIX}").relative_to(output)))
        else:
            row.update(status="clash")
        if fixed is not None:
            row.update(fixed_energy_ev=fixed_energy,
                       fixed_energy_ev_per_atom=fixed_energy / len(fixed),
                       fixed_cif=str((trial_dir / f"{formula}{KEPT_FIXED_CIF_SUFFIX}").relative_to(output)))
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        LOG.exception("Failed gene %d trial %d", gene, trial)
        clear_cuda_cache = device.startswith("cuda") and (
            "out of memory" in str(exc).lower()
            or "CUBLAS_STATUS_ALLOC_FAILED" in str(exc)
        )
    if clear_cuda_cache:
        import torch

        torch.cuda.empty_cache()
    row["seconds"] = round(time.monotonic() - started, 3)
    return row


_WORKER_CALCULATOR = None
_WORKER_OPTIONS = None


def initialise_worker(mlip, device, factory, output, fmax, relax_timeout):
    global _WORKER_CALCULATOR, _WORKER_OPTIONS
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(processName)s %(levelname)s %(message)s")
    _print_logm_warning_once()
    _WORKER_OPTIONS = {"output": output, "mlip": mlip, "device": device,
                       "fmax": fmax, "relax_timeout": relax_timeout}
    _WORKER_CALCULATOR = build_calculator(mlip, device, factory)


def execute_worker_trial(atoms):
    return execute_trial(atoms, calculator=_WORKER_CALCULATOR, **_WORKER_OPTIONS)


def parallel_rows(draws, *, workers: int, mlip: str, device: str, factory,
                  output: Path, fmax: float, relax_timeout: float):
    """Keep at most one outstanding trial per spawned calculator process.

    Return finished rows immediately, so a slow trial cannot delay recording
    other results. The parent remains the sole ledger and artifact writer.
    """
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=initialise_worker,
        initargs=(mlip, device, factory, output, fmax, relax_timeout),
    ) as pool:
        draws = iter(draws)
        pending = set()
        for _ in range(workers):
            atoms = next(draws, None)
            if atoms is None:
                break
            pending.add(pool.submit(execute_worker_trial, atoms))
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                yield future.result()
                atoms = next(draws, None)
                if atoms is not None:
                    pending.add(pool.submit(execute_worker_trial, atoms))


def pending_draws(draws_path, done, *, limit_genes=None, max_trials=None):
    count = 0
    for atoms in iread(str(draws_path), index=":", format="extxyz"):
        gene, trial = int(atoms.info["gene"]), int(atoms.info["trial"])
        if limit_genes is not None and gene >= limit_genes:
            continue
        if (gene, trial) in done:
            continue
        if max_trials is not None and count >= max_trials:
            break
        count += 1
        yield atoms


def main() -> None:
    _print_logm_warning_once()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Downloaded protocol artifact directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mlip", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent calculator processes; one coordinator writes results")
    parser.add_argument("--calculator-factory",
                        help="Optional MODULE:FUNCTION taking device and returning ASE calculator")
    parser.add_argument("--checkpoint-id",
                        help="Required with a custom factory; exact weights identifier")
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--relax-timeout", type=float,
                        help="Override seconds per trial; otherwise use saved setting or input manifest")
    parser.add_argument("--limit-genes", type=int)
    parser.add_argument("--max-trials", type=int,
                        help="Bound this invocation; rerun to continue")
    retry = parser.add_mutually_exclusive_group()
    retry.add_argument("--retry-failed", action="store_true",
                       help="Rerun failed trials, replacing their ledger rows")
    retry.add_argument("--retry-timeouts", action="store_true",
                       help="Rerun timeout failures, preserving other failed trials")
    retry.add_argument("--retry-oom-timeouts", action="store_true",
                       help="Rerun OOM/timeout failures once at this budget; resume unfinished retries")
    parser.add_argument("--input-artifact", default=DEFAULT_INPUT_ARTIFACT)
    parser.add_argument("--wandb-entity", default="symmetry-advantage")
    parser.add_argument("--wandb-project", default="WyckoffTransformer")
    parser.add_argument("--sync-every", type=int, default=25,
                        help="Upload a resumable W&B artifact every N new trials")
    parser.add_argument("--no-wandb", action="store_true", help="Local smoke tests only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.fmax <= 0 or (args.limit_genes is not None and args.limit_genes < 1):
        parser.error("--fmax and --limit-genes must be positive")
    if args.calculator_factory and not args.checkpoint_id:
        parser.error("--checkpoint-id is required with --calculator-factory")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.sync_every < 1:
        parser.error("--sync-every must be positive")
    if args.relax_timeout is not None and (
            not math.isfinite(args.relax_timeout) or args.relax_timeout <= 0):
        parser.error("--relax-timeout must be finite and positive")
    draws_path = args.input / "pyxtal.extxyz"
    source = args.input / "manifest.json"
    if not args.no_wandb and (not draws_path.is_file() or not source.is_file()):
        import wandb

        LOG.info("Downloading input artifact %s", args.input_artifact)
        wandb.Api().artifact(args.input_artifact, type="protocol_eval").download(
            root=str(args.input)
        )
    if not draws_path.is_file() or not source.is_file():
        parser.error("--input must contain pyxtal.extxyz and manifest.json")
    source_manifest = json.loads(source.read_text())
    source_timeout = float(source_manifest["relax_timeout"])
    if not math.isfinite(source_timeout) or source_timeout <= 0:
        parser.error("Input manifest relax_timeout must be finite and positive")
    manifest_path = args.output / "study_manifest.json"
    manifest = {
        "input_sha256": sha256_file(draws_path),
        "input_manifest_sha256": sha256_file(source),
        "mlip": args.mlip,
        "calculator_factory": args.calculator_factory,
        "checkpoint_id": args.checkpoint_id or getattr(
            MLIP_REGISTRY.get(args.mlip) or HULL_MLIPS.get(args.mlip), "checkpoint", None
        ),
        "fmax_ev_per_angstrom": args.fmax,
        "limit_genes": args.limit_genes,
        "input_artifact": args.input_artifact,
        "schedule": "WyFormer relax_trial: fixed cell, symmetric cell, released cell, rattle",
        "rattle_seed": "wyckoff_transformer.cryspr.generator._trial_seed(gene, trial)",
    }
    run_id, artifact_name = wandb_identity(manifest)
    if not args.no_wandb and not (args.output / "trials.csv").is_file():
        restore_from_wandb(args.output, artifact_name,
                           entity=args.wandb_entity, project=args.wandb_project)
    args.output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing output manifest differs; use a separate output directory")
    else:
        write_json(manifest_path, manifest)
    rows_path = args.output / "trials.csv"
    done = load_rows(rows_path)
    record_execution(args.output, args.workers, args.device, done)
    relax_timeout = resolve_relax_timeout(args.output, source_timeout, args.relax_timeout, done)
    LOG.info("Relaxation timeout: %g s per trial (input manifest: %g s)",
             relax_timeout, source_timeout)
    invalidated = apply_historical_timeout(rows_path, done, relax_timeout)
    if invalidated:
        LOG.warning("Invalidated %d earlier trials exceeding %.0f s", invalidated,
                    relax_timeout)
    recovery = None
    if args.retry_oom_timeouts:
        recovery = prepare_recovery(args.output, done, relax_timeout, args.workers)
        finished = {tuple(key) for key in recovery["history"][-1]["completed"]}
        retry_keys = {(int(row["gene"]), int(row["trial"]))
                      for row in recovery["history"][-1]["original_failures"]} - finished
    if args.retry_failed or args.retry_timeouts or args.retry_oom_timeouts:
        failed = [key for key, row in done.items() if row["status"] == "failed"
                  and (args.retry_failed
                       or (args.retry_timeouts and "TimeoutError" in row["error"])
                       or (args.retry_oom_timeouts and key in retry_keys))]
        if failed:
            for key in failed:
                del done[key]
            replacement = rows_path.with_suffix(".csv.tmp")
            with replacement.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(done.values())
            os.replace(replacement, rows_path)
            LOG.info("Requeued %d failed trials", len(failed))
    run = None
    if not args.no_wandb:
        import wandb

        from wyckoff_transformer.paths import wandb_dir

        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                         id=run_id, resume="allow", dir=str(wandb_dir()),
                         job_type="mlip_bias_relax", config=manifest)
        run.summary["relax_timeout_s"] = relax_timeout
        run.summary["source_relax_timeout_s"] = source_timeout
        run.summary["relaxation_workers"] = args.workers
    if invalidated:
        select_structures(args.output, done)
        if run is not None:
            snapshot_to_wandb(run, args.output, artifact_name, done)
    try:
        calculator = (build_calculator(args.mlip, args.device, args.calculator_factory)
                      if args.workers == 1 else None)
    except Exception as exc:
        if run is not None:
            run.summary["setup_failure"] = f"{type(exc).__name__}: {exc}"
            run.finish(exit_code=1)
        raise
    if (run is not None and args.mlip in MLIP_REGISTRY
            and MLIP_REGISTRY[args.mlip].backend == "tace"
            and hasattr(calculator, "model")):
        oeq_modules = sum("openequivariance" in type(module).__module__
                          for module in calculator.model.modules())
        cue_modules = sum("cuequivariance" in type(module).__module__
                          for module in calculator.model.modules())
        run.summary["tace_oeq_modules"] = oeq_modules
        run.summary["tace_acceleration_backend"] = (
            "openequivariance" if oeq_modules else "cuequivariance" if cue_modules else "e3nn"
        )
        LOG.info("Loaded TACE checkpoint with %d OpenEquivariance modules", oeq_modules)
    completed = 0
    results = None
    failed = False
    try:
        with rows_path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if stream.tell() == 0:
                writer.writeheader()
            draws = pending_draws(draws_path, done, limit_genes=args.limit_genes,
                                  max_trials=args.max_trials)
            options = {"output": args.output, "mlip": args.mlip, "device": args.device,
                       "fmax": args.fmax, "relax_timeout": relax_timeout}
            if args.workers == 1:
                results = (execute_trial(atoms, calculator=calculator, **options) for atoms in draws)
            else:
                results = parallel_rows(draws, workers=args.workers,
                                        factory=args.calculator_factory, **options)
            LOG.info("Relaxing with %d calculator worker(s)", args.workers)
            for row in results:
                gene, trial = int(row["gene"]), int(row["trial"])
                writer.writerow(row)
                stream.flush()
                done[(gene, trial)] = row
                if recovery is not None and (gene, trial) in retry_keys:
                    recovery["history"][-1]["completed"].append([gene, trial])
                    write_json(args.output / "recovery_settings.json", recovery)
                completed += 1
                LOG.info("%s gene %d trial %d: %s", args.mlip, gene, trial, row["status"])
                if run is not None and completed % args.sync_every == 0:
                    select_structures(args.output, done)
                    snapshot_to_wandb(run, args.output, artifact_name, done)
    except BaseException as exc:
        failed = True
        if run is not None:
            run.summary["execution_failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if results is not None:
            results.close()
        select_structures(args.output, done)
        if run is not None:
            snapshot_to_wandb(run, args.output, artifact_name, done)
            run.finish(exit_code=1 if failed else 0)
    LOG.info("%d trials recorded; %d completed this invocation", len(done), completed)


if __name__ == "__main__":
    main()
