"""Finish each migrated arm before proceeding; never repeatedly requeue failures."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import wandb

from wyckoff_transformer.paths import wandb_dir

STUDY = Path("generated/mlip_bias_study")
WORK = STUDY / "zeus_migration"
SETUP = STUDY / "setup"
GPU = int(os.environ.get("MLIP_BIAS_GPU", "1"))
ARMS = {
    "tece_oam_rra_1_0": ("retry_failed", ["migration", "genbench"], 1, 3600),
    "equiformerv3_dens_oam": ("retry_failed", ["equiformer_v3_src", "fairchem"], 1, 3600),
    "esen_30m_oam": ("retry_failed", ["fairchem"], 1, 3600),
    "tace_oam_l": ("resume", ["migration", "genbench"], 1, 3600),
    "orb_conserv_inf": ("resume", ["genbench"], 2, 1200),
    "prophet_oame_mbd": ("resume", ["migration", "genbench"], 2, 1200),
}
# TECE/Equiformer first passes are confirmed finished in the incoming handoff.
# eSEN's W&B state changed during preparation; require the migration confirmation.
NEEDS_BOUNDARY = {"esen_30m_oam", "tace_oam_l", "orb_conserv_inf", "prophet_oame_mbd"}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def rows_in(directory):
    with (directory / "trials.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = {(int(r["gene"]), int(r["trial"])) for r in rows}
    if len(keys) != len(rows):
        raise ValueError(f"Duplicate trial identities in {directory}")
    return rows


def recovery_remaining(directory):
    settings = directory / "recovery_settings.json"
    if settings.is_file():
        history = json.loads(settings.read_text())["history"]
        if history and history[-1]["relax_timeout_s"] == 3600:
            attempt = history[-1]
            completed = {tuple(key) for key in attempt["completed"]}
            return sum((int(r["gene"]), int(r["trial"])) not in completed
                       for r in attempt["original_failures"])
    return sum(row["status"] == "failed" and any(
        pattern in row["error"].lower() for pattern in
        ("timeouterror", "out of memory", "outofmemoryerror",
         "cudaerrormemoryallocation", "failed to allocate")) for row in rows_in(directory))


def prepare_retry(directory, source):
    """Persist the original failures before atomically removing them once.

    A crash between filtering and recording 'prepared' is safe: the original
    rows are either unchanged or missing, and no calculator has run yet.
    """
    settings = directory / "zeus_failed_retry.json"
    if settings.exists():
        plan = json.loads(settings.read_text())
        if plan["source_artifact"] != source["artifact"]:
            raise ValueError("Retry source identity changed")
        if plan["phase"] != "planned":
            return plan
    else:
        rows = rows_in(directory)
        if len(rows) != 2698:
            raise ValueError("A new failed-trial retry requires a complete first pass")
        original = directory / "zeus_before_failed_retry.csv"
        original.write_bytes((directory / "trials.csv").read_bytes())
        plan = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "source_artifact": source["artifact"],
            "phase": "planned",
            "original_ledger_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "original_failures": [r for r in rows if r["status"] == "failed"],
        }
        atomic_json(settings, plan)
    failures = {(r["gene"], r["trial"]): r for r in plan["original_failures"]}
    rows = rows_in(directory)
    for row in rows:
        key = (row["gene"], row["trial"])
        if key in failures and row != failures[key]:
            raise ValueError("A trial changed before its retry preparation finished")
    kept = [r for r in rows if (r["gene"], r["trial"]) not in failures]
    temporary = directory / "trials.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(kept)
    temporary.replace(directory / "trials.csv")
    plan["phase"] = "prepared"
    atomic_json(settings, plan)
    print(f"Requeued {len(failures)} original failures once; retained {len(kept)} rows", flush=True)
    return plan


def upload_preparation(arm, source, run):
    directory = WORK / arm
    artifact = wandb.Artifact(f"mlip-bias-zeus-migration-{arm}-20261006", type="mlip_bias_compat")
    paths = [WORK / "migration_plan.json", SETUP / f"migration_probe_{arm}_20261006.json",
             SETUP / f"migration_probe_{arm}_20261006.log",
             directory / "zeus_failed_retry.json", directory / "zeus_before_failed_retry.csv",
             directory / "study_manifest.json", directory / "zeus_source_snapshot.json",
             directory / "zeus_execution_started.json", WORK / "boundary_confirmation.json",
             directory / "zeus_relaxation_complete.json", directory / "recovery_settings.json",
             Path("scripts/platforms/zeus/run_mlip_bias_migration_gpu1.py"),
             Path("scripts/platforms/zeus/run_mlip_bias_migration_gpu1.sh"),
             Path("src/wyckoff_transformer/cryspr/mlips.py")]
    for path in paths:
        if path.is_file():
            artifact.add_file(str(path), name=path.name)
    artifact.metadata.update(arm=arm, source_artifact=source["artifact"], run_id=source["run_id"])
    run.log_artifact(artifact).wait()


def stop_requested():
    return ((STUDY / f"zeus_gpu{GPU}.stop").exists()
            or (WORK / f"gpu{GPU}.stop").exists()
            or (GPU == 1 and (WORK / "stop").exists()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="*", default=list(ARMS))
    args = parser.parse_args()
    if any(arm not in ARMS for arm in args.arms):
        parser.error(f"Unknown arm; choose from {list(ARMS)}")
    plan = json.loads((WORK / "migration_plan.json").read_text())
    sources = {r["arm"]: r for r in plan["arms"]}
    run = wandb.init(entity="symmetry-advantage", project="WyckoffTransformer",
                     id=f"zeusmigrationgpu{GPU}20261006", resume="allow",
                     job_type="mlip_bias_compat", dir=wandb_dir())
    try:
        for arm in args.arms:
            if stop_requested():
                return
            while arm in NEEDS_BOUNDARY and not (WORK / "boundary_confirmation.json").is_file():
                print(f"{arm}: waiting for iapetus single-writer boundary confirmation", flush=True)
                if stop_requested():
                    return
                time.sleep(30)
            source = sources[arm]
            directory = WORK / arm
            source_path = directory / "zeus_source_snapshot.json"
            if source_path.is_file():
                source = json.loads(source_path.read_text())
            mode, overlays, workers, timeout = ARMS[arm]
            probe = json.loads((SETUP / f"migration_probe_{arm}_20261006.json").read_text())
            if not probe["passed"] or probe["manifest"] != source["manifest"]:
                raise ValueError(f"Model probe failed or mismatched for {arm}")
            # Refresh only before this host has prepared/run the arm. New remote
            # boundary rows must not be discarded in favor of a stale snapshot.
            started = directory / "zeus_execution_started.json"
            if not started.exists() and arm in NEEDS_BOUNDARY:
                latest = wandb.Api().artifact(
                    "symmetry-advantage/WyckoffTransformer/" + source["artifact"].split(":")[0] + ":latest",
                    type="mlip_bias_relax")
                latest_pinned = latest.name.split(":")[0] + ":" + latest.version
                if latest_pinned != source["artifact"]:
                    latest.download(root=str(directory))
                    source.update(artifact=latest_pinned, rows=len(rows_in(directory)))
                    source["ledger_sha256"] = hashlib.sha256((directory / "trials.csv").read_bytes()).hexdigest()
                if json.loads((directory / "study_manifest.json").read_text()) != source["manifest"]:
                    raise ValueError("Refreshed scientific manifest changed")
            atomic_json(source_path, source)
            if mode == "retry_failed":
                prepare_retry(directory, source)
            upload_preparation(arm, source, run)
            atomic_json(started, dict(arm=arm, source_artifact=source["artifact"], workers=workers, gpu=GPU,
                                     timeout_seconds=timeout, utc=datetime.now(timezone.utc).isoformat()))
            environment = os.environ.copy()
            # Separate each calculator runner's W&B service from the long-lived
            # queue, whose API requests otherwise compete with artifact uploads.
            environment.pop("WANDB_SERVICE", None)
            environment["PYTHONFAULTHANDLER"] = "1"
            environment["PYTHONPATH"] = os.pathsep.join(
                [str(Path("scripts").resolve())] +
                [str((STUDY / "deps" / overlay).resolve()) for overlay in overlays])
            environment.update(TACE_USE_CUE="0", TACE_USE_COMPILE="0", TACE_USE_EQT="0",
                               TACE_USE_OEQ="1" if arm == "tece_oam_rra_1_0" else "0")
            manifest = source["manifest"]
            while len(rows_in(directory)) < 2698:
                if stop_requested():
                    return
                before = len(rows_in(directory))
                command = [".venv/bin/python", "scripts/run_mlip_bias_relax.py",
                           "--input", str(STUDY / "input"), "--output", str(directory),
                           "--mlip", manifest["mlip"], "--device", "cuda", "--workers", str(workers),
                           "--relax-timeout", str(timeout), "--max-trials", "25", "--sync-every", "25"]
                if manifest.get("calculator_factory"):
                    command.extend(["--calculator-factory", manifest["calculator_factory"],
                                    "--checkpoint-id", manifest["checkpoint_id"]])
                print(f"{arm}: {before}/2698 rows; {workers} worker(s), {timeout}s", flush=True)
                # Retain the lock in the child if the coordinator is interrupted.
                subprocess.run(command, env=environment, check=True, pass_fds=(9,))
                if len(rows_in(directory)) <= before:
                    raise RuntimeError(f"No ledger progress for {arm}")
            # A full first-pass/interrupted TACE retry ledger is followed by one
            # persisted allocator/timeout recovery pass. The runner records
            # finished attempts so a resumed queue never requeues them again.
            if mode == "resume":
                remaining = recovery_remaining(directory)
                while remaining:
                    if stop_requested():
                        return
                    command = [".venv/bin/python", "scripts/run_mlip_bias_relax.py",
                               "--input", str(STUDY / "input"), "--output", str(directory),
                               "--mlip", manifest["mlip"], "--device", "cuda", "--workers", "1",
                               "--relax-timeout", "3600", "--retry-oom-timeouts",
                               "--max-trials", "25", "--sync-every", "25"]
                    if manifest.get("calculator_factory"):
                        command.extend(["--calculator-factory", manifest["calculator_factory"],
                                        "--checkpoint-id", manifest["checkpoint_id"]])
                    print(f"{arm}: {remaining} allocator/timeout retries; one worker, 3600s", flush=True)
                    subprocess.run(command, env=environment, check=True, pass_fds=(9,))
                    after = recovery_remaining(directory)
                    if after >= remaining:
                        raise RuntimeError(f"No recovery progress for {arm}")
                    remaining = after
            rows = rows_in(directory)
            if mode == "retry_failed":
                retry_path = directory / "zeus_failed_retry.json"
                retry = json.loads(retry_path.read_text())
                retry["phase"] = "complete"
                atomic_json(retry_path, retry)
            run.summary[f"migration/{arm}/rows"] = len(rows)
            run.summary[f"migration/{arm}/successful"] = sum(r["status"] == "ok" for r in rows)
            ledger_sha256 = hashlib.sha256((directory / "trials.csv").read_bytes()).hexdigest()
            latest = wandb.Api().artifact(
                "symmetry-advantage/WyckoffTransformer/" + source["artifact"].split(":")[0] + ":latest",
                type="mlip_bias_relax")
            mirrored = Path(latest.get_entry("trials.csv").download(
                root=str(directory / "final_snapshot_verification")))
            if hashlib.sha256(mirrored.read_bytes()).hexdigest() != ledger_sha256:
                raise ValueError(f"Final W&B ledger differs from local ledger for {arm}")
            atomic_json(directory / "zeus_relaxation_complete.json", dict(
                arm=arm, gpu=GPU, rows=len(rows), successful=sum(r["status"] == "ok" for r in rows),
                source_artifact=source["artifact"], complete=True,
                artifact=latest.name.split(":")[0] + ":" + latest.version,
                ledger_sha256=ledger_sha256, all_planned_passes_complete=True,
                utc=datetime.now(timezone.utc).isoformat()))
            print(f"{arm}: complete; {run.summary[f'migration/{arm}/successful']} successful", flush=True)
            upload_preparation(arm, source, run)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
