"""Retry frozen campaign failures once, with a fresh process and a separate identity."""

import argparse
import csv
import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from scripts.platforms.zeus.watch_mlip_bias_genbench import validate_snapshot
from scripts.run_mlip_bias_relax import (
    FIELDS,
    build_calculator,
    execute_trial,
    load_rows,
    record_execution,
    resolve_relax_timeout,
    select_structures,
    sha256_file,
    snapshot_to_wandb,
    wandb_identity,
    write_json,
)
from wyckoff_transformer.paths import wandb_dir

STUDY = Path("generated/mlip_bias_study")


def utc():
    return datetime.now(UTC).isoformat()


def artifact_pin(artifact):
    # Api().artifact("...:latest").name retains that alias in W&B 0.30.
    return artifact.name.split(":")[0] + ":" + artifact.version


def scalar_trial_row(row):
    import numpy as np

    return {key: value.item() if isinstance(value, (np.generic, np.ndarray)) else value
            for key, value in row.items()}


def replace_ledger(output, rows):
    path = output / "trials.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows.values())
    temporary.replace(path)


def prepare_arm(output, spec, namespace):
    """Copy immutable CIFs; keep the complete baseline ledger until results arrive."""
    source = Path(spec["source_directory"])
    baseline = validate_snapshot(source, spec.get("source_ledger_sha256"))
    manifest = {**baseline["manifest"], "calculator_factory": spec["factory"],
                "failure_pass_namespace": namespace, "failure_pass_source_artifact": spec["artifact"]}
    settings_path = output / "failure_pass_settings.json"
    if settings_path.is_file():
        settings = json.loads(settings_path.read_text())
        if settings["spec"] != spec or settings["namespace"] != namespace:
            raise ValueError("A recovery plan changed after preparation")
        if settings["phase"] != "planned":
            if json.loads((output / "study_manifest.json").read_text()) != manifest:
                raise ValueError("Recovery manifest changed")
            return settings
    else:
        output.mkdir(parents=True, exist_ok=True)
        original = load_rows(source / "trials.csv")
        settings = {"utc": utc(), "namespace": namespace, "spec": spec, "phase": "planned",
                    "baseline_ledger_sha256": baseline["trials_sha256"],
                    "baseline_selected_cif_hashes": baseline["selected_cif_hashes"],
                    "original_failures": [row for row in original.values() if row["status"] == "failed"],
                    "completed": [], "relax_timeout_s": 3600, "workers": 1}
        write_json(settings_path, settings)
    original = load_rows(source / "trials.csv")
    for row in original.values():
        for field in ("cif", "fixed_cif"):
            if row[field]:
                target = output / row[field]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / row[field], target)
    shutil.copyfile(source / "trials.csv", output / "before_failure_pass.csv")
    replace_ledger(output, original)
    write_json(output / "study_manifest.json", manifest)
    select_structures(output, original)
    settings["phase"] = "prepared"
    write_json(settings_path, settings)
    return settings


def commit_trial(output, settings, result):
    key = (int(result["gene"]), int(result["trial"]))
    planned = {(int(row["gene"]), int(row["trial"])) for row in settings["original_failures"]}
    if key not in planned:
        raise ValueError("Worker returned a trial outside the frozen failure set")
    rows = load_rows(output / "trials.csv")
    rows[key] = result
    replace_ledger(output, rows)
    select_structures(output, rows)
    if list(key) not in settings["completed"]:
        settings["completed"].append(list(key))
    write_json(output / "failure_pass_settings.json", settings)
    return rows


def worker(args):
    """The process and calculator are discarded after exactly one raw draw."""
    from ase.io import iread

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = json.loads((args.output / "study_manifest.json").read_text())
    atoms = next(atoms for atoms in iread(str(STUDY / "input" / "pyxtal.extxyz"), format="extxyz")
                 if (int(atoms.info["gene"]), int(atoms.info["trial"])) == (args.gene, args.trial))
    started = time.monotonic()
    calculator = build_calculator(manifest["mlip"], "cuda", manifest["calculator_factory"])
    import torch

    torch.cuda.reset_peak_memory_stats()
    # Compile a small graph first. Compilation is setup, not relaxation time.
    if manifest["calculator_factory"] == "mlip_bias_factories:build_prophet_kernel_compile":
        from ase.build import bulk

        warmup = bulk("Si", "diamond", a=5.43)
        warmup.calc = calculator
        warmup.get_potential_energy()
        warmup.get_forces()
        warmup.get_stress()
        calculator.reset()
    write_json(args.result.with_suffix(".setup.json"), {
        "utc": utc(), "initialization_seconds": time.monotonic() - started,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "factory": manifest["calculator_factory"],
        "worker_sha256": sha256_file(Path(__file__)),
        "factories_sha256": sha256_file(Path("scripts/mlip_bias_factories.py")),
    })
    result = execute_trial(atoms, calculator=calculator, output=args.output, mlip=manifest["mlip"],
                           device="cuda", fmax=manifest["fmax_ev_per_angstrom"], relax_timeout=3600)
    result = scalar_trial_row(result)
    resources = {"peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved()}
    try:
        free, total = torch.cuda.mem_get_info()
        resources.update(final_gpu_free_bytes=free, total_gpu_bytes=total)
    except RuntimeError as exc:
        resources["memory_query_error"] = str(exc)
    write_json(args.result.with_suffix(".resources.json"), resources)
    write_json(args.result, result)


def run_trial(args, spec, output, original):
    key = (int(original["gene"]), int(original["trial"]))
    diagnostics = output / "failure_pass_diagnostics"
    diagnostics.mkdir(exist_ok=True)
    result_path = diagnostics / f"gene-{key[0]}-trial-{key[1]}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    environment.update(spec.get("environment", {}))
    environment["PYTHONPATH"] = ":".join([".", "scripts"] +
                                        [str(STUDY / "deps" / name) for name in spec["overlays"]])
    command = [".venv/bin/python", __file__, "--worker", "--output", str(output),
               "--gene", str(key[0]), "--trial", str(key[1]), "--result", str(result_path)]
    # A native crash gets one fresh-parent retry; successful results are durable.
    for attempt in range(2):
        result_path.with_suffix(".setup.json").unlink(missing_ok=True)
        log_path = diagnostics / f"gene-{key[0]}-trial-{key[1]}-attempt-{attempt}.log"
        started = time.monotonic()
        with log_path.open("w") as log:
            child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                setup_finished = None
                while child.poll() is None:
                    setup_path = result_path.with_suffix(".setup.json")
                    if setup_finished is None and setup_path.exists():
                        setup_finished = time.monotonic()
                    elapsed = time.monotonic() - (setup_finished or started)
                    if elapsed > (3660 if setup_finished else 1800):
                        raise subprocess.TimeoutExpired(command, elapsed)
                    time.sleep(2)
                returncode = child.returncode
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                if not result_path.with_suffix(".setup.json").exists():
                    raise RuntimeError(f"Backend setup exceeded 1800s allowance: {log_path}")
                result = {**original, "status": "failed", "seconds": round(time.monotonic() - started, 3),
                          "error": "TimeoutError: fresh worker exceeded the 5400s setup+relaxation ceiling"}
                write_json(result_path, result)
                return result
        if result_path.is_file():
            return json.loads(result_path.read_text())
        if returncode >= 0:
            raise RuntimeError(f"Worker setup/execution failed (exit {returncode}); inspect {log_path}")
        print(f"Fresh worker terminated by signal {-returncode}: {log_path}", flush=True)
    result = {**original, "status": "failed", "seconds": round(time.monotonic() - started, 3),
              "error": f"NativeWorkerError: fresh process terminated by signal {-returncode} twice; see diagnostics"}
    write_json(result_path, result)
    return result


def mirror_provenance(run, root, plan):
    import wandb

    artifact = wandb.Artifact(f"mlip-bias-failure-pass-{plan['namespace']}", type="mlip_bias_compat")
    paths = [root / "plan.json", Path(__file__), Path("scripts/mlip_bias_factories.py"),
             Path("scripts/platforms/zeus/probe_mlip_failure_backend.py"),
             Path("scripts/platforms/zeus/monitor_mlip_failure_pass.py"),
             Path("scripts/platforms/zeus/watch_mlip_bias_genbench.py"),
             Path("scripts/run_mlip_bias_relax.py"), Path("scripts/run_mlip_bias_genbench.py"),
             Path("scripts/report_mlip_bias_genbench.py"), Path("scripts/compute_mlip_bias_common_genbench.py"),
             Path("scripts/verify_mlip_bias_genbench_report.py"),
             Path("src/wyckoff_transformer/cryspr/mlips.py"), Path("src/wyckoff_transformer/cryspr/relaxer.py")]
    for spec in plan["arms"].values():
        for probe in spec.get("probes", []) + spec.get("optional_probe_diagnostics", []):
            paths.extend([Path(probe), Path(probe).with_suffix(".log")])
    for arm in plan["arms"]:
        paths.extend([root / arm / "failure_pass_settings.json", root / arm / "failure_pass_complete.json"])
    for path in paths:
        if path.is_file():
            artifact.add_file(str(path), name=str(path.resolve().relative_to(Path.cwd())))
    run.log_artifact(artifact).wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--gpu", type=int, choices=(0, 1))
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--gene", type=int)
    parser.add_argument("--trial", type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.plan is None or args.gpu is None or not args.arms:
        parser.error("--plan, --gpu and --arms are required")
    plan = json.loads(args.plan.read_text())
    root = args.plan.parent
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with (STUDY / f"zeus_gpu{args.gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for arm in args.arms:
            if (STUDY / f"zeus_gpu{args.gpu}.stop").exists() or (root / f"gpu{args.gpu}.stop").exists():
                print("Stopped at a trial boundary", flush=True)
                return
            spec = plan["arms"][arm]
            for probe_path in spec.get("probes", []):
                while not Path(probe_path).is_file():
                    print("Waiting for backend probe:", probe_path, flush=True)
                    time.sleep(30)
                if not json.loads(Path(probe_path).read_text())["passed"]:
                    raise ValueError(f"Required backend probe did not pass: {probe_path}")
            output = root / arm
            if (output / "failure_pass_complete.json").is_file():
                print(arm, "already complete", flush=True)
                continue
            settings = prepare_arm(output, spec, plan["namespace"])
            manifest = json.loads((output / "study_manifest.json").read_text())
            rows = load_rows(output / "trials.csv")
            record_execution(output, 1, f"cuda (physical GPU {args.gpu})", rows)
            resolve_relax_timeout(output, 300, 3600, rows)
            run_id, artifact_name = wandb_identity(manifest)
            import wandb

            run = wandb.init(entity="symmetry-advantage", project="WyckoffTransformer", id=run_id,
                             resume="allow", dir=wandb_dir(), job_type="mlip_bias_failure_pass", config=manifest)
            try:
                mirror_provenance(run, root, plan)
                snapshot_to_wandb(run, output, artifact_name, rows)
                run.summary["original_failures"] = len(settings["original_failures"])
                for original in settings["original_failures"]:
                    key = [int(original["gene"]), int(original["trial"])]
                    if key in settings["completed"]:
                        continue
                    if (STUDY / f"zeus_gpu{args.gpu}.stop").exists() or (root / f"gpu{args.gpu}.stop").exists():
                        return
                    print(utc(), arm, "retrying", key, flush=True)
                    result = run_trial(args, spec, output, original)
                    rows = commit_trial(output, settings, result)
                    run.summary["failure_pass_completed"] = len(settings["completed"])
                    run.summary["failure_pass_recovered"] = sum(rows[(int(r["gene"]), int(r["trial"]))]["status"] == "ok"
                                                                 for r in settings["original_failures"])
                    snapshot_to_wandb(run, output, artifact_name, rows)
                    print(utc(), arm, key, result["status"], result["error"], flush=True)
                validation = validate_snapshot(output)
                final = wandb.Api().artifact(f"symmetry-advantage/WyckoffTransformer/{artifact_name}:latest")
                final_file = final.get_path("trials.csv").download(root=str(output / "publication_verification"))
                if sha256_file(Path(final_file)) != validation["trials_sha256"]:
                    raise ValueError("Final W&B ledger differs from the local completed ledger")
                settings["phase"] = "complete"
                write_json(output / "failure_pass_settings.json", settings)
                marker = {"utc": utc(), "arm": arm, "complete": True, "artifact": artifact_pin(final),
                          "source_artifact": spec["artifact"], "ledger_sha256": validation["trials_sha256"],
                          "rows": len(rows), "successful": sum(row["status"] == "ok" for row in rows.values()),
                          "retried": len(settings["completed"]), "recovered": run.summary["failure_pass_recovered"],
                          "selected_cif_hashes": validation["selected_cif_hashes"], "run_id": run_id}
                write_json(output / "failure_pass_complete.json", marker)
                mirror_provenance(run, root, plan)
                run.summary["failure_pass_complete"] = True
                print("COMPLETE", arm, marker["artifact"], "recovered", marker["recovered"], flush=True)
            finally:
                run.finish()


if __name__ == "__main__":
    main()
