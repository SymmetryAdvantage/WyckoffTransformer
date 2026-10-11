"""Publish recovery progress, then audit the new nine-arm comparison."""

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

import wandb

from scripts.platforms.zeus.retry_mlip_bias_failures import STUDY, artifact_pin, utc
from scripts.platforms.zeus.supervise_mlip_bias_campaign import (
    gpu_idle,
    refresh_budget,
    spend_restart,
)
from scripts.platforms.zeus.watch_mlip_bias_genbench import ARMS, atomic_json
from wyckoff_transformer.paths import wandb_dir

QUEUES = {0: ["equflashv2", "pet_oam_xl", "tace_oam_l", "equiformerv3_dens_oam", "nequip_oam_xl"],
          1: ["prophet_oame_mbd"]}


def supervise_jobs(root, plan_path, payload, run):
    """Restart absent jobs with bounded retries and durable progress fingerprints."""
    state_path = root / "supervision_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": {}}
    live = set(subprocess.check_output(["tmux", "list-windows", "-t", "0", "-F", "#{window_name}"],
                                      text=True).splitlines())
    namespace = payload["namespace"]
    index_path = STUDY / "metrics" / namespace / "scoring_source_index.json"
    scored = json.loads(index_path.read_text())["arms"] if index_path.is_file() else {}
    jobs = {}
    for gpu, arms in QUEUES.items():
        name = f"mlip-failures-gpu{gpu}"
        cpus = "8-15" if gpu == 0 else "16-23"
        command = (f"CUDA_VISIBLE_DEVICES={gpu} WANDB_MODE=online WANDB_ENTITY=symmetry-advantage "
                   "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 "
                   f"PYTHONPATH=.:scripts taskset -c {cpus} .venv/bin/python "
                   "scripts/platforms/zeus/retry_mlip_bias_failures.py "
                   f"--plan {shlex.quote(str(plan_path))} --gpu {gpu} --arms {' '.join(arms)}")
        jobs[name] = {"gpu": gpu, "command": command, "log": root / f"gpu{gpu}.log",
                      "complete": all(arm in payload["arms"] for arm in arms),
                      "progress": json.dumps({arm: payload["progress"].get(arm, {}) for arm in arms}, sort_keys=True)}
    jobs["mlip-failure-scores"] = {
        "gpu": 0, "complete": set(scored) == set(ARMS), "progress": json.dumps(sorted(scored)),
        "log": root / "scoring.log",
        "command": ("MLIP_BIAS_GPU=0 scripts/platforms/zeus/run_updated_mlip_bias_genbench.sh "
                    f"--namespace {namespace} --completion {shlex.quote(str(root / 'completion_index.json'))} "
                    "--completion-only --reuse-source-index "
                    "generated/mlip_bias_study/metrics/recovery-20261006/scoring_source_index.json")}
    for name, job in jobs.items():
        record = state["jobs"].setdefault(name, {})
        refresh_budget(record, job["progress"])
        gpu = job["gpu"]
        paused = ((STUDY / f"zeus_gpu{gpu}.stop").exists() or (root / f"gpu{gpu}.stop").exists()
                  or (name == "mlip-failure-scores" and (STUDY / "zeus_genbench_gpu0.stop").exists()))
        if job["complete"]:
            record["status"] = "complete"
        elif paused:
            record["status"] = "stopped_by_marker"
        elif name in live:
            record["status"] = "running"
        elif name != "mlip-failure-scores" and not gpu_idle(gpu):
            record["status"] = "waiting_for_gpu_owner_to_exit"
        elif spend_restart(record):
            command = job["command"] + " >> " + shlex.quote(str(job["log"])) + " 2>&1"
            subprocess.run(["tmux", "new-window", "-d", "-t", "0:", "-c", str(Path.cwd()), "-n", name, command],
                           check=True)
            record["status"] = "restarted"
            print(utc(), name, "restarted", flush=True)
        else:
            record["status"] = "review_required_after_no_progress_restarts"
    atomic_json(state_path, state)
    run.summary["supervision"] = state
    return state


def collect(root, plan, baseline):
    arms = {arm: {**source["completion_evidence"], "complete": True,
                  "artifact": source["artifact"], "reused_from_namespace": baseline["snapshot_namespace"]}
            for arm, source in baseline["arms"].items() if arm not in plan["arms"]}
    progress = {}
    for arm in plan["arms"]:
        marker = root / arm / "failure_pass_complete.json"
        settings = root / arm / "failure_pass_settings.json"
        if settings.is_file():
            record = json.loads(settings.read_text())
            progress[arm] = {"planned": len(record["original_failures"]), "completed": len(record["completed"]),
                             "phase": record["phase"]}
        if marker.is_file():
            arms[arm] = json.loads(marker.read_text())
            progress[arm]["recovered"] = arms[arm]["recovered"]
    return {"utc": utc(), "namespace": plan["namespace"], "arms": arms,
            "progress": progress, "all_nine_complete": set(arms) == set(ARMS)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    root = args.plan.parent
    plan = json.loads(args.plan.read_text())
    baseline = json.loads((STUDY / "metrics" / "recovery-20261006" / "scoring_source_index.json").read_text())
    run = wandb.init(entity="symmetry-advantage", project="WyckoffTransformer", id="mlipfailurepass20261009",
                     resume="allow", dir=wandb_dir(), job_type="mlip_bias_compat",
                     config={"namespace": plan["namespace"], "baseline_report_run": plan["baseline_report_run"]})
    previous = None
    try:
        while not (root / "monitor.stop").exists():
            payload = collect(root, plan, baseline)
            supervision = supervise_jobs(root, args.plan, payload, run)
            identity = {key: value for key, value in payload.items() if key != "utc"}
            if identity != previous:
                path = root / "completion_index.json"
                atomic_json(path, payload)
                artifact = wandb.Artifact("mlip-bias-failure-pass-progress-20261009", type="mlip_bias_compat")
                for source in (path, args.plan, Path(__file__)):
                    artifact.add_file(str(source), name=source.name)
                artifact.add_file(str(root / "supervision_state.json"), name="supervision_state.json")
                run.log_artifact(artifact).wait()
                # W&B 0.30 can retain old leaves when replacing a nested dict
                # on a resumed run. Publish each scalar explicitly instead.
                run.summary.update({f"progress.{arm}.{key}": value
                                    for arm, values in payload["progress"].items()
                                    for key, value in values.items()})
                run.summary["all_nine_relaxations_complete"] = payload["all_nine_complete"]
                print(utc(), json.dumps(payload["progress"]), flush=True)
                previous = identity
            if any(record["status"] == "review_required_after_no_progress_restarts"
                   for record in supervision["jobs"].values()):
                print("A job exhausted its bounded restart allowance; see supervision_state.json", flush=True)
            index = STUDY / "metrics" / plan["namespace"] / "scoring_source_index.json"
            if (payload["all_nine_complete"] and index.is_file()
                    and set(json.loads(index.read_text())["arms"]) == set(ARMS)):
                log = root / "report.log"
                with log.open("a") as stream:
                    subprocess.run(["scripts/platforms/zeus/report_updated_mlip_bias_genbench.sh", plan["namespace"]],
                                   stdout=stream, stderr=subprocess.STDOUT, check=True)
                audit = json.loads((index.parent / "report" / "genbench_results_validation.json").read_text())
                if not audit["passed"]:
                    raise ValueError("The recovery comparison audit failed")
                records = [line.split() for line in log.read_text().splitlines() if line.startswith("REPORT_RUN ")]
                if not records:
                    raise ValueError("The comparison was not published to W&B")
                _, report_run, _, report_artifact = records[-1]
                api = wandb.Api()
                if api.run("symmetry-advantage/WyckoffTransformer/" + report_run).state != "finished":
                    raise ValueError("The comparison W&B run did not finish")
                published = api.artifact(report_artifact, type="mlip_bias_metrics_report")
                downloaded = Path(published.get_entry("scoring_source_index.json").download(
                    root=str(root / "report_publication_verification")))
                index_hash = hashlib.sha256(index.read_bytes()).hexdigest()
                if hashlib.sha256(downloaded.read_bytes()).hexdigest() != index_hash:
                    raise ValueError("The uploaded comparison source index differs")
                publication = {"complete": True, "run_id": report_run, "artifact": artifact_pin(published),
                               "source_index_sha256": index_hash}
                run.summary["report_audit_passed"] = True
                run.summary["report_run_id"] = report_run
                atomic_json(root / "study_complete.json", {"utc": utc(), "namespace": plan["namespace"],
                                                           "comparison_audit": audit, "publication": publication})
                final_artifact = wandb.Artifact("mlip-bias-failure-pass-progress-20261009", type="mlip_bias_compat")
                for source in (root / "study_complete.json", index, log, root / "completion_index.json"):
                    final_artifact.add_file(str(source), name=source.name)
                run.log_artifact(final_artifact).wait()
                print("Failure recovery and comparison complete", flush=True)
                return
            time.sleep(30)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
