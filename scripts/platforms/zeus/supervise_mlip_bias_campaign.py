"""Resume interrupted study queues with persistent, bounded restart budgets."""

import fcntl
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import wandb

from wyckoff_transformer.paths import wandb_dir

STUDY = Path("generated/mlip_bias_study")
WORK = STUDY / "zeus_migration"
METRICS = STUDY / "metrics/recovery-20261006"
STATE = WORK / "supervision_state.json"
MAX_NO_PROGRESS_RESTARTS = 3
ASSIGNMENTS = {
    0: ("tace_oam_l", "orb_conserv_inf", "prophet_oame_mbd"),
    1: ("tece_oam_rra_1_0", "equiformerv3_dens_oam", "esen_30m_oam"),
}
ALL_ARMS = {"equflashv2", "pet_oam_xl", "nequip_oam_xl", *ASSIGNMENTS[0], *ASSIGNMENTS[1]}
JOBS = {
    "mlip-migration-gpu0": {
        "gpu": 0, "kind": "relaxation",
        "command": "MLIP_BIAS_GPU=0 scripts/platforms/zeus/run_mlip_bias_migration_gpu1.sh "
                   "--wait-for-lock tace_oam_l orb_conserv_inf prophet_oame_mbd",
        "log": STUDY / "zeus_migration_gpu0_20261006.log",
    },
    "mlip-recovery-gpu1": {
        "gpu": 1, "kind": "relaxation",
        "command": "scripts/platforms/zeus/run_mlip_bias_migration_gpu1.sh "
                   "--wait-for-lock tece_oam_rra_1_0 equiformerv3_dens_oam esen_30m_oam",
        "log": STUDY / "zeus_recovery_gpu1_20261006.log",
    },
    "mlip-completion": {
        "kind": "completion", "command": ".venv/bin/python "
            "scripts/platforms/zeus/monitor_mlip_bias_completion.py",
        "log": STUDY / "study_completion_20261006.log",
    },
    "mlip-updated-genbench": {
        "gpu": 1, "kind": "scoring",
        "command": "MLIP_BIAS_GPU=1 scripts/platforms/zeus/run_updated_mlip_bias_genbench.sh "
                   "--namespace recovery-20261006",
        "log": STUDY / "zeus_updated_genbench_20261006.log",
    },
    "mlip-updated-report": {
        "kind": "report", "command": "scripts/platforms/zeus/report_updated_mlip_bias_genbench.sh "
                                      "recovery-20261006",
        "log": STUDY / "zeus_updated_report_20261006.log",
    },
}


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def final_arm(arm):
    return read_json(WORK / arm / "zeus_relaxation_complete.json", {}).get("complete") is True


def completed_index(path):
    arms = read_json(path, {}).get("arms", {})
    return set(arms) == ALL_ARMS


def paused(job):
    gpu = job.get("gpu")
    if gpu is not None:
        if (STUDY / f"zeus_gpu{gpu}.stop").exists() or (WORK / f"gpu{gpu}.stop").exists():
            return True
        if gpu == 1 and (WORK / "stop").exists():
            return True
        if gpu == 0:
            loan = read_json(STUDY / "setup/gpu0_loan_20261007.json", {})
            if loan.get("phase") in {"waiting_for_saved_trial", "checkpointing", "gpu0_lent"}:
                return True
    if job["kind"] == "scoring":
        return (STUDY / f"zeus_genbench_gpu{gpu}.stop").exists()
    if job["kind"] == "completion":
        return (WORK / "completion_monitor.stop").exists()
    return False


def progress(job):
    """Hash durable trial/journal data; scorer progress uses saved checkpoints."""
    if job["kind"] == "relaxation":
        paths = [WORK / arm / name for arm in ASSIGNMENTS[job["gpu"]]
                 for name in ("trials.csv", "recovery_settings.json", "zeus_failed_retry.json",
                              "zeus_relaxation_complete.json")]
    elif job["kind"] == "completion":
        paths = [WORK / "study_relaxation_completion.json"]
    else:
        paths = [METRICS / "scoring_source_index.json", METRICS / "report/report_provenance.json",
                 METRICS / "report/common_cohort_metrics.json"]
    payload = {}
    for path in paths:
        if not path.is_file():
            continue
        content = path.read_bytes()
        if path.suffix == ".json":
            value = json.loads(content)
            # Re-publishing identical completion evidence with a new timestamp
            # is not a recovered trial and must not reset a crash budget.
            for key in ("utc", "date_utc"):
                value.pop(key, None)
            content = json.dumps(value, sort_keys=True).encode()
        payload[str(path)] = hashlib.sha256(content).hexdigest()
    if job["kind"] == "scoring":
        for path in METRICS.glob("*/*/*/single_point_checkpoint.json.gz"):
            stat = path.stat()
            payload[str(path)] = [stat.st_size, stat.st_mtime_ns]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def refresh_budget(record, fingerprint):
    if record.get("progress") != fingerprint:
        record.update(progress=fingerprint, no_progress_restarts=0, review_required=False)
    record.setdefault("no_progress_restarts", 0)


def spend_restart(record):
    if record["no_progress_restarts"] >= MAX_NO_PROGRESS_RESTARTS:
        record["review_required"] = True
        return False
    record["no_progress_restarts"] += 1
    return True


def report_complete():
    """The final report must have passed audit and actually reached W&B."""
    publication = read_json(METRICS / "report/supervisor_publication_verification.json", {})
    index = METRICS / "scoring_source_index.json"
    if not completed_index(index):
        return False
    index_hash = hashlib.sha256(index.read_bytes()).hexdigest()
    if publication.get("complete") and publication.get("source_index_sha256") == index_hash:
        return True
    audit = read_json(METRICS / "report/genbench_results_validation.json", {})
    provenance = read_json(METRICS / "report/report_provenance.json", {})
    if not audit.get("passed") or provenance.get("scoring_source_index_sha256") != index_hash:
        return False
    log = JOBS["mlip-updated-report"]["log"]
    if not log.is_file():
        return False
    lines = log.read_text().splitlines()
    reports = [line.split() for line in lines if line.startswith("REPORT_RUN ")]
    if not reports:
        return False
    _, run_id, _, artifact_name = reports[-1]
    api = wandb.Api()
    if api.run("symmetry-advantage/WyckoffTransformer/" + run_id).state != "finished":
        return False
    artifact = api.artifact(artifact_name, type="mlip_bias_metrics_report")
    downloaded = Path(artifact.get_entry("scoring_source_index.json").download(
        root=str(WORK / "report_publication_verification")))
    if hashlib.sha256(downloaded.read_bytes()).hexdigest() != index_hash:
        raise ValueError("Final report artifact differs from the final scoring source index")
    publication = dict(complete=True, source_index_sha256=index_hash, run_id=run_id,
                       artifact=artifact.name.split(":")[0] + ":" + artifact.version)
    (METRICS / "report/supervisor_publication_verification.json").write_text(
        json.dumps(publication, indent=2) + "\n")
    return True


def job_done(job):
    kind = job["kind"]
    if kind == "relaxation":
        return all(final_arm(arm) for arm in ASSIGNMENTS[job["gpu"]])
    if kind == "completion":
        index = read_json(WORK / "study_relaxation_completion.json", {})
        return index.get("all_nine_complete") is True and set(index.get("arms", {})) == ALL_ARMS
    if kind == "scoring":
        return completed_index(METRICS / "scoring_source_index.json")
    return report_complete()


def panes():
    output = subprocess.check_output(
        ["tmux", "list-panes", "-a", "-F", "#{session_name}\t#{window_name}\t#{pane_dead}\t#{pane_id}"],
        text=True)
    return {name: {"dead": dead == "1", "pane": pane}
            for session, name, dead, pane in (line.split("\t") for line in output.splitlines())
            if session == "0"}


def gpu_idle(gpu):
    try:
        gpu_uuid = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
        applications = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    except subprocess.CalledProcessError:
        print("GPU occupancy check temporarily unavailable; no launch", gpu, flush=True)
        return False
    return not any(line.split(",")[0].strip() == gpu_uuid for line in applications.splitlines())


def launch(name, job):
    target = subprocess.check_output(
        ["tmux", "new-window", "-d", "-P", "-F", "#{session_name}:#{window_index}",
         "-t", "0:", "-c", str(Path.cwd()), "-n", name,
         job["command"] + " >> " + str(job["log"]) + " 2>&1"], text=True).strip()
    subprocess.run(["tmux", "set-window-option", "-t", target, "remain-on-exit", "off"], check=True)
    return target


def save(state, run):
    state["utc"] = datetime.now(timezone.utc).isoformat()
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(STATE)
    artifact = wandb.Artifact("mlip-bias-zeus-supervision-20261007", type="mlip_bias_compat",
                              metadata={"complete": state.get("complete", False)})
    for path in [STATE, Path(__file__), Path("docs/platforms/zeus/mlip_bias.md")]:
        artifact.add_file(str(path), name=path.name)
    publication = METRICS / "report/supervisor_publication_verification.json"
    if publication.exists():
        artifact.add_file(str(publication), name=publication.name)
    run.log_artifact(artifact).wait()
    run.summary["complete"] = state.get("complete", False)
    run.summary["jobs_requiring_review"] = [name for name, record in state["jobs"].items()
                                           if record.get("review_required")]


def main():
    with (WORK / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read_json(STATE, {"jobs": {}, "complete": False})
        run = wandb.init(entity="symmetry-advantage", project="WyckoffTransformer",
                         id="zeussupervision20261007", resume="allow",
                         job_type="mlip_bias_compat", dir=wandb_dir())
        previous = None
        try:
            while not (WORK / "supervisor.stop").exists():
                live = panes()
                for name, job in JOBS.items():
                    record = state["jobs"].setdefault(name, {})
                    refresh_budget(record, progress(job))
                    try:
                        done = job_done(job)
                    except wandb.errors.Error as exc:
                        record["status"] = "waiting_for_wandb_verification"
                        print(name, "W&B verification temporarily unavailable:", type(exc).__name__, flush=True)
                        continue
                    if paused(job):
                        record["status"] = "paused_by_marker_or_gpu_loan"
                    elif done:
                        record["status"] = "complete"
                    elif name in live and not live[name]["dead"]:
                        record["status"] = "running"
                    elif job["kind"] == "relaxation" and not gpu_idle(job["gpu"]):
                        record["status"] = "waiting_for_gpu_owner_to_exit"
                    elif spend_restart(record):
                        if name in live:
                            subprocess.run(["tmux", "kill-pane", "-t", live[name]["pane"]], check=True)
                        record.update(status="restarted", window=launch(name, job))
                        print(name, "restarted", record, flush=True)
                    else:
                        record["status"] = "review_required_after_no_progress_restarts"
                        print(name, "held after three restarts without durable progress", flush=True)
                state["complete"] = all(record.get("status") == "complete"
                                        for record in state["jobs"].values())
                summary = json.dumps(state, sort_keys=True)
                if summary != previous:
                    state.pop("utc", None)
                    try:
                        save(state, run)
                    except wandb.errors.Error as exc:
                        print("Supervision upload temporarily unavailable:", type(exc).__name__, flush=True)
                        previous = None
                    else:
                        previous = json.dumps(state, sort_keys=True)
                if state["complete"]:
                    print("All relaxations, scores and audited W&B report complete", flush=True)
                    return
                time.sleep(30)
        finally:
            run.finish()


if __name__ == "__main__":
    main()
