"""Mirror final recovery evidence from both machines into a pinned study index."""

import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import wandb

from wyckoff_transformer.paths import wandb_dir

STUDY = Path("generated/mlip_bias_study")
WORK = STUDY / "zeus_migration"
HISTORICAL = {
    "equflashv2": "mlip-bias-9b5e07e1c31040ee:v116",
    "pet_oam_xl": "mlip-bias-a4bbbd2f9825a3a3:v130",
}
LOCAL_ARMS = ("tece_oam_rra_1_0", "equiformerv3_dens_oam", "esen_30m_oam",
              "tace_oam_l", "orb_conserv_inf", "prophet_oame_mbd")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def remote_final():
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "iapetus",
         "cat /home/kna/WyckoffTransformer/generated/mlip_bias_study/nequip_iapetus_final_20261006.json"],
        capture_output=True, text=True, timeout=30)
    if result.returncode:
        return None
    marker = json.loads(result.stdout)
    if marker.get("complete") is not True:
        return None
    if not re.fullmatch(r"[^:]+:v\d+", marker["artifact"]):
        raise ValueError("NequIP requires a final pinned artifact version")
    artifact_name = marker["artifact"]
    if "/" not in artifact_name:
        artifact_name = "symmetry-advantage/WyckoffTransformer/" + artifact_name
    artifact = wandb.Api().artifact(artifact_name, type="mlip_bias_relax")
    downloaded = Path(artifact.get_entry("trials.csv").download(root=str(WORK / "nequip_verification")))
    if hashlib.sha256(downloaded.read_bytes()).hexdigest() != marker["ledger_sha256"]:
        raise ValueError("NequIP final completion marker and W&B ledger differ")
    atomic_json(WORK / "nequip_iapetus_final_20261006.json", marker)
    return marker


def main():
    path = WORK / "study_relaxation_completion.json"
    remote = None
    previous = None
    run = wandb.init(entity="symmetry-advantage", project="WyckoffTransformer",
                     id="mlipstudycompletion20261006", resume="allow",
                     job_type="mlip_bias_compat", dir=wandb_dir())
    try:
        while not (WORK / "completion_monitor.stop").exists():
            arms = {arm: dict(complete=True, artifact=artifact) for arm, artifact in HISTORICAL.items()}
            for arm in LOCAL_ARMS:
                marker = WORK / arm / "zeus_relaxation_complete.json"
                if marker.is_file():
                    record = json.loads(marker.read_text())
                    if record.get("complete") is True:
                        arms[arm] = record
            if remote is None:
                try:
                    remote = remote_final()
                except (subprocess.TimeoutExpired, OSError, wandb.errors.CommError) as exc:
                    print(f"Iapetus marker temporarily unavailable: {type(exc).__name__}", flush=True)
            if remote is not None:
                arms["nequip_oam_xl"] = remote
            if arms != previous:
                payload = dict(utc=datetime.now(timezone.utc).isoformat(), arms=arms,
                               all_nine_complete=len(arms) == 9)
                atomic_json(path, payload)
                artifact = wandb.Artifact("mlip-bias-final-relaxation-index-20261006",
                                          type="mlip_bias_compat", metadata={"completed_arms": len(arms)})
                artifact.add_file(str(path), name=path.name)
                artifact.add_file("scripts/platforms/zeus/monitor_mlip_bias_completion.py",
                                  name="monitor_mlip_bias_completion.py")
                run.log_artifact(artifact).wait()
                run.summary["complete_relaxation_arms"] = len(arms)
                run.summary["all_nine_complete"] = len(arms) == 9
                print(f"Final recovery snapshots: {len(arms)}/9", flush=True)
                previous = arms
            if len(arms) == 9:
                return
            time.sleep(30)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
