"""Score explicitly completed recovery snapshots without changing prior results."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import re
import struct
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from wyckoff_transformer.paths import wandb_dir

STUDY = Path("generated/mlip_bias_study")
PROJECT = "symmetry-advantage/WyckoffTransformer"
ARMS = ("equflashv2", "pet_oam_xl", "tece_oam_rra_1_0", "equiformerv3_dens_oam",
        "esen_30m_oam", "tace_oam_l", "orb_conserv_inf", "prophet_oame_mbd", "nequip_oam_xl")
ARM_MODELS = dict(zip(ARMS, ("EquFlashV2", "PET-OAM-XL", "TECE-OAM-RRA-1.0",
                            "EquiformerV3+DeNS-OAM", "eSEN-30M-OAM", "TACE-OAM-L",
                            "orb_conserv_inf", "Prophet-OAME-MBD", "Nequip-OAM-XL")))
FROZEN_COMMIT = "fbf1ba4855934acb8fe87135315504899064080d"
HISTORICAL = {"equflashv2": "mlip-bias-9b5e07e1c31040ee:v116",
              "pet_oam_xl": "mlip-bias-a4bbbd2f9825a3a3:v130"}
INPUT_HASHES = {"input_sha256": "52583736efd05509c747e27b42c26fabb0104f53723222f8171a5091dc873c17",
                "input_manifest_sha256": "023d6c927dfc634f9b7b88e4a0c5b483cd49abc2141f1d264cf46d2b42b185bc"}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned_artifact(value):
    if not re.fullmatch(r"(?:[\w.-]+/[\w.-]+/)?[\w.-]+:v\d+", value):
        raise ValueError(f"A final pinned artifact version is required: {value!r}")
    return value if "/" in value else PROJECT + "/" + value


def energy_matches_serialized_ledger(selected, recorded):
    """CSV's short float32 decimal and JSON's expanded float can represent one value."""
    if not math.isfinite(selected) or not math.isfinite(recorded):
        return False
    if selected == recorded:
        return True
    try:
        packed = struct.pack("!f", selected)
        return selected == struct.unpack("!f", packed)[0] and packed == struct.pack("!f", recorded)
    except OverflowError:
        return False


def validate_snapshot(directory, expected_ledger_hash=None):
    """Completion requires unique original draws, both exports and verified CIFs."""
    manifest = json.loads((directory / "study_manifest.json").read_text())
    if any(manifest.get(key) != value for key, value in INPUT_HASHES.items()):
        raise ValueError("Snapshot uses different raw pyXtal inputs")
    with (directory / "trials.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = {(int(row["gene"]), int(row["trial"])) for row in rows}
    by_key = {(int(row["gene"]), int(row["trial"])): row for row in rows}
    if len(rows) != 2698 or len(keys) != 2698:
        raise ValueError(f"Snapshot has {len(rows)} rows and {len(keys)} distinct trials, expected2698")
    original_keys = None
    original = STUDY / "equflashv2" / "trials.csv"
    if original.is_file():
        with original.open(newline="") as stream:
            original_keys = {(int(row["gene"]), int(row["trial"])) for row in csv.DictReader(stream)}
    if original_keys is not None and keys != original_keys:
        raise ValueError("Snapshot trial identities differ from the original completed study")
    ledger_hash = sha256(directory / "trials.csv")
    if expected_ledger_hash and expected_ledger_hash != ledger_hash:
        raise ValueError("Final completion marker and uploaded ledger disagree")
    hashes = {}
    for track in ("free", "fixed_symmetry"):
        energy_key, cif_key = (("energy_ev_per_atom", "cif") if track == "free"
                               else ("fixed_energy_ev_per_atom", "fixed_cif"))
        selected = json.loads((directory / f"selected_{track}.json").read_text())
        if not selected:
            raise ValueError("No selected structures")
        hashes[track] = {}
        chosen = {}
        for (gene, trial), ledger_row in by_key.items():
            if ledger_row[energy_key]:
                candidate = (float(ledger_row[energy_key]), trial)
                if not math.isfinite(candidate[0]):
                    raise ValueError("Nonfinite selected-track relaxation energy")
                if gene not in chosen or candidate < chosen[gene]:
                    chosen[gene] = candidate
        if set(map(int, selected)) != set(chosen):
            raise ValueError("Selected genes differ from the authoritative track's ledger")
        for gene, row in selected.items():
            trial_key = (int(gene), int(row["trial"]))
            ledger_row = by_key.get(trial_key)
            expected_energy, expected_trial = chosen[int(gene)]
            if (ledger_row is None or int(row["trial"]) != expected_trial
                    or not energy_matches_serialized_ledger(row["energy_ev_per_atom"], expected_energy)):
                raise ValueError("Selected trial or energy differs from the authoritative track's ledger")
            cif = (directory / row["cif"]).resolve()
            if not cif.is_relative_to(directory.resolve()):
                raise ValueError("Selected CIF path escapes snapshot")
            hashes[track][gene] = sha256(cif)
            source = (directory / ledger_row[cif_key]).resolve()
            if not source.is_relative_to(directory.resolve()) or hashes[track][gene] != sha256(source):
                raise ValueError("Selected CIF differs from the authoritative track's trial CIF")
    return {"manifest": manifest, "trials_sha256": ledger_hash, "selected_cif_hashes": hashes}


def completed_sources(completion_path, completion_only=False):
    sources = ({} if completion_only else
               {arm: {"complete": True, "artifact": artifact} for arm, artifact in HISTORICAL.items()})
    if completion_path.is_file():
        payload = json.loads(completion_path.read_text())
        sources.update(payload["arms"])
    for arm in (() if completion_only else ARMS):
        marker = STUDY / "zeus_migration" / arm / "zeus_relaxation_complete.json"
        if marker.is_file():
            sources[arm] = json.loads(marker.read_text())
    return {arm: record for arm, record in sources.items() if record.get("complete") is True}


def completed_attempt(output, artifact, namespace, hashes):
    """A documented scorer failure is a finished attempt, not a retry invitation."""
    required = ("metrics.json", "single_point_energies.json", "checkpoint_manifest.json",
                "single_point_checkpoint.json.gz", "novelty_cache_checkpoint.json.gz")
    if not all((output / name).is_file() for name in required):
        return False
    metrics = json.loads((output / "metrics.json").read_text())
    identity = json.loads((output / "checkpoint_manifest.json").read_text())
    return (metrics["arm_complete"] and metrics["arm_artifact"] == artifact
            and metrics.get("snapshot_namespace") == namespace
            and metrics["lemat_genbench_commit"] == FROZEN_COMMIT
            and identity["selected_cif_hashes"] == hashes
            and set(metrics["results"]) == {"validity", "distribution", "diversity", "novelty",
                                           "uniqueness", "hhi", "sun", "stability"})


def publish_index(index_path, namespace):
    import wandb

    digest = hashlib.sha256(namespace.encode()).hexdigest()
    run = wandb.init(entity="symmetry-advantage", project="WyckoffTransformer",
                     id="pins" + digest[:4], resume="allow", dir=wandb_dir(),
                     job_type="mlip_bias_compat", config={"snapshot_namespace": namespace})
    try:
        artifact = wandb.Artifact("mlip-bias-genbench-pins-" + digest[:16], type="mlip_bias_compat")
        artifact.add_file(str(index_path), name="scoring_source_index.json")
        for path in [Path("scripts/run_mlip_bias_genbench.py"),
                     Path("scripts/platforms/zeus/watch_mlip_bias_genbench.py"),
                     Path("scripts/platforms/zeus/run_updated_mlip_bias_genbench.sh"),
                     Path("scripts/platforms/zeus/report_updated_mlip_bias_genbench.sh"),
                     Path("scripts/compute_mlip_bias_common_genbench.py"),
                     Path("scripts/report_mlip_bias_genbench.py"),
                     Path("scripts/verify_mlip_bias_genbench_report.py"),
                     Path("docs/platforms/zeus/updated_genbench.md"),
                     Path("src/wyckoff_transformer/tests/test_mlip_bias_genbench.py"),
                     Path("src/wyckoff_transformer/tests/test_zeus_genbench_snapshot.py"),
                     Path("src/wyckoff_transformer/tests/test_mlip_bias_genbench_report.py")]:
            artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact).wait()
    finally:
        run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--namespace", default="recovery-20261006")
    parser.add_argument("--completion", type=Path,
                        default=STUDY / "zeus_migration" / "study_relaxation_completion.json")
    parser.add_argument("--completion-only", action="store_true",
                        help="Use only the specified completion index, without historical local markers")
    parser.add_argument("--reuse-source-index", type=Path,
                        help="Reuse frozen single points from a prior index only when CIF hashes match")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true", help="Process currently ready arms, then exit")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.namespace) or args.workers < 1 or args.poll_seconds < 1:
        parser.error("Use a safe namespace, positive worker count and polling interval")
    root = STUDY / "metrics" / args.namespace
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "scoring_source_index.json"
    # One coordinator per namespace; never write the same scoring outputs from two GPUs.
    coordinator_lock = (root / "coordinator.lock").open("a")
    try:
        fcntl.flock(coordinator_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another scoring coordinator owns this namespace")
    index = (json.loads(index_path.read_text()) if index_path.exists()
             else {"snapshot_namespace": args.namespace, "arms": {}})
    reuse = (json.loads(args.reuse_source_index.read_text())["arms"]
             if args.reuse_source_index else {})
    assert index["snapshot_namespace"] == args.namespace
    lemat = STUDY / "deps" / "lemat_genbench"
    if subprocess.check_output(["git", "-C", str(lemat), "rev-parse", "HEAD"], text=True).strip() != FROZEN_COMMIT:
        raise ValueError("LeMat-GenBench checkout is not the frozen study version")
    stop = STUDY / f"zeus_genbench_gpu{args.gpu}.stop"
    while not stop.exists():
        ready = completed_sources(args.completion, args.completion_only)
        for arm, previous in index["arms"].items():
            if arm in ready and pinned_artifact(ready[arm]["artifact"]) != previous["artifact"]:
                raise ValueError(f"Completed artifact changed for {arm}; choose a new scoring namespace")
            for track, path in previous["outputs"].items():
                if sha256(Path(path) / "metrics.json") != previous["metrics_sha256"][track]:
                    raise ValueError(f"Recorded scoring output changed for {arm}/{track}")
        progressed = False
        for arm in args.arms:
            if stop.exists():
                return
            if arm in index["arms"] or arm not in ready:
                continue
            with (STUDY / f"zeus_gpu{args.gpu}.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                # Refresh readiness after locking; no live local relaxation writer can mutate its ledger.
                source = completed_sources(args.completion, args.completion_only).get(arm)
                if source is None:
                    continue
                artifact = pinned_artifact(source["artifact"])
                version = artifact.rsplit(":", 1)[1]
                directory = root / "snapshots" / arm / version
                snapshot_marker = directory / "snapshot_pinned.json"
                if not snapshot_marker.exists():
                    import wandb

                    remote = wandb.Api().artifact(artifact, type="mlip_bias_relax")
                    actual = PROJECT + "/" + remote.name.split(":")[0] + ":" + remote.version
                    if actual != artifact:
                        raise ValueError("Requested final pinned version resolved to different source")
                    remote.download(root=str(directory))
                validation = validate_snapshot(directory, source.get("ledger_sha256"))
                if validation["manifest"]["mlip"] != ARM_MODELS[arm]:
                    raise ValueError(f"Pinned artifact contains a different model for {arm}")
                pinned_identity = {"artifact": artifact, "validation": validation}
                if snapshot_marker.exists() and json.loads(snapshot_marker.read_text()) != pinned_identity:
                    raise ValueError("Pinned local snapshot changed")
                atomic_json(snapshot_marker, pinned_identity)
                outputs = {}
                for track in ("free", "fixed_symmetry"):
                    output = root / arm / version / track
                    legacy = STUDY / "metrics" / arm / track
                    if arm in HISTORICAL and artifact == pinned_artifact(HISTORICAL[arm]) and completed_attempt(
                            legacy, artifact, None, validation["selected_cif_hashes"][track]):
                        output = legacy
                    elif not completed_attempt(output, artifact, args.namespace,
                                               validation["selected_cif_hashes"][track]):
                        output.mkdir(parents=True, exist_ok=True)
                        command = [".venv/bin/python", "scripts/run_mlip_bias_genbench.py", "--arm", str(directory),
                                   "--track", track, "--arm-artifact", artifact, "--snapshot-namespace", args.namespace,
                                   "--lemat-root", str(lemat), "--output", str(output), "--workers", str(args.workers)]
                        if arm in reuse:
                            reused_output = Path(reuse[arm]["outputs"][track])
                            if sha256(reused_output / "metrics.json") != reuse[arm]["metrics_sha256"][track]:
                                raise ValueError("Reusable scoring source changed")
                            command.extend(["--reuse-scores-from", str(reused_output)])
                        elif track == "fixed_symmetry":
                            command.extend(["--reuse-scores-from", outputs["free"]])
                        print(f"{datetime.now(timezone.utc).isoformat()} scoring {arm}/{track} {artifact}", flush=True)
                        lock_fd = lock.fileno()
                        with (output / "scoring.log").open("a") as log:
                            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                                           pass_fds=(lock_fd,))
                        if not completed_attempt(output, artifact, args.namespace,
                                                 validation["selected_cif_hashes"][track]):
                            raise RuntimeError(f"Scoring attempt did not finish: {arm}/{track}")
                    outputs[track] = str(output)
                index["arms"][arm] = {"artifact": artifact, "snapshot": str(directory),
                                      "validation": validation, "outputs": outputs,
                                      "metrics_sha256": {track: sha256(Path(path) / "metrics.json")
                                                         for track, path in outputs.items()},
                                      "completion_evidence": source}
                index["date_utc"] = datetime.now(timezone.utc).isoformat()
                atomic_json(index_path, index)
                publish_index(index_path, args.namespace)
                progressed = True
        if all(arm in index["arms"] for arm in args.arms):
            print("All requested snapshots scored; source index:", index_path, flush=True)
            return
        if args.once:
            return
        if not progressed:
            print("Waiting for final recovery markers and the GPU study lock", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
