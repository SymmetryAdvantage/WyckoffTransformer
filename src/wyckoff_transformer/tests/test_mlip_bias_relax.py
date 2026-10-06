"""The study's resumable snapshots keep every trial that can win selection."""

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import wandb

from scripts.run_mlip_bias_relax import (
    FIELDS,
    apply_historical_timeout,
    is_oom_or_timeout,
    load_rows,
    prepare_recovery,
    record_execution,
    resolve_relax_timeout,
    restore_from_wandb,
    select_structures,
    snapshot_to_wandb,
    trial_timeout,
    wandb_identity,
)


def test_snapshot_restores_trial_ledger_and_both_cifs(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "study_manifest.json").write_text(json.dumps({"mlip": "test"}))
    rows = {}
    for trial, energy in [(0, -1.0), (1, -2.0)]:
        directory = source / "trials" / "0" / f"trial-{trial}"
        directory.mkdir(parents=True)
        for name in ("kept.cif", "fixed.cif"):
            (directory / name).write_text(f"trial {trial} {name}")
        row = dict.fromkeys(FIELDS, "")
        row.update(gene=0, trial=trial, status="ok", formula="Si2", n_atoms=2,
                   energy_ev=energy * 2, energy_ev_per_atom=energy,
                   fixed_energy_ev=energy * 2,
                   fixed_energy_ev_per_atom=energy,
                   cif=f"trials/0/trial-{trial}/kept.cif",
                   fixed_cif=f"trials/0/trial-{trial}/fixed.cif")
        rows[(0, trial)] = row
    with (source / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows.values())
    select_structures(source, rows)
    resolve_relax_timeout(source, 300, 600, rows)
    record_execution(source, 2, "cuda", rows)
    prepare_recovery(source, rows, 3600, 1)

    class FakeArtifact:
        def __init__(self, *args, **kwargs):
            self.files = {}

        def add_file(self, path, name):
            self.files[name] = Path(path)

        def add_dir(self, path, name):
            for file in Path(path).iterdir():
                self.files[f"{name}/{file.name}"] = file

    class FakeRun:
        summary = {}

        def log_artifact(self, artifact):
            self.artifact = artifact
            return self

        def wait(self):
            return self

    monkeypatch.setattr(wandb, "Artifact", FakeArtifact)
    run = FakeRun()
    snapshot_to_wandb(run, source, "test-artifact", rows)
    assert "trials/0/trial-0/kept.cif" in run.artifact.files
    assert "trials/0/trial-1/kept.cif" in run.artifact.files
    assert "cifs/0.cif" in run.artifact.files
    assert "execution_settings.json" in run.artifact.files
    assert "recovery_settings.json" in run.artifact.files

    class RemoteArtifact:
        name = "test-artifact:v0"

        def download(self, root):
            for relative, file in run.artifact.files.items():
                target = Path(root) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, target)

    class FakeApi:
        def artifact(self, *args, **kwargs):
            return RemoteArtifact()

    monkeypatch.setattr(wandb, "Api", FakeApi)
    restored = tmp_path / "restored"
    assert restore_from_wandb(restored, "test-artifact", entity="e", project="p")
    loaded = load_rows(restored / "trials.csv")
    assert len(loaded) == 2
    assert resolve_relax_timeout(restored, 300, None, loaded) == 600
    execution = json.loads((restored / "execution_settings.json").read_text())
    assert execution["current"]["workers"] == 2
    select_structures(restored, loaded)
    assert (restored / "cifs" / "0.cif").read_text() == "trial 1 kept.cif"


def test_wandb_identity_changes_with_checkpoint():
    one = {"mlip": "test", "checkpoint_id": "a"}
    two = {"mlip": "test", "checkpoint_id": "b"}
    assert wandb_identity(one) != wandb_identity(two)


def test_trial_timeout_interrupts_and_clears_alarm():
    with pytest.raises(TimeoutError):
        with trial_timeout(0.01):
            time.sleep(0.1)
    with trial_timeout(0.1):
        time.sleep(0.02)


def test_increased_timeout_survives_resume_and_preserves_long_success(tmp_path):
    rows_path = tmp_path / "trials.csv"
    rows = {(1, 0): {"status": "ok", "seconds": "450", "energy_ev": "-4"}}
    assert resolve_relax_timeout(tmp_path, 300, 600, rows) == 600
    effective = resolve_relax_timeout(tmp_path, 300, None, rows)
    assert effective == 600
    assert apply_historical_timeout(rows_path, rows, effective) == 0
    assert rows[(1, 0)]["status"] == "ok"
    settings = json.loads((tmp_path / "relaxation_settings.json").read_text())
    assert len(settings["history"]) == 1
    assert settings["history"][0]["completed_trials_at_change"] == 1


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_override_is_rejected(tmp_path, timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        resolve_relax_timeout(tmp_path, 300, timeout, {})


def test_historical_timeout_excludes_old_winner(tmp_path):
    rows_path = tmp_path / "trials.csv"
    trial_dir = tmp_path / "trials" / "0" / "trial-0"
    trial_dir.mkdir(parents=True)
    (trial_dir / "kept.cif").write_text("late trial")
    row = dict.fromkeys(FIELDS, "")
    row.update(gene=0, trial=0, status="ok", formula="Si2", n_atoms=2,
               energy_ev=-4.0, energy_ev_per_atom=-2.0,
               cif="trials/0/trial-0/kept.cif", seconds=301)
    with rows_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)
    rows = load_rows(rows_path)
    select_structures(tmp_path, rows)
    assert (tmp_path / "cifs" / "0.cif").exists()

    assert apply_historical_timeout(rows_path, rows, 300) == 1
    select_structures(tmp_path, rows)
    assert rows[(0, 0)]["status"] == "failed"
    assert rows[(0, 0)]["energy_ev"] == ""
    assert not (tmp_path / "cifs" / "0.cif").exists()


@pytest.mark.parametrize(("error", "eligible"), [
    ("TimeoutError: Relaxation exceeded 1200 seconds", True),
    ("RuntimeError: CUDA out of memory", True),
    ("RuntimeError: TorchScript failed\nOutOfMemoryError: allocation failed", True),
    ("RuntimeError: Reason=cudaErrorMemoryAllocation:out of memory", True),
    ("RuntimeError: Failed to allocate 36 bytes on device 'cuda:0'", True),
    ("NeighborOverflowError: 502 > 500", False),
    ("RuntimeError: CUDA illegal memory access", False),
])
def test_recovery_failure_filter(error, eligible):
    assert is_oom_or_timeout({"status": "failed", "error": error}) == eligible
    assert not is_oom_or_timeout({"status": "ok", "error": error})


def test_recovery_plan_handles_interruption_before_and_after_ledger_write(tmp_path):
    row = {"gene": "1", "trial": "0", "status": "failed", "error": "CUDA out of memory"}
    rows = {(1, 0): row.copy()}
    first = prepare_recovery(tmp_path, rows, 3600, 1)
    assert first["history"][-1]["completed"] == []
    # Original rows retained on a crash before requeue remain pending.
    assert prepare_recovery(tmp_path, rows, 3600, 1) == first
    # A replacement flushed just before a crash is a finished attempt.
    rows[(1, 0)]["error"] = "TimeoutError: Relaxation exceeded 3600 seconds"
    saved = prepare_recovery(tmp_path, rows, 3600, 1)
    assert saved["history"][-1]["completed"] == [[1, 0]]
    assert len(saved["history"]) == 1


def test_spawned_workers_timeout_and_resume_without_duplicate_trials(tmp_path):
    """Real child processes isolate a timeout and leave one recoverable ledger."""
    from ase import Atoms
    from ase.io import write

    source = tmp_path / "input"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"relax_timeout": 2}))
    draws = []
    for gene, symbol in enumerate(("Si", "Ne", "Si", "Si")):
        atoms = Atoms(symbol + "2", scaled_positions=[[0, 0, 0], [.25, .25, .25]],
                      cell=[5.43] * 3, pbc=True)
        atoms.info.update(gene=gene, trial=0)
        draws.append(atoms)
    write(source / "pyxtal.extxyz", draws, format="extxyz")
    (tmp_path / "worker_probe.py").write_text('''
import os, time
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
class Probe(Calculator):
    implemented_properties = ["energy", "forces", "stress"]
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        with open(os.environ["WORKER_PROBE_PIDS"], "a") as stream:
            stream.write(str(os.getpid()) + "\\n")
        time.sleep(float(os.environ.get("WORKER_PROBE_DELAY", "10")) if atoms[0].symbol == "Ne" else .01)
        self.results = dict(energy=-float(len(atoms)), forces=np.zeros((len(atoms), 3)),
                            stress=np.zeros(6))
def build(*, device):
    return Probe()
''')
    root = Path(__file__).resolve().parents[3]
    pids = tmp_path / "worker_pids.txt"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONPATH=str(tmp_path), WORKER_PROBE_PIDS=str(pids))
    output = tmp_path / "output"
    command = [sys.executable, str(root / "scripts/run_mlip_bias_relax.py"),
               "--input", str(source), "--output", str(output), "--mlip", "probe",
               "--calculator-factory", "worker_probe:build", "--checkpoint-id", "probe-v1",
               "--workers", "2", "--max-trials", "4", "--no-wandb"]
    result = subprocess.run(command, cwd=root, env=environment, capture_output=True,
                            text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = load_rows(output / "trials.csv")
    assert set(rows) == {(gene, 0) for gene in range(4)}
    assert rows[(1, 0)]["status"] == "failed"
    assert "TimeoutError" in rows[(1, 0)]["error"]
    for gene in (0, 2, 3):
        assert rows[(gene, 0)]["status"] == "ok"
        assert (output / rows[(gene, 0)]["cif"]).is_file()
        assert (output / rows[(gene, 0)]["fixed_cif"]).is_file()
    assert len(set(pids.read_text().splitlines())) == 2
    original = (output / "trials.csv").read_bytes()
    before_pids = pids.read_text()
    resumed = subprocess.run(command, cwd=root, env=environment, capture_output=True,
                             text=True, timeout=60, check=False)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (output / "trials.csv").read_bytes() == original
    assert pids.read_text() == before_pids

    # A longer budget retries the timeout without redoing successes or OOMs.
    rows[(2, 0)].update(status="failed", error="RuntimeError: CUDA out of memory")
    with (output / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows.values())
    environment["WORKER_PROBE_DELAY"] = "0.01"
    retried = subprocess.run(command + ["--retry-timeouts", "--relax-timeout", "4"],
                             cwd=root, env=environment, capture_output=True,
                             text=True, timeout=60, check=False)
    assert retried.returncode == 0, retried.stdout + retried.stderr
    updated = load_rows(output / "trials.csv")
    assert updated[(1, 0)]["status"] == "ok"
    for gene in (0, 2, 3):
        assert updated[(gene, 0)] == rows[(gene, 0)]
    assert len(updated) == 4
    settings = json.loads((output / "relaxation_settings.json").read_text())
    assert settings["relax_timeout_s"] == 4
    assert settings["history"][-1]["previous_timeout_s"] == 2

    updated[(1, 0)].update(status="failed", error="TimeoutError: previous attempt")
    updated[(3, 0)].update(status="failed", error="NeighborOverflowError: 502 > 500")
    with (output / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(updated.values())
    environment["WORKER_PROBE_DELAY"] = "10"
    recovery_command = command + ["--workers", "1", "--retry-oom-timeouts", "--relax-timeout", "4"]
    recovered = subprocess.run(recovery_command, cwd=root, env=environment, capture_output=True,
                               text=True, timeout=60, check=False)
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    final = load_rows(output / "trials.csv")
    assert final[(1, 0)]["status"] == "failed"
    assert "TimeoutError" in final[(1, 0)]["error"]
    assert final[(2, 0)]["status"] == "ok"
    assert final[(0, 0)] == updated[(0, 0)]
    assert final[(3, 0)] == updated[(3, 0)]
    assert len(final) == 4
    saved = json.loads((output / "recovery_settings.json").read_text())
    assert saved["history"][-1]["workers"] == 1
    assert set(map(tuple, saved["history"][-1]["completed"])) == {(1, 0), (2, 0)}
    before_pids = pids.read_text()
    resumed = subprocess.run(recovery_command, cwd=root, env=environment, capture_output=True,
                             text=True, timeout=60, check=False)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert load_rows(output / "trials.csv") == final
    assert pids.read_text() == before_pids
