"""The study's resumable snapshots keep every trial that can win selection."""

import csv
import json
import shutil
import time
from pathlib import Path

import pytest
import wandb

from scripts.run_mlip_bias_relax import (
    FIELDS,
    apply_historical_timeout,
    load_rows,
    restore_from_wandb,
    resolve_relax_timeout,
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
