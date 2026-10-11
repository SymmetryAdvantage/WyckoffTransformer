"""Failure recovery preserves the published baseline and resumes without repeated trials."""

import json
from pathlib import Path

import pytest

from scripts.mlip_bias_factories import _with_neighbor_capacity
from scripts.platforms.zeus import retry_mlip_bias_failures as recovery
from scripts.run_mlip_bias_relax import FIELDS, load_rows, sha256_file, wandb_identity


def test_completion_pins_the_resolved_version_instead_of_latest_alias():
    from types import SimpleNamespace

    assert recovery.artifact_pin(SimpleNamespace(name="relaxation:latest", version="v4")) == "relaxation:v4"


def test_scalar_array_calculator_energy_serializes_without_precision_loss():
    import numpy as np

    row = recovery.scalar_trial_row({"energy_ev": np.array(-308.19946, dtype=np.float32),
                                    "energy_ev_per_atom": np.float32(-2.9634564), "status": "ok"})
    assert json.loads(json.dumps(row)) == row
    assert row["energy_ev"] == -308.199462890625
    with pytest.raises(ValueError):
        recovery.scalar_trial_row({"energy_ev": np.array([1.0, 2.0])})


def test_capacity_growth_preserves_edges_and_propagates_other_errors():
    class NeighborOverflowError(Exception):
        def __init__(self, required):
            self.num_neighbors = required

    capacities = []
    edges = object()

    def neighbors(positions, *, cutoff, max_neighbors):
        capacities.append(max_neighbors)
        assert cutoff == 6.0
        if max_neighbors < 12000:
            raise NeighborOverflowError(12000)
        return edges

    assert _with_neighbor_capacity(neighbors)(object(), cutoff=6.0, max_neighbors=500) is edges
    assert capacities == [4096, 12000]
    with pytest.raises(NeighborOverflowError):
        _with_neighbor_capacity(neighbors, maximum=8192)(object(), cutoff=6.0)

    def bad(*args, **kwargs):
        raise RuntimeError("CUDA allocation failed")

    with pytest.raises(RuntimeError, match="CUDA allocation"):
        _with_neighbor_capacity(bad)(object())


def test_recovery_preserves_baseline_and_completed_trial_on_resume(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    rows = {}
    for trial in (0, 1):
        row = dict.fromkeys(FIELDS, "")
        row.update(gene=3, trial=trial, status="ok" if trial == 0 else "failed",
                   n_atoms=2, formula="Si2")
        if trial == 0:
            row.update(cif="kept.cif", fixed_cif="fixed.cif", energy_ev_per_atom=-1,
                       fixed_energy_ev_per_atom=-0.9)
        else:
            row["error"] = "OutOfMemoryError: baseline"
        rows[(3, trial)] = row
    for name in ("kept.cif", "fixed.cif"):
        (source / name).write_text("original " + name)
    recovery.replace_ledger(source, rows)
    original_bytes = (source / "trials.csv").read_bytes()
    baseline = {"manifest": {"mlip": "Si", "calculator_factory": None},
                "trials_sha256": sha256_file(source / "trials.csv"), "selected_cif_hashes": {}}
    monkeypatch.setattr(recovery, "validate_snapshot", lambda *args: baseline)
    spec = {"source_directory": str(source), "artifact": "baseline:v1", "factory": None}
    output = tmp_path / "recovery"
    settings = recovery.prepare_arm(output, spec, "new-pass")
    assert wandb_identity(baseline["manifest"]) != wandb_identity(json.loads((output / "study_manifest.json").read_text()))
    retried = {**load_rows(output / "trials.csv")[(3, 1)], "error": "OutOfMemoryError: retry"}
    recovery.commit_trial(output, settings, retried)
    resumed = recovery.prepare_arm(output, spec, "new-pass")
    assert resumed["completed"] == [[3, 1]]
    assert load_rows(output / "trials.csv")[(3, 1)]["error"] == "OutOfMemoryError: retry"
    assert (source / "trials.csv").read_bytes() == original_bytes
    (output / "kept.cif").write_text("new CIF bytes")
    assert (source / "kept.cif").read_text() == "original kept.cif"
    with pytest.raises(ValueError, match="plan changed"):
        recovery.prepare_arm(output, {**spec, "artifact": "other:v0"}, "new-pass")
    with pytest.raises(ValueError, match="outside the frozen"):
        recovery.commit_trial(output, settings, {**retried, "gene": 10})


def test_only_prophet_optimized_factory_enables_requested_flags(monkeypatch):
    import sys
    import types

    from scripts import mlip_bias_factories as factories

    prophet = types.ModuleType("prophet")
    prophet.KairosCalculator = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "prophet", prophet)
    monkeypatch.setattr(factories, "resolve_model_path", lambda url: Path("checkpoint.pt"))
    baseline = factories.build_prophet(device="cuda")
    proposed = factories.build_prophet_kernel_compile(device="cuda")
    assert (baseline["use_kernel"], baseline["use_compile"]) == (False, False)
    assert (proposed["use_kernel"], proposed["use_compile"]) == (True, True)
    assert baseline["model_path"] == proposed["model_path"]


def test_supervision_bounds_missing_jobs_and_resets_only_on_durable_progress(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts.platforms.zeus import monitor_mlip_failure_pass as monitor

    monkeypatch.setattr(monitor, "STUDY", tmp_path)
    monkeypatch.setattr(monitor.subprocess, "check_output", lambda *args, **kwargs: "")
    launches = []
    monkeypatch.setattr(monitor.subprocess, "run", lambda cmd, **kwargs: launches.append(cmd))
    monkeypatch.setattr(monitor, "gpu_idle", lambda gpu: True)
    run = SimpleNamespace(summary={})
    payload = {"namespace": "new-pass", "arms": {}, "progress": {}}
    for _ in range(4):
        state = monitor.supervise_jobs(tmp_path, tmp_path / "plan.json", payload, run)
    assert len(launches) == 9  # Three jobs, each with three bounded attempts.
    assert all(record["status"] == "review_required_after_no_progress_restarts" for record in state["jobs"].values())
    payload["progress"]["nequip_oam_xl"] = {"completed": 1, "planned": 40}
    state = monitor.supervise_jobs(tmp_path, tmp_path / "plan.json", payload, run)
    assert state["jobs"]["mlip-failures-gpu0"]["status"] == "restarted"
    assert state["jobs"]["mlip-failures-gpu1"]["status"] == "review_required_after_no_progress_restarts"
    (tmp_path / "gpu0.stop").touch()
    state = monitor.supervise_jobs(tmp_path, tmp_path / "plan.json", payload, run)
    assert state["jobs"]["mlip-failures-gpu0"]["status"] == "stopped_by_marker"
