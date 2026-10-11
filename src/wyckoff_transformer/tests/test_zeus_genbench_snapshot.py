"""Do not score batch snapshots or silently change a completed scoring attempt."""

import json

import pytest

from scripts.platforms.zeus import watch_mlip_bias_genbench as watcher


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher, "STUDY", tmp_path)
    directory = tmp_path / "snapshot"
    directory.mkdir()
    (directory / "study_manifest.json").write_text(json.dumps(watcher.INPUT_HASHES))
    (directory / "trials.csv").write_text(
        "gene,trial,status,energy_ev_per_atom,cif,fixed_energy_ev_per_atom,fixed_cif\n" +
        "".join(f"{gene},0,ok,-1,source.cif,-1,source.cif\n" if gene in (1, 2)
                else f"{gene},0,ok,,,,\n" for gene in range(2698)))
    (directory / "one.cif").write_text("immutable-CIF")
    (directory / "source.cif").write_text("immutable-CIF")
    for track in ("free", "fixed_symmetry"):
        (directory / f"selected_{track}.json").write_text(json.dumps(
            {str(gene): {"trial": 0, "cif": "one.cif", "energy_ev_per_atom": -1} for gene in (1, 2)}))
    return directory


def test_snapshot_requires_complete_unique_ledger_and_matching_cifs(snapshot):
    payload = watcher.validate_snapshot(snapshot)
    assert payload["selected_cif_hashes"]["free"] == payload["selected_cif_hashes"]["fixed_symmetry"]
    watcher.validate_snapshot(snapshot, payload["trials_sha256"])
    with pytest.raises(ValueError, match="uploaded ledger disagree"):
        watcher.validate_snapshot(snapshot, "wrong-ledger")
    original = (snapshot / "trials.csv").read_text()
    (snapshot / "trials.csv").write_text(original.replace("2697,0,ok", "2696,0,ok"))
    with pytest.raises(ValueError, match="distinct trials"):
        watcher.validate_snapshot(snapshot)
    (snapshot / "trials.csv").write_text(original.replace("2697,0,ok,,,,\n", ""))
    with pytest.raises(ValueError, match="expected2698"):
        watcher.validate_snapshot(snapshot)


def test_snapshot_keeps_fixed_only_failed_free_trial_and_checks_track_energy(snapshot):
    ledger = snapshot / "trials.csv"
    ledger.write_text(ledger.read_text().replace("1,0,ok,-1,source.cif,-1,source.cif",
                                                "1,0,failed,,,-1,source.cif"))
    selected_free = json.loads((snapshot / "selected_free.json").read_text())
    del selected_free["1"]
    (snapshot / "selected_free.json").write_text(json.dumps(selected_free))
    validation = watcher.validate_snapshot(snapshot)
    assert "1" not in validation["selected_cif_hashes"]["free"]
    assert "1" in validation["selected_cif_hashes"]["fixed_symmetry"]
    fixed = json.loads((snapshot / "selected_fixed_symmetry.json").read_text())
    fixed["1"]["energy_ev_per_atom"] = -2
    (snapshot / "selected_fixed_symmetry.json").write_text(json.dumps(fixed))
    with pytest.raises(ValueError, match="trial or energy differs"):
        watcher.validate_snapshot(snapshot)
    (snapshot / "study_manifest.json").write_text(json.dumps({**watcher.INPUT_HASHES, "input_sha256": "different"}))
    with pytest.raises(ValueError, match="different raw pyXtal"):
        watcher.validate_snapshot(snapshot)


def test_completion_requires_explicit_final_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher, "STUDY", tmp_path)
    completion = tmp_path / "completion.json"
    completion.write_text(json.dumps({"arms": {"nequip_oam_xl": {"complete": False, "artifact": "arm:v1"},
                                               "orb_conserv_inf": {"complete": True, "artifact": "orb:v2"}}}))
    ready = watcher.completed_sources(completion)
    assert "nequip_oam_xl" not in ready
    assert ready["orb_conserv_inf"]["artifact"] == "orb:v2"
    marker = tmp_path / "zeus_migration" / "orb_conserv_inf" / "zeus_relaxation_complete.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"complete": False, "artifact": "orb:v3"}))
    assert "orb_conserv_inf" not in watcher.completed_sources(completion)
    with pytest.raises(ValueError, match="pinned artifact"):
        watcher.pinned_artifact("orb:latest")
    assert watcher.pinned_artifact("orb:v2") == watcher.PROJECT + "/orb:v2"


def test_separate_failure_pass_ignores_old_completion_markers(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher, "STUDY", tmp_path)
    completion = tmp_path / "completion.json"
    completion.write_text(json.dumps({"arms": {"pet_oam_xl": {"complete": True, "artifact": "new-pet:v3"}}}))
    marker = tmp_path / "zeus_migration" / "pet_oam_xl" / "zeus_relaxation_complete.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"complete": True, "artifact": "old-pet:v2"}))
    assert watcher.completed_sources(completion, completion_only=True) == {
        "pet_oam_xl": {"complete": True, "artifact": "new-pet:v3"}}


def test_float32_csv_round_trip_matches_json_but_rejects_changed_energy(snapshot):
    ledger = snapshot / "trials.csv"
    ledger.write_text(ledger.read_text().replace("1,0,ok,-1,source.cif", "1,0,ok,-1.8897468,source.cif"))
    path = snapshot / "selected_free.json"
    selected = json.loads(path.read_text())
    selected["1"]["energy_ev_per_atom"] = -1.8897467851638794
    path.write_text(json.dumps(selected))
    watcher.validate_snapshot(snapshot)
    # The adjacent float32 value is a real energy change, not serialization.
    for changed_energy in (-1.889746904373169, -1.889746786):
        selected["1"]["energy_ev_per_atom"] = changed_energy
        path.write_text(json.dumps(selected))
        with pytest.raises(ValueError, match="trial or energy differs"):
            watcher.validate_snapshot(snapshot)


def test_documented_scoring_failure_finishes_attempt_without_claiming_complete(snapshot):
    hashes = watcher.validate_snapshot(snapshot)["selected_cif_hashes"]["free"]
    metadata = {"arm_complete": True, "final_results": False, "scoring_complete": False,
                "arm_artifact": "entity/project/arm:v2", "snapshot_namespace": "recovery",
                "lemat_genbench_commit": watcher.FROZEN_COMMIT,
                "results": dict.fromkeys(["validity", "distribution", "diversity", "novelty",
                                          "uniqueness", "hhi", "sun", "stability"], {})}
    (snapshot / "metrics.json").write_text(json.dumps(metadata))
    (snapshot / "checkpoint_manifest.json").write_text(json.dumps({"selected_cif_hashes": hashes}))
    (snapshot / "single_point_energies.json").write_text("[]")
    for name in ("single_point_checkpoint.json.gz", "novelty_cache_checkpoint.json.gz"):
        (snapshot / name).touch()
    assert watcher.completed_attempt(snapshot, "entity/project/arm:v2", "recovery", hashes)
    assert not watcher.completed_attempt(snapshot, "entity/project/arm:v3", "recovery", hashes)
    assert not watcher.completed_attempt(snapshot, "entity/project/arm:v2", "another", hashes)
    assert not watcher.completed_attempt(snapshot, "entity/project/arm:v2", "recovery", {"1": "changed"})
