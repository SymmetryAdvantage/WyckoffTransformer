"""The updated report derives its cohort and failure audit from its inputs."""

import hashlib
import json

import pytest
from pymatgen.core import Lattice, Structure

from scripts import verify_mlip_bias_genbench_report as audit


def test_audit_derives_mask_and_failures_without_historical_gene_assumptions(tmp_path, monkeypatch):
    index = {"snapshot_namespace": "new", "arms": {"new_model": {"outputs": {}, "metrics_sha256": {}}}}
    source_hashes, saved_structures = {}, {}
    families = {family: {} for family in audit.scorer.FAMILIES}
    fd = {"uncertainties": {"n_models_computed": 3}}
    families["distribution"] = {"evaluator_results": {"FrechetDistance": {
        "metric_results": {"FrechetDistance": fd}}}}
    for track in ("free", "fixed_symmetry"):
        directory = tmp_path / track
        directory.mkdir()
        source = index["arms"]["new_model"]
        source["outputs"][track] = str(directory)
        quantities = ("energy", "energy_per_atom", "formation_energy_per_atom", "energy_above_hull")
        scores = [{"gene": gene, "by_scorer": {model: dict.fromkeys(quantities, 1.)
                                               for model in audit.scorer.SCORING_MLIPS}} for gene in (7, 8)]
        if track == "free":
            scores[0]["by_scorer"]["uma"] = dict.fromkeys(quantities)
        metadata = {"arm_complete": True, "completed_trials": 2698,
                    "scored_gene_ids": [7, 8], "selected_gene_ids": [7, 8], "selected_genes": 2,
                    "single_point_coverage": {model: 1 if track == "free" and model == "uma" else 2
                                               for model in audit.scorer.SCORING_MLIPS},
                    "single_point_coverage_by_quantity": {model: dict.fromkeys(
                        quantities, 1 if track == "free" and model == "uma" else 2)
                        for model in audit.scorer.SCORING_MLIPS},
                    "scoring_failure_gene_ids": {model: [7] if track == "free" and model == "uma" else []
                                                 for model in audit.scorer.SCORING_MLIPS},
                    "scoring_complete": track != "free", "metrics_complete": True,
                    "final_results": track != "free", "results": {"validity": {}, **families}}
        path = directory / "metrics.json"
        path.write_text(json.dumps(metadata))
        (directory / "single_point_energies.json").write_text(json.dumps(scores))
        source["metrics_sha256"][track] = hashlib.sha256(path.read_bytes()).hexdigest()
        source_hashes[f"new_model/{track}"] = source["metrics_sha256"][track]
        saved_structures[directory] = [Structure(Lattice.cubic(3), ["Cu"], [[0, 0, 0]],
                                                properties={f"graph_embedding_{model}": [1.]
                                                            for model in audit.scorer.SCORING_MLIPS}) for _ in scores]
    monkeypatch.setattr(audit.scorer, "_load_checkpoint", lambda path: saved_structures[path.parent])
    source_index = tmp_path / "scoring_source_index.json"
    source_index.write_text(json.dumps(index))
    common = {"all_metrics_complete": True, "common_gene_compositions_match": True,
              "identity": {"common_valid_gene_ids": [8], "source_metrics_sha256": source_hashes,
                           "scoring_source_index_sha256": hashlib.sha256(source_index.read_bytes()).hexdigest()},
              "results": {key: {"metrics_complete": True, "results": families} for key in source_hashes}}
    common_path = tmp_path / "common_cohort_metrics.json"
    common_path.write_text(json.dumps(common))
    for name, count in [("full_cohort_metrics", 2), ("common_cohort_metrics", 2),
                        ("paired_free_fixed_energy_statistics", 9), ("common_gene_energy_statistics", 18),
                        ("formation_reference_cancellation_residuals", 3)]:
        (tmp_path / (name + ".csv")).write_text("column\n" + "row\n" * count)
    result = audit.validate(source_index, tmp_path)
    assert result["passed"]
    assert result["scoring_failures"] == {"new_model/free/uma": [7]}
    assert result["common_valid_fully_scored_genes"] == 1
    assert not result["source_tracks_checked"][0]["final_results"]
    common["identity"]["common_valid_gene_ids"] = [7, 8]
    common_path.write_text(json.dumps(common))
    with pytest.raises(AssertionError):
        audit.validate(source_index, tmp_path)
    common["identity"]["common_valid_gene_ids"] = [8]
    common_path.write_text(json.dumps(common))
    saved_structures[tmp_path / "fixed_symmetry"][1] = Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]],
                                                               properties={f"graph_embedding_{model}": [1.]
                                                                           for model in audit.scorer.SCORING_MLIPS})
    with pytest.raises(AssertionError):
        audit.validate(source_index, tmp_path)
