"""Validate a snapshot-indexed report without assuming a fixed failure or mask."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

spec = importlib.util.spec_from_file_location("bias_scorer", "scripts/run_mlip_bias_genbench.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def validate(source_index, output):
    index = json.loads(source_index.read_text())
    rows, eligible, source_hashes, failures = [], [], {}, {}
    compositions = []
    for arm, source in index["arms"].items():
        for track in ("free", "fixed_symmetry"):
            directory = Path(source["outputs"][track])
            path = directory / "metrics.json"
            data = json.loads(path.read_text())
            scores = json.loads((directory / "single_point_energies.json").read_text())
            structures = scorer._load_checkpoint(directory / "single_point_checkpoint.json.gz")
            assert data["arm_complete"] and data["completed_trials"] == 2698
            assert set(data["results"]) == {"validity", *scorer.FAMILIES}
            assert len(scores) == len(data["scored_gene_ids"]) == len(set(data["scored_gene_ids"]))
            assert len(structures) == len(scores)
            assert [row["gene"] for row in scores] == data["scored_gene_ids"]
            assert len(data["selected_gene_ids"]) == data["selected_genes"]
            assert set(data["scored_gene_ids"]) <= set(data["selected_gene_ids"])
            for model in scorer.SCORING_MLIPS:
                bad = [row["gene"] for row in scores if any(value is None for value in row["by_scorer"][model].values())]
                assert bad == data["scoring_failure_gene_ids"][model]
                assert len(scores) - len(bad) == data["single_point_coverage"][model]
                for quantity, count in data["single_point_coverage_by_quantity"][model].items():
                    assert count == sum(row["by_scorer"][model][quantity] is not None for row in scores)
                if bad:
                    failures[f"{arm}/{track}/{model}"] = bad
            complete_genes = set()
            for row, structure in zip(scores, structures):
                finite = all(value is not None and math.isfinite(value)
                             for values in row["by_scorer"].values() for value in values.values())
                embeddings = all(structure.properties.get(f"graph_embedding_{model}") is not None
                                 for model in scorer.SCORING_MLIPS)
                if finite and embeddings:
                    complete_genes.add(row["gene"])
            eligible.append(complete_genes)
            compositions.append({row['gene']: structure.composition.fractional_composition
                                 for row, structure in zip(scores, structures)})
            key = f"{arm}/{track}"
            source_hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
            assert source_hashes[key] == source["metrics_sha256"][track]
            assert data["final_results"] == (data["scoring_complete"] and data["metrics_complete"] and bool(scores))
            assert data["metrics_complete"] == scorer._metrics_complete(data["results"])
            if not data["scoring_complete"]:
                assert not data["final_results"]
            rows.append({"arm": arm, "track": track, "selected_genes": data["selected_genes"],
                         "valid_genes": len(scores), "coverage": data["single_point_coverage"],
                         "final_results": data["final_results"]})
    common = json.loads((output / "common_cohort_metrics.json").read_text())
    assert common["all_metrics_complete"] and common["common_gene_compositions_match"]
    assert set(common["identity"]["common_valid_gene_ids"]) == set.intersection(*eligible)
    for gene in common["identity"]["common_valid_gene_ids"]:
        assert all(mapping[gene] == compositions[0][gene] for mapping in compositions)
    assert common["identity"]["source_metrics_sha256"] == source_hashes
    assert common["identity"]["scoring_source_index_sha256"] == hashlib.sha256(source_index.read_bytes()).hexdigest()
    n_arms = len(index["arms"])
    assert len(common["results"]) == n_arms * 2
    assert set(common["results"]) == set(source_hashes)
    for result in common["results"].values():
        assert set(result["results"]) == set(scorer.FAMILIES)
        assert result["metrics_complete"] and scorer._metrics_complete(result["results"])
        fd = result["results"]["distribution"]["evaluator_results"]["FrechetDistance"]["metric_results"]["FrechetDistance"]
        assert fd["uncertainties"]["n_models_computed"] == 3
    for name, expected in [("full_cohort_metrics", n_arms * 2), ("common_cohort_metrics", n_arms * 2),
                           ("paired_free_fixed_energy_statistics", n_arms * 9),
                           ("common_gene_energy_statistics", n_arms * 18),
                           ("formation_reference_cancellation_residuals", n_arms * 3)]:
        with (output / (name + ".csv")).open(newline="") as stream:
            assert len(list(csv.DictReader(stream))) == expected
    return {"date_utc": datetime.now(timezone.utc).isoformat(), "source_tracks_checked": rows,
            "source_metrics_sha256": source_hashes, "scoring_failures": failures,
            "common_valid_fully_scored_genes": len(common["identity"]["common_valid_gene_ids"]),
            "all_matched_benchmarks_complete": True, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.source_index.parent / "report"
    record = validate(args.source_index, output)
    (output / "genbench_results_validation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
