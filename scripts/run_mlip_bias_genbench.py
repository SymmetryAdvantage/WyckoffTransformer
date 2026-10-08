#!/usr/bin/env python3
"""Score a completed MLIP relaxation arm with LeMat-GenBench, without re-relaxing.

Run in an environment with LeMat-GenBench and its ORB/MACE/UMA dependencies.
The LeMat checkout must be visible at --lemat-root. Results and per-structure
single-point energies are saved locally and uploaded to W&B as an artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import numbers
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pymatgen.core import Structure

from wyckoff_transformer.paths import wandb_dir

FAMILIES = ("distribution", "diversity", "novelty", "uniqueness", "hhi", "sun", "stability")
SCORING_MLIPS = ("orb", "mace", "uma")


def _import_runner(lemat_root: Path):
    runner_path = lemat_root / "scripts" / "run_benchmarks.py"
    if not runner_path.is_file():
        raise FileNotFoundError(runner_path)
    sys.path[:0] = [str(lemat_root / "src"), str(lemat_root / "scripts")]
    spec = importlib.util.spec_from_file_location("lemat_bias_benchmarks", runner_path)
    if spec is None or spec.loader is None:
        raise ImportError(runner_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _drop_relaxation_rmsd(value):
    if isinstance(value, dict):
        return {key: _drop_relaxation_rmsd(item) for key, item in value.items()
                if "relaxation_rmse" not in key.lower() and "relaxation_rmsd" not in key.lower()}
    if isinstance(value, list):
        return [_drop_relaxation_rmsd(item) for item in value]
    return value


def _single_point_record(gene: int, structure: Structure) -> dict:
    props = structure.properties
    record = {"gene": gene, "formula": structure.composition.reduced_formula,
              "by_scorer": {}}
    for mlip in SCORING_MLIPS:
        scores = {}
        for source, target in (("energy", "energy_ev"),
                               ("formation_energy", "formation_energy_ev_per_atom"),
                               ("e_above_hull", "energy_above_hull_ev_per_atom")):
            value = props.get(f"{source}_{mlip}")
            scores[target] = float(value) if isinstance(value, numbers.Real) else None
        record["by_scorer"][mlip] = scores
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--arm-artifact", help="W&B relaxation artifact path, e.g. entity/project/name:latest")
    parser.add_argument("--lemat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track", choices=("free", "fixed_symmetry"), default="free")
    parser.add_argument("--expected-trials", type=int, default=2698)
    parser.add_argument("--allow-partial", action="store_true", help="Diagnostic subset only")
    parser.add_argument("--no-wandb", action="store_true", help="Diagnostic local run only")
    args = parser.parse_args()

    if args.arm_artifact and not (args.arm / "trials.csv").is_file():
        import wandb

        wandb.Api().artifact(args.arm_artifact, type="mlip_bias_relax").download(
            root=str(args.arm)
        )

    arm_manifest = json.loads((args.arm / "study_manifest.json").read_text())
    settings_path = args.arm / "relaxation_settings.json"
    relaxation_settings = json.loads(settings_path.read_text()) if settings_path.is_file() else None
    selected = json.loads((args.arm / f"selected_{args.track}.json").read_text())
    with (args.arm / "trials.csv").open(newline="") as stream:
        completed_trials = sum(1 for _ in csv.DictReader(stream))
    if not args.allow_partial and completed_trials != args.expected_trials:
        parser.error(f"Arm has {completed_trials}/{args.expected_trials} trials; use --allow-partial only for a diagnostic")
    if not selected:
        parser.error("No selected CIFs")
    genes = sorted(map(int, selected))
    structures = [Structure.from_file(args.arm / selected[str(gene)]["cif"]) for gene in genes]

    bench = _import_runner(args.lemat_root)
    lemat_commit = subprocess.check_output(
        ["git", "-C", str(args.lemat_root), "rev-parse", "HEAD"], text=True
    ).strip()
    # Upstream's run_remaining_preprocessors derives relax_structures from the
    # presence of the stability family and ignores its YAML setting. Override
    # that constructor explicitly while preserving energy and hull scoring.
    base = bench.MultiMLIPStabilityPreprocessor

    class SinglePointPreprocessor(base):
        def __init__(self, *positional, **keyword):
            keyword.update(relax_structures=False, calculate_formation_energy=True,
                           calculate_energy_above_hull=True, n_jobs=1)
            super().__init__(*positional, **keyword)

    bench.MultiMLIPStabilityPreprocessor = SinglePointPreprocessor
    # Upstream's helper saves an optional embedding visualization beside its
    # checkout; the benchmark uses the in-memory embeddings for its metrics.
    bench.save_embeddings_from_structures = lambda *positional, **keyword: None
    config = bench.load_benchmark_config("comprehensive_multi_mlip_hull")
    config["cache_dir"] = str(args.lemat_root / "data")
    for key in ("js_distributions_file", "mmd_values_file"):
        config[key] = str(args.lemat_root / config[key])
    config["preprocessor_config"]["relax_structures"] = False

    validity, valid_structures, filtering = bench.run_validity_preprocessing_and_filtering(
        structures, config)
    requirements = bench.create_preprocessor_config(list(FAMILIES), "structure-matcher")
    requirements["validity"] = False
    scored, _ = bench.run_remaining_preprocessors(
        valid_structures, requirements, config, "mlip_bias", False, False)
    metrics = bench.run_remaining_benchmarks(scored, list(FAMILIES), config)
    if len(scored) != len(valid_structures):
        raise RuntimeError("LeMat-GenBench changed the structure count during single-point preprocessing")

    valid_genes = []
    for structure in valid_structures:
        source = structure.properties.get("original_source", "")
        if not source.startswith("structure_"):
            raise RuntimeError(f"Missing LeMat-GenBench source index: {source!r}")
        valid_genes.append(genes[int(source.removeprefix("structure_"))])
    if len(valid_genes) != len(scored):
        raise RuntimeError("Cannot map scored structures to their source genes")
    energies = [_single_point_record(gene, structure)
                for gene, structure in zip(valid_genes, scored)]
    coverage = {mlip: sum(row["by_scorer"][mlip]["energy_ev"] is not None and
                          row["by_scorer"][mlip]["energy_above_hull_ev_per_atom"] is not None
                          for row in energies)
                for mlip in SCORING_MLIPS}
    scoring_complete = all(count == len(scored) for count in coverage.values())
    arm_complete = completed_trials == args.expected_trials
    metrics_complete = all(not (isinstance(result, dict) and "error" in result)
                           for result in metrics.values())
    final_results = arm_complete and scoring_complete and metrics_complete and bool(scored)

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "arm_manifest": arm_manifest,
        "relaxation_settings": relaxation_settings,
        "track": args.track,
        "completed_trials": completed_trials,
        "selected_genes": len(genes),
        "validity_filtering": filtering,
        "single_point_coverage": coverage,
        "arm_complete": arm_complete,
        "scoring_complete": scoring_complete,
        "metrics_complete": metrics_complete,
        "final_results": final_results,
        "scoring_mlips": list(SCORING_MLIPS),
        "scoring_config": "comprehensive_multi_mlip_hull; relaxation disabled",
        "lemat_genbench_commit": lemat_commit,
        "results": _drop_relaxation_rmsd({"validity": validity, **metrics}),
    }
    results_path = args.output / "metrics.json"
    results_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    energies_path = args.output / "single_point_energies.json"
    energies_path.write_text(json.dumps(energies, indent=2, default=str) + "\n")

    if not args.no_wandb:
        import wandb

        identity = json.dumps({"arm": arm_manifest, "track": args.track}, sort_keys=True)
        digest = hashlib.sha256(identity.encode()).hexdigest()
        run = wandb.init(entity="symmetry-advantage", project="WyckoffTransformer",
                         id=digest[:8], resume="allow", dir=str(wandb_dir()),
                         job_type="mlip_bias_genbench",
                         config={"track": args.track, "arm": arm_manifest,
                                 "relaxation_settings": relaxation_settings,
                                 "relax_structures": False,
                                 "lemat_genbench_commit": lemat_commit})
        artifact = wandb.Artifact(f"mlip-bias-metrics-{digest[:16]}", type="mlip_bias_metrics",
                                  metadata={"track": args.track, "completed_trials": completed_trials,
                                            "selected_genes": len(genes), "coverage": coverage})
        artifact.add_file(str(results_path), name="metrics.json")
        artifact.add_file(str(energies_path), name="single_point_energies.json")
        run.log_artifact(artifact).wait()
        run.summary.update({"selected_genes": len(genes), "valid_genes": len(scored),
                            "scoring_complete": scoring_complete,
                            "arm_complete": arm_complete,
                            "metrics_complete": metrics_complete,
                            "final_results": final_results,
                            **{f"coverage_{key}": value for key, value in coverage.items()}})
        run.finish()

    print(json.dumps({"metrics": str(results_path), "energies": str(energies_path),
                      "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
