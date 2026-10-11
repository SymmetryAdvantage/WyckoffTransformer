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
import math
import multiprocessing
import numbers
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from itertools import combinations_with_replacement
from pathlib import Path

from monty.json import MontyDecoder
from monty.serialization import dumpfn, loadfn
from pymatgen.core import Composition, Structure

from wyckoff_transformer.energy_fields import (
    EnergyField,
    canonical_id,
    check_compatible,
    mlip_hull_energy_field,
)
from wyckoff_transformer.paths import wandb_dir

FAMILIES = ("distribution", "diversity", "novelty", "uniqueness", "hhi", "sun", "stability")
SCORING_MLIPS = ("orb", "mace", "uma")
SCORING_HULLS = {"orb": "orb_conserv_inf", "mace": "mace_mp", "uma": "uma"}
NOVELTY_REFERENCE_REVISION = "0dc17eea904b860ad7288141e9870f67f8e6bb2c"


def _validate_score_reuse(target, source):
    """Require identical scoring physics; match each geometry separately by CIF hash.

    Relaxation provenance identifies the output snapshot, not the single-point
    scorer. It may change across a failure recovery or another relaxation arm.
    These fields still participate in the target's checkpoint/run identity.
    """
    provenance = {"arm", "track", "selected_cif_hashes", "snapshot_namespace",
                  "arm_artifact", "trials_sha256"}
    for key in (target.keys() | source.keys()) - provenance:
        if key not in target or key not in source or target[key] != source[key]:
            raise ValueError(f"Cannot reuse scores with different {key}")


def _scoring_digest(arm_manifest, track, checkpoint_identity, snapshot_namespace=None):
    """Keep historical IDs; isolate explicitly versioned inputs by their contents."""
    identity = {"arm": arm_manifest, "track": track}
    if snapshot_namespace is not None:
        identity.update(snapshot_namespace=snapshot_namespace,
                        checkpoint_identity=checkpoint_identity)
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
_WORKER_PROCESS = None


def _initialize_scoring_worker(lemat_root):
    global _WORKER_PROCESS
    runner = _import_runner(Path(lemat_root))
    _install_matched_formation_references()
    _WORKER_PROCESS = runner.MultiMLIPStabilityPreprocessor.process_structure


def _score_worker(job):
    structure, settings = job
    try:
        result = _WORKER_PROCESS(structure, **settings)
    except Exception as exc:
        result = structure.copy()
        result.properties["scoring_exception"] = f"{type(exc).__name__}: {exc}"
    return _compact_score(result)


def _pairwise_matching(fingerprinter, structures, threshold=None):
    """Skip only composition pairs rejected by the same StructureMatcher."""
    import numpy as np

    matrix = np.zeros((len(structures), len(structures)), dtype=int)
    matcher = fingerprinter.matcher
    groups = {None: list(range(len(structures)))}
    hashes = None
    if not matcher._subset and not matcher._ignored_species:
        hashes = [matcher._comparator.get_hash(item.composition) for item in structures]
        groups = {}
        try:
            for index, value in enumerate(hashes):
                groups.setdefault(hash(value), []).append(index)
        except TypeError:
            groups, hashes = {None: list(range(len(structures)))}, None
    for indices in groups.values():
        for left, right in combinations_with_replacement(indices, 2):
            if hashes is not None and hashes[left] != hashes[right]:
                continue
            if threshold is None:
                matched = fingerprinter.is_equivalent(structures[left], structures[right])
            else:
                matched = fingerprinter.is_equivalent(structures[left], structures[right], threshold)
            matrix[left, right] = matrix[right, left] = int(matched)
    return matrix


def _composition_can_match(fingerprinter, left, right):
    """Cache composition rejection for immutable benchmark structures."""
    matcher = fingerprinter.matcher
    if matcher._subset or matcher._ignored_species:
        return True
    cache = getattr(fingerprinter, "_mlip_bias_composition_cache", None)
    if cache is None:
        cache = fingerprinter._mlip_bias_composition_cache = {}
    if len(cache) > 4096:
        # Bound references retained while matching against the full dataset.
        cache.clear()
    hashes = []
    for structure in (left, right):
        key = id(structure)
        if key not in cache:
            cache[key] = (structure, matcher._comparator.get_hash(structure.composition))
        hashes.append(cache[key][1])
    return hashes[0] == hashes[1]


def _novelty_match(structure, dataset_information, fingerprinter):
    """Apply the matcher's own composition rejection before building references."""
    import numpy as np
    from lemat_genbench.fingerprinting.encode_compositions import (
        filter_df,
        lematbulk_item_to_structure,
    )

    matcher = fingerprinter.matcher
    query_hash = matcher._comparator.get_hash(structure.composition)
    filtered = filter_df(dataset_information["dataset_dataframe"],
                         dataset_information["all_compositions"], structure)
    matrix = dataset_information["all_compositions"]
    if matrix is not None:
        if "element_counts" not in dataset_information:
            dataset_information["element_counts"] = np.asarray(matrix.sum(axis=1)).ravel()
        # The upstream subset-of-elements filter includes all unary/binary
        # subcompositions. The default comparator cannot match those when the
        # query contains a different number of elements.
        counts = dataset_information["element_counts"]
        indices = filtered["index_number"].to_numpy()
        filtered = filtered.loc[counts[indices] == len({element.Z for element in structure.composition.elements})]
    indices = filtered["index_number"].tolist()
    candidates = dataset_information["dataset"].select(indices)
    for item in candidates:
        reference_composition = Composition(Counter(item["species_at_sites"]))
        if query_hash != matcher._comparator.get_hash(reference_composition):
            continue
        if fingerprinter.is_equivalent(structure, lematbulk_item_to_structure(item)):
            return 0.0
    return 1.0


def _install_novelty_acceleration(lemat_root, cache, save_cache):
    import numpy as np
    from lemat_genbench.metrics import novelty_metric
    from material_hasher.similarity.structure_matchers import (
        PymatgenStructureSimilarity,
    )
    from scipy.sparse import load_npz

    original_load_dataset = novelty_metric.load_dataset

    def pinned_dataset(*args, **kwargs):
        if args and args[0] == "LeMaterial/LeMat-Bulk":
            kwargs["revision"] = NOVELTY_REFERENCE_REVISION
        return original_load_dataset(*args, **kwargs)

    novelty_metric.load_dataset = pinned_dataset
    matrix = None

    def compositions():
        nonlocal matrix
        if matrix is None:
            matrix = load_npz(lemat_root / "data/all_compositions.npz")
        return matrix

    novelty_metric.get_all_compositions = compositions
    reference_cache = {}
    original_load = novelty_metric.NoveltyMetric._load_reference_dataset

    def shared_reference(metric):
        key = tuple(getattr(metric.config, name) for name in
                    ("reference_dataset", "reference_config", "fingerprint_method", "max_reference_size"))
        if key not in reference_cache:
            reference_cache[key] = original_load(metric)
        metric._dataset_information = reference_cache[key]
        metric._reference_loaded = True
        return reference_cache[key]

    novelty_metric.NoveltyMetric._load_reference_dataset = shared_reference
    original_compute = novelty_metric.NoveltyMetric.compute_structure
    checked = 0

    def accelerated(structure, dataset_information, fingerprinter, verbose=False):
        nonlocal checked
        matcher = getattr(fingerprinter, "matcher", None)
        if matcher is None or matcher._subset or matcher._ignored_species:
            return original_compute(structure, dataset_information, fingerprinter, verbose)
        key = hashlib.sha256(structure.lattice.matrix.tobytes() + structure.frac_coords.tobytes()
                             + json.dumps([str(site.species) for site in structure]).encode()
                             + json.dumps(matcher.as_dict(), sort_keys=True).encode()
                             + NOVELTY_REFERENCE_REVISION.encode()).hexdigest()
        if key in cache:
            return cache[key]
        try:
            value = _novelty_match(structure, dataset_information, fingerprinter)
        except Exception as exc:
            print(f"Novelty failed for {structure.composition}: {exc}", file=sys.stderr, flush=True)
            return float("nan")
        if np.isfinite(value):
            cache[key] = value
            checked += 1
            if checked % 100 == 0:
                save_cache(cache)
            if checked % 25 == 0:
                print(f"Novelty checked {checked} new structures; cached {len(cache)}", flush=True)
        return value

    novelty_metric.NoveltyMetric.compute_structure = staticmethod(accelerated)
    original_equivalence = PymatgenStructureSimilarity.is_equivalent

    def composition_filtered_equivalence(fingerprinter, left, right, threshold=None):
        if not _composition_can_match(fingerprinter, left, right):
            return False
        return original_equivalence(fingerprinter, left, right, threshold)

    PymatgenStructureSimilarity.is_equivalent = composition_filtered_equivalence
    PymatgenStructureSimilarity.get_pairwise_equivalence = _pairwise_matching


def _compact_score(structure):
    # Requested metrics use total energies and graph embeddings, not node
    # embeddings or forces. Keep checkpoints proportional to structure count.
    structure.properties = {key: value for key, value in structure.properties.items()
                            if not key.startswith(("node_embeddings_", "forces_"))}
    return structure


def _load_checkpoint(path):
    data = loadfn(path)

    def restore(value):
        if isinstance(value, Structure):
            value.properties = MontyDecoder().process_decoded(value.properties)
        elif isinstance(value, dict):
            for item in value.values():
                restore(item)
        elif isinstance(value, list):
            for item in value:
                restore(item)

    restore(data)
    return data


def _energy_definitions():
    definitions = {}
    for scorer, hull in SCORING_HULLS.items():
        above_hull = mlip_hull_energy_field(hull)
        fields = [EnergyField("energy", "total", above_hull.source),
                  EnergyField("energy", "per_atom", above_hull.source),
                  EnergyField("formation_energy", "per_atom", above_hull.source, above_hull.reference),
                  above_hull]
        definitions[scorer] = {canonical_id(field): field.to_dict() for field in fields}
    return definitions


def _matched_formation_energy(total_energy, composition, references, predicted_field, reference_field):
    check_compatible(predicted_field, reference_field, "MLIP elemental formation-energy references")
    return (total_energy - sum(count * references[element.symbol]
                              for element, count in composition.items())) / composition.num_atoms


def _install_matched_formation_references():
    """Use each MLIP's elemental energies instead of upstream's PBE energies."""
    import numpy as np
    from lemat_genbench.models.mace.calculator import MACECalculator
    from lemat_genbench.models.orb.calculator import ORBCalculator
    from lemat_genbench.models.uma.calculator import UMACalculator
    from lemat_genbench.preprocess import multi_mlip_preprocess
    from lemat_genbench.preprocess.reference_energies import (
        _retrieve_df,
        _retrieve_matrix,
    )

    elemental = {}

    def calculate_formation_energy(calculator, structure):
        hull = calculator.hull_type
        if hull not in elemental:
            frame, matrix = _retrieve_df(hull), _retrieve_matrix(hull)
            references = {}
            for index in np.flatnonzero(matrix.sum(axis=1) == 1):
                row = frame.iloc[index]
                species = row["species_at_sites"]
                energy = float(row["energy"]) / len(species)
                if math.isfinite(energy):
                    symbol = str(species[0])
                    references[symbol] = min(references.get(symbol, math.inf), energy)
            elemental[hull] = references
        source = mlip_hull_energy_field(hull).source
        field = EnergyField("energy", "per_atom", source)
        total_energy = calculator.calculate_energy_forces(structure).energy
        return _matched_formation_energy(total_energy, structure.composition,
                                         elemental[hull], field, field)

    for calculator in (ORBCalculator, MACECalculator, UMACalculator):
        calculator.calculate_formation_energy = calculate_formation_energy
    # ASE cache comparisons cannot compare arrays from earlier scorers inside
    # atoms.info. Retain annotations in the outer pipeline, and give each model
    # the same physical structure with only its general metadata cleared.
    process_mlip = multi_mlip_preprocess._process_single_mlip

    def process_clean_structure(structure, *positional, **keyword):
        physical = structure.copy()
        physical.properties = {}
        return process_mlip(physical, *positional, **keyword)

    multi_mlip_preprocess._process_single_mlip = process_clean_structure


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
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(key): _drop_relaxation_rmsd(item) for key, item in value.items()
                if "relaxation_rmse" not in str(key).lower()
                and "relaxation_rmsd" not in str(key).lower()
                and "relaxation_stability" not in str(key).lower()}
    if isinstance(value, (list, tuple)):
        return [_drop_relaxation_rmsd(item) for item in value]
    if hasattr(value, "tolist"):
        return _drop_relaxation_rmsd(value.tolist())
    return value


def _adjacent_oxidation_combinations(values, count):
    """Generate precisely the multisets retained by LeMat's max-min <= 1 rule."""
    values = tuple(values)
    if count < 1 or any(not isinstance(value, numbers.Integral) for value in values) \
            or list(values) != sorted(set(values)):
        for combo in combinations_with_replacement(values, count):
            if not combo or max(combo) - min(combo) <= 1:
                yield combo
        return
    for index, value in enumerate(values):
        yield (value,) * count
        if index + 1 < len(values) and values[index + 1] - value == 1:
            for copies in range(count - 1, 0, -1):
                yield (value,) * copies + (values[index + 1],) * (count - copies)


def _metrics_complete(results):
    """Require each requested evaluator's primary metric to be finite."""
    if isinstance(results, dict):
        if "error" in results:
            return False
        if "n_models_computed" in results and results["n_models_computed"] != len(SCORING_MLIPS):
            return False
        if "primary_metric" in results and "metrics" in results:
            primary = results["metrics"].get(results["primary_metric"])
            if primary is None or (isinstance(primary, numbers.Real) and not math.isfinite(primary)):
                return False
        return all(_metrics_complete(value) for value in results.values())
    if isinstance(results, list):
        return all(_metrics_complete(value) for value in results)
    return True


def _single_point_record(gene: int, structure: Structure) -> dict:
    props = structure.properties
    record = {"gene": gene, "formula": structure.composition.reduced_formula,
              "by_scorer": {}}
    for mlip in SCORING_MLIPS:
        scores = {}
        for source, target in (("energy", "energy"),
                               ("formation_energy", "formation_energy_per_atom"),
                               ("e_above_hull", "energy_above_hull")):
            value = props.get(f"{source}_{mlip}")
            scores[target] = (float(value) if isinstance(value, numbers.Real)
                              and math.isfinite(value) else None)
        record["by_scorer"][mlip] = scores
        scores["energy_per_atom"] = (scores["energy"] / len(structure)
                                      if scores["energy"] is not None else None)
    record["scoring_errors"] = {
        str(key): _drop_relaxation_rmsd(value) for key, value in props.items()
        if "error" in str(key).lower() or "exception" in str(key).lower()
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--arm-artifact", help="W&B relaxation artifact path, e.g. entity/project/name:latest")
    parser.add_argument("--snapshot-namespace",
                        help="New snapshot scoring identity; includes pinned artifact, ledger and CIF hashes")
    parser.add_argument("--lemat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-scores-from", type=Path,
                        help="Completed frozen scores; reuse only byte-identical CIFs")
    parser.add_argument("--workers", type=int, default=1,
                        help="Spawned single-point scoring workers; parent owns checkpoints")
    parser.add_argument("--track", choices=("free", "fixed_symmetry"), default="free")
    parser.add_argument("--expected-trials", type=int, default=2698)
    parser.add_argument("--allow-partial", action="store_true", help="Diagnostic subset only")
    parser.add_argument("--no-wandb", action="store_true", help="Diagnostic local run only")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.snapshot_namespace and (not args.arm_artifact or ":v" not in args.arm_artifact or
                                   not args.arm_artifact.rsplit(":v", 1)[-1].isdigit()):
        parser.error("--snapshot-namespace requires a pinned --arm-artifact ending in :v<number>")

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
    # Enumerate only combinations surviving the upstream rule, in its original
    # order. Large cells otherwise spend hours constructing rejected multisets.
    from lemat_genbench.utils import oxidation_state

    oxidation_state.combinations_with_replacement = _adjacent_oxidation_combinations
    _install_matched_formation_references()
    lemat_commit = subprocess.check_output(
        ["git", "-C", str(args.lemat_root), "rev-parse", "HEAD"], text=True
    ).strip()
    # Upstream's run_remaining_preprocessors derives relax_structures from the
    # presence of the stability family and ignores its YAML setting. Override
    # that constructor explicitly while preserving energy and hull scoring.
    base = bench.MultiMLIPStabilityPreprocessor
    single_point_progress = []
    reusable_scores = {}

    def cached_score(structure, index):
        if index < len(single_point_progress):
            return single_point_progress[index]
        source_index = int(structure.properties["original_source"].removeprefix("structure_"))
        cif_hash = checkpoint_identity["selected_cif_hashes"][str(genes[source_index])]
        cached = reusable_scores.get(cif_hash)
        if cached is None:
            return None
        result = structure.copy()
        for key, value in cached.properties.items():
            if key.endswith(tuple(f"_{mlip}" for mlip in SCORING_MLIPS)) or key.endswith(
                    ("_mean", "_std", "_n_mlips")):
                result.properties[key] = deepcopy(value)
        result.properties["scores_reused_from_cif_sha256"] = cif_hash
        return result

    def remember_score(result):
        single_point_progress.append(_compact_score(result))
        # Reused scores already have a durable source checkpoint. Avoid
        # repeatedly uploading their growing prefix; reconstruct it on resume.
        # During reuse, persist every new model result immediately instead.
        new_during_reuse = (args.reuse_scores_from
                            and "scores_reused_from_cif_sha256" not in result.properties)
        regular_batch = not args.reuse_scores_from and len(single_point_progress) % 25 == 0
        if new_during_reuse or regular_batch:
            save_checkpoint("single_point_partial_checkpoint", single_point_progress)
        return result

    class SinglePointPreprocessor(base):
        def __init__(self, *positional, **keyword):
            self._seen = 0
            keyword.update(relax_structures=False, calculate_formation_energy=True,
                           calculate_energy_above_hull=True, extract_embeddings=True, n_jobs=1)
            super().__init__(*positional, **keyword)

        def __call__(self, structures):
            if args.workers == 1:
                return super().__call__(structures)
            from lemat_genbench.preprocess.base import PreprocessorResult

            started = time.monotonic()
            initial_count = len(single_point_progress)
            # Resolve cache hits before appending new scores, so indices always
            # refer to the original ordered validity cohort.
            ready = {index: cached_score(structure, index)
                     for index, structure in enumerate(structures)}
            settings = self._get_process_attributes()
            jobs = [(structure, settings) for index, structure in enumerate(structures)
                    if ready[index] is None]
            results = []
            with ProcessPoolExecutor(max_workers=args.workers,
                                     mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_initialize_scoring_worker,
                                     initargs=(str(args.lemat_root.resolve()),)) as pool:
                pending = iter(pool.map(_score_worker, jobs, chunksize=1))
                for index in range(len(structures)):
                    result = ready[index]
                    if result is None:
                        result = next(pending)
                    if index >= initial_count:
                        remember_score(result)
                    results.append(result)
                    if (index + 1) % 25 == 0:
                        print(f"Parallel scoring: {index + 1}/{len(structures)}; workers={args.workers}", flush=True)
            self.config.n_jobs = args.workers
            return PreprocessorResult(processed_structures=results, config=self.config,
                                      computation_time=time.monotonic() - started,
                                      n_input_structures=len(structures), failed_indices=[], warnings=[])

        def process_structure(self, structure, **keyword):
            index = self._seen
            self._seen += 1
            if index < len(single_point_progress):
                return single_point_progress[index]
            try:
                result = cached_score(structure, index)
                if result is None:
                    result = base.process_structure(structure, **keyword)
            except Exception as exc:
                result = structure.copy()
                result.properties["scoring_exception"] = f"{type(exc).__name__}: {exc}"
                print(f"Single-point scoring failed: {exc}", file=sys.stderr, flush=True)
            return remember_score(result)

    bench.MultiMLIPStabilityPreprocessor = SinglePointPreprocessor
    # Upstream's helper saves an optional embedding visualization beside its
    # checkout; the benchmark uses the in-memory embeddings for its metrics.
    bench.save_embeddings_from_structures = lambda *positional, **keyword: None
    config = bench.load_benchmark_config("comprehensive_multi_mlip_hull")
    config["cache_dir"] = str(args.lemat_root / "data")
    for key in ("js_distributions_file", "mmd_values_file"):
        config[key] = str(args.lemat_root / config[key])
    config["preprocessor_config"]["relax_structures"] = False

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_identity = {
        "arm": arm_manifest, "track": args.track, "lemat_commit": lemat_commit,
        "scoring_config": config,
        "formation_energy_reference": "matched_mlip_hull_elemental_minima_v1",
        "scorer_input_metadata": "isolated_per_scorer_v1",
        "extract_embeddings": True,
        "selected_cif_hashes": {str(gene): hashlib.sha256(
            (args.arm / selected[str(gene)]["cif"]).read_bytes()).hexdigest() for gene in genes},
    }
    if args.snapshot_namespace:
        checkpoint_identity.update(
            snapshot_namespace=args.snapshot_namespace,
            arm_artifact=args.arm_artifact,
            trials_sha256=hashlib.sha256((args.arm / "trials.csv").read_bytes()).hexdigest())
    digest = _scoring_digest(arm_manifest, args.track, checkpoint_identity, args.snapshot_namespace)
    checkpoint_manifest = args.output / "checkpoint_manifest.json"
    # Reject changed inputs before opening a W&B run or recovering old checkpoints.
    if checkpoint_manifest.exists() and json.loads(checkpoint_manifest.read_text()) != checkpoint_identity:
        raise ValueError("Scoring checkpoint inputs changed; use a separate output directory")
    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(entity="symmetry-advantage", project="WyckoffTransformer",
                         id=digest[:8], resume="allow", dir=str(wandb_dir()),
                         job_type="mlip_bias_genbench",
                         config={"track": args.track, "arm": arm_manifest,
                                 "snapshot_namespace": args.snapshot_namespace,
                                 "arm_artifact": args.arm_artifact,
                                 "relaxation_settings": relaxation_settings,
                                 "relax_structures": False,
                                 "single_point_workers": args.workers,
                                 "lemat_genbench_commit": lemat_commit})
    if not checkpoint_manifest.exists() and run is not None:
        try:
            wandb.Api().artifact(
                f"symmetry-advantage/WyckoffTransformer/mlip-bias-scoring-checkpoint-{digest[:16]}:latest",
                type="mlip_bias_scoring_checkpoint",
            ).download(root=str(args.output))
        except wandb.errors.CommError as exc:
            if "not found" not in str(exc).lower() and "does not exist" not in str(exc).lower():
                raise
    if checkpoint_manifest.exists():
        if json.loads(checkpoint_manifest.read_text()) != checkpoint_identity:
            raise ValueError("Scoring checkpoint inputs changed; use a separate output directory")
    else:
        checkpoint_manifest.write_text(json.dumps(checkpoint_identity, indent=2, default=str) + "\n")

    if args.reuse_scores_from:
        source_identity = json.loads((args.reuse_scores_from / "checkpoint_manifest.json").read_text())
        _validate_score_reuse(checkpoint_identity, source_identity)
        # Validity is also invariant when every gene has the exact same CIF,
        # order and frozen preprocessing configuration. Never reuse it for a
        # changed geometry or overwrite the target's own resume checkpoint.
        source_validity = args.reuse_scores_from / "validity_checkpoint.json.gz"
        target_validity = args.output / "validity_checkpoint.json.gz"
        if (source_identity["selected_cif_hashes"] == checkpoint_identity["selected_cif_hashes"]
                and source_validity.is_file() and not target_validity.exists()):
            temporary = target_validity.with_suffix(".tmp")
            shutil.copyfile(source_validity, temporary)
            temporary.replace(target_validity)
            print("Reused validity checkpoint: all selected CIF hashes and scoring configuration match", flush=True)
        source_metrics = json.loads((args.reuse_scores_from / "metrics.json").read_text())
        source_scores = _load_checkpoint(args.reuse_scores_from / "single_point_checkpoint.json.gz")
        if len(source_scores) != len(source_metrics["scored_gene_ids"]):
            raise ValueError("Reuse checkpoint gene mapping does not match scores")
        for gene, score in zip(source_metrics["scored_gene_ids"], source_scores):
            record = _single_point_record(gene, score)
            if all(value is not None for values in record["by_scorer"].values()
                   for value in values.values()) and all(
                       score.properties.get(f"graph_embedding_{mlip}") is not None
                       for mlip in SCORING_MLIPS):
                reusable_scores[source_identity["selected_cif_hashes"][str(gene)]] = score

    def save_checkpoint(name, data):
        temporary = args.output / f"{name}.tmp.json.gz"
        # Pymatgen's Structure.as_dict can mutate nested properties while
        # encoding numpy arrays. Keep the live metric inputs untouched.
        dumpfn(deepcopy(data), temporary)
        temporary.replace(args.output / f"{name}.json.gz")
        if run is not None:
            artifact = wandb.Artifact(f"mlip-bias-scoring-checkpoint-{digest[:16]}",
                                      type="mlip_bias_scoring_checkpoint", metadata={"stage": name})
            artifact.add_file(str(checkpoint_manifest), name=checkpoint_manifest.name)
            for path in args.output.glob("*_checkpoint.json.gz"):
                artifact.add_file(str(path), name=path.name)
            run.log_artifact(artifact).wait()
            run.summary["scoring_stage"] = name

    novelty_path = args.output / "novelty_cache_checkpoint.json.gz"
    novelty_cache = _load_checkpoint(novelty_path) if novelty_path.exists() else {}
    if args.reuse_scores_from:
        source_novelty = args.reuse_scores_from / "novelty_cache_checkpoint.json.gz"
        if source_novelty.exists():
            novelty_cache.update(_load_checkpoint(source_novelty))
    _install_novelty_acceleration(args.lemat_root, novelty_cache,
                                 lambda cache: save_checkpoint("novelty_cache_checkpoint", cache))

    validity_path = args.output / "validity_checkpoint.json.gz"
    if validity_path.exists():
        checkpoint = _load_checkpoint(validity_path)
        validity, valid_structures, filtering = (checkpoint[key] for key in
                                                 ("validity", "structures", "filtering"))
    else:
        validity, valid_structures, filtering = bench.run_validity_preprocessing_and_filtering(
            structures, config)
        save_checkpoint("validity_checkpoint", {"validity": _drop_relaxation_rmsd(validity),
                                                "structures": valid_structures, "filtering": filtering})
    requirements = bench.create_preprocessor_config(list(FAMILIES), "structure-matcher")
    requirements["validity"] = False
    scored_path = args.output / "single_point_checkpoint.json.gz"
    if scored_path.exists():
        scored = [_compact_score(item) for item in _load_checkpoint(scored_path)]
    else:
        partial_path = args.output / "single_point_partial_checkpoint.json.gz"
        if partial_path.exists():
            single_point_progress.extend(_compact_score(item) for item in _load_checkpoint(partial_path))
            if len(single_point_progress) > len(valid_structures):
                raise ValueError("Partial single-point checkpoint exceeds validity cohort")
        scored, _ = bench.run_remaining_preprocessors(
            valid_structures, requirements, config, "mlip_bias", False, False)
        save_checkpoint("single_point_checkpoint", scored)
    benchmarks_path = args.output / "benchmark_checkpoint.json.gz"
    metric_identity = {"novelty_reference_revision": NOVELTY_REFERENCE_REVISION,
                       "novelty_implementation": "composition_rejection_and_first_match_v1"}
    metrics = {}
    if benchmarks_path.exists():
        saved = _load_checkpoint(benchmarks_path)
        if saved["identity"] == metric_identity:
            metrics = saved["results"]
    for family in FAMILIES:
        if family in metrics:
            continue
        metrics.update(_drop_relaxation_rmsd(bench.run_remaining_benchmarks(scored, [family], config)))
        save_checkpoint("benchmark_checkpoint", {"identity": metric_identity, "results": metrics})
    save_checkpoint("novelty_cache_checkpoint", novelty_cache)
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
    coverage_by_quantity = {
        mlip: {quantity: sum(row["by_scorer"][mlip][quantity] is not None
                             for row in energies)
               for quantity in ("energy", "energy_per_atom", "formation_energy_per_atom",
                                "energy_above_hull")}
        for mlip in SCORING_MLIPS
    }
    coverage = {mlip: sum(all(value is not None for value in row["by_scorer"][mlip].values())
                          for row in energies)
                for mlip in SCORING_MLIPS}
    scoring_complete = all(count == len(scored) for count in coverage.values())
    arm_complete = completed_trials == args.expected_trials
    serialized_results = _drop_relaxation_rmsd({"validity": validity, **metrics})
    metrics_complete = _metrics_complete(serialized_results)
    final_results = (arm_complete and scoring_complete and metrics_complete and bool(scored)
                     and not args.allow_partial)

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "wyformer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "arm_manifest": arm_manifest,
        "arm_artifact": args.arm_artifact,
        "snapshot_namespace": args.snapshot_namespace,
        "scoring_run_id": digest[:8],
        "checkpoint_identity_sha256": hashlib.sha256(
            json.dumps(checkpoint_identity, sort_keys=True).encode()).hexdigest(),
        "relaxation_settings": relaxation_settings,
        "track": args.track,
        "completed_trials": completed_trials,
        "selected_genes": len(genes),
        "selected_gene_ids": genes,
        "scored_gene_ids": valid_genes,
        "validity_filtering": filtering,
        "single_point_coverage": coverage,
        "reused_byte_identical_scores": sum("scores_reused_from_cif_sha256" in item.properties
                                            for item in scored),
        "reuse_scores_from": str(args.reuse_scores_from) if args.reuse_scores_from else None,
        "single_point_coverage_by_quantity": coverage_by_quantity,
        "scoring_failure_gene_ids": {
            mlip: [row["gene"] for row in energies
                   if any(value is None for value in row["by_scorer"][mlip].values())]
            for mlip in SCORING_MLIPS},
        "arm_complete": arm_complete,
        "scoring_complete": scoring_complete,
        "metrics_complete": metrics_complete,
        "final_results": final_results,
        "scoring_mlips": list(SCORING_MLIPS),
        "single_point_workers": args.workers,
        "energy_fields": _energy_definitions(),
        "scoring_config": "comprehensive_multi_mlip_hull; relaxation disabled",
        "reference_warnings": ["Frozen LeMat source records an unresolved UMA energy/hull reference discrepancy."],
        "lemat_genbench_commit": lemat_commit,
        "novelty_reference_revision": NOVELTY_REFERENCE_REVISION,
        "novelty_implementation": "matcher_composition_rejection_and_first_match_short_circuit_v1",
        "results": serialized_results,
    }
    results_path = args.output / "metrics.json"
    results_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    energies_path = args.output / "single_point_energies.json"
    energies_path.write_text(json.dumps(energies, indent=2, default=str) + "\n")

    if not args.no_wandb:
        import wandb

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
