"""Scoring preserves benchmark values and the frozen validity predicate."""

import hashlib
import json
import sys
import types
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np
import pytest
from monty.serialization import dumpfn
from pymatgen.core import Composition, Element, Lattice, Structure

from scripts import run_mlip_bias_genbench as scorer
from scripts.run_mlip_bias_genbench import (
    _adjacent_oxidation_combinations,
    _drop_relaxation_rmsd,
    _energy_definitions,
    _matched_formation_energy,
)
from wyckoff_transformer.energy_fields import EnergyField, IncompatibleEnergyFieldError


def test_snapshot_identity_preserves_legacy_and_isolates_changed_sources():
    manifest = {"mlip": "probe", "input_sha256": "same-initial-structures"}
    checkpoint = {"selected_cif_hashes": {"7": "first"}, "arm_artifact": "arm:v1",
                  "trials_sha256": "ledger1", "scoring_config": {"relax_structures": False}}
    legacy = hashlib.sha256(json.dumps({"arm": manifest, "track": "free"},
                                      sort_keys=True).encode()).hexdigest()
    assert scorer._scoring_digest(manifest, "free", checkpoint) == legacy
    digest = scorer._scoring_digest(manifest, "free", checkpoint, "recovery-20261006")
    assert digest != legacy
    assert digest == scorer._scoring_digest(manifest, "free", dict(reversed(list(checkpoint.items()))),
                                            "recovery-20261006")
    for key, value in [("selected_cif_hashes", {"7": "second"}), ("arm_artifact", "arm:v2"),
                       ("trials_sha256", "ledger2"), ("scoring_config", {"relax_structures": True})]:
        assert scorer._scoring_digest(manifest, "free", {**checkpoint, key: value},
                                     "recovery-20261006") != digest
    assert scorer._scoring_digest(manifest, "fixed_symmetry", checkpoint, "recovery-20261006") != digest
    assert scorer._scoring_digest(manifest, "free", checkpoint, "another-recovery") != digest


def test_score_reuse_separates_relaxation_provenance_from_scoring_contract():
    source = {"arm": {"calculator_factory": None}, "track": "free",
              "selected_cif_hashes": {"7": "old-cif"}, "lemat_commit": "frozen",
              "scoring_config": {"model": "pinned", "reference": "hull"},
              "formation_energy_reference": "matched", "scorer_input_metadata": "isolated",
              "extract_embeddings": True}
    target = {**source, "arm": {"calculator_factory": "optimized"},
              "track": "fixed_symmetry", "selected_cif_hashes": {"7": "new-cif"},
              "snapshot_namespace": "failure-recovery", "arm_artifact": "arm:v2",
              "trials_sha256": "new-ledger"}
    scorer._validate_score_reuse(target, source)
    for key in source.keys() - {"arm", "track", "selected_cif_hashes"}:
        with pytest.raises(ValueError, match=key):
            scorer._validate_score_reuse({**target, key: "changed"}, source)
        incomplete = dict(source)
        del incomplete[key]
        with pytest.raises(ValueError, match=key):
            scorer._validate_score_reuse(target, incomplete)
    with pytest.raises(ValueError, match="new_scoring_setting"):
        scorer._validate_score_reuse(target, {**source, "new_scoring_setting": True})


def test_optimized_oxidation_search_preserves_candidates_and_order():
    states = [element.oxidation_states for element in Element]
    states.extend([(2, -1, 0), (-1.0, -.5, 0), (), (1, 1, 2)])
    for values in states:
        for count in range(1, 5):
            expected = [combo for combo in combinations_with_replacement(values, count)
                        if max(combo) - min(combo) <= 1]
            assert list(_adjacent_oxidation_combinations(values, count)) == expected


def test_benchmark_dataclasses_are_numeric_json_without_relaxation_rmsd():
    @dataclass
    class Result:
        evaluator_results: list
        final_scores: dict
        metadata: dict

    result = Result([{"score": np.float64(.8), "relaxation_rmse": 3, 2: "count"}],
                    {"stability": .9}, {"embedding": np.array([1, 2])})
    saved = json.loads(json.dumps(_drop_relaxation_rmsd(result)))
    assert saved["evaluator_results"] == [{"score": .8, "2": "count"}]
    assert saved["final_scores"] == {"stability": .9}
    assert saved["metadata"]["embedding"] == [1, 2]


def test_scoring_resumes_partial_checkpoint_without_recomputing_or_relaxing(tmp_path, monkeypatch):
    arm, output = tmp_path / "arm", tmp_path / "output"
    arm.mkdir()
    selected = {}
    structure = Structure(Lattice.cubic(3.61), ["Cu"], [[0, 0, 0]])
    for gene in range(26):
        structure.to(filename=str(arm / f"{gene}.cif"))
        selected[str(gene)] = {"cif": f"{gene}.cif"}
    (arm / "study_manifest.json").write_text(json.dumps({"mlip": "probe"}))
    (arm / "selected_free.json").write_text(json.dumps(selected))
    (arm / "trials.csv").write_text("gene,trial\n" + "".join(f"{i},0\n" for i in range(26)))
    calls = {"validity": 0, "single_point": 0}

    class Base:
        def __init__(self, **settings):
            self.config = types.SimpleNamespace(n_jobs=1)
            assert settings["relax_structures"] is False
            assert settings["calculate_formation_energy"] is True
            assert settings["calculate_energy_above_hull"] is True
            assert settings["n_jobs"] == 1

        def _get_process_attributes(self):
            return {}

        @staticmethod
        def process_structure(structure, **keyword):
            calls["single_point"] += 1
            for mlip in scorer.SCORING_MLIPS:
                structure.properties.update({f"energy_{mlip}": -4.,
                                             f"e_above_hull_{mlip}": 0.,
                                             f"formation_energy_{mlip}": -.5,
                                             f"graph_embedding_{mlip}": np.array([1., 2.])})
            return structure

    def validity(structures, config):
        calls["validity"] += 1
        for i, item in enumerate(structures):
            item.properties["original_source"] = f"structure_{i}"
        return {"score": 1.}, structures, {"valid_structures": len(structures)}

    def runner(root):
        bench = types.SimpleNamespace(
            MultiMLIPStabilityPreprocessor=Base,
            load_benchmark_config=lambda name: {"preprocessor_config": {},
                                                "js_distributions_file": "data/js.json",
                                                "mmd_values_file": "data/mmd.pkl"},
            run_validity_preprocessing_and_filtering=validity,
            create_preprocessor_config=lambda *args: {"validity": True},
            run_remaining_benchmarks=lambda *args: {family: {"score": .5} for family in scorer.FAMILIES},
        )

        def preprocess(structures, *args):
            processor = bench.MultiMLIPStabilityPreprocessor()
            if '--workers' in sys.argv:
                return processor(structures).processed_structures, {}
            return [processor.process_structure(item) for item in structures], {}

        bench.run_remaining_preprocessors = preprocess
        return bench

    monkeypatch.setattr(scorer, "_import_runner", runner)
    monkeypatch.setattr(scorer, "_install_matched_formation_references", lambda: None)
    monkeypatch.setattr(scorer, "_install_novelty_acceleration", lambda *args: None)
    monkeypatch.setattr(scorer.subprocess, "check_output", lambda *args, **kwargs: "test-commit\n")
    monkeypatch.setitem(sys.modules, "lemat_genbench.utils", types.SimpleNamespace(
        oxidation_state=types.SimpleNamespace(combinations_with_replacement=None)))
    monkeypatch.setattr(sys, "argv", ["scorer", "--arm", str(arm), "--output", str(output),
                                     "--lemat-root", str(tmp_path), "--expected-trials", "26", "--no-wandb"])
    scorer.main()
    assert calls == {"validity": 1, "single_point": 26}
    assert json.loads((output / "metrics.json").read_text())["scoring_complete"]
    (output / "single_point_checkpoint.json.gz").unlink()
    scorer.main()
    assert calls == {"validity": 1, "single_point": 27}
    scorer.main()
    assert calls == {"validity": 1, "single_point": 27}
    results = json.loads((output / "metrics.json").read_text())
    assert results["scored_gene_ids"] == list(range(26))
    assert results["single_point_coverage"] == dict.fromkeys(scorer.SCORING_MLIPS, 26)
    (arm / "selected_fixed_symmetry.json").write_text(json.dumps(selected))
    fixed_output = tmp_path / "fixed_output"
    sys.argv.extend(["--track", "fixed_symmetry", "--reuse-scores-from", str(output)])
    sys.argv[sys.argv.index("--output") + 1] = str(fixed_output)
    scorer.main()
    assert calls == {"validity": 1, "single_point": 27}
    fixed = json.loads((fixed_output / "metrics.json").read_text())
    assert fixed["scored_gene_ids"] == list(range(26))
    assert fixed["reused_byte_identical_scores"] == 26
    assert not (fixed_output / 'single_point_partial_checkpoint.json.gz').exists()

    class ImmediatePool:
        def __init__(self, **settings):
            assert settings['max_workers'] == 4
            assert settings['mp_context'].get_start_method() == 'spawn'
            settings['initializer'](*settings['initargs'])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, jobs, chunksize):
            assert chunksize == 1
            return map(function, jobs)

    monkeypatch.setattr(scorer, 'ProcessPoolExecutor', ImmediatePool)
    monkeypatch.setitem(sys.modules, 'lemat_genbench.preprocess.base',
                        types.SimpleNamespace(PreprocessorResult=lambda **settings: types.SimpleNamespace(**settings)))
    sys.argv[sys.argv.index('--track') + 1] = 'free'
    reuse_index = sys.argv.index('--reuse-scores-from')
    del sys.argv[reuse_index:reuse_index + 2]
    parallel_output = tmp_path / 'parallel_output'
    sys.argv[sys.argv.index('--output') + 1] = str(parallel_output)
    sys.argv.extend(['--workers', '4'])
    scorer.main()
    assert calls == {'validity': 2, 'single_point': 53}
    (parallel_output / 'single_point_checkpoint.json.gz').unlink()
    scorer.main()
    assert calls == {'validity': 2, 'single_point': 54}
    parallel = json.loads((parallel_output / 'metrics.json').read_text())
    assert parallel['scored_gene_ids'] == list(range(26))
    assert parallel['single_point_workers'] == 4

    # A new snapshot cannot resume an older local checkpoint, even without W&B.
    sys.argv.extend(['--snapshot-namespace', 'recovery-20261006', '--arm-artifact', 'entity/project/probe:v2'])
    with pytest.raises(ValueError, match='Scoring checkpoint inputs changed'):
        scorer.main()
    snapshot_output = tmp_path / 'snapshot_output'
    sys.argv[sys.argv.index('--output') + 1] = str(snapshot_output)
    # Reuse across namespaces and a changed relaxation factory; only exact CIF
    # matches can reuse scores, so changing one CIF must trigger one new score.
    (arm / 'study_manifest.json').write_text(json.dumps({"mlip": "probe", "calculator_factory": "optimized"}))
    changed = Structure(Lattice.cubic(3.7), ["Cu"], [[0, 0, 0]])
    changed.to(filename=str(arm / '0.cif'))
    sys.argv.extend(['--reuse-scores-from', str(parallel_output)])
    before_reuse = calls['single_point']
    before_validity = calls['validity']
    scorer.main()
    snapshot = json.loads((snapshot_output / 'metrics.json').read_text())
    assert snapshot['reused_byte_identical_scores'] == 25
    assert calls['single_point'] == before_reuse + 1
    assert calls['validity'] == before_validity + 1
    # The newly evaluated CIF is durable even when fewer than 25 scores changed.
    assert len(scorer._load_checkpoint(snapshot_output / 'single_point_partial_checkpoint.json.gz')) == 1
    (snapshot_output / 'single_point_checkpoint.json.gz').unlink()
    assert snapshot['snapshot_namespace'] == 'recovery-20261006'
    assert snapshot['scoring_run_id'] != parallel['scoring_run_id']
    before_resume = calls.copy()
    scorer.main()
    assert calls == before_resume
    sys.argv[sys.argv.index('--arm-artifact') + 1] = 'entity/project/probe:latest'
    with pytest.raises(SystemExit):
        scorer.main()


def test_formation_energy_requires_matched_elemental_references():
    fields = _energy_definitions()
    predicted = EnergyField.from_dict(fields["orb"]["energy_per_atom"])
    assert _matched_formation_energy(-17., Composition("Cu2O3"), {"Cu": -3., "O": -2.},
                                      predicted, predicted) == pytest.approx(-1.)
    wrong_reference = EnergyField.from_dict(fields["mace"]["energy_per_atom"])
    with pytest.raises(IncompatibleEnergyFieldError):
        _matched_formation_energy(-17., Composition("Cu2O3"), {"Cu": -3., "O": -2.},
                                   predicted, wrong_reference)


def test_checkpoint_restores_numpy_properties_and_failed_metrics_are_not_complete(tmp_path):
    structure = Structure(Lattice.cubic(3.61), ["Cu"], [[0, 0, 0]],
                          properties={"graph_embedding_orb": np.array([1., 2.])})
    checkpoint = tmp_path / "checkpoint.json.gz"
    dumpfn([structure], checkpoint)
    restored = scorer._load_checkpoint(checkpoint)
    np.testing.assert_array_equal(restored[0].properties["graph_embedding_orb"], [1., 2.])
    assert scorer._metrics_complete({"primary_metric": "fd", "metrics": {"fd": .5}})
    assert not scorer._metrics_complete({"primary_metric": "fd", "metrics": {"fd": float('nan')}})
    assert not scorer._metrics_complete({"primary_metric": "fd", "metrics": {}})
    assert not scorer._metrics_complete({"primary_metric": "fd", "metrics": {"fd": .5},
                                        "uncertainties": {"n_models_computed": 2}})


def test_novelty_prefilter_and_short_circuit_preserve_structure_matcher_results(monkeypatch):
    import pandas as pd
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from scipy.sparse import csr_matrix

    references = [Structure(Lattice.cubic(3.61), ['Cu'] * 4,
                            [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]),
                  Structure(Lattice.cubic(3.61), ['Cu'], [[0, 0, 0]]),
                  Structure(Lattice.cubic(5.6), ['Na', 'Cl'], [[0, 0, 0], [.5, .5, .5]])]
    larger = references[0].copy()
    larger.make_supercell([2, 1, 1])
    charged = Structure(Lattice.cubic(4), ['Fe2+', 'O2-'], [[0, 0, 0], [.5, .5, .5]])
    references.extend([larger, charged])
    items = [{'species_at_sites': [str(site.specie) for site in structure],
              'lattice_vectors': structure.lattice.matrix,
              'cartesian_site_positions': structure.cart_coords} for structure in references]

    def convert(item):
        return Structure(item['lattice_vectors'], item['species_at_sites'],
                         item['cartesian_site_positions'], coords_are_cartesian=True)

    monkeypatch.setitem(sys.modules, 'lemat_genbench.fingerprinting.encode_compositions',
                        types.SimpleNamespace(filter_df=lambda frame, matrix, structure: frame,
                                              lematbulk_item_to_structure=convert))
    matcher = StructureMatcher(ltol=.1)
    fingerprinter = types.SimpleNamespace(matcher=matcher, is_equivalent=matcher.fit)
    dataset = types.SimpleNamespace(select=lambda indices: [items[index] for index in indices])
    matrix = np.zeros((len(references), 118))
    for index, structure in enumerate(references):
        for element in structure.composition.elements:
            matrix[index, element.Z - 1] = 1
    info = {'dataset_dataframe': pd.DataFrame({'index_number': range(len(items))}),
            'all_compositions': csr_matrix(matrix), 'dataset': dataset}
    queries = references + [Structure(Lattice.cubic(4), ['Xe'], [[0, 0, 0]]),
                            Structure(Lattice.cubic(4), ['Na', 'Na', 'Cl'],
                                      [[0, 0, 0], [.2, .2, .2], [.5, .5, .5]])]
    for structure in queries:
        expected = float(not any(matcher.fit(structure, reference) for reference in references))
        assert scorer._novelty_match(structure, info, fingerprinter) == expected
    expected_matrix = np.array([[int(matcher.fit(left, right)) for right in queries]
                                for left in queries])
    np.testing.assert_array_equal(scorer._pairwise_matching(fingerprinter, queries), expected_matrix)
    cached_matrix = np.array([[int(scorer._composition_can_match(fingerprinter, left, right)
                                  and matcher.fit(left, right)) for right in queries]
                              for left in queries])
    np.testing.assert_array_equal(cached_matrix, expected_matrix)
    assert len(fingerprinter._mlip_bias_composition_cache) == len(queries)
    # Ignored species and subset matching must retain the full matcher path.
    fingerprinter.matcher = StructureMatcher(allow_subset=True)
    assert scorer._composition_can_match(fingerprinter, queries[0], queries[-1])
    fingerprinter.matcher = StructureMatcher(ignored_species=['Cu'])
    assert scorer._composition_can_match(fingerprinter, queries[0], queries[-1])
