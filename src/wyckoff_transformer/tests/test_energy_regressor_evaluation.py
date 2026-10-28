"""Numerical and representation checks for the saved-energy evaluation."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def evaluation():
    path = Path(__file__).resolve().parents[3] / "scripts/evaluate_energy_regressor.py"
    spec = importlib.util.spec_from_file_location("energy_regressor_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_errors_use_only_finite_pairs_and_signed_bias(evaluation):
    result = evaluation.statistics(
        pd.DataFrame(
            dict(prediction=[1.0, 3.0, np.nan, np.inf], truth=[2.0, 1.0, 0.0, 0.0])
        )
    )
    assert result["n"] == 2
    assert result["excluded"] == 2
    assert result["mae"] == pytest.approx(1.5)
    assert result["rmse"] == pytest.approx(np.sqrt(2.5))
    assert result["bias"] == pytest.approx(0.5)


def test_no_target_has_explicit_zero_coverage(evaluation):
    result = evaluation.statistics(pd.DataFrame(dict(prediction=[1.0], truth=[np.nan])))
    assert result == dict(n_rows=1, n=0, excluded=1)


def test_common_elemental_reference_preserves_energy_residual(evaluation):
    # Different cell sizes must not change the intensive error.
    for atoms in [5, 20]:
        predicted_formation = -2.0
        elemental_baseline = -4.0
        total = -5.9 * atoms
        truth = total / atoms - elemental_baseline
        r = evaluation.statistics(
            pd.DataFrame(dict(prediction=[predicted_formation], truth=[truth]))
        )
        assert r["bias"] == pytest.approx(-0.1)


def test_gene_roundtrip_preserves_repeated_element_orbits(evaluation):
    from wyckoff_transformer.evaluation.protocol import GeneFingerprinter

    fp = GeneFingerprinter()
    gene = dict(
        group=62, species=["Ca", "Ti", "O"], sites=[["4c"], ["4b"], ["4c", "8d"]]
    )
    record = fp.record(gene)
    assert fp.fingerprint(evaluation.gene_from_record(record)) == fp.fingerprint(gene)
