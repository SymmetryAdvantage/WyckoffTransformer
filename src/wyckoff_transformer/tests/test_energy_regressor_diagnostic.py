"""Check the scientific identities and denominators of the residual diagnostic."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def diagnostic():
    path = Path(__file__).resolve().parents[3] / "scripts/diagnose_energy_regressor.py"
    spec = importlib.util.spec_from_file_location("energy_regressor_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_minimum_uses_all_splits_and_detects_changed_labels(diagnostic):
    matched = pd.DataFrame(dict(
        key=["a", "a", "b"], split=["train", "test", "val"],
        immutable_id=["1", "2", "3"], formation_energy_per_atom=[-1., -2., -3.],
        gene_min_formation_energy_per_atom=[-2., -2., -3.],
    ))
    minima = diagnostic.reference_minima(matched)
    assert minima.loc["a", diagnostic.MINIMUM] == -2.
    assert minima.loc["a", "train_min_formation_energy_per_atom"] == -1.
    assert minima.loc["a", "minimum_split"] == "test"
    assert minima.loc["a", "gene_in_train"]
    assert not minima.loc["b", "gene_in_train"]
    matched.loc[0, diagnostic.MINIMUM] = -1.
    with pytest.raises(ValueError, match="stored training target"):
        diagnostic.reference_minima(matched)


def test_decomposition_allows_new_structures_below_archive_and_keeps_exclusions(diagnostic):
    frame = pd.DataFrame(dict(
        final_key=["a", "a", "novel", "a"], final_gene_novel_fixed=[False, False, True, False],
        final_prediction=[-2.1, -2.1, -2., np.nan], truth=[-1.8, -2.3, -1., -2.],
    ))
    minima = pd.DataFrame({diagnostic.MINIMUM: [-2.]}, index=["a"])
    result = diagnostic.decompose(frame, minima)
    assert result.regressor_error.iloc[0] == pytest.approx(-.1)
    assert result.structure_gap.iloc[0] == pytest.approx(.2)
    assert result.structure_gap.iloc[1] == pytest.approx(-.3)
    assert result.structure_error.iloc[1] == pytest.approx(.2)
    assert result.decomposition_available.tolist() == [True, True, False, False]
    assert result.decomposition_exclusion.tolist() == ["", "", "unknown final gene", "missing prediction"]
    summary = diagnostic.summarize(result)
    assert summary["n"] == 2
    assert summary["excluded"] == 2
    assert summary["regressor_error_mae"] == pytest.approx(.1)


def test_changed_membership_is_refused(diagnostic):
    frame = pd.DataFrame(dict(
        final_key=["a"], final_gene_novel_fixed=[True], final_prediction=[-2.], truth=[-2.],
    ))
    minima = pd.DataFrame({diagnostic.MINIMUM: [-2.]}, index=["a"])
    with pytest.raises(ValueError, match="membership"):
        diagnostic.decompose(frame, minima)


def test_archive_control_uses_dft_while_protocol_uses_saved_orb(diagnostic):
    frame = pd.DataFrame(dict(
        dataset=["a", "b"], final_key=["a", "a"], final_gene_novel_fixed=[False, False],
        final_prediction=[-2., -2.], truth=[-.8, -.8], formation_energy_per_atom=[-1.8, np.nan],
    ))
    minima = pd.DataFrame({diagnostic.MINIMUM: [-2.]}, index=["a"])
    result = diagnostic.decompose(frame, minima)
    assert result.structure_gap.tolist() == pytest.approx([.2, 1.2])
    assert result.truth.tolist() == [-.8, -.8]


def test_large_components_can_cancel_instead_of_adding_maes(diagnostic):
    frame = pd.DataFrame(dict(
        final_key=["a", "b"], decomposition_available=[True, True],
        regressor_error=[2., -2.], structure_gap=[2., -2.], structure_error=[0., 0.],
    ))
    result = diagnostic.summarize(frame)
    assert result["regressor_error_mae"] == 2.
    assert result["structure_gap_mae"] == 2.
    assert result["structure_error_mae"] == 0.
    assert result["regressor_mse"] == 4.
    assert result["gap_mse"] == 4.
    assert result["cross_term"] == -8.


def test_equal_gene_weighting_does_not_weight_repeated_geometries_as_new_genes(diagnostic):
    frame = pd.DataFrame(dict(
        dataset=["c"] * 4, final_key=["a", "a", "a", "b"], geometry=["1", "1", "2", "3"],
        cohort=["x"] * 4, reference_known=[True] * 4,
        track=[None] * 4, valid_structure=[None] * 4,
        same_gene=[None] * 4, novel_structure=[None] * 4,
        novel_structure_fixed=[True] * 4, gene_in_train=[True] * 4,
        decomposition_available=[True] * 4,
        regressor_error=[1., 1., 1., 3.], structure_gap=[0.] * 4, structure_error=[1., 1., 1., 3.],
    ))
    result = diagnostic.summaries(frame)
    result = result[result.description == "DFT all"].set_index("weighting")
    assert result.loc["source_rows", "regressor_error_mae"] == 1.5
    assert result.loc["unique_geometry", "regressor_error_mae"] == pytest.approx(5 / 3)
    assert result.loc["equal_gene", "regressor_error_mae"] == 2.
    assert result.loc["equal_gene", "regressor_error_rmse"] == pytest.approx(np.sqrt(5))
