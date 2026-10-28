"""Scientific selection controls for the minimum-per-gene repeat."""

import numpy as np
import pandas as pd
import pytest

from wyckoff_transformer.evaluation.gene_minimum_evaluation import (
    add_archive_candidates, error_statistics, select_minima,
)


def candidates():
    return pd.DataFrame(dict(
        candidate_id=["a", "b", "c", "d"], pool=["DFT"] * 4,
        group_key=["g", "g", "h", "h"], final_key=["g", "g", "h", "h"],
        geometry=["a", "b", "c", "d"], truth=[-.8, -1., -2., -1.9],
        final_prediction=[-.8, -.8, np.nan, -1.9],
        novel_structure_fixed=[True, False, True, True], origin=["generated"] * 4,
    ))


def test_selection_precedes_novelty_and_prediction_filtering():
    chosen, audit = select_minima(candidates())
    assert chosen.candidate_id.tolist() == ["b", "c"]
    assert chosen.novel_structure_fixed.tolist() == [False, True]
    assert error_statistics(chosen)["n"] == 1
    assert error_statistics(chosen)["excluded"] == 1
    assert audit.selection_status.tolist() == ["higher energy or tie", "selected", "selected", "higher energy or tie"]


def test_sources_stay_separate_and_ties_are_deterministic():
    frame = candidates().iloc[:2].copy()
    frame.truth = -1.
    other = frame.iloc[:1].assign(candidate_id="orb", pool="ORB", truth=-3.)
    frame = pd.concat([frame, other], ignore_index=True)
    chosen, _ = select_minima(frame.sample(frac=1, random_state=42))
    assert chosen.candidate_id.tolist() == ["a", "orb"]


def test_unresolved_and_unlabelled_rows_are_audited():
    frame = candidates()
    frame.loc[0, "group_key"] = None
    frame.loc[1, "truth"] = np.inf
    chosen, audit = select_minima(frame)
    assert chosen.candidate_id.tolist() == ["c"]
    assert audit.selection_status.iloc[:2].tolist() == ["unresolved gene", "missing energy"]


def test_minimum_is_intensive_energy_not_total_cell_energy():
    frame = candidates().iloc[:2].copy()
    frame["energy"] = [-8., -4.]
    # First candidate has more atoms, and the higher energy per atom.
    chosen, _ = select_minima(frame)
    assert chosen.candidate_id.tolist() == ["b"]


def test_archive_union_uses_same_genes_and_can_keep_a_lower_generated_structure():
    generated = candidates().assign(dataset="c", cohort="generated", material_id="example")
    reference = pd.DataFrame(dict(
        key=["g", "g", "h", "archive-only"],
        formation_energy_per_atom=[-1.1, -.9, -1.5, -10.],
        immutable_id=["ref1", "ref2", "ref3", "ref4"], split=["train"] * 4,
    ))
    union = add_archive_candidates(generated, reference)
    union["group_key"] = union.final_key
    union["pool"] = "union"
    chosen, _ = select_minima(union)
    assert chosen.candidate_id.tolist() == ["archive:ref1", "c"]
    assert chosen.truth.tolist() == [-1.1, -2.]
    assert chosen.final_key.tolist() == ["g", "h"]
    assert chosen.iloc[0].novel_structure_fixed == False
    assert pd.isna(chosen.iloc[0].get("energy"))


def test_archive_union_refuses_different_predictions_for_one_gene():
    frame = candidates()
    frame.loc[1, "final_prediction"] = -1.
    with pytest.raises(ValueError, match="inconsistent"):
        add_archive_candidates(frame, pd.DataFrame())
