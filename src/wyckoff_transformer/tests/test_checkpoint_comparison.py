"""Check augmentation replay and the meaning of paired checkpoint differences."""

import numpy as np
import pandas as pd
import pytest
import torch

from wyckoff_transformer.evaluation.checkpoint_comparison import (
    paired_formula_bootstrap, predict_frozen_batch,
)


class ToyModel(torch.nn.Module):
    def forward(self, starts, tokens, padding, mask, cond=None):
        return tokens[0].float().sum(dim=1, keepdim=True) + starts[:, :1]


def test_vectorized_augmentations_preserve_draw_and_gene_order():
    batch = dict(keys=["a", "b"], start_tokens=torch.tensor([[1.], [2.]]),
                 padding_mask=torch.zeros((2, 3), dtype=torch.bool),
                 cascade_samples=[[torch.tensor([[i, 0], [0, i + 1]])] for i in range(5)])
    single = predict_frozen_batch(ToyModel(), batch, "cpu", augmentation_block=1)
    together = predict_frozen_batch(ToyModel(), batch, "cpu", augmentation_block=4)
    np.testing.assert_array_equal(single, together)
    np.testing.assert_array_equal(single, [[1, 3], [2, 4], [3, 5], [4, 6], [5, 7]])


def test_single_gene_batch_retains_augmentation_axis():
    batch = dict(keys=["a"], start_tokens=torch.ones((1, 1)),
                 padding_mask=torch.zeros((1, 2), dtype=torch.bool),
                 cascade_samples=[[torch.ones((1, 1), dtype=torch.long)] for _ in range(3)])
    assert predict_frozen_batch(ToyModel(), batch, "cpu").shape == (3, 1)


def test_identical_checkpoints_have_zero_paired_change():
    result = paired_formula_bootstrap([.1, -.2, .3], [.1, -.2, .3], ["x", "x", "y"], repetitions=100)
    assert result["n_formulas"] == 2
    assert result["final_minus_early_mae"] == 0
    assert result["ci95_lower"] == result["ci95_upper"] == 0


def test_difference_is_per_gene_and_positive_when_final_worsens():
    result = paired_formula_bootstrap([0., 0., 0.], [.1, .1, .4], ["x", "x", "y"], repetitions=1000)
    assert result["final_minus_early_mae"] == pytest.approx(.2)
    assert result["ci95_lower"] > 0
    assert result["ci95_upper"] > result["ci95_lower"]


def test_missing_pairs_are_never_compared_as_zero_errors():
    result = paired_formula_bootstrap([.1, np.nan], [.2, .1], ["x", "y"], repetitions=100)
    assert result["n"] == 1
    assert result["final_minus_early_mae"] == pytest.approx(.1)


def test_csv_roundtrip_preserves_valid_rows_and_fixed_exclusions():
    import importlib.util
    from io import StringIO
    from pathlib import Path
    path = Path(__file__).resolve().parents[3] / "scripts/compare_energy_regressor_checkpoints.py"
    spec = importlib.util.spec_from_file_location("checkpoint_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.DataFrame(dict(
        population=["archive", "generated_dft", "generated_dft"],
        comparison_exclusion=["", "", "bad composition"],
        cohort=["test", "cohort", "cohort"],
        gene_in_train=[False, False, False], final_gene_novel_fixed=[False, True, True],
        contains_gd=[False, False, True], template_known_corpus=[1, 0, 0],
        novel_structure_fixed=[False, True, True],
    ))
    restored = pd.read_csv(StringIO(frame.to_csv(index=False)))
    populations = module.subsets(restored)
    assert len(populations["archive_test_unseen_genes"]) == 1
    assert len(populations["dft_novel_genes"]) == 1
    assert len(populations["dft_novel_no_gd"]) == 1
    assert len(populations["dft_novel_gd"]) == 0
