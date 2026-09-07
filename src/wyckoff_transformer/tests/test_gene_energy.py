"""Tests for the observed-minimum target used by the initial gene-energy critic."""
import unittest

import pandas as pd
import torch
from unittest.mock import patch

from wyckoff_transformer.cascade.dataset import TargetClass
from wyckoff_transformer.cli import gene_screen
from wyckoff_transformer.gene_energy import (
    GENE_MIN_FORMATION_ENERGY_COLUMN,
    add_observed_gene_minimum,
    build_clean_relaxation_condition,
)


def _row(elements, symmetries, enumerations, energy):
    return {
        "spacegroup_number": 1,
        "elements": elements,
        "site_symmetries": symmetries,
        "sites_enumeration_augmented": enumerations,
        "formation_energy_per_atom": energy,
    }


class TestObservedGeneMinimum(unittest.TestCase):
    def test_uses_one_augmentation_invariant_minimum_across_splits(self):
        # The first two records differ only by site order and the arbitrary
        # enumeration of equivalent Wyckoff positions, so they are one gene.
        train = pd.DataFrame([
            _row(["Na", "Cl"], ["m", "-1"], [[0, 1], [1, 0]], -1.0),
            _row(["K"], ["1"], [[0]], -0.2),
        ])
        val = pd.DataFrame([
            _row(["Cl", "Na"], ["-1", "m"], [[1, 0], [0, 1]], -1.4),
        ])

        add_observed_gene_minimum({"train": train, "val": val})

        self.assertEqual(train[GENE_MIN_FORMATION_ENERGY_COLUMN].tolist(), [-1.4, -0.2])
        self.assertEqual(val[GENE_MIN_FORMATION_ENERGY_COLUMN].tolist(), [-1.4])

    def test_rejects_a_non_finite_source_energy(self):
        frame = pd.DataFrame([
            _row(["Na"], ["1"], [[0]], float("nan")),
        ])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            add_observed_gene_minimum({"train": frame})


class _ConditionedTrainer:
    condition_features = ("max_force",)
    device = torch.device("cpu")

    def build_condition_from_values(self, values, n_rows, device=None):
        self.values = values
        self.n_rows = n_rows
        self.condition_device = device
        return torch.full((n_rows, 1), values["max_force"], device=device)


class TestCleanRelaxationCondition(unittest.TestCase):
    def test_sets_max_force_to_zero_for_inference(self):
        trainer = _ConditionedTrainer()
        condition = build_clean_relaxation_condition(trainer, 3)
        self.assertEqual(trainer.values, {"max_force": 0.0})
        self.assertEqual(condition.tolist(), [[0.0], [0.0], [0.0]])

    def test_rejects_a_regressor_with_an_unrelated_condition(self):
        trainer = _ConditionedTrainer()
        trainer.condition_features = ("energy_above_hull",)
        with self.assertRaisesRegex(ValueError, "max_force"):
            build_clean_relaxation_condition(trainer, 1)


class _Regressor(_ConditionedTrainer):
    target = TargetClass.Scalar
    scalar_loss = "mse"
    target_name = GENE_MIN_FORMATION_ENERGY_COLUMN

    def predict_scalars(self, tensors, augmentation_samples, cond):
        self.predict_condition = cond
        values = torch.tensor([-1.2, -0.3])
        return values, values.unsqueeze(0)


class TestGeneScreen(unittest.TestCase):
    def test_clean_energy_is_compared_to_the_composition_hull(self):
        regressor = _Regressor()
        records = [
            {
                "elements": ["Na"],
                "multiplicity": [1],
                "site_symmetries": ["1"],
                "sites_enumeration": [0],
                "sites_enumeration_augmented": [[0]],
                "spacegroup_number": 1,
            },
            {
                "elements": ["Cl"],
                "multiplicity": [1],
                "site_symmetries": ["1"],
                "sites_enumeration": [0],
                "sites_enumeration_augmented": [[0]],
                "spacegroup_number": 1,
            },
        ]

        with (
            patch("wyckoff_transformer.cli.gene_screen.GeneFingerprinter") as fingerprinter,
            patch(
                "wyckoff_transformer.cli.gene_screen.filter_supported_tokens",
                side_effect=lambda frame, _: (frame, []),
            ),
            patch(
                "wyckoff_transformer.cli.gene_screen.build_tokenised_prediction_tensors",
                return_value={},
            ),
            patch("wyckoff_transformer.cli.gene_screen.HullLookup") as lookup,
        ):
            fingerprinter.return_value.record.side_effect = records
            lookup.return_value.hull_energy_per_atom.side_effect = [-1.0, -0.5]
            scored = gene_screen.score_genes(
                [{"gene": 0}, {"gene": 1}],
                regressor,
                pd.DataFrame(),
            )

        self.assertEqual(regressor.predict_condition.tolist(), [[0.0], [0.0]])
        self.assertAlmostEqual(scored.loc[0, "score"], -0.2)
        self.assertTrue(scored.loc[0, "predicted_below_hull"])
        self.assertAlmostEqual(scored.loc[1, "score"], 0.2)
        self.assertFalse(scored.loc[1, "predicted_below_hull"])
