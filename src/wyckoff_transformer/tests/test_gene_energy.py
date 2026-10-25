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
    """One row of the gene table.

    The augmentation carries a site-symmetry tuple per enumeration variant, since
    a relabelling can change the oriented symbol (docs/wyckoff_augmentation_audit.md).
    These fixtures permute only the indices, so every variant repeats *symmetries*.
    """
    return {
        "spacegroup_number": 1,
        "elements": elements,
        "site_symmetries": symmetries,
        "site_symmetries_augmented": [list(symmetries)] * len(enumerations),
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

    def test_per_split_keeps_each_splits_labels_to_itself(self):
        # The same gene in train and val: no split's energy sets the other's target.
        train = pd.DataFrame([
            _row(["Na", "Cl"], ["m", "-1"], [[0, 1], [1, 0]], -1.0),
            _row(["Na", "Cl"], ["m", "-1"], [[0, 1], [1, 0]], -1.1),
        ])
        val = pd.DataFrame([
            _row(["Cl", "Na"], ["-1", "m"], [[1, 0], [0, 1]], -1.4),
        ])

        add_observed_gene_minimum({"train": train, "val": val}, per_split=True)

        self.assertEqual(train[GENE_MIN_FORMATION_ENERGY_COLUMN].tolist(), [-1.1, -1.1])
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
                "site_symmetries_augmented": [["1"]],
                "sites_enumeration_augmented": [[0]],
                "spacegroup_number": 1,
            },
            {
                "elements": ["Cl"],
                "multiplicity": [1],
                "site_symmetries": ["1"],
                "sites_enumeration": [0],
                "site_symmetries_augmented": [["1"]],
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

    def test_any_scalar_target_is_predicted_without_a_hull(self):
        # A band-gap regressor: unconditioned, not a screen target, so score_genes
        # would refuse it; predict_gene_scalars reports the mean and spread of the
        # augmentation draws and why an unreadable gene has none.
        class _BandGap:
            target = TargetClass.Scalar
            target_name = "band_gap"

            def predict_scalars(self, tensors, augmentation_samples, cond):
                self.cond = cond
                return (torch.tensor([2.0]),
                        torch.tensor([[1.0], [3.0]])[:augmentation_samples])

        record = {
            "elements": ["Li", "Cl"],
            "multiplicity": [4, 4],
            "site_symmetries": ["m-3m", "m-3m"],
            "sites_enumeration": [0, 1],
            "spacegroup_number": 225,
        }
        regressor = _BandGap()
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
        ):
            fingerprinter.return_value.record.side_effect = [KeyError("4z"), record]
            predicted = gene_screen.predict_gene_scalars(
                [{"gene": 0}, {"gene": 1}], regressor, augmentation_samples=2)

        self.assertIsNone(regressor.cond)
        self.assertTrue(pd.isna(predicted.loc[0, "prediction"]))
        self.assertIn("KeyError", predicted.loc[0, "reason"])
        self.assertEqual(predicted.loc[1, "formula"], "Cl4Li4")
        self.assertAlmostEqual(predicted.loc[1, "prediction"], 2.0)
        self.assertAlmostEqual(predicted.loc[1, "prediction_std"], 2 ** 0.5)
        self.assertTrue(pd.isna(predicted.loc[1, "reason"]))

    def test_repeated_genes_keep_their_best_scored_row(self):
        # Two orders of the same sites are one gene; the screen's score order decides
        # which row survives, so the input order of the gene list must not.
        nacl = {"group": 225, "species": ["Na", "Cl"], "numIons": [4, 4],
                "sites": [["4a"], ["4b"]]}
        clna = {"group": 225, "species": ["Cl", "Na"], "numIons": [4, 4],
                "sites": [["4b"], ["4a"]]}
        kcl = {"group": 225, "species": ["K", "Cl"], "numIons": [4, 4],
               "sites": [["4a"], ["4b"]]}
        scored = pd.DataFrame({"score": [-0.3, -0.2, 0.1]}, index=[2, 0, 1])
        kept = gene_screen.drop_repeated_genes(scored, [nacl, kcl, clna])
        self.assertEqual(list(kept.index), [2, 1])

    def test_written_genes_round_trip(self):
        import gzip
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        genes = [{"group": 225, "species": ["Na"], "numIons": [4], "sites": [["4a"]]}]
        with TemporaryDirectory() as tmp:
            for name, opener in (("genes.json", open), ("genes.json.gz", gzip.open)):
                path = Path(tmp) / "out" / name
                gene_screen.write_genes(genes, path)
                with opener(path, "rt", encoding="utf-8") as handle:
                    self.assertEqual(json.load(handle), genes)
