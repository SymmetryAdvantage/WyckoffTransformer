"""Tests for the DFT-only composition-plus-gene fixed-hull screen."""
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from wyckoff_transformer.cli import dft_screen


def _gene_scores() -> pd.DataFrame:
    return pd.DataFrame({
        "formula": ["NaCl", "KBr", "Li2F2", pd.NA],
        "predicted_formation_energy": [-1.2, -0.7, -0.8, np.nan],
        "hull_energy": [-1.0, -0.8, -0.5, np.nan],
        "score": [-0.2, 0.1, -0.3, np.nan],
        "predicted_below_hull": pd.array([True, False, True, pd.NA], dtype="boolean"),
        "reason": [pd.NA, pd.NA, pd.NA, "invalid gene"],
    })


def _composition_predictions() -> pd.DataFrame:
    return pd.DataFrame({
        "location": [-1.3, -1.0, -0.4],
        "sigma_epistemic": [0.1, 0.05, 0.1],
        "scale": [0.4, 0.9, 0.2],
        "target": np.nan,
        "hull": [-1.0, -0.8, -0.5],
    }, index=pd.Index(["Cl1Na1", "Br1K1", "F1Li1"]))


class TestFormulaValidation(unittest.TestCase):
    def test_requires_the_censored_floor_estimand(self):
        with self.assertRaisesRegex(ValueError, "latent composition floor"):
            dft_screen.validate_formula_ensemble(
                SimpleNamespace(
                    loss="mse",
                    location_features=(),
                    energy_scale=dft_screen.formula_train.LEMAT_BULK_PBE_ENERGY_SCALE,
                ),
                (),
            )

    def test_rejects_formula_specific_provenance_at_inference(self):
        with self.assertRaisesRegex(ValueError, "cannot supply"):
            dft_screen.validate_formula_ensemble(
                SimpleNamespace(
                    loss="censored",
                    location_features=("log1p_n_rows",),
                    energy_scale=dft_screen.formula_train.LEMAT_BULK_PBE_ENERGY_SCALE,
                ),
                ("log1p_n_rows",),
            )

    def test_accepts_chemistry_only_and_system_density_floors(self):
        dft_screen.validate_formula_ensemble(
            SimpleNamespace(
                loss="censored",
                location_features=("log1p_sys_entries_per_binary",),
                energy_scale=dft_screen.formula_train.LEMAT_BULK_PBE_ENERGY_SCALE,
            ),
            ("log1p_sys_entries_per_binary",),
        )

    def test_legacy_energy_scale_fails_closed_without_an_override(self):
        legacy = SimpleNamespace(loss="censored", location_features=(), energy_scale=None)
        with self.assertRaisesRegex(ValueError, "does not verify"):
            dft_screen.validate_formula_ensemble(legacy, ())
        dft_screen.validate_formula_ensemble(
            legacy,
            (),
            allow_unverified_energy_scale=True,
        )

    def test_gene_critic_must_record_the_dft_training_dataset(self):
        verified = SimpleNamespace(training_dataset_name="lemat_bulk_fmax1")
        dft_screen.validate_gene_energy_scale(verified)
        with self.assertRaisesRegex(ValueError, "does not verify"):
            dft_screen.validate_gene_energy_scale(
                SimpleNamespace(training_dataset_name="mlip_relaxations")
            )


class TestScoreCombination(unittest.TestCase):
    def test_joint_score_is_a_conservative_conjunction(self):
        combined = dft_screen.combine_scores(
            _gene_scores(),
            _composition_predictions(),
            pd.DataFrame(index=["Cl1Na1"]),
        )

        # NaCl clears both adjusted component screens.
        self.assertAlmostEqual(combined.loc[0, "composition_score_adjusted"], -0.2)
        self.assertAlmostEqual(combined.loc[0, "gene_score"], -0.2)
        self.assertAlmostEqual(combined.loc[0, "joint_score_adjusted"], -0.2)
        self.assertTrue(combined.loc[0, "joint_score_adjusted_below_hull"])

        # KBr clears the formula screen but not the gene screen; the max fails.
        self.assertLess(combined.loc[1, "composition_score_adjusted"], 0.0)
        self.assertGreater(combined.loc[1, "gene_score"], 0.0)
        self.assertAlmostEqual(
            combined.loc[1, "joint_score_adjusted"],
            combined.loc[1, "gene_score"],
        )
        self.assertFalse(combined.loc[1, "joint_score_adjusted_below_hull"])

        # LiF clears the gene screen but not the uncertainty-adjusted formula screen.
        self.assertGreaterEqual(combined.loc[2, "composition_score_adjusted"], 0.0)
        self.assertLess(combined.loc[2, "gene_score"], 0.0)
        self.assertFalse(combined.loc[2, "joint_score_adjusted_below_hull"])

    def test_excess_scale_does_not_enter_the_floor_score(self):
        predictions = _composition_predictions()
        first = dft_screen.combine_scores(
            _gene_scores(), predictions, pd.DataFrame(index=[])
        )
        predictions["scale"] = predictions["scale"] + 1000.0
        second = dft_screen.combine_scores(
            _gene_scores(), predictions, pd.DataFrame(index=[])
        )
        pd.testing.assert_series_equal(
            first["composition_score_adjusted"],
            second["composition_score_adjusted"],
        )

    def test_cell_multiples_share_one_formula_prediction(self):
        genes = _gene_scores().iloc[[0, 2]].copy()
        genes.loc[2, "formula"] = "Na2Cl2"
        predictions = _composition_predictions().loc[["Cl1Na1"]]
        combined = dft_screen.combine_scores(genes, predictions, pd.DataFrame(index=[]))
        self.assertEqual(combined.loc[0, "reduced_formula"], "Cl1Na1")
        self.assertEqual(combined.loc[2, "reduced_formula"], "Cl1Na1")
        self.assertEqual(
            combined.loc[0, "composition_floor"],
            combined.loc[2, "composition_floor"],
        )

    def test_unscorable_rows_remain_unknown_and_sort_last(self):
        combined = dft_screen.combine_scores(
            _gene_scores(),
            _composition_predictions(),
            pd.DataFrame(index=[]),
        )
        self.assertTrue(pd.isna(combined.loc[3, "joint_score_adjusted"]))
        self.assertTrue(pd.isna(combined.loc[3, "joint_score_adjusted_below_hull"]))
        self.assertEqual(combined.index[-1], 3)
        self.assertEqual(combined.loc[3, "reason"], "invalid gene")

    def test_known_formula_is_reported_but_not_used_as_a_filter(self):
        combined = dft_screen.combine_scores(
            _gene_scores(),
            _composition_predictions(),
            pd.DataFrame(index=["Cl1Na1"]),
        )
        self.assertTrue(combined.loc[0, "formula_known"])
        self.assertFalse(combined.loc[1, "formula_known"])
        self.assertIn(0, combined.index)

    def test_formula_table_and_live_reference_must_share_the_hull(self):
        table = pd.DataFrame(
            {"e_hull_at_composition": [-0.9]},
            index=pd.Index(np.asarray(["Cl1Na1"], dtype=object)),
        )
        with self.assertRaisesRegex(ValueError, "inconsistent hull energies"):
            dft_screen.combine_scores(
                _gene_scores(),
                _composition_predictions(),
                table,
            )

    def test_inconsistent_hull_for_one_reduced_formula_is_rejected(self):
        frame = _gene_scores().iloc[[0, 2]].copy()
        frame.loc[2, "formula"] = "Na2Cl2"
        frame.loc[2, "hull_energy"] = -0.7
        frame["reduced_formula"] = ["Cl1Na1", "Cl1Na1"]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            dft_screen._formula_hulls(frame)


class TestParser(unittest.TestCase):
    def test_cli_is_explicitly_dft_only(self):
        parser = dft_screen.build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertNotIn("--mlip", option_strings)
        self.assertIn("--allow-unverified-energy-scale", option_strings)
        self.assertEqual(
            parser.get_default("reference"),
            dft_screen.DEFAULT_REFERENCE,
        )
        self.assertEqual(
            parser.get_default("rank_by"),
            "joint_score_adjusted",
        )


if __name__ == "__main__":
    unittest.main()
