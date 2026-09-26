"""Tests for the DFT-only composition-plus-gene fixed-hull screen."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from wyckoff_transformer import dataset_manifest as dm
from wyckoff_transformer.cli import dft_screen
from wyckoff_transformer.energy_fields import IncompatibleEnergyFieldError
from wyckoff_transformer.field_provenance import build_field_provenance


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
                ),
                (),
            )

    def test_rejects_formula_specific_provenance_at_inference(self):
        with self.assertRaisesRegex(ValueError, "cannot supply"):
            dft_screen.validate_formula_ensemble(
                SimpleNamespace(
                    loss="censored",
                    location_features=("log1p_n_rows",),
                ),
                ("log1p_n_rows",),
            )

    def test_accepts_chemistry_only_and_system_density_floors(self):
        dft_screen.validate_formula_ensemble(
            SimpleNamespace(
                loss="censored",
                location_features=("log1p_sys_entries_per_binary",),
            ),
            ("log1p_sys_entries_per_binary",),
        )



def _gene_regressor(dataset: str, target: str = "gene_min_formation_energy_per_atom"):
    """A stand-in regressor whose provenance is inferred from its dataset's manifest."""
    config = {"dataset": dataset, "model": {"WyckoffTrainer_args": {
        "target": "Scalar", "target_name": target, "condition_feature": "max_force"}}}
    return SimpleNamespace(field_provenance=build_field_provenance(config, recorded=False))


class TestEnergyFields(unittest.TestCase):
    """All four energy inputs of the screen must mean the same thing."""

    TABLE = Path("formula_table.parquet")
    REFERENCE = Path("lemat_pbe_ehull.csv.gz")

    def setUp(self):
        labelled = {
            self.TABLE: dm.formation_energy_field(
                dm.load_manifest("formula_energy"), "formula_table"),
            self.REFERENCE: dm.formation_energy_field(
                dm.load_manifest("lemat-bulk"), "lemat_pbe_ehull"),
        }
        patcher = patch.object(dft_screen, "file_formation_energy_field",
                               lambda path: labelled.get(Path(path)))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.formula = {"fields": {"target": labelled[self.TABLE].to_dict()}}

    def test_todays_inputs_agree(self):
        # The regressor dft-screen is run with today, trained on the superseded variant:
        # same reference entry set, so its formation energies are the same quantity.
        self.assertEqual(dft_screen.check_energy_fields(
            _gene_regressor("lemat_bulk_fmax1"), self.formula, self.TABLE,
            self.REFERENCE), [])

    def test_an_mp2020_regressor_is_refused_unless_allowed(self):
        mp_20 = _gene_regressor("mp_20")
        with self.assertRaisesRegex(IncompatibleEnergyFieldError, "source.correction"):
            dft_screen.check_energy_fields(mp_20, self.formula, self.TABLE, self.REFERENCE)
        lines = dft_screen.check_energy_fields(
            mp_20, self.formula, self.TABLE, self.REFERENCE, allow_incompatible_energy=True)
        self.assertTrue(any("correction" in line for line in lines))

    def test_an_unlabelled_input_is_refused(self):
        with self.assertRaisesRegex(IncompatibleEnergyFieldError, "unknown"):
            dft_screen.check_energy_fields(
                _gene_regressor("lemat_bulk_fmax1"), self.formula, Path("elsewhere.parquet"),
                self.REFERENCE)
        with self.assertRaisesRegex(IncompatibleEnergyFieldError, "unknown"):
            dft_screen.check_energy_fields(
                _gene_regressor("alex_mp_20"), self.formula, self.TABLE, self.REFERENCE)


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
        self.assertIn("--allow-incompatible-energy", option_strings)
        # The old name still works, for the commands already written down.
        self.assertTrue(parser.parse_args(
            ["genes.json", "--regressor-path", "r", "--allow-unverified-energy-scale"]
        ).allow_incompatible_energy)
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
