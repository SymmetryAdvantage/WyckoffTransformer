"""Tests for joining the screener's verdict to a protocol run.

Enrichment here is a ratio of two rates measured on the same relaxed structures,
so the arithmetic has to be right at the edges: a perfect ranking, a useless one,
and rows whose reconstruction failed and have no energy at all.
"""
import unittest

import numpy as np
import pandas as pd

from wyckoff_transformer.formula_energy import prefilter


def _scored(n=100, perfect=True):
    """Ten metastable rows in a hundred, ranked either perfectly or backwards."""
    good = np.arange(n) < 10
    score = np.where(good, -1.0, 1.0) if perfect else np.where(good, 1.0, -1.0)
    return pd.DataFrame({
        "e_above_hull": np.where(good, 0.05, 0.5),
        "valid_structure": True, "unique_structure": True, "novel_structure": True,
        "score_naive": score, "score_adjusted": score,
    })


class TestReducedKeys(unittest.TestCase):
    def test_cell_formulas_are_reduced(self):
        # structures.csv carries the cell formula, the screener wants the compound.
        self.assertEqual(prefilter.reduced_keys(["Al4Mn2Nd12"]), ["Al2Mn1Nd6"])

    def test_already_reduced_formulas_survive(self):
        self.assertEqual(prefilter.reduced_keys(["AuSc3Zn2"]), ["Au1Sc3Zn2"])


    def test_a_missing_formula_does_not_raise(self):
        # structures.csv carries no formula for a gene whose reconstruction failed.
        self.assertEqual(prefilter.reduced_keys([float("nan"), "NaCl"]), [None, "Cl1Na1"])


class TestFormulasFromGenes(unittest.TestCase):
    def test_composition_is_read_from_the_gene(self):
        import gzip, json, tempfile
        from pathlib import Path

        genes = [{"group": 225, "sites": [["4a"], ["4b"]],
                  "species": ["Na", "Cl"], "numIons": [4, 4]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "genes.json.gz"
            with gzip.open(path, "wt") as handle:
                json.dump(genes, handle)
            self.assertEqual(prefilter.formulas_from_genes(path), ["Na4Cl4"])
            self.assertEqual(prefilter.reduced_keys(prefilter.formulas_from_genes(path)),
                             ["Cl1Na1"])


class TestSliceTable(unittest.TestCase):
    def test_a_perfect_ranking_concentrates_everything(self):
        table = prefilter.slice_table(_scored(), budgets=(10, 50))
        row = table[(table.outcome == "metastable") & (table.budget == 10)].iloc[0]
        self.assertEqual(row["hits"], 10)
        self.assertAlmostEqual(row["rate"], 1.0)
        self.assertAlmostEqual(row["base_rate"], 0.1)
        self.assertAlmostEqual(row["enrichment"], 10.0)

    def test_the_whole_run_always_has_enrichment_one(self):
        table = prefilter.slice_table(_scored(), budgets=(10,))
        whole = table[(table.outcome == "metastable") & (table.budget == 100)].iloc[0]
        self.assertAlmostEqual(whole["enrichment"], 1.0)

    def test_a_backwards_ranking_scores_below_one(self):
        table = prefilter.slice_table(_scored(perfect=False), budgets=(10,))
        row = table[(table.outcome == "metastable") & (table.budget == 10)].iloc[0]
        self.assertEqual(row["hits"], 0)
        self.assertAlmostEqual(row["enrichment"], 0.0)

    def test_failed_reconstructions_count_as_misses(self):
        # No energy means the budget was spent and nothing came back. Dropping
        # those rows would flatter every rate.
        frame = _scored()
        frame.loc[0, "e_above_hull"] = np.nan
        table = prefilter.slice_table(frame, budgets=(10,))
        row = table[(table.outcome == "metastable") & (table.budget == 10)].iloc[0]
        self.assertEqual(row["hits"], 9)
        self.assertAlmostEqual(row["base_rate"], 0.09)

    def test_sun_demands_novelty_as_well_as_energy(self):
        frame = _scored()
        frame["novel_structure"] = False
        table = prefilter.slice_table(frame, budgets=(10,))
        self.assertEqual(table[(table.outcome == "metasun")]["hits"].sum(), 0)
        self.assertGreater(table[(table.outcome == "metastable")]["hits"].sum(), 0)

    def test_the_interval_brackets_the_estimate_and_widens_when_rare(self):
        table = prefilter.slice_table(_scored(), budgets=(10,))
        row = table[(table.outcome == "metastable") & (table.budget == 10)].iloc[0]
        self.assertLessEqual(row["enrichment_low"], row["enrichment"])
        self.assertGreaterEqual(row["enrichment_high"], row["enrichment"])
        # Ten of ten is still only ten events, so the lower end stays well below.
        self.assertLess(row["enrichment_low"], row["enrichment"])

    def test_an_interval_spanning_one_means_indistinguishable_from_random(self):
        # A single hit in a slice of ten, against a 10% base rate.
        frame = _scored()
        frame["score_adjusted"] = np.arange(len(frame), dtype=float)
        frame["e_above_hull"] = 0.5
        frame.loc[5, "e_above_hull"] = 0.05
        frame.loc[50:58, "e_above_hull"] = 0.05
        table = prefilter.slice_table(frame, budgets=(10,))
        row = table[(table.outcome == "metastable") & (table.budget == 10)].iloc[0]
        self.assertLess(row["enrichment_low"], 1.0)
        self.assertGreater(row["enrichment_high"], 1.0)

    def test_novel_only_drops_the_known_formulas(self):
        frame = _scored()
        frame["formula_known"] = np.arange(len(frame)) < 10   # exactly the good ones
        whole = prefilter.slice_table(frame, budgets=(10,))
        novel = prefilter.slice_table(frame, budgets=(10,), novel_only=True)
        # Ranking looks perfect until the known formulas are removed, at which
        # point there is nothing left to find.
        self.assertAlmostEqual(
            whole[(whole.outcome == "metastable") & (whole.budget == 10)].iloc[0]["rate"], 1.0)
        self.assertAlmostEqual(
            novel[(novel.outcome == "metastable") & (novel.budget == 10)].iloc[0]["rate"], 0.0)

    def test_novel_only_needs_the_column(self):
        with self.assertRaises(ValueError):
            prefilter.slice_table(_scored(), novel_only=True)

    def test_budgets_beyond_the_run_are_dropped(self):
        table = prefilter.slice_table(_scored(), budgets=(10, 10_000))
        self.assertEqual(sorted(table["budget"].unique()), [10, 100])

    def test_an_unscorable_run_is_refused(self):
        frame = _scored()
        frame["score_adjusted"] = np.nan
        with self.assertRaises(ValueError):
            prefilter.slice_table(frame)


if __name__ == "__main__":
    unittest.main()
