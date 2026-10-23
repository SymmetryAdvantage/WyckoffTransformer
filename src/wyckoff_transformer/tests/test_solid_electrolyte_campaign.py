"""The pure parts of scripts/solid_electrolyte_campaign.py: systems, screens, selection."""
import sys
import threading
import unittest
from pathlib import Path

import pandas as pd

scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))
from solid_electrolyte_campaign import (
    DEFAULT_CONFIG,
    RelaxQueue,
    expand_systems,
    load_config,
    oxidation_states,
    select_system,
)


class TestSystems(unittest.TestCase):
    def test_templates_expand_to_canonical_systems_and_runs(self):
        config = {"families": [
            {"name": "halide", "templates": [["Li", ["Y", "In"], ["Cl", "Br"]]],
             "split_by": ["Cl", "Br"]},
            {"name": "nitride", "templates": [["Li", "B", "N"], ["Li", "N", ["Cl"]]]},
        ]}
        systems = expand_systems(config)
        self.assertEqual(list(systems["system"]),
                         ["Cl-Li-Y", "Br-Li-Y", "Cl-In-Li", "Br-In-Li", "B-Li-N", "Cl-Li-N"])
        self.assertEqual(list(systems["run"]), ["halide_Cl", "halide_Br", "halide_Cl",
                                                "halide_Br", "nitride", "nitride"])

    def test_a_system_listed_twice_is_refused(self):
        config = {"families": [{"name": "a", "templates": [["Li", "O", "Cl"]]},
                               {"name": "b", "templates": [["Cl", "Li", "O"]]}]}
        with self.assertRaisesRegex(ValueError, "twice"):
            expand_systems(config)

    def test_the_campaign_config_expands(self):
        systems = expand_systems(load_config(DEFAULT_CONFIG))
        self.assertEqual(len(systems), 126)
        self.assertTrue(systems["system"].str.contains("Li").all())


class TestChargeBalance(unittest.TestCase):
    def test_family_prototypes_balance(self):
        for formula in ("Li3YCl6", "LiNbOCl4", "Li6PS5Cl", "Li7Si2S7I", "Li10GeP2S12",
                        "Li2HOCl", "Li9AlP4", "Li7PN4", "Li5NCl2", "Li7La3Zr2O12",
                        "LiTi2(PO4)3", "Li5B7S13", "LiBH4"):
            self.assertIsNotNone(oxidation_states(formula), formula)
        self.assertEqual(oxidation_states("Li3YCl6"), "Cl-1 Li+1 Y+3")

    def test_impossible_stoichiometries_do_not(self):
        for formula in ("Li46P4S7Cl4", "LiY2Cl8", "Li10P3S21Cl4", "LiLa10Zr4O19"):
            self.assertIsNone(oxidation_states(formula), formula)
        self.assertIsNone(oxidation_states(None))


def _pool(rows):
    columns = ["reduced_formula", "predicted_e_hull", "band_gap_pred", "gene_novel",
               "obelix_known", "exact_system", "count"]
    return pd.DataFrame(rows, columns=columns).assign(n_atoms=10)


_SELECTION_CONFIG = {"screens": {"band_gap": {"min_ev": 0.5}},
                     "selection": {"top_k_per_system": 3, "max_per_formula": 2}}


class TestSelection(unittest.TestCase):

    def test_filters_in_order_then_rank_with_a_per_formula_cap(self):
        pool = _pool([
            ("Li3YCl6", -0.30, 4.0, True, True, True, 1),     # known SE composition
            ("LiYCl4", -0.20, 4.0, False, False, True, 5),    # gene in LeMat-Bulk
            ("LiY2Cl8", -0.50, 4.0, True, False, True, 1),    # not charge balanced
            ("Li2YCl5", -0.10, 0.2, True, False, True, 1),    # no gap
            ("Li4YCl7", 0.05, 4.0, True, False, True, 2),
            ("Li4YCl7", 0.01, 4.0, True, False, True, 1),
            ("Li4YCl7", 0.02, 4.0, True, False, True, 1),     # third of its formula
            ("Li5YCl8", 0.30, 3.0, True, False, True, 1),
            ("Li6YCl9", 0.40, 3.0, True, False, True, 1),     # past top-K
            ("Li6YCl9", float("nan"), 3.0, True, False, True, 1),
        ])
        chosen, counts = select_system(pool, _SELECTION_CONFIG)
        self.assertEqual(list(chosen.index), [5, 6, 7])
        self.assertEqual(counts["gene_novel"], 9)
        self.assertEqual(counts["not_obelix"], 8)
        self.assertEqual(counts["charge_balanced"], 7)
        self.assertEqual(counts["band_gap"], 6)
        self.assertEqual(counts["energy_scored"], 5)
        self.assertEqual(counts["energy_scored_draws"], 6)
        self.assertEqual(counts["selected"], 3)
        self.assertEqual(counts["selected_atoms"], 30)

    def test_an_atom_budget_stops_the_ranking_early(self):
        pool = _pool([("Li3Y2Cl9", -0.1, 4.0, True, False, True, 1),
                      ("Li5YCl8", 0.0, 4.0, True, False, True, 1),
                      ("Li6YCl9", 0.1, 4.0, True, False, True, 1)])
        pool["n_atoms"] = [14, 28, 16]
        config = {**_SELECTION_CONFIG,
                  "selection": {**_SELECTION_CONFIG["selection"], "atom_budget_per_system": 45}}
        chosen, counts = select_system(pool, config)
        self.assertEqual(list(chosen.index), [0, 1])
        self.assertEqual(counts["selected_atoms"], 42)


class TestRelaxQueue(unittest.TestCase):
    def test_a_capped_lane_skips_large_runs_and_stops_when_closed(self):
        lanes = RelaxQueue()
        lanes.put("big", 5000)
        lanes.put("small", 100)
        self.assertEqual(lanes.take(max_genes=2500, remote=True), "small")
        lanes.remote_done("small", 100, succeeded=True)
        lanes.close()
        self.assertIsNone(lanes.take(max_genes=2500, remote=True))
        self.assertEqual(lanes.take(), "big")
        self.assertIsNone(lanes.take())

    def test_a_failed_remote_run_comes_back_for_the_local_lane_only(self):
        lanes = RelaxQueue()
        lanes.put("run", 100)
        self.assertEqual(lanes.take(max_genes=2500, remote=True), "run")
        lanes.close()
        taken = []
        local = threading.Thread(target=lambda: taken.append(lanes.take()))
        local.start()               # must wait: the remote run may come back
        local.join(timeout=0.2)
        self.assertTrue(local.is_alive())
        lanes.remote_done("run", 100, succeeded=False)
        local.join(timeout=5)
        self.assertEqual(taken, ["run"])
        self.assertIsNone(lanes.take(max_genes=2500, remote=True))
        self.assertIsNone(lanes.take())

    def test_a_waiting_lane_wakes_for_a_new_run(self):
        lanes, taken = RelaxQueue(), []
        waiter = threading.Thread(target=lambda: taken.append(lanes.take()))
        waiter.start()
        lanes.put("run", 10)
        waiter.join(timeout=5)
        self.assertEqual(taken, ["run"])


if __name__ == "__main__":
    unittest.main()
