"""A CUDA OOM is a one-sided loss, and the sweep has to say how big it could be.

`analyse_temperature_sweep` compares arms drawn at different sampling
temperatures. A cold arm's cohort carries a large-cell tail -- at T=0.7, 3.0% of
genes ask for 100+ atoms against 1.1% at T=1.0 -- and a large cell is what runs
a shared 4.6 GiB K20c out of memory. So the OOM rate is correlated with the arm,
and it can only ever *remove* a relaxation: every rate it touches is biased
downwards, on the cold side, which would read as the cold sampler being worse.

These pin the accounting that keeps that separable from the measurement.
"""
import sys
import unittest
from pathlib import Path

import pandas as pd

_scripts = Path(__file__).resolve().parents[3] / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from analyse_temperature_sweep import oom_accounting, oom_metasun_bound  # noqa: E402

OOM = "OutOfMemoryError: CUDA out of memory. Tried to allocate 94.00 MiB."


def _relaxations(rows):
    return pd.DataFrame(
        rows, columns=["index", "trial", "status", "error"])


def _structures(has_structure):
    return pd.DataFrame(
        {"has_structure": list(has_structure.values())},
        index=pd.Index(list(has_structure), name="index"))


class TestOomAccounting(unittest.TestCase):
    def test_an_arm_with_no_failures_reports_zero(self):
        """The all-empty error column reads back as float64, which has no .str."""
        relaxations = _relaxations([
            (0, 0, "ok", None), (0, 1, "ok", None), (1, 0, "ok", None)])
        accounting = oom_accounting(relaxations, _structures({0: True, 1: True}))
        self.assertEqual(accounting["trials_lost_to_oom"], 0)
        self.assertEqual(accounting["genes_touched_by_oom"], 0)
        self.assertEqual(oom_metasun_bound(accounting, 1000), 0.0)

    def test_only_out_of_memory_counts_as_oom(self):
        """A PyXtal timeout is a different loss and must not inflate the bound."""
        relaxations = _relaxations([
            (0, 0, "failed", OOM),
            (1, 0, "failed", "Timeout: relaxation exceeded 1800 s"),
            (2, 0, "ok", None)])
        accounting = oom_accounting(relaxations, _structures({0: False, 1: False, 2: True}))
        self.assertEqual(accounting["trials_lost_to_oom"], 1)
        self.assertEqual(accounting["genes_touched_by_oom"], 1)

    def test_a_gene_that_lost_every_trial_is_separated_from_one_that_lost_some(self):
        """They bound the readout differently: one is a zero, the other a worse best."""
        relaxations = _relaxations([
            (0, 0, "failed", OOM), (0, 1, "failed", OOM),   # lost everything
            (1, 0, "failed", OOM), (1, 1, "ok", None)])     # lost one of two
        accounting = oom_accounting(relaxations, _structures({0: False, 1: True}))
        self.assertEqual(accounting["trials_lost_to_oom"], 3)
        self.assertEqual(accounting["genes_touched_by_oom"], 2)
        self.assertEqual(accounting["genes_all_trials_lost"], 1)
        self.assertEqual(accounting["genes_some_trials_lost"], 1)

    def test_the_bound_is_one_sided_and_per_sampled_gene(self):
        """OOM cannot have created a MetaSUN hit, so the band only opens upwards."""
        relaxations = _relaxations([(g, 0, "failed", OOM) for g in range(25)])
        accounting = oom_accounting(relaxations, _structures({g: False for g in range(25)}))
        self.assertEqual(oom_metasun_bound(accounting, 1000), 0.025)
        # Denominator is the sampled cohort, not the trials.
        self.assertEqual(oom_metasun_bound(accounting, 500), 0.05)


if __name__ == "__main__":
    unittest.main()
