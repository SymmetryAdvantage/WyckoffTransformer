"""STOP tokens must not be scored as invalid Wyckoff positions.

Regression test for the defect that made the logged `ss_validity` / `enumeration_validity`
curves read as the stop rate. Two things went wrong together:

- `WyckoffTrainer.generate_structures` built `WyckoffGenerator` without `stops`, so
  `stop_generated` stayed all-False and the skip for finished structures never fired.
- The validity check is a membership test against `token_engineers["multiplicity"].db`,
  whose keys are real (space group, site symmetry, enumeration) triples. STOP is not among
  them, so every sequence that legitimately ended was counted as invalid at that position
  and at every position after it.

The visible effect was validity falling to ~0.5 past three sites on runs whose real per-site
validity is above 0.99.
"""
import unittest
from types import SimpleNamespace

import torch

from wyckoff_transformer.generator import WyckoffGenerator

CASCADE = ("elements", "site_symmetries", "sites_enumeration")
VOCAB = 4
STOP = 3
MASK = 2
# The one real site the stub model knows how to emit, and the only key in the stub table.
REAL = {"elements": 0, "site_symmetries": 1, "sites_enumeration": 0}
START = 0


class _StubTable:
    """Stands in for `token_engineers["multiplicity"].db`: real triples only, no STOP."""

    def __init__(self):
        self._keys = {
            (START, REAL["site_symmetries"]),
            (START, REAL["site_symmetries"], REAL["sites_enumeration"]),
        }

    def __contains__(self, key):
        return key in self._keys


class _StubModel:
    """Emits `length[b]` real sites for structure b, then STOP in every field."""

    def __init__(self, lengths):
        self.lengths = lengths

    def eval(self):
        return self

    def __call__(self, start, cascade, padding_mask, prediction_head, cond=None):
        known_seq_len = cascade[0].size(1) - 1
        field = CASCADE[prediction_head]
        logits = torch.zeros(start.size(0), VOCAB)
        for b in range(start.size(0)):
            token = REAL[field] if known_seq_len < self.lengths[b] else STOP
            logits[b, token] = 1e4
        return logits


def _generator(lengths, stops):
    return WyckoffGenerator(
        model=_StubModel(lengths),
        cascade_order=CASCADE,
        cascade_is_target={f: True for f in CASCADE},
        token_engineers={"multiplicity": SimpleNamespace(db=_StubTable())},
        masks={f: MASK for f in CASCADE},
        max_sequence_len=4,
        stops=stops,
    )


class TestValidityStopHandling(unittest.TestCase):
    def test_stop_is_not_a_key_of_the_multiplicity_table(self):
        """The premise of the bug: a STOP site symmetry looks invalid to the membership test."""
        table = _StubTable()
        self.assertIn((START, REAL["site_symmetries"]), table)
        self.assertNotIn((START, STOP), table)

    def test_stopped_structures_are_excluded_from_the_average(self):
        # Structure 0 places one real site then stops; structure 1 places two.
        lengths = [1, 2]
        start = torch.zeros(2, dtype=torch.int64)
        gen = _generator(lengths, stops={f: STOP for f in CASCADE})
        _, ss_validity, enum_validity = gen.generate_tensors(
            start, compute_validity=True, max_length=4)

        # Position 0: both structures place a real, valid site.
        # Position 1: structure 0 stops and drops out, structure 1 places a valid site.
        # Position 2 onwards: nothing is live, so no value is recorded at all.
        self.assertEqual(ss_validity, [1.0, 1.0])
        self.assertEqual(enum_validity, [1.0, 1.0])

    def test_without_the_fix_the_stop_rate_would_be_reported(self):
        """Scoring every structure at every position reproduces the broken curve."""
        lengths = [1, 2]
        start = torch.zeros(2, dtype=torch.int64)
        gen = _generator(lengths, stops={f: STOP for f in CASCADE})
        generated = gen.generate_tensors(start, compute_validity=False, max_length=4)
        table = _StubTable()
        ss_index = CASCADE.index("site_symmetries")
        unguarded = [
            sum((START, generated[ss_index][b, k].item()) in table for b in range(2)) / 2
            for k in range(3)
        ]
        # One structure stops at position 1 and both have stopped by position 2, which is
        # exactly what the old metric reported as validity.
        self.assertEqual(unguarded, [1.0, 0.5, 0.0])

    def test_compute_validity_requires_stops(self):
        gen = _generator([1, 2], stops=None)
        with self.assertRaises(ValueError) as caught:
            gen.generate_tensors(torch.zeros(2, dtype=torch.int64), compute_validity=True)
        self.assertIn("stops", str(caught.exception))

    def test_missing_stop_token_for_a_field_is_tolerated(self):
        """Tokenisers built with include_stop=False have stop_token None."""
        stops = {f: STOP for f in CASCADE}
        stops["elements"] = None
        gen = _generator([1, 2], stops=stops)
        _, ss_validity, _ = gen.generate_tensors(
            torch.zeros(2, dtype=torch.int64), compute_validity=True, max_length=4)
        self.assertEqual(ss_validity, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
