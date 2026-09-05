"""Tests for composition-constrained CSP decoding.

Two claims are worth defending.  `SpaceGroupCombinatorics` must never call a
reachable composition unreachable, because a false negative forbids a legal
structure; it is checked against exhaustive search on real space groups.  And
the decoder must emit only genes with the target formula, which is checked with
a stub model so that the masking, not the weights, is what is under test.
"""
import random
import unittest

import torch

from wyckoff_transformer.csp import (
    CompositionTarget,
    ConstrainedDecoder,
    CSPCandidate,
    SpaceGroupCombinatorics,
    WyckoffPosition,
    parse_formula,
)
from wyckoff_transformer.tokenization import get_wp_index

#: A spread of settings, symmetries and centrings.
SPACE_GROUPS = (1, 2, 12, 14, 62, 63, 123, 139, 166, 194, 216, 221, 225, 227, 229)

_WP_INDEX = None


def _wp_index():
    global _WP_INDEX
    if _WP_INDEX is None:
        _WP_INDEX = get_wp_index()
    return _WP_INDEX


def _combinatorics(sg_number, max_atoms=64):
    """Build the tables straight from pyxtal, with tokens standing in for themselves."""
    positions, token = [], 0
    for site_symmetry, letters in _wp_index()[sg_number].items():
        for letter, (multiplicity, dof) in letters.items():
            positions.append(WyckoffPosition(token, token, letter, multiplicity, dof))
            token += 1
    return SpaceGroupCombinatorics(sg_number, positions, max_atoms=max_atoms)


def _exhaustive(combinatorics, deficits, slots):
    """Ground truth: place positions one at a time under the once-only rule."""
    seen = set()

    def recurse(remaining, used, slots_left):
        if all(d == 0 for d in remaining):
            return True
        if slots_left <= 0:
            return False
        key = (remaining, used, slots_left)
        if key in seen:
            return False
        seen.add(key)
        for position in combinatorics.positions:
            handle = (position.ss_token, position.enum_token)
            if not position.reusable and handle in used:
                continue
            for index, owed in enumerate(remaining):
                if owed < position.multiplicity:
                    continue
                nxt = list(remaining)
                nxt[index] -= position.multiplicity
                extra = {handle} if not position.reusable else set()
                if recurse(tuple(nxt), used | extra, slots_left - 1):
                    return True
        return False

    return recurse(tuple(deficits), frozenset(), slots)


class TestFormulaParsing(unittest.TestCase):
    def test_parses_counts_and_implicit_ones(self):
        self.assertEqual(dict(parse_formula("BaTiO3")), {"Ba": 1, "Ti": 1, "O": 3})
        self.assertEqual(dict(parse_formula("Na2Cl2")), {"Na": 2, "Cl": 2})

    def test_repeated_element_accumulates(self):
        self.assertEqual(dict(parse_formula("CH3CH3")), {"C": 2, "H": 6})

    def test_rejects_junk(self):
        for bad in ("", "2Na", "Na!", "na"):
            with self.assertRaises(ValueError, msg=bad):
                parse_formula(bad)


class TestSpaceGroupCombinatorics(unittest.TestCase):
    def test_never_rejects_a_reachable_composition(self):
        """The property that matters: a False must be trustworthy."""
        random.seed(0)
        checked = false_negatives = 0
        for sg_number in SPACE_GROUPS:
            combinatorics = _combinatorics(sg_number)
            for _ in range(40):
                deficits = [random.choice([1, 2, 3, 4, 6, 8, 12, 16, 24])
                            for _ in range(random.randint(1, 3))]
                slots = random.randint(1, 6)
                got = combinatorics.can_complete(deficits, frozenset(), slots)
                if _exhaustive(combinatorics, deficits, slots) and not got:
                    false_negatives += 1
                checked += 1
        self.assertGreater(checked, 500)
        self.assertEqual(false_negatives, 0)

    def test_agrees_exactly_with_exhaustive_search(self):
        # Stronger than the above, and expected to hold: the node budget is not
        # reached at these sizes, so the optimistic fallback never fires.
        random.seed(1)
        for sg_number in SPACE_GROUPS:
            combinatorics = _combinatorics(sg_number)
            for _ in range(40):
                deficits = [random.choice([1, 2, 3, 4, 6, 8, 12])
                            for _ in range(random.randint(1, 3))]
                slots = random.randint(1, 5)
                self.assertEqual(
                    combinatorics.can_complete(deficits, frozenset(), slots),
                    _exhaustive(combinatorics, deficits, slots),
                    f"sg {sg_number}, deficits {deficits}, slots {slots}")

    def test_zero_deficit_is_complete(self):
        self.assertTrue(_combinatorics(225).can_complete([0, 0], frozenset(), 0))

    def test_negative_deficit_is_impossible(self):
        self.assertFalse(_combinatorics(225).can_complete([-1], frozenset(), 5))

    def test_odd_counts_are_unreachable_in_a_face_centred_group(self):
        # Every position of Fm-3m has multiplicity divisible by 4, so 1 and 2 atoms
        # cannot be placed however many sites are allowed.
        combinatorics = _combinatorics(225)
        self.assertFalse(combinatorics.can_complete([1], frozenset(), 20))
        self.assertFalse(combinatorics.can_complete([2], frozenset(), 20))
        self.assertTrue(combinatorics.can_complete([4], frozenset(), 20))

    def test_slot_budget_is_respected(self):
        # P1 has one position, of multiplicity 1, so n atoms need n sites.
        combinatorics = _combinatorics(1)
        self.assertTrue(combinatorics.can_complete([4], frozenset(), 4))
        self.assertFalse(combinatorics.can_complete([4], frozenset(), 3))

    def test_consuming_a_fixed_position_can_make_a_composition_impossible(self):
        """The scarce-resource half of the rule, which the relaxation cannot see."""
        combinatorics = _combinatorics(221)
        fixed_ones = [p for p in combinatorics.positions
                      if not p.reusable and p.multiplicity == 1]
        self.assertGreaterEqual(len(fixed_ones), 2, "Pm-3m should have 1a and 1b")
        target = [len(fixed_ones)]
        self.assertTrue(combinatorics.can_complete(target, frozenset(), 8))
        used = frozenset((p.ss_token, p.enum_token) for p in fixed_ones)
        self.assertFalse(combinatorics.can_complete(target, used, 8))

    def test_beyond_the_table_is_rejected_not_truncated(self):
        combinatorics = _combinatorics(225, max_atoms=32)
        self.assertFalse(combinatorics.can_complete([64], frozenset(), 20))


class _UniformModel(torch.nn.Module):
    """Stub backbone: uniform logits over each cascade field's vocabulary.

    Puts every legal choice on an equal footing, so what the decoder produces is
    decided by the composition mask alone.
    """

    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes
        self.calls = 0
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def forward(self, start, cascade, padding_mask, cascade_index, cond=None):
        self.calls += 1
        return torch.zeros(start.shape[0], self.sizes[cascade_index])


def _decoder_for(sg_number, max_sequence_len=8, max_atoms=64):
    """A decoder whose tables are injected, so no tokenisers are needed."""
    combinatorics = _combinatorics(sg_number, max_atoms=max_atoms)
    vocabulary = max(max(p.ss_token, p.enum_token) for p in combinatorics.positions) + 2
    cascade_order = ("elements", "site_symmetries", "sites_enumeration")
    decoder = ConstrainedDecoder(
        model=_UniformModel([vocabulary] * 3),
        cascade_order=cascade_order,
        cascade_is_target={name: True for name in cascade_order},
        tokenisers={"elements": None, "site_symmetries": None, "sites_enumeration": None},
        token_engineers={},
        masks={name: vocabulary - 1 for name in cascade_order},
        stops=None,
        max_sequence_len=max_sequence_len,
        device=torch.device("cpu"))
    decoder._combinatorics[sg_number] = combinatorics
    return decoder, combinatorics


def _target(counts, tokens):
    return CompositionTarget(
        symbols=tuple(f"E{i}" for i in range(len(counts))),
        counts=tuple(counts),
        element_tokens=tuple(tokens))


def _composition_of(candidate, combinatorics, target):
    """Atoms per element token in a decoded gene."""
    totals = {token: 0 for token in target.element_tokens}
    for element, ss, enum in candidate.rows:
        totals[element] += combinatorics.by_key[(ss, enum)].multiplicity
    return totals


class TestConstrainedDecoding(unittest.TestCase):
    def _check_all_on_target(self, sg_number, counts, **kwargs):
        decoder, combinatorics = _decoder_for(sg_number)
        # Element tokens must be distinct and inside the stub's vocabulary.
        target = _target(counts, tuple(range(len(counts))))
        candidates = decoder.decode(
            start=torch.zeros(1, dtype=torch.int64), sg_number=sg_number,
            target=target, **kwargs)
        self.assertGreater(len(candidates), 0, f"nothing decoded for {counts} in {sg_number}")
        for candidate in candidates:
            self.assertEqual(
                _composition_of(candidate, combinatorics, target),
                dict(zip(target.element_tokens, target.counts)))
        return candidates

    def test_sampling_always_hits_the_target_composition(self):
        torch.manual_seed(0)
        self._check_all_on_target(225, [4, 8], n_candidates=32, strategy="sample")

    def test_beam_always_hits_the_target_composition(self):
        torch.manual_seed(0)
        self._check_all_on_target(
            62, [4, 4, 12], n_candidates=16, strategy="beam", beam_width=16)

    def test_single_element_composition(self):
        torch.manual_seed(0)
        self._check_all_on_target(194, [4], n_candidates=8, strategy="sample")

    def test_low_symmetry_group_with_one_multiplicity(self):
        torch.manual_seed(0)
        candidates = self._check_all_on_target(1, [3], n_candidates=4, strategy="sample")
        # P1's only position is 1a, with dof, so three atoms means exactly three sites.
        for candidate in candidates:
            self.assertEqual(candidate.n_sites, 3)

    def test_fixed_positions_are_never_reused(self):
        torch.manual_seed(0)
        decoder, combinatorics = _decoder_for(221)
        target = _target([2, 6], (0, 1))
        for candidate in decoder.decode(
                start=torch.zeros(1, dtype=torch.int64), sg_number=221,
                target=target, n_candidates=48, strategy="sample"):
            occupied = [(ss, enum) for _, ss, enum in candidate.rows
                        if not combinatorics.by_key[(ss, enum)].reusable]
            self.assertEqual(len(occupied), len(set(occupied)),
                             "a position with no positional freedom was occupied twice")

    def test_impossible_composition_returns_nothing(self):
        # One atom cannot be placed in Fm-3m, whose smallest multiplicity is 4.
        decoder, _ = _decoder_for(225)
        self.assertEqual(
            decoder.decode(start=torch.zeros(1, dtype=torch.int64), sg_number=225,
                           target=_target([1], (0,)), n_candidates=8),
            [])

    def test_composition_too_large_for_the_site_budget_returns_nothing(self):
        decoder, _ = _decoder_for(1, max_sequence_len=3)
        self.assertEqual(
            decoder.decode(start=torch.zeros(1, dtype=torch.int64), sg_number=1,
                           target=_target([5], (0,)), n_candidates=4),
            [])

    def test_sampling_is_reproducible_and_diverse(self):
        decoder, _ = _decoder_for(62)
        target = _target([4, 8], (0, 1))
        def draw(seed):
            return decoder.decode(
                start=torch.zeros(1, dtype=torch.int64), sg_number=62, target=target,
                n_candidates=24, strategy="sample",
                generator=torch.Generator().manual_seed(seed))
        self.assertEqual([c.rows for c in draw(7)], [c.rows for c in draw(7)])
        # Under a uniform stub every legal gene is equally likely, so a sample of 24
        # should not collapse onto one gene.
        self.assertGreater(len({c.rows for c in draw(7)}), 1)

    def test_beam_returns_distinct_genes(self):
        decoder, _ = _decoder_for(62)
        candidates = decoder.decode(
            start=torch.zeros(1, dtype=torch.int64), sg_number=62,
            target=_target([4, 8], (0, 1)), n_candidates=8, strategy="beam", beam_width=8)
        self.assertEqual(len({c.rows for c in candidates}), len(candidates))

    def test_candidates_come_back_sorted_by_likelihood(self):
        decoder, _ = _decoder_for(62)
        candidates = decoder.decode(
            start=torch.zeros(1, dtype=torch.int64), sg_number=62,
            target=_target([4, 8], (0, 1)), n_candidates=16, strategy="beam", beam_width=16)
        self.assertEqual([c.log_prob for c in candidates],
                         sorted((c.log_prob for c in candidates), reverse=True))

    def test_rejects_unknown_strategy(self):
        decoder, _ = _decoder_for(225)
        with self.assertRaises(ValueError):
            decoder.decode(start=torch.zeros(1, dtype=torch.int64), sg_number=225,
                           target=_target([4], (0,)), strategy="greedy")

    def test_conditioning_vector_reaches_the_model(self):
        seen = []

        decoder, _ = _decoder_for(225)
        wrapped = decoder.model.forward

        def record(start, cascade, padding_mask, cascade_index, cond=None):
            seen.append(None if cond is None else cond.shape)
            return wrapped(start, cascade, padding_mask, cascade_index, cond=cond)

        decoder.model.forward = record
        decoder.decode(start=torch.zeros(1, dtype=torch.int64), sg_number=225,
                       target=_target([4], (0,)), n_candidates=4,
                       cond=torch.zeros(1, 1))
        self.assertTrue(seen)
        self.assertTrue(all(shape is not None and shape[0] == 4 for shape in seen))


class TestCSPCandidate(unittest.TestCase):
    def test_length_penalty_normalises_the_sum_over_sites(self):
        short = CSPCandidate(225, ((0, 0, 0),), log_prob=-4.0, n_sites=1)
        long = CSPCandidate(225, ((0, 0, 0),) * 4, log_prob=-8.0, n_sites=4)
        # Unnormalised, the four-site gene looks worse purely for having more terms.
        self.assertLess(long.log_prob, short.log_prob)
        self.assertGreater(long.normalised_log_prob(), short.normalised_log_prob())

    def test_zero_penalty_leaves_the_raw_sum(self):
        candidate = CSPCandidate(225, ((0, 0, 0),) * 3, log_prob=-6.0, n_sites=3)
        self.assertEqual(candidate.normalised_log_prob(0), -6.0)


if __name__ == "__main__":
    unittest.main()
