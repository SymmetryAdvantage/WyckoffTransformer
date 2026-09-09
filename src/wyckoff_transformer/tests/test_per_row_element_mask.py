"""Every row of a batch may be asked for a different chemical system.

`allowed_element_set` is one set for the whole batch, which is the right shape when
the caller names a system and the wrong one when the systems were drawn per
structure by `wyckoff_transformer.system_prior`: a batch-wide union would re-admit
the rest of the palette into every row and undo the sampling. The property tested
here is exactly that -- what a row may place is its own mask and nothing else --
plus the refusals that keep a malformed mask from silently producing a batch nobody
asked for.

The model is a stub emitting uniform element logits, so anything that reaches the
output does so because the mask let it.
"""
import unittest

import torch

from wyckoff_transformer.generator import WyckoffGenerator

CASCADE = ("elements", "site_symmetries", "sites_enumeration")
#: 0-5 are elements, 6 is STOP, 7 is MASK. Wide enough that a uniform draw would
#: leave the mask's complement in nearly every row if the mask were not applied.
VOCAB = 8
STOP = 6
MASK = 7
ELEMENTS_VOCAB = {"STOP": STOP}


class _UniformModel:
    """Uniform over the element vocabulary; a fixed real token for the other fields."""

    def eval(self):
        return self

    def __call__(self, start, cascade, padding_mask, prediction_head, cond=None):
        batch_size = start.size(0)
        if CASCADE[prediction_head] == "elements":
            return torch.zeros(batch_size, VOCAB)
        logits = torch.full((batch_size, VOCAB), -1e4)
        logits[:, 0] = 1e4
        return logits


def _generator():
    return WyckoffGenerator(
        model=_UniformModel(),
        cascade_order=CASCADE,
        cascade_is_target={field: True for field in CASCADE},
        token_engineers={},
        masks={field: MASK for field in CASCADE},
        max_sequence_len=6,
    )


def _mask(rows):
    """One boolean row per structure, from the element tokens it may place, plus STOP."""
    mask = torch.zeros(len(rows), VOCAB, dtype=torch.bool)
    for index, tokens in enumerate(rows):
        mask[index, list(tokens)] = True
        mask[index, STOP] = True
    return mask


class TestPerRowMask(unittest.TestCase):
    def setUp(self):
        self.rows = [(0, 1), (2,), (3, 4, 5), (0, 5)]
        self.mask = _mask(self.rows)
        self.start = torch.zeros(len(self.rows), dtype=torch.int64)

    def _elements(self, **kwargs):
        generated = _generator().generate_tensors(
            self.start, elements_vocab=ELEMENTS_VOCAB, allowed_element_mask=self.mask, **kwargs)
        return generated[CASCADE.index("elements")]

    def test_each_row_places_only_what_its_own_mask_permits(self):
        elements = self._elements()
        for index, tokens in enumerate(self.rows):
            placed = set(elements[index].tolist())
            self.assertTrue(placed <= set(tokens) | {STOP}, f"row {index} placed {placed}")

    def test_a_row_is_not_confined_to_what_the_other_rows_allow(self):
        # The intersection of the four systems is empty and their union is everything;
        # a batch-wide mask could only be one of the two, and neither is what was asked.
        elements = self._elements()
        self.assertTrue(set(elements[2].tolist()) & {3, 4, 5},
                        "the row asked for 3-4-5 placed none of them")

    def test_a_single_element_row_places_that_element(self):
        elements = self._elements()
        self.assertTrue(set(elements[1].tolist()) <= {2, STOP})

    def test_the_mask_activates_constrained_generation_without_a_required_set(self):
        # required_element_set is None here: masking alone is enough, and nothing is
        # forced in. That is the point -- forcing is what distorts the sample.
        elements = self._elements()
        self.assertEqual(elements.size(0), len(self.rows))

    def test_required_elements_still_force_when_asked(self):
        # Forcing and per-row masking compose, as long as every row's mask permits what
        # is being forced -- which is exactly what the refusal below checks.
        mask = _mask([(0, 1), (0, 2), (0, 3, 4)])
        elements = _generator().generate_tensors(
            torch.zeros(3, dtype=torch.int64), elements_vocab=ELEMENTS_VOCAB,
            allowed_element_mask=mask, required_element_set={0})[CASCADE.index("elements")]
        self.assertEqual(elements[:, 0].tolist(), [0, 0, 0])

    def test_a_required_element_outside_a_row_mask_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self._elements(required_element_set={3})
        self.assertIn("forbids a required element", str(caught.exception))

    def test_a_row_that_cannot_stop_is_refused(self):
        mask = self.mask.clone()
        mask[1, STOP] = False
        with self.assertRaises(ValueError) as caught:
            _generator().generate_tensors(
                self.start, elements_vocab=ELEMENTS_VOCAB, allowed_element_mask=mask)
        self.assertIn("STOP", str(caught.exception))

    def test_a_row_with_no_element_is_refused(self):
        mask = torch.zeros(len(self.rows), VOCAB, dtype=torch.bool)
        mask[:, STOP] = True
        with self.assertRaises(ValueError) as caught:
            _generator().generate_tensors(
                self.start, elements_vocab=ELEMENTS_VOCAB, allowed_element_mask=mask)
        self.assertIn("at least one element", str(caught.exception))

    def test_a_mask_with_the_wrong_number_of_rows_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            _generator().generate_tensors(
                self.start, elements_vocab=ELEMENTS_VOCAB, allowed_element_mask=self.mask[:2])
        self.assertIn("expected", str(caught.exception))

    def test_the_mask_and_the_set_are_not_passed_together(self):
        with self.assertRaises(ValueError) as caught:
            _generator().generate_tensors(
                self.start, elements_vocab=ELEMENTS_VOCAB, allowed_element_mask=self.mask,
                allowed_element_set={0, 1})
        self.assertIn("pass one", str(caught.exception))

    def test_a_mask_of_the_wrong_width_is_refused(self):
        narrow = torch.ones(len(self.rows), VOCAB - 2, dtype=torch.bool)
        with self.assertRaises(ValueError) as caught:
            _generator().generate_tensors(
                self.start, elements_vocab=ELEMENTS_VOCAB, allowed_element_mask=narrow)
        self.assertIn("wide", str(caught.exception))

    def test_a_non_boolean_mask_is_accepted(self):
        # A caller building the mask with arithmetic gets floats; the meaning is the same.
        elements = _generator().generate_tensors(
            self.start, elements_vocab=ELEMENTS_VOCAB,
            allowed_element_mask=self.mask.to(torch.float32))[CASCADE.index("elements")]
        for index, tokens in enumerate(self.rows):
            self.assertTrue(set(elements[index].tolist()) <= set(tokens) | {STOP})

    def test_the_batch_set_path_still_works(self):
        elements = _generator().generate_tensors(
            self.start, elements_vocab=ELEMENTS_VOCAB, required_element_set=set(),
            allowed_element_set={0, 1})[CASCADE.index("elements")]
        self.assertTrue(set(elements.reshape(-1).tolist()) <= {0, 1, STOP})


class TestDrawsProduceAUsableMask(unittest.TestCase):
    """The mask the sampler builds is the one the generator accepts, end to end."""

    def test_a_plan_masks_its_own_rows(self):
        from wyckoff_transformer.system_prior import SystemSpaceGroupPrior

        symbols = [f"E{index}" for index in range(VOCAB)]
        symbols[STOP] = "STOP"
        symbols[MASK] = "MASK"
        prior = SystemSpaceGroupPrior.from_rows(
            systems=[(0, 1), (0, 1), (2, 3), (0, 1, 4)],
            space_groups=[12, 225, 12, 2],
            element_symbols=symbols)
        draws = prior.sample(4, required=[0], allowed=[0, 1, 4], novel_fraction=0.0, rng=0)
        mask = draws.element_mask(VOCAB, stop_token=STOP)
        elements = _generator().generate_tensors(
            torch.zeros(len(draws), dtype=torch.int64),
            elements_vocab=ELEMENTS_VOCAB,
            allowed_element_mask=mask)[CASCADE.index("elements")]
        for row in range(len(draws)):
            self.assertTrue(
                set(elements[row].tolist()) <= set(draws.element_tokens[row]) | {STOP})


if __name__ == "__main__":
    unittest.main()
