"""`read_cif` drops pymatgen's coordinate-rounding note and nothing else."""
import unittest
import warnings
from unittest.mock import MagicMock, patch

from wyckoff_transformer import data


def _parser_that_warns(message: str, category=UserWarning):
    """Stand in for CifParser, warning the way pymatgen does while parsing."""
    def from_str(cif):
        warnings.warn(message, category)
        parsed = MagicMock()
        parsed.parse_structures.return_value = ["a structure"]
        return parsed

    parser = MagicMock()
    parser.from_str.side_effect = from_str
    return parser


class TestReadCif(unittest.TestCase):
    ROUNDING = ("Issues encountered while parsing CIF: 8 fractional coordinates "
                "rounded to ideal values to avoid issues with finite precision.")

    def read(self, message, category=UserWarning):
        with patch.object(data, "CifParser", _parser_that_warns(message, category)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                structure = data.read_cif("a cif")
        return structure, [str(entry.message) for entry in caught]

    def test_the_rounding_note_is_dropped(self):
        # It fires for most structures of a million-row dataset and says nothing
        # a caller can act on.
        structure, caught = self.read(self.ROUNDING)
        self.assertEqual(structure, "a structure")
        self.assertEqual(caught, [])

    def test_it_is_dropped_whatever_the_count_in_it(self):
        # The count varies, so Python's own per-message deduplication never
        # collapses these: to it they are different warnings.
        for count in (1, 7, 8, 132):
            with self.subTest(count=count):
                message = self.ROUNDING.replace("8 fractional", f"{count} fractional")
                self.assertEqual(self.read(message)[1], [])

    def test_every_other_warning_still_gets_through(self):
        _, caught = self.read("Issues encountered while parsing CIF: no structure found")
        self.assertEqual(caught, ["Issues encountered while parsing CIF: no structure found"])

    def test_a_warning_of_another_category_gets_through(self):
        _, caught = self.read(self.ROUNDING, DeprecationWarning)
        self.assertEqual(caught, [self.ROUNDING])

    def test_the_filter_does_not_outlive_the_call(self):
        # catch_warnings restores what was there; a caller's own filters, and
        # the ones a Pool's workers inherit, must be left as they were.
        before = list(warnings.filters)
        self.read(self.ROUNDING)
        self.assertEqual(warnings.filters, before)


if __name__ == "__main__":
    unittest.main()
