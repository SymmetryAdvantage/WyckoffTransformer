"""The pymatgen warnings a bulk build would otherwise repeat until it drowns."""
import os
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


class TestNoisyPymatgenWarnings(unittest.TestCase):
    PAULING = ("No Pauling electronegativity for Ne. Setting to NaN. This has no "
               "physical meaning, and is mainly done to avoid errors caused by the "
               "code expecting a float.")

    def caught(self, *messages):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data.filter_noisy_pymatgen_warnings()
            for message in messages:
                warnings.warn(message, UserWarning)
        return [str(entry.message) for entry in caught]

    def test_the_pauling_note_is_dropped(self):
        # Element.X is a cached_property, so it fires once per element per
        # process -- which is once per Pool worker, two dozen times over.
        self.assertEqual(self.caught(self.PAULING), [])

    def test_it_is_dropped_for_every_element(self):
        for element in ("He", "Ne", "Ar"):
            with self.subTest(element=element):
                message = self.PAULING.replace("Ne", element)
                self.assertEqual(self.caught(message), [])

    def test_every_other_warning_still_gets_through(self):
        self.assertEqual(self.caught("Something worth reading"), ["Something worth reading"])

    def test_it_is_scoped_by_the_caller(self):
        # A library must put the filters back; only a command owns its process.
        before = list(warnings.filters)
        with warnings.catch_warnings():
            data.filter_noisy_pymatgen_warnings()
        self.assertEqual(warnings.filters, before)


class TestSpglibWarnings(unittest.TestCase):
    """spglib prints from C, so this is an environment switch, not a filter."""

    def test_it_asks_spglib_to_be_quiet(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SPGLIB_WARNING", None)
            data.silence_spglib_warnings()
            self.assertEqual(os.environ["SPGLIB_WARNING"], "OFF")

    def test_the_value_is_the_one_spglib_compares_against(self):
        # spglib uses strcmp, so "off" and "0" would leave the messages on.
        self.assertEqual(data.SPGLIB_WARNINGS_OFF, "OFF")

    def test_a_caller_who_wants_them_keeps_them(self):
        with patch.dict(os.environ, {"SPGLIB_WARNING": "ON"}):
            data.silence_spglib_warnings()
            self.assertEqual(os.environ["SPGLIB_WARNING"], "ON")


if __name__ == "__main__":
    unittest.main()
