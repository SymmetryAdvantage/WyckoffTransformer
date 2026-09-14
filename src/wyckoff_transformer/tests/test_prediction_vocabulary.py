"""The prediction path must survive a JSON round-trip of the vocabulary.

A processor saved with ``save_pretrained`` writes its ``elements`` keys -- pymatgen
``Element`` in the training cache -- as plain symbols, so a record built by
``pyxtal_notation_to_sites``, which yields ``Element``, used to match nothing at
all in a reloaded vocabulary and every generated gene was dropped as
"outside the regressor vocabulary".
"""
import unittest

import pandas as pd
from omegaconf import OmegaConf
from pymatgen.core import Element

from wyckoff_transformer.prediction import filter_supported_tokens, tokeniser_key
from wyckoff_transformer.tokenization import EnumeratingTokeniser, SpaceGroupEncoder


def _elements_tokeniser(keys):
    return EnumeratingTokeniser.from_token_set(frozenset(keys))


class _Trainer:
    """The two attributes ``filter_supported_tokens`` reads."""

    def __init__(self, tokenisers):
        self.tokenisers = tokenisers
        self.tokeniser_config = OmegaConf.create({
            "dtype": "int16",
            "token_fields": {"pure_categorical": ["elements", "site_symmetries"]},
            "sequence_fields": {"space_group": ["spacegroup_number"]},
            "augmented_token_fields": [],
        })


def _trainer(element_keys):
    return _Trainer({
        "elements": _elements_tokeniser(element_keys),
        "site_symmetries": _elements_tokeniser(["1", "m"]),
        "spacegroup_number": SpaceGroupEncoder.from_sg_set(frozenset({1, 225})),
    })


_RECORD = pd.DataFrame.from_records([{
    "elements": [Element("Na"), Element("Cl")],
    "site_symmetries": ["1", "m"],
    "spacegroup_number": 225,
}])


class TestTokeniserKey(unittest.TestCase):
    def test_symbol_keys_accept_an_element(self):
        tokeniser = _elements_tokeniser(["Na", "Cl"])
        self.assertEqual(tokeniser_key(Element("Na"), tokeniser), "Na")

    def test_element_keys_accept_a_symbol(self):
        tokeniser = _elements_tokeniser([Element("Na"), Element("Cl")])
        self.assertEqual(tokeniser_key("Na", tokeniser), Element("Na"))

    def test_an_absent_token_has_no_key(self):
        self.assertIsNone(tokeniser_key("Fe", _elements_tokeniser(["Na", "Cl"])))


class TestFilterSupportedTokens(unittest.TestCase):
    def test_element_record_against_a_json_restored_symbol_vocabulary(self):
        supported, dropped = filter_supported_tokens(_RECORD, _trainer(["Na", "Cl"]))
        self.assertEqual(dropped, [])
        self.assertEqual(len(supported), 1)

    def test_element_record_against_a_cache_loaded_element_vocabulary(self):
        supported, dropped = filter_supported_tokens(
            _RECORD, _trainer([Element("Na"), Element("Cl")]))
        self.assertEqual(dropped, [])
        self.assertEqual(len(supported), 1)

    def test_a_genuinely_unknown_element_is_still_dropped(self):
        with self.assertRaisesRegex(ValueError, "All structures were dropped"):
            filter_supported_tokens(_RECORD, _trainer(["K", "Br"]))


if __name__ == "__main__":
    unittest.main()
