"""Tests for the crystallographic and chemical pairwise relational attention biases.

The properties worth pinning down are the ones the design rests on: the bias is symmetric
in the pair, it moves with a permutation of the sites rather than staying put, it leaves
the start token alone, and it is identically zero at initialisation, so a run with it
starts from the same model as a run without.
"""

import unittest

import numpy as np
import torch

from wyckoff_transformer.cascade.model import CascadeTransformer
from wyckoff_transformer.cascade.relational import (
    CHEM_PAIR_FEATURES,
    RelationalAttentionBias,
    build_element_pair_features,
)
from wyckoff_transformer.tokenization import EnumeratingTokeniser

N_HEAD = 4
N_START = 11


def _element_tokeniser():
    # A slice of the real vocabulary: two alkali metals, two halogens, a transition metal,
    # an oxide former, and a noble gas, which has neither a Pauling electronegativity nor
    # a Slater radius and so exercises both fallbacks.
    return EnumeratingTokeniser.from_token_set({"Cs", "K", "F", "Cl", "Fe", "O", "He"})


def _site_symmetry_tokeniser():
    return EnumeratingTokeniser.from_token_set({"1", "2", "m", "4/mmm"})


def _bias_module(**kwargs):
    elements = _element_tokeniser()
    features = torch.from_numpy(build_element_pair_features(elements))
    return RelationalAttentionBias(
        nhead=N_HEAD,
        num_elements=len(elements),
        num_site_symmetries=len(_site_symmetry_tokeniser()),
        element_pair_features=features,
        n_start=N_START,
        start_type="one_hot",
        **kwargs)


class TestElementPairFeatures(unittest.TestCase):
    def setUp(self):
        self.tokeniser = _element_tokeniser()
        self.features = build_element_pair_features(self.tokeniser)

    def test_shape_and_finiteness(self):
        n = len(self.tokeniser)
        self.assertEqual(self.features.shape, (n, n, len(CHEM_PAIR_FEATURES)))
        self.assertTrue(np.isfinite(self.features).all())

    def test_symmetric(self):
        np.testing.assert_allclose(self.features, self.features.transpose(1, 0, 2))

    def test_service_tokens_are_zero(self):
        for name in ("MASK", "STOP", "PAD"):
            index = self.tokeniser[name]
            np.testing.assert_array_equal(self.features[index], 0.)
            np.testing.assert_array_equal(self.features[:, index], 0.)

    def test_noble_gas_is_filled_not_nan(self):
        # He has no Pauling electronegativity and no `atomic_radius`; the pair with itself
        # must still be a finite, standardised row rather than NaN.
        helium = self.tokeniser["He"]
        self.assertTrue(np.isfinite(self.features[helium, helium]).all())

    def test_chemically_ordered(self):
        # The electronegativity channel is a difference, so the most ionic pair in this
        # vocabulary must outrank the pair of two alkali metals.
        channel = CHEM_PAIR_FEATURES.index("abs_electronegativity_difference")
        cs_f = self.features[self.tokeniser["Cs"], self.tokeniser["F"], channel]
        cs_k = self.features[self.tokeniser["Cs"], self.tokeniser["K"], channel]
        self.assertGreater(cs_f, cs_k)

    def test_rejects_vocabulary_without_elements(self):
        with self.assertRaises(ValueError):
            build_element_pair_features({"MASK": 0, "STOP": 1, "PAD": 2})


class TestRelationalAttentionBias(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.module = _bias_module()
        self.batch, self.sites = 3, 5
        self.elements = torch.randint(0, 7, (self.batch, self.sites))
        self.site_symmetries = torch.randint(0, 4, (self.batch, self.sites))
        self.start = torch.randn(self.batch, N_START)

    def _excite(self):
        """Move every zero-initialised output off zero, so the tests see a real bias."""
        with torch.no_grad():
            self.module.elem_interaction.weight.normal_()
            self.module.chem_mlp[-1].weight.normal_()
            self.module.chem_mlp[-1].bias.normal_()
            self.module.ss_pair_bias.normal_()
            self.module.sg_gate.weight.normal_(std=0.1)

    def _bias(self, elements=None, site_symmetries=None):
        flat = self.module(
            self.elements if elements is None else elements,
            self.site_symmetries if site_symmetries is None else site_symmetries,
            self.start)
        return flat.view(self.batch, N_HEAD, self.sites + 1, self.sites + 1)

    def test_shape_is_multihead_attn_mask(self):
        flat = self.module(self.elements, self.site_symmetries, self.start)
        self.assertEqual(
            flat.shape, (self.batch * N_HEAD, self.sites + 1, self.sites + 1))

    def test_zero_at_initialisation(self):
        self.assertTrue(bool((self._bias() == 0).all()))

    def test_start_token_takes_no_bias(self):
        self._excite()
        bias = self._bias()
        self.assertTrue(bool((bias[:, :, 0, :] == 0).all()))
        self.assertTrue(bool((bias[:, :, :, 0] == 0).all()))
        self.assertTrue(bool((bias[:, :, 1:, 1:] != 0).any()))

    def test_symmetric_in_the_pair(self):
        self._excite()
        sites = self._bias()[:, :, 1:, 1:]
        torch.testing.assert_close(sites, sites.transpose(-1, -2))

    def test_permutation_equivariant(self):
        self._excite()
        reference = self._bias()[:, :, 1:, 1:]
        permutation = torch.randperm(self.sites)
        permuted = self._bias(
            self.elements[:, permutation], self.site_symmetries[:, permutation])[:, :, 1:, 1:]
        torch.testing.assert_close(permuted, reference[:, :, permutation][:, :, :, permutation])

    def test_depends_on_the_space_group(self):
        self._excite()
        reference = self._bias()
        self.start = torch.randn(self.batch, N_START)
        self.assertFalse(torch.allclose(self._bias(), reference))

    def test_space_group_gate_can_be_disabled(self):
        module = _bias_module(space_group_gate=False)
        self.assertIsNone(module.sg_gate)
        with torch.no_grad():
            module.ss_pair_bias.normal_()
        first = module(self.elements, self.site_symmetries, self.start)
        second = module(self.elements, self.site_symmetries, torch.randn(self.batch, N_START))
        torch.testing.assert_close(first, second)

    def test_rejects_non_categorial_tokens(self):
        with self.assertRaises(ValueError):
            self.module(self.elements.unsqueeze(-1).float(), self.site_symmetries, self.start)

    def test_features_survive_a_state_dict_round_trip(self):
        # The table is a buffer precisely so that generation cannot disagree with training
        # about the physics; check it actually travels.
        state = self.module.state_dict()
        self.assertIn("element_pair_features", state)
        fresh = _bias_module()
        fresh.load_state_dict(state)
        torch.testing.assert_close(fresh.element_pair_features, self.module.element_pair_features)


class TestCascadeTransformerIntegration(unittest.TestCase):
    """The bias is wired in as the encoder's float `attn_mask`; check the model still runs."""

    @staticmethod
    def _model(with_bias: bool, condition_dim=None):
        elements = _element_tokeniser()
        site_symmetries = _site_symmetry_tokeniser()
        relational = None
        if with_bias:
            relational = {
                "tokenisers": {"elements": elements, "site_symmetries": site_symmetries},
                "cascade_order": ["elements", "site_symmetries"],
            }
        return CascadeTransformer(
            start_type="one_hot",
            n_start=N_START,
            cascade=((len(elements), 8, elements.pad_token, True),
                     (len(site_symmetries), 8, site_symmetries.pad_token, True)),
            token_aggregation=None,
            aggregate_after_encoder=False,
            include_start_in_aggregation=False,
            aggregation_inclsion="None",
            concat_token_counts=False,
            concat_token_presence=False,
            num_fully_connected_layers=1,
            mixer_layers=1,
            outputs="token_scores",
            perceptron_shape="input",
            TransformerEncoderLayer_args={"nhead": N_HEAD, "dim_feedforward": 16, "dropout": 0.0},
            TransformerEncoder_args={"num_layers": 2, "enable_nested_tensor": False},
            learned_positional_encoding_max_size=0,
            learned_positional_encoding_only_masked=True,
            condition_dim=condition_dim,
            relational_attention_bias=relational)

    @staticmethod
    def _batch(batch=3, sites=4):
        start = torch.randn(batch, N_START)
        cascade = [torch.randint(0, 7, (batch, sites)), torch.randint(0, 4, (batch, sites))]
        return start, cascade

    def test_forward_shape(self):
        torch.manual_seed(0)
        model = self._model(with_bias=True)
        start, cascade = self._batch()
        out = model(start, cascade, padding_mask=None, prediction_head=0)
        self.assertEqual(out.shape, (3, len(_element_tokeniser())))

    def test_identical_to_the_baseline_at_initialisation(self):
        # Zero-init makes the bias exactly absent at step 0, which is what makes a run with
        # it an ablation of the same starting model rather than a different experiment.
        torch.manual_seed(0)
        with_bias = self._model(with_bias=True)
        torch.manual_seed(0)
        without = self._model(with_bias=False)
        without.load_state_dict(
            {k: v for k, v in with_bias.state_dict().items() if not k.startswith("relational_bias.")})
        start, cascade = self._batch()
        torch.testing.assert_close(
            with_bias(start, cascade, padding_mask=None, prediction_head=0),
            without(start, cascade, padding_mask=None, prediction_head=0))

    def test_forward_with_padding_mask(self):
        torch.manual_seed(0)
        model = self._model(with_bias=True)
        start, cascade = self._batch(batch=3, sites=4)
        padding_mask = torch.zeros(3, 5, dtype=torch.bool)
        padding_mask[:, -1] = True
        out = model(start, cascade, padding_mask=padding_mask, prediction_head=0)
        self.assertTrue(torch.isfinite(out).all())

    def test_forward_with_an_empty_batch(self):
        # `WyckoffGenerator.calibrate` sweeps every sequence length, so splits that hold no
        # structure of that length reach the model as a batch of zero.
        torch.manual_seed(0)
        model = self._model(with_bias=True)
        start, cascade = self._batch(batch=0, sites=4)
        out = model(start, cascade, padding_mask=None, prediction_head=0)
        self.assertEqual(out.shape, (0, len(_element_tokeniser())))

    def test_forward_with_adaln_conditioning(self):
        torch.manual_seed(0)
        model = self._model(with_bias=True, condition_dim=2)
        start, cascade = self._batch()
        out = model(start, cascade, padding_mask=None, prediction_head=0,
                    cond=torch.randn(3, 2))
        self.assertEqual(out.shape, (3, len(_element_tokeniser())))

    def test_unknown_cascade_field_is_rejected(self):
        elements = _element_tokeniser()
        with self.assertRaises(ValueError):
            CascadeTransformer(
                start_type="one_hot", n_start=N_START,
                cascade=((len(elements), 8, elements.pad_token, True),),
                token_aggregation=None, aggregate_after_encoder=False,
                include_start_in_aggregation=False, aggregation_inclsion="None",
                concat_token_counts=False, concat_token_presence=False,
                num_fully_connected_layers=1, mixer_layers=1, outputs="token_scores",
                perceptron_shape="input",
                TransformerEncoderLayer_args={"nhead": N_HEAD, "dim_feedforward": 16},
                TransformerEncoder_args={"num_layers": 1, "enable_nested_tensor": False},
                learned_positional_encoding_max_size=0,
                learned_positional_encoding_only_masked=True,
                relational_attention_bias={
                    "tokenisers": {"elements": elements},
                    "cascade_order": ["elements"]})


if __name__ == "__main__":
    unittest.main()
