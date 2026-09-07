"""Conditioning on several scalars at once.

The AdaLN machinery was always width-agnostic -- the modulation is a
`nn.Linear(condition_dim, 2 * d_model)` and there has been a passing `condition_dim=2`
layer test for as long as it has existed. What assumed a single channel was everything
above it: one feature name, one transform applied to the whole tensor, a literal `1` in
the width, and CLIs that filled every column of the conditioning vector with the same
number. These tests pin the plural behaviour and, just as importantly, the column order,
which is what every learned AdaLN weight is tied to and the one thing that can go wrong
silently.
"""
import math
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch
from omegaconf import OmegaConf

from wyckoff_transformer.cli import (
    describe_condition, parse_condition_assignments, resolve_condition_values)
from wyckoff_transformer.trainer import (
    WyckoffTrainer,
    get_condition_transform,
    normalise_condition_features,
    normalise_condition_scales,
    normalise_condition_transforms,
)

FEATURES = ("energy_above_hull", "delta_e_polymorph", "max_force")


def trainer(condition_feature=FEATURES, condition_transform="log1p", condition_scale=None,
            composition_conditioning=False, n_elements=None, condition_on_cell_size=True):
    """A skeleton carrying only what the conditioning code path reads."""
    skeleton = WyckoffTrainer.__new__(WyckoffTrainer)
    skeleton.condition_feature = condition_feature
    skeleton.condition_scale = condition_scale
    skeleton.condition_transform = condition_transform
    skeleton.composition_conditioning = composition_conditioning
    skeleton.n_elements = n_elements
    skeleton.condition_on_cell_size = condition_on_cell_size
    skeleton.device = torch.device("cpu")
    return skeleton


def dataset(**columns):
    frame = MagicMock()
    frame.data = columns
    return frame


class TestNormalisation(unittest.TestCase):
    def test_a_single_name_becomes_a_one_tuple(self):
        self.assertEqual(normalise_condition_features("energy_above_hull"),
                         ("energy_above_hull",))

    def test_none_becomes_empty(self):
        self.assertEqual(normalise_condition_features(None), ())

    def test_a_list_keeps_its_order(self):
        # Not sorted, not deduplicated into a set: the order is the column order.
        self.assertEqual(normalise_condition_features(["b", "a", "c"]), ("b", "a", "c"))

    def test_an_omegaconf_list_is_accepted(self):
        listed = OmegaConf.create({"f": list(FEATURES)}).f
        self.assertEqual(normalise_condition_features(listed), FEATURES)

    def test_a_repeated_feature_is_refused(self):
        with self.assertRaisesRegex(ValueError, "repeats"):
            normalise_condition_features(["a", "b", "a"])

    def test_one_transform_name_applies_to_every_feature(self):
        self.assertEqual(normalise_condition_transforms("log1p", FEATURES),
                         ("log1p", "log1p", "log1p"))

    def test_a_transform_list_is_positional(self):
        self.assertEqual(normalise_condition_transforms(["log1p", None, "log1p"], FEATURES),
                         ("log1p", None, "log1p"))

    def test_a_transform_mapping_is_by_name_and_defaults_to_none(self):
        self.assertEqual(
            normalise_condition_transforms({"max_force": "log1p"}, FEATURES),
            (None, None, "log1p"))

    def test_a_transform_list_of_the_wrong_length_is_refused(self):
        with self.assertRaisesRegex(ValueError, "one per feature"):
            normalise_condition_transforms(["log1p"], FEATURES)

    def test_a_transform_mapping_naming_an_unknown_feature_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not conditioning features"):
            normalise_condition_transforms({"band_gap": "log1p"}, FEATURES)

    def test_scales_default_to_one(self):
        self.assertEqual(normalise_condition_scales(None, FEATURES), (1.0, 1.0, 1.0))

    def test_a_scale_mapping_leaves_the_others_at_one(self):
        self.assertEqual(normalise_condition_scales({"max_force": 0.01}, FEATURES),
                         (1.0, 1.0, 0.01))

    def test_a_non_positive_scale_is_refused(self):
        # Zero would divide by zero and a negative would flip the sign into log1p's
        # undefined half, both of which are much easier to catch here than in a loss curve.
        for bad in (0.0, -1.0):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "positive"):
                normalise_condition_scales({"max_force": bad}, FEATURES)


class TestWidthAndOrder(unittest.TestCase):
    def test_condition_dim_counts_the_features(self):
        self.assertEqual(trainer().condition_dim, 3)

    def test_condition_dim_is_none_when_unconditional(self):
        self.assertIsNone(trainer(condition_feature=None, condition_transform=None).condition_dim)

    def test_the_composition_block_is_added_after_the_scalars(self):
        wide = trainer(composition_conditioning=True, n_elements=5)
        self.assertEqual(wide.condition_dim, 3 + 5 + 1)

    def test_build_cond_uses_the_configured_order_not_the_dataset_order(self):
        # The dict is deliberately in a different order from `condition_feature`.
        data = dataset(max_force=torch.tensor([[0.5]]),
                       energy_above_hull=torch.tensor([[1.0]]),
                       delta_e_polymorph=torch.tensor([[3.0]]))
        cond = trainer(condition_transform=None).build_cond(data)
        self.assertEqual(cond.shape, (1, 3))
        self.assertEqual(cond[0].tolist(), [1.0, 3.0, 0.5])

    def test_build_cond_transforms_each_column(self):
        data = dataset(energy_above_hull=torch.tensor([[1.0]]),
                       delta_e_polymorph=torch.tensor([[3.0]]),
                       max_force=torch.tensor([[0.5]]))
        cond = trainer(condition_transform={"energy_above_hull": "log1p"}).build_cond(data)
        self.assertAlmostEqual(cond[0, 0].item(), math.log1p(1.0), places=6)
        self.assertAlmostEqual(cond[0, 1].item(), 3.0, places=6)
        self.assertAlmostEqual(cond[0, 2].item(), 0.5, places=6)

    def test_condition_dim_equals_what_build_cond_produces(self):
        data = dataset(energy_above_hull=torch.zeros(4, 1),
                       delta_e_polymorph=torch.zeros(4, 1),
                       max_force=torch.zeros(4, 1))
        built = trainer().build_cond(data)
        self.assertEqual(built.shape[-1], trainer().condition_dim)


class TestTransformAndScale(unittest.TestCase):
    def test_the_uniform_fast_path_matches_the_per_column_path(self):
        # Same transform and scale everywhere takes a single-call branch; it must agree
        # with the column-by-column one, which is the only thing exercised when they differ.
        values = torch.tensor([[0.0, 0.1, 0.5], [1.0, 2.0, 0.25]])
        uniform = trainer(condition_transform="log1p")
        mixed = trainer(condition_transform=["log1p", "log1p", "log1p"])
        self.assertTrue(uniform._condition_is_uniform)
        self.assertTrue(torch.allclose(
            uniform.transform_condition(values), mixed.transform_condition(values)))
        self.assertTrue(torch.allclose(uniform.transform_condition(values),
                                       torch.log1p(values)))

    def test_a_scale_divides_before_the_transform(self):
        scaled = trainer(condition_scale={"max_force": 0.01})
        out = scaled.transform_condition(torch.tensor([[0.0, 0.0, 0.5]]))
        self.assertAlmostEqual(out[0, 2].item(), math.log1p(50.0), places=5)

    def test_zero_stays_zero_under_every_channel(self):
        # Generating at "on the hull, ground state, fully converged" must land on exactly
        # the origin of the conditioning space, not near it.
        out = trainer(condition_scale={"max_force": 0.01}).transform_condition(torch.zeros(2, 3))
        self.assertTrue(torch.equal(out, torch.zeros(2, 3)))

    def test_the_callers_tensor_is_not_modified(self):
        raw = torch.tensor([[0.1, 0.2, 0.3]])
        trainer(condition_scale={"max_force": 0.01}).transform_condition(raw)
        self.assertAlmostEqual(raw[0, 2].item(), 0.3, places=6)

    def test_identity_returns_the_same_tensor(self):
        plain = trainer(condition_transform=None)
        values = torch.rand(3, 3)
        self.assertIs(plain.transform_condition(values), values)


class TestValidation(unittest.TestCase):
    def test_a_negative_value_names_the_offending_feature(self):
        with self.assertRaisesRegex(ValueError, "delta_e_polymorph"):
            trainer()._validate_condition_values(torch.tensor([[0.0, -1.0, 0.0]]))

    def test_a_negative_value_in_an_untransformed_column_is_allowed(self):
        # log1p is what rejects negatives, so a channel without it may go negative.
        trainer(condition_transform={"energy_above_hull": "log1p"})._validate_condition_values(
            torch.tensor([[0.0, -1.0, 0.0]]))

    def test_the_wrong_width_is_refused_rather_than_broadcast(self):
        with self.assertRaisesRegex(ValueError, "1 wide"):
            trainer()._validate_condition_values(torch.zeros(4, 1))


class TestBuildConditionFromValues(unittest.TestCase):
    def test_values_land_in_the_configured_order(self):
        built = trainer().build_condition_from_values(
            {"max_force": 0.5, "energy_above_hull": 0.1, "delta_e_polymorph": 0.2}, 3)
        self.assertEqual(built.shape, (3, 3))
        self.assertEqual([round(v, 4) for v in built[0].tolist()], [0.1, 0.2, 0.5])
        self.assertTrue(torch.equal(built[0], built[2]))

    def test_a_bare_number_works_for_a_single_feature_model(self):
        single = trainer(condition_feature="energy_above_hull")
        self.assertEqual(single.build_condition_from_values(0.25, 2).tolist(),
                         [[0.25], [0.25]])

    def test_a_bare_number_is_refused_for_several_features(self):
        # This is the bug the old CLIs had: one number silently filled every column.
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            trainer().build_condition_from_values(0.0, 4)

    def test_a_missing_feature_is_refused(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            trainer().build_condition_from_values({"energy_above_hull": 0.0}, 2)

    def test_an_unknown_feature_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            trainer().build_condition_from_values(
                dict.fromkeys(FEATURES, 0.0) | {"band_gap": 1.0}, 2)

    def test_the_result_is_in_physical_units(self):
        # generate_structures transforms what it is handed, so this must not be pre-scaled.
        built = trainer(condition_scale={"max_force": 0.01}).build_condition_from_values(
            dict.fromkeys(FEATURES, 0.5), 1)
        self.assertAlmostEqual(built[0, 2].item(), 0.5, places=6)


class TestBackwardCompatibility(unittest.TestCase):
    def test_a_single_string_round_trips_as_a_string(self):
        single = trainer(condition_feature="energy_above_hull")
        self.assertEqual(single.condition_feature, "energy_above_hull")
        self.assertEqual(single.condition_features, ("energy_above_hull",))

    def test_several_features_report_as_a_tuple(self):
        self.assertEqual(trainer().condition_feature, FEATURES)

    def test_unconditional_reports_none(self):
        plain = trainer(condition_feature=None, condition_transform=None)
        self.assertIsNone(plain.condition_feature)
        self.assertEqual(plain.condition_features, ())

    def test_the_setters_are_order_independent(self):
        first = WyckoffTrainer.__new__(WyckoffTrainer)
        first.condition_feature = FEATURES
        first.condition_transform = "log1p"
        second = WyckoffTrainer.__new__(WyckoffTrainer)
        second.condition_transform = "log1p"
        second.condition_feature = FEATURES
        self.assertEqual(first.condition_transforms, second.condition_transforms)
        self.assertEqual(first.condition_scales, second.condition_scales)


class TestModelRoundTrip(unittest.TestCase):
    """A three-wide condition has to survive the AdaLN stack, not just the trainer."""

    def test_three_channels_reach_every_encoder_layer(self):
        from wyckoff_transformer.cascade.model import AdaLNTransformerEncoderLayer
        layer = AdaLNTransformerEncoderLayer(
            d_model=16, nhead=4, condition_dim=3, batch_first=True, norm_first=False)
        source = torch.randn(4, 6, 16)
        out = layer(source, cond=torch.randn(4, 3))
        self.assertEqual(out.shape, source.shape)

    def test_the_layer_is_the_identity_at_initialisation(self):
        # The modulation is zero-initialised, which is what lets an unnormalised
        # conditioning input be safe at step zero.
        from wyckoff_transformer.cascade.model import AdaLNTransformerEncoderLayer
        layer = AdaLNTransformerEncoderLayer(
            d_model=16, nhead=4, condition_dim=3, batch_first=True, norm_first=False)
        # eval(), or dropout makes two forward passes differ for reasons unrelated to cond.
        layer.eval()
        source = torch.randn(2, 3, 16)
        cond = torch.randn(2, 3)
        self.assertTrue(torch.allclose(
            layer(source, cond=cond), layer(source, cond=cond * 1000.0), atol=1e-5))

    def test_a_condition_that_differs_changes_the_output_once_trained(self):
        from wyckoff_transformer.cascade.model import AdaLNTransformerEncoderLayer
        torch.manual_seed(0)
        layer = AdaLNTransformerEncoderLayer(
            d_model=16, nhead=4, condition_dim=3, batch_first=True, norm_first=False)
        layer.eval()
        # Break the zero initialisation the way a single optimiser step would.
        with torch.no_grad():
            layer.adaLN_modulation1.weight.normal_(std=0.1)
        source = torch.randn(2, 3, 16)
        a = layer(source, cond=torch.zeros(2, 3))
        b = layer(source, cond=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]))
        self.assertFalse(torch.allclose(a, b, atol=1e-4))


class TestCliArguments(unittest.TestCase):
    def test_assignments_parse_to_a_mapping(self):
        self.assertEqual(
            parse_condition_assignments(["energy_above_hull=0", "max_force=0.05"]),
            {"energy_above_hull": 0.0, "max_force": 0.05})

    def test_a_repeated_name_is_refused(self):
        with self.assertRaisesRegex(ValueError, "more than once"):
            parse_condition_assignments(["a=1", "a=2"])

    def test_the_legacy_single_value_still_works_for_one_feature(self):
        single = trainer(condition_feature="energy_above_hull")
        self.assertEqual(resolve_condition_values(single, None, 0.0),
                         {"energy_above_hull": 0.0})

    def test_the_legacy_single_value_is_refused_for_several_features(self):
        with self.assertRaisesRegex(ValueError, "once per feature"):
            resolve_condition_values(trainer(), None, 0.0)

    def test_the_two_forms_cannot_be_mixed(self):
        with self.assertRaisesRegex(ValueError, "alternatives"):
            resolve_condition_values(trainer(), ["max_force=0"], 0.0)

    def test_nothing_given_means_sample_from_training(self):
        self.assertIsNone(resolve_condition_values(trainer(), None, None))
        self.assertEqual(describe_condition(None), "sampled from the training distribution")


if __name__ == "__main__":
    unittest.main()


class TestSingleChannelCondition(unittest.TestCase):
    """Sweeping a target moves one variable, not all of them."""

    def test_the_target_lands_on_its_own_channel_and_the_rest_stay_at_baseline(self):
        from wyckoff_transformer.cli import single_channel_condition
        built = single_channel_condition(trainer(), 0.2, 3)
        self.assertEqual(built.shape, (3, 3))
        self.assertEqual([round(v, 4) for v in built[0].tolist()], [0.2, 0.0, 0.0])

    def test_another_channel_can_be_swept_by_name(self):
        from wyckoff_transformer.cli import single_channel_condition
        built = single_channel_condition(trainer(), 0.2, 1, feature="max_force")
        self.assertEqual([round(v, 4) for v in built[0].tolist()], [0.0, 0.0, 0.2])

    def test_a_single_feature_model_is_unaffected(self):
        from wyckoff_transformer.cli import single_channel_condition
        single = trainer(condition_feature="energy_above_hull")
        built = single_channel_condition(single, 0.2, 2)
        self.assertEqual(built.shape, (2, 1))
        self.assertEqual([[round(v, 4) for v in row] for row in built.tolist()],
                         [[0.2], [0.2]])

    def test_an_unknown_channel_name_falls_back_to_the_first(self):
        from wyckoff_transformer.cli import single_channel_condition
        other = trainer(condition_feature=["band_gap", "max_force"])
        built = single_channel_condition(other, 1.5, 1)
        self.assertEqual([round(v, 4) for v in built[0].tolist()], [1.5, 0.0])

    def test_an_unconditional_model_gets_none(self):
        from wyckoff_transformer.cli import single_channel_condition
        plain = trainer(condition_feature=None, condition_transform=None)
        self.assertIsNone(single_channel_condition(plain, 0.2, 4))


class TestShippedConfigs(unittest.TestCase):
    """Every model yaml has to declare a conditioning width that matches its channels.

    `from_config` derives the width and refuses a config that disagrees, so a mismatch is
    caught at startup rather than inside `nn.Linear`. This checks the same thing without a
    dataset, so adding a fourth channel and forgetting `condition_dim` fails in a second
    rather than after a model has been built.
    """

    YAMLS = sorted((Path(__file__).resolve().parents[3] / "yamls" / "models").rglob("*.yaml"))

    def test_condition_dim_matches_the_channel_count(self):
        self.assertTrue(self.YAMLS, "no model yamls found")
        for path in self.YAMLS:
            config = OmegaConf.load(path)
            trainer_args = config.get("model", {}).get("WyckoffTrainer_args")
            model_args = config.get("model", {}).get("CascadeTransformer_args")
            if trainer_args is None or model_args is None:
                continue
            # The composition block's width depends on the element vocabulary, which needs a
            # cache to size; those configs are covered by from_config instead.
            if trainer_args.get("composition_conditioning", False):
                continue
            features = normalise_condition_features(trainer_args.get("condition_feature"))
            declared = model_args.get("condition_dim")
            with self.subTest(config=path.name):
                if features:
                    self.assertEqual(
                        declared, len(features),
                        f"{path} conditions on {list(features)} but declares "
                        f"condition_dim={declared}")
                else:
                    self.assertIsNone(
                        declared,
                        f"{path} declares condition_dim={declared} but has no conditioning")

    def test_transforms_and_scales_resolve(self):
        for path in self.YAMLS:
            config = OmegaConf.load(path)
            trainer_args = config.get("model", {}).get("WyckoffTrainer_args")
            if trainer_args is None:
                continue
            features = normalise_condition_features(trainer_args.get("condition_feature"))
            with self.subTest(config=path.name):
                transforms = normalise_condition_transforms(
                    trainer_args.get("condition_transform"), features)
                scales = normalise_condition_scales(trainer_args.get("condition_scale"), features)
                self.assertEqual(len(transforms), len(features))
                self.assertEqual(len(scales), len(features))
                for name in transforms:
                    get_condition_transform(name)


class TestThreeChannelConfigIsBalanced(unittest.TestCase):
    """The scale that run wjwmgjag diverged on must not come back.

    0.01 was chosen by matching the channels' medians, which for a variable that is exactly
    zero on a quarter of rows and has a long right tail equalises the middle and widens the
    end: it gave max_force the largest spread of the three, and AdaLN puts no bound on the
    modulation the tail of its input produces.
    """

    CONFIG = (Path(__file__).resolve().parents[3] / "yamls" / "models" / "lemat_bulk_ehull"
              / "e_all_adamw_wsd.yaml")

    def test_max_force_is_not_scaled_back_to_the_diverging_value(self):
        if not self.CONFIG.exists():
            self.skipTest(f"{self.CONFIG} is not present")
        args = OmegaConf.load(self.CONFIG).model.WyckoffTrainer_args
        features = normalise_condition_features(args.get("condition_feature"))
        scales = dict(zip(features, normalise_condition_scales(args.get("condition_scale"),
                                                               features)))
        self.assertIn("max_force", scales)
        self.assertGreaterEqual(
            scales["max_force"], 0.02,
            "max_force is scaled at least as aggressively as the 0.01 that diverged")
