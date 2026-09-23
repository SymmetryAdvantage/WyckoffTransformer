"""Classifier-free guidance: condition dropout in training, guided logits in sampling.

What can go wrong silently, and is therefore pinned here:

- the null condition coinciding with a real one. AdaLN is affine in its input, so a zeroed
  log1p(e_hull) column *is* e_hull = 0 -- the target the protocol samples at. The null
  indicator column is what separates the two;
- a conditioned row reaching AdaLN differently from a model trained without dropout. The
  indicator is 0 on those rows for exactly that reason: a presence flag that is a constant
  1 on them duplicates the AdaLN bias, and destabilised run
  ehull_adamw_wsd_5x_cfg-20260916-015500 (see guidance_conditioning_width);
- dropout leaking out of training. Evaluation, calibration and generation all go through
  `build_cond`, and a validation loss computed with dropped conditions would not be
  comparable with a run trained without it;
- a model without dropout accepting a guidance scale, and producing a "guided" sample
  from an unconditional branch it never learned;
- the guidance formula itself, and every generation path forwarding it.
"""
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from wyckoff_transformer.cascade.dataset import TargetClass
from wyckoff_transformer.generator import WyckoffGenerator
from wyckoff_transformer.trainer import WyckoffTrainer, guidance_conditioning_width

FIXTURE_RUN = Path(__file__).resolve().parent / "fixtures" / "ioi8tycx"


def _skeleton(condition_dropout=0.1, condition_feature="energy_above_hull"):
    """A trainer carrying only what the conditioning path reads."""
    trainer = WyckoffTrainer.__new__(WyckoffTrainer)
    trainer.condition_feature = condition_feature
    trainer.condition_scale = None
    trainer.condition_transform = "log1p"
    trainer.condition_dropout = condition_dropout
    trainer.device = torch.device("cpu")
    return trainer


def _dataset(**columns):
    frame = MagicMock()
    frame.data = columns
    return frame


class TestWidth(unittest.TestCase):
    def test_no_dropout_adds_nothing(self):
        self.assertEqual(guidance_conditioning_width(0.0, 1), 0)
        self.assertEqual(guidance_conditioning_width(None, 1), 0)

    def test_dropout_adds_the_null_indicator_column(self):
        self.assertEqual(guidance_conditioning_width(0.1, 3), 1)

    def test_a_dropout_of_one_or_more_is_refused(self):
        for bad in (1.0, 1.5, -0.1):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
                guidance_conditioning_width(bad, 1)

    def test_dropout_without_a_condition_is_refused(self):
        with self.assertRaisesRegex(ValueError, "nothing conditions"):
            guidance_conditioning_width(0.1, 0)

    def test_condition_dim_is_one_wider(self):
        self.assertEqual(_skeleton(condition_dropout=0.0).condition_dim, 1)
        self.assertEqual(_skeleton(condition_dropout=0.1).condition_dim, 2)
        self.assertTrue(_skeleton().classifier_free_guidance)
        self.assertFalse(_skeleton(condition_dropout=0.0).classifier_free_guidance)


class TestBuildCond(unittest.TestCase):
    def setUp(self):
        self.values = torch.linspace(0.0, 2.0, 1000).unsqueeze(1)
        self.data = _dataset(energy_above_hull=self.values)

    def test_a_conditioned_row_is_the_baseline_row_plus_a_zero(self):
        guided = _skeleton().build_cond(self.data)
        plain = _skeleton(condition_dropout=0.0).build_cond(self.data)
        self.assertEqual(guided.shape, (1000, 2))
        self.assertTrue(torch.equal(guided[:, :1], plain))
        self.assertTrue(torch.equal(guided[:, 1], torch.zeros(1000)))

    def test_dropped_rows_are_exactly_the_null_condition(self):
        torch.manual_seed(0)
        trainer = _skeleton(condition_dropout=0.3)
        cond = trainer.build_cond(self.data, drop_condition=True)
        dropped = cond[:, 1] == 1
        # The indicator and the zeroed value go together: never a set indicator on a live value.
        self.assertTrue(torch.equal(cond[dropped], trainer.null_condition(int(dropped.sum()))))
        kept = ~dropped
        self.assertTrue(torch.equal(cond[kept, 1], torch.zeros(int(kept.sum()))))
        self.assertTrue(torch.allclose(cond[kept, 0], torch.log1p(self.values[kept, 0])))
        # 300 expected of 1000; the binomial sd is 14.5.
        self.assertAlmostEqual(dropped.float().mean().item(), 0.3, delta=0.06)

    def test_the_null_condition_differs_from_a_zero_target(self):
        # The reason the indicator column exists: e_hull = 0 transforms to 0.
        trainer = _skeleton()
        at_zero = trainer.build_cond(_dataset(energy_above_hull=torch.zeros(1, 1)))
        self.assertTrue(torch.equal(at_zero, torch.zeros(1, 2)))
        self.assertTrue(torch.equal(trainer.null_condition(1), torch.tensor([[0.0, 1.0]])))

    def test_without_drop_condition_nothing_is_dropped(self):
        cond = _skeleton(condition_dropout=0.9).build_cond(self.data)
        self.assertTrue(bool((cond[:, 1] == 0).all()))

    def test_unconditional_is_all_null(self):
        cond = _skeleton().build_cond(self.data, unconditional=True)
        self.assertTrue(torch.equal(cond, torch.tensor([[0.0, 1.0]]).repeat(1000, 1)))

    def test_a_model_without_dropout_is_unchanged_by_drop_condition(self):
        trainer = _skeleton(condition_dropout=0.0)
        torch.manual_seed(0)
        state = torch.get_rng_state()
        cond = trainer.build_cond(self.data, drop_condition=True)
        self.assertEqual(cond.shape, (1000, 1))
        self.assertTrue(torch.allclose(cond[:, 0], torch.log1p(self.values[:, 0])))
        # No RNG draw either: an ordinary conditional run trains exactly as it did.
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_a_model_without_dropout_has_no_unconditional_mode(self):
        with self.assertRaisesRegex(ValueError, "without condition_dropout"):
            _skeleton(condition_dropout=0.0).build_cond(self.data, unconditional=True)
        with self.assertRaisesRegex(ValueError, "without condition_dropout"):
            _skeleton(condition_dropout=0.0).null_condition(3)


class TestTrainingPaths(unittest.TestCase):
    """Dropout is asked for by train_epoch, and only there."""

    def _trainer(self):
        trainer = _skeleton()
        trainer.target = TargetClass.NextToken
        trainer.multiclass_next_token_with_order_permutation = True
        trainer.cascade_target_indices = (0,)
        trainer.cascade_target_count = 1
        trainer.clip_grad_norm = None
        trainer.evaluation_samples = 1
        trainer.scheduler_steps_per_batch = False
        trainer.model = torch.nn.Linear(1, 1)
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
        trainer.criterion = torch.nn.Identity()
        trainer.train_dataset = MagicMock()
        trainer.train_dataset.sample_known_seq_len.return_value = 0
        trainer.train_loader = MagicMock(batches_per_epoch=2)
        return trainer

    def test_train_epoch_drops_conditions(self):
        trainer = self._trainer()
        calls = []

        def fake_loss(*args, **kwargs):
            calls.append(kwargs)
            return trainer.model.weight.sum() * 0, 4

        with patch.object(WyckoffTrainer, "get_loss", side_effect=fake_loss), \
                patch("wyckoff_transformer.trainer.wandb"):
            trainer.train_epoch()
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call["drop_condition"] for call in calls))

    def test_evaluate_conditions_unless_asked_not_to(self):
        trainer = self._trainer()
        dataset = MagicMock(max_sequence_length=2)
        dataset.viable_count.return_value = 5
        dataset.__len__.return_value = 5
        for unconditional in (False, True):
            calls = []

            def fake_loss(*args, **kwargs):
                calls.append(kwargs)
                return torch.tensor(1.0)

            with self.subTest(unconditional=unconditional), \
                    patch.object(WyckoffTrainer, "get_loss", side_effect=fake_loss):
                trainer.evaluate(dataset, unconditional=unconditional)
                self.assertTrue(calls)
                self.assertTrue(all(call["unconditional"] is unconditional for call in calls))
                self.assertFalse(any(call.get("drop_condition") for call in calls))

    def test_train_epoch_drops_conditions_on_start(self):
        trainer = self._trainer()
        trainer.predict_start = True
        trainer.start_loss_weight = 1.0
        start_calls = []

        def fake_start_loss(*args, **kwargs):
            start_calls.append(kwargs)
            return trainer.model.weight.sum() * 0, 4

        with patch.object(WyckoffTrainer, "get_loss", side_effect=lambda *a, **k: (trainer.model.weight.sum() * 0, 4)), \
                patch.object(WyckoffTrainer, "get_start_loss", side_effect=fake_start_loss), \
                patch("wyckoff_transformer.trainer.wandb"):
            trainer.train_epoch()
        self.assertEqual(len(start_calls), 2)
        self.assertTrue(all(call.get("drop_condition") for call in start_calls))

    def test_evaluate_conditions_start_unless_asked_not_to(self):
        trainer = self._trainer()
        trainer.predict_start = True
        dataset = MagicMock(max_sequence_length=2)
        dataset.viable_count.return_value = 5
        dataset.__len__.return_value = 5
        for unconditional in (False, True):
            calls = []

            def fake_loss(*args, **kwargs):
                return torch.tensor(1.0)

            def fake_start_loss(*args, **kwargs):
                calls.append(kwargs)
                return torch.tensor(1.0), 5

            with self.subTest(unconditional=unconditional), \
                    patch.object(WyckoffTrainer, "get_loss", side_effect=fake_loss), \
                    patch.object(WyckoffTrainer, "get_start_loss", side_effect=fake_start_loss):
                trainer.evaluate(dataset, unconditional=unconditional)
                self.assertTrue(calls)
                self.assertTrue(all(call["unconditional"] is unconditional for call in calls))
                self.assertFalse(any(call.get("drop_condition") for call in calls))


class _CondModel:
    """Logits that depend on the null indicator only: conditional rows favour token 0.

    Records every call, so the tests can see how many forward passes a step costs.
    """

    CONDITIONAL = torch.tensor([1.0, 0.0, -1e4, -1e4])
    UNCONDITIONAL = torch.tensor([0.0, 0.5, -1e4, -1e4])

    def __init__(self):
        self.batch_sizes = []

    def eval(self):
        return self

    def __call__(self, start, cascade, padding_mask, prediction_head, cond=None):
        self.batch_sizes.append(start.size(0))
        null = cond[:, -1:]
        return null * self.UNCONDITIONAL + (1 - null) * self.CONDITIONAL

    def forward_start(self, batch_size, cond=None):
        self.batch_sizes.append(batch_size)
        null = cond[:, -1:]
        return null * self.UNCONDITIONAL + (1 - null) * self.CONDITIONAL


CASCADE = ("elements", "site_symmetries", "sites_enumeration")


def _generator(model):
    return WyckoffGenerator(
        model=model, cascade_order=CASCADE,
        cascade_is_target={field: True for field in CASCADE}, token_engineers={},
        masks={field: 2 for field in CASCADE}, max_sequence_len=1,
        stops={field: 3 for field in CASCADE})


class TestGuidedLogits(unittest.TestCase):
    def setUp(self):
        self.start = torch.zeros(3, dtype=torch.int64)
        self.cascade = [torch.zeros(3, 1, dtype=torch.int64)]
        self.cond = torch.zeros(3, 2)
        self.uncond = torch.tensor([[0.0, 1.0]]).repeat(3, 1)

    def _logits(self, scale):
        model = _CondModel()
        logits = _generator(model).guided_logits(
            self.start, self.cascade, 0, self.cond, self.uncond, scale)
        return logits, model

    def test_scale_one_is_the_conditional_model_in_one_pass(self):
        logits, model = self._logits(1.0)
        self.assertTrue(torch.equal(logits[0], _CondModel.CONDITIONAL))
        self.assertEqual(model.batch_sizes, [3])

    def test_scale_zero_is_the_unconditional_model(self):
        logits, _ = self._logits(0.0)
        self.assertTrue(torch.allclose(logits[0, :2], _CondModel.UNCONDITIONAL[:2]))

    def test_the_combination_extrapolates_in_one_doubled_pass(self):
        logits, model = self._logits(3.0)
        expected = _CondModel.UNCONDITIONAL + 3.0 * (
            _CondModel.CONDITIONAL - _CondModel.UNCONDITIONAL)
        self.assertTrue(torch.allclose(logits[:, :2], expected[:2].expand(3, 2)))
        self.assertEqual(model.batch_sizes, [6])

    def test_sampling_follows_the_guided_distribution(self):
        start = torch.zeros(20000, dtype=torch.int64)
        cond = torch.zeros(20000, 2)
        torch.manual_seed(0)
        generated = _generator(_CondModel()).generate_tensors(
            start, cond=cond, uncond=torch.tensor([[0.0, 1.0]]).repeat(20000, 1),
            guidance_scale=2.0)
        share = (generated[0][:, 0] == 0).float().mean().item()
        # Guided logits are [2.0, -0.5]: p(0) = 1 / (1 + e^-2.5) = 0.924, against 0.731
        # for the conditional model alone.
        self.assertAlmostEqual(share, 1 / (1 + math.exp(-2.5)), delta=0.01)

    def test_a_scale_other_than_one_needs_the_null_condition(self):
        with self.assertRaisesRegex(ValueError, "uncond"):
            _generator(_CondModel()).generate_tensors(
                self.start, cond=self.cond, guidance_scale=2.0)

    def test_a_negative_scale_is_refused(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _generator(_CondModel()).generate_tensors(
                self.start, cond=self.cond, uncond=self.uncond, guidance_scale=-1.0)


class TestGuidedStartLogits(unittest.TestCase):
    def setUp(self):
        self.batch_size = 3
        self.cond = torch.zeros(3, 2)
        self.uncond = torch.tensor([[0.0, 1.0]]).repeat(3, 1)

    def _logits(self, scale):
        model = _CondModel()
        logits = _generator(model).guided_start_logits(
            self.batch_size, self.cond, self.uncond, scale)
        return logits, model

    def test_scale_one_is_the_conditional_model_in_one_pass(self):
        logits, model = self._logits(1.0)
        self.assertTrue(torch.equal(logits[0], _CondModel.CONDITIONAL))
        self.assertEqual(model.batch_sizes, [3])

    def test_scale_zero_is_the_unconditional_model(self):
        logits, _ = self._logits(0.0)
        self.assertTrue(torch.allclose(logits[0, :2], _CondModel.UNCONDITIONAL[:2]))

    def test_the_combination_extrapolates_in_one_doubled_pass(self):
        logits, model = self._logits(3.0)
        expected = _CondModel.UNCONDITIONAL + 3.0 * (
            _CondModel.CONDITIONAL - _CondModel.UNCONDITIONAL)
        self.assertTrue(torch.allclose(logits[:, :2], expected[:2].expand(3, 2)))
        self.assertEqual(model.batch_sizes, [6])

    def test_sampling_follows_the_guided_distribution(self):
        n = 20000
        cond = torch.zeros(n, 2)
        torch.manual_seed(0)
        drawn = _generator(_CondModel()).sample_start_classes(
            n, cond=cond, uncond=torch.tensor([[0.0, 1.0]]).repeat(n, 1),
            guidance_scale=2.0)
        share = (drawn == 0).float().mean().item()
        self.assertAlmostEqual(share, 1 / (1 + math.exp(-2.5)), delta=0.01)

    def test_a_scale_other_than_one_needs_the_null_condition(self):
        with self.assertRaisesRegex(ValueError, "uncond"):
            _generator(_CondModel()).sample_start_classes(
                self.batch_size, cond=self.cond, guidance_scale=2.0)

    def test_a_negative_scale_is_refused(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _generator(_CondModel()).sample_start_classes(
                self.batch_size, cond=self.cond, uncond=self.uncond, guidance_scale=-1.0)


class _Recorded(Exception):
    pass


class _RecordingGenerator:
    calls: list = []
    start_calls: list = []

    def __init__(self, *args, **kwargs):
        pass

    def generate_tensors(self, *args, **kwargs):
        type(self).calls.append(kwargs)
        raise _Recorded

    def sample_start_classes(self, *args, **kwargs):
        type(self).start_calls.append((args, kwargs))
        return torch.zeros(args[0], dtype=torch.int64)


class TestGenerateStructuresForwardsGuidance(unittest.TestCase):
    def setUp(self):
        _RecordingGenerator.calls = []
        _RecordingGenerator.start_calls = []
        patcher = patch("wyckoff_transformer.trainer.WyckoffGenerator", _RecordingGenerator)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _trainer(self, condition_dropout):
        trainer = _skeleton(condition_dropout=condition_dropout)
        trainer.model = MagicMock()
        trainer.cascade_order = list(CASCADE)
        trainer.cascade_is_target = {field: True for field in CASCADE}
        trainer.token_engineers = {}
        trainer.masks_dict = {field: 2 for field in CASCADE}
        trainer.stops_dict = {field: 3 for field in CASCADE}
        trainer.max_sequence_length = 1
        trainer.tokenisers = {}
        trainer.train_dataset = None
        trainer._sample_start_tokens_from_distribution = (
            lambda n: torch.zeros(n, dtype=torch.int64))
        return trainer

    def _call(self, trainer, **kwargs):
        with self.assertRaises(_Recorded):
            trainer.generate_structures(n_structures=4, calibrate=False, **kwargs)
        return _RecordingGenerator.calls[-1]

    def test_the_caller_condition_gains_the_indicator_and_the_null_is_forwarded(self):
        call = self._call(self._trainer(0.1), cond=torch.full((4, 1), 0.1),
                          guidance_scale=2.5)
        self.assertEqual(call["guidance_scale"], 2.5)
        self.assertTrue(torch.allclose(
            call["cond"], torch.tensor([[math.log1p(0.1), 0.0]]).repeat(4, 1)))
        self.assertTrue(torch.equal(call["uncond"], torch.tensor([[0.0, 1.0]]).repeat(4, 1)))

    def test_every_path_forwards_it(self):
        trainer = self._trainer(0.1)
        call = self._call(trainer, cond=torch.zeros(4, 1), guidance_scale=2.0,
                          compute_validity_per_known_sequence_length=True)
        self.assertEqual(call["guidance_scale"], 2.0)
        trainer.tokenisers = {"elements": {}}
        call = self._call(trainer, cond=torch.zeros(4, 1), guidance_scale=2.0,
                          required_element_set=set())
        self.assertEqual(call["guidance_scale"], 2.0)
        self.assertIsNotNone(call["uncond"])

    def test_scale_one_skips_the_null_condition(self):
        call = self._call(self._trainer(0.1), cond=torch.zeros(4, 1))
        self.assertEqual(call["guidance_scale"], 1.0)
        self.assertIsNone(call["uncond"])
        # Still two columns: the conditional branch of a guided model always carries the
        # (clear) indicator.
        self.assertEqual(call["cond"].shape, (4, 2))

    def test_a_model_without_dropout_refuses_a_guidance_scale(self):
        trainer = self._trainer(0.0)
        with self.assertRaisesRegex(ValueError, "condition_dropout"):
            trainer.generate_structures(n_structures=4, calibrate=False,
                                        cond=torch.zeros(4, 1), guidance_scale=2.0)
        # And at 1 it behaves exactly as before: one column, no indicator.
        call = self._call(trainer, cond=torch.zeros(4, 1))
        self.assertEqual(call["cond"].shape, (4, 1))

    def test_predict_start_forwards_guidance_to_sample_start_classes(self):
        trainer = self._trainer(0.1)
        trainer.predict_start = True
        trainer.start_classes_to_tokens = lambda c: c
        call = self._call(trainer, cond=torch.full((4, 1), 0.1), guidance_scale=2.5)
        self.assertEqual(len(_RecordingGenerator.start_calls), 1)
        start_args, start_kwargs = _RecordingGenerator.start_calls[0]
        self.assertEqual(start_args[0], 4)
        self.assertEqual(start_kwargs["guidance_scale"], 2.5)
        self.assertTrue(torch.allclose(
            start_kwargs["cond"], torch.tensor([[math.log1p(0.1), 0.0]]).repeat(4, 1)))
        self.assertTrue(torch.equal(start_kwargs["uncond"], torch.tensor([[0.0, 1.0]]).repeat(4, 1)))


@pytest.mark.filterwarnings("ignore:No Pauling electronegativity for .*")
class TestRealModelEndToEnd(unittest.TestCase):
    """The doubled batch has to survive the real CascadeTransformer, token presence and all."""

    def setUp(self):
        if not FIXTURE_RUN.exists():
            self.skipTest(f"{FIXTURE_RUN} not found")
        config = OmegaConf.load(FIXTURE_RUN / "config.yaml")
        config.model.WyckoffTrainer_args.condition_feature = "energy_above_hull"
        config.model.WyckoffTrainer_args.condition_transform = "log1p"
        config.model.WyckoffTrainer_args.condition_dropout = 0.1
        torch.manual_seed(0)
        self.trainer = WyckoffTrainer.from_config(
            config, device=torch.device("cpu"), use_cached_tensors=False,
            run_path=FIXTURE_RUN, load_datasets=False)
        # Break the zero-initialised AdaLN so that the condition changes the logits.
        with torch.no_grad():
            for name, parameter in self.trainer.model.named_parameters():
                if "adaLN" in name:
                    parameter.normal_(std=0.5)

    def test_condition_dim_is_derived_with_the_null_indicator_column(self):
        self.assertEqual(self.trainer.model.condition_dim, 2)
        self.assertEqual(self.trainer.condition_dim, 2)

    def test_guided_generation_runs_every_position(self):
        # Randomised weights rarely produce a formally valid gene, so this is about the
        # plumbing: a full-length guided draw through the real model, decoded without error.
        torch.manual_seed(1)
        with patch.object(WyckoffGenerator, "guided_logits",
                          autospec=True, side_effect=WyckoffGenerator.guided_logits) as spy:
            structures = self.trainer.generate_structures(
                n_structures=16, calibrate=False, guidance_scale=3.0,
                cond=self.trainer.build_condition_from_values(
                    0.0, 16, device=torch.device("cpu")))
        self.assertIsInstance(structures, list)
        n_targets = sum(self.trainer.cascade_is_target.values())
        self.assertEqual(spy.call_count, self.trainer.max_sequence_length * n_targets)
        # (self, start, cascade, prediction_head, cond, uncond, guidance_scale)
        self.assertTrue(all(call.args[6] == 3.0 for call in spy.call_args_list))

    def test_guidance_changes_the_distribution(self):
        model = self.trainer.model
        model.eval()
        n = 8
        start = self.trainer._sample_start_tokens_from_distribution(n)
        cascade = [torch.full((n, 1), self.trainer.masks_dict[field], dtype=torch.int64)
                   for field in self.trainer.cascade_order]
        cond = self.trainer.with_null_indicator(torch.zeros(n, 1))
        uncond = self.trainer.null_condition(n, device=torch.device("cpu"))
        generator = WyckoffGenerator(
            model, self.trainer.cascade_order, self.trainer.cascade_is_target,
            self.trainer.token_engineers, self.trainer.masks_dict,
            self.trainer.max_sequence_length)
        with torch.no_grad():
            single = model(start, cascade, None, 0, cond=cond)
            at_one = generator.guided_logits(start, cascade, 0, cond, uncond, 1.0)
            at_three = generator.guided_logits(start, cascade, 0, cond, uncond, 3.0)
            unconditional = model(start, cascade, None, 0, cond=uncond)
        self.assertTrue(torch.allclose(single, at_one))
        # Row-wise batching must not change a row's logits.
        self.assertTrue(torch.allclose(
            at_three, unconditional + 3.0 * (single - unconditional), atol=1e-4))
        self.assertFalse(torch.allclose(single, unconditional, atol=1e-3))


class TestShippedGuidanceConfig(unittest.TestCase):
    """The CFG run must differ from its baseline in the dropout and the width, nothing else."""

    ROOT = Path(__file__).resolve().parents[3] / "yamls" / "models" / "lemat_bulk_ehull"

    def test_only_the_guidance_keys_differ(self):
        base_path = self.ROOT / "ehull_adamw_wsd_5x.yaml"
        cfg_path = self.ROOT / "ehull_adamw_wsd_5x_cfg.yaml"
        if not cfg_path.exists():
            self.skipTest(f"{cfg_path} not present")
        from wyckoff_transformer.trainer import flatten_config
        base = flatten_config(OmegaConf.to_container(OmegaConf.load(base_path)))
        cfg = flatten_config(OmegaConf.to_container(OmegaConf.load(cfg_path)))
        differing = {key for key in set(base) | set(cfg) if base.get(key) != cfg.get(key)}
        self.assertEqual(differing, {
            "model.WyckoffTrainer_args.condition_dropout",
            "model.CascadeTransformer_args.condition_dim"})
        self.assertEqual(cfg["model.CascadeTransformer_args.condition_dim"], 2)


if __name__ == "__main__":
    unittest.main()
