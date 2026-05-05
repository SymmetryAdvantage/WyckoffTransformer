"""Tests for AdaLN conditioning: model layers, dataset extra_fields, calibrate cond
slicing, and generate_structures cond handling."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from wyckoff_transformer.cascade.dataset import AugmentedCascadeDataset, TargetClass
from wyckoff_transformer.cascade.model import (
    AdaLNTransformerEncoder,
    AdaLNTransformerEncoderLayer,
    CascadeTransformer,
)
from wyckoff_transformer.generator import WyckoffGenerator
from wyckoff_transformer.trainer import WyckoffTrainer


def _make_minimal_dataset(extra_fields=None, energy_values=None, energy_dtype=torch.float64):
    """Build a tiny AugmentedCascadeDataset suitable for unit tests."""
    data = {
        "field1": torch.tensor([[0, 1, 2], [1, 0, 2], [2, 1, 0]], dtype=torch.int64),
        "spacegroup": torch.tensor([5, 7, 9], dtype=torch.int64),
    }
    if energy_values is None:
        energy_values = [1.0, 2.0, 3.0]
    data["energy"] = torch.tensor(energy_values, dtype=energy_dtype)
    return AugmentedCascadeDataset(
        data=data,
        cascade_order=("field1",),
        masks={"field1": 99},
        pads={"field1": 100},
        stops={"field1": 101},
        num_classes={"field1": 3},
        start_field="spacegroup",
        augmented_fields=None,
        extra_fields=extra_fields,
    )


class TestAdaLNTransformerEncoderLayer(unittest.TestCase):
    def test_inherits_from_transformer_encoder_layer(self):
        layer = AdaLNTransformerEncoderLayer(d_model=16, nhead=4, condition_dim=1, batch_first=True)
        self.assertIsInstance(layer, TransformerEncoderLayer)

    def test_inherits_parent_submodules(self):
        layer = AdaLNTransformerEncoderLayer(d_model=16, nhead=4, condition_dim=1, batch_first=True)
        for name in ("self_attn", "linear1", "linear2", "dropout", "dropout1", "dropout2", "activation"):
            self.assertTrue(hasattr(layer, name), f"missing inherited attribute {name}")

    def test_layernorms_have_no_affine(self):
        layer = AdaLNTransformerEncoderLayer(d_model=16, nhead=4, condition_dim=1, batch_first=True)
        # AdaLN modulation provides the affine; the LayerNorms must not double up.
        self.assertFalse(layer.norm1.elementwise_affine)
        self.assertFalse(layer.norm2.elementwise_affine)

    def test_modulation_zero_init(self):
        layer = AdaLNTransformerEncoderLayer(d_model=16, nhead=4, condition_dim=1, batch_first=True)
        for m in (layer.adaLN_modulation1, layer.adaLN_modulation2):
            self.assertEqual(m.weight.abs().max().item(), 0.0)
            self.assertEqual(m.bias.abs().max().item(), 0.0)

    def test_forward_shape_post_norm(self):
        torch.manual_seed(0)
        layer = AdaLNTransformerEncoderLayer(
            d_model=16, nhead=4, condition_dim=2, batch_first=True, norm_first=False)
        src = torch.randn(4, 6, 16)
        cond = torch.randn(4, 2)
        out = layer(src, cond=cond)
        self.assertEqual(out.shape, src.shape)

    def test_forward_shape_pre_norm(self):
        torch.manual_seed(0)
        layer = AdaLNTransformerEncoderLayer(
            d_model=16, nhead=4, condition_dim=2, batch_first=True, norm_first=True)
        src = torch.randn(4, 6, 16)
        cond = torch.randn(4, 2)
        out = layer(src, cond=cond)
        self.assertEqual(out.shape, src.shape)

    def test_cond_changes_output(self):
        torch.manual_seed(0)
        layer = AdaLNTransformerEncoderLayer(
            d_model=16, nhead=4, condition_dim=1, batch_first=True, dropout=0.0)
        # Unblock zero-init so cond actually matters; seed for determinism.
        torch.nn.init.normal_(layer.adaLN_modulation1.weight, std=0.1)
        torch.nn.init.normal_(layer.adaLN_modulation2.weight, std=0.1)
        src = torch.randn(2, 3, 16)
        out_a = layer(src, cond=torch.zeros(2, 1))
        out_b = layer(src, cond=torch.ones(2, 1))
        self.assertFalse(torch.allclose(out_a, out_b))


class TestAdaLNTransformerEncoder(unittest.TestCase):
    def test_inherits_from_transformer_encoder(self):
        layer = AdaLNTransformerEncoderLayer(d_model=16, nhead=4, condition_dim=1, batch_first=True)
        encoder = AdaLNTransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.assertIsInstance(encoder, TransformerEncoder)

    def test_layers_are_independent_clones(self):
        layer = AdaLNTransformerEncoderLayer(d_model=16, nhead=4, condition_dim=1, batch_first=True)
        encoder = AdaLNTransformerEncoder(layer, num_layers=3, enable_nested_tensor=False)
        ids = {id(l) for l in encoder.layers}
        self.assertEqual(len(ids), 3)

    def test_threads_cond_through_all_layers(self):
        torch.manual_seed(0)
        layer = AdaLNTransformerEncoderLayer(d_model=16, nhead=4, condition_dim=1, batch_first=True)
        encoder = AdaLNTransformerEncoder(layer, num_layers=3, enable_nested_tensor=False)

        called_with_cond = []
        for sublayer in encoder.layers:
            sublayer.forward = MagicMock(side_effect=lambda src, **kw: (called_with_cond.append(kw.get("cond")), src)[1])

        src = torch.randn(2, 5, 16)
        cond = torch.randn(2, 1)
        encoder(src, cond=cond)
        self.assertEqual(len(called_with_cond), 3)
        for received in called_with_cond:
            self.assertIs(received, cond)


class TestCascadeTransformerWithConditionDim(unittest.TestCase):
    @staticmethod
    def _build_model(condition_dim):
        return CascadeTransformer(
            start_type="categorial",
            n_start=4,
            cascade=((3, 4, 0, True),),
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
            TransformerEncoderLayer_args={"nhead": 2, "dim_feedforward": 16, "dropout": 0.0},
            TransformerEncoder_args={"num_layers": 1, "enable_nested_tensor": False},
            learned_positional_encoding_max_size=0,
            learned_positional_encoding_only_masked=True,
            condition_dim=condition_dim,
        )

    def test_forward_with_cond_returns_correct_shape(self):
        torch.manual_seed(0)
        model = self._build_model(condition_dim=1)
        start = torch.tensor([0, 1, 2], dtype=torch.int64)
        cascade = [torch.tensor([[0, 1], [1, 0], [2, 0]], dtype=torch.int64)]
        # Match the model's parameter dtype: torch.set_default_dtype may have been
        # changed by another test (e.g. MACE calculator init flips it to float64).
        cond = torch.tensor([[0.5], [-0.5], [1.5]], dtype=next(model.parameters()).dtype)
        out = model(start, cascade, padding_mask=None, prediction_head=0, cond=cond)
        # Output is the prediction head applied to the last token; shape [batch_size, num_classes]
        self.assertEqual(out.shape, (3, 3))

    def test_forward_without_cond_raises(self):
        model = self._build_model(condition_dim=1)
        start = torch.tensor([0, 1, 2], dtype=torch.int64)
        cascade = [torch.tensor([[0, 1], [1, 0], [2, 0]], dtype=torch.int64)]
        with self.assertRaises(ValueError):
            model(start, cascade, padding_mask=None, prediction_head=0, cond=None)

    def test_no_condition_dim_does_not_require_cond(self):
        torch.manual_seed(0)
        model = self._build_model(condition_dim=None)
        start = torch.tensor([0, 1, 2], dtype=torch.int64)
        cascade = [torch.tensor([[0, 1], [1, 0], [2, 0]], dtype=torch.int64)]
        out = model(start, cascade, padding_mask=None, prediction_head=0)
        self.assertEqual(out.shape, (3, 3))


class TestAugmentedCascadeDatasetExtraFields(unittest.TestCase):
    def test_no_extra_fields_keeps_only_cascade(self):
        ds = _make_minimal_dataset(extra_fields=None)
        self.assertNotIn("energy", ds.data)
        self.assertNotIn("spacegroup", ds.data)
        self.assertIn("field1", ds.data)

    def test_extra_fields_are_added(self):
        ds = _make_minimal_dataset(extra_fields=["energy"])
        self.assertIn("energy", ds.data)
        # Should still preserve source dtype (trainer pre-casts to float32 separately).
        self.assertEqual(ds.data["energy"].dtype, torch.float64)
        self.assertEqual(ds.data["energy"].shape, (3,))

    def test_extra_fields_put_on_device(self):
        ds = _make_minimal_dataset(extra_fields=["energy"])
        self.assertEqual(ds.data["energy"].device.type, "cpu")

    def test_unlisted_keys_are_not_silently_absorbed(self):
        # Adding a field to `data` without listing it in extra_fields must not
        # leak into self.data, otherwise random columns (formation_energy,
        # band_gap, etc.) would pollute the dataset.
        data = {
            "field1": torch.tensor([[0, 1, 2], [1, 0, 2]], dtype=torch.int64),
            "spacegroup": torch.tensor([5, 7], dtype=torch.int64),
            "energy": torch.tensor([1.0, 2.0]),
            "leakage_field": torch.tensor([10.0, 20.0]),
        }
        ds = AugmentedCascadeDataset(
            data=data,
            cascade_order=("field1",),
            masks={"field1": 99}, pads={"field1": 100}, stops={"field1": 101},
            num_classes={"field1": 3},
            start_field="spacegroup",
            augmented_fields=None,
            extra_fields=["energy"],
        )
        self.assertNotIn("leakage_field", ds.data)


class TestWyckoffGeneratorCalibrateConditioning(unittest.TestCase):
    def test_cond_is_sliced_per_iteration(self):
        """When `pure_sequences_lengths` filter excludes rows, cond must be sliced
        to match the kept rows so the model isn't called with mismatched batch sizes."""
        torch.manual_seed(0)
        model = MagicMock(spec=torch.nn.Module)
        param = torch.nn.Parameter(torch.ones(1))
        model.parameters.side_effect = lambda: iter([param])

        # Capture cond batch sizes the model sees.
        observed_cond_sizes = []

        def model_fn(start, data, padding, cascade_idx, cond=None):
            observed_cond_sizes.append(cond.shape[0])
            # Sanity: cond batch must match start batch
            assert cond.shape[0] == start.shape[0], "cond/start batch mismatch"
            return torch.randn(start.shape[0], 4)

        model.side_effect = model_fn

        full_n = 200
        # Two iterations: first keeps everything, second keeps only half.
        chosen_full = torch.ones(full_n, dtype=torch.bool)
        chosen_half = torch.zeros(full_n, dtype=torch.bool)
        chosen_half[:full_n // 2] = True

        iterations = [chosen_full, chosen_half]

        def get_masked(*_args, return_chosen_indices=False, **_kwargs):
            chosen = iterations.pop(0)
            n = int(chosen.sum().item())
            start_tokens = torch.zeros(n, dtype=torch.long)
            masked_data = [torch.zeros(n, 1, dtype=torch.long)]
            target = torch.zeros(n, dtype=torch.long)
            return (start_tokens, masked_data, target, chosen)

        full_cond = torch.randn(full_n, 1)
        dataset = MagicMock(spec=AugmentedCascadeDataset)
        dataset.cascade_order = ("field1",)
        dataset.max_sequence_length = 2  # 2 known-seq-len iterations
        dataset.data = {"energy": full_cond}
        dataset.get_masked_multiclass_cascade_data.side_effect = get_masked

        generator = WyckoffGenerator(
            model=model,
            cascade_order=("field1",),
            cascade_is_target={"field1": True},
            token_engineers={},
            masks={"field1": 0},
            max_sequence_len=2,
        )
        generator.calibrate(
            dataset, calibration_element_count_threshold=1,
            condition_feature="energy")

        self.assertEqual(observed_cond_sizes, [full_n, full_n // 2])

    def test_no_condition_feature_passes_none(self):
        """When `condition_feature` is None, the model receives `cond=None`."""
        model = MagicMock(spec=torch.nn.Module)
        param = torch.nn.Parameter(torch.ones(1))
        model.parameters.side_effect = lambda: iter([param])

        seen = []

        def model_fn(start, data, padding, cascade_idx, cond=None):
            seen.append(cond)
            return torch.randn(start.shape[0], 4)

        model.side_effect = model_fn

        n = 110
        chosen = torch.ones(n, dtype=torch.bool)
        dataset = MagicMock(spec=AugmentedCascadeDataset)
        dataset.cascade_order = ("field1",)
        dataset.max_sequence_length = 1
        dataset.get_masked_multiclass_cascade_data.return_value = (
            torch.zeros(n, dtype=torch.long),
            [torch.zeros(n, 1, dtype=torch.long)],
            torch.zeros(n, dtype=torch.long),
            chosen,
        )

        generator = WyckoffGenerator(
            model=model,
            cascade_order=("field1",),
            cascade_is_target={"field1": True},
            token_engineers={},
            masks={"field1": 0},
            max_sequence_len=1,
        )
        generator.calibrate(dataset, calibration_element_count_threshold=1)
        self.assertTrue(seen)
        for c in seen:
            self.assertIsNone(c)


class TestGenerateStructuresConditioning(unittest.TestCase):
    def _make_trainer_skeleton(self, condition_feature, train_dataset=None):
        trainer = WyckoffTrainer.__new__(WyckoffTrainer)
        trainer.model = MagicMock()
        trainer.model.start_type = "categorial"
        trainer.model.start_embedding.num_embeddings = 5
        trainer.cascade_order = ["spacegroup", "harmonic_site_symmetries"]
        trainer.cascade_is_target = {"spacegroup": True}
        trainer.token_engineers = {}
        trainer.tokenisers = {"elements": MagicMock()}
        trainer.masks_dict = {}
        trainer.start_name = "spacegroup_number"
        trainer.start_token_distribution = None
        trainer.train_dataset = train_dataset
        trainer.val_dataset = MagicMock()
        trainer.val_dataset.start_tokens = torch.tensor([1, 2])
        trainer.max_sequence_length = 10
        trainer.device = torch.device("cpu")
        trainer.processor = MagicMock()
        trainer.processor.tensor_to_pyxtal.return_value = "pyxtal_mock"
        trainer.production_training = False
        trainer.run_path = None
        trainer.condition_feature = condition_feature
        return trainer

    def test_missing_train_dataset_raises_clear_error(self):
        """generate_structures with a condition_feature but no train_dataset must
        raise a clear ValueError instead of silently feeding `cond=None` to a
        condition-aware model."""
        trainer = self._make_trainer_skeleton(
            condition_feature="energy_above_hull", train_dataset=None)
        # Skip start-distribution sampling by passing an explicit start_tensor.
        start = torch.tensor([0, 1], dtype=torch.int64)
        with self.assertRaises(ValueError) as ctx:
            trainer.generate_structures(
                n_structures=2, calibrate=False, start_tensor=start)
        self.assertIn("energy_above_hull", str(ctx.exception))

    @patch("wyckoff_transformer.trainer.WyckoffGenerator")
    @patch("wyckoff_transformer.trainer.load_wyckoff_mappings")
    @patch("wyckoff_transformer.trainer.get_wp_index")
    def test_cond_sampled_from_train_dataset_on_correct_device(
            self, _mock_wp_index, mock_load_mappings, MockGen):
        mock_load_mappings.return_value.ss_from_letter = "ss_from_letter_mock"
        mock_gen_inst = MockGen.return_value
        mock_gen_inst.generate_tensors.return_value = [torch.zeros((4, 5)), torch.ones((4, 5))]

        train_ds = MagicMock()
        train_ds.start_tokens = torch.tensor([0, 1, 2, 3])
        train_ds.num_examples = 100
        # Already-prepared cond tensor (trainer pre-casts in __init__).
        train_ds.data = {"energy": torch.arange(100, dtype=torch.float32).unsqueeze(1)}

        trainer = self._make_trainer_skeleton(
            condition_feature="energy", train_dataset=train_ds)

        trainer.generate_structures(
            n_structures=4, calibrate=False,
            compute_validity_per_known_sequence_length=False)

        call_kwargs = mock_gen_inst.generate_tensors.call_args.kwargs
        cond = call_kwargs["cond"]
        self.assertIsNotNone(cond)
        self.assertEqual(cond.shape, (4, 1))
        self.assertEqual(cond.device.type, trainer.device.type)
        self.assertEqual(cond.dtype, torch.float32)


class TestGenerateEvaluateAndLogWp(unittest.TestCase):
    def _make_trainer(self, tmp):
        trainer = WyckoffTrainer.__new__(WyckoffTrainer)
        trainer.run_path = Path(tmp)
        trainer.generate_structures = MagicMock(
            return_value=([{"k": 1}], [0.5, 0.5], [1.0, 1.0]))
        return trainer

    @patch("wyckoff_transformer.trainer.evaluate_and_log")
    @patch("wyckoff_transformer.trainer.wandb")
    def test_skips_evaluate_and_log_when_evaluator_is_none(self, _mock_wandb, mock_eval):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._make_trainer(tmp)
            trainer.generate_evaluate_and_log_wp(
                generation_name="t", calibrate=False, n_structures=1, evaluator=None)
        mock_eval.assert_not_called()

    @patch("wyckoff_transformer.trainer.evaluate_and_log")
    @patch("wyckoff_transformer.trainer.wandb")
    def test_calls_evaluate_and_log_when_evaluator_provided(self, _mock_wandb, mock_eval):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = self._make_trainer(tmp)
            evaluator = MagicMock()
            trainer.generate_evaluate_and_log_wp(
                generation_name="t", calibrate=False, n_structures=1, evaluator=evaluator)
        mock_eval.assert_called_once()


if __name__ == "__main__":
    unittest.main()
