"""Gradient clipping is optional, and the pre-clip norm is logged whether or not it binds."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from wyckoff_transformer.cascade.dataset import (
    AugmentedCascadeDataset, AugmentedCascadeLoader, TargetClass)
from wyckoff_transformer.trainer import WyckoffTrainer


# Small enough that the stub's gradients always exceed it, so it is guaranteed to bind.
BINDING = 1e-3


class _TinyModel(nn.Module):
    """Enough of the model interface for train_epoch, with real parameters to differentiate."""

    def __init__(self, n_classes=102):
        super().__init__()
        self.head = nn.Linear(1, n_classes)

    def forward(self, start_tokens, masked_data, padding_mask, known_cascade_len, cond=None):
        return self.head(start_tokens.float().unsqueeze(-1))


def _make_dataset(lengths, max_sequence_length=8, batch_size=None):
    n = len(lengths)
    data = {
        "field1": torch.arange(n * max_sequence_length, dtype=torch.int64).reshape(
            n, max_sequence_length) % 3,
        "spacegroup": torch.zeros(n, dtype=torch.int64),
        "pure_sequence_length": torch.tensor(lengths, dtype=torch.int64),
    }
    return AugmentedCascadeDataset(
        data=data, cascade_order=("field1",), masks={"field1": 99}, pads={"field1": 100},
        stops={"field1": 101}, num_classes={"field1": 102}, start_field="spacegroup",
        augmented_fields=None, batch_size=batch_size)


def _make_trainer(clip_grad_norm):
    dataset = _make_dataset([0, 1, 2, 3, 4, 5, 6, 7] * 8, batch_size=16)
    trainer = WyckoffTrainer.__new__(WyckoffTrainer)
    trainer.target = TargetClass.NextToken
    trainer.multiclass_next_token_with_order_permutation = True
    trainer.condition_feature = None
    trainer.cascade_target_count = 1
    trainer.clip_grad_norm = clip_grad_norm
    trainer.train_dataset = dataset
    trainer.train_loader = AugmentedCascadeLoader.from_dataset(dataset)
    trainer.criterion = nn.CrossEntropyLoss(reduction="sum")
    trainer.model = _TinyModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
    return trainer


def _run_epoch(trainer):
    """Run one epoch, returning the logged grad_norm values and the post-clip norms."""
    logged, applied = [], []
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):
        norm = real_clip(parameters, max_norm, *args, **kwargs)
        applied.append(torch.cat([p.grad.flatten() for p in trainer.model.parameters()
                                  if p.grad is not None]).norm().item())
        return norm

    with patch("wyckoff_transformer.trainer.wandb") as wandb_mock, \
         patch("torch.nn.utils.clip_grad_norm_", side_effect=spy):
        trainer.train_epoch()
        for call in wandb_mock.log.call_args_list:
            logged.append(float(call.args[0]["grad_norm"]))
    return logged, applied


class TestGradientClippingIsOptional(unittest.TestCase):
    def test_absent_config_key_means_no_clipping(self):
        """`optimisation.clip_grad_norm` may be omitted entirely."""
        from omegaconf import OmegaConf
        config = OmegaConf.create({"epochs": 1, "validation_period": 1})
        self.assertIsNone(config.get("clip_grad_norm", None))

    def test_none_leaves_the_gradient_untouched(self):
        trainer = _make_trainer(clip_grad_norm=None)
        logged, applied = _run_epoch(trainer)
        self.assertTrue(logged, "no step was logged")
        # The same gradients would have been rescaled by the threshold the tests below use.
        self.assertTrue(all(n > BINDING for n in logged), "gradients were too small to be a test")
        for before, after in zip(logged, applied):
            self.assertAlmostEqual(before, after, places=6,
                                   msg="an unconstrained gradient was rescaled")

    def test_a_binding_threshold_rescales_the_gradient(self):
        trainer = _make_trainer(clip_grad_norm=BINDING)
        logged, applied = _run_epoch(trainer)
        self.assertTrue(all(n > BINDING for n in logged), "the threshold did not bind")
        for after in applied:
            self.assertLessEqual(after, BINDING * (1 + 1e-3))

    def test_a_slack_threshold_leaves_the_gradient_alone(self):
        trainer = _make_trainer(clip_grad_norm=1e9)
        logged, applied = _run_epoch(trainer)
        self.assertTrue(all(n < 1e9 for n in logged), "gradients were unexpectedly huge")
        for before, after in zip(logged, applied):
            self.assertAlmostEqual(before, after, places=6)

    def test_the_pre_clip_norm_is_logged_even_when_clipping_binds(self):
        """The point of the metric: it must report the norm before the constraint, not after."""
        trainer = _make_trainer(clip_grad_norm=BINDING)
        logged, _ = _run_epoch(trainer)
        self.assertTrue(all(n > BINDING for n in logged),
                        "grad_norm was logged post-clip, which would hide a binding threshold")


if __name__ == "__main__":
    unittest.main()
