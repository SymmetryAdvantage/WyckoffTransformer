"""The training-time monitor for schedule-free's averaged iterate going stale."""
import unittest

import schedulefree
import torch
from torch import nn

from wyckoff_transformer.cascade.dataset import (
    AugmentedCascadeDataset, AugmentedCascadeLoader, TargetClass)
from wyckoff_transformer.trainer import WyckoffTrainer

MAX_SEQ = 6


def _make_dataset(batch_size=None):
    lengths = [1, 1, 2, 2, 3, 3, 4, 5] * 3
    n = len(lengths)
    rows = []
    for length in lengths:
        row = torch.randint(0, 5, (MAX_SEQ,))
        row[length] = 7
        row[length + 1:] = 5
        rows.append(row)
    data = {"field1": torch.stack(rows).to(torch.int64),
            "spacegroup": torch.zeros(n, dtype=torch.int64),
            "pure_sequence_length": torch.tensor(lengths, dtype=torch.int64)}
    return AugmentedCascadeDataset(
        data=data, cascade_order=("field1",), masks={"field1": 6}, pads={"field1": 5},
        stops={"field1": 7}, num_classes={"field1": 8}, start_field="spacegroup",
        augmented_fields=None, batch_size=batch_size)


class _TinyModel(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.head = nn.Linear(1, n_classes)

    def forward(self, start_tokens, masked_data, padding_mask, known_cascade_len, cond=None):
        return self.head(start_tokens.float().unsqueeze(-1))


def _make_trainer(optimizer_cls, **opt_kwargs):
    torch.manual_seed(0)
    dataset = _make_dataset(batch_size=8)
    t = WyckoffTrainer.__new__(WyckoffTrainer)
    t.target = TargetClass.NextToken
    t.multiclass_next_token_with_order_permutation = True
    t.condition_feature = None
    t.cascade_len = 1
    t.cascade_target_count = 1
    t.cascade_order = ("field1",)
    t.evaluation_samples = 1
    t.device = torch.device("cpu")
    t.clip_grad_norm = None
    t.criterion = nn.CrossEntropyLoss(reduction="sum")
    t.model = _TinyModel()
    t.train_dataset = dataset
    t.train_loader = AugmentedCascadeLoader.from_dataset(dataset)
    t.optimizer = optimizer_cls(t.model.parameters(), **opt_kwargs)
    return t


def _take_steps(t, n):
    from unittest.mock import patch
    with patch("wyckoff_transformer.trainer.wandb"):
        for _ in range(n):
            t.train_epoch()


class TestScheduleFreeLag(unittest.TestCase):
    def test_reports_nothing_for_an_optimiser_without_an_average(self):
        t = _make_trainer(torch.optim.SGD, lr=0.01)
        _take_steps(t, 2)
        self.assertEqual(t.schedule_free_lag(), {},
                         "plain SGD keeps no averaged iterate; there is nothing to report")

    def test_reports_nothing_before_the_first_step(self):
        t = _make_trainer(schedulefree.SGDScheduleFree, lr=0.01, warmup_steps=0)
        self.assertEqual(t.schedule_free_lag(), {})

    def test_reports_the_lag_once_the_average_exists(self):
        t = _make_trainer(schedulefree.SGDScheduleFree, lr=0.05, warmup_steps=0)
        _take_steps(t, 3)
        m = t.schedule_free_lag()
        for key in ("lag", "lag_relative", "norm_x", "norm_z", "loss_z", "loss_x_minus_z"):
            self.assertIn(key, m)
        self.assertGreater(m["lag"], 0.0, "x and z cannot coincide after several steps")
        self.assertAlmostEqual(m["lag_relative"], m["lag"] / m["norm_x"], places=6)

    def test_x_and_z_are_compared_on_the_same_batches(self):
        """An unpaired difference would be swamped by the estimator's own sampling noise."""
        t = _make_trainer(schedulefree.SGDScheduleFree, lr=0.05, warmup_steps=0)
        _take_steps(t, 4)
        m = t.schedule_free_lag()
        t.optimizer.eval()
        params = [p for group in t.optimizer.param_groups for p in group["params"]]
        x = [p.detach().clone() for p in params]
        z = [t.optimizer.state[p].get("z", p).detach().clone() for p in params]

        def at(weights):
            for p, w in zip(params, weights):
                p.data.copy_(w)
            torch.manual_seed(WyckoffTrainer.LAG_EVAL_SEED)
            return t.evaluate(t.train_dataset, t.train_loader).sum().item()

        # Both iterates must have been measured under the one seed, not merely deterministically.
        self.assertAlmostEqual(m["loss_x"], at(x), places=6, msg="loss_x used other batches")
        self.assertAlmostEqual(m["loss_z"], at(z), places=6, msg="loss_z used other batches")
        for p, w in zip(params, x):
            p.data.copy_(w)
        self.assertAlmostEqual(m["loss_x_minus_z"], m["loss_x"] - m["loss_z"], places=9)

    def test_repeated_measurements_agree(self):
        t = _make_trainer(schedulefree.SGDScheduleFree, lr=0.05, warmup_steps=0)
        _take_steps(t, 4)
        first, second = t.schedule_free_lag(), t.schedule_free_lag()
        self.assertAlmostEqual(first["loss_x_minus_z"], second["loss_x_minus_z"], places=6)

    def test_the_training_random_stream_is_left_alone(self):
        """Seeding the measurement must not make the following training steps deterministic."""
        t = _make_trainer(schedulefree.SGDScheduleFree, lr=0.05, warmup_steps=0)
        _take_steps(t, 3)
        torch.manual_seed(1234)
        expected = torch.randn(4)
        torch.manual_seed(1234)
        t.schedule_free_lag()
        self.assertTrue(torch.equal(torch.randn(4), expected),
                        "the monitor perturbed the training random stream")

    def test_the_weights_are_left_exactly_as_found(self):
        """The monitor swaps z into the model to evaluate it; training must not notice."""
        t = _make_trainer(schedulefree.SGDScheduleFree, lr=0.05, warmup_steps=0)
        _take_steps(t, 3)
        t.optimizer.eval()
        before = [p.detach().clone() for p in t.model.parameters()]
        train_mode_before = t.optimizer.param_groups[0]["train_mode"]
        t.schedule_free_lag()
        for a, b in zip(before, t.model.parameters()):
            self.assertTrue(torch.equal(a, b), "the monitor perturbed the model weights")
        self.assertEqual(t.optimizer.param_groups[0]["train_mode"], train_mode_before)

    def test_training_continues_correctly_afterwards(self):
        """A botched restore would show up as the loss jumping after a measurement."""
        t = _make_trainer(schedulefree.SGDScheduleFree, lr=0.05, warmup_steps=0)
        _take_steps(t, 4)
        t.optimizer.eval()
        # evaluate() samples its batches and permutations, so seed it: we are testing that the
        # weights come back, not that the estimator is deterministic.
        torch.manual_seed(7)
        before = t.evaluate(t.train_dataset, t.train_loader).sum().item()
        t.schedule_free_lag()
        t.optimizer.eval()
        torch.manual_seed(7)
        after = t.evaluate(t.train_dataset, t.train_loader).sum().item()
        self.assertAlmostEqual(before, after, places=4)


if __name__ == "__main__":
    unittest.main()
