"""Properties the loss estimators must have, independent of the model.

`train_epoch` and `evaluate` reach the same criterion by different routes -- one samples a
single (known_seq_len, known_cascade_len) subproblem per step and reduces to a per-example mean,
the other walks every subproblem and rescales each to the whole viable set. These tests pin down
that the two agree in expectation, that neither depends on the batch size, and that the only
thing separating a training step from an evaluation of the same weights is dropout.
"""
import unittest

import torch
from torch import nn

from wyckoff_transformer.cascade.dataset import (
    AugmentedCascadeDataset, AugmentedCascadeLoader, TargetClass)
from wyckoff_transformer.cascade.model import CascadeTransformer
from wyckoff_transformer.trainer import WyckoffTrainer

N_CLASSES = 8
PAD, MASK, STOP = 5, 6, 7
LENGTHS = [1, 1, 2, 2, 2, 3, 3, 4, 4, 5] * 4      # skewed, like the real Wyckoff data
MAX_SEQ = 7


def _make_dataset(batch_size=None, lengths=LENGTHS):
    """A dataset whose sequences have the given pure lengths, then STOP, then PAD."""
    rows = []
    for length in lengths:
        row = torch.randint(0, PAD, (MAX_SEQ,))
        row[length] = STOP
        row[length + 1:] = PAD
        rows.append(row)
    data = {
        "field1": torch.stack(rows).to(torch.int64),
        "spacegroup": torch.randint(0, 4, (len(lengths),)),
        "pure_sequence_length": torch.tensor(lengths, dtype=torch.int64),
    }
    return AugmentedCascadeDataset(
        data=data, cascade_order=("field1",), masks={"field1": MASK}, pads={"field1": PAD},
        stops={"field1": STOP}, num_classes={"field1": N_CLASSES}, start_field="spacegroup",
        augmented_fields=None, batch_size=batch_size)


def _make_model(dropout):
    return CascadeTransformer(
        start_type="categorial", n_start=4,
        cascade=((N_CLASSES, 4, PAD, True),),
        token_aggregation=None, aggregate_after_encoder=False,
        include_start_in_aggregation=False, aggregation_inclsion="None",
        concat_token_counts=False, concat_token_presence=False,
        num_fully_connected_layers=1, mixer_layers=1, outputs="token_scores",
        perceptron_shape="input",
        TransformerEncoderLayer_args={"nhead": 2, "dim_feedforward": 16, "dropout": dropout},
        TransformerEncoder_args={"num_layers": 1, "enable_nested_tensor": False},
        learned_positional_encoding_max_size=0, learned_positional_encoding_only_masked=True,
        condition_dim=None)


def _make_trainer(model):
    t = WyckoffTrainer.__new__(WyckoffTrainer)
    t.target = TargetClass.NextToken
    t.multiclass_next_token_with_order_permutation = True
    t.condition_feature = None
    t.cascade_len = 1
    t.cascade_target_count = 1
    t.cascade_order = ("field1",)
    t.evaluation_samples = 1
    t.device = torch.device("cpu")
    t.optimizer = None
    t.criterion = nn.CrossEntropyLoss(reduction="sum")
    t.model = model
    return t


@torch.no_grad()
def _mean_over_draws(fn, draws, seed=0):
    torch.manual_seed(seed)
    return sum(float(fn()) for _ in range(draws)) / draws


class TestBatchedMatchesExhaustive(unittest.TestCase):
    """A sampled batch, rescaled, must estimate the pass over every viable example."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = _make_model(dropout=0.0).eval()

    def test_per_subproblem_expectation_matches(self):
        dataset = _make_dataset(batch_size=8)
        loader = AugmentedCascadeLoader.from_dataset(dataset)
        t = _make_trainer(self.model)
        for k in range(MAX_SEQ):
            if dataset.viable_count(k) == 0:
                continue
            exhaustive = _mean_over_draws(
                lambda: t.get_loss(dataset, k, 0, no_batch=True), 60, seed=1)
            batched = _mean_over_draws(
                lambda: t.get_loss(dataset, k, 0, loader=loader, no_batch=False), 600, seed=2)
            self.assertAlmostEqual(
                batched, exhaustive, delta=0.02 * max(1.0, abs(exhaustive)),
                msg=f"batched estimator is biased at known_seq_len={k}")

    def test_expectation_does_not_depend_on_batch_size(self):
        t = _make_trainer(self.model)
        # One dataset, several loaders: _make_dataset randomises the sequences, so rebuilding it
        # per batch size would compare different data rather than different batchings.
        torch.manual_seed(0)
        dataset = _make_dataset(batch_size=None)
        means = {}
        for batch_size in (4, 8, 20, len(LENGTHS)):
            loader = AugmentedCascadeLoader(dataset, batch_size=batch_size)
            means[batch_size] = sum(
                _mean_over_draws(
                    lambda k=k: t.get_loss(dataset, k, 0, loader=loader, no_batch=False),
                    600, seed=4 + k)
                for k in range(MAX_SEQ) if dataset.viable_count(k) > 0)
        reference = means[len(LENGTHS)]   # a full-batch loader: the exact value
        for batch_size, value in means.items():
            self.assertAlmostEqual(
                value, reference, delta=0.02 * abs(reference),
                msg=f"split loss depends on batch_size (got {value} at {batch_size})")


class TestTrainingObjectiveTracksTheMetric(unittest.TestCase):
    """What train_epoch descends must be proportional to what evaluate reports.

    A step draws known_seq_len from p(k) proportional to viable_count(k) and reduces to a
    per-example mean; the metric sums viable_count(k) * mean(k) over k and divides by the split
    size. So E[step loss] = metric * N / Z with Z = sum_k viable_count(k) -- one positive
    constant, hence the same minimiser. If this fails, training and evaluation are chasing
    different optima and a run can degrade on the metric while the optimiser is still winning.
    """

    def test_expected_step_loss_is_proportional_to_the_split_loss(self):
        torch.manual_seed(0)
        model = _make_model(dropout=0.0).eval()
        dataset = _make_dataset(batch_size=8)
        loader = AugmentedCascadeLoader.from_dataset(dataset)
        t = _make_trainer(model)

        metric = _mean_over_draws(lambda: t.evaluate(dataset, loader).sum(), 400, seed=5)

        def one_step():
            k = dataset.sample_known_seq_len()
            loss, n = t.get_loss(dataset, k, 0, loader=loader,
                                 rescale_to_viable=False, return_n_samples=True)
            return loss / n

        import random
        random.seed(6)
        expected_step = _mean_over_draws(one_step, 20000, seed=7)

        z = sum(dataset.viable_count(k) for k in range(dataset.max_sequence_length))
        predicted = metric * len(dataset) / z
        self.assertAlmostEqual(
            expected_step, predicted, delta=0.03 * abs(predicted),
            msg=f"E[train step] = {expected_step:.5f} but the metric predicts {predicted:.5f}; "
                "the training objective is not proportional to the reported one")


class TestTrainEvalModeAgreement(unittest.TestCase):
    """Dropout must be the only thing that makes a training step differ from an evaluation."""

    @torch.no_grad()
    def _loss_in_mode(self, t, dataset, loader, train_mode, seed):
        t.model.train(train_mode)
        torch.manual_seed(seed)
        return sum(float(t.get_loss(dataset, k, 0, loader=loader, no_batch=False))
                   for k in range(MAX_SEQ) if dataset.viable_count(k) > 0)

    def test_identical_without_dropout(self):
        torch.manual_seed(0)
        t = _make_trainer(_make_model(dropout=0.0))
        dataset = _make_dataset(batch_size=8)
        loader = AugmentedCascadeLoader.from_dataset(dataset)
        train = self._loss_in_mode(t, dataset, loader, True, seed=11)
        evalm = self._loss_in_mode(t, dataset, loader, False, seed=11)
        self.assertAlmostEqual(
            train, evalm, places=4,
            msg="train and eval mode disagree with dropout off -- some other layer is "
                "mode-dependent, and the training objective is not the reported one")

    def test_dropout_alone_reproduces_the_discrepancy(self):
        torch.manual_seed(0)
        t = _make_trainer(_make_model(dropout=0.5))
        dataset = _make_dataset(batch_size=8)
        loader = AugmentedCascadeLoader.from_dataset(dataset)
        train = self._loss_in_mode(t, dataset, loader, True, seed=11)
        evalm = self._loss_in_mode(t, dataset, loader, False, seed=11)
        self.assertGreater(
            train, evalm,
            "dropout should make the training-mode loss the larger of the two")


class TestMulticlassTargetIsTheRaoBlackwellisation(unittest.TestCase):
    """The soft multiclass target must have the same expectation as the sampled next token.

    At known_cascade_len 0 the target is the distribution of the tokens still to come, which is
    exactly the expectation over permutations of the one-hot next token. Averaging the sampled
    version over many permutations must land on the soft-target loss.
    """

    def test_soft_target_matches_the_permutation_average(self):
        torch.manual_seed(0)
        model = _make_model(dropout=0.0).eval()
        t = _make_trainer(model)
        dataset = _make_dataset(batch_size=None)

        def loss_with(multiclass):
            start, masked, target = dataset.get_masked_multiclass_cascade_data(
                k, 0, target_type=TargetClass.NextToken, multiclass_target=multiclass)
            return float(t.criterion(model(start, masked, None, 0, cond=None), target)) / start.size(0)

        for k in range(MAX_SEQ):
            if dataset.viable_count(k) == 0:
                continue
            soft = _mean_over_draws(lambda: loss_with(True), 200, seed=21)
            sampled = _mean_over_draws(lambda: loss_with(False), 4000, seed=22)
            self.assertAlmostEqual(
                sampled, soft, delta=0.03 * max(1.0, abs(soft)),
                msg=f"soft multiclass target is not the permutation average at known_seq_len={k}")


if __name__ == "__main__":
    unittest.main()
