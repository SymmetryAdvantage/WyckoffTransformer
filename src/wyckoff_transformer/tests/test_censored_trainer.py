"""The censored loss wired through WyckoffTrainer's Scalar path.

`test_censored.py` checks the likelihood itself. These tests check the plumbing:
that a real `CascadeTransformer` with a two-column head trains through
`get_loss`, that `predict_scalars` hands back the location rather than the
scale, and that the end-to-end fit still finds the floor of a synthetic
gene-to-energy relation rather than its mean.
"""
import unittest

import torch

from wyckoff_transformer.cascade.dataset import AugmentedCascadeDataset, TargetClass
from wyckoff_transformer.cascade.model import CascadeTransformer
from wyckoff_transformer.censored import CensoredMinDiagnostics, CensoredMinLoss
from wyckoff_transformer.trainer import WyckoffTrainer

VOCABULARY = 8
PAD, STOP, MASK = 5, 6, 7
SEQUENCE_LENGTH = 4


def _model(outputs):
    return CascadeTransformer(
        # An embedded integer start token, standing in for the space group.
        start_type="categorial",
        n_start=4,
        # (vocabulary, embedding size, pad token, is a prediction target)
        cascade=((VOCABULARY, 8, PAD, True),),
        token_aggregation="mean",
        aggregate_after_encoder=True,
        include_start_in_aggregation=True,
        aggregation_inclsion="aggr",
        concat_token_counts=False,
        concat_token_presence=False,
        num_fully_connected_layers=2,
        mixer_layers=1,
        outputs=outputs,
        perceptron_shape="pyramid",
        TransformerEncoderLayer_args={"nhead": 2, "dim_feedforward": 16, "dropout": 0.0},
        TransformerEncoder_args={"num_layers": 1, "enable_nested_tensor": False},
        learned_positional_encoding_max_size=0,
        learned_positional_encoding_only_masked=True)


def _dataset(genes, energies):
    """One cascade field standing in for a gene, plus its energy label."""
    rows = []
    for gene in genes:
        row = list(gene) + [STOP]
        rows.append(row + [PAD] * (SEQUENCE_LENGTH - len(row)))
    return AugmentedCascadeDataset(
        data={"field": torch.tensor(rows, dtype=torch.int64),
              "spacegroup": torch.zeros(len(genes), dtype=torch.int64),
              "energy": torch.tensor(energies, dtype=torch.float32)},
        cascade_order=("field",),
        masks={"field": MASK}, pads={"field": PAD}, stops={"field": STOP},
        num_classes={"field": VOCABULARY},
        start_field="spacegroup", augmented_fields=None, target_name="energy")


def _trainer(model, dataset, scalar_loss, censored_loss_args=None):
    """A trainer assembled directly, as the other tests in this suite do."""
    trainer = WyckoffTrainer.__new__(WyckoffTrainer)
    trainer.model = model
    trainer.target = TargetClass.Scalar
    trainer.scalar_loss = scalar_loss
    trainer.censored_diagnostics = None
    trainer.multiclass_next_token_with_order_permutation = False
    trainer.cascade_order = ("field",)
    trainer.cascade_is_target = {"field": True}
    trainer.masks_dict = {"field": MASK}
    trainer.pad_dict = {"field": PAD}
    trainer.stops_dict = {"field": STOP}
    trainer.num_classes_dict = {"field": VOCABULARY}
    trainer.start_name = "spacegroup"
    trainer.augmented_fields = None
    trainer.max_sequence_length = SEQUENCE_LENGTH
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.int64
    trainer.train_dataset = dataset
    trainer.condition_feature = None
    trainer.tokeniser_config = None
    trainer.token_engineers = {}
    trainer.tokenisers = {}
    if scalar_loss == "censored":
        trainer.criterion = CensoredMinLoss(reduction="mean", **(censored_loss_args or {}))
        trainer.testing_criterion = trainer.criterion
        trainer.censored_diagnostics = CensoredMinDiagnostics(trainer.criterion)
    else:
        trainer.criterion = torch.nn.MSELoss(reduction="mean")
        trainer.testing_criterion = torch.nn.L1Loss(reduction="mean")
    return trainer


class TestScalarPathAcceptsTheCensoredLoss(unittest.TestCase):
    def test_loss_is_finite_and_backpropagates(self):
        torch.manual_seed(0)
        dataset = _dataset([(0, 1), (1, 2), (2, 0)], [1.0, 2.0, 3.0])
        trainer = _trainer(_model(2), dataset, "censored")
        loss = trainer.get_loss(dataset, SEQUENCE_LENGTH - 1, None, no_batch=True)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in trainer.model.parameters()))

    def test_a_batch_of_one_keeps_the_two_columns(self):
        # The reshape, rather than a bare squeeze(), is what makes this work.
        torch.manual_seed(0)
        dataset = _dataset([(0, 1)], [1.0])
        trainer = _trainer(_model(2), dataset, "censored")
        self.assertTrue(torch.isfinite(
            trainer.get_loss(dataset, SEQUENCE_LENGTH - 1, None, no_batch=True)))

    def test_mse_path_is_untouched(self):
        torch.manual_seed(0)
        dataset = _dataset([(0, 1), (1, 2)], [1.0, 2.0])
        trainer = _trainer(_model(1), dataset, "mse")
        self.assertTrue(torch.isfinite(
            trainer.get_loss(dataset, SEQUENCE_LENGTH - 1, None, no_batch=True)))

    def test_predict_scalars_returns_the_location_column(self):
        torch.manual_seed(0)
        dataset = _dataset([(0, 1), (1, 2), (2, 0)], [1.0, 2.0, 3.0])
        trainer = _trainer(_model(2), dataset, "censored")
        data = {"field": dataset.data["field"], "spacegroup": dataset.start_tokens}
        mean, all_samples = trainer.predict_scalars(data)
        self.assertEqual(mean.shape, (3,))
        self.assertEqual(all_samples.shape, (1, 3))
        raw = trainer.model(dataset.start_tokens, [dataset.data["field"]],
                            dataset.padding_mask, None)
        self.assertEqual(raw.shape, (3, 2))
        self.assertTrue(torch.allclose(mean, raw[:, 0], atol=1e-5))


class TestEndToEndRecoversTheFloor(unittest.TestCase):
    """The behaviour the whole change exists for, through the real training path."""

    def test_fitted_model_predicts_the_minimum_not_the_mean(self):
        torch.manual_seed(0)
        # Two genes. Each is observed several times, with energies scattered above
        # its own floor by an exponential excess -- what a dataset of structures
        # sharing a Wyckoff gene looks like.
        floors = {(0, 1): -2.0, (2, 3): -1.0}
        scale, repeats = 0.4, 60
        genes, energies = [], []
        for gene, floor in floors.items():
            draws = torch.distributions.Exponential(1 / scale).sample((repeats,))
            genes.extend([gene] * repeats)
            energies.extend((floor + draws).tolist())
        dataset = _dataset(genes, energies)

        trainer = _trainer(_model(2), dataset, "censored", {"noise": 0.02})
        optimizer = torch.optim.Adam(trainer.trainable_parameters(), lr=0.02)
        # Slower to converge than the MSE fit on the same data: the location starts
        # near the conditional mean and has to be pushed down through the barrier,
        # which is one-sided by construction.
        for _ in range(2500):
            optimizer.zero_grad()
            trainer.get_loss(dataset, SEQUENCE_LENGTH - 1, None, no_batch=True).backward()
            optimizer.step()

        data = {"field": dataset.data["field"], "spacegroup": dataset.start_tokens}
        predicted, _ = trainer.predict_scalars(data)
        for gene, floor in floors.items():
            rows = [i for i, g in enumerate(genes) if g == gene]
            estimate = predicted[rows].mean().item()
            observed_mean = torch.tensor([energies[i] for i in rows]).mean().item()
            self.assertAlmostEqual(estimate, floor, delta=0.25)
            # And it is meaningfully below the mean an MSE fit would have found.
            self.assertLess(estimate, observed_mean - 0.5 * scale)

    def test_diagnostics_report_a_calibrated_fit(self):
        torch.manual_seed(1)
        scale, repeats = 0.3, 80
        draws = torch.distributions.Exponential(1 / scale).sample((repeats,))
        dataset = _dataset([(0, 1)] * repeats, (draws - 1.0).tolist())
        trainer = _trainer(_model(2), dataset, "censored", {"noise": 0.02})
        optimizer = torch.optim.Adam(trainer.trainable_parameters(), lr=0.02)
        for _ in range(1500):
            optimizer.zero_grad()
            trainer.get_loss(dataset, SEQUENCE_LENGTH - 1, None, no_batch=True).backward()
            optimizer.step()

        start, cascade, target, padding = dataset.get_augmented_data()
        prediction = trainer.model(start, cascade, padding, None).reshape(-1, 2)
        reported = trainer.censored_diagnostics(prediction, target)
        # Almost nothing may sit below a minimum, and the fitted excess should be
        # near the one the data were generated with.
        self.assertLess(reported["violation"].item(), 0.05)
        self.assertAlmostEqual(reported["scale"].item(), scale, delta=0.15)


if __name__ == "__main__":
    unittest.main()
