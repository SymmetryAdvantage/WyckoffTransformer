"""The sampling temperature must reach the generator on every path.

Regression test for a silent defect: `WyckoffTrainer.generate_structures` took a
`temperature` argument and documented it, but only forwarded it on the
element-constrained branch. Ordinary de novo generation -- what
`wyformer-generate` and the de novo ranking protocol both do -- fell through to
the two unconstrained `generate_tensors` calls, which took the default of 1.0.
Nothing raised and nothing looked wrong: a sweep over the temperature simply
produced the same distribution at every setting.

The two halves are tested separately, because they fail separately:

- `WyckoffGenerator.generate_tensors` must actually apply the temperature to
  the logits of every *target* cascade field.
- `WyckoffTrainer.generate_structures` must hand its `temperature` to the
  generator on the unconstrained path, not just the constrained one.
"""
import unittest
from types import SimpleNamespace

import torch

from wyckoff_transformer.generator import WyckoffGenerator
from wyckoff_transformer.trainer import WyckoffTrainer

CASCADE = ("elements", "site_symmetries", "sites_enumeration")
VOCAB = 4
STOP = 3
MASK = 2


class _TwoWayModel:
    """Emits a fixed, *non*-degenerate distribution: token 0 favoured over token 1.

    The gap is small enough that the temperature visibly moves the split, unlike
    the near-one-hot logits a stub usually emits.
    """

    LOGITS = [1.0, 0.0, -1e4, -1e4]

    def eval(self):
        return self

    def __call__(self, start, cascade, padding_mask, prediction_head, cond=None):
        return torch.tensor(self.LOGITS).repeat(start.size(0), 1)


def _generator(model, max_sequence_len=1):
    return WyckoffGenerator(
        model=model,
        cascade_order=CASCADE,
        cascade_is_target={field: True for field in CASCADE},
        token_engineers={},
        masks={field: MASK for field in CASCADE},
        max_sequence_len=max_sequence_len,
        stops={field: STOP for field in CASCADE},
    )


class TestGeneratorTemperature(unittest.TestCase):
    def test_temperature_moves_the_sampled_split(self):
        """A logit gap of 1 nat is a 73/27 split at T=1 and sharper at T=0.2."""
        start = torch.zeros(4000, dtype=torch.int64)
        shares = {}
        for temperature in (0.2, 1.0, 5.0):
            torch.manual_seed(0)
            generated = _generator(_TwoWayModel()).generate_tensors(
                start, temperature=temperature)
            first_field = generated[0][:, 0]
            shares[temperature] = (first_field == 0).float().mean().item()

        # Sharpening drives the favoured token towards 1.0, flattening towards 0.5.
        self.assertGreater(shares[0.2], 0.98)
        self.assertAlmostEqual(shares[1.0], 0.731, delta=0.03)
        self.assertLess(shares[5.0], 0.60)
        self.assertGreater(shares[0.2], shares[1.0])
        self.assertGreater(shares[1.0], shares[5.0])


class _RecordingGenerator:
    """Stands in for `WyckoffGenerator`, records the call and stops the caller.

    `generate_structures` goes on to rebuild structures from the returned
    tensors, which a stub cannot supply; raising here is what keeps the test to
    the one thing it is about.
    """

    calls: list = []

    def __init__(self, *args, **kwargs):
        pass

    def generate_tensors(self, *args, **kwargs):
        type(self).calls.append(kwargs)
        raise _Recorded


class _Recorded(Exception):
    pass


def _trainer_stub():
    """The attributes `generate_structures` touches before it calls the generator."""
    return SimpleNamespace(
        model=_TwoWayModel(),
        cascade_order=list(CASCADE),
        cascade_is_target={field: True for field in CASCADE},
        token_engineers={},
        masks_dict={field: MASK for field in CASCADE},
        max_sequence_length=1,
        stops_dict={field: STOP for field in CASCADE},
        condition_features=(),
        condition_dim=None,
        formula_conditioning_field=None,
        composition_conditioning=False,
        chemical_system_conditioning=False,
        train_dataset=None,
        val_dataset=None,
        tokenisers={},
        device=torch.device("cpu"),
        _sample_start_tokens_from_distribution=(
            lambda n_structures: torch.zeros(n_structures, dtype=torch.int64)),
    )


class TestTrainerForwardsTemperature(unittest.TestCase):
    """`generate_structures` must forward the temperature it was given."""

    def setUp(self):
        _RecordingGenerator.calls = []
        self._real = getattr(
            __import__("wyckoff_transformer.trainer", fromlist=["WyckoffGenerator"]),
            "WyckoffGenerator")
        import wyckoff_transformer.trainer as trainer_module
        self._module = trainer_module
        trainer_module.WyckoffGenerator = _RecordingGenerator

    def tearDown(self):
        self._module.WyckoffGenerator = self._real

    def _call(self, **kwargs):
        with self.assertRaises(_Recorded):
            WyckoffTrainer.generate_structures(
                _trainer_stub(), n_structures=2, calibrate=False, **kwargs)
        self.assertEqual(len(_RecordingGenerator.calls), 1)
        return _RecordingGenerator.calls[0]

    def test_unconstrained_generation_forwards_it(self):
        self.assertEqual(self._call(temperature=0.5)["temperature"], 0.5)

    def test_validity_measurement_forwards_it(self):
        call = self._call(
            temperature=0.5, compute_validity_per_known_sequence_length=True)
        self.assertEqual(call["temperature"], 0.5)
        self.assertTrue(call["compute_validity"])

    def test_the_default_is_one(self):
        self.assertEqual(self._call()["temperature"], 1.0)


if __name__ == "__main__":
    unittest.main()
