"""The generative novelty score has to be an actual log-density.

Two things can go quietly wrong and leave a score that still looks plausible.
The combinatorial factor |R(G)| -- how many token sequences decode to one gene --
is of the order of log n!, so getting it wrong turns the score into a measure of
gene size. And the stopping term is carried by every cascade field at once in the
tokenised data, so counting all of them charges one stopping decision three times
over, again in proportion to nothing in particular.

Both are checked against a model whose distribution is known exactly: with
uniform logits every representation has the same likelihood, so the estimator
must return the closed form, whatever the ordering it happens to draw.
"""
import math
import unittest
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from wyckoff_transformer.cascade.dataset import TargetClass
from wyckoff_transformer.gene_likelihood import (
    gene_representations,
    log_orderings,
    representation_log_likelihood,
    score_gene_likelihood,
    start_log_prior,
)
from wyckoff_transformer.tokenization import EnumeratingTokeniser

CASCADE_ORDER = ("elements", "site_symmetries", "sites_enumeration")
#: Deliberately different per field, so a test that passes cannot be doing so by
#: multiplying the right count by the wrong vocabulary.
TOKENS = {
    "elements": {"Fe", "O", "Li"},
    "site_symmetries": {"1", "2", "4", "m"},
    "sites_enumeration": {"a", "b", "c", "d", "e"},
}


class _UniformModel(torch.nn.Module):
    """Uniform logits over each field's vocabulary, ignoring its input entirely."""

    start_type = "one_hot"

    def __init__(self, num_classes: List[int]):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, start, cascade, padding_mask, known_cascade_len, cond=None):
        return torch.zeros(cascade[0].size(0), self.num_classes[known_cascade_len])


class _SpaceGroupStub(dict):
    """The two-space-group encoder the stub trainer needs, without pyxtal."""

    def __init__(self, vectors: Dict[int, tuple]):
        super().__init__({sg: index for index, sg in enumerate(vectors)})
        self.np_dict = {sg: np.asarray(vector, dtype=float) for sg, vector in vectors.items()}

    def encode_spacegroups(self, space_groups, **tensor_args):
        return torch.stack(
            [torch.from_numpy(self.np_dict[sg]) for sg in space_groups]).to(**tensor_args)


class _TrainerStub:
    """The surface `representation_log_likelihood` reads, and nothing else."""

    def __init__(self):
        self.cascade_order = CASCADE_ORDER
        self.tokenisers = {
            field: EnumeratingTokeniser.from_token_set(tokens)
            for field, tokens in TOKENS.items()}
        self.tokenisers["spacegroup_number"] = _SpaceGroupStub({1: (1.0, 0.0), 2: (0.0, 1.0)})
        self.token_engineers = {}
        self.masks_dict = {f: self.tokenisers[f].mask_token for f in CASCADE_ORDER}
        self.pad_dict = {f: self.tokenisers[f].pad_token for f in CASCADE_ORDER}
        self.stops_dict = {f: self.tokenisers[f].stop_token for f in CASCADE_ORDER}
        self.num_classes_dict = {f: len(self.tokenisers[f]) for f in CASCADE_ORDER}
        self.start_name = "spacegroup_number"
        self.dtype = torch.int64
        self.device = torch.device("cpu")
        self.target = TargetClass.NextToken
        self.condition_features = ()
        self.composition_conditioning = False
        self.cascade_target_indices = (0, 1, 2)
        self.train_dataset = None
        self.model = _UniformModel([self.num_classes_dict[f] for f in CASCADE_ORDER])
        self.start_token_distribution = {
            "start_name": "spacegroup_number",
            "start_type": "one_hot",
            "max_sequence_length": 8,
            "vectors": [[1.0, 0.0], [0.0, 1.0]],
            "counts": [3, 1],
        }
        self.tokeniser_config = OmegaConf.create({
            "dtype": "int64",
            "token_fields": {"pure_categorical": list(CASCADE_ORDER)},
            "sequence_fields": {"space_group": ["spacegroup_number"]},
        })

    def log_uniform_per_site(self) -> float:
        return -sum(math.log(self.num_classes_dict[f]) for f in CASCADE_ORDER)

    def log_uniform_stop(self) -> float:
        return -math.log(self.num_classes_dict[CASCADE_ORDER[0]])


def _record(elements, symmetries, enumerations, space_group=1, augmented=None):
    record = {
        "elements": list(elements),
        "site_symmetries": list(symmetries),
        "sites_enumeration": list(enumerations),
        "spacegroup_number": space_group,
    }
    if augmented is not None:
        record["sites_enumeration_augmented"] = frozenset(map(tuple, augmented))
    return record


class TestOrderingCounts(unittest.TestCase):
    def test_distinct_sites_give_factorial(self):
        sites = [("Fe", "1", "a"), ("O", "2", "b"), ("Li", "4", "c")]
        self.assertAlmostEqual(log_orderings(sites), math.log(6))

    def test_repeated_sites_are_not_distinct_orderings(self):
        # Three identical tokens can be arranged exactly one way, not six: the
        # model sees a sequence of tokens, not a labelled set of sites.
        sites = [("Fe", "1", "a")] * 3
        self.assertAlmostEqual(log_orderings(sites), 0.0)
        mixed = [("Fe", "1", "a"), ("Fe", "1", "a"), ("O", "2", "b")]
        self.assertAlmostEqual(log_orderings(mixed), math.log(3))

    def test_equivalent_enumerations_that_collapse_are_counted_once(self):
        # Two augmentations that permute the enumerations between two sites of
        # the same element and symmetry give the same multiset of tokens, so
        # they are one representation, not two.
        record = _record(
            ["Fe", "Fe"], ["1", "1"], ["a", "b"],
            augmented=[("a", "b"), ("b", "a"), ("c", "d")])
        multisets, weights = gene_representations(record)
        self.assertEqual(len(multisets), 2)
        self.assertTrue(np.allclose(weights, math.log(2)))

    def test_plain_enumeration_is_used_without_augmentation(self):
        multisets, weights = gene_representations(
            _record(["Fe", "O"], ["1", "2"], ["a", "b"]))
        self.assertEqual(len(multisets), 1)
        self.assertAlmostEqual(float(weights[0]), math.log(2))


class TestRepresentationLikelihood(unittest.TestCase):
    def setUp(self):
        self.trainer = _TrainerStub()

    def _frame(self, records):
        frame = pd.DataFrame.from_records(records)
        frame.index.name = "index"
        return frame

    def test_uniform_model_charges_one_term_per_token_plus_one_stop(self):
        records = [
            _record(["Fe", "O"], ["1", "2"], ["a", "b"]),
            _record(["Fe", "O", "Li"], ["1", "2", "4"], ["a", "b", "c"]),
        ]
        got = representation_log_likelihood(self.trainer, self._frame(records))
        for row, record in enumerate(records):
            n_sites = len(record["elements"])
            expected = (n_sites * self.trainer.log_uniform_per_site()
                        + self.trainer.log_uniform_stop())
            self.assertAlmostEqual(float(got[row]), expected, places=4)

    def test_a_shorter_gene_is_not_charged_for_the_padding(self):
        # Both genes go through the same padded tensor; the shorter one must stop
        # accumulating at its own STOP rather than at the width of the batch.
        alone = representation_log_likelihood(
            self.trainer, self._frame([_record(["Fe"], ["1"], ["a"])]))
        padded = representation_log_likelihood(
            self.trainer,
            self._frame([
                _record(["Fe"], ["1"], ["a"]),
                _record(["Fe", "O", "Li"], ["1", "2", "4"], ["a", "b", "c"]),
            ]))
        self.assertAlmostEqual(float(alone[0]), float(padded[0]), places=4)


class TestGeneLikelihood(unittest.TestCase):
    def setUp(self):
        self.trainer = _TrainerStub()

    def test_density_is_the_closed_form_under_a_uniform_model(self):
        # Every representation has the same likelihood here, so the estimator has
        # nothing to average over and must return |R| * p(r) * p(spacegroup)
        # exactly -- which is the whole normalisation, checked in one place.
        records = pd.DataFrame.from_records([
            _record(["Fe", "O"], ["1", "2"], ["a", "b"], space_group=1),
            _record(["Fe", "Fe", "O"], ["1", "1", "2"], ["a", "b", "c"], space_group=2,
                    augmented=[("a", "b", "c"), ("b", "a", "c"), ("c", "d", "e")]),
        ])
        records.index.name = "index"
        got = score_gene_likelihood(records, self.trainer, permutation_samples=4, seed=3)

        multiset_counts = [1, 2]  # the second gene's two augmentations collapse to one
        for row, (record, n_multisets) in enumerate(zip(
                records.to_dict("records"), multiset_counts)):
            n_sites = len(record["elements"])
            log_representations = math.log(n_multisets) + math.lgamma(n_sites + 1)
            log_p_start = math.log({1: 3 / 4, 2: 1 / 4}[record["spacegroup_number"]])
            expected = (log_representations
                        + n_sites * self.trainer.log_uniform_per_site()
                        + self.trainer.log_uniform_stop()
                        + log_p_start)
            self.assertAlmostEqual(float(got["log_likelihood"].iloc[row]), expected, places=4)
            # With no spread across representations the bound is tight.
            self.assertAlmostEqual(
                float(got["log_likelihood_elbo"].iloc[row]), expected, places=4)
            self.assertAlmostEqual(float(got["surprisal"].iloc[row]), -expected, places=4)

    def test_the_bound_never_exceeds_the_importance_estimate(self):
        records = pd.DataFrame.from_records([
            _record(["Fe", "O", "Li"], ["1", "2", "4"], ["a", "b", "c"])])
        records.index.name = "index"
        got = score_gene_likelihood(records, self.trainer, permutation_samples=8, seed=0)
        self.assertLessEqual(
            float(got["log_likelihood_elbo"].iloc[0]),
            float(got["log_likelihood"].iloc[0]) + 1e-9)

    def test_a_scalar_regressor_is_refused(self):
        self.trainer.target = TargetClass.Scalar
        records = pd.DataFrame.from_records([_record(["Fe"], ["1"], ["a"])])
        with self.assertRaisesRegex(ValueError, "generative"):
            score_gene_likelihood(records, self.trainer)

    def test_space_group_prior_follows_the_sampled_distribution(self):
        records = pd.DataFrame.from_records([
            _record(["Fe"], ["1"], ["a"], space_group=1),
            _record(["Fe"], ["1"], ["a"], space_group=2)])
        prior = start_log_prior(self.trainer, records)
        self.assertAlmostEqual(float(prior.iloc[0]), math.log(0.75), places=6)
        self.assertAlmostEqual(float(prior.iloc[1]), math.log(0.25), places=6)


if __name__ == "__main__":
    unittest.main()
