"""A model that generates the space group rather than being handed it.

`predict_start` adds p(start | cond) to a model whose sequence factor p(sites | start, cond)
is untouched. Two things make the comparison with its parent meaningful and are pinned here:
the shared weights are initialised exactly as the parent's are, and the start token is
mapped to a class and back without loss. The rest checks the loss and sampling plumbing.
"""
import unittest
from types import SimpleNamespace

import torch

from wyckoff_transformer.cascade.model import CascadeTransformer
from wyckoff_transformer.generator import WyckoffGenerator
from wyckoff_transformer.trainer import WyckoffTrainer

N_START = 5
CONDITION_DIM = 3


def build_model(predict_start: bool, start_type: str = "one_hot", condition_dim=CONDITION_DIM):
    return CascadeTransformer(
        start_type=start_type,
        n_start=N_START,
        cascade=((6, 4, 0, True), (5, 4, 0, True)),
        token_aggregation=None,
        aggregate_after_encoder=False,
        include_start_in_aggregation=False,
        aggregation_inclsion="None",
        concat_token_counts=False,
        concat_token_presence=True,
        num_fully_connected_layers=3,
        mixer_layers=1,
        outputs="token_scores",
        perceptron_shape="pyramid",
        TransformerEncoderLayer_args={"nhead": 2, "dim_feedforward": 16, "dropout": 0.0},
        TransformerEncoder_args={"num_layers": 2, "enable_nested_tensor": False},
        learned_positional_encoding_max_size=0,
        learned_positional_encoding_only_masked=True,
        condition_dim=condition_dim,
        predict_start=predict_start,
        n_start_classes=N_START if predict_start else None,
    )


class TestModel(unittest.TestCase):
    def test_shared_weights_are_initialised_as_without_it(self):
        # The new parameters are created after every existing one, so under one seed the
        # two models start from the same site model -- the premise of comparing them.
        torch.manual_seed(0)
        parent = build_model(predict_start=False)
        torch.manual_seed(0)
        child = build_model(predict_start=True)
        child_state = child.state_dict()
        for name, value in parent.state_dict().items():
            self.assertTrue(torch.equal(value, child_state[name]), name)
        extra = set(child_state) - set(parent.state_dict())
        self.assertTrue(extra)
        self.assertTrue(all(name.startswith(("start_query", "start_prediction_head"))
                            for name in extra), extra)

    def test_forward_start_shape_and_gradients(self):
        torch.manual_seed(0)
        model = build_model(predict_start=True)
        cond = torch.randn(7, CONDITION_DIM)
        logits = model.forward_start(7, cond=cond)
        self.assertEqual(logits.shape, (7, N_START))
        torch.nn.functional.cross_entropy(logits, torch.randint(0, N_START, (7,))).backward()
        self.assertIsNotNone(model.start_query.grad)
        self.assertGreater(model.start_query.grad.abs().sum().item(), 0)

    def test_forward_start_reads_the_conditioning(self):
        torch.manual_seed(0)
        model = build_model(predict_start=True)
        # AdaLN is zero-initialised, so give it something to say first.
        for layer in model.transformer_encoder.layers:
            torch.nn.init.normal_(layer.adaLN_modulation1.weight)
        a = model.forward_start(1, cond=torch.zeros(1, CONDITION_DIM))
        b = model.forward_start(1, cond=torch.ones(1, CONDITION_DIM))
        self.assertFalse(torch.allclose(a, b))

    def test_unconditional_model(self):
        model = build_model(predict_start=True, condition_dim=None)
        self.assertEqual(model.forward_start(3).shape, (3, N_START))

    def test_refused_when_not_built_for_it(self):
        with self.assertRaises(ValueError):
            build_model(predict_start=False).forward_start(2, cond=torch.zeros(2, CONDITION_DIM))


def skeleton_trainer(start_type: str):
    trainer = WyckoffTrainer.__new__(WyckoffTrainer)
    trainer.model = SimpleNamespace(start_type=start_type)
    trainer.predict_start = True
    trainer.start_name = "spacegroup_number"
    trainer.cascade_order = ("elements", "site_symmetries")
    trainer.cascade_target_indices = (0, 1)
    # Three space groups, encoded as distinct 0/1 vectors.
    trainer.start_class_vectors = torch.tensor(
        [[1., 0., 0., 1.], [0., 1., 0., 1.], [0., 0., 1., 0.]])
    return trainer


class TestStartClasses(unittest.TestCase):
    def test_one_hot_round_trip(self):
        trainer = skeleton_trainer("one_hot")
        classes = torch.tensor([2, 0, 0, 1, 2])
        tokens = trainer.start_classes_to_tokens(classes)
        self.assertTrue(torch.equal(trainer.start_tokens_to_classes(tokens), classes))

    def test_an_unknown_vector_is_refused(self):
        trainer = skeleton_trainer("one_hot")
        with self.assertRaises(ValueError):
            trainer.start_tokens_to_classes(torch.tensor([[1., 1., 1., 1.]]))

    def test_categorial_is_the_token(self):
        trainer = skeleton_trainer("categorial")
        tokens = torch.tensor([3, 1, 4])
        self.assertTrue(torch.equal(trainer.start_tokens_to_classes(tokens), tokens))

    def test_loss_fields_put_the_start_first(self):
        trainer = skeleton_trainer("one_hot")
        self.assertEqual(trainer.loss_field_names,
                         ("spacegroup_number", "elements", "site_symmetries"))
        trainer.predict_start = False
        self.assertEqual(trainer.loss_field_names, ("elements", "site_symmetries"))


class TestSampling(unittest.TestCase):
    def test_samples_follow_the_head(self):
        torch.manual_seed(0)
        model = build_model(predict_start=True)
        with torch.no_grad():
            final = model.start_prediction_head[-1]
            final.weight.zero_()
            final.bias.copy_(torch.tensor([-50., -50., 50., -50., -50.]))
        generator = WyckoffGenerator(model, ("a", "b"), {"a": True, "b": True}, {}, {}, 4)
        drawn = generator.sample_start_classes(64, cond=torch.randn(64, CONDITION_DIM))
        self.assertEqual(drawn.shape, (64,))
        self.assertTrue(bool((drawn == 2).all()))

    def test_samples_follow_guidance(self):
        torch.manual_seed(0)
        model = build_model(predict_start=True, condition_dim=2)
        def mock_forward_start(batch_size, cond=None):
            out = torch.zeros(batch_size, N_START)
            is_uncond = cond[:, 1] == 1.0
            out[~is_uncond, 1] = 50.0
            out[is_uncond, 3] = 50.0
            return out
        model.forward_start = mock_forward_start
        generator = WyckoffGenerator(model, ("a", "b"), {"a": True, "b": True}, {}, {}, 4)
        cond = torch.tensor([[1.0, 0.0]]).repeat(32, 1)
        uncond = torch.tensor([[0.0, 1.0]]).repeat(32, 1)
        drawn_cond = generator.sample_start_classes(32, cond=cond, uncond=uncond, guidance_scale=1.0)
        self.assertTrue(bool((drawn_cond == 1).all()))
        drawn_uncond = generator.sample_start_classes(32, cond=cond, uncond=uncond, guidance_scale=0.0)
        self.assertTrue(bool((drawn_uncond == 3).all()))
        drawn_guided = generator.sample_start_classes(32, cond=cond, uncond=uncond, guidance_scale=2.0)
        self.assertTrue(bool((drawn_guided == 1).all()))


if __name__ == "__main__":
    unittest.main()
