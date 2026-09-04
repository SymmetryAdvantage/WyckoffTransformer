"""Generation with a frozen-table auxiliary field.

``site_symmetry_ops_id`` is not predicted: it is the last cascade field, and the
generator fills it from (spacegroup_number, site_symmetries) through its engineer after
each cascade step. The model therefore sees the site symmetry's operations vector for a
token in the same step it sees the token itself.
"""
import unittest

import torch

from ..generator import WyckoffGenerator
from ..wyckoff_processor import FeatureEngineer

N_SS = 3
MASK_ID, STOP_ID, PAD_ID = 6, 7, 8
# (spacegroup one-hot tuple, site symmetry token) -> ops id
OPS_IDS = {
    ((1, 0), 0): 0,
    ((1, 0), 1): 1,
    ((1, 0), 2): 2,
    ((0, 1), 0): 3,
    ((0, 1), 1): 4,
    ((0, 1), 2): 5,
}


class ConstantModel(torch.nn.Module):
    """Always names the site symmetry given by the structure's index."""
    def __init__(self, choice):
        super().__init__()
        self.choice = choice
        self.seen_ops_ids = []

    def forward(self, start, cascade, padding_mask, prediction_head, cond=None):
        # The auxiliary column of every fully known token, as the model receives it.
        self.seen_ops_ids.append(cascade[1][:, :-1].clone())
        logits = torch.full((start.size(0), N_SS), -30.)
        logits[torch.arange(start.size(0)), self.choice] = 30.
        return logits


class TestGenerationWithOpsId(unittest.TestCase):
    def setUp(self):
        self.engineer = FeatureEngineer(
            OPS_IDS, ("spacegroup_number", "site_symmetries"),
            name="site_symmetry_ops_id",
            mask_token=MASK_ID, stop_token=STOP_ID, pad_token=PAD_ID,
            default_value=PAD_ID)
        self.choice = torch.tensor([2, 0])
        self.model = ConstantModel(self.choice)
        self.generator = WyckoffGenerator(
            model=self.model,
            cascade_order=("site_symmetries", "site_symmetry_ops_id"),
            cascade_is_target={"site_symmetries": True, "site_symmetry_ops_id": False},
            token_engineers={"site_symmetry_ops_id": self.engineer},
            masks={"site_symmetries": N_SS, "site_symmetry_ops_id": MASK_ID},
            max_sequence_len=3)

    def test_an_engineer_reading_a_later_field_is_rejected(self):
        """The fill only works if the fields it reads are already decided for this token;
        the opposite order would silently feed the model MASK."""
        generator = WyckoffGenerator(
            model=self.model,
            cascade_order=("site_symmetry_ops_id", "site_symmetries"),
            cascade_is_target={"site_symmetries": True, "site_symmetry_ops_id": False},
            token_engineers={"site_symmetry_ops_id": self.engineer},
            masks={"site_symmetries": N_SS, "site_symmetry_ops_id": MASK_ID},
            max_sequence_len=3)
        with self.assertRaisesRegex(ValueError, "generated later in the cascade"):
            with torch.no_grad():
                generator.generate_tensors(
                    torch.tensor([[1., 0.], [0., 1.]]), compute_validity=False)

    def test_auxiliary_field_is_filled_from_the_engineer(self):
        start = torch.tensor([[1., 0.], [0., 1.]])
        with torch.no_grad():
            generated = self.generator.generate_tensors(start, compute_validity=False)

        site_symmetries, ops_ids = generated
        self.assertTrue(torch.equal(
            site_symmetries, self.choice.unsqueeze(1).expand(2, 3)))
        expected = torch.tensor([
            [OPS_IDS[((1, 0), 2)]] * 3,
            [OPS_IDS[((0, 1), 0)]] * 3])
        self.assertTrue(torch.equal(ops_ids, expected),
                        f"expected {expected.tolist()}, got {ops_ids.tolist()}")

    def test_the_model_sees_the_filled_ids_of_earlier_tokens(self):
        start = torch.tensor([[1., 0.], [0., 1.]])
        with torch.no_grad():
            self.generator.generate_tensors(start, compute_validity=False)

        # Third token: the two preceding ones are known, and carry real ids rather than
        # the MASK the tensors were initialised with.
        last_seen = self.model.seen_ops_ids[-1]
        self.assertEqual(tuple(last_seen.shape), (2, 2))
        self.assertFalse((last_seen == MASK_ID).any(),
                         "known tokens must carry engineer-filled ids, not MASK")
        self.assertTrue(torch.equal(
            last_seen,
            torch.tensor([[OPS_IDS[((1, 0), 2)]] * 2, [OPS_IDS[((0, 1), 0)]] * 2])))


if __name__ == "__main__":
    unittest.main()
