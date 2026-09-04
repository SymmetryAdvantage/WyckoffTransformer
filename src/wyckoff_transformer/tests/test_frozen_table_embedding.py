"""The frozen-table cascade embedding.

A field declared as ``{frozen_table: <name>}`` carries an integer id per token and the
model expands it through a lookup table shipped as package data. The point of the
indirection is storage, not behaviour: the encoder must see exactly what it would see
if the same vectors were fed in directly as a ``pass_through_vector`` field.
"""
import functools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from ..cascade.model import CascadeEmbedding, FrozenTableEmbedding
from ..wyckoff_processor import load_frozen_table, save_frozen_table


class TestFrozenTableEmbedding(unittest.TestCase):
    N_IDS = 7
    N_FEATURES = 5

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        rng = np.random.default_rng(0)
        self.table = rng.integers(0, 2, (self.N_IDS, self.N_FEATURES)).astype(np.float32)
        save_frozen_table("test_field", self.table, engineers_dir=Path(self.tmp.name))
        # The model looks the table up in the package's engineers directory; point it at
        # the temporary one instead of writing into the installed package.
        patcher = patch(
            "wyckoff_transformer.cascade.model.load_frozen_table",
            functools.partial(load_frozen_table, engineers_dir=Path(self.tmp.name)))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _cascade(self, second):
        return ((4, 3, 0, True), (self.N_IDS, second, None, False))

    def test_matches_the_pass_through_vector_path(self):
        dense = CascadeEmbedding(self._cascade({"pass_through_vector": self.N_FEATURES}))
        table = CascadeEmbedding(self._cascade({"frozen_table": "test_field"}))
        table.embeddings[0].load_state_dict(dense.embeddings[0].state_dict())

        self.assertEqual(dense.total_embedding_dim, table.total_embedding_dim)
        ids = torch.tensor([[0, 3, 6], [2, 2, 5]])
        tokens = torch.tensor([[0, 1, 2], [3, 0, 1]])
        vectors = torch.from_numpy(self.table[ids.numpy()])

        self.assertTrue(torch.equal(
            dense([tokens, vectors]), table([tokens, ids])))

    def test_table_is_a_buffer_not_a_parameter(self):
        embedding = FrozenTableEmbedding(torch.from_numpy(self.table))
        self.assertEqual(list(embedding.parameters()), [])
        self.assertIn("table", dict(embedding.named_buffers()))
        # Buffers are checkpointed, so a reloaded model does not depend on the package
        # data still being byte-identical.
        self.assertIn("table", embedding.state_dict())

    def test_survives_an_optimiser_step(self):
        cascade = self._cascade({"frozen_table": "test_field"})
        embedding = CascadeEmbedding(cascade)
        before = embedding.embeddings[1].table.clone()
        optimiser = torch.optim.AdamW(embedding.parameters(), lr=1.)
        output = embedding([torch.tensor([[0, 1]]), torch.tensor([[3, 4]])])
        output.sum().backward()
        optimiser.step()
        self.assertTrue(torch.equal(before, embedding.embeddings[1].table))

    def test_rejects_a_table_too_small_for_the_token_count(self):
        with self.assertRaisesRegex(ValueError, "too few"):
            CascadeEmbedding(((4, 3, 0, True),
                              (self.N_IDS + 1, {"frozen_table": "test_field"}, None, False)))

    def test_rejects_a_non_2d_table(self):
        with self.assertRaisesRegex(ValueError, "2-D"):
            FrozenTableEmbedding(torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
