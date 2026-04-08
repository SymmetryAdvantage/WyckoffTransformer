import gzip
import json
import pickle
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from wyckoff_transformer.cli.generate import (
    _decode_space_groups,
    _resolve_sg_cache_path,
    _select_tensor_and_tokeniser,
    main,
    prepare_start_tensor_from_cache,
)


# ---------------------------------------------------------------------------
# _resolve_sg_cache_path
# ---------------------------------------------------------------------------

class TestResolveSgCachePath(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_exact_name_found(self):
        (self.cache_root / "mp-20").mkdir()
        result = _resolve_sg_cache_path("mp-20", cache_root=self.cache_root)
        self.assertEqual(result, self.cache_root / "mp-20")

    def test_hyphen_to_underscore_fallback(self):
        (self.cache_root / "mp_20").mkdir()
        result = _resolve_sg_cache_path("mp-20", cache_root=self.cache_root)
        self.assertEqual(result, self.cache_root / "mp_20")

    def test_exact_takes_priority_over_underscore(self):
        (self.cache_root / "mp-20").mkdir()
        (self.cache_root / "mp_20").mkdir()
        result = _resolve_sg_cache_path("mp-20", cache_root=self.cache_root)
        self.assertEqual(result, self.cache_root / "mp-20")

    def test_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            _resolve_sg_cache_path("nonexistent", cache_root=self.cache_root)

    def test_no_hyphen_no_fallback(self):
        # Name without hyphen: only exact match attempted
        with self.assertRaises(FileNotFoundError):
            _resolve_sg_cache_path("mp20", cache_root=self.cache_root)


# ---------------------------------------------------------------------------
# _select_tensor_and_tokeniser
# ---------------------------------------------------------------------------

class TestSelectTensorAndTokeniser(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_path = Path(self._tmp.name)
        (self.cache_path / "tensors").mkdir()
        (self.cache_path / "tokenisers").mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _write_pair(self, stem):
        (self.cache_path / "tensors" / f"{stem}.pt").write_bytes(b"")
        (self.cache_path / "tokenisers" / f"{stem}.pkl.gz").write_bytes(b"")

    def test_finds_matching_pair(self):
        self._write_pair("v1")
        tensor_path, tokeniser_path = _select_tensor_and_tokeniser(self.cache_path)
        self.assertEqual(tensor_path.name, "v1.pt")
        self.assertEqual(tokeniser_path.name, "v1.pkl.gz")

    def test_skips_tensor_without_tokeniser(self):
        (self.cache_path / "tensors" / "orphan.pt").write_bytes(b"")
        self._write_pair("v2")
        _, tokeniser_path = _select_tensor_and_tokeniser(self.cache_path)
        self.assertEqual(tokeniser_path.name, "v2.pkl.gz")

    def test_missing_tensors_dir_raises(self):
        import shutil
        shutil.rmtree(self.cache_path / "tensors")
        with self.assertRaises(FileNotFoundError):
            _select_tensor_and_tokeniser(self.cache_path)

    def test_no_pair_raises(self):
        with self.assertRaises(FileNotFoundError):
            _select_tensor_and_tokeniser(self.cache_path)


# ---------------------------------------------------------------------------
# _decode_space_groups
# ---------------------------------------------------------------------------

class TestDecodeSpaceGroups(unittest.TestCase):
    def _make_enum_tokeniser(self, mapping):
        """Simulate EnumeratingTokeniser with a to_token dict."""
        tok = SimpleNamespace(to_token=mapping)
        return tok

    def test_1d_tensor_counts(self):
        tok = self._make_enum_tokeniser({0: 1, 1: 225, 2: 225})
        tensor = torch.tensor([0, 1, 2, 1])
        counts = _decode_space_groups(tensor, tok)
        self.assertEqual(counts[1], 1)
        self.assertEqual(counts[225], 3)

    def test_2d_tensor_counts(self):
        sg_numbers = [1, 2, 225]
        encoded = torch.eye(3, dtype=torch.float32)

        tok = MagicMock()
        tok.keys.return_value = sg_numbers
        tok.encode_spacegroups.return_value = encoded

        # Two rows of sg=1, one of sg=225
        start_tensor = torch.stack([encoded[0], encoded[0], encoded[2]])
        counts = _decode_space_groups(start_tensor, tok)
        self.assertEqual(counts[1], 2)
        self.assertEqual(counts[225], 1)
        self.assertNotIn(2, counts)

    def test_2d_unknown_row_raises(self):
        sg_numbers = [1]
        encoded = torch.eye(1, dtype=torch.float32)

        tok = MagicMock()
        tok.keys.return_value = sg_numbers
        tok.encode_spacegroups.return_value = encoded

        bad_tensor = torch.tensor([[0.5, 0.5]])
        with self.assertRaises(ValueError, msg="unknown space group encoding"):
            _decode_space_groups(bad_tensor, tok)

    def test_2d_no_encode_spacegroups_raises(self):
        tok = SimpleNamespace()  # no encode_spacegroups
        tensor = torch.zeros((2, 3))
        with self.assertRaises(ValueError):
            _decode_space_groups(tensor, tok)


# ---------------------------------------------------------------------------
# prepare_start_tensor_from_cache
# ---------------------------------------------------------------------------

class TestPrepareStartTensorFromCache(unittest.TestCase):
    """Integration test for prepare_start_tensor_from_cache using temp files."""

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_root = Path(self._tmp.name)
        dataset_dir = self.cache_root / "test-ds"
        (dataset_dir / "tensors").mkdir(parents=True)
        (dataset_dir / "tokenisers").mkdir()

        # Build a minimal cached tensors dict: train split with 1D start tokens
        # Token index 0 -> sg 225, index 1 -> sg 1
        cached_tensors = {
            "train": {"spacegroup_number": torch.tensor([0, 0, 1])},
        }
        torch.save(cached_tensors, dataset_dir / "tensors" / "v1.pt")

        # Source tokeniser: EnumeratingTokeniser-style with to_token
        source_tok = SimpleNamespace(to_token={0: 225, 1: 1})
        with gzip.open(dataset_dir / "tokenisers" / "v1.pkl.gz", "wb") as f:
            pickle.dump({"spacegroup_number": source_tok}, f)
            pickle.dump({}, f)  # token engineers placeholder

    def tearDown(self):
        self._tmp.cleanup()

    def _make_trainer(self, start_type, target_tok):
        trainer = MagicMock()
        trainer.start_name = "spacegroup_number"
        trainer.device = torch.device("cpu")
        trainer.model.start_type = start_type
        trainer.train_dataset.start_tokens = torch.zeros(1, dtype=torch.int64)
        trainer.tokenisers = {"spacegroup_number": target_tok}
        return trainer

    def test_categorial_output_shape(self):
        # Target tokeniser maps sg -> token index; supports __contains__ and __getitem__
        target_tok = {225: 0, 1: 1}
        trainer = self._make_trainer("categorial", target_tok)

        result = prepare_start_tensor_from_cache(
            trainer, "test-ds", n_samples=10, cache_root=self.cache_root
        )
        self.assertEqual(result.shape, (10,))
        self.assertEqual(result.dtype, torch.int64)
        # All sampled tokens must be valid (0 or 1)
        self.assertTrue(((result == 0) | (result == 1)).all())

    def test_one_hot_output_shape(self):
        encoded = torch.eye(2, dtype=torch.float32)
        target_tok = MagicMock()
        # __contains__: both 225 and 1 are in tokeniser
        target_tok.__contains__ = lambda self_, sg: sg in (225, 1)
        target_tok.encode_spacegroups.return_value = encoded.repeat(5, 1)[:10]

        trainer = self._make_trainer("one_hot", target_tok)
        trainer.train_dataset.start_tokens = torch.zeros((1, 2), dtype=torch.float32)

        result = prepare_start_tensor_from_cache(
            trainer, "test-ds", n_samples=10, cache_root=self.cache_root
        )
        self.assertEqual(result.shape[0], 10)

    def test_missing_start_field_raises(self):
        target_tok = {225: 0}
        trainer = self._make_trainer("categorial", target_tok)
        trainer.start_name = "nonexistent_field"

        with self.assertRaises(ValueError, msg="missing in cached tokenisers"):
            prepare_start_tensor_from_cache(
                trainer, "test-ds", n_samples=5, cache_root=self.cache_root
            )

    def test_no_overlapping_sgs_raises(self):
        # Target tokeniser contains only sg=999, which is not in source data
        target_tok = {999: 0}
        trainer = self._make_trainer("categorial", target_tok)

        with self.assertRaises(ValueError, msg="None of the space groups"):
            prepare_start_tensor_from_cache(
                trainer, "test-ds", n_samples=5, cache_root=self.cache_root
            )


# ---------------------------------------------------------------------------
# main() — argument parsing and output validation
# ---------------------------------------------------------------------------

class TestMain(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_non_json_gz_output_raises(self):
        out = str(self.tmp_path / "out.json")
        sys.argv = ["wyformer-generate", out, "--model-path", "/fake"]
        with self.assertRaises(ValueError, msg="Output file must be a .json.gz file"):
            main()

    def test_update_wandb_without_wandb_run_raises(self):
        out = str(self.tmp_path / "out.json.gz")
        with self.assertRaises(SystemExit):
            sys.argv = ["wyformer-generate", out, "--hf-model", "x/y", "--update-wandb"]
            main()

    @patch("wyckoff_transformer.cli.generate.WyckoffTrainer")
    def test_hf_model_writes_output(self, MockTrainer):
        out = self.tmp_path / "out.json.gz"
        mock_trainer = MockTrainer.from_huggingface.return_value
        mock_trainer.generate_structures.return_value = [{"a": 1}] * 1000

        sys.argv = [
            "wyformer-generate", str(out),
            "--hf-model", "fake/model",
            "--initial-n-samples", "1000",
            "--firm-n-samples", "1",
        ]
        main()

        self.assertTrue(out.exists())
        with gzip.open(out, "rt") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)

    @patch("wyckoff_transformer.cli.generate.WyckoffTrainer")
    def test_not_enough_structures_raises(self, MockTrainer):
        out = self.tmp_path / "out.json.gz"
        mock_trainer = MockTrainer.from_huggingface.return_value
        mock_trainer.generate_structures.return_value = [{"a": 1}] * 5

        sys.argv = [
            "wyformer-generate", str(out),
            "--hf-model", "fake/model",
            "--initial-n-samples", "10",
            "--firm-n-samples", "100",
        ]
        with self.assertRaises(ValueError, msg="Not enough valid structures"):
            main()


if __name__ == "__main__":
    unittest.main()
