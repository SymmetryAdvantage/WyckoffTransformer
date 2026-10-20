"""Tests for the hull reference built from LeMat-Bulk on the HuggingFace Hub."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import fsspec
import pandas as pd

from wyckoff_transformer.dataset_manifest import formation_energy_field, load_manifest
from wyckoff_transformer.formula_energy import hf_reference

SHA = "bf8f0b3112a0c522ccd7f90da9748e9240ee80a5"


class TestProvenance(unittest.TestCase):
    def test_known_revision_matches_the_data_store_manifest(self):
        # The data-store pipeline builds lemat_pbe_ehull from the same revision;
        # a reference built here must be accepted and refused exactly as that is.
        manifest = load_manifest("lemat-bulk-fixed")
        self.assertIn(SHA[:8], manifest.description)
        self.assertEqual(
            hf_reference.reference_field(hf_reference.DEFAULT_REPO, SHA),
            formation_energy_field(manifest, "lemat_pbe_ehull"))

    def test_bare_repo_takes_the_pinned_revision(self):
        self.assertEqual(hf_reference.parse_spec(hf_reference.DEFAULT_REPO),
                         (hf_reference.DEFAULT_REPO, SHA))
        self.assertEqual(hf_reference.parse_spec("someone/else@main"), ("someone/else", "main"))
        self.assertEqual(hf_reference.parse_spec("someone/else"), ("someone/else", None))

    def test_abbreviated_commit_is_known_and_a_branch_is_not(self):
        repo = hf_reference.DEFAULT_REPO
        self.assertIsNotNone(hf_reference.reference_field(repo, SHA[:8]))
        self.assertIsNone(hf_reference.reference_field(repo, "main"))
        self.assertIsNone(hf_reference.reference_field(repo, SHA[:4]))
        self.assertIsNone(hf_reference.reference_field("someone/else", SHA))


class TestLoad(unittest.TestCase):
    def test_builds_the_process_lemat_table_and_caches_it(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shards = tmp / "repo" / hf_reference.CONFIG
            shards.mkdir(parents=True)
            pd.DataFrame({
                "immutable_id": ["mp-1", "mp-2"],
                "species_at_sites": [["O", "Ba", "Ti", "O", "O"], ["Na", "Cl"]],
                "energy": [-40.0, -7.0],
                "forces": [[[0.0]], [[0.0]]],
            }).to_parquet(shards / "train-00000-of-00002.parquet")
            pd.DataFrame({
                "immutable_id": ["oqmd-3"],
                "species_at_sites": [["Na"]],
                "energy": [-1.3],
                "forces": [[[0.0]]],
            }).to_parquet(shards / "train-00001-of-00002.parquet")

            local = fsspec.filesystem("file")
            glob = local.glob
            hub = SimpleNamespace(glob=lambda pattern: glob(
                str(shards / Path(pattern).name)), open=local.open)
            api = SimpleNamespace(dataset_info=lambda repo, revision: SimpleNamespace(sha=SHA))
            with patch("huggingface_hub.HfApi", return_value=api), \
                    patch("huggingface_hub.HfFileSystem", return_value=hub):
                built = hf_reference.load_hf_reference(cache_dir=tmp / "cache")
                cached = hf_reference.load_hf_reference(cache_dir=tmp / "cache")

            self.assertEqual(list(built.index), ["mp-1", "mp-2", "oqmd-3"])
            self.assertEqual(list(built.columns), ["full_formula", "chemsys", "energy_corrected"])
            self.assertEqual(built.loc["mp-1", "full_formula"], "Ba1 O3 Ti1")
            self.assertEqual(built.loc["mp-1", "chemsys"], "Ba-O-Ti")
            self.assertEqual(built.loc["oqmd-3", "energy_corrected"], -1.3)
            pd.testing.assert_frame_equal(built, cached)
            self.assertTrue((tmp / "cache" / f"kazeevn__LeMat-Bulk-fixed@{SHA}.parquet").is_file())


if __name__ == "__main__":
    unittest.main()
