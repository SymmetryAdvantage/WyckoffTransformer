"""The training run records which dataset cache it was tokenised from."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from pymatgen.core import Element

from wyckoff_transformer import dataset_cache as dc
from wyckoff_transformer.trainer import log_dataset_cache_provenance
from wyckoff_transformer.trainer import logger as trainer_logger


def a_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_symmetries": [["m"]],
            "elements": [[Element("Na")]],
            "spacegroup_number": [8],
        },
        index=pd.Index(["mp-1"], name="material_id"),
    )


class TestProvenanceLogging(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache = Path(tmp.name) / "toy"
        self.run = MagicMock()
        for target, value in (("wandb", MagicMock(run=self.run)),
                              ("dataset_cache_dir", lambda name: self.cache)):
            patcher = patch(f"wyckoff_transformer.trainer.{target}", value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.wandb = __import__("wyckoff_transformer.trainer", fromlist=["wandb"]).wandb

    def logged(self):
        self.wandb.config.update.assert_called_once()
        payload, kwargs = self.wandb.config.update.call_args
        self.assertTrue(kwargs["allow_val_change"])
        return payload[0]["dataset_cache"]

    def test_the_build_record_reaches_the_run_config(self):
        build = dc.provenance("wyformer-cache-dataset", max_sites=61)
        dc.save_cache({"train": a_frame(), "val": a_frame()}, self.cache, build)
        log_dataset_cache_provenance("toy")
        self.assertEqual(self.logged(), build)

    def test_splits_built_apart_are_all_reported(self):
        dc.save_split(a_frame(), self.cache, "train", dc.provenance("first"))
        dc.save_split(a_frame(), self.cache, "val", dc.provenance("second"))
        log_dataset_cache_provenance("toy")
        self.assertEqual({split: record["tool"] for split, record in self.logged().items()},
                         {"train": "first", "val": "second"})

    def test_a_cache_that_records_nothing_is_logged_as_nothing(self):
        # "the cache does not say" must stay distinguishable from "nobody looked".
        dc.save_cache({"train": a_frame()}, self.cache)
        log_dataset_cache_provenance("toy")
        self.assertIsNone(self.logged())

    def test_a_missing_cache_does_not_fail_the_run(self):
        # Tensors outlive the splits they came from; a cluster trains that way.
        with self.assertLogs(trainer_logger, "WARNING"):
            log_dataset_cache_provenance("toy")
        self.assertIsNone(self.logged())

    def test_nothing_is_logged_without_a_run(self):
        dc.save_cache({"train": a_frame()}, self.cache, dc.provenance("a-tool"))
        with patch("wyckoff_transformer.trainer.wandb", MagicMock(run=None)) as no_run:
            log_dataset_cache_provenance("toy")
        no_run.config.update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
