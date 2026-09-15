"""The probe must never report "nothing to resume" when it simply could not tell.

`scripts/platforms/aspire2a/train_in_pbs.sh` mints a fresh W&B run id on status 1, so conflating "W&B is
unreachable" with "this run has no checkpoint" is how a purged run loses its trained
epochs. Status 2 exists to keep those apart.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from wyckoff_transformer.cli.resume_probe import (
    AVAILABLE,
    NOTHING_TO_RESUME,
    UNKNOWN,
    local_checkpoint,
    main,
    wandb_checkpoint_exists,
)
from wyckoff_transformer.trainer import CHECKPOINT_FILENAME

RUN_ID = "probe123"


def _api_raising(message):
    """A `wandb.Api()` whose `.run()` fails with *message*."""
    api = MagicMock()
    api.run.side_effect = RuntimeError(message)
    return api


def _api_with_file(size):
    """A `wandb.Api()` whose run reports the checkpoint file at *size* bytes.

    Zero is how W&B reports a file the run never uploaded, which is why the probe
    reads absence off the size rather than expecting an error.
    """
    api = MagicMock()
    api.run.return_value.file.return_value = MagicMock(size=size)
    return api


class TestLocalCheckpoint(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.runs = Path(self._dir.name)

    def test_found(self):
        (self.runs / RUN_ID).mkdir()
        (self.runs / RUN_ID / CHECKPOINT_FILENAME).touch()
        self.assertIsNotNone(local_checkpoint(RUN_ID, self.runs))

    def test_absent(self):
        self.assertIsNone(local_checkpoint(RUN_ID, self.runs))


class TestWandbCheckpointExists(unittest.TestCase):
    def _call(self, api):
        with patch("wandb.Api", return_value=api):
            return wandb_checkpoint_exists(RUN_ID, "ent", "proj")

    def test_mirror_present(self):
        self.assertIs(self._call(_api_with_file(4096)), True)

    def test_run_exists_but_never_uploaded_the_checkpoint(self):
        self.assertIs(self._call(_api_with_file(0)), False)

    def test_a_run_wandb_never_saw_is_nothing_to_resume(self):
        """An id minted locally by a link that crashed before logging anything."""
        api = _api_raising(f"Could not find run <Run ent/proj/{RUN_ID} (not found)>")
        self.assertIs(self._call(api), False)

    def test_an_unreachable_api_is_unknown_not_absent(self):
        """The distinction the whole module exists for."""
        api = _api_raising("Max retries exceeded: connection refused")
        self.assertIsNone(self._call(api))

    def test_file_listing_failure_is_unknown(self):
        api = MagicMock()
        api.run.return_value.file.side_effect = RuntimeError("gateway timeout")
        self.assertIsNone(self._call(api))


class TestExitCodes(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.runs = Path(self._dir.name)

    def _main(self, api=None):
        argv = ["resume_probe", RUN_ID, "--runs-path", str(self.runs)]
        with patch("sys.argv", argv):
            if api is None:
                return main()
            with patch("wandb.Api", return_value=api):
                return main()

    def test_local_checkpoint_is_available(self):
        (self.runs / RUN_ID).mkdir()
        (self.runs / RUN_ID / CHECKPOINT_FILENAME).touch()
        self.assertEqual(self._main(), AVAILABLE)

    def test_wandb_mirror_is_available(self):
        self.assertEqual(self._main(_api_with_file(4096)), AVAILABLE)

    def test_no_state_anywhere_is_nothing_to_resume(self):
        self.assertEqual(self._main(_api_with_file(0)), NOTHING_TO_RESUME)

    def test_unreachable_wandb_is_unknown(self):
        api = _api_raising("connection refused")
        self.assertEqual(self._main(api), UNKNOWN)


if __name__ == "__main__":
    unittest.main()
