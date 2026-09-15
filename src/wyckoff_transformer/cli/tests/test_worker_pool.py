"""The supervised pool tells a broken worker from a failed trial.

Every test runs real spawned processes: what is being tested is how the pool
behaves when one of them dies, hangs or loses its GPU.
"""
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from wyckoff_transformer.cli.tests import _pool_tasks
from wyckoff_transformer.cli.worker_pool import (
    NOT_RUN_ERROR,
    WORKER_HUNG_ERROR,
    is_retryable_error,
    is_technical_error,
    run_supervised,
)


class _Run:
    """Collects what the pool reports back."""

    def __init__(self):
        self.results, self.failures = {}, {}

    def __call__(self, tasks, slots, mode, broken_at_start=(), **kwargs):
        flags = tempfile.TemporaryDirectory()
        try:
            report = run_supervised(
                [(i, (i, mode, flags.name)) for i in tasks],
                _pool_tasks.task,
                slots=slots,
                initializer=_pool_tasks.init,
                initargs=lambda counter, live: (counter, live, tuple(broken_at_start)),
                on_result=lambda key, result: self.results.__setitem__(key, result),
                on_failure=lambda key, status, error: self.failures.__setitem__(
                    key, (status, error)
                ),
                **kwargs,
            )
        finally:
            flags.cleanup()
        return report


class TestClassification(unittest.TestCase):
    def test_the_iapetus_error_is_technical_and_retryable(self):
        self.assertTrue(is_technical_error(_pool_tasks.CUDA_FAILURE))
        self.assertTrue(is_retryable_error(_pool_tasks.CUDA_FAILURE))

    def test_a_trial_failure_is_neither(self):
        for error in ("ValueError: the cell collapsed", None, float("nan"),
                      f"{WORKER_HUNG_ERROR}: exceeded 1800 s"):
            self.assertFalse(is_retryable_error(error), error)


class TestRunSupervised(unittest.TestCase):
    def test_a_poisoned_gpu_loses_no_trials(self):
        """The failure that cost ehull-ssops a third of its genes.

        Whether the bad card is also retired depends on whether it gets work
        again before the good one finishes, so that is tested on its own below.
        """
        run = _Run()
        report = run(range(6), ["good", "bad"], "cuda_on_bad", max_device_faults=2)
        self.assertEqual(run.failures, {})
        self.assertEqual(sorted(run.results), list(range(6)))
        self.assertTrue(all(row["device"] == "good" for row, _ in run.results.values()))
        self.assertEqual(report.unanswered, 0)

    def test_a_gpu_that_fails_to_start_does_not_blame_trials(self):
        """With one attempt per trial, any charge would turn into a failure."""
        run = _Run()
        report = run(range(4), ["good", "bad"], "ok", broken_at_start=("bad",),
                     max_device_faults=2, max_attempts=1)
        self.assertEqual(run.failures, {})
        self.assertEqual(sorted(run.results), list(range(4)))
        self.assertEqual(report.worker_faults, 0)

    def test_a_crashed_worker_does_not_lose_trials(self):
        run = _Run()
        report = run(range(4), ["cpu", "cpu"], "crash_once")
        self.assertEqual(run.failures, {})
        self.assertEqual(sorted(run.results), list(range(4)))
        self.assertGreaterEqual(report.pool_breaks, 1)

    def test_a_segfaulting_trial_does_not_use_up_its_neighbours_attempts(self):
        """Each crash takes down every running trial; only the culprit pays."""
        run = _Run()
        report = run(range(8), ["cpu"] * 3, "crash_zero", max_attempts=2)
        self.assertEqual(sorted(run.results), list(range(1, 8)))
        self.assertEqual(list(run.failures), [0])
        status, error = run.failures[0]
        self.assertIn("BrokenProcessPool", error)
        self.assertEqual(report.unanswered, 1)

    def test_a_hung_worker_is_killed_and_its_trial_retried(self):
        run = _Run()
        report = run([0], ["cpu"], "hang_once", task_timeout=0.5, hang_grace=3.0)
        self.assertEqual(run.failures, {})
        self.assertEqual(list(run.results), [0])
        self.assertEqual(report.hung_workers, 1)

    def test_a_trial_that_hangs_every_worker_is_recorded_as_a_timeout(self):
        """As the in-task limit would have recorded it: answered, not a hole."""
        run = _Run()
        report = run([0], ["cpu"], "hang_always", task_timeout=0.2, hang_grace=1.0,
                     max_attempts=1)
        status, error = run.failures[0]
        self.assertEqual(status, "timeout")
        self.assertTrue(error.startswith(WORKER_HUNG_ERROR))
        self.assertFalse(is_retryable_error(error))
        self.assertEqual(report.unanswered, 0)

    def test_a_trial_that_always_fails_its_device_is_given_up_on(self):
        run = _Run()
        report = run([0, 1], ["gpu"], "oom_always", max_attempts=2, max_device_faults=99)
        self.assertEqual(run.results, {})
        self.assertEqual(sorted(run.failures), [0, 1])
        for status, error in run.failures.values():
            self.assertEqual(status, "failed")
            self.assertTrue(is_retryable_error(error))
        self.assertEqual(report.unanswered, 2)

    def test_a_retired_last_device_leaves_retryable_holes(self):
        run = _Run()
        report = run(range(3), ["bad"], "cuda_on_bad", max_device_faults=1)
        self.assertEqual(report.retired_devices, ["bad"])
        self.assertEqual(sorted(run.failures), [0, 1, 2])
        for _, error in run.failures.values():
            self.assertTrue(is_retryable_error(error))
        self.assertTrue(any(e.startswith(NOT_RUN_ERROR) for _, e in run.failures.values()))
        self.assertEqual(report.unanswered, 3)

    def test_a_trial_that_raises_is_answered_once(self):
        run = _Run()
        report = run([0], ["cpu"], "raise")
        self.assertEqual(run.failures, {0: ("failed", "ValueError: the cell collapsed")})
        self.assertEqual(report.rounds, 1)
        self.assertEqual(report.retried, 0)


class TestRequireComplete(unittest.TestCase):
    """The score stage must not publish a funnel with holes in it."""

    def _run_dir(self, rows):
        from wyckoff_transformer.cli.protocol import RELAXATIONS_FILE

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with (Path(tmp.name) / RELAXATIONS_FILE).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "trial", "status", "error"])
            writer.writeheader()
            writer.writerows(rows)
        return Path(tmp.name)

    def test_a_gpu_failure_is_refused(self):
        from wyckoff_transformer.cli.protocol import (
            TRIAL_LOGS, IncompleteStageError, require_complete,
        )

        out = self._run_dir([
            {"index": 0, "trial": 0, "status": "ok", "error": ""},
            {"index": 1, "trial": 0, "status": "failed", "error": _pool_tasks.CUDA_FAILURE},
        ])
        with self.assertRaises(IncompleteStageError):
            require_complete(SimpleNamespace(output_dir=out), TRIAL_LOGS)
        require_complete(SimpleNamespace(output_dir=out, allow_incomplete=True), TRIAL_LOGS)

    def test_holes_beyond_the_limit_are_not_this_runs_business(self):
        from wyckoff_transformer.cli.protocol import TRIAL_LOGS, require_complete

        out = self._run_dir([
            {"index": 700, "trial": 0, "status": "failed", "error": _pool_tasks.CUDA_FAILURE},
        ])
        require_complete(SimpleNamespace(output_dir=out, limit=80), TRIAL_LOGS)

    def test_a_resumed_retry_answers_the_trial(self):
        from wyckoff_transformer.cli.protocol import TRIAL_LOGS, require_complete

        out = self._run_dir([
            {"index": 1, "trial": 0, "status": "failed", "error": _pool_tasks.CUDA_FAILURE},
            {"index": 1, "trial": 0, "status": "ok", "error": ""},
        ])
        require_complete(SimpleNamespace(output_dir=out), TRIAL_LOGS)

    def test_a_genuine_failure_is_not_a_hole(self):
        from wyckoff_transformer.cli.protocol import TRIAL_LOGS, require_complete

        out = self._run_dir([
            {"index": 1, "trial": 0, "status": "failed", "error": "ValueError: collapsed"},
        ])
        require_complete(SimpleNamespace(output_dir=out), TRIAL_LOGS)


if __name__ == "__main__":
    unittest.main()
