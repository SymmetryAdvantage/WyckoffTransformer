"""A resumed migration cannot discard successful retries or requeue them twice."""

import csv
import json

from scripts.platforms.zeus.run_mlip_bias_migration_gpu1 import (
    prepare_retry,
    recovery_remaining,
)


def write_rows(directory, rows):
    with (directory / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["gene", "trial", "status", "error"])
        writer.writeheader()
        writer.writerows(rows)


def original_rows():
    return [dict(gene=str(i), trial="0", status="failed" if i == 5 else "ok",
                 error="OOM\nsecond line" if i == 5 else "") for i in range(2698)]


def test_retry_preserves_new_success_and_failure_after_resume(tmp_path):
    rows = original_rows()
    write_rows(tmp_path, rows)
    source = {"artifact": "source:v1"}
    plan = prepare_retry(tmp_path, source)
    assert len(plan["original_failures"]) == 1
    assert (tmp_path / "zeus_before_failed_retry.csv").is_file()
    for status in ["ok", "failed"]:
        rows[5].update(status=status, error="" if status == "ok" else "new failure")
        write_rows(tmp_path, rows)
        expected = (tmp_path / "trials.csv").read_bytes()
        prepare_retry(tmp_path, source)
        assert (tmp_path / "trials.csv").read_bytes() == expected


def test_retry_recovers_crash_between_filter_and_prepared_marker(tmp_path):
    rows = original_rows()
    write_rows(tmp_path, rows)
    source = {"artifact": "source:v1"}
    plan = prepare_retry(tmp_path, source)
    plan["phase"] = "planned"
    (tmp_path / "zeus_failed_retry.json").write_text(json.dumps(plan))
    prepare_retry(tmp_path, source)
    with (tmp_path / "trials.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [r for r in rows if r["status"] == "ok"]


def test_completed_failed_recovery_attempt_is_not_requeued(tmp_path):
    rows = original_rows()
    rows[5]["error"] = "CUDA out of memory"
    write_rows(tmp_path, rows)
    assert recovery_remaining(tmp_path) == 1
    settings = {"history": [{"relax_timeout_s": 3600, "original_failures": [rows[5]],
                              "completed": [[5, 0]]}]}
    (tmp_path / "recovery_settings.json").write_text(json.dumps(settings))
    assert recovery_remaining(tmp_path) == 0
