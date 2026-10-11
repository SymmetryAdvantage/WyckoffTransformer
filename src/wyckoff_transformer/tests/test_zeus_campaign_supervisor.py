"""Campaign recovery must be bounded and respect scientific completion/loans."""

import json

from scripts.platforms.zeus import supervise_mlip_bias_campaign as supervisor


def test_no_progress_crash_budget_is_bounded_and_new_saved_progress_resets_it():
    record = {}
    supervisor.refresh_budget(record, "saved-ledger-before-native-crash")
    for _ in range(3):
        assert supervisor.spend_restart(record)
        supervisor.refresh_budget(record, "saved-ledger-before-native-crash")
    assert not supervisor.spend_restart(record)
    assert record["review_required"]
    supervisor.refresh_budget(record, "newly-saved-trial-and-recovery-journal")
    assert not record["review_required"]
    assert supervisor.spend_restart(record)


def configure_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor, "STUDY", tmp_path)
    monkeypatch.setattr(supervisor, "WORK", tmp_path / "zeus_migration")
    monkeypatch.setattr(supervisor, "METRICS", tmp_path / "metrics")
    (tmp_path / "zeus_migration").mkdir()
    (tmp_path / "metrics").mkdir()
    (tmp_path / "setup").mkdir()


def test_gpu0_loan_blocks_both_gpu0_queues_without_pausing_gpu1(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)
    loan = tmp_path / "setup/gpu0_loan_20261007.json"
    gpu0_scoring = {**supervisor.JOBS["mlip-updated-genbench"], "gpu": 0}
    gpu1_scoring = {**supervisor.JOBS["mlip-updated-genbench"], "gpu": 1}
    loan.write_text(json.dumps({"phase": "gpu0_lent"}))
    assert supervisor.paused(supervisor.JOBS["mlip-migration-gpu0"])
    assert supervisor.paused(gpu0_scoring)
    assert not supervisor.paused(gpu1_scoring)
    assert not supervisor.paused(supervisor.JOBS["mlip-recovery-gpu1"])
    loan.write_text(json.dumps({"phase": "resumed"}))
    assert not supervisor.paused(supervisor.JOBS["mlip-migration-gpu0"])
    (tmp_path / "zeus_gpu0.stop").touch()
    assert supervisor.paused(supervisor.JOBS["mlip-migration-gpu0"])
    assert supervisor.paused(gpu0_scoring)
    assert not supervisor.paused(gpu1_scoring)
    (tmp_path / "zeus_genbench_gpu1.stop").touch()
    assert supervisor.paused(gpu1_scoring)


def test_full_ledger_is_not_a_completed_recovery_queue(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)
    for arm in supervisor.ASSIGNMENTS[0]:
        path = supervisor.WORK / arm
        path.mkdir()
        (path / "trials.csv").write_text("gene,trial\n" + "1,0\n" * 2698)
    job = supervisor.JOBS["mlip-migration-gpu0"]
    assert not supervisor.job_done(job)
    for arm in supervisor.ASSIGNMENTS[0]:
        (supervisor.WORK / arm / "zeus_relaxation_complete.json").write_text(
            json.dumps({"complete": True}))
    assert supervisor.job_done(job)


def test_source_index_needs_all_nine_arms_before_reporting(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)
    index = supervisor.METRICS / "scoring_source_index.json"
    index.write_text(json.dumps({"arms": {"tece_oam_rra_1_0": {}}}))
    assert not supervisor.job_done(supervisor.JOBS["mlip-updated-genbench"])
    assert not supervisor.report_complete()
    index.write_text(json.dumps({"arms": {arm: {} for arm in supervisor.ALL_ARMS}}))
    assert supervisor.job_done(supervisor.JOBS["mlip-updated-genbench"])
    assert not supervisor.report_complete()


def test_timestamp_only_completion_changes_do_not_reset_a_crash_budget(monkeypatch, tmp_path):
    configure_paths(monkeypatch, tmp_path)
    path = supervisor.WORK / "tace_oam_l"
    path.mkdir()
    marker = path / "zeus_relaxation_complete.json"
    marker.write_text(json.dumps({"complete": True, "ledger_sha256": "same", "utc": "before"}))
    job = supervisor.JOBS["mlip-migration-gpu0"]
    before = supervisor.progress(job)
    marker.write_text(json.dumps({"complete": True, "ledger_sha256": "same", "utc": "after"}))
    assert supervisor.progress(job) == before
    marker.write_text(json.dumps({"complete": True, "ledger_sha256": "new-trial", "utc": "after"}))
    assert supervisor.progress(job) != before
