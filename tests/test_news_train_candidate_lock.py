"""Pin one TRAIN-only candidate without TEST access or model promotion."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_news_research_worker import files

from tris_market_event_intelligence.news_research_policy import (
    MAX_HYPOTHESES,
    run_policy_once,
)
from tris_market_event_intelligence.news_test_usage_ledger import (
    initialize_usage_ledger,
)
from tris_market_event_intelligence.news_train_candidate_lock import (
    audit_candidate_lock,
    create_candidate_lock,
)


def _inputs(tmp_path: Path):
    return files(tmp_path)


def _run(paths):
    return run_policy_once(
        cohort_path=paths[0], plan_path=paths[1],
        review_path=paths[2], journal_path=paths[3],
    )


def _lock(paths, manifest):
    return create_candidate_lock(
        output_path=manifest, cohort_path=paths[0], plan_path=paths[1],
        review_path=paths[2], journal_path=paths[3],
    )


def test_lock_requires_all_train_hypotheses_and_never_overwrites(tmp_path: Path):
    paths = _inputs(tmp_path)
    manifest = tmp_path / "candidate-lock.json"
    with pytest.raises(ValueError, match="12 pinned"):
        _lock(paths, manifest)
    assert not manifest.exists()
    for _ in range(MAX_HYPOTHESES):
        assert _run(paths)["newResearchExperiment"] is True
    result = _lock(paths, manifest)
    assert result["status"] == "TRAIN_CANDIDATE_LOCKED_NOT_VALIDATED"
    assert result["promotionAllowed"] is False
    assert manifest.stat().st_mode & 0o777 == 0o600
    audit = audit_candidate_lock(
        lock_path=manifest, cohort_path=paths[0], plan_path=paths[1],
        review_path=paths[2], journal_path=paths[3],
    )
    assert audit["pinnedInputsReplayed"] is True
    assert audit["independentImprovementVerified"] is False
    with pytest.raises(ValueError, match="New ordinary"):
        _lock(paths, manifest)


def test_lock_rejects_digest_tamper_and_audits_test_budget(tmp_path: Path):
    paths = _inputs(tmp_path)
    for _ in range(MAX_HYPOTHESES):
        _run(paths)
    manifest = tmp_path / "locked.json"
    _lock(paths, manifest)
    ledger = tmp_path / "reserved-test.sqlite3"
    initialize_usage_ledger(ledger)
    budget = audit_candidate_lock(lock_path=manifest, usage_ledger_path=ledger)
    assert budget["finalTestReservationsInCooperativeLedger"] == 0
    assert budget["finalTestAccessOutsideLedgerVerified"] is False
    assert budget["promotionAllowed"] is False
    document = json.loads(manifest.read_text())
    document["payload"]["alpha"] = 500
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="digest mismatch"):
        audit_candidate_lock(lock_path=manifest)


def test_changed_train_journal_detected_by_replay(tmp_path: Path):
    paths = _inputs(tmp_path)
    for _ in range(MAX_HYPOTHESES):
        _run(paths)
    manifest = tmp_path / "locked.json"
    _lock(paths, manifest)
    import sqlite3

    selected = json.loads(manifest.read_text())["payload"]["experimentSha256"]
    with sqlite3.connect(paths[3]) as db:
        db.execute(
            "UPDATE news_research_policy_experiments "
            "SET train_cv_log_loss=123 WHERE experiment_sha256=?",
            (selected,),
        )
    with pytest.raises(ValueError, match="differs"):
        audit_candidate_lock(
            lock_path=manifest, cohort_path=paths[0],
            plan_path=paths[1], review_path=paths[2], journal_path=paths[3],
        )
