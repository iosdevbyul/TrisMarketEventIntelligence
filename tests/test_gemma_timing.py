"""Measured Gemma provider runtime versus preceding queue time."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tris_market_event_intelligence.gemma_timing import audit_gemma_timing
from tris_market_event_intelligence.impact import ImpactResult
from tris_market_event_intelligence.impact_store import ImpactProcessor
from tris_market_event_intelligence.operational_evidence import build_operational_evidence

from test_local_impact import insert_fixture


class TimedMockProvider:
    model_id = "gemma2:local-stub"

    def analyze(self, event: object) -> ImpactResult:
        return ImpactResult(impacts=[])


class FailedMockProvider:
    model_id = "gemma2:local-failure"

    def analyze(self, event: object) -> ImpactResult:
        raise RuntimeError("private provider error details")


def test_success_records_measured_provider_wall_time_and_queue(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    insert_fixture(path)
    stats = ImpactProcessor(path, TimedMockProvider()).process_pending()
    assert stats.processed == 2
    with sqlite3.connect(path) as db:
        rows = db.execute(
            "SELECT attempted_at,provider_returned_at,completed_at,"
            "provider_elapsed_ms,processing_elapsed_ms,outcome "
            "FROM impact_attempt_timings ORDER BY id"
        ).fetchall()
    assert len(rows) == 2
    for started, returned, done, provider_ms, total_ms, outcome in rows:
        assert datetime.fromisoformat(started) <= datetime.fromisoformat(returned)
        assert datetime.fromisoformat(returned) <= datetime.fromisoformat(done)
        assert provider_ms >= 0
        assert total_ms >= provider_ms
        assert outcome == "NO_HYPOTHESIS"
    audit = audit_gemma_timing(path, as_of=datetime.now(UTC))
    assert audit["status"] == "OBSERVED"
    assert audit["timed_attempts"] == 2
    assert audit["sources"][0]["provider_wall_time"]["count"] == 2
    assert audit["sources"][0]["extraction_to_attempt_wait"]["count"] == 2
    assert audit["first_token_latency_observed"] is False


def test_failed_provider_is_measured_without_logging_sensitive_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "news.sqlite3"
    insert_fixture(path)
    stats = ImpactProcessor(path, FailedMockProvider()).process_pending()
    assert stats.failed == 2
    with sqlite3.connect(path) as db:
        attempts = db.execute(
            "SELECT outcome, provider_returned_at, provider_elapsed_ms "
            "FROM impact_attempt_timings"
        ).fetchall()
        error_codes = db.execute("SELECT last_error FROM impact_errors").fetchall()
    assert all(outcome == "ERROR" and returned and duration >= 0 
               for outcome, returned, duration in attempts)
    assert error_codes == [("RuntimeError",), ("RuntimeError",)]
    audit = audit_gemma_timing(path, as_of=datetime.now(UTC))
    assert audit["sources"][0]["provider_duration_missing"] == 0
    assert audit["sources"][0]["errors"] == 2
    assert audit["sources"][0]["provider_wall_time"]["count"] == 2


def test_legacy_completion_is_not_falsely_called_model_runtime(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    insert_fixture(path)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE impact_runs (event_id INTEGER,model_id TEXT,"
            "prompt_version TEXT,processed_at TEXT,outcome TEXT)"
        )
        db.execute(
            "INSERT INTO impact_runs VALUES (1,'gemma2:latest','v8',?,'COMPLETED')",
            (datetime.now(UTC).isoformat(),),
        )
    audit = audit_gemma_timing(path, as_of=datetime.now(UTC))
    assert audit["status"] == "LEGACY_TIMING_UNAVAILABLE"
    assert audit["historical_model_runtime_known"] is False
    assert audit["timed_attempts"] == 0


def test_existing_operational_report_includes_new_timing_and_blocker(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    insert_fixture(path)
    before = build_operational_evidence(
        path, as_of=datetime.now(UTC), sources=("bbc_business",),
    )
    assert before["checks"]["gemma_timing"]["data"]["status"] == (
        "LEGACY_TIMING_UNAVAILABLE"
    )
    assert "GEMMA_PROVIDER_DURATION_NOT_YET_OBSERVED" in (
        before["stage_1_evidence_review"]["blockers"]
    )
    ImpactProcessor(path, TimedMockProvider()).process_pending()
    after = build_operational_evidence(
        path, as_of=datetime.now(UTC), sources=("bbc_business",),
    )
    assert after["checks"]["gemma_timing"]["status"] == "OBSERVED"
    assert after["checks"]["gemma_timing"]["data"]["timed_attempts"] == 2
    assert "GEMMA_PROVIDER_DURATION_NOT_YET_OBSERVED" not in (
        after["stage_1_evidence_review"]["blockers"]
    )


def test_missing_or_anomalous_timing_remains_explicit(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    insert_fixture(path)
    ImpactProcessor(path, TimedMockProvider()).process_pending()
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE impact_attempt_timings SET attempted_at=? WHERE id=1",
            ((datetime.now(UTC) + timedelta(days=1)).isoformat(),),
        )
    audit = audit_gemma_timing(path, as_of=datetime.now(UTC))
    assert audit["sources"][0]["chronology_anomalies"] == 1
    assert audit["sources"][0]["provider_wall_time"]["count"] == 1
