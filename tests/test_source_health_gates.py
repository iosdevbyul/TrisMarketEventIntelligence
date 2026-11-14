"""Source health/freshness controls and integration before model calls."""
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tris_market_event_intelligence.collector import Article, NewsStore
from tris_market_event_intelligence.impact import ImpactResult
from tris_market_event_intelligence.impact_store import ImpactProcessor
from tris_market_event_intelligence.source_health import (
    assess_live_event,
    audit_source_health,
)


def observed(seconds: int = 0) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


def healthy_store(path: Path, *, success_age_seconds: int = 0) -> NewsStore:
    store = NewsStore(path)
    store.initialize()
    stamp = observed(-success_age_seconds)
    store.save("bbc_business", "https://example.org/feed", None, [], stamp, None, None)
    store.record_attempt(
        "bbc_business", stamp, stamp, stamp, 12.0, "ok", 0, 0,
    )
    return store


def test_source_health_recent_success_and_old_success(tmp_path: Path) -> None:
    path = tmp_path / "live.db"
    healthy_store(path)
    now = datetime.now(UTC)
    outcome = audit_source_health(
        path, expected_sources=("bbc_business",), now=now,
    )
    assert outcome["status"] == "OK"
    assert outcome["sources"][0]["reason"] == "RECENT_SUCCESSFUL_POLL"
    assert outcome["live_news_coverage_verified"] is False
    stale = audit_source_health(
        path, expected_sources=("bbc_business",),
        now=now + timedelta(minutes=20),
    )
    assert stale["status"] == "BLOCKED"
    assert stale["sources"][0]["reason"] == "SOURCE_POLL_STALE"


def test_source_failure_isolation_and_recovery(tmp_path: Path) -> None:
    path = tmp_path / "live.db"
    store = healthy_store(path)
    stamp = observed()
    store.record_error("bbc_business", "HTTP_429", stamp)
    store.record_attempt("bbc_business", stamp, stamp, stamp, 5, "error", 0, 0)
    state = audit_source_health(
        path, expected_sources=("bbc_business",), now=datetime.now(UTC),
    )
    assert state["status"] == "BLOCKED"
    assert state["sources"][0]["reason"] == "SOURCE_RATE_LIMITED"
    store.save("bbc_business", "https://example.org/feed", None, [], stamp, None, None)
    store.record_attempt("bbc_business", stamp, stamp, stamp, 5, "not_modified", 0, 0)
    state = audit_source_health(
        path, expected_sources=("bbc_business",), now=datetime.now(UTC),
    )
    assert state["status"] == "OK"


def test_missing_telemetry_and_unknown_source_fail_closed(tmp_path: Path) -> None:
    absent = audit_source_health(
        tmp_path / "not-here.db", expected_sources=("bbc_business",),
        now=datetime.now(UTC),
    )
    assert absent["status"] == "BLOCKED"
    path = tmp_path / "existing.db"
    store = healthy_store(path)
    state = audit_source_health(
        path, expected_sources=("bbc_business", "gdelt_doc"), now=datetime.now(UTC),
    )
    by_source = {item["source"]: item for item in state["sources"]}
    assert by_source["bbc_business"]["status"] == "OK"
    assert by_source["gdelt_doc"]["reason"] == "NO_SUCCESSFUL_POLL"
    assert state["status"] == "BLOCKED"
    with store.connect() as db:
        db.execute("DROP TABLE fetch_attempts")
    assert audit_source_health(
        path, expected_sources=("bbc_business",), now=datetime.now(UTC),
    )["reason"] == "SOURCE_TELEMETRY_UNAVAILABLE"


def test_live_event_requires_both_recent_receipt_and_plausible_publication() -> None:
    now = datetime.now(UTC)
    known = (now - timedelta(seconds=10)).isoformat()
    assert assess_live_event(
        detected_at=known,
        published_at=(now - timedelta(seconds=30)).isoformat(),
        now=now, max_received_age_seconds=300,
        max_declared_publish_age_seconds=300,
    ) == "ELIGIBLE_WITH_UNVERIFIED_PUBLICATION"
    assert assess_live_event(
        detected_at=(now - timedelta(minutes=30)).isoformat(),
        published_at=(now - timedelta(minutes=31)).isoformat(),
        now=now, max_received_age_seconds=300,
        max_declared_publish_age_seconds=300,
    ) == "RECEIPT_STALE"
    assert assess_live_event(
        detected_at=known, published_at=None, now=now,
        max_received_age_seconds=300, max_declared_publish_age_seconds=300,
    ) == "PUBLICATION_UNVERIFIED"
    assert assess_live_event(
        detected_at=known,
        published_at=(now + timedelta(seconds=1)).isoformat(),
        now=now, max_received_age_seconds=300,
        max_declared_publish_age_seconds=300,
    ) == "PUBLICATION_AFTER_RECEIPT"
    assert assess_live_event(
        detected_at=known,
        published_at=(now - timedelta(minutes=10)).isoformat(),
        now=now, max_received_age_seconds=300,
        max_declared_publish_age_seconds=300,
    ) == "DECLARED_PUBLICATION_STALE"


class NoImpactProvider:
    model_id = "fixture-cpu"
    def __init__(self) -> None:
        self.calls = 0
    def analyze(self, event: object) -> ImpactResult:
        self.calls += 1
        return ImpactResult(impacts=[])


def seed_event(
    db: Path, *, first_seen: str, published: str | None,
) -> NewsStore:
    store = healthy_store(db)
    store.save(
        "bbc_business", "https://example.org/feed", b"<local>test</local>",
        [Article(
            "bbc_business", "id-1", "https://example.org/1",
            "NVIDIA CEO resigns", None, published, first_seen,
        )], first_seen, None, None,
    )
    processor = ImpactProcessor(db, NoImpactProvider())
    processor.initialize()
    with store.connect() as conn:
        conn.execute(
            "INSERT INTO economic_events("
            "article_id,extractor_version,classification,status,evidence,source,"
            "source_url,published_at,detected_at,processed_at,capture_sha256"
            ") VALUES (1,'test','CORPORATE_EVENT','CANDIDATE',"
            "'NVIDIA CEO resigns','bbc_business','https://example.org/1',?,?,?,'sha')",
            (published, first_seen, first_seen),
        )
        conn.execute(
            "INSERT INTO event_metadata(event_id,claim_kind,tags_json,verification_note)"
            " VALUES (1,'REPORTED','[]','test')",
        )
    return store


def test_fresh_live_analysis_calls_model_and_checkpoints(tmp_path: Path) -> None:
    path = tmp_path / "news.db"
    seed_event(
        path, first_seen=observed(-15), published=observed(-20),
    )
    model = NoImpactProvider()
    result = ImpactProcessor(path, model).process_pending(
        live_freshness_seconds=300,
    )
    assert result.processed == 1
    assert result.deferred == 0
    assert model.calls == 1


def test_stale_live_analysis_defers_without_model_or_checkpoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "news.db"
    seed_event(
        path, first_seen=observed(-1800), published=observed(-1900),
    )
    model = NoImpactProvider()
    processor = ImpactProcessor(path, model)
    result = processor.process_pending(live_freshness_seconds=300)
    assert result.deferred == 1
    assert result.processed == result.failed == 0
    assert model.calls == 0
    with processor.store.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM impact_runs").fetchone()[0] == 0


def test_unhealthy_source_blocks_live_model_even_for_recent_event(tmp_path: Path) -> None:
    path = tmp_path / "news.db"
    store = seed_event(
        path, first_seen=observed(-5), published=observed(-15),
    )
    stamp = observed()
    store.record_error("bbc_business", "HTTP_429", stamp)
    store.record_attempt("bbc_business", stamp, stamp, stamp, 10, "error", 0, 0)
    model = NoImpactProvider()
    stats = ImpactProcessor(path, model).process_pending(live_freshness_seconds=300)
    assert stats.deferred == 1
    assert model.calls == 0


def test_legacy_historical_analysis_path_remains_available(tmp_path: Path) -> None:
    path = tmp_path / "news.db"
    seed_event(
        path, first_seen=observed(-86400), published=observed(-86500),
    )
    model = NoImpactProvider()
    result = ImpactProcessor(path, model).process_pending()
    assert result.processed == 1
    assert result.deferred == 0
    assert model.calls == 1


def test_invalid_threshold_rejected(tmp_path: Path) -> None:
    path = tmp_path / "news.db"
    with pytest.raises(ValueError, match="unique"):
        audit_source_health(path, expected_sources=("a", "a"), now=datetime.now(UTC))
    with pytest.raises(ValueError, match="positive"):
        assess_live_event(
            detected_at=observed(), published_at=observed(),
            now=datetime.now(UTC), max_received_age_seconds=0,
            max_declared_publish_age_seconds=300,
        )
    with pytest.raises(ValueError, match="live_freshness_seconds"):
        ImpactProcessor(path, NoImpactProvider()).process_pending(
            live_freshness_seconds=0,
        )
