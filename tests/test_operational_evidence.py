"""Integrated evidence report is truthful about missing/legacy data."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tris_market_event_intelligence.collector import Article, NewsStore
from tris_market_event_intelligence.exchange_sessions import SessionCalendar, SessionDay
from tris_market_event_intelligence.operational_evidence import build_operational_evidence

NOW = datetime(2026, 11, 16, 15, tzinfo=UTC)
SOURCE = "fed_monetary"


def observed(seconds: int) -> str:
    return (NOW + timedelta(seconds=seconds)).isoformat()


def seed(path: Path) -> None:
    store = NewsStore(path)
    store.initialize()
    store.save(
        SOURCE, "https://example.org/feed", b"<sample/>",
        [
            Article(
                SOURCE, "one", "https://example.org/one", "sample",
                None, observed(-50), observed(-30),
            ),
            Article(
                SOURCE, "old", "https://example.org/old", "sample",
                None, observed(-7200), observed(-20),
            ),
            Article(
                SOURCE, "unknown", "https://example.org/missing", "sample",
                None, None, observed(-10),
            ),
        ],
        observed(-10), None, None,
    )
    store.record_attempt(
        SOURCE, observed(-10), observed(-9), observed(-8), 10.0, "ok", 3, 3,
    )
    with store.connect() as db:
        db.executescript(
            """CREATE TABLE IF NOT EXISTS economic_events (
                id INTEGER PRIMARY KEY,
                article_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                detected_at TEXT NOT NULL,
                processed_at TEXT NOT NULL,
                published_at TEXT
            );
            CREATE TABLE IF NOT EXISTS impact_runs (
                event_id INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                processed_at TEXT NOT NULL,
                outcome TEXT NOT NULL
            );"""
        )
        db.execute(
            "INSERT INTO economic_events VALUES (1,1,?,?,?,?)",
            (SOURCE, observed(-30), observed(-25), observed(-50)),
        )
        db.execute(
            "INSERT INTO impact_runs VALUES (1,'gemma2:latest','v1',?,'COMPLETED')",
            (observed(-15),),
        )


def calendar() -> SessionCalendar:
    return SessionCalendar([SessionDay(
        "XNAS", NOW.date(), "OPEN",
        NOW - timedelta(minutes=30), NOW + timedelta(hours=5),
    )])


def test_all_metrics_in_single_report_with_read_only_db(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    seed(path)
    result = build_operational_evidence(
        path, as_of=NOW, sources=(SOURCE,),
        calendar=calendar(), markets=("XNAS",),
    )
    assert result["status"] == "RESEARCH_OBSERVATIONS_ONLY"
    assert result["real_market_data_observed_on_this_host"] is True
    assert result["missing_headline_recall_measured"] is False
    assert result["four_stage_product_complete"] is False
    assert set(result["checks"]) == {
        "news_latency", "event_lifecycle", "recent_freshness", "source_health",
    }
    assert result["checks"]["event_lifecycle"]["data"]["model_checkpoints"] == 1
    assert result["checks"]["event_lifecycle"]["data"]["sources"][0][
        "first_seen_to_checkpoint_latency"
    ]["median_seconds"] == 15
    freshness = result["checks"]["recent_freshness"]["data"]["sources"][0]
    assert freshness["articles"] == 3
    assert freshness["freshness_reasons"]["ELIGIBLE_WITH_UNVERIFIED_PUBLICATION"] == 1
    assert freshness["freshness_reasons"]["DECLARED_PUBLICATION_STALE"] == 1
    assert freshness["freshness_reasons"]["PUBLICATION_UNVERIFIED"] == 1
    assert result["session_plan"]["decision"] == "RUN"


def test_missing_db_not_misreported_as_zero_events(tmp_path: Path) -> None:
    path = tmp_path / "absent.sqlite3"
    result = build_operational_evidence(path, as_of=NOW, sources=(SOURCE,))
    assert result["status"] == "UNVERIFIED"
    assert set(result["blocked_checks"]) == {
        "news_latency", "event_lifecycle", "source_health", "recent_freshness",
        "exchange_sessions",
    }
    assert result["checks"]["news_latency"] == {
        "status": "UNAVAILABLE", "reason": "DATABASE_NOT_AVAILABLE",
    }
    assert result["real_market_data_observed_on_this_host"] is False
    assert not path.exists()


def test_legacy_schema_reports_specific_unavailable_checks(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    store = NewsStore(path)
    store.initialize()
    with store.connect() as db:
        db.execute("DROP TABLE fetch_attempts")
    report = build_operational_evidence(
        path, as_of=NOW, sources=(SOURCE,),
        calendar=calendar(), markets=("XNAS",),
    )
    assert report["status"] == "BLOCKED"
    assert report["checks"]["news_latency"]["status"] == "OBSERVED"
    assert report["checks"]["event_lifecycle"]["status"] == "UNAVAILABLE"
    assert report["checks"]["source_health"]["status"] == "BLOCKED"
    assert report["checks"]["source_health"]["data"]["reason"] == "SOURCE_TELEMETRY_UNAVAILABLE"


def test_unknown_market_day_fails_closed_even_if_news_observed(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    seed(path)
    report = build_operational_evidence(
        path, as_of=NOW, sources=(SOURCE,),
        calendar=calendar(), markets=("XKRX", "XNAS"),
    )
    assert report["status"] == "BLOCKED"
    assert report["session_plan"]["markets"]["XKRX"] == "CALENDAR_UNAVAILABLE"
    assert "exchange_sessions" in report["blocked_checks"]


def test_recent_window_does_not_relabel_all_time_history(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    seed(path)
    report = build_operational_evidence(
        path, as_of=NOW, sources=(SOURCE,), calendar=calendar(),
        markets=("XNAS",), window_hours=1,
    )
    assert report["checks"]["recent_freshness"]["data"]["total_articles"] == 3
    assert report["checks"]["news_latency"]["data"]["articles_count"] == 3
    assert report["observation_start_utc"] == (NOW - timedelta(hours=1)).isoformat()


@pytest.mark.parametrize("hours,minutes", [(0, 15), (200, 15), (24, 0)])
def test_rejects_unbounded_or_invalid_thresholds(
    tmp_path: Path, hours: int, minutes: int,
) -> None:
    with pytest.raises(ValueError, match="bounded"):
        build_operational_evidence(
            tmp_path / "none.db", as_of=NOW,
            window_hours=hours, freshness_minutes=minutes,
        )


def test_malformed_event_audit_does_not_destroy_other_evidence(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    seed(path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE economic_events SET processed_at='bad' WHERE id=1")
    report = build_operational_evidence(
        path, as_of=NOW, sources=(SOURCE,),
        calendar=calendar(), markets=("XNAS",),
    )
    assert report["status"] == "BLOCKED"
    assert report["checks"]["event_lifecycle"]["status"] == "UNAVAILABLE"
    assert report["checks"]["recent_freshness"]["status"] == "OBSERVED"
    assert report["checks"]["source_health"]["status"] == "OBSERVED"
