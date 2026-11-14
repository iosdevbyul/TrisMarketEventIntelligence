"""Synthetic-only lifecycle audit regression and privacy checks."""
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tris_market_event_intelligence.event_store import EventProcessor
from tris_market_event_intelligence.event_lifecycle_latency import audit_event_lifecycle


NOW = datetime(2026, 11, 18, tzinfo=UTC)


def setup(db_path: Path, *, model_table: bool = True) -> None:
    processor = EventProcessor(db_path)
    processor.initialize()
    if model_table:
        with processor.store.connect() as db:
            db.executescript(
                """CREATE TABLE impact_runs (
                    event_id INTEGER NOT NULL,
                    model_id TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    PRIMARY KEY(event_id,model_id,prompt_version)
                );"""
            )


def insert_event(
    db_path: Path, *, event_id: int,
    source: str = "fed_monetary",
    first: str = "2026-11-16T10:00:30+00:00",
    extraction: str = "2026-11-16T10:00:35+00:00",
    publication: str | None = "2026-11-16T10:00:00+00:00",
    copied_first: str | None = None,
) -> None:
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO captures(id,source,feed_url,fetched_at,sha256,payload) "
            "VALUES (?,?,'https://example.org/rss',?,'sha',X'00')",
            (event_id, source, first),
        )
        db.execute(
            "INSERT INTO articles(id,source,external_id,url,title,published_at,"
            "detected_at,capture_id) VALUES (?,?,?,'https://example.org/a','News',?,?,?)",
            (event_id, source, str(event_id), publication, first, event_id),
        )
        db.execute(
            "INSERT INTO economic_events("
            "id,article_id,extractor_version,classification,status,evidence,"
            "source,source_url,published_at,detected_at,processed_at,capture_sha256"
            ") VALUES (?,?,'v1','REGULATION','UNVERIFIED','Evidence',?,"
            "'https://example.org/a',?,?,?,'sha')",
            (event_id, event_id, source, publication, copied_first or first, extraction),
        )


def insert_checkpoint(
    db_path: Path, event_id: int, *,
    completed: str = "2026-11-16T10:00:45+00:00",
    model: str = "gemma2:latest", outcome: str = "COMPLETED",
) -> None:
    with sqlite3.connect(db_path) as db:
        db.execute(
            "INSERT INTO impact_runs VALUES (?,?,?,?,?)",
            (event_id, model, "market-impact-v8", completed, outcome),
        )


def test_end_to_end_local_stages_from_first_seen_to_model_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    setup(path)
    insert_event(path, event_id=1)
    insert_checkpoint(path, 1)
    original = path.read_bytes()
    report = audit_event_lifecycle(path, as_of=NOW)
    source = report["sources"][0]
    assert source["economic_events"] == 1
    assert source["checkpointed_events"] == 1
    assert source["first_seen_to_extraction_latency"]["median_seconds"] == 5
    assert source["extraction_to_checkpoint_latency"]["median_seconds"] == 10
    assert source["first_seen_to_checkpoint_latency"]["median_seconds"] == 15
    assert source["published_to_checkpoint_latency"]["median_seconds"] == 45
    assert report["gemma_token_start_observed"] is False
    assert report["news_to_usable_inference_verified"] is False
    assert path.read_bytes() == original


def test_no_checkpoint_keeps_latency_null_instead_of_zero(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    setup(path)
    insert_event(path, event_id=1)
    report = audit_event_lifecycle(path, as_of=NOW)
    source = report["sources"][0]
    assert source["uncheckpointed_events"] == 1
    assert source["first_seen_to_checkpoint_latency"] == {
        "count": 0, "median_seconds": None, "p95_seconds": None,
    }


def test_multiple_models_and_abstention_are_separate_checkpoints(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    setup(path)
    insert_event(path, event_id=1)
    insert_checkpoint(path, 1, outcome="NO_HYPOTHESIS")
    insert_checkpoint(path, 1, model="gemma2:v2",
                      completed="2026-11-16T10:00:50+00:00")
    source = audit_event_lifecycle(path, as_of=NOW)["sources"][0]
    assert source["checkpointed_events"] == 1
    assert source["model_checkpoints"] == 2
    assert source["no_hypothesis_checkpoints"] == 1
    assert source["first_seen_to_checkpoint_latency"]["count"] == 2
    assert source["first_seen_to_checkpoint_latency"]["median_seconds"] == 17.5


def test_anomalous_source_timestamps_not_falsely_scored(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    setup(path)
    insert_event(path, event_id=1,
                 copied_first="2026-11-16T09:59:30+00:00")
    insert_checkpoint(path, 1)
    source = audit_event_lifecycle(path, as_of=NOW)["sources"][0]
    assert source["chronology_anomalies"] == 1
    assert source["first_seen_to_checkpoint_latency"]["count"] == 0


def test_model_checkpoint_before_extraction_is_not_falsely_scored(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    setup(path)
    insert_event(path, event_id=1)
    insert_checkpoint(path, 1, completed="2026-11-16T10:00:34+00:00")
    source = audit_event_lifecycle(path, as_of=NOW)["sources"][0]
    assert source["chronology_anomalies"] == 1
    assert source["extraction_to_checkpoint_latency"]["count"] == 0


def test_missing_or_future_feed_publication_is_separate(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    setup(path)
    insert_event(path, event_id=1, publication=None)
    insert_event(
        path, event_id=2, source="naver_news",
        publication="2026-11-16T10:02:00+00:00",
    )
    insert_checkpoint(path, 1)
    insert_checkpoint(path, 2)
    output = audit_event_lifecycle(path, as_of=NOW)
    items = {source["source"]: source for source in output["sources"]}
    assert items["fed_monetary"]["publication_missing"] == 1
    assert items["naver_news"]["publication_after_first_seen"] == 1
    assert items["fed_monetary"]["published_to_checkpoint_latency"]["count"] == 0
    assert items["naver_news"]["published_to_checkpoint_latency"]["count"] == 0


def test_legacy_database_with_no_model_table_remains_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    setup(path, model_table=False)
    insert_event(path, event_id=1)
    report = audit_event_lifecycle(path, as_of=NOW)
    assert report["model_checkpoint_table_available"] is False
    assert report["model_checkpoints"] == 0
    with sqlite3.connect(path) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='impact_runs'"
        ).fetchone()[0] == 0


def test_missing_database_or_naive_audit_clock_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Existing"):
        audit_event_lifecycle(tmp_path / "missing.db", as_of=NOW)
    path = tmp_path / "news.sqlite3"
    setup(path)
    with pytest.raises(ValueError, match="timezone"):
        audit_event_lifecycle(path, as_of=datetime(2026, 11, 18))


def test_orphan_checkpoint_rejected_and_no_source_text_leaked(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    setup(path)
    insert_checkpoint(path, 200)
    with pytest.raises(ValueError, match="Orphan"):
        audit_event_lifecycle(path, as_of=NOW)


def test_malformed_model_and_date_out_of_asof_rejected(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    setup(path)
    insert_event(path, event_id=1)
    insert_checkpoint(path, 1, completed="2026-11-20T00:00:00+00:00")
    with pytest.raises(ValueError, match="audit time"):
        audit_event_lifecycle(path, as_of=NOW)
