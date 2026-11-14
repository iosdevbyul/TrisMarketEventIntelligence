"""Manual pilot selection should be predictable, bounded and safe."""
import sqlite3
from pathlib import Path

import pytest

from tris_market_event_intelligence.impact import EventInput, ImpactResult
from tris_market_event_intelligence.impact_pilot import pilot_candidates
from tris_market_event_intelligence.impact_store import ImpactProcessor


class EmptyModel:
    model_id = "gemma2:latest"

    def __init__(self):
        self.calls = []

    def analyze(self, event: EventInput) -> ImpactResult:
        self.calls.append(event.event_id)
        return ImpactResult(impacts=[])


def database(db: Path) -> None:
    # Use the production event schema, but avoid remote collection.
    from tris_market_event_intelligence.event_store import EventProcessor

    EventProcessor(db).initialize()
    model = EmptyModel()
    ImpactProcessor(db, model).initialize()
    with sqlite3.connect(db) as conn:
        for event_id, category, status, kind in [
            (1, "INDUSTRY", "UNVERIFIED", "REPORTED"),
            (2, "INDUSTRY", "UNVERIFIED", "REPORTED"),
            (3, "EARNINGS", "UNVERIFIED", "REPORTED"),
            (4, "GEOPOLITICAL", "UNVERIFIED", "REPORTED"),
            (5, "INDUSTRY", "UNVERIFIED", "FORECAST"),
            (6, "MONETARY_POLICY", "CANDIDATE", "REPORTED"),
            (7, "SUPPLY_CHAIN", "UNVERIFIED", "REPORTED"),
        ]:
            conn.execute(
                """INSERT INTO economic_events (
                    id, article_id, extractor_version, classification, status,
                    evidence, source, source_url, detected_at, processed_at,
                    capture_sha256
                ) VALUES (?, ?, 'fixture', ?, ?, ?,
                          'fixture', ?, 'now', 'now', 'sha')""",
                (event_id, event_id, category, status,
                 f"Fixture evidence for independent event {event_id}",
                 f"https://example.org/story/{event_id}"),
            )
            conn.execute(
                "INSERT INTO event_metadata VALUES (?, ?, '[]', 'not verified')",
                (event_id, kind),
            )
        conn.execute(
            """INSERT INTO impact_runs VALUES
               (2, 'gemma2:latest', 'market-impact-v7', 'now', 'NO_HYPOTHESIS')"""
        )
        conn.execute(
            """INSERT INTO impact_runs VALUES
               (2, 'gemma2:latest', 'market-impact-v8', 'now', 'NO_HYPOTHESIS')"""
        )
        conn.execute(
            """INSERT INTO impact_errors VALUES
               (1, 'gemma2:latest', 'market-impact-v7',
                'RATIONALE_INCOMPLETE', 'now', 1)"""
        )


def test_diverse_fresh_candidates_avoid_failed_and_completed(tmp_path: Path) -> None:
    db = tmp_path / "sample.sqlite3"
    database(db)
    report = pilot_candidates(db, model_id="gemma2:latest", prompt_version="market-impact-v7")
    assert [r["event_id"] for r in report["candidates"]] == [3, 4, 7]
    assert report["eligible_events"] == 5
    assert report["already_checkpointed_events"] == 1
    assert report["fresh_pending_events"] == 3
    assert report["pending_with_prior_failure"] == 1
    assert report["quality_verified"] is False


def test_explicit_selection_cannot_run_unrequested_events(tmp_path: Path) -> None:
    db = tmp_path / "sample.sqlite3"
    database(db)
    model = EmptyModel()
    processor = ImpactProcessor(db, model)
    assert processor.process_pending(event_ids=(3, 7), batch_size=2).processed == 2
    assert model.calls == [3, 7]
    with pytest.raises(ValueError, match="already checkpointed"):
        processor.process_pending(event_ids=(3,), batch_size=1)


def test_invalid_or_ineligible_selection_rejected_before_inference(tmp_path: Path) -> None:
    db = tmp_path / "sample.sqlite3"
    database(db)
    model = EmptyModel()
    processor = ImpactProcessor(db, model)
    for ids in [(5,), (6,), (99,), (2,), (1, 1), (1, 3, 4, 7)]:
        with pytest.raises(ValueError):
            processor.process_pending(event_ids=ids, batch_size=3)
    assert model.calls == []


def test_missing_db_and_limits(tmp_path: Path) -> None:
    db = tmp_path / "missing.sqlite3"
    assert pilot_candidates(db, model_id="gemma2:latest")["status"] == "database_missing"
    assert not db.exists()
    for limit in (0, 4):
        with pytest.raises(ValueError, match="limit"):
            pilot_candidates(db, model_id="gemma2:latest", limit=limit)


def test_new_only_excludes_all_previous_versions_and_failed_attempts(
    tmp_path: Path,
) -> None:
    db = tmp_path / "sample.sqlite3"
    database(db)
    before = db.read_bytes()
    report = pilot_candidates(
        db, model_id="gemma2:latest", prompt_version="market-impact-v8",
        new_only=True,
    )
    assert [row["event_id"] for row in report["candidates"]] == [3, 4, 7]
    assert report["new_only"] is True
    assert report["fresh_pending_events"] == 3
    assert db.read_bytes() == before


def test_new_only_is_scoped_to_requested_model(tmp_path: Path) -> None:
    db = tmp_path / "sample.sqlite3"
    database(db)
    report = pilot_candidates(
        db, model_id="another-model", prompt_version="market-impact-v8",
        new_only=True,
    )
    assert [row["event_id"] for row in report["candidates"]] == [1, 3, 4]


def test_new_cohort_skips_duplicate_articles_and_title_only_ppi(tmp_path: Path) -> None:
    db = tmp_path / "sample.sqlite3"
    database(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE economic_events SET source_url=? WHERE id=7",
            ("https://example.org/story/3?utm_source=feed",),
        )
        conn.execute(
            "UPDATE economic_events SET classification='ECONOMIC_INDICATOR', evidence=? WHERE id=4",
            ("2026년 8월 생산자물가지수(잠정)",),
        )
    before = db.read_bytes()
    report = pilot_candidates(
        db, model_id="gemma2:latest", new_only=True,
    )
    assert [row["event_id"] for row in report["candidates"]] == [3]
    assert report["skipped_duplicate_sources"] == 1
    assert report["skipped_sparse_evidence"] == 1
    assert db.read_bytes() == before


def test_new_cohort_skips_previous_version_copy_of_same_article(tmp_path: Path) -> None:
    db = tmp_path / "sample.sqlite3"
    database(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE economic_events SET source_url=? WHERE id=7",
            ("https://example.org/story/2",),
        )
    report = pilot_candidates(
        db, model_id="gemma2:latest", new_only=True,
    )
    assert [row["event_id"] for row in report["candidates"]] == [3, 4]
    assert report["skipped_duplicate_sources"] == 1


def test_new_cohort_collapses_identical_long_evidence_across_feeds(tmp_path: Path) -> None:
    db = tmp_path / "sample.sqlite3"
    database(db)
    long_evidence = (
        "Microsoft says it is continuing to restore services including "
        "Outlook and Teams following a global outage."
    )
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE economic_events SET evidence=? WHERE id=3", (long_evidence,))
        conn.execute("UPDATE economic_events SET evidence=? WHERE id=7", (long_evidence,))
    report = pilot_candidates(
        db, model_id="gemma2:latest", new_only=True,
    )
    assert [row["event_id"] for row in report["candidates"]] == [3, 4]
    assert report["skipped_duplicate_sources"] == 1
