"""Historical news cohort must preserve all news and abstain on missing labels."""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tris_market_event_intelligence.historical_news_cohort import historical_news_cohort


def seeded(tmp_path: Path) -> Path:
    path = tmp_path / "archived-news.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript(
            """CREATE TABLE captures (
                id INTEGER PRIMARY KEY, source TEXT, feed_url TEXT,
                fetched_at TEXT, sha256 TEXT, payload BLOB
            );
            CREATE TABLE articles (
                id INTEGER PRIMARY KEY, source TEXT, external_id TEXT,
                url TEXT, title TEXT, summary TEXT, published_at TEXT,
                detected_at TEXT, capture_id INTEGER
            );
            CREATE TABLE story_members (
                article_id INTEGER PRIMARY KEY, story_id INTEGER,
                canonical_url TEXT, title_fingerprint TEXT,
                match_reason TEXT, indexed_at TEXT
            );
            CREATE TABLE article_event_claims (
                id INTEGER PRIMARY KEY, article_id INTEGER, processed_at TEXT
            );
            CREATE TABLE economic_events (
                id INTEGER PRIMARY KEY, article_id INTEGER, processed_at TEXT
            );"""
        )
        db.execute(
            "INSERT INTO captures VALUES (1,?,?,?,?,?)",
            ("RSS", "https://example.org/feed", "2026-11-01T12:00:00+00:00",
             hashlib.sha256(b"archive").hexdigest(), b"archive"),
        )
        for row in [
            (1, "Important release", "2026-11-01T12:00:01+00:00"),
            (2, "Everyday publicity", "2026-11-01T12:00:02+00:00"),
            (3, "Unclassified commentary", "2026-11-01T12:00:03+00:00"),
            (4, "Future article", "2026-11-09T12:00:03+00:00"),
        ]:
            i, title, received = row
            db.execute(
                "INSERT INTO articles VALUES (?,?,?,?,?,?,?,?,?)",
                (i, "RSS", str(i), f"https://example.org/{i}", title, "Summary",
                 "2026-11-01T11:00:00+00:00", received, 1),
            )
        db.execute("INSERT INTO story_members VALUES(1,4,'x',NULL,'url','2026-11-01')")
        db.execute(
            "INSERT INTO article_event_claims VALUES (1,1,'2026-11-01T12:10:00+00:00')"
        )
        db.execute(
            "INSERT INTO economic_events VALUES (1,1,'2026-11-02T10:00:00+00:00')"
        )
    return path


def run(path: Path, **options: object) -> dict[str, object]:
    return historical_news_cohort(
        source_db=path, as_of=datetime(2026, 11, 1, 13, tzinfo=UTC), **options,
    )


def test_all_observed_articles_survive_even_without_claims(tmp_path: Path) -> None:
    source = seeded(tmp_path)
    with sqlite3.connect(source) as db:
        baseline = tuple(db.iterdump())
    result = run(source)
    assert result["totalSourceArticles"] == 4
    assert result["totalObservedByCutoff"] == 3
    assert result["excludedCounts"] == {"AFTER_AS_OF": 1}
    assert result["sourceCountsByCutoff"] == {"RSS": 3}
    assert [row["articleId"] for row in result["items"]] == [1, 2, 3]
    assert result["items"][0]["claimCountProcessedByCutoff"] == 1
    assert result["items"][0]["eventCountProcessedByCutoff"] == 0
    assert result["items"][0]["storyGroupId"] == 4
    assert result["items"][0]["captureChecksumVerified"] is True
    assert result["items"][1]["claimCountProcessedByCutoff"] == 0
    assert result["items"][2]["eventCountProcessedByCutoff"] == 0
    assert all(row["importanceHumanGold"] == "UNKNOWN" for row in result["items"])
    assert all(row["observedMarketReaction"] == "UNMEASURED" for row in result["items"])
    assert all(row["negativeSampleConfirmed"] is False for row in result["items"])
    assert all(row["trainingEligible"] is False for row in result["items"])
    assert result["trainingDatasetEligible"] is False
    assert result["hasMore"] is False
    with sqlite3.connect(source) as db:
        assert tuple(db.iterdump()) == baseline


def test_paginated_cohort_does_not_lose_unclassified_article(tmp_path: Path) -> None:
    source = seeded(tmp_path)
    first = run(source, limit=2)
    assert [item["articleId"] for item in first["items"]] == [1, 2]
    assert first["hasMore"] is True
    follow = run(source, after_id=first["nextCursor"], limit=2)
    assert [item["articleId"] for item in follow["items"]] == [3]
    assert follow["hasMore"] is False
    empty = run(source, after_id=999, limit=2)
    assert empty["items"] == []
    assert empty["nextCursor"] == 999


def test_missing_optional_stages_are_not_zero_or_important(tmp_path: Path) -> None:
    source = seeded(tmp_path)
    with sqlite3.connect(source) as db:
        db.execute("DROP TABLE article_event_claims")
        db.execute("DROP TABLE economic_events")
        db.execute("DROP TABLE story_members")
    result = run(source)
    assert result["items"][0]["claimCountProcessedByCutoff"] is None
    assert result["items"][0]["eventCountProcessedByCutoff"] is None
    assert result["items"][0]["storyGroupId"] is None


def test_reported_future_publication_does_not_set_future_label(tmp_path: Path) -> None:
    source = seeded(tmp_path)
    with sqlite3.connect(source) as db:
        db.execute(
            "UPDATE articles SET published_at='not-a-date' WHERE id=2"
        )
    result = run(source)
    assert result["excludedCounts"]["PUBLISHED_AT_UNTRUSTED"] == 1
    assert result["items"][1]["publishedAtReported"] is None
    assert result["items"][1]["firstObservedAt"] == "2026-11-01T12:00:02+00:00"
    expected = hashlib.sha256(b"Everyday publicity\nSummary").hexdigest()
    assert result["items"][1]["textSha256"] == expected


@pytest.mark.parametrize("options", [
    {"after_id": True}, {"after_id": -1},
    {"limit": 0}, {"limit": 5001}, {"limit": True},
])
def test_invalid_paging_rejected(tmp_path: Path, options: dict[str, object]) -> None:
    source = seeded(tmp_path)
    with pytest.raises(ValueError):
        run(source, **options)


def test_source_mutation_and_missing_source_fail_closed(tmp_path: Path) -> None:
    source = seeded(tmp_path)
    with sqlite3.connect(source) as db:
        db.execute(
            "UPDATE articles SET detected_at='2026-11-01' WHERE id=1"
        )
    with pytest.raises(ValueError, match="Timezone"):
        run(source)
    missing = tmp_path / "not-existing.sqlite3"
    with pytest.raises(ValueError, match="non-symlink"):
        run(missing)
    assert not missing.exists()


def test_inconsistent_receipt_capture_fails_closed(tmp_path: Path) -> None:
    source = seeded(tmp_path)
    with sqlite3.connect(source) as db:
        db.execute(
            "UPDATE captures SET fetched_at='2026-11-02T12:00:00+00:00'"
        )
    with pytest.raises(ValueError, match="before"):
        run(source)


def test_changed_original_rss_payload_checksum_fails_closed(tmp_path: Path) -> None:
    source = seeded(tmp_path)
    with sqlite3.connect(source) as db:
        db.execute("UPDATE captures SET payload=? WHERE id=1", (b"tampered",))
    with pytest.raises(ValueError, match="checksum mismatch"):
        run(source)
