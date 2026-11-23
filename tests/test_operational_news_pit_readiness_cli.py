"""Tests for non-destructive audits of the genuine news SQLite schema.

Fixture records simulate the collector; they are not real historical articles.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from tris_market_event_intelligence.operational_news_pit_readiness_cli import (
    inspect_live_news,
)


def create_db(path: Path, *, duplicate: bool = False, broken: bool = False,
              no_original: bool = False) -> None:
    feed = b"<rss><item><title>Actual news</title></item></rss>"
    raw = b"<html><main>Actual issuer filing content</main></html>"
    body = "Actual issuer filing content"
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE captures (
          id INTEGER PRIMARY KEY, source TEXT, feed_url TEXT, fetched_at TEXT,
          sha256 TEXT, payload BLOB
        );
        CREATE TABLE articles (
          id INTEGER PRIMARY KEY, source TEXT, external_id TEXT, url TEXT,
          title TEXT, summary TEXT, published_at TEXT, detected_at TEXT,
          capture_id INTEGER
        );
        CREATE TABLE article_full_text (
          article_id INTEGER PRIMARY KEY, status TEXT, observed_at TEXT,
          body TEXT, body_sha256 TEXT, raw_payload BLOB, raw_sha256 TEXT,
          sparse_pages TEXT
        );
        CREATE TABLE story_members (
          article_id INTEGER PRIMARY KEY, story_id INTEGER, match_reason TEXT
        );
        CREATE TABLE automatic_news_analyses (
          article_id INTEGER, status TEXT
        );
        """)
        db.execute("INSERT INTO captures VALUES (1, ?, ?, ?, ?, ?)", (
            "fed_all", "https://example.org/rss", "2026-11-23T14:01:00+09:00",
            hashlib.sha256(feed).hexdigest(), feed,
        ))
        for article_id in (1, 2) if duplicate else (1,):
            db.execute("INSERT INTO articles VALUES (?,?,?,?,?,?,?,?,?)", (
                article_id, "fed_all", str(article_id),
                "https://example.org/article", "News", None,
                "2026-11-23T14:00:00+09:00",
                "2026-11-23T14:01:00+09:00", 1,
            ))
            db.execute("INSERT INTO story_members VALUES (?, ?, ?)", (
                article_id, 22, "new_story" if article_id == 1 else "canonical_url",
            ))
            if not no_original:
                db.execute(
                    "INSERT INTO article_full_text VALUES (?,?,?,?,?,?,?,?)",
                    (article_id, "COMPLETE", "2026-11-23T14:02:00+09:00",
                     body, "bad" if broken else hashlib.sha256(body.encode()).hexdigest(),
                     raw, hashlib.sha256(raw).hexdigest(), "[]"),
                )
            db.execute("INSERT INTO automatic_news_analyses VALUES (?,?)",
                       (article_id, "COMPLETED"))


def test_reads_real_sqlite_schema_without_modifications(tmp_path: Path) -> None:
    database = tmp_path / "news.sqlite3"
    create_db(database)
    before = database.read_bytes()
    report = inspect_live_news(database)
    assert database.read_bytes() == before
    assert report["scanned"] == 1
    assert report["rows"][0]["blockers"] == []
    assert report["rows"][0]["sourceIntegrityReviewed"] is True
    assert report["rows"][0]["firstPublicTimeAuthenticated"] is False
    assert report["independentlyVerifiedMarketOutcomeCount"] == 0
    assert report["learningEligibleCount"] == 0
    assert report["databaseModified"] is False


def test_duplicate_story_is_reported_not_counted_as_independent(tmp_path: Path) -> None:
    database = tmp_path / "news.sqlite3"
    create_db(database, duplicate=True)
    report = inspect_live_news(database)
    assert report["scanned"] == 2
    assert "EXACT_URL_ALREADY_PRESENT" in report["rows"][1]["blockers"]
    assert "STORY_GROUP_HAS_EARLIER_ARTICLE" in report["rows"][1]["blockers"]
    assert report["rows"][1]["eventFamilyIndependentlyVerified"] is False


def test_body_hash_mismatch_and_missing_text_are_explicit(tmp_path: Path) -> None:
    bad = tmp_path / "bad.sqlite3"
    create_db(bad, broken=True)
    report = inspect_live_news(bad)
    assert "BODY_INTEGRITY_OR_PDF_SPARSITY" in report["rows"][0]["blockers"]
    absent = tmp_path / "missing.sqlite3"
    create_db(absent, no_original=True)
    missing = inspect_live_news(absent)
    assert "ORIGINAL_BODY_INCOMPLETE" in missing["rows"][0]["blockers"]


def test_missing_database_and_invalid_limit_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Existing"):
        inspect_live_news(tmp_path / "absent.sqlite3")
    database = tmp_path / "news.sqlite3"
    create_db(database)
    with pytest.raises(ValueError, match="bound"):
        inspect_live_news(database, limit=0)


def test_missing_optional_legacy_tables_is_not_silent_evidence(tmp_path: Path) -> None:
    database = tmp_path / "legacy.sqlite3"
    feed = b"<rss/>"
    with sqlite3.connect(database) as db:
        db.executescript("""
        CREATE TABLE captures (
          id INTEGER PRIMARY KEY, source TEXT, feed_url TEXT,
          fetched_at TEXT, sha256 TEXT, payload BLOB
        );
        CREATE TABLE articles (
          id INTEGER PRIMARY KEY, source TEXT, url TEXT, published_at TEXT,
          detected_at TEXT, capture_id INTEGER
        );
        """)
        db.execute("INSERT INTO captures VALUES (1,?,?,?,?,?)", (
            "fed_all", "https://example.org/rss", "2026-11-23T14:00:00+09:00",
            hashlib.sha256(feed).hexdigest(), feed,
        ))
        db.execute("INSERT INTO articles VALUES (1,?,?,?,?,?)", (
            "fed_all", "https://example.org/1", "2026-11-23T14:00:00+09:00",
            "2026-11-23T14:00:00+09:00", 1,
        ))
    report = inspect_live_news(database)
    blockers = report["rows"][0]["blockers"]
    assert "ORIGINAL_BODY_INCOMPLETE" in blockers
    assert "STORY_GROUP_UNAVAILABLE" in blockers
    assert "NO_COMPLETED_AUTOMATIC_GEMMA_ANALYSIS" in blockers
