"""Read-only dashboard contract tests against representative collector schema."""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from tris_market_event_intelligence.dashboard_server import create_dashboard_app, snapshot


def test_snapshot_reads_real_article_rows_without_writes(tmp_path):
    path = tmp_path / "news.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE articles(
            id INTEGER PRIMARY KEY, source TEXT, url TEXT, title TEXT,
            summary TEXT, published_at TEXT, detected_at TEXT
        );
        CREATE TABLE fetch_state(
            source TEXT, last_checked_at TEXT, last_success_at TEXT,
            last_error TEXT
        );
        CREATE TABLE fetch_attempts(
            id INTEGER PRIMARY KEY, source TEXT, status TEXT,
            completed_at TEXT, returned INTEGER, inserted INTEGER
        );
        CREATE TABLE economic_events(
            id INTEGER PRIMARY KEY, article_id INTEGER, status TEXT, classification TEXT
        );
        CREATE TABLE market_impacts(
            id INTEGER PRIMARY KEY, event_id INTEGER, direction TEXT,
            sector TEXT, rationale TEXT, model_id TEXT
        );
        CREATE TABLE impact_runs(
            event_id INTEGER, outcome TEXT
        );
        INSERT INTO articles VALUES
        (1,'fed_monetary','https://example.org/1','Sample rate decision',
         'Sample summary','2026-11-21T10:00:00+00:00','2026-11-21T10:01:00+00:00');
        INSERT INTO fetch_state VALUES
        ('fed_monetary','2026-11-21T10:01:00+00:00','2026-11-21T10:01:00+00:00',NULL);
        INSERT INTO fetch_attempts VALUES
        (1,'fed_monetary','ok','2026-11-21T10:01:00+00:00',10,1);
        INSERT INTO economic_events VALUES (7,1,'VERIFIED','RATE_DECISION');
        INSERT INTO market_impacts VALUES
        (8,7,'DOWN','Technology','Discount-rate pressure','test-model');
        INSERT INTO impact_runs VALUES (7,'COMPLETED');
        """)
    before = path.read_bytes()
    report = snapshot(path)
    assert report["totals"]["articles"] == 1
    assert report["totals"]["impacts"] == 1
    assert report["articles"][0]["latestDirection"] == "DOWN"
    assert report["articles"][0]["latestRationale"] == "Discount-rate pressure"
    assert report["sources"][0]["recentAttempt"]["inserted"] == 1
    assert path.read_bytes() == before

    client = TestClient(create_dashboard_app(path))
    assert client.get("/dashboard").status_code == 200
    assert "News Intelligence Monitor" in client.get("/dashboard").text
    assert client.get("/api/v1/dashboard").json()["articles"][0]["id"] == 1
    assert client.get("/api/v1/dashboard?limit=301").status_code == 422


def test_snapshot_missing_db_is_explicitly_unavailable(tmp_path):
    path = tmp_path / "not-created.sqlite3"
    client = TestClient(create_dashboard_app(path))
    response = client.get("/api/v1/dashboard")
    assert response.status_code == 503
    assert not path.exists()


def test_snapshot_missing_optional_tables_is_not_fabricated(tmp_path):
    path = tmp_path / "news.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE articles (
            id INTEGER PRIMARY KEY, source TEXT, url TEXT, title TEXT,
            summary TEXT, published_at TEXT, detected_at TEXT
        )""")
        db.execute("""INSERT INTO articles VALUES
            (1,'example','https://example.org','Headline',NULL,NULL,
             '2026-11-21T10:01:00+00:00')""")
    report = snapshot(path)
    assert report["totals"]["events"] is None
    assert report["articles"][0]["impactCount"] is None
    assert report["articles"][0]["extractionCount"] is None
