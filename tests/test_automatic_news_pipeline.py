"""Automatic news-to-Gemma pipeline offline regression tests."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from unittest.mock import patch

from tris_market_event_intelligence.automatic_news_pipeline import process
from tris_market_event_intelligence.dashboard_server import snapshot


def test_news_analysis_persisted_and_idempotent(tmp_path):
    path = tmp_path / "news.sqlite3"
    body = "The central bank changed its policy guidance and discussed inflation."
    digest = hashlib.sha256(body.encode()).hexdigest()
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE articles (
          id INTEGER PRIMARY KEY, source TEXT, url TEXT, title TEXT, summary TEXT,
          published_at TEXT, detected_at TEXT
        );
        CREATE TABLE article_full_text (
          article_id INTEGER PRIMARY KEY, status TEXT, body TEXT, body_sha256 TEXT,
          sparse_pages TEXT
        );
        """)
        db.execute("INSERT INTO articles VALUES (1,?,?,?,?,?,?)", (
            "fed_monetary", "https://example.org/1", "New guidance", "summary",
            "2026-11-20T10:00:00+00:00", "2026-11-20T10:01:00+00:00",
        ))
        db.execute("INSERT INTO article_full_text VALUES (1,'COMPLETE',?,?,?)",
                   (body, digest, "[]"))
    result = {
        "summary": "Guidance change", "limitations": "Conditional impact",
        "impacts": [{
            "target": "Technology", "direction": "DOWN", "reason": "Higher discount rates",
            "evidence_id": "E1", "evidence_quote": body,
        }],
    }
    with patch(
        "tris_market_event_intelligence.automatic_news_pipeline.ask_gemma",
        return_value=result,
    ) as ask:
        first = process(path, collect=False, limit=3)
        second = process(path, collect=False, limit=3)
    assert first["completed"] == 1
    assert second["completed"] == 0
    assert ask.call_count == 1
    report = snapshot(path)
    assert report["articles"][0]["automaticStatus"] == "COMPLETED"
    assert report["automaticAnalysisCounts"] == {"COMPLETED": 1, "FAILED": 0}
    assert report["automaticFailureCodes"] == {}
    saved = json.loads(report["articles"][0]["automaticAnalysis"])
    assert {key: saved[key] for key in result} == result
    assert saved["_repeatability"]["status"] == "SINGLE_RUN_UNVERIFIED"
    assert saved["_repeatability"]["semanticCorrectnessVerified"] is False
    with sqlite3.connect(path) as db:
        assert db.execute(
            "SELECT attempts FROM automatic_news_analyses"
        ).fetchone()[0] == 1


def test_missing_fulltext_does_not_invoke_model(tmp_path):
    path = tmp_path / "news.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE articles (
          id INTEGER PRIMARY KEY, source TEXT, url TEXT, title TEXT, summary TEXT,
          published_at TEXT, detected_at TEXT
        );
        INSERT INTO articles VALUES
          (1,'fed','https://example.org','Headline',NULL,NULL,'2026-11-20');
        """)
    with patch(
        "tris_market_event_intelligence.automatic_news_pipeline.ask_gemma"
    ) as ask:
        report = process(path, collect=False)
    assert report["completed"] == 0
    ask.assert_not_called()


def test_error_is_saved_and_not_immediately_retried(tmp_path):
    path = tmp_path / "news.sqlite3"
    body = "A sufficiently long source sentence of new public information."
    digest = hashlib.sha256(body.encode()).hexdigest()
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE articles (
          id INTEGER PRIMARY KEY, source TEXT, url TEXT, title TEXT, summary TEXT,
          published_at TEXT, detected_at TEXT
        );
        CREATE TABLE article_full_text (
          article_id INTEGER PRIMARY KEY, status TEXT, body TEXT, body_sha256 TEXT,
          sparse_pages TEXT
        );
        INSERT INTO articles VALUES
          (1,'fed','https://example.org','Headline',NULL,NULL,'2026-11-20');
        """)
        db.execute("INSERT INTO article_full_text VALUES (1,'COMPLETE',?,?,?)",
                   (body, digest, "[]"))
    with patch(
        "tris_market_event_intelligence.automatic_news_pipeline.ask_gemma",
        side_effect=ValueError("invalid model response"),
    ) as ask:
        first = process(path, collect=False)
        second = process(path, collect=False)
    assert first["failed"] == 1
    assert second["failed"] == 0
    assert ask.call_count == 1
    assert snapshot(path)["articles"][0]["automaticStatus"] == "FAILED"


def test_degraded_collector_is_not_reported_as_healthy(tmp_path):
    path = tmp_path / "news.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE articles (id INTEGER PRIMARY KEY)")
    with patch(
        "tris_market_event_intelligence.automatic_news_pipeline.run_cycle",
        return_value={"status": "degraded", "successful_sources": 7, "total_sources": 8},
    ):
        result = process(path, collect=True)
    assert result["status"] == "DEGRADED_COLLECTION"
    assert result["collectionHealthy"] is False
    assert result["collectionStatus"] == "degraded"
    assert result["completed"] == 0
    assert result["failed"] == 0


def test_newest_news_not_starved_by_existing_historical_backlog(tmp_path):
    path = tmp_path / "news.sqlite3"
    body = "The central bank described a significant updated policy decision."
    digest = hashlib.sha256(body.encode()).hexdigest()
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE articles (
          id INTEGER PRIMARY KEY, source TEXT, url TEXT, title TEXT, summary TEXT,
          published_at TEXT, detected_at TEXT
        );
        CREATE TABLE article_full_text (
          article_id INTEGER PRIMARY KEY, status TEXT, body TEXT, body_sha256 TEXT,
          sparse_pages TEXT
        );
        """)
        for article_id in range(1, 13):
            db.execute(
                "INSERT INTO articles VALUES (?,?,?,?,?,?,?)",
                (article_id, "fed_all", f"https://example.org/{article_id}",
                 f"Policy {article_id}", None, "2026-11-20T00:00:00+00:00",
                 "2026-11-20T00:01:00+00:00"),
            )
            db.execute(
                "INSERT INTO article_full_text VALUES (?,'COMPLETE',?,?,?)",
                (article_id, body, digest, "[]"),
            )
    answer = {"summary": "Policy news", "limitations": "Uncertain impact", "impacts": []}
    with patch(
        "tris_market_event_intelligence.automatic_news_pipeline.ask_gemma",
        return_value=answer,
    ):
        report = process(path, collect=False, limit=3)
    assert report["completed"] == 3
    selected = [row["articleId"] for row in report["results"]]
    assert 12 in selected
    assert 1 in selected
    assert len(set(selected)) == 3


def test_sec_only_analyses_include_partial_scope_and_exclude_news(tmp_path):
    path = tmp_path / "news.sqlite3"
    body = "The issuer announced a material contract."
    digest = hashlib.sha256(body.encode()).hexdigest()
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE articles (
          id INTEGER PRIMARY KEY, source TEXT, url TEXT, title TEXT, summary TEXT,
          published_at TEXT, detected_at TEXT
        );
        CREATE TABLE article_full_text (
          article_id INTEGER PRIMARY KEY, status TEXT, body TEXT, body_sha256 TEXT,
          sparse_pages TEXT, format TEXT, error_code TEXT
        );
        """)
        for article_id, source, fmt, error in (
            (1, "bbc_business", "HTML", None),
            (2, "sec_8k_789019", "SEC_BUNDLE_PARTIAL", "SEC_EXHIBIT_EMPTY_CONTENT"),
            (3, "sec_8k_1045810", "SEC_BUNDLE", None),
            (4, "sec_8k_2488", "HTML", None),
        ):
            db.execute("INSERT INTO articles VALUES (?,?,?,?,?,?,?)", (
                article_id, source, "https://example.org/filing", "Disclosure", None,
                "2026-11-20T10:00:00+00:00", "2026-11-20T10:01:00+00:00",
            ))
            db.execute("INSERT INTO article_full_text VALUES (?,'COMPLETE',?,?,?,?,?)",
                       (article_id, body, digest, "[]", fmt, error))
    def answer(_client, _model, request):
        assert request["article"]["source"].startswith("sec_8k_")
        return {"summary": "Material contract", "limitations": "Limited", "impacts": []}
    with patch(
        "tris_market_event_intelligence.automatic_news_pipeline.ask_gemma",
        side_effect=answer,
    ) as ask:
        first = process(path, collect=False, sec_only=True, limit=24)
        second = process(path, collect=False, sec_only=True, limit=24)
    assert first["completed"] == 2
    assert second["completed"] == 0
    assert ask.call_count == 2
    assert not first["forecastPrecommitmentsEnabled"]
    with sqlite3.connect(path) as db:
        raw = db.execute(
            "SELECT analysis_json FROM automatic_news_analyses WHERE article_id=2"
        ).fetchone()[0]
    scope = json.loads(raw)["_sourceScope"]
    assert scope["scope"] == "PRIMARY_WITH_PARTIAL_EXHIBITS"
    assert scope["missingEvidence"] == "SEC_EXHIBIT_EMPTY_CONTENT"
    assert scope["historicalResearchOnly"]
