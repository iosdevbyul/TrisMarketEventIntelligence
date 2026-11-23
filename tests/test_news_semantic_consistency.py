"""No unreviewed model output may be interpreted as validated market impact."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from unittest.mock import patch

from tris_market_event_intelligence.automatic_news_pipeline import process
from tris_market_event_intelligence.news_semantic_consistency import compare_analyses
from tris_market_event_intelligence.dashboard_server import snapshot


def _analysis(target: str, direction: str) -> dict:
    return {
        "summary": "Policy publication",
        "limitations": "No demonstrated causal relationship",
        "impacts": [{
            "target": target, "direction": direction,
            "reason": "A conditional economic transmission mechanism",
            "evidence_id": "E1", "evidence_quote": "The policy report was published.",
        }],
    }


def test_exact_repeat_not_claimed_semantically_correct():
    output = compare_analyses(_analysis("Technology", "UNCERTAIN"),
                              _analysis("Technology", "UNCERTAIN"))
    assert output["status"] == "REPEATABLE_UNREVIEWED"
    assert output["exactTargetDirections"] is True
    assert output["semanticCorrectnessVerified"] is False


def test_disjoint_targets_explicitly_need_review():
    output = compare_analyses(_analysis("Technology", "UP"),
                              _analysis("Banking", "UP"))
    assert output["status"] == "REVIEW_REQUIRED"
    assert output["commonTargets"] == []
    assert output["firstOnlyTargets"] == ["technology"]
    assert output["secondOnlyTargets"] == ["banking"]
    assert output["directionConflicts"] == []


def test_direction_conflict_explicitly_needs_review():
    output = compare_analyses(_analysis("Technology", "UP"),
                              _analysis("Technology", "DOWN"))
    assert output["status"] == "REVIEW_REQUIRED"
    assert output["directionConflicts"] == [
        {"target": "technology", "first": "UP", "second": "DOWN"},
    ]


def test_automatic_worker_keeps_both_model_calls_for_review(tmp_path):
    path = tmp_path / "news.sqlite3"
    body = "The federal reserve reported a policy decision impacting markets."
    digest = hashlib.sha256(body.encode()).hexdigest()
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE articles (
          id INTEGER PRIMARY KEY, source TEXT, url TEXT, title TEXT,
          summary TEXT, published_at TEXT, detected_at TEXT
        );
        CREATE TABLE article_full_text (
          article_id INTEGER PRIMARY KEY, status TEXT, body TEXT,
          body_sha256 TEXT, sparse_pages TEXT
        );
        """)
        db.execute(
            "INSERT INTO articles VALUES (1,'fed_all','https://example.org/1',"
            "'Policy',NULL,NULL,'2026-11-23T00:00:00+00:00')"
        )
        db.execute(
            "INSERT INTO article_full_text VALUES (1,'COMPLETE',?,?,?)",
            (body, digest, "[]"),
        )
    with patch(
        "tris_market_event_intelligence.automatic_news_pipeline.ask_gemma",
        side_effect=[_analysis("Technology", "UP"),
                     _analysis("Technology", "DOWN")],
    ) as ask:
        result = process(path, collect=False, verification_runs=2)
    assert ask.call_count == 2
    assert result["completed"] == 1
    assert result["verificationRuns"] == 2
    with sqlite3.connect(path) as db:
        stored = json.loads(db.execute(
            "SELECT analysis_json FROM automatic_news_analyses"
        ).fetchone()[0])
    assert stored["_repeatability"]["status"] == "REVIEW_REQUIRED"
    assert stored["_repeatability"]["secondAnalysis"]["impacts"][0]["direction"] == "DOWN"
    record = snapshot(path)["articles"][0]
    assert record["automaticVerificationStatus"] == "REVIEW_REQUIRED"


def test_invalid_verification_setting_rejected(tmp_path):
    from pytest import raises
    with raises(ValueError, match="verification_runs"):
        process(tmp_path / "missing.sqlite3", collect=False, verification_runs=3)
