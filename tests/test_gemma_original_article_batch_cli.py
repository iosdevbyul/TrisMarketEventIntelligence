"""Offline tests for direct original-document Gemma batch transfer."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from tris_market_event_intelligence.collector import Article, NewsStore
from tris_market_event_intelligence.gemma_original_article_batch_cli import (
    analyze,
    validate_answer,
    verified_articles,
)
from tris_market_event_intelligence.source_article_content import initialize

BODY = ("This report describes the regional market and company activity. " * 8)


def database(path: Path) -> Path:
    store = NewsStore(path)
    store.initialize()
    for index, status in enumerate(("COMPLETE", "REVIEW_REQUIRED", "NON_ARTICLE"), 1):
        article = Article(
            source="bbc_business",
            external_id=f"source-{index}",
            url=f"https://www.bbc.com/news/articles/article-{index}",
            title=f"Market article {index}",
            summary="RSS summary", published_at=None,
            detected_at="2026-11-21T00:00:00+00:00",
        )
        store.save(
            "bbc_business", "https://feeds.bbci.co.uk/news/business/rss.xml",
            b"<rss/>", [article], article.detected_at, None, None,
        )
    initialize(store)
    with store.connect() as db:
        for article_id, state in enumerate(("COMPLETE", "REVIEW_REQUIRED", "NON_ARTICLE"), 1):
            content = BODY if state != "NON_ARTICLE" else None
            digest = hashlib.sha256(content.encode()).hexdigest() if content else None
            db.execute(
                """INSERT INTO article_full_text
                    (article_id,status,observed_at,body,body_sha256,sparse_pages)
                    VALUES (?,?,?,?,?,?)""",
                (article_id, state, "2026-11-21T00:00:00+00:00",
                 content, digest, "[1]" if state == "REVIEW_REQUIRED" else "[]"),
            )
    return path


def test_verified_only_and_source_database_read_only(tmp_path: Path) -> None:
    db = database(tmp_path / "news.sqlite3")
    articles, coverage = verified_articles(db, 20)
    assert coverage["completeRows"] == 1
    assert coverage["selected"] == 1
    assert articles[0]["body"] == BODY
    assert articles[0]["articleId"] == 1


def test_model_gets_original_and_creates_resumable_json(tmp_path: Path) -> None:
    db = database(tmp_path / "news.sqlite3")
    inputs: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "127.0.0.1"
        payload = json.loads(request.content)
        inputs.append(payload)
        assert BODY in payload["messages"][1]["content"]
        assert payload["model"] == "gemma4:e2b"
        return httpx.Response(
            200,
            json={"message": {"content": json.dumps({
                "summary": "Regional market and company activity",
                "impacts": [{
                    "target": "FINANCIALS",
                    "direction": "UNCERTAIN",
                    "reason": "The report is broad and no prices are provided.",
                    "evidence_quote": "regional market and company activity",
                }],
                "limitations": "No market reaction observed",
            })}},
        )

    output = tmp_path / "results"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = analyze(db, output, client=client)
        second = analyze(db, output, client=client)
    assert first["completed"] == 1 and first["failed"] == 0
    assert second["completed"] == 0 and second["alreadyCompleted"] == 1
    assert len(inputs) == 1
    reports = list(output.glob("*.json"))
    assert len(reports) == 1
    saved = json.loads(reports[0].read_text(encoding="utf-8"))
    assert saved["article"]["bodySha256"] == hashlib.sha256(BODY.encode()).hexdigest()
    assert saved["analysis"]["impacts"][0]["direction"] == "UNCERTAIN"
    assert saved["priceReturnsEvaluated"] is False
    assert saved["predictionSealedBeforeOutcome"] is False
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM article_full_text").fetchone()[0] == 3


def test_invalid_quote_cannot_be_saved(tmp_path: Path) -> None:
    db = database(tmp_path / "news.sqlite3")
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": json.dumps({
            "summary": "Report", "limitations": "None",
            "impacts": [{
                "target": "TECH", "direction": "UP", "reason": "Hypothesis",
                "evidence_quote": "not in article at all",
            }],
        })}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = analyze(db, tmp_path / "results", client=client)
    assert report["failed"] == 1
    assert not list((tmp_path / "results").glob("*.json"))


def test_reject_corrupt_body_and_nonsource_quotes(tmp_path: Path) -> None:
    db = database(tmp_path / "news.sqlite3")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE article_full_text SET body_sha256='tampered' WHERE article_id=1")
    selected, coverage = verified_articles(db, 20)
    assert not selected
    assert coverage["invalidRowsExcluded"] == 1
    with pytest.raises(ValueError, match="QUOTE_NOT_IN_SOURCE"):
        validate_answer(json.dumps({
            "summary": "", "limitations": "",
            "impacts": [{
                "target": "TECH", "direction": "UP", "reason": "Justification",
                "evidence_quote": "fake",
            }],
        }), BODY)
