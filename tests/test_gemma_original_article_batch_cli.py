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
        assert "regional market and company activity" in payload["messages"][1]["content"]
        assert payload["model"] == "gemma4:e2b"
        return httpx.Response(
            200,
            json={"message": {"content": json.dumps({
                "summary": "Regional market and company activity",
                "impacts": [{
                    "target": "FINANCIALS",
                    "direction": "UNCERTAIN",
                    "reason": "The report is broad and no prices are provided.",
                    "evidence_id": "E1",
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
    assert saved["analysis"]["impacts"][0]["evidence_quote"] in BODY
    assert saved["analysis"]["impacts"][0]["evidence_id"] == "E1"
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
                "evidence_id": "E999",
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
    with pytest.raises(ValueError, match="EVIDENCE_ID_NOT_IN_SOURCE"):
        validate_answer(json.dumps({
            "summary": "", "limitations": "",
            "impacts": [{
                "target": "TECH", "direction": "UP", "reason": "Justification",
                "evidence_id": "E999",
            }],
        }), {"E1": BODY})


def test_invalid_first_response_repairs_with_same_source(tmp_path: Path) -> None:
    db = database(tmp_path / "news.sqlite3")
    attempts: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        attempts.append(payload)
        if len(attempts) == 1:
            return httpx.Response(
                200, json={"message": {"content": '{"summary":'}}
            )
        return httpx.Response(
            200, json={"message": {"content": json.dumps({
                "summary": "Source-based review",
                "limitations": "No observed price returns",
                "impacts": [{
                    "target": "FINANCIALS", "direction": "UNCERTAIN",
                    "reason": "No evidence about future returns",
                    "evidence_id": "E1",
                }],
            })}},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = analyze(db, tmp_path / "reviews", client=client)
    assert result["completed"] == 1 and result["failed"] == 0
    assert len(attempts) == 2
    assert all(payload["format"]["type"] == "object" for payload in attempts)
    assert "regional market and company activity" in attempts[0]["messages"][1]["content"]
    assert "regional market and company activity" in attempts[1]["messages"][1]["content"]


def test_failed_twice_has_specific_diagnostic(tmp_path: Path, capsys: object) -> None:
    db = database(tmp_path / "news.sqlite3")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"message": {"content": "{}"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = analyze(db, tmp_path / "reviews", client=client)
    assert result["completed"] == 0 and result["failed"] == 1
    assert result["failureCodes"] == {"SUMMARY_MISSING": 1}
    assert calls == 2
    assert '"errorCode": "SUMMARY_MISSING"' in capsys.readouterr().out


def test_ollama_schema_fallback(tmp_path: Path) -> None:
    db = database(tmp_path / "news.sqlite3")
    formats: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        formats.append(payload["format"])
        if isinstance(payload["format"], dict):
            return httpx.Response(422)
        return httpx.Response(
            200, json={"message": {"content": json.dumps({
                "summary": "Financial overview",
                "limitations": "No observed returns",
                "impacts": [],
            })}},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = analyze(db, tmp_path / "reviews", client=client)
    assert result["completed"] == 1
    assert formats[1] == "json"


def test_source_quote_is_attached_by_python_not_generated_by_gemma() -> None:
    from tris_market_event_intelligence.gemma_original_article_batch_cli import (
        evidence_passages,
    )

    source = ("US central bank announced a policy decision. "
              "The report does not provide realized stock returns.")
    passages = evidence_passages(source)
    assert passages[0]["id"] == "E1"
    assert passages[0]["text"] in source
    answer = validate_answer(json.dumps({
        "summary": "Policy decision", "limitations": "No prices",
        "impacts": [{
            "target": "US_MARKET", "direction": "UNCERTAIN",
            "reason": "The report lacks forward price data",
            "evidence_id": "E1",
        }],
    }), {item["id"]: item["text"] for item in passages})
    assert answer["impacts"][0]["evidence_quote"] == passages[0]["text"]


def test_oversized_source_has_bounded_deterministic_passages() -> None:
    from tris_market_event_intelligence.gemma_original_article_batch_cli import (
        evidence_passages,
    )

    source = "Financial markets face uncertainty. " * 1000
    first = evidence_passages(source)
    assert first == evidence_passages(source)
    assert 1 <= len(first) <= 24
    assert all(item["text"] in source[:12000] for item in first)


@pytest.mark.parametrize("done_reason", ["length", "stop"])
def test_empty_generation_retries_with_thinking_disabled(tmp_path, done_reason):
    db = database(tmp_path / "news.sqlite3")
    attempts = []

    def handler(request):
        payload = json.loads(request.content)
        attempts.append(payload)
        assert payload["think"] is False
        assert payload["options"]["num_ctx"] == 8192
        if len(attempts) == 1:
            return httpx.Response(200, json={
                "done_reason": done_reason, "message": {"content": "", "thinking": "internal"},
            })
        return httpx.Response(200, json={
            "done_reason": "stop", "message": {"content": json.dumps({
                "summary": "Source-based review", "limitations": "Limited evidence", "impacts": [],
            })},
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = analyze(db, tmp_path / "reviews", client=client)
    assert result["completed"] == 1
    assert len(attempts) == 2
    assert attempts[1]["format"] == "json"
    assert attempts[1]["options"]["num_predict"] > attempts[0]["options"]["num_predict"]
    assert attempts[0]["messages"][1] == attempts[1]["messages"][1]
    assert attempts[0]["options"]["seed"] != attempts[1]["options"]["seed"]
