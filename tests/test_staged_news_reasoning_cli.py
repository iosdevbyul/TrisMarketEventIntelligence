"""Offline regression for the separately staged Gemma research path."""
from __future__ import annotations

import hashlib
import json
import sqlite3

import httpx
import pytest

from tris_market_event_intelligence.gemma_original_article_batch_cli import AnswerError
from tris_market_event_intelligence.staged_news_reasoning_cli import (
    _validated_event,
    analyze_staged,
    run_pilot,
)


def _event(kind: str, *, historic: bool = False) -> dict:
    return {
        "eventKind": kind, "actor": "Federal Reserve",
        "eventFact": "The committee announced a revised meeting schedule.",
        "evidence_id": "E1", "isHistoricalObservation": historic,
        "limitations": "No policy decision was issued.",
    }


def _request() -> dict:
    return {
        "article": {
            "articleId": 1, "source": "fed_monetary",
            "title": "Meeting schedule", "bodySha256": "0" * 64,
        },
        "body": (
            "The committee announced a revised meeting schedule. "
            "No policy decision was made in this announcement."
        ),
        "bodyTruncated": False,
    }


def test_schedule_event_does_not_generate_directional_market_impacts() -> None:
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={
            "message": {"content": json.dumps(_event("SCHEDULE_ONLY"))},
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = analyze_staged(client, "gemma4:e2b", _request())
    assert len(calls) == 1
    assert result["event"]["eventKind"] == "SCHEDULE_ONLY"
    assert result["event"]["eventGroundTruthVerified"] is False
    assert result["analysis"]["impacts"] == []
    assert "source" not in result["event"]
    assert result["priceDataUsed"] is False
    assert calls[0]["format"]["properties"]["eventKind"]["enum"]


def test_policy_event_requires_separate_source_backed_impact_step() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 1:
            return httpx.Response(200, json={
                "message": {"content": json.dumps(_event("POLICY_DECISION"))},
            })
        assert payload["messages"][0]["content"] != calls[0]["messages"][0]["content"]
        user = json.loads(payload["messages"][1]["content"])
        assert user["extractedEvent"]["eventKind"] == "POLICY_DECISION"
        assert user["extractedEvent"]["evidence_quote"] in (
            "The committee announced a new monetary policy decision. "
            "The policy action may affect funding conditions."
        )
        return httpx.Response(200, json={
            "message": {"content": json.dumps({
                "summary": "Policy decision affects funding conditions",
                "limitations": "Direction of market effects remains uncertain",
                "impacts": [{
                    "target": "Banking System", "direction": "UNCERTAIN",
                    "reason": "Funding cost changes can affect bank margins",
                    "evidence_id": "E1",
                }],
            })},
        })

    policy_request = {
        **_request(),
        "article": {**_request()["article"], "title": "Federal Reserve issues policy statement"},
        "body": ("The committee announced a new monetary policy decision. "
                 "The policy action may affect funding conditions."),
    }
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = analyze_staged(client, "gemma4:e2b", policy_request)
    assert len(calls) == 2
    assert result["analysis"]["impacts"][0]["target_key"] == "banking system"
    assert result["analysis"]["impacts"][0]["evidence_quote"] in policy_request["body"]
    assert result["eventAssessment"]["status"] == "UNVERIFIED"


def test_prior_market_observation_is_not_projected_as_new_return() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "message": {"content": json.dumps(
                _event("RETROSPECTIVE_MARKET", historic=True)
            )},
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = analyze_staged(client, "gemma4:e2b", _request())
    assert result["analysis"]["impacts"] == []


def test_malformed_event_evidence_and_timing_rejected() -> None:
    with pytest.raises(AnswerError, match="EVENT_EVIDENCE_INVALID"):
        _validated_event({**_event("POLICY_DECISION"), "evidence_id": "E999"},
                         {"E1": "Real exact source"})
    with pytest.raises(AnswerError, match="EVENT_TIMING_CONFLICT"):
        _validated_event(_event("RETROSPECTIVE_MARKET"),
                         {"E1": "Real exact source"})


def test_staged_pilot_writes_create_only_and_never_modifies_source_database(
    tmp_path,
) -> None:
    from test_gemma_original_article_batch_cli import database

    db_path = database(tmp_path / "news.sqlite3")
    first = hashlib.sha256(db_path.read_bytes()).hexdigest()
    output = tmp_path / "staged"
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={
            "message": {"content": json.dumps(_event("SCHEDULE_ONLY"))},
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        run1 = run_pilot(db_path, output, limit=5, client=client)
        run2 = run_pilot(db_path, output, limit=5, client=client)
    assert run1["completed"] == 1
    assert run2["alreadyCompleted"] == 1
    assert run1["failed"] == 0
    assert len(calls) == 1
    saved = json.loads(next(output.glob("*.json")).read_text())
    assert saved["schemaVersion"] == "gemma_original_article_review_v2"
    assert saved["reasoningSchema"] == "staged_event_impact_v1"
    assert saved["analysis"]["impacts"] == []
    assert saved["event"]["evidence_quote"]
    assert run1["productionDatabaseModified"] is False
    # SQLite may create a journal externally only for writers; pilot is strictly read-only.
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == first
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM article_full_text").fetchone()[0] == 3


def test_meeting_schedule_labeled_policy_decision_is_withheld_for_review() -> None:
    request = _request()
    request["article"]["title"] = (
        "Federal Reserve announces revised tentative schedule of "
        "Federal Open Market Committee meetings in 2027"
    )
    request["body"] = (
        "Federal Reserve announces revised tentative schedule of Federal "
        "Open Market Committee meetings in 2027. "
        "The Federal Reserve announced a revised tentative schedule of "
        "Federal Open Market Committee meetings in 2027."
    )
    calls = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(http_request.content))
        return httpx.Response(200, json={
            "message": {"content": json.dumps(_event("POLICY_DECISION"))},
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = analyze_staged(client, "gemma4:e2b", request)

    assert len(calls) == 1
    assert result["event"]["eventKind"] == "POLICY_DECISION"
    assert result["eventAssessment"]["status"] == "REVIEW_REQUIRED"
    assert "SOURCE_SCHEDULE_KIND_CONFLICT" in result["eventAssessment"]["flags"]
    assert result["impactDecisionStatus"] == "WITHHELD_EVENT_REVIEW"
    assert result["analysis"]["impacts"] == []
    assert result["eventFactVerified"] is False


def test_fomc_minutes_policy_decision_requires_timing_review() -> None:
    request = _request()
    request["article"]["title"] = (
        "Minutes of the Federal Open Market Committee, September 15-16, 2026"
    )
    request["body"] = (
        "Minutes of the Federal Open Market Committee. "
        "Treasury yields increased over the intermeeting period."
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "message": {"content": json.dumps(_event("POLICY_DECISION"))},
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = analyze_staged(client, "gemma4:e2b", request)

    assert result["impactDecisionStatus"] == "WITHHELD_EVENT_REVIEW"
    assert result["eventAssessment"]["flags"] == [
        "MINUTES_DECISION_TIMING_REVIEW"
    ]
    assert result["analysis"]["impacts"] == []
