"""Synthetic retrospective research contract tests; not a claimed AI win."""
from __future__ import annotations

import json

import httpx

from tris_market_event_intelligence.experience_memory import digest
from tris_market_event_intelligence.historical_experience_abc_v2_cli import (
    compare,
    generate_rule_v2,
    retrieve,
)


def experience(article_id: int, verdict: str, exit_day: str) -> dict:
    row = {
        "articleId": article_id, "bodySha256": str(article_id),
        "publishedAt": "2026-01-01T10:00:00+00:00",
        "exitDay": exit_day, "status": "RETROSPECTIVE_UNSEALED_OBSERVATION",
        "verdict": verdict, "relativeVerdict": verdict,
        "target": "US technology stocks", "direction": "UP",
        "actualDirection": "UP" if verdict == "CORRECT" else "DOWN",
        "horizonSessions": 1, "reason": "technology earnings",
    }
    return {"id": digest(row), "outcome": row,
            "reflection": {"hypothesis": "earnings expectations may matter"}}


def packet() -> dict:
    return {
        "title": "Technology earnings outlook",
        "publishedAt": "2026-02-01T10:00:00+00:00",
        "sourcePassages": [{"id": "E1", "text": "Technology earnings may grow"}],
        "instruments": [],
    }


def test_retrieval_filters_same_article_body_and_future_outcomes():
    rows = [
        experience(1, "CORRECT", "2026-01-20"),
        experience(2, "INCORRECT", "2026-02-05"),
        experience(3, "INCORRECT", "2026-01-18"),
    ]
    selected = retrieve(packet(), 1, "3", rows)
    assert selected == []
    selected = retrieve(packet(), 99, "unused", rows)
    assert {x["articleId"] for x in selected} == {1, 3}
    assert all(x["historicallyBackfilled"] for x in selected)


def test_rule_v2_retries_distinct_article_support_without_relaxing_gate():
    experiences = [
        experience(1, "CORRECT", "2026-01-06"),
        experience(2, "INCORRECT", "2026-01-07"),
        experience(3, "CORRECT", "2026-01-08"),
    ]
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        user = json.loads(payload["messages"][1]["content"])
        calls.append(user)
        supporting = ["R1", "R1"] if len(calls) == 1 else ["R1", "R2"]
        return httpx.Response(200, json={"message": {"content": json.dumps({
            "rules": [{
                "condition": "Preannouncement earnings guidance",
                "analysis_action": "Consider expectations before directional forecast",
                "exceptions": "Guidance too ambiguous",
                "hypothesis": "Guidance surprises can affect prices",
                "supporting_ids": supporting,
                "counterexample_ids": [],
            }],
        })}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        rules, errors = generate_rule_v2(
            client, "gemma4:e2b", experiences, "2026-03-01T00:00:00+00:00")
    assert len(calls) == 2
    assert len(errors) == 1
    assert "Invalid" in errors[0]["reason"]
    assert len(rules) == 1
    assert len({next(x["outcome"]["articleId"] for x in experiences if x["id"] == i)
                for i in rules[0]["supporting_ids"]}) == 2
    assert rules[0]["promotionAllowed"] is False


def test_paired_union_tracks_unselected_targets(tmp_path):
    price_path = tmp_path / "prices.csv"
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
             "2026-01-09", "2026-01-12", "2026-01-13"]
    price_path.write_text(
        "symbol,date,adjusted_close,source\n" +
        "".join(f"AAA,{day},{100 + i},test\n" for i, day in enumerate(dates)) +
        "".join(f"SPY,{day},{100},test\n" for day in dates),
        encoding="utf-8",
    )
    info = {"name": "AAA", "market": "XNYS", "symbol": "AAA", "benchmark": "SPY"}
    forecast = {"instrument": info, "direction": "UP",
                "mechanism": "expectations", "counterargument": "uncertain",
                "evidenceQuote": "earnings"}
    identity = {
        "articleId": 50, "bodySha256": "source",
        "packet": {"publishedAt": "2026-01-06T14:00:00+00:00"},
    }
    result = compare([{
        "identity": identity, "arms": {"A": [forecast], "B": []},
        "generatedAt": "2026-11-28T00:00:00+00:00", "recordSha256": "record",
    }], price_path)
    a = result["metrics"]["A:1d:verdict"]
    b = result["metrics"]["B:1d:verdict"]
    assert a["universe"] == b["universe"] == 1
    assert a["scored"] == 1
    assert b["scored"] == 0
    assert b["counts"]["NOT_SELECTED"] == 1
    assert b["conditionalAccuracy"] is None
    assert result["promotionAllowed"] is False
