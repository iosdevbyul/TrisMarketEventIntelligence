"""Synthetic unit tests, not empirical forecasting evidence."""
import sqlite3

import pytest

from tris_market_event_intelligence.cross_event_validator import compare_observed_pairs
from tris_market_event_intelligence.experience_memory import ExperienceMemory, digest
from tris_market_event_intelligence.experience_source_evidence import story_overlap
from tris_market_event_intelligence.reflection_evidence_validator import validate_review
from tris_market_event_intelligence.self_reflection import validate_reflection


def test_reflection_citation_does_not_establish_causality():
    answer = {
        "category": "MARKET_RESPONSE", "hypothesis": "Expectations may differ",
        "evidence_id": "E1", "missing_evidence": "Survey", "falsification": "Check survey",
    }
    result = validate_reflection(
        answer, {"E1": "Rate changed"},
        observed_at="2026-01-01T00:00:00+00:00",
        sealed_at="2026-01-01T01:00:00+00:00",
    )
    assert result["causalStatus"] == "UNVERIFIED"
    assert not result["semanticEntailmentVerified"]
    with pytest.raises(ValueError):
        validate_reflection(
            answer, {},
            observed_at="2026-01-01T00:00:00+00:00",
            sealed_at="2026-01-01T01:00:00+00:00",
        )


def test_evidence_review_rejects_invented_source():
    answer = {
        "verdict": "SOURCE_SUPPORT_CANDIDATE", "support_ids": ["E2"],
        "contradiction_ids": [], "reason": "Possible support", "missing_evidence": "Survey",
    }
    with pytest.raises(ValueError):
        validate_review(answer, {"E1": "Original source"})


def test_memory_rejects_unsealed_report(tmp_path):
    memory = ExperienceMemory(tmp_path / "memory.sqlite3")
    with pytest.raises(ValueError):
        memory.record({}, {"schemaVersion": "automatic_gemma_daily_outcomes_v1"}, {})
    assert memory.records() == []


def test_title_overlap():
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE articles(id INTEGER, title TEXT, url TEXT)")
        db.executemany("INSERT INTO articles VALUES (?, ?, ?)", [
            (1, "Central bank raises policy interest rate", "https://a.example"),
            (2, "Central bank raises policy interest rate", "https://b.example"),
            (3, "Company reports unexpected revenue growth", "https://c.example"),
        ])
        assert story_overlap(db, 2, [1])
        assert not story_overlap(db, 3, [1])


def test_paired_score_and_digest():
    event = {
        "articleId": 1, "bodySha256": "body", "publisherStatedAt": "2026-01-01T01:00:00+00:00",
        "target": "Example", "symbol": "EX", "benchmark": "SPY", "market": "US",
        "horizonSessions": 1,
    }
    payload = {
        "schemaVersion": "experience_paired_forecast_v1", "event": event,
        "ruleId": "rule", "sealedAt": "2026-01-01T02:00:00+00:00",
        "baseline": {"direction": "DOWN"}, "candidate": {"direction": "UP"},
    }
    pair = {**payload, "pairSha256": digest(payload)}
    report = {
        "schemaVersion": "committed_forecast_daily_price_evaluation_v1",
        "commitmentChainVerifiedLocally": True,
        "rows": [{
            **event, "status": "OBSERVED_UNAUTHENTICATED_PRECOMMITMENT",
            "entryDay": "2026-01-02", "actualDirection": "UP",
            "relativeActualDirection": "UP", "stockReturn": 0.03, "excessReturn": 0.02,
        }],
    }
    result = compare_observed_pairs([pair], report)
    assert result["metrics"]["rule:1d"]["candidateOnlyCorrect"] == 1
    assert result["promotionAllowed"] is False
    assert result["generalizationImprovementEstablished"] is False
    with pytest.raises(ValueError):
        compare_observed_pairs([{**pair, "ruleId": "changed"}], report)
