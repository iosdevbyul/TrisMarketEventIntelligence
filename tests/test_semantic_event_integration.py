import json

import pytest

from tris_market_event_intelligence.semantic_event_integration_cli import hints, retrieve, validate


def test_bilingual_retrieval():
    base = {"detectedAt": "2026-11-13T13:00:00+00:00",
            "publisherDeclaredAt": "2026-11-13T12:00:00+00:00", "secAcceptanceAt": None,
            "tokens": {"results", "quarterly"}, "bodyHints": {"NVDA", "EARNINGS"}}
    official = {**base, "eventId": "sec", "kind": "SEC",
                "primaryHints": hints("NVIDIA quarterly results")}
    news = {**base, "eventId": "news", "kind": "NEWS",
            "primaryHints": hints("엔비디아 실적")}
    pairs = list(retrieve([official, news]))
    assert len(pairs) == 1
    assert pairs[0][2]["primaryHints"] == ["EARNINGS", "NVDA"]


def test_exact_quote_validation():
    left = [{"id": "1", "text": "NVIDIA announced quarterly results.", "start": 100,
             "bodySha256": "a", "rawSha256": "b", "observedAt": "stamp"}]
    right = [{**left[0], "text": "엔비디아가 분기 실적을 공식 발표했다."}]
    answer = {"relation": "SAME_EVENT", "left_id": "1", "right_id": "1",
              "left_quote": "NVIDIA announced quarterly results.",
              "right_quote": "엔비디아가 분기 실적을 공식 발표했다.",
              "reason": "candidate only", "new_fact": ""}
    result = validate(json.dumps(answer), left, right)
    assert result["anchors"]["left"]["start"] == 100
    assert not result["humanVerified"]
    assert not result["predictionEligible"]
    answer["right_quote"] = "Invented quotation not found."
    with pytest.raises(ValueError, match="UNSUPPORTED_RIGHT_QUOTE"):
        validate(json.dumps(answer), left, right)


def test_no_shared_entity_is_not_candidate():
    base = {"detectedAt": "2026-11-13T13:00:00+00:00",
            "publisherDeclaredAt": "2026-11-13T12:00:00+00:00", "secAcceptanceAt": None,
            "tokens": set(), "bodyHints": set()}
    assert not list(retrieve([
        {**base, "eventId": "a", "kind": "SEC", "primaryHints": {"NVDA"}},
        {**base, "eventId": "b", "kind": "NEWS", "primaryHints": {"AAPL"}},
    ]))
