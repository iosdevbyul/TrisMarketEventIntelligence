"""Market-pressure analysis across multiple news and indicator sources."""
from __future__ import annotations

import json
from datetime import date

import pytest

from tris_market_event_intelligence.impact import ImpactAnalysisError
from tris_market_event_intelligence.market_evidence_pressure import (
    MarketEvidencePacket,
    analyze_market_evidence,
)


class FakeMarketProvider:
    model_id = "mock-market-research"

    def __init__(self, result: dict | None = None) -> None:
        self.inputs: list[dict] = []
        self.result = result

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        envelope = json.loads(user)
        self.inputs.append(envelope)
        assert "stockPricesAvailable" in envelope
        assert envelope["stockPricesAvailable"] is False
        assert envelope["outcomesAvailable"] is False
        return json.dumps(self.result or {"impacts": []})


def sample() -> MarketEvidencePacket:
    return MarketEvidencePacket.model_validate({
        "eventId": "macro-news-research-demo",
        "subject": "Inflation and the cost of borrowing",
        "asOfDateExclusive": "2024-02-15",
        "evidences": [
            {
                "sourceId": "indicator-one",
                "kind": "ECONOMIC_INDICATOR",
                "publishedDate": "2024-02-13",
                "sourceName": "Example official indicator source",
                "sourceUrl": "https://www.bls.gov/news.release/cpi.toc.htm",
                "excerpt": (
                    "Consumer prices rose 0.3 percent in the reference month "
                    "according to the official index release."
                ),
                "verification": "VERIFIED_SOURCE",
                "indicatorName": "Consumer prices",
                "indicatorValue": "0.3 percent",
                "indicatorPeriod": "reference month",
            },
            {
                "sourceId": "news-one",
                "kind": "NEWS",
                "publishedDate": "2024-02-14",
                "sourceName": "Example financial report",
                "sourceUrl": "https://www.example.org/industry-news",
                "excerpt": (
                    "Companies reported that higher borrowing costs could "
                    "delay investments in new facilities."
                ),
                "verification": "REPORTED_UNVERIFIED",
            },
            {
                "sourceId": "future-one",
                "kind": "NEWS",
                "publishedDate": "2024-02-16",
                "sourceName": "Future article",
                "sourceUrl": "https://www.example.org/future",
                "excerpt": (
                    "Future earnings release discussed later corporate "
                    "guidance not yet known at this cutoff."
                ),
                "verification": "REPORTED_UNVERIFIED",
            },
        ],
    })


def impact(*, excerpt: str, uncertainty: str = "HIGH") -> dict:
    return {
        "market": "US",
        "sector": "FINANCIALS",
        "direction": "MIXED",
        "horizon": "SHORT_TERM",
        "rationale": (
            "The reported rise in borrowing costs could increase financing "
            "expenses for companies and conditionally affect US financial "
            "sector earnings expectations, although the market consensus "
            "and whether the information was already priced in are unknown."
        ),
        "evidence_excerpt": excerpt,
        "uncertainty": uncertainty,
        "drivers": ["Conditional financing costs"],
        "entities": [],
    }


def test_cutoff_excludes_future_news_and_keeps_multiple_source_kinds() -> None:
    provider = FakeMarketProvider()
    output = analyze_market_evidence(sample(), provider)
    assert output["impacts"] == []
    assert output["visibleSourceIds"] == ["indicator-one", "news-one"]
    assert output["excludedFutureSourceIds"] == ["future-one"]
    assert output["modelCalls"] == 1
    assert output["historicalPITVerified"] is False
    assert output["marketReturnsAnalyzed"] is False
    assert len(provider.inputs[0]["evidence"]) == 2
    assert {x["kind"] for x in provider.inputs[0]["evidence"]} == {
        "NEWS", "ECONOMIC_INDICATOR",
    }


def test_verified_source_quotation_is_bound_to_original_source_id() -> None:
    provider = FakeMarketProvider({
        "impacts": [impact(excerpt="higher borrowing costs could delay investments")]
    })
    output = analyze_market_evidence(sample(), provider)
    assert output["impacts"][0]["supportingSourceIds"] == ["news-one"]
    assert output["impacts"][0]["pressureNotPriceForecast"] is True


def test_unverified_evidence_cannot_claim_low_uncertainty() -> None:
    provider = FakeMarketProvider({
        "impacts": [
            impact(excerpt="higher borrowing costs could delay investments",
                   uncertainty="LOW"),
        ],
    })
    with pytest.raises((ImpactAnalysisError, ValueError)):
        analyze_market_evidence(sample(), provider)


def test_future_quote_is_rejected_even_when_in_packet() -> None:
    provider = FakeMarketProvider({
        "impacts": [impact(excerpt="Future earnings release")]
    })
    with pytest.raises(ImpactAnalysisError, match="unavailable"):
        analyze_market_evidence(sample(), provider)


def test_indicators_reject_fabricated_consensus_or_figures() -> None:
    raw = sample().model_dump(mode="json")
    raw["evidences"][0]["expectation"] = "0.1 percent"
    with pytest.raises(ValueError, match="UNSUPPORTED_EXPECTATION"):
        MarketEvidencePacket.model_validate(raw)


def test_empty_cutoff_rejected_without_model_call() -> None:
    raw = sample().model_dump(mode="json")
    raw["asOfDateExclusive"] = date(2024, 1, 1).isoformat()
    provider = FakeMarketProvider()
    with pytest.raises(ValueError, match="MARKET_EVIDENCE_EMPTY_AT_CUTOFF"):
        analyze_market_evidence(MarketEvidencePacket.model_validate(raw), provider)
    assert provider.inputs == []


def test_exact_cutoff_uses_utc_instant_not_publisher_calendar_day() -> None:
    raw = sample().model_dump(mode="json")
    raw["asOfDateExclusive"] = "2024-02-14"
    raw["asOfTimestampExclusive"] = "2024-02-14T10:00:00+09:00"
    # Cutoff is 2024-02-14 01:00 UTC. The same local-date article
    # published at 00:30 UTC is visible; a 03:00 UTC article is not.
    raw["evidences"][0]["publishedAt"] = "2024-02-13T18:30:00-05:00"
    raw["evidences"][1]["publishedAt"] = "2024-02-14T00:30:00+00:00"
    raw["evidences"][2]["publishedAt"] = "2024-02-16T00:30:00+00:00"
    p = MarketEvidencePacket.model_validate(raw)
    provider = FakeMarketProvider()
    report = analyze_market_evidence(p, provider)
    assert report["visibleSourceIds"] == ["indicator-one", "news-one"]
    assert report["excludedFutureSourceIds"] == ["future-one"]


def test_exact_cutoff_refuses_date_only_sources() -> None:
    raw = sample().model_dump(mode="json")
    raw["asOfTimestampExclusive"] = "2024-02-15T09:30:00+00:00"
    with pytest.raises(ValueError, match="PRECISE_REPLAY_REQUIRES_SOURCE_TIMESTAMPS"):
        MarketEvidencePacket.model_validate(raw)
