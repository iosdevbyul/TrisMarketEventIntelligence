"""Two-pass event understanding and qualitative pressure validation."""
from __future__ import annotations

import json

import pytest

from tris_market_event_intelligence.impact import ImpactAnalysisError
from tris_market_event_intelligence.market_evidence_pressure import MarketEvidencePacket
from tris_market_event_intelligence.market_pressure_reasoning import (
    analyze_pressure_two_stage,
)


def example() -> MarketEvidencePacket:
    return MarketEvidencePacket.model_validate({
        "eventId": "multi-source-market",
        "subject": "Financing conditions and industry investment",
        "asOfDateExclusive": "2024-02-15",
        "evidences": [
            {
                "sourceId": "s1", "kind": "ECONOMIC_INDICATOR",
                "publishedDate": "2024-02-13",
                "sourceName": "Official price release",
                "sourceUrl": "https://www.bls.gov/news.release/cpi.toc.htm",
                "excerpt": (
                    "The official agency release reported that consumer prices "
                    "increased 0.3 percent during the reference month."
                ),
                "verification": "VERIFIED_SOURCE",
                "indicatorName": "Consumer price index",
                "indicatorValue": "0.3 percent",
            },
            {
                "sourceId": "s2", "kind": "NEWS",
                "publishedDate": "2024-02-14",
                "sourceName": "Industry report",
                "sourceUrl": "https://www.example.org/news",
                "excerpt": (
                    "A news report said that companies might postpone "
                    "expansion plans because of higher financing costs."
                ),
                "verification": "REPORTED_UNVERIFIED",
            },
            {
                "sourceId": "future", "kind": "NEWS",
                "publishedDate": "2024-02-20",
                "sourceName": "Future earnings",
                "sourceUrl": "https://www.example.org/future",
                "excerpt": (
                    "A later financial release described a surprise increase "
                    "in earnings that was not yet available."
                ),
                "verification": "REPORTED_UNVERIFIED",
            },
        ],
    })


FACT_QUOTE = "consumer prices increased 0.3 percent"
NEWS_QUOTE = "companies might postpone expansion plans"


def first_pass() -> dict:
    return {
        "claims": [
            {"sourceId": "s1", "excerpt": FACT_QUOTE,
             "factKind": "REPORTED_FACT", "novelty": "UNKNOWN", "entities": []},
            {"sourceId": "s2", "excerpt": NEWS_QUOTE,
             "factKind": "FORECAST", "novelty": "UNKNOWN", "entities": []},
        ],
    }


def impact() -> dict:
    return {
        "market": "US", "sector": "BROAD_MARKET",
        "direction": "NEGATIVE", "horizon": "SHORT_TERM",
        "rationale": (
            "The published price index could constrain rate-cut expectations, "
            "raising the discount rate applied to future earnings and "
            "potentially weighing on US equity valuations; the market "
            "consensus, positioning, and priced-in effects are unknown."
        ),
        "evidence_excerpt": FACT_QUOTE,
        "uncertainty": "HIGH",
        "drivers": ["conditional discount rate effect"],
        "entities": [], "targetType": "MARKET", "targetName": "US equities",
        "strength": "UNASSESSED",
        "strengthRationale": (
            "The release alone cannot determine magnitude or positioning."
        ),
        "transmissionChannel": "DISCOUNT_RATE",
        "claimIndexes": [0],
        "countervailingFactors": ["other demand factors may offset impact"],
    }


class QueueProvider:
    model_id = "mock-two-stage"

    def __init__(self, facts: dict | None = None, impacts: dict | None = None):
        self.facts = facts if facts is not None else first_pass()
        self.impacts = impacts if impacts is not None else {"impacts": [impact()]}
        self.prompts: list[dict] = []

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        self.prompts.append(json.loads(user))
        return json.dumps(self.facts if len(self.prompts) == 1 else self.impacts)


def test_two_stage_market_pressure_is_grounded_and_excludes_future() -> None:
    p = QueueProvider()
    result = analyze_pressure_two_stage(example(), p)
    assert result["modelCalls"] == 2
    assert result["visibleSourceIds"] == ["s1", "s2"]
    assert result["excludedFutureSourceIds"] == ["future"]
    assert result["impacts"][0]["direction"] == "NEGATIVE"
    assert result["impacts"][0]["strength"] == "UNASSESSED"
    assert result["impacts"][0]["supportingSourceIds"] == ["s1"]
    assert result["historicalPITVerified"] is False
    assert result["marketReturnsAnalyzed"] is False
    assert len(p.prompts) == 2
    assert "later financial release" not in json.dumps(p.prompts)
    assert "outcomeLabelsAvailable" in p.prompts[1]


def test_fact_hallucination_rejected_before_second_llm_call() -> None:
    facts = first_pass()
    facts["claims"][0]["excerpt"] = "A fabricated rate cut was announced."
    p = QueueProvider(facts=facts)
    with pytest.raises(ImpactAnalysisError, match="Claim not present"):
        analyze_pressure_two_stage(example(), p)
    assert len(p.prompts) == 1


def test_impact_citation_must_match_extracted_claim() -> None:
    i = impact()
    i["claimIndexes"] = [1]
    p = QueueProvider(impacts={"impacts": [i]})
    with pytest.raises(ImpactAnalysisError, match="missing from cited claims"):
        analyze_pressure_two_stage(example(), p)


def test_company_target_must_be_named_in_source() -> None:
    i = impact()
    i["targetType"], i["targetName"] = "COMPANY", "ImaginaryCorp"
    p = QueueProvider(impacts={"impacts": [i]})
    with pytest.raises(ImpactAnalysisError, match="Company not identified"):
        analyze_pressure_two_stage(example(), p)


def test_unknown_direction_cannot_confidently_claim_strength() -> None:
    i = impact()
    i["direction"], i["strength"] = "UNCERTAIN", "HIGH"
    p = QueueProvider(impacts={"impacts": [i]})
    with pytest.raises(ValueError, match="UNCERTAIN_DIRECTION"):
        analyze_pressure_two_stage(example(), p)


def test_high_strength_from_only_unverified_reports_rejected() -> None:
    i = impact()
    i["claimIndexes"] = [1]
    i["evidence_excerpt"] = NEWS_QUOTE
    i["strength"] = "HIGH"
    p = QueueProvider(impacts={"impacts": [i]})
    with pytest.raises(ImpactAnalysisError, match="High pressure strength"):
        analyze_pressure_two_stage(example(), p)


def test_model_returns_no_claims_without_inventing_impacts() -> None:
    p = QueueProvider(facts={"claims": []})
    output = analyze_pressure_two_stage(example(), p)
    assert output["modelCalls"] == 1
    assert output["impacts"] == []
    assert len(p.prompts) == 1


def test_unverified_news_does_not_allow_medium_uncertainty() -> None:
    i = impact()
    i["uncertainty"] = "MEDIUM"
    p = QueueProvider(impacts={"impacts": [i]})
    with pytest.raises((ValueError, ImpactAnalysisError)):
        analyze_pressure_two_stage(example(), p)



def test_economic_surprise_requires_sourced_actual_and_consensus() -> None:
    proposed = impact()
    proposed["expectationComparison"] = "ABOVE_EXPECTATION"
    p = QueueProvider(impacts={"impacts": [proposed]})
    with pytest.raises(ImpactAnalysisError, match="lacks sourced consensus"):
        analyze_pressure_two_stage(example(), p)


def test_ollama_impact_uses_portable_json_with_postvalidation() -> None:
    from types import SimpleNamespace

    provider = QueueProvider()
    provider.settings = SimpleNamespace(kind="ollama")
    original = provider.generate_json
    observed: list[tuple[int, bool]] = []

    def recorder(*, system: str, user: str, schema: dict | None) -> str:
        observed.append((len(observed), schema is None))
        return original(system=system, user=user, schema=schema)

    provider.generate_json = recorder
    result = analyze_pressure_two_stage(example(), provider)
    assert observed == [(0, False), (1, True)]
    assert result["impactGenerationMode"] == "JSON_PYDANTIC_VALIDATED"
    assert result["impacts"][0]["strength"] == "UNASSESSED"


def test_ollama_portable_json_still_rejects_invalid_impact() -> None:
    from types import SimpleNamespace

    invalid = impact()
    invalid["direction"] = "ALWAYS_RISE"
    provider = QueueProvider(impacts={"impacts": [invalid]})
    provider.settings = SimpleNamespace(kind="ollama")
    with pytest.raises(ValueError, match="direction"):
        analyze_pressure_two_stage(example(), provider)


def test_empty_real_model_impact_must_report_explainable_abstention() -> None:
    provider = QueueProvider(impacts={
        "impacts": [],
        "noImpactCategory": "NO_IDENTIFIABLE_AFFECTED_TARGET",
        "noImpactExplanation": (
            "The cited headlines contain no reliably identifiable economic "
            "exposure for a particular market, industry or company."
        ),
    })
    result = analyze_pressure_two_stage(example(), provider)
    assert result["impacts"] == []
    assert result["noImpactAssessment"]["category"] == (
        "NO_IDENTIFIABLE_AFFECTED_TARGET"
    )
    assert result["noImpactAssessment"]["explanation"]
    assert result["modelCalls"] == 2


def test_unexplained_empty_result_is_not_marked_quality_pass() -> None:
    result = analyze_pressure_two_stage(
        example(), QueueProvider(impacts={"impacts": []}),
    )
    assert result["noImpactAssessment"]["category"] == "UNEXPLAINED_ABSTENTION"
    assert result["semanticAccuracyVerified"] is False


def test_unverified_news_can_support_qualified_pressure_not_forced_silence() -> None:
    proposed = impact()
    proposed["evidence_excerpt"] = NEWS_QUOTE
    proposed["claimIndexes"] = [1]
    proposed["strength"] = "LOW"
    proposed["uncertainty"] = "HIGH"
    provider = QueueProvider(impacts={"impacts": [proposed]})
    result = analyze_pressure_two_stage(example(), provider)
    assert result["impacts"][0]["strength"] == "LOW"
    assert result["impacts"][0]["uncertainty"] == "HIGH"
    assert result["impacts"][0]["supportingSourceIds"] == ["s2"]
    assert result["noImpactAssessment"] is None


def test_empty_explanation_with_nonempty_impact_rejected() -> None:
    provider = QueueProvider(impacts={
        "impacts": [impact()],
        "noImpactCategory": "OTHER",
        "noImpactExplanation": (
            "The model contradicts itself by both answering and abstaining."
        ),
    })
    with pytest.raises(ValueError, match="IMPACT_ABSTENTION_CONTRADICTION"):
        analyze_pressure_two_stage(example(), provider)
