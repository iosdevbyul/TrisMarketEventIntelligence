"""Article-independent causal analysis contract across unrelated event domains.

These are synthetic contract tests, NOT empirical Gemma generalization proof.
"""
from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tris_market_event_intelligence.market_evidence_pressure import (
    MarketEvidencePacket,
)
from tris_market_event_intelligence.market_pressure_causal_engine import (
    analyze_general_causal,
    causal_output_schema,
)


SCENARIOS = [
    (
        "inflation-release",
        "The statistics office reported consumer prices grew 0.6% in October.",
        "MARKET", "US equities", "DISCOUNT_RATE", "NEGATIVE",
    ),
    (
        "corporate-earnings",
        "Atlas Devices reported higher quarterly operating costs.",
        "COMPANY", "Atlas Devices", "MARGINS", "NEGATIVE",
    ),
    (
        "energy-supply",
        "The pipeline operator reported a reduction in fuel shipments.",
        "SECTOR", "Energy producers", "SUPPLY", "POSITIVE",
    ),
    (
        "monetary-policy",
        "The central bank announced a higher benchmark interest rate.",
        "SECTOR", "Financial institutions", "FUNDING_COST", "NEGATIVE",
    ),
    (
        "regulatory-event",
        "The ministry announced a new licensing requirement for exports.",
        "SECTOR", "Exporting manufacturers", "REGULATORY_OPPORTUNITY", "NEGATIVE",
    ),
    (
        "consumer-demand",
        "The retail survey reported fewer new durable goods orders.",
        "SECTOR", "Consumer products", "DEMAND", "NEGATIVE",
    ),
    (
        "transport-supply",
        "The port authority reported a closure of a main freight terminal.",
        "SECTOR", "Shipping and logistics", "COST", "NEGATIVE",
    ),
]


def packet(name: str, text: str) -> MarketEvidencePacket:
    return MarketEvidencePacket.model_validate({
        "eventId": name, "subject": text,
        "asOfDateExclusive": "2026-11-19",
        "evidences": [{
            "sourceId": "source-1",
            "kind": "NEWS",
            "publishedDate": "2026-11-18",
            "sourceName": "Example bulletin",
            "sourceUrl": "https://example.org/reports/unique",
            "excerpt": text,
            "verification": "REPORTED_UNVERIFIED",
        }],
    })


def proposal(
    text: str, target_type: str, target: str,
    channel: str, sign: str,
) -> dict:
    return {
        "targetType": target_type, "targetName": target,
        "market": "GLOBAL", "sector": "BROAD_MARKET",
        "horizon": "SHORT_TERM", "strength": "UNASSESSED",
        "causalPaths": [{
            "pressureDirection": sign,
            "evidenceClaimIndexes": [0],
            "observedExcerpt": text,
            "economicSteps": [{
                "cause": "Observed event affects an economic input",
                "channel": channel,
                "effect": "Conditional change in future company cash flows",
                "condition": "Only if the observation changes the target exposure",
            }],
            "competingExplanation": (
                "Other macroeconomic pressures may offset this effect."
            ),
        }],
        "missingEvidence": ["Firm exposure not directly measured"],
        "uncertaintyExplanation": (
            "This reported observation has not independently established "
            "a quantified forward economic exposure."
        ),
    }


class MockCausalProvider:
    model_id = "mock-general-causal"

    def __init__(self, text: str, impact: dict | None) -> None:
        self.text = text
        self.impact = impact
        self.calls = 0
        self.user_inputs: list[dict] = []

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        self.calls += 1
        self.user_inputs.append(json.loads(user))
        if self.calls == 1:
            return json.dumps({"claims": [{
                "sourceId": "source-1", "excerpt": self.text,
                "factKind": "REPORTED_FACT", "novelty": "UNKNOWN",
                "entities": [],
            }]})
        assert "candidateImpacts" in schema["properties"]
        return json.dumps({"candidateImpacts": [self.impact] if self.impact else []})


@pytest.mark.parametrize(
    ("name", "text", "target_type", "target", "channel", "sign"),
    SCENARIOS,
)
def test_general_causal_contract_across_distinct_domains(
    name: str, text: str, target_type: str, target: str,
    channel: str, sign: str,
) -> None:
    model = MockCausalProvider(
        text, proposal(text, target_type, target, channel, sign),
    )
    report = analyze_general_causal(packet(name, text), model)
    assert report["analysisMode"] == "GENERAL_CAUSAL_PATHS_V1"
    assert report["status"] == "CONDITIONAL_MARKET_PRESSURE_REQUIRES_REVIEW"
    assert report["modelCalls"] == 2
    assert report["rejectedImpacts"] == []
    assert len(report["impacts"]) == 1
    item = report["impacts"][0]
    assert item["direction"] == sign
    assert item["strength"] == "UNASSESSED"
    assert item["supportingSourceIds"] == ["source-1"]
    assert item["economicCausalityVerified"] is False
    assert report["semanticAccuracyVerified"] is False
    assert report["generalizationVerified"] is False
    assert len(model.user_inputs[1]["groundedClaims"]) == 1


def test_bidirectional_paths_derive_mixed_without_word_matching() -> None:
    name, text, _, _, channel, sign = SCENARIOS[0]
    proposed = proposal(text, "MARKET", "US equities", channel, sign)
    opposing = deepcopy(proposed["causalPaths"][0])
    opposing["pressureDirection"] = "POSITIVE"
    opposing["economicSteps"][0]["channel"] = "DEMAND"
    proposed["causalPaths"].append(opposing)
    model = MockCausalProvider(text, proposed)
    report = analyze_general_causal(packet(name, text), model)
    assert report["impacts"][0]["direction"] == "MIXED"
    assert len(report["impacts"][0]["causalPaths"]) == 2


@pytest.mark.parametrize("mutation,code", [
    (lambda x: x["causalPaths"][0].update(
        evidenceClaimIndexes=[999],
    ), "CAUSAL_CLAIM_INDEX_INVALID"),
    (lambda x: x["causalPaths"][0].update(
        observedExcerpt="Fabricated cited release happened today",
    ), "CAUSAL_OBSERVATION_NOT_CITED"),
    (lambda x: x.update(strength="MODERATE"), "CAUSAL_SCHEMA_INVALID"),
    (lambda x: x["causalPaths"][0]["economicSteps"][0].update(
        condition="unknown",
    ), "CAUSAL_SCHEMA_INVALID"),
])
def test_invalid_cross_domain_proposals_quarantined(
    mutation: object, code: str,
) -> None:
    name, text, kind, target, channel, sign = SCENARIOS[1]
    candidate = proposal(text, kind, target, channel, sign)
    mutation(candidate)
    report = analyze_general_causal(
        packet(name, text), MockCausalProvider(text, candidate),
    )
    assert report["impacts"] == []
    assert report["rejectedImpacts"][0]["code"] == code
    assert report["status"] == "MODEL_OUTPUT_REJECTED_REQUIRES_REVIEW"


def test_unsourced_named_company_is_not_accepted() -> None:
    name, text, _, _, channel, sign = SCENARIOS[0]
    candidate = proposal(text, "COMPANY", "Unknown Corporation", channel, sign)
    report = analyze_general_causal(
        packet(name, text), MockCausalProvider(text, candidate),
    )
    assert report["impacts"] == []
    assert report["rejectedImpacts"][0]["code"] == (
        "CAUSAL_COMPANY_NOT_IN_CITED_OBSERVATIONS"
    )


def test_future_sources_cannot_reach_model_and_absence_is_not_success() -> None:
    name, text, kind, target, channel, sign = SCENARIOS[2]
    raw = packet(name, text).model_dump(mode="json")
    later = deepcopy(raw["evidences"][0])
    later["sourceId"] = "future-source"
    later["publishedDate"] = "2026-11-25"
    later["excerpt"] = "A later press release reveals the future outcome."
    raw["evidences"].append(later)
    model = MockCausalProvider(
        text, proposal(text, kind, target, channel, sign),
    )
    report = analyze_general_causal(
        MarketEvidencePacket.model_validate(raw), model,
    )
    assert len(model.user_inputs[0]["visibleSources"]) == 1
    assert "future-source" not in json.dumps(model.user_inputs)
    empty = analyze_general_causal(
        packet(name, text), MockCausalProvider(text, None),
    )
    assert empty["impacts"] == []
    assert empty["noImpactAssessment"] is not None


def test_native_schema_bounded_and_no_price_prediction_or_legacy_fields() -> None:
    schema = causal_output_schema()
    assert set(schema["properties"]) == {"candidateImpacts"}
    assert schema["additionalProperties"] is False
    item = schema["properties"]["candidateImpacts"]["items"]
    assert item["properties"]["strength"]["enum"] == ["UNASSESSED"]
    assert "noImpactCategory" not in item["properties"]
    assert "priceTarget" not in item["properties"]
    assert "$defs" not in json.dumps(schema)
    assert "$ref" not in json.dumps(schema)
