"""Deterministic quality concerns, never falsely certified market impact."""
from __future__ import annotations

from tris_market_event_intelligence.market_pressure_quality_gate import (
    assess_reasoning_quality,
)


def _hypothesis() -> dict:
    return {
        "rationale": (
            "The CEO transition introduces uncertainty regarding future "
            "strategic execution and investor perceptions."
        ),
        "strengthRationale": (
            "The reported CEO exit creates a mixed signal regarding "
            "operational stability and leadership."
        ),
        "transmissionChannel": "REGULATORY_OPPORTUNITY",
        "direction": "UNCERTAIN",
        "strength": "MODERATE",
        "drivers": ["Leadership continuity risk"],
        "countervailingFactors": ["Investor calls for return"],
        "expectationComparison": "NOT_AVAILABLE",
    }


def _claims() -> list[dict]:
    return [{
        "sourceId": "article-20726",
        "sourceVerification": "REPORTED_UNVERIFIED",
        "excerpt": (
            "Huang says he's willing to return as Nvidia boss after "
            "leadership transition and calls from investors"
        ),
    }]


def test_actual_nvidia_governance_regulatory_mismatch_is_flagged() -> None:
    hypothesis = _hypothesis()
    before = hypothesis.copy()
    review = assess_reasoning_quality(hypothesis, _claims())
    assert hypothesis == before
    assert review["economicAccuracyVerified"] is False
    assert review["gateStatus"] == "HUMAN_SEMANTIC_REVIEW_REQUIRED"
    assert "REGULATORY_CHANNEL_WITHOUT_REGULATORY_EVIDENCE" in (
        review["riskFlags"]
    )
    assert "STRENGTH_JUSTIFICATION_LACKS_ECONOMIC_MAGNITUDE_BRIDGE" in (
        review["riskFlags"]
    )
    assert review["modelTransmissionChannelPreserved"] == (
        "REGULATORY_OPPORTUNITY"
    )
    assert review["modelStrengthPreserved"] == "MODERATE"


def test_real_regulation_context_does_not_trigger_governance_mismatch() -> None:
    hypothesis = _hypothesis()
    hypothesis["rationale"] = (
        "The reported new export control restrictions may alter "
        "access to international sales."
    )
    cited = _claims()
    cited[0]["excerpt"] = (
        "Regulator announces new licensing requirements for exports "
        "following company leadership changes."
    )
    flags = assess_reasoning_quality(hypothesis, cited)["riskFlags"]
    assert "REGULATORY_CHANNEL_WITHOUT_REGULATORY_EVIDENCE" not in flags


def test_mixed_direction_and_unproven_magnitude_require_review() -> None:
    hypothesis = _hypothesis()
    hypothesis["transmissionChannel"] = "RISK_PREMIUM"
    hypothesis["direction"] = "MIXED"
    hypothesis["countervailingFactors"] = []
    flags = assess_reasoning_quality(hypothesis, _claims())["riskFlags"]
    assert "MIXED_DIRECTION_REQUIRES_TWO_SIDED_TRANSMISSION_REVIEW" in flags
    assert "MIXED_DIRECTION_WITHOUT_COUNTERVAILING_FACTORS" in flags
    assert "PRIOR_MARKET_EXPECTATIONS_UNAVAILABLE" in flags
    assert "REPORTED_NEWS_NOT_INDEPENDENTLY_CONFIRMED" in flags


def test_unassessed_strength_keeps_uncertainty_without_inventing_magnitude() -> None:
    hypothesis = _hypothesis()
    hypothesis["strength"] = "UNASSESSED"
    hypothesis["direction"] = "UNCERTAIN"
    hypothesis["transmissionChannel"] = "RISK_PREMIUM"
    result = assess_reasoning_quality(hypothesis, _claims())
    assert "STRENGTH_JUSTIFICATION_LACKS_ECONOMIC_MAGNITUDE_BRIDGE" not in (
        result["riskFlags"]
    )
    assert result["modelStrengthPreserved"] == "UNASSESSED"
    assert result["observedMarketReactionVerified"] is False
