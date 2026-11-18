"""Conservative, deterministic economic-reasoning quality diagnostics.

These checks detect unsupported *structure* of a hypothesis, not empirical
market impact. They never invent a direction, magnitude or transmission.
A passing check is NOT evidence of economic correctness.
"""
from __future__ import annotations

import re
from typing import Any

_REGULATORY = re.compile(
    r"\b(regulat(?:ion|ions|ory|or)|policy|policies|legislat(?:ion|ive)|"
    r"law|laws|rule|rules|license|licens(?:ing|es)|permit|permits|"
    r"tariff|tariffs|sanction|sanctions|export.control|approval|"
    r"compliance|ban|bans|restriction|restrictions)\b",
    re.IGNORECASE,
)
_GOVERNANCE = re.compile(
    r"\b(ceo|chief.executive|boss|board|leader(?:ship)?|governance|"
    r"executive|management|resign(?:ation|ed)?|oust(?:ed|er)?|"
    r"return|reinstat(?:ed|ement)|succession|investor.calls)\b",
    re.IGNORECASE,
)
_QUANT_ECONOMICS = re.compile(
    r"\b(?:revenue|earnings|profit|margin|cash.flow|cost.of.capital|"
    r"discount.rate|funding.cost|profitability|valuation|"
    r"operating.cost|capital.expenditure|investment.cost|"
    r"sales|orders|backlog|contracts?|guidance|risk.premium)\b",
    re.IGNORECASE,
)


def assess_reasoning_quality(
    impact: dict[str, Any], cited_claims: list[dict[str, Any]],
) -> dict:
    """Provide named concerns and evidence gaps, never an accuracy score.

    Each flag remains a review requirement rather than an automatic
    rejection. Evidence vocabulary is not treated as causal verification.
    """
    rationale = str(impact.get("rationale", ""))
    strength_rationale = str(impact.get("strengthRationale", ""))
    channel = impact.get("transmissionChannel")
    direction = impact.get("direction")
    strength = impact.get("strength")
    countervailing = impact.get("countervailingFactors") or []
    cited_text = " ".join(
        str(claim.get("excerpt", "")) for claim in cited_claims
    )
    hypothesis_text = " ".join([
        rationale, strength_rationale,
        " ".join(str(x) for x in impact.get("drivers", [])),
    ])
    flags: list[str] = []
    explanations: dict[str, str] = {}

    def flag(code: str, explanation: str) -> None:
        flags.append(code)
        explanations[code] = explanation

    if (
        channel == "REGULATORY_OPPORTUNITY"
        and _GOVERNANCE.search(cited_text + " " + hypothesis_text)
        and not _REGULATORY.search(cited_text + " " + hypothesis_text)
    ):
        flag(
            "REGULATORY_CHANNEL_WITHOUT_REGULATORY_EVIDENCE",
            "Leadership/governance language is present, but no regulatory "
            "fact or mechanism appears in cited observations or rationale.",
        )

    if direction == "MIXED":
        flag(
            "MIXED_DIRECTION_REQUIRES_TWO_SIDED_TRANSMISSION_REVIEW",
            "A MIXED direction requires separately justified positive and "
            "negative mechanisms; a single channel and a list of opposing "
            "factors do not independently prove both effects.",
        )
        if not countervailing:
            flag(
                "MIXED_DIRECTION_WITHOUT_COUNTERVAILING_FACTORS",
                "No offsetting economic factors were supplied.",
            )

    if strength in {"MODERATE", "HIGH"}:
        flag(
            "QUALITATIVE_STRENGTH_NOT_EMPIRICALLY_VERIFIED",
            "The model's qualitative strength is not an estimated exposure, "
            "causal effect, return or probability and requires human review.",
        )
        if not _QUANT_ECONOMICS.search(strength_rationale):
            flag(
                "STRENGTH_JUSTIFICATION_LACKS_ECONOMIC_MAGNITUDE_BRIDGE",
                "The strength rationale does not explain financial exposure, "
                "earnings, costs, valuation or other economic magnitude.",
            )

    if any(
        claim.get("sourceVerification") == "REPORTED_UNVERIFIED"
        for claim in cited_claims
    ):
        flag(
            "REPORTED_NEWS_NOT_INDEPENDENTLY_CONFIRMED",
            "Source provenance is not independent verification of the event.",
        )
    if len({claim.get("sourceId") for claim in cited_claims}) == 1:
        flag(
            "SINGLE_CITED_SOURCE_LIMITS_CORROBORATION",
            "The proposal is supported by only one cited source.",
        )
    if (
        impact.get("expectationComparison") == "NOT_AVAILABLE"
        and strength in {"MODERATE", "HIGH"}
    ):
        flag(
            "PRIOR_MARKET_EXPECTATIONS_UNAVAILABLE",
            "Strength is proposed without a sourced prior market expectation.",
        )

    return {
        "schemaVersion": "market_pressure_reasoning_quality_v1",
        "gateStatus": "HUMAN_SEMANTIC_REVIEW_REQUIRED",
        "economicAccuracyVerified": False,
        "observedMarketReactionVerified": False,
        "riskFlags": flags,
        "riskExplanations": explanations,
        "modelDirectionPreserved": direction,
        "modelStrengthPreserved": strength,
        "modelTransmissionChannelPreserved": channel,
    }
