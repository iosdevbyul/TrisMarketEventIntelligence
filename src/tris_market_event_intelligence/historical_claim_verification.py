"""Fail-closed verification for atomic, source-grounded news impact claims.

Exact source excerpts are the only mechanically verified facts. Economic
mechanisms remain clearly labeled conditional hypotheses; neither an LLM
judgment nor exact quotation is proof of an asserted causal market outcome.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_MARKET_OUTCOME = re.compile(
    r"\b(?:stock|stocks|shares|index|indices|nasdaq|s&p\s*500|market)\b"
    r".{0,100}\b(?:fell|dropped|declined|slumped|plunged|rallied|rose|"
    r"sold\s+off|sell-?off|downward\s+pressure)\b|"
    r"\b(?:sell-?off|decline|drop|rally|rise)\b.{0,80}"
    r"\b(?:stocks|shares|index|indices|market)\b",
    re.IGNORECASE,
)
_FACT_ASSERTION = re.compile(
    r"\b(?:has|have|had|was|were|did|launched|confirmed|proved|"
    r"resulted|caused|triggered|led\s+to|announced|"
    r"implemented|imposed)\b",
    re.IGNORECASE,
)
_CONDITIONAL = re.compile(
    r"\b(?:if|could|might|may|would|potentially|conditional|"
    r"possible|possibility)\b",
    re.IGNORECASE,
)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvidenceClaim(Strict):
    quoteId: str
    sourceExcerpt: str = Field(min_length=12, max_length=280)
    epistemicType: Literal["REPORTED_FACT", "ALLEGATION", "DENIAL", "OPINION"]


class CausalClaim(Strict):
    triggerCondition: str = Field(min_length=12, max_length=400)
    transmissionChannel: Literal[
        "REVENUE", "COSTS", "DEMAND", "SUPPLY", "REGULATION",
        "REPUTATION", "RISK_PREMIUM", "FINANCING", "OTHER",
    ]
    potentialEffect: str = Field(min_length=20, max_length=500)
    counterHypothesis: str = Field(min_length=20, max_length=500)


def verify_claims(
    *, evidence: list[EvidenceClaim], causal: CausalClaim,
    quote_lookup: dict[str, dict], permitted_quote_ids: list[str],
    claimed_name: str | None = None,
) -> dict:
    """Never rewrite generated statements or silently invent source support."""
    errors: list[dict] = []
    if not evidence:
        errors.append({"reason": "NO_ATOMIC_SOURCE_CLAIMS"})
    if len(evidence) > 6:
        errors.append({"reason": "TOO_MANY_SOURCE_CLAIMS"})
    if len(set(permitted_quote_ids)) != len(permitted_quote_ids):
        errors.append({"reason": "DUPLICATE_EVIDENCE_REFERENCE"})
    allowed = set(permitted_quote_ids)
    used: set[str] = set()
    verified: list[dict] = []
    for index, item in enumerate(evidence):
        quote = quote_lookup.get(item.quoteId)
        if item.quoteId in used:
            errors.append({"claimIndex": index, "reason": "DUPLICATE_SOURCE_CLAIM"})
            continue
        used.add(item.quoteId)
        if quote is None or item.quoteId not in allowed:
            errors.append({"claimIndex": index, "reason": "CLAIM_QUOTE_NOT_ALLOWED"})
            continue
        # Exact substring, preserving the original source text verbatim.
        if item.sourceExcerpt not in quote["text"]:
            errors.append({"claimIndex": index, "reason": "EXCERPT_NOT_IN_SOURCE"})
            continue
        if item.epistemicType == "REPORTED_FACT" and re.search(
            r"\b(?:alleged|allegedly|suspected|rumou?red|reportedly|"
            r"according\s+to|accused|claims?|unconfirmed)\b",
            item.sourceExcerpt, re.IGNORECASE,
        ):
            errors.append({"claimIndex": index, "reason": "ALLEGATION_AS_FACT"})
            continue
        verified.append({
            "quoteId": item.quoteId,
            "sourceExcerpt": item.sourceExcerpt,
            "epistemicType": item.epistemicType,
            "sourceMatch": "EXACT_SUBSTRING",
            "semanticEntailmentVerified": False,
        })
    if claimed_name is not None and not any(
        claimed_name.casefold() in quote_lookup[item.quoteId]["text"].casefold()
        for item in evidence if item.quoteId in quote_lookup
    ):
        errors.append({"reason": "ENTITY_NOT_IN_EVIDENCE"})
    for field, statement in (
        ("triggerCondition", causal.triggerCondition),
        ("potentialEffect", causal.potentialEffect),
        ("counterHypothesis", causal.counterHypothesis),
    ):
        if _MARKET_OUTCOME.search(statement):
            # No historical prices/outcomes were loaded: do not promote an
            # asserted market move, including sentences containing "could".
            errors.append({"field": field, "reason": "UNVERIFIED_MARKET_OUTCOME"})
        if field == "potentialEffect" and not _CONDITIONAL.search(statement):
            errors.append({"field": field, "reason": "NONCONDITIONAL_IMPACT"})
        if field == "triggerCondition" and not _CONDITIONAL.search(statement):
            errors.append({"field": field, "reason": "NONCONDITIONAL_TRIGGER"})
        if field == "counterHypothesis" and not _CONDITIONAL.search(statement):
            errors.append({"field": field, "reason": "NONCONDITIONAL_COUNTER_HYPOTHESIS"})
        # Unconditional new official acts should not be smuggled into the
        # economic hypothesis; source-asserted facts belong in evidence.
        if field != "counterHypothesis" and _FACT_ASSERTION.search(statement):
            sentences = re.split(r"(?<=[.!?])\s+", statement)
            if any(_FACT_ASSERTION.search(line) and not _CONDITIONAL.search(line)
                   for line in sentences):
                errors.append({"field": field, "reason": "UNCONDITIONAL_NEW_FACT"})
    return {
        "status": "SOURCE_QUOTE_MATCHED_CONDITIONAL" if not errors
        else "REJECTED_UNSUPPORTED_CLAIM",
        "evidence": verified,
        "claimChecks": errors,
        "semanticallyVerified": False,
        "causalityProven": False,
        "marketOutcomesVerified": False,
        "humanReviewRequired": True,
    }
