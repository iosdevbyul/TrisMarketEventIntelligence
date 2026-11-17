"""Conditional *scenario* assessment over a complete economic mechanism.

An implicit antecedent and a declarative alternative can be legitimate
research hypotheses, but only when explicitly scoped as unverified model
assumptions, never as established facts or causal results.
"""
from __future__ import annotations

import re

from tris_market_event_intelligence.historical_claim_verification import (
    _CONDITIONAL,
    _FACT_ASSERTION,
)
from tris_market_event_intelligence.historical_evidence_scope import (
    is_observed_market_result,
)

_REALISED_OUTCOME = re.compile(
    r"\b(?:shares?|stocks?|indices|nasdaq|s&p\s*500)\b.{0,100}?"
    r"\b(?:fell|rose|plunged|rallied|dropped|gained)\b",
    re.IGNORECASE,
)
_EXPLICIT_UNCERTAINTY = re.compile(
    r"\b(?:could|might|may|would|potentially|possibly)\b", re.IGNORECASE,
)
_ANTECEDENT = re.compile(r"\b(?:if|when|provided\s+that|assuming)\b", re.IGNORECASE)


def _price_outcome(text: str) -> bool:
    return is_observed_market_result(text) or bool(_REALISED_OUTCOME.search(text))


def review_scenario(proposal: dict) -> dict:
    """Check scenario shape without rewriting author text.

    Hard errors: no conditional outcome, unsupported actual market result,
    or an unconditional, asserted new outcome. Recoverable forms: nominal
    trigger with explicit downstream IF, and declarative alternative tagged
    as an *unverified* competing scenario requiring human review.
    """
    trigger = proposal["triggerCondition"]
    effect = proposal["conditionalEffect"]
    counter = proposal["counterHypothesis"]
    missing = proposal["missingEvidence"]
    reasons: list[str] = []
    notes: list[str] = []
    if any(_price_outcome(x) for x in (trigger, effect, counter, missing)):
        reasons.append("OBSERVED_MARKET_RESULT_NOT_SUPPORTED")
    if not _EXPLICIT_UNCERTAINTY.search(effect):
        reasons.append("POTENTIAL_EFFECT_LACKS_UNCERTAINTY")
    if _FACT_ASSERTION.search(effect) and not _CONDITIONAL.search(effect):
        reasons.append("UNCONDITIONAL_NEW_ECONOMIC_FACT")
    if not _ANTECEDENT.search(trigger):
        if _ANTECEDENT.search(effect) and _EXPLICIT_UNCERTAINTY.search(effect):
            notes.append("IMPLICIT_TRIGGER_CONDITION_SCOPED_BY_EFFECT")
        else:
            reasons.append("TRIGGER_UNCONDITIONAL_WITHOUT_DOWNSTREAM_IF")
    if not _ANTECEDENT.search(counter):
        # A statement in the counterHypothesis *slot* is an alternative
        # proposition, not an observed fact. Preserve author wording, mark it
        # unverified and demand human review. It is not promoted to fact.
        notes.append("DECLARATIVE_COUNTER_AS_UNVERIFIED_ALTERNATIVE")
    return {
        "status": (
            "REJECTED_UNSUPPORTED_SCENARIO" if reasons
            else "CONDITIONAL_RESEARCH_CANDIDATE"
        ),
        "blockers": reasons,
        "warnings": notes,
        "triggerIsExplicitlyConditional": bool(_ANTECEDENT.search(trigger)),
        "effectContainsAntecedent": bool(_ANTECEDENT.search(effect)),
        "counterIsExplicitlyConditional": bool(_ANTECEDENT.search(counter)),
        "allStatementsAreAssumptions": True,
        "semanticEntailmentVerified": False,
        "causalityProven": False,
        "humanReviewRequired": True,
    }
