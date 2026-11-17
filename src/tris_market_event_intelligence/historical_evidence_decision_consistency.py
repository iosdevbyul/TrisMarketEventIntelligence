"""Conservative evidence/decision consistency checks for model-authored drafts.

Diagnoses known vacuous responses and overt logical contradictions. Does not
pretend general natural-language entailment or rewrite original model output.
"""
from __future__ import annotations

import re

from tris_market_event_intelligence.historical_evidence_scope import (
    is_observed_market_result,
)

_PLACEHOLDER = re.compile(
    r"^\s*(?:included|provided|present|listed|mentioned|found)"
    r"\s+in\s+(?:the\s+)?(?:visible\s+)?(?:historical\s+)?"
    r"(?:source|evidence|context|record|input|data)\.?\s*$",
    re.IGNORECASE,
)
_RECORD_NOUN = re.compile(
    r"\b(?:records?|documents?|filings?|receipts?|data|evidence)\b",
    re.IGNORECASE,
)
_ABSENCE = re.compile(
    r"\b(?:not\s+(?:yet\s+)?(?:available|visible|provided|received|"
    r"confirmed|observed|submitted|reviewed|established)|"
    r"(?:remain|remains|remained)\s+(?:unavailable|unknown|unverified|"
    r"unobserved|missing)|(?:still\s+needed)|(?:no\s+records?))\b",
    re.IGNORECASE,
)


def review_evidence_decisions(packet: dict, draft: dict) -> dict:
    """Flag concrete observed failures, conservatively marked for human review.

    The contradiction rule checks the actual research question and model
    explanation: saying required documentary records are NOT available cannot
    establish that a question asking for those records was RESOLVED. By
    contrast, a resolved product-authorization question can reasonably retain
    unknown manufacturing details. It therefore must not be rejected just
    because *some* remaining uncertainty exists.
    """
    questions = {q["questionId"]: q for q in draft.get("visibleQuestions", [])}
    facts = {
        (s["sourceId"], c["claimId"]): c["text"]
        for s in packet["eligibleSources"] for c in s["claims"]
    }
    vacuous = []
    for change in draft.get("modelAuthoredChangeItems", []):
        action = change["changedAction"]
        if _PLACEHOLDER.fullmatch(action):
            vacuous.append({
                "sourceId": change["sourceId"],
                "claimId": change["claimId"],
                "issueCode": "CHANGE_ACTION_IS_EVIDENCE_PRESENCE_PLACEHOLDER",
            })
    contradicted = []
    for decision in draft.get("modelAuthoredQuestionAssessments", []):
        question = questions.get(decision["questionId"])
        if question is None or decision["status"] != "RESOLVED_BY_VISIBLE_SOURCE":
            continue
        # Only when the question requests missing documentary evidence and
        # the MODEL explanation itself says the evidence is not available.
        # Don't scan stillUnknown alone: resolved decisions may retain
        # legitimate uncertainty about different aspects of an event.
        query = question["question"]
        rationale = decision["explanation"]
        if (
            _RECORD_NOUN.search(query)
            and _RECORD_NOUN.search(rationale)
            and _ABSENCE.search(rationale)
        ):
            contradicted.append({
                "questionId": decision["questionId"],
                "sourceId": decision["sourceId"],
                "claimId": decision["claimId"],
                "issueCode": "RESOLVED_STATUS_CONTRADICTS_MISSING_REQUIRED_RECORDS",
            })
    codes = []
    if vacuous:
        codes.append("VACUOUS_MODEL_CHANGE_ACTION")
    if contradicted:
        codes.append("MODEL_STATUS_CONTRADICTS_OWN_EVIDENCE_EXPLANATION")
    return {
        "issueCodes": codes,
        "vacuousChangeItems": vacuous,
        "contradictoryDecisions": contradicted,
        "narrowPatternWarningOnly": True,
        "doesNotInferCausalProof": True,
        "semanticAccuracyVerified": False,
        "modelOutputRewritten": False,
        "humanReviewRequired": True,
    }


def check_action_evidence(packet: dict, changes: list[dict]) -> list[dict]:
    """Only mechanically vacuous field text fails closed; no lexical similarity
    scoring to curated claim paraphrases, since truthful synonyms differ.
    """
    return [
        {
            "sourceId": x["sourceId"],
            "claimId": x["claimId"],
            "reason": "CHANGE_ACTION_IS_EVIDENCE_PRESENCE_PLACEHOLDER",
        }
        for x in changes
        if _PLACEHOLDER.fullmatch(x["changedAction"])
        or is_observed_market_result(x["changedAction"])
    ]
