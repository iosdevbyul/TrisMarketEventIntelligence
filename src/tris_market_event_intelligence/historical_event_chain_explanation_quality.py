"""Conservative semantic quality evidence checks on dated event-chain research.

This is NOT a truth-or-entailment classifier. It identifies inspectable
omissions, circular "missing evidence", and unresolved decision omissions
using the source IDs and actual statement text that the model received.
The checks never invent evidence, repair model text or award causal proof.
"""
from __future__ import annotations

import re
from collections import Counter

_DECISION_PENDING = re.compile(
    r"\b(?:before approving|would audit|would assess|assessment of|"
    r"required .* before|will keep|pending|would not grant)\b",
    re.IGNORECASE,
)
_FALSE_COMPLETENESS = re.compile(
    r"\b(?:no|none|not|without)\b.{0,80}?"
    r"\b(?:unknown|unresolved|future evidence|additional information|"
    r"remaining questions|necessary future evidence)\b",
    re.IGNORECASE,
)
_ACTIONABLE_EVIDENCE = re.compile(
    r"\b(?:inspection results?|actual findings?|underlying data|"
    r"completed audits?|completion reports?|records?|documents?|"
    r"manufacturer cost|operator cost|financial statements?|"
    r"independent|verification|subsequent release|"
    r"decision notice|approval status|authorization records?)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z][a-z0-9-]*", re.IGNORECASE)
_STOP = frozenset({
    "the", "that", "this", "were", "with", "from", "their", "would",
    "said", "while", "after", "before", "also", "which", "these",
    "about", "into", "additional", "under", "agency", "release", "announced",
})


def _words(value: str) -> set[str]:
    return {token for token in _WORD.findall(value.casefold())
            if len(token) > 3 and token not in _STOP}


def source_claim_index(packet: dict) -> dict[tuple[str, str], str]:
    return {
        (source["sourceId"], claim["claimId"]): claim["text"]
        for source in packet["eligibleSources"]
        for claim in source["claims"]
    }


def inspect_missing_evidence(edge: dict, source: dict) -> dict:
    """Flag restatement of editor-provided relationship rather than new needs."""
    explanation = edge["missingEvidence"]
    basis = source["basis"]
    overlap = len(_words(explanation) & _words(basis))
    base = len(_words(explanation) | _words(basis))
    similarity = overlap / base if base else 0.0
    issues: list[str] = []
    if similarity >= 0.45:
        issues.append("MISSING_EVIDENCE_RESTATES_KNOWN_EDGE_BASIS")
    if not _ACTIONABLE_EVIDENCE.search(explanation):
        issues.append("MISSING_EVIDENCE_NOT_ACTIONABLE")
    return {
        "edgeId": edge["edgeId"],
        "issueCodes": issues,
        "wordOverlapRatio": round(similarity, 3),
        "specificEvidenceRequirementVerified": False,
        "humanReviewRequired": True,
    }


def audit_explanation(packet: dict, assessment: dict) -> dict:
    """Audit the saved output as-is, never rewriting or accepting model claims."""
    facts = source_claim_index(packet)
    source_to_facts: dict[str, set[str]] = {}
    for source_id, fact_id in facts:
        source_to_facts.setdefault(source_id, set()).add(fact_id)
    edges = {e["edgeId"]: e for e in packet["visibleCuratedEdges"]}
    edge_reviews = [
        inspect_missing_evidence(row, edges[row["edgeId"]])
        for row in assessment["acceptedEdgeExplanations"]
        if row["edgeId"] in edges
    ]
    updates = assessment.get("changeItems", [])
    questions = assessment.get("unresolvedQuestions", [])
    valid_updates: list[dict] = []
    invalid_updates: list[dict] = []
    for item in updates:
        key = (item.get("sourceId"), item.get("claimId"))
        problems = []
        if key not in facts:
            problems.append("CHANGE_REFERENCES_UNAVAILABLE_SOURCE_CLAIM")
        if not item.get("changedAction") or not item.get("scope"):
            problems.append("CHANGE_DESCRIPTION_INCOMPLETE")
        if problems:
            invalid_updates.append({"proposal": item, "issueCodes": problems})
        else:
            valid_updates.append(item)
    valid_questions: list[dict] = []
    invalid_questions: list[dict] = []
    for item in questions:
        key = (item.get("sourceId"), item.get("claimId"))
        problems = []
        if key not in facts:
            problems.append("QUESTION_REFERENCES_UNAVAILABLE_SOURCE_CLAIM")
        if not _ACTIONABLE_EVIDENCE.search(item.get("evidenceToResolve", "")):
            problems.append("QUESTION_NOT_TIED_TO_ACTIONABLE_EVIDENCE")
        if problems:
            invalid_questions.append({"proposal": item, "issueCodes": problems})
        else:
            valid_questions.append(item)
    visible_ids = set(source_to_facts)
    covered_ids = {item["sourceId"] for item in valid_updates}
    recent_ids = {
        item["sourceId"] for item in packet["eligibleSources"]
        if item["sourceId"] not in set(packet.get("previouslySeenSourceIds", []))
    }
    pending_facts = [
        {"sourceId": key[0], "claimId": key[1]}
        for key, fact in facts.items() if _DECISION_PENDING.search(fact)
    ]
    cited_question_keys = {
        (item["sourceId"], item["claimId"]) for item in valid_questions
    }
    missing_questions = [
        key for key in pending_facts
        if (key["sourceId"], key["claimId"]) not in cited_question_keys
    ]
    warnings: list[str] = []
    if _FALSE_COMPLETENESS.search(assessment["whatRemainsUnknown"]) and pending_facts:
        warnings.append("UNKNOWN_NARRATIVE_MINIMIZES_OPEN_REGULATORY_QUESTIONS")
    if recent_ids - covered_ids:
        warnings.append("INCOMPLETE_NEW_SOURCE_CHANGE_COVERAGE")
    if missing_questions:
        warnings.append("UNRESOLVED_DECISIONS_LACK_FOLLOWUP_QUESTIONS")
    if any(row["issueCodes"] for row in edge_reviews):
        warnings.append("LINK_EVIDENCE_REQUIREMENTS_NEED_REVIEW")
    if invalid_updates or invalid_questions:
        warnings.append("UNSUPPORTED_CHANGE_OR_QUESTION_REFERENCES")
    # Even exhaustive reference coverage is not proof of semantic explanation.
    return {
        "sourceClaimCount": len(facts),
        "visibleSourceCount": len(visible_ids),
        "recentSourceCount": len(recent_ids),
        "changeSourceCoverage": len(covered_ids & visible_ids),
        "missingRecentSourceIds": sorted(recent_ids - covered_ids),
        "pendingDecisionClaims": pending_facts,
        "unansweredPendingDecisionClaims": missing_questions,
        "validChangeItems": valid_updates,
        "invalidChangeItems": invalid_updates,
        "validUnresolvedQuestions": valid_questions,
        "invalidUnresolvedQuestions": invalid_questions,
        "edgeEvidenceReviews": edge_reviews,
        "issueCodes": warnings,
        "semanticAccuracyVerified": False,
        "causalityVerified": False,
        "PITVerified": False,
        "humanReviewRequired": True,
    }


def audit_sequence(doc: dict, saved: dict) -> dict:
    """Offline review of existing v1 or future saved replay with 0 model calls."""
    from tris_market_event_intelligence.historical_independent_news_cli import _digest
    from tris_market_event_intelligence.historical_official_event_chain_cli import (
        prepare_replay,
        validate_chain,
    )

    validate_chain(doc)
    if (
        not isinstance(saved, dict)
        or saved.get("schemaVersion") not in {
            "historical_official_event_chain_sequence_v1",
            "historical_official_event_chain_sequence_v2",
        }
        or saved.get("eventId") != doc["eventId"]
        or saved.get("sourceCohortSha256") != _digest(doc)
        or saved.get("historicalPITVerified") is not False
        or saved.get("causalityProven") is not False
        or not isinstance(saved.get("replaySteps"), list)
        or len(saved["replaySteps"]) != 2
    ):
        raise ValueError("EVENT_CHAIN_SAVED_SEQUENCE_INTEGRITY_INVALID")
    reviewed: list[dict] = []
    previous_ids: set[str] = set()
    previous_cutoff = ""
    for step in saved["replaySteps"]:
        cutoff = step.get("asOfDateExclusive")
        if not isinstance(cutoff, str) or cutoff <= previous_cutoff:
            raise ValueError("EVENT_CHAIN_SAVED_CUTOFF_INVALID")
        try:
            from datetime import date

            expected = prepare_replay(doc, date.fromisoformat(cutoff))
        except (TypeError, ValueError) as exc:
            raise ValueError("EVENT_CHAIN_SAVED_CUTOFF_INVALID") from exc
        if (
            step.get("sourceCohortSha256") != expected["sourceCohortSha256"]
            or step.get("eligibleSources") != expected["eligibleSources"]
            or step.get("visibleCuratedEdges") != expected["visibleCuratedEdges"]
            or step.get("pointInTimeVerified") is not False
            or not isinstance(step.get("assessment"), dict)
        ):
            raise ValueError("EVENT_CHAIN_SAVED_SOURCE_SCOPE_INVALID")
        packet = {**expected, "previouslySeenSourceIds": sorted(previous_ids)}
        quality = audit_explanation(packet, step["assessment"])
        reviewed.append({
            "asOfDateExclusive": cutoff,
            "qualityAudit": quality,
            "savedOriginalModelClaimsUnchanged": True,
        })
        previous_ids = {s["sourceId"] for s in expected["eligibleSources"]}
        previous_cutoff = cutoff
    return {
        "schemaVersion": "historical_official_event_chain_quality_audit_v1",
        "status": "EVENT_CHAIN_EXPLANATION_QUALITY_REQUIRES_HUMAN_REVIEW",
        "originalSourceCohortSha256": _digest(doc),
        "originalSequenceModelCalls": saved.get("modelCalls"),
        "auditModelCalls": 0,
        "replayStepAudits": reviewed,
        "warningCounts": dict(Counter(
            flag for row in reviewed for flag in row["qualityAudit"]["issueCodes"]
        )),
        "historicalPITVerified": False,
        "economicCausalityVerified": False,
        "semanticAccuracyVerified": False,
        "generalizationVerified": False,
        "originalSavedArtifactUnchanged": True,
        "promotionAllowed": False,
        "tradesPlaced": False,
    }
