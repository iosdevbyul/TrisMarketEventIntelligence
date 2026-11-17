"""Source-bound factual change ledger and historical question lifecycle.

Question tracks are *editorially curated*, not verified model inferences.
Only dated source-claim references visible at a replay cutoff are usable.
The saved model text is never edited and no retrospective answer is inserted
into an earlier replay. This is not historical archived-page PIT proof.
"""
from __future__ import annotations

import re
from datetime import date

STATES = {"OPEN", "PARTIALLY_RESOLVED", "RESOLVED_BY_LATER_SOURCE"}
STATE_ORDER = {
    "OPEN": 0,
    "PARTIALLY_RESOLVED": 1,
    "RESOLVED_BY_LATER_SOURCE": 2,
}


def _claims(doc: dict) -> dict[tuple[str, str], tuple[str, dict]]:
    return {
        (source["sourceId"], claim["claimId"]): (source["sourceDate"], claim)
        for source in doc["sources"]
        for claim in source["claims"]
    }


def validate_question_tracks(doc: dict) -> None:
    refs = _claims(doc)
    ids: set[str] = set()
    tracks = doc.get("questionTracks", [])
    if not isinstance(tracks, list) or len(tracks) > 20:
        raise ValueError("QUESTION_TRACKS_INVALID")
    for track in tracks:
        if not isinstance(track, dict):
            raise ValueError("QUESTION_TRACK_INVALID")
        qid = track.get("questionId")
        transitions = track.get("transitions")
        if (
            not isinstance(qid, str) or not qid or qid in ids
            or not isinstance(track.get("question"), str)
            or len(track["question"]) < 30
            or not isinstance(track.get("scope"), str)
            or len(track["scope"]) < 15
            or not isinstance(transitions, list)
            or not 1 <= len(transitions) <= 12
        ):
            raise ValueError("QUESTION_TRACK_INVALID")
        ids.add(qid)
        last_day = ""
        last_state = -1
        for transition in transitions:
            if not isinstance(transition, dict):
                raise ValueError("QUESTION_TRANSITION_INVALID")
            key = (transition.get("sourceId"), transition.get("claimId"))
            if key not in refs:
                raise ValueError("QUESTION_TRANSITION_SOURCE_INVALID")
            source_day, _ = refs[key]
            state = transition.get("status")
            if (
                state not in STATES
                or source_day <= last_day
                or STATE_ORDER[state] < last_state
                or (last_state == 2 and STATE_ORDER[state] != 2)
                or not isinstance(transition.get("interpretation"), str)
                or len(transition["interpretation"]) < 30
                or not isinstance(transition.get("stillUnknown"), str)
                or len(transition["stillUnknown"]) < 15
            ):
                raise ValueError("QUESTION_TRANSITION_INVALID")
            last_day = source_day
            last_state = STATE_ORDER[state]
        if transitions[0]["status"] != "OPEN":
            raise ValueError("QUESTION_TRACK_MUST_OPEN_FIRST")


def evidence_ledger(packet: dict) -> list[dict]:
    """Every visible source-claim is logged without needing a model to restate it."""
    old = set(packet.get("previouslySeenSourceIds", []))
    return [
        {
            "sourceId": s["sourceId"],
            "claimId": claim["claimId"],
            "sourceDate": s["sourceDate"],
            "sourceUrl": s["url"],
            "factParaphrase": claim["text"],
            "newSincePreviousCutoff": s["sourceId"] not in old,
            "origin": "CURATOR_SOURCE_CLAIM_NOT_MODEL_INFERENCE",
            "historicalPublicationTimeVerified": False,
            "economicCausalityVerified": False,
        }
        for s in packet["eligibleSources"] for claim in s["claims"]
    ]


def question_snapshot(doc: dict, packet: dict) -> list[dict]:
    """Only transitions whose source facts are visible may change a question."""
    visible = {
        (s["sourceId"], c["claimId"]) for s in packet["eligibleSources"]
        for c in s["claims"]
    }
    result = []
    for question in doc.get("questionTracks", []):
        visible_transitions = [
            t for t in question["transitions"]
            if (t["sourceId"], t["claimId"]) in visible
        ]
        if not visible_transitions:
            continue
        latest = visible_transitions[-1]
        result.append({
            "questionId": question["questionId"],
            "question": question["question"],
            "scope": question["scope"],
            "status": latest["status"],
            "supportingSourceId": latest["sourceId"],
            "supportingClaimId": latest["claimId"],
            "interpretation": latest["interpretation"],
            "stillUnknown": latest["stillUnknown"],
            "priorVisibleTransitions": [
                {
                    "sourceId": t["sourceId"],
                    "claimId": t["claimId"],
                    "status": t["status"],
                } for t in visible_transitions[:-1]
            ],
            "origin": "CURATOR_REPLAY_VISIBLE_EVIDENCE_NOT_MODEL_INFERENCE",
            "semanticAccuracyVerified": False,
        })
    return result


def lifecycle_delta(early: dict, late: dict) -> list[dict]:
    previous = {x["questionId"]: x for x in early["questionLifecycle"]}
    return [
        {
            "questionId": item["questionId"],
            "previousStatus": previous[item["questionId"]]["status"]
            if item["questionId"] in previous else "NOT_YET_VISIBLE",
            "currentStatus": item["status"],
            "currentSupportingSourceId": item["supportingSourceId"],
            "currentSupportingClaimId": item["supportingClaimId"],
            "currentStillUnknown": item["stillUnknown"],
            "causalityVerified": False,
        }
        for item in late["questionLifecycle"]
        if item["questionId"] not in previous
        or item["status"] != previous[item["questionId"]]["status"]
    ]


def audit_temporal_consistency(
    packet: dict, assessment: dict, lifecycle: list[dict],
) -> dict:
    """Flag anchored model questions and narrow narrative contradictions.

    No model text is 'corrected' in place. No arbitrary question is matched
    by loose semantic similarity; only source+claim anchors are compared.
    """
    claims = {
        (s["sourceId"], c["claimId"]) for s in packet["eligibleSources"]
        for c in s["claims"]
    }
    model_changes = assessment.get("changeItems", [])
    missing = sorted({
        s["sourceId"] for s in packet["eligibleSources"]
        if s["sourceId"] not in set(packet.get("previouslySeenSourceIds", []))
    } - {
        x.get("sourceId") for x in model_changes
        if (x.get("sourceId"), x.get("claimId")) in claims
    })
    questions = assessment.get("unresolvedQuestions", [])
    resolved = [q for q in lifecycle if q["status"] == "RESOLVED_BY_LATER_SOURCE"]
    stale: list[dict] = []
    for q in resolved:
        anchors = {
            (t["sourceId"], t["claimId"])
            for t in q["priorVisibleTransitions"]
        }
        matching = [
            item for item in questions
            if (item.get("sourceId"), item.get("claimId")) in anchors
        ]
        if matching:
            stale.append({
                "questionId": q["questionId"],
                "resolutionSourceId": q["supportingSourceId"],
                "resolutionClaimId": q["supportingClaimId"],
                "staleUnresolvedModelQuestions": matching,
                "issueCode": "MODEL_QUESTION_ANCHOR_RETAINS_RESOLVED_STATE",
                "humanReviewRequired": True,
            })
    # This is a *warning*, not semantic verification. Narrative mentions may
    # refer to detailed committee minutes rather than a product decision.
    text = assessment.get("whatRemainsUnknown", "")
    formula = any(
        q["questionId"] == "fda-2023-xbb15-authorized-formula"
        for q in resolved
    )
    obsolete_formula = bool(
        formula
        and re.search(
            r"\b(?:composition|strain)\b.{0,100}"
            r"\b(?:remains?\s+(?:an?\s+)?unobserved\s+decision|"
            r"undetermined|not\s+(?:yet\s+)?decided)\b",
            text, flags=re.IGNORECASE,
        )
    )
    issue_codes = []
    if missing:
        issue_codes.append("MODEL_CHANGE_ITEMS_OMIT_VISIBLE_NEW_SOURCES")
    if stale:
        issue_codes.append("MODEL_RETAINS_POTENTIALLY_RESOLVED_QUESTION")
    if obsolete_formula:
        issue_codes.append("MODEL_NARRATIVE_POTENTIALLY_CONTRADICTS_LATER_FORMULA")
    return {
        "modelMissingChangeSourceIds": missing,
        "modelChangeItemCount": len(model_changes),
        "curatedChangeLedgerCount": len(evidence_ledger(packet)),
        "staleQuestionAnchors": stale,
        "potentialNarrativeFormulaContradiction": obsolete_formula,
        "issueCodes": issue_codes,
        "automaticModelCorrectionPerformed": False,
        "humanReviewRequired": True,
        "semanticAccuracyVerified": False,
        "historicalPITVerified": False,
    }
