"""Offline cross-event transfer review of saved date-only staged Gemma studies.

No model calls, no model output rewriting and no assertion of causal or
generalized semantic validity. FAA is a separate regulatory event challenge.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from tris_market_event_intelligence.historical_cross_domain_official_study_cli import (
    _require_new_output,
)
from tris_market_event_intelligence.historical_event_chain_explanation_quality import (
    audit_sequence,
)
from tris_market_event_intelligence.historical_evidence_decision_consistency import (
    review_evidence_decisions,
)
from tris_market_event_intelligence.historical_independent_news_cli import (
    _digest,
    _read,
    _write,
)
from tris_market_event_intelligence.historical_official_event_chain_cli import (
    prepare_replay,
)
from tris_market_event_intelligence.historical_staged_reasoning import (
    compare_model_questions_to_curator,
    review_staged_model_output,
    visible_questions,
)
from tris_market_event_intelligence.historical_temporal_question_lifecycle import (
    question_snapshot,
    validate_question_tracks,
)

VERSION = "historical_cross_event_staged_transfer_v1"


def review_study(doc: dict, saved: dict, annotations: dict) -> dict:
    """Reconstruct *only* eligible question text at each saved cutoff."""
    audit_sequence(doc, saved)
    validate_question_tracks(doc, annotations)
    if saved.get("reasoningMode") != "staged":
        raise ValueError("TRANSFER_REQUIRES_STAGED_MODEL_OUTPUT")
    old_ids: set[str] = set()
    rows = []
    for step in saved["replaySteps"]:
        packet = prepare_replay(doc, date.fromisoformat(step["asOfDateExclusive"]))
        packet["previouslySeenSourceIds"] = sorted(old_ids)
        assessment = step["assessment"]
        original = assessment.get("stageOneModelDraft")
        if (
            not isinstance(original, dict)
            or not isinstance(original.get("modelAuthoredChangeItems"), list)
            or not isinstance(original.get("modelAuthoredQuestionAssessments"), list)
        ):
            raise TypeError("TRANSFER_MISSING_STAGED_MODEL_DRAFT")
        draft = {
            **original,
            "visibleQuestions": visible_questions(annotations, packet),
        }
        valid_refs = {
            (s["sourceId"], c["claimId"])
            for s in packet["eligibleSources"] for c in s["claims"]
        }
        recent = {
            s["sourceId"] for s in packet["eligibleSources"]
            if s["sourceId"] not in old_ids
        }
        changes = draft["modelAuthoredChangeItems"]
        models = draft["modelAuthoredQuestionAssessments"]
        if any(
            (x.get("sourceId"), x.get("claimId")) not in valid_refs
            for x in changes + models
        ):
            raise ValueError("TRANSFER_UNAVAILABLE_MODEL_CLAIM_REFERENCE")
        expected_questions = {x["questionId"] for x in draft["visibleQuestions"]}
        if (
            {x.get("questionId") for x in models} != expected_questions
            or len(models) != len(expected_questions)
        ):
            raise ValueError("TRANSFER_MODEL_QUESTION_COVERAGE_INVALID")
        stage_review = review_staged_model_output(packet, assessment)
        consistency = review_evidence_decisions(packet, draft)
        curator = question_snapshot(annotations, packet)
        status_comparison = compare_model_questions_to_curator(draft, curator)
        covered = {x["sourceId"] for x in changes}
        missing_new = sorted(recent - covered)
        issue_codes = list(consistency["issueCodes"])
        if missing_new:
            issue_codes.append("MISSING_NEW_SOURCE_COVERAGE")
        if not stage_review["stageTwoExplanationUsable"]:
            issue_codes.append("UNUSABLE_STAGE_TWO_NARRATIVE")
        rows.append({
            "asOfDateExclusive": step["asOfDateExclusive"],
            "newEligibleSourceIds": sorted(recent),
            "newSourceChangeCoverage": len(recent) - len(missing_new),
            "missingNewSourceIds": missing_new,
            "sourceChangeCount": len(changes),
            "visibleResearchQuestionCount": len(models),
            "modelVsCuratorStatusDisagreements": status_comparison[
                "modelVsCuratorDisagreements"
            ],
            "stagedNarrativeReview": stage_review,
            "evidenceDecisionReview": consistency,
            "issueCodes": issue_codes,
            "historicalPITVerified": False,
            "semanticAccuracyVerified": False,
        })
        old_ids = {s["sourceId"] for s in packet["eligibleSources"]}
    return {
        "eventId": doc["eventId"],
        "sourceCohortSha256": _digest(doc),
        "savedModelId": saved["replaySteps"][0].get("modelId"),
        "savedModelCalls": saved.get("modelCalls"),
        "replayStepReviews": rows,
        "newSourceCoverageComplete": all(
            not x["missingNewSourceIds"] for x in rows
        ),
        "noMechanicallyRejectedNarratives": all(
            x["stagedNarrativeReview"]["stageTwoExplanationUsable"]
            for x in rows
        ),
        "noKnownPatternIssues": all(not x["issueCodes"] for x in rows),
        "modelCuratorDisagreementCount": sum(
            len(x["modelVsCuratorStatusDisagreements"]) for x in rows
        ),
        "historicalPITVerified": False,
        "semanticAccuracyVerified": False,
    }


def compare_independent_events(
    reference_doc: dict, reference_run: dict, reference_tracks: dict,
    challenge_doc: dict, challenge_run: dict, challenge_tracks: dict,
) -> dict:
    if (
        reference_doc.get("eventId") == challenge_doc.get("eventId")
        or _digest(reference_doc) == _digest(challenge_doc)
    ):
        raise ValueError("TRANSFER_EVENTS_MUST_BE_INDEPENDENT")
    reference = review_study(reference_doc, reference_run, reference_tracks)
    challenge = review_study(challenge_doc, challenge_run, challenge_tracks)
    if reference["savedModelId"] != challenge["savedModelId"]:
        raise ValueError("TRANSFER_MODEL_ID_MISMATCH")
    return {
        "schemaVersion": VERSION,
        "status": "CROSS_EVENT_TRANSFER_REQUIRES_HUMAN_REVIEW",
        "referenceEvent": reference,
        "independentChallengeEvent": challenge,
        "differentEventIdsVerified": True,
        "sameModelIdVerified": True,
        "auditModelCalls": 0,
        "modelOutputsRewritten": False,
        "curatorQuestionTransitionsNotSuppliedToModel": True,
        "curatorStatesAreEditorialNotGroundTruth": True,
        "historicalPITVerified": False,
        "semanticAccuracyVerified": False,
        "generalizationVerified": False,
        "causalityVerified": False,
        "autoPromotion": False,
        "pricesRead": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare saved FDA baseline and independent FAA staged study",
    )
    for prefix in ("reference", "challenge"):
        parser.add_argument(f"--{prefix}-chain", type=Path, required=True)
        parser.add_argument(f"--{prefix}-run", type=Path, required=True)
        parser.add_argument(f"--{prefix}-tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require_new_output(args.output)
    report = compare_independent_events(
        _read(args.reference_chain),
        _read(args.reference_run),
        _read(args.reference_tracks),
        _read(args.challenge_chain),
        _read(args.challenge_run),
        _read(args.challenge_tracks),
    )
    _write(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "auditModelCalls": 0,
        "referenceEventId": report["referenceEvent"]["eventId"],
        "challengeEventId": report["independentChallengeEvent"]["eventId"],
        "referenceDisagreements": report["referenceEvent"][
            "modelCuratorDisagreementCount"
        ],
        "challengeDisagreements": report["independentChallengeEvent"][
            "modelCuratorDisagreementCount"
        ],
        "generalizationVerified": False,
        "historicalPITVerified": False,
        "outputCreated": True,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
