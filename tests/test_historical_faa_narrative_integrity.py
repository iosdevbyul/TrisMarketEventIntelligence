"""Regression: real FAA v1 clipped narrative and resolved N/A answer."""
from __future__ import annotations

import copy
from datetime import date

from test_historical_official_event_chain import load as faa

from tris_market_event_intelligence import historical_official_event_chain_cli as chain
from tris_market_event_intelligence.historical_staged_reasoning import (
    review_staged_model_output,
)


def assessment(what_changed: str, unknown: str) -> dict:
    return {
        "stageOneModelDraft": {
            "modelAuthoredChangeItems": [{
                "sourceId": "faa-20240124-production",
                "claimId": "return-to-service",
                "changedAction": (
                    "The FAA approved the procedures for 737-9 MAX "
                    "return to service after required inspections."
                ),
            }],
            "modelAuthoredQuestionAssessments": [{
                "questionId": "faa-2024-7379-inspection-process-approval",
                "status": "RESOLVED_BY_VISIBLE_SOURCE",
                "sourceId": "faa-20240124-production",
                "claimId": "return-to-service",
                "explanation": (
                    "The FAA approved the 737-9 MAX inspection and "
                    "maintenance process, subject to aircraft inspection."
                ),
                "stillUnknown": unknown,
            }],
        },
        "whatChanged": what_changed,
        "whatRemainsUnknown": (
            "Individual aircraft inspection records and final Boeing "
            "production audit findings are not established in this release."
        ),
    }


def test_real_faa_clipped_sentence_and_na_placeholder_are_detected() -> None:
    packet = chain.prepare_replay(faa(), date(2024, 1, 25))
    a = assessment(
        "The FAA approved inspection and maintenance procedures for "
        "grounded Boeing 737-9 MAX aircraft, making them eligible to "
        "return to service after the required work is 1",
        "N/A_RESOLVED_BY_VISIBLE_SOURCE",
    )
    original = copy.deepcopy(a)
    review = review_staged_model_output(packet, a)
    assert a == original
    assert review["status"] == "REJECTED_MECHANICALLY_CORRUPTED_EXPLANATION"
    assert review["issues"] == [
        "STAGED_NARRATIVE_TRAILING_FRAGMENT",
        "STAGED_QUESTION_REMAINDER_PLACEHOLDER",
    ]
    assert review["stageTwoExplanationUsable"] is False
    assert review["placeholderQuestionUnknownFields"][0]["rawStillUnknown"] == (
        "N/A_RESOLVED_BY_VISIBLE_SOURCE"
    )
    assert review["modelTextRewritten"] is False
    assert review["semanticAccuracyVerified"] is False


def test_fda_style_resolved_decision_with_real_remaining_unknown_is_not_rejected() -> None:
    packet = chain.prepare_replay(faa(), date(2024, 1, 25))
    a = assessment(
        "The FAA approved the inspection procedures for Boeing 737-9 MAX "
        "aircraft on January 24. This does not mean that every airplane "
        "has already completed the mandatory maintenance.",
        "Individual aircraft inspection completion and return-to-service "
        "logs are not visible in the official release.",
    )
    review = review_staged_model_output(packet, a)
    assert review["issues"] == []
    assert review["placeholderQuestionUnknownFields"] == []
    assert review["stageTwoExplanationUsable"] is True
    assert review["semanticAccuracyVerified"] is False


def test_na_placeholder_alone_warns_without_falsely_rejecting_narrative() -> None:
    packet = chain.prepare_replay(faa(), date(2024, 1, 25))
    a = assessment(
        "The FAA approved inspection and maintenance procedures for "
        "grounded 737-9 MAX airplanes while maintaining oversight.",
        "N/A_RESOLVED_BY_VISIBLE_SOURCE",
    )
    review = review_staged_model_output(packet, a)
    assert review["issues"] == ["STAGED_QUESTION_REMAINDER_PLACEHOLDER"]
    assert review["stageTwoExplanationUsable"] is True
    assert review["semanticAccuracyVerified"] is False


def test_common_contracted_sentence_is_not_mistaken_for_real_faa_trailing_fragment() -> None:
    packet = chain.prepare_replay(faa(), date(2024, 1, 25))
    a = assessment(
        "The FAA approved the inspection process for the 737-9 MAX "
        "on January 24. The announcement does not establish completion "
        "of all aircraft inspections.",
        "Detailed inspection records were not published.",
    )
    review = review_staged_model_output(packet, a)
    assert "STAGED_NARRATIVE_TRAILING_FRAGMENT" not in review["issues"]
