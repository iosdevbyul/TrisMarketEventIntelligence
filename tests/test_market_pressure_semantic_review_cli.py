"""Tests for source-bound human semantic review, not mocked AI quality."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_market_pressure_replay_cli import _trace_fixture

from tris_market_event_intelligence.market_pressure_semantic_review_cli import (
    _create_only,
    semantic_review,
)


def test_review_exposes_actual_reasoning_and_provenance(tmp_path: Path) -> None:
    trace = tmp_path / "private.jsonl"
    _trace_fixture(trace)
    before = trace.read_bytes()
    report = semantic_review(trace)
    assert trace.read_bytes() == before
    assert report["schemaVersion"] == "market_pressure_semantic_review_v1"
    assert report["newModelCalls"] == 0
    assert report["dbWrites"] == 0
    assert report["economicAccuracyVerified"] is False
    assert report["acceptedBySchemaCount"] == 1
    assert report["rejectedBySchemaCount"] == 1
    assert len(report["reviewItems"]) == 2
    failed, accepted = report["reviewItems"]
    assert failed["proposalIndex"] == 0
    assert failed["rejectionCode"] == "IMPACT_ENTITY_MISMATCH"
    assert accepted["proposalIndex"] == 1
    assert accepted["validatorDisposition"] == "ACCEPTED_REQUIRES_SEMANTIC_REVIEW"
    assert "published price index" in accepted["mechanismRationale"]
    assert accepted["references"][0]["quotedObservation"] == (
        "consumer prices increased 0.3 percent"
    )
    assert accepted["references"][0]["sourceId"] == "s1"
    assert accepted["humanReview"]["humanDecision"] == "PENDING"
    assert accepted["economicReasoningQuality"]["gateStatus"] == (
        "HUMAN_SEMANTIC_REVIEW_REQUIRED"
    )
    assert accepted["economicReasoningQuality"]["economicAccuracyVerified"] is False
    assert accepted["humanReview"]["strengthSupported"] is None
    assert "SINGLE_CITED_SOURCE_NOT_INDEPENDENT_CORROBORATION" in (
        accepted["reviewFlags"]
    )


def test_semantic_review_template_is_create_only_and_not_auto_approved(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "private.jsonl"
    _trace_fixture(trace)
    result = semantic_review(trace)
    output = tmp_path / "semantic-review.json"
    _create_only(output, result)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "HUMAN_REVIEW_REQUIRED"
    assert saved["humanReviewedCount"] == 0
    assert saved["economicAccuracyVerified"] is False
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        _create_only(output, result)
    assert original == output.read_bytes()


def test_missing_model_response_does_not_claim_valid_semantics(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "incomplete.jsonl"
    _trace_fixture(trace, missing_impact=True)
    with pytest.raises(ValueError, match="REPLAY_MODEL_RESPONSE_MISSING"):
        semantic_review(trace)
