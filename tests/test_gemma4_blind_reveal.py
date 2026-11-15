"""Human-only post-blind reveal: mapping pins, reviewer agreement and no promotion."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from test_gemma4_blind_review import complete, generate

from tris_market_event_intelligence.gemma4_blind_reveal import (
    reveal_blind_review,
)


def reveal(packet, mapping, reviews):
    return reveal_blind_review(
        packet=packet, mapping=mapping, reviews=reviews,
    )


def test_model_specific_counts_only_after_valid_human_review(
    monkeypatch, tmp_path: Path,
) -> None:
    packet, mapping, _, _ = generate(monkeypatch, tmp_path)
    a = complete(packet)
    b = copy.deepcopy(a)
    b["items"][0]["humanReview"]["preferredResponse"] = "B"
    first = reveal(packet, mapping, [("reviewer-1", a)])
    assert first["status"] == "HUMAN_REVIEW_REVEALED_DESCRIPTIVE_ONLY"
    assert first["modelPreferences"] == {"gemma2:latest": 1}
    assert first["modelRubricGradeCounts"]["gemma4:e4b"]["evidenceAccuracy"] == {
        "UNASSESSABLE": 1,
    }
    assert first["interReviewerPreferenceAgreement"]["status"] == (
        "NOT_ESTIMABLE_SINGLE_REVIEWER"
    )
    assert first["improvementProven"] is False
    assert first["modelPromotionAllowed"] is False
    second = reveal(packet, mapping, [
        ("reviewer-1", a), ("reviewer-2", b),
    ])
    assert second["reviewerCount"] == 2
    assert second["modelPreferences"] == {
        "gemma2:latest": 1, "gemma4:e4b": 1,
    }
    assert second["interReviewerPreferenceAgreement"]["exactAgreementRate"] == 0.0
    assert second["interReviewerPreferenceAgreement"]["cohensKappa"] == 0.0
    assert second["independentReviewerIdentityVerified"] is False
    assert "Consumer Price Index" not in json.dumps(second)
    assert a["items"][0]["humanReview"]["preferredResponse"] == "A"


def test_reversed_assignment_does_not_bias_models(
    monkeypatch, tmp_path: Path,
) -> None:
    packet, mapping, _, _ = generate(monkeypatch, tmp_path, reverse=True)
    summary = reveal(packet, mapping, [("reviewer-1", complete(packet))])
    assert summary["modelPreferences"] == {"gemma4:e4b": 1}
    assert summary["modelRubricGradeCounts"]["gemma4:e4b"]["evidenceAccuracy"] == {
        "SUPPORTED": 1,
    }


def test_no_unblind_if_any_review_unfinished(monkeypatch, tmp_path: Path) -> None:
    packet, mapping, _, _ = generate(monkeypatch, tmp_path)
    a = complete(packet)
    b = copy.deepcopy(a)
    b["humanReviewCompleted"] = False
    with pytest.raises(ValueError, match="completion"):
        reveal(packet, mapping, [("reviewer-a", a), ("reviewer-b", b)])


def test_no_unblind_with_tampered_packet_or_mapping(
    monkeypatch, tmp_path: Path,
) -> None:
    packet, mapping, _, _ = generate(monkeypatch, tmp_path)
    review = complete(packet)
    altered = copy.deepcopy(mapping)
    altered["packetSha256"] = "0" * 64
    with pytest.raises(ValueError, match="mapping"):
        reveal(packet, altered, [("reviewer-1", review)])
    altered = copy.deepcopy(mapping)
    altered["assignments"][0]["A"] = "gemma4:e4b"
    with pytest.raises(ValueError, match="assignment"):
        reveal(packet, altered, [("reviewer-1", review)])
    altered = copy.deepcopy(packet)
    altered["items"][0]["sourceContext"]["evidence"] = "changed"
    with pytest.raises(ValueError, match="different blind packet"):
        reveal(altered, mapping, [("reviewer-1", review)])


def test_mapping_wrong_event_and_noncanonical_model_fails(
    monkeypatch, tmp_path: Path,
) -> None:
    packet, mapping, _, _ = generate(monkeypatch, tmp_path)
    review = complete(packet)
    wrong = copy.deepcopy(mapping)
    wrong["assignments"][0]["eventId"] = 123
    with pytest.raises(ValueError, match="assignment"):
        reveal(packet, wrong, [("reviewer-1", review)])
    wrong = copy.deepcopy(mapping)
    wrong["assignments"][0]["B"] = "gemma4:latest"
    with pytest.raises(ValueError, match="explicitly sized"):
        reveal(packet, wrong, [("reviewer-1", review)])


def test_same_reviewer_identity_and_mismatched_document_rejected(
    monkeypatch, tmp_path: Path,
) -> None:
    packet, mapping, _, _ = generate(monkeypatch, tmp_path)
    one = complete(packet)
    with pytest.raises(ValueError, match="unique"):
        reveal(packet, mapping, [("reviewer-x", one), ("reviewer-x", one)])
    tampered = copy.deepcopy(one)
    tampered["items"][0]["eventId"] = 999
    with pytest.raises(ValueError, match="misaligned"):
        reveal(packet, mapping, [("reviewer-x", one), ("reviewer-y", tampered)])


def test_agreement_unanimous_has_undefined_kappa(
    monkeypatch, tmp_path: Path,
) -> None:
    packet, mapping, _, _ = generate(monkeypatch, tmp_path)
    one = complete(packet)
    summary = reveal(packet, mapping, [("human-1", one), ("human-2", one)])
    agreement = summary["interReviewerPreferenceAgreement"]
    assert agreement["exactAgreementRate"] == 1.0
    assert agreement["cohensKappa"] is None
    assert agreement["kappaUnavailableReason"] == "DEGENERATE_MARGINALS"


def test_invalid_reviewer_count_or_empty_id_rejected(
    monkeypatch, tmp_path: Path,
) -> None:
    packet, mapping, _, _ = generate(monkeypatch, tmp_path)
    review = complete(packet)
    with pytest.raises(ValueError, match="1..10"):
        reveal(packet, mapping, [])
    with pytest.raises(ValueError, match="identifier"):
        reveal(packet, mapping, [("", review)])
    with pytest.raises(ValueError, match="1..10"):
        reveal(packet, mapping, [
            (f"human-{n}", review) for n in range(11)
        ])
