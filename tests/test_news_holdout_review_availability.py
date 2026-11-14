"""Gold availability must be time-gated without exposing news gold contents."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_news_holdout_plan import cohort

from tris_market_event_intelligence.news_holdout_plan import plan_news_holdout
from tris_market_event_intelligence.news_importance_review import prepare_review

TRAIN = datetime(2026, 11, 2, tzinfo=UTC)
VALIDATION = datetime(2026, 11, 6, tzinfo=UTC)


def _inputs(tmp_path: Path, *, reviewed_at: str) -> tuple[Path, Path]:
    original = cohort(tmp_path)
    review_path = tmp_path / "human-review.json"
    review = prepare_review(original)
    review["judgments"][0].update({
        "importance": "IMPORTANT",
        "sourceAlignment": "SUPPORTED",
        "reviewer": "human-reviewer",
        "reviewedAt": reviewed_at,
        "reason": "The original article has independent news significance.",
    })
    review["judgments"][1].update({
        "importance": "UNIMPORTANT",
        "sourceAlignment": "SUPPORTED",
        "reviewer": "human-reviewer",
        "reviewedAt": "2026-11-09T00:00:00+00:00",
        "reason": "Ordinary news without a documented substantial new event.",
    })
    review_path.write_text(json.dumps(review), encoding="utf-8")
    return original, review_path


def _plan(original: Path, review: Path) -> dict[str, object]:
    return plan_news_holdout(
        original, train_end=TRAIN, validation_end=VALIDATION,
        embargo_hours=24, review_path=review,
    )


def test_train_gold_available_before_cutoff_without_label_leak(tmp_path: Path) -> None:
    original, review = _inputs(
        tmp_path, reviewed_at="2026-11-01T13:00:00+00:00",
    )
    before = original.read_bytes(), review.read_bytes()
    result = _plan(original, review)
    assert result["humanReviewTimeGateApplied"] is True
    assert len(result["humanReviewSha256"]) == 64
    assert result["items"][0]["split"] == "TRAIN"
    assert result["items"][0]["trainGoldAvailability"] == (
        "LABEL_AVAILABLE_BY_TRAIN_END_NOT_AUDITED"
    )
    assert result["items"][1]["split"] == "VALIDATION"
    assert result["items"][1]["trainGoldAvailability"] == "NOT_TRAIN_LABEL"
    assert result["items"][2]["split"] == "TEST"
    assert result["items"][2]["trainGoldAvailability"] == "NOT_TRAIN_LABEL"
    assert result["humanGoldValuesExposed"] is False
    assert result["trainingDatasetEligible"] is False
    assert "IMPORTANT" not in json.dumps(result)
    assert "UNIMPORTANT" not in json.dumps(result)
    assert (original.read_bytes(), review.read_bytes()) == before


def test_train_gold_written_after_cutoff_is_unavailable(tmp_path: Path) -> None:
    original, review = _inputs(
        tmp_path, reviewed_at="2026-11-04T00:00:00+00:00",
    )
    result = _plan(original, review)
    assert result["items"][0]["trainGoldAvailability"] == "REVIEWED_AFTER_TRAIN_END"
    assert result["trainingLabelsIndependentlyAudited"] is False


def test_unreviewed_training_item_is_not_assumed_negative(tmp_path: Path) -> None:
    original, review = _inputs(
        tmp_path, reviewed_at="2026-11-01T13:00:00+00:00",
    )
    doc = json.loads(review.read_text())
    for key in ("importance", "sourceAlignment", "reviewer", "reviewedAt", "reason"):
        doc["judgments"][0][key] = None
    review.write_text(json.dumps(doc))
    report = _plan(original, review)
    assert report["items"][0]["trainGoldAvailability"] == "UNREVIEWED"


def test_modified_human_review_fails_closed(tmp_path: Path) -> None:
    original, review = _inputs(
        tmp_path, reviewed_at="2026-11-01T13:00:00+00:00",
    )
    doc = json.loads(review.read_text())
    doc["judgments"][0]["textSha256"] = "0" * 64
    review.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="provenance"):
        _plan(original, review)


def test_retroactive_review_before_article_observed_fails(tmp_path: Path) -> None:
    original, review = _inputs(
        tmp_path, reviewed_at="2026-10-01T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="precede first observed"):
        _plan(original, review)


def test_existing_without_review_stays_compatible(tmp_path: Path) -> None:
    original = cohort(tmp_path)
    result = plan_news_holdout(
        original, train_end=TRAIN, validation_end=VALIDATION, embargo_hours=24,
    )
    assert result["humanReviewTimeGateApplied"] is False
    assert result["humanReviewSha256"] is None
    assert all("trainGoldAvailability" not in x for x in result["items"])
