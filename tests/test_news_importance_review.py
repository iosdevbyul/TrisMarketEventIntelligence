"""Human labels are explicit, provenance-pinned and news-only."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_historical_news_cohort import seeded

from tris_market_event_intelligence.historical_news_cohort import historical_news_cohort
from tris_market_event_intelligence.news_importance_review import (
    prepare_review,
    score_review,
)


def files(tmp_path: Path) -> tuple[Path, Path]:
    source = seeded(tmp_path)
    exported = historical_news_cohort(
        source_db=source, as_of=datetime(2026, 11, 1, 13, tzinfo=UTC),
    )
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps(exported), encoding="utf-8")
    return source, cohort_path


def review(tmp_path: Path, cohort: Path) -> tuple[Path, dict]:
    original = prepare_review(cohort)
    path = tmp_path / "review.json"
    path.write_text(json.dumps(original), encoding="utf-8")
    return path, original


def fill(record: dict, *, importance: str, alignment: str = "SUPPORTED") -> None:
    record.update({
        "importance": importance,
        "sourceAlignment": alignment,
        "reviewer": "independent-person",
        "reviewedAt": "2026-11-02T00:00:00+00:00",
        "reason": "Human reviewed the complete article context and news significance.",
    })


def test_blank_review_preserves_all_and_never_infers_importance(tmp_path: Path) -> None:
    source, cohort = files(tmp_path)
    with source.open("rb") as f:
        original_bytes = f.read()
    path, data = review(tmp_path, cohort)
    assert [row["articleId"] for row in data["judgments"]] == [1, 2, 3]
    assert all(row["importance"] is None for row in data["judgments"])
    assert all(row["sourceAlignment"] is None for row in data["judgments"])
    assert data["generatedLabels"] == 0
    result = score_review(cohort, path)
    assert result["status"] == "NO_HUMAN_REVIEWS"
    assert result["humanReviewedArticles"] == 0
    assert result["unreviewedArticles"] == 3
    assert result["stockPricesUsed"] is False
    assert result["trainingDatasetEligible"] is False
    with source.open("rb") as f:
        assert f.read() == original_bytes


def test_explicit_important_unimportant_uncertain_and_partial(tmp_path: Path) -> None:
    _, cohort = files(tmp_path)
    path, document = review(tmp_path, cohort)
    fill(document["judgments"][0], importance="IMPORTANT")
    fill(document["judgments"][1], importance="UNIMPORTANT", alignment="UNRELATED")
    path.write_text(json.dumps(document), encoding="utf-8")
    result = score_review(cohort, path)
    assert result["humanReviewedArticles"] == 2
    assert result["unreviewedArticles"] == 1
    assert result["importanceCounts"] == {"IMPORTANT": 1, "UNIMPORTANT": 1}
    assert result["sourceAlignmentCounts"] == {"SUPPORTED": 1, "UNRELATED": 1}
    assert result["potentialIndependentStoryUnits"] == 2
    assert result["reviewedStoryGroupCount"] == 1
    assert result["modelAccuracyVerified"] is False
    assert result["humanLabelsIndependentlyAudited"] is False
    assert result["stockPricesUsed"] is False


@pytest.mark.parametrize("tampering", [
    ("sourceUrl", "https://fake.org/story"),
    ("textSha256", "0" * 64),
    ("captureSha256", "1" * 64),
    ("firstObservedAt", "2026-11-01T12:00:04+00:00"),
    ("storyGroupId", 999),
])
def test_tampered_review_article_provenance_fails(
    tmp_path: Path, tampering: tuple[str, object],
) -> None:
    _, cohort = files(tmp_path)
    path, document = review(tmp_path, cohort)
    fill(document["judgments"][0], importance="IMPORTANT")
    field, new_value = tampering
    document["judgments"][0][field] = new_value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="provenance"):
        score_review(cohort, path)


@pytest.mark.parametrize("change", [
    {"importance": "VERY_IMPORTANT"},
    {"importance": "IMPORTANT", "sourceAlignment": None},
    {"importance": "IMPORTANT", "reviewer": ""},
    {"importance": "IMPORTANT", "reason": "short"},
    {"importance": "IMPORTANT", "reviewedAt": "2026-11-01T00:00:00+00:00"},
    {"importance": "IMPORTANT", "reviewedAt": "2026-11-02T00:00:00"},
])
def test_invalid_gold_or_review_chronology_fails(
    tmp_path: Path, change: dict[str, object],
) -> None:
    _, cohort = files(tmp_path)
    path, document = review(tmp_path, cohort)
    fill(document["judgments"][0], importance="UNIMPORTANT")
    document["judgments"][0].update(change)
    path.write_text(json.dumps(document))
    with pytest.raises((ValueError, TypeError)):
        score_review(cohort, path)


def test_missing_duplicate_or_added_articles_fail(tmp_path: Path) -> None:
    _, cohort = files(tmp_path)
    path, document = review(tmp_path, cohort)
    document["judgments"][1]["articleId"] = 1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="duplicated"):
        score_review(cohort, path)
    document["judgments"].pop()
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="preserve every article"):
        score_review(cohort, path)


def test_cohort_changed_after_review_fails(tmp_path: Path) -> None:
    _, cohort = files(tmp_path)
    path, _ = review(tmp_path, cohort)
    data = json.loads(cohort.read_text())
    data["items"][0]["title"] = "Post-hoc modified story"
    cohort.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="text hash mismatch"):
        score_review(cohort, path)


def test_review_must_be_after_source_and_cannot_be_auto_filled(tmp_path: Path) -> None:
    _, cohort = files(tmp_path)
    path, doc = review(tmp_path, cohort)
    doc["judgments"][0]["importance"] = "IMPORTANT"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Complete human importance"):
        score_review(cohort, path)
