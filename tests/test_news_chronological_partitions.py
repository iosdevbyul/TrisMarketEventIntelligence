"""Fail-closed research partition tests without prices or fictitious human gold."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_historical_news_cohort import seeded

from tris_market_event_intelligence.historical_news_cohort import historical_news_cohort
from tris_market_event_intelligence.news_chronological_partitions import (
    split_news_chronologically,
)
from tris_market_event_intelligence.news_importance_review import prepare_review


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    source = seeded(tmp_path)
    with sqlite3.connect(source) as db:
        db.execute(
            "UPDATE articles SET detected_at='2026-11-03T12:00:00+00:00' WHERE id=2"
        )
        db.execute(
            "UPDATE articles SET detected_at='2026-11-06T12:00:00+00:00' WHERE id=3"
        )
        db.execute(
            "INSERT INTO story_members VALUES (3,4,'y',NULL,'same_story','2026-11-06')"
        )
    cohort = historical_news_cohort(
        source_db=source, as_of=datetime(2026, 11, 10, tzinfo=UTC),
    )
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(prepare_review(cohort_path)), encoding="utf-8")
    return cohort_path, review_path


def partitions(paths: tuple[Path, Path], *, hours: int = 0) -> dict:
    return split_news_chronologically(
        cohort_path=paths[0], review_path=paths[1],
        train_end=datetime(2026, 11, 2, tzinfo=UTC),
        validation_end=datetime(2026, 11, 5, tzinfo=UTC),
        embargo_hours=hours,
    )


def test_cross_window_story_group_quarantined(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    prior = [p.read_bytes() for p in paths]
    outcome = partitions(paths)
    by_id = {row["articleId"]: row for row in outcome["items"]}
    assert by_id[1]["partition"] == "QUARANTINED"
    assert by_id[3]["partition"] == "QUARANTINED"
    assert by_id[1]["reason"] == "STORY_OR_DUPLICATE_CROSSES_SPLIT"
    assert by_id[2]["partition"] == "VALIDATION"
    assert by_id[4]["partition"] == "TEST"
    assert outcome["groupCount"] == 3
    assert outcome["partitionCounts"] == {"QUARANTINED": 2, "TEST": 1, "VALIDATION": 1}
    assert outcome["humanGoldExposedInOutput"] is False
    assert outcome["holdoutRepeatedTuningPrevented"] is False
    assert outcome["trainingDatasetEligible"] is False
    assert all(p.read_bytes() == snapshot for p, snapshot in zip(paths, prior, strict=True))


def test_same_article_text_hash_across_sources_quarantined(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    doc = json.loads(paths[0].read_text())
    # Artificial test case: same article text/body, different story index and URL.
    doc["items"][3]["textSha256"] = doc["items"][0]["textSha256"]
    doc["items"][3]["title"] = doc["items"][0]["title"]
    doc["items"][3]["summary"] = doc["items"][0]["summary"]
    paths[0].write_text(json.dumps(doc))
    paths[1].write_text(json.dumps(prepare_review(paths[0])))
    report = partitions(paths)
    assert report["items"][0]["partition"] == "QUARANTINED"
    assert report["items"][3]["partition"] == "QUARANTINED"


def test_same_canonical_url_cross_window_quarantined(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    doc = json.loads(paths[0].read_text())
    doc["items"][3]["sourceUrl"] = doc["items"][0]["sourceUrl"] + "?utm_source=weekly"
    paths[0].write_text(json.dumps(doc))
    paths[1].write_text(json.dumps(prepare_review(paths[0])))
    result = partitions(paths)
    assert result["items"][3]["partition"] == "QUARANTINED"


def test_human_labels_after_training_cutoff_not_available(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    doc = json.loads(paths[0].read_text())
    # Remove grouping in the test inputs to focus only on label availability.
    doc["items"][0]["storyGroupId"] = None
    doc["items"][2]["storyGroupId"] = None
    paths[0].write_text(json.dumps(doc))
    review = prepare_review(paths[0])
    review["judgments"][0].update({
        "importance": "IMPORTANT",
        "sourceAlignment": "SUPPORTED",
        "reviewer": "human-reviewer",
        "reviewedAt": "2026-11-07T00:00:00+00:00",
        "reason": "Human reviewed the original story and judged the news importance.",
    })
    paths[1].write_text(json.dumps(review))
    result = partitions(paths)
    assert result["items"][0]["partition"] == "TRAIN"
    assert result["items"][0]["labelAvailability"] == "REVIEW_NOT_AVAILABLE_BY_TRAIN_CUTOFF"
    assert "IMPORTANT" not in json.dumps(result)


def test_train_boundary_embargo_purges_nearby_article(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    doc = json.loads(paths[0].read_text())
    # Article 2 is within the embargo around 2026-11-02.
    doc["items"][1]["firstObservedAt"] = "2026-11-02T03:00:00+00:00"
    paths[0].write_text(json.dumps(doc))
    paths[1].write_text(json.dumps(prepare_review(paths[0])))
    result = partitions(paths, hours=12)
    assert result["items"][1]["partition"] == "QUARANTINED"
    assert result["items"][1]["reason"] == "PURGED_TRAIN_BOUNDARY"


@pytest.mark.parametrize("hours", [-1, 721, True, 40])
def test_invalid_embargo_or_collapsed_windows(tmp_path: Path, hours: int) -> None:
    paths = fixture(tmp_path)
    with pytest.raises(ValueError):
        partitions(paths, hours=hours)


def test_modified_human_review_provenance_fails(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    document = json.loads(paths[1].read_text())
    document["judgments"][0]["textSha256"] = "0" * 64
    paths[1].write_text(json.dumps(document))
    with pytest.raises(ValueError, match="provenance"):
        partitions(paths)


def test_invalid_naive_boundaries_fail(tmp_path: Path) -> None:
    paths = fixture(tmp_path)
    with pytest.raises(ValueError, match="timezones"):
        split_news_chronologically(
            cohort_path=paths[0], review_path=paths[1],
            train_end=datetime(2026, 11, 2),
            validation_end=datetime(2026, 11, 5, tzinfo=UTC),
        )
