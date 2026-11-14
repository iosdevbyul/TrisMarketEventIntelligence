"""News temporal partitions quarantine duplicate events crossing evaluation periods."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tris_market_event_intelligence.historical_news_cohort import historical_news_cohort
from tris_market_event_intelligence.news_holdout_plan import plan_news_holdout
from test_historical_news_cohort import seeded


TRAIN_END = datetime(2026, 11, 2, tzinfo=UTC)
VALIDATION_END = datetime(2026, 11, 6, tzinfo=UTC)


def cohort(tmp_path: Path) -> Path:
    archive = seeded(tmp_path)
    original = historical_news_cohort(
        source_db=archive, as_of=datetime(2026, 11, 10, tzinfo=UTC),
    )
    assert len(original["items"]) == 4
    for item, received in zip(original["items"], [
        "2026-11-01T12:00:01+00:00",
        "2026-11-04T12:00:00+00:00",
        "2026-11-08T12:00:00+00:00",
        "2026-11-08T12:00:01+00:00",
    ], strict=True):
        item["firstObservedAt"] = received
    path = tmp_path / "cohort.json"
    path.write_text(json.dumps(original), encoding="utf-8")
    return path


def run(path: Path, **options: object) -> dict[str, object]:
    return plan_news_holdout(
        path, train_end=TRAIN_END, validation_end=VALIDATION_END, **options,
    )


def test_chronological_news_split_reserves_final_holdout(tmp_path: Path) -> None:
    path = cohort(tmp_path)
    original = path.read_bytes()
    report = run(path)
    assert [row["split"] for row in report["items"]] == [
        "TRAIN", "VALIDATION", "TEST", "TEST",
    ]
    assert report["counts"] == {"TEST": 2, "TRAIN": 1, "VALIDATION": 1}
    assert report["groupsConsidered"] == 4
    assert report["testSetReservedForFinalIndependentEvaluation"] is True
    assert report["trainingDatasetEligible"] is False
    assert report["stockPricesUsed"] is False
    assert report["nearDuplicateDetectionComplete"] is False
    assert path.read_bytes() == original
    assert "importanceHumanGold" not in report["items"][0]


def test_duplicate_text_spanning_train_validation_is_quarantined(tmp_path: Path) -> None:
    path = cohort(tmp_path)
    document = json.loads(path.read_text())
    document["items"][1]["title"] = document["items"][0]["title"]
    document["items"][1]["summary"] = document["items"][0]["summary"]
    document["items"][1]["textSha256"] = document["items"][0]["textSha256"]
    path.write_text(json.dumps(document))
    report = run(path)
    assert [row["split"] for row in report["items"]] == [
        "QUARANTINED", "QUARANTINED", "TEST", "TEST",
    ]
    assert all(
        row["exclusionReason"] == "DUPLICATE_GROUP_CROSSES_SPLITS"
        for row in report["items"][:2]
    )
    assert report["quarantinedArticles"] == 2
    assert report["groupsConsidered"] == 3


def test_same_story_group_spanning_validation_and_test_quarantined(
    tmp_path: Path,
) -> None:
    path = cohort(tmp_path)
    document = json.loads(path.read_text())
    document["items"][1]["storyGroupId"] = 42
    document["items"][2]["storyGroupId"] = 42
    path.write_text(json.dumps(document))
    report = run(path)
    assert [row["split"] for row in report["items"]] == [
        "TRAIN", "QUARANTINED", "QUARANTINED", "TEST",
    ]


def test_embargo_and_group_touching_embargo_are_excluded(tmp_path: Path) -> None:
    path = cohort(tmp_path)
    doc = json.loads(path.read_text())
    doc["items"][1]["firstObservedAt"] = "2026-11-02T12:00:00+00:00"
    report = run_with_document(path, doc)
    assert report["items"][1]["split"] == "EMBARGO"
    assert report["items"][1]["exclusionReason"] == "TEMPORAL_EMBARGO"
    doc["items"][1]["storyGroupId"] = 4
    report = run_with_document(path, doc)
    assert report["items"][0]["split"] == "QUARANTINED"
    assert report["items"][1]["split"] == "QUARANTINED"
    assert report["items"][0]["exclusionReason"] == (
        "DUPLICATE_GROUP_TOUCHES_EMBARGO"
    )


def run_with_document(path: Path, doc: dict) -> dict[str, object]:
    path.write_text(json.dumps(doc), encoding="utf-8")
    return run(path)


@pytest.mark.parametrize("overrides", [
    {"embargo_hours": 0},
    {"embargo_hours": True},
    {"embargo_hours": 721},
    {"embargo_hours": 1.2},
    {"train_end": datetime(2026, 11, 5, tzinfo=UTC)},
    {"validation_end": datetime(2026, 11, 2, tzinfo=UTC)},
    {"train_end": datetime(2026, 11, 2)},
])
def test_invalid_split_policy_fails_closed(
    tmp_path: Path, overrides: dict[str, object],
) -> None:
    path = cohort(tmp_path)
    kwargs = {
        "train_end": TRAIN_END,
        "validation_end": VALIDATION_END,
        "embargo_hours": 24,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        plan_news_holdout(path, **kwargs)


def test_source_text_tampering_is_rejected(tmp_path: Path) -> None:
    path = cohort(tmp_path)
    doc = json.loads(path.read_text())
    doc["items"][0]["title"] = "silently replaced with a later correction"
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="text hash"):
        run(path)


def test_future_availability_and_invalid_group_fail_closed(tmp_path: Path) -> None:
    path = cohort(tmp_path)
    doc = json.loads(path.read_text())
    doc["items"][0]["storyGroupId"] = True
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="story group"):
        run(path)


def test_no_market_returns_are_ever_included(tmp_path: Path) -> None:
    path = cohort(tmp_path)
    report = run(path)
    assert report["stockPricesUsed"] is False
    assert all(set(item) == {
        "articleId", "textSha256", "firstObservedAt", "split", "exclusionReason",
    } for item in report["items"])
