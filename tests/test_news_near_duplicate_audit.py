"""Candidate audit must detect paraphrased headlines but never assert truth."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_news_holdout_plan import cohort

from tris_market_event_intelligence.news_holdout_plan import plan_news_holdout
from tris_market_event_intelligence.news_near_duplicate_audit import (
    audit_news_near_duplicates,
)

TRAIN = datetime(2026, 11, 2, tzinfo=UTC)
VALIDATION = datetime(2026, 11, 6, tzinfo=UTC)


def files(tmp_path: Path) -> tuple[Path, Path]:
    news = cohort(tmp_path)
    document = json.loads(news.read_text())
    original = (
        "Regional banking regulator announces revised consumer loan "
        "policy review expected next week"
    )
    later = (
        "Regional banking regulator announces revised consumer loan "
        "policy review expected next month"
    )
    for row, title in zip(document["items"][:2], (original, later), strict=True):
        row["title"] = title
        row["textSha256"] = hashlib.sha256(
            (title + "\n" + row["summary"]).encode()
        ).hexdigest()
    news.write_text(json.dumps(document), encoding="utf-8")
    holdout = tmp_path / "holdout.json"
    plan = plan_news_holdout(
        news, train_end=TRAIN, validation_end=VALIDATION,
    )
    holdout.write_text(json.dumps(plan), encoding="utf-8")
    return news, holdout


def audit(paths: tuple[Path, Path], **kwargs: object) -> dict:
    return audit_news_near_duplicates(
        cohort_path=paths[0], holdout_path=paths[1], **kwargs,
    )


def test_cross_split_lexical_candidate_is_review_only(tmp_path: Path) -> None:
    paths = files(tmp_path)
    originals = [path.read_bytes() for path in paths]
    report = audit(paths)
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["crossPartitionCandidates"] == 1
    result = report["items"][0]
    assert (result["earlierArticleId"], result["laterArticleId"]) == (1, 2)
    assert (result["earlierPartition"], result["laterPartition"]) == (
        "TRAIN", "VALIDATION",
    )
    assert result["reviewStatus"] == "HUMAN_REVIEW_REQUIRED"
    assert result["confirmedSameStory"] is None
    assert report["candidateMatchesAreConfirmedDuplicates"] is False
    assert report["crossPageNearDuplicatesAudited"] is False
    assert report["humanGoldExposed"] is False
    assert report["stockPricesUsed"] is False
    assert [p.read_bytes() for p in paths] == originals


def test_no_cross_split_candidate_is_not_clearance(tmp_path: Path) -> None:
    paths = files(tmp_path)
    report = audit(paths, threshold=0.95)
    assert report["crossPartitionCandidates"] == 0
    assert report["status"] == "NO_LEXICAL_CANDIDATES_NOT_VERIFIED"
    assert report["semanticOrTranslatedNearDuplicatesFullyDetected"] is False
    assert report["trainingDatasetEligible"] is False


def test_same_split_near_duplicate_does_not_raise_cross_split_alarm(
    tmp_path: Path,
) -> None:
    paths = files(tmp_path)
    document = json.loads(paths[0].read_text())
    document["items"][1]["firstObservedAt"] = "2026-11-01T13:00:00+00:00"
    paths[0].write_text(json.dumps(document), encoding="utf-8")
    plan = plan_news_holdout(
        paths[0], train_end=TRAIN, validation_end=VALIDATION,
    )
    paths[1].write_text(json.dumps(plan), encoding="utf-8")
    assert audit(paths)["crossPartitionCandidates"] == 0


def test_modified_holdout_article_hash_rejected(tmp_path: Path) -> None:
    paths = files(tmp_path)
    doc = json.loads(paths[1].read_text())
    doc["items"][0]["textSha256"] = "0" * 64
    paths[1].write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="provenance"):
        audit(paths)


def test_modified_holdout_counts_rejected(tmp_path: Path) -> None:
    paths = files(tmp_path)
    doc = json.loads(paths[1].read_text())
    doc["counts"]["TRAIN"] = 99
    paths[1].write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="counts"):
        audit(paths)


def test_new_article_omitted_from_holdout_rejected(tmp_path: Path) -> None:
    paths = files(tmp_path)
    doc = json.loads(paths[1].read_text())
    doc["items"].pop()
    paths[1].write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="cover all"):
        audit(paths)


@pytest.mark.parametrize("settings", [
    {"threshold": 0.50},
    {"threshold": 0.96},
    {"threshold": True},
    {"max_days": 0},
    {"max_days": 31},
    {"max_days": True},
])
def test_invalid_research_limits_rejected(
    tmp_path: Path, settings: dict[str, object],
) -> None:
    paths = files(tmp_path)
    with pytest.raises(ValueError):
        audit(paths, **settings)
