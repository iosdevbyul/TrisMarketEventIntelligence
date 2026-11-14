"""News-only CPU experiments must be bounded, repeatable and label-safe."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_news_holdout_plan import cohort

from tris_market_event_intelligence.news_holdout_plan import plan_news_holdout
from tris_market_event_intelligence.news_importance_review import prepare_review
from tris_market_event_intelligence.news_research_worker import (
    audit_research_journal,
    research_news_once,
)

TRAIN_END = datetime(2026, 11, 3, tzinfo=UTC)
VALIDATION_END = datetime(2026, 11, 9, tzinfo=UTC)


def files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    original = json.loads(cohort(tmp_path).read_text())
    prototype = original["items"][0]
    items = []
    for number in range(12):
        important = number % 2 == 0
        if number < 6:
            receipt = "2026-11-01T12:00:00+00:00"
        elif number < 10:
            receipt = "2026-11-06T12:00:00+00:00"
        else:
            receipt = "2026-11-11T12:00:00+00:00"
        title = (
            f"Major regulatory decision approved market-wide safety standard {number}"
            if important else
            f"Routine local community weekend library activities update {number}"
        )
        item = dict(prototype)
        item.update({
            "articleId": number + 1, "title": title, "summary": "Source-only detail",
            "firstObservedAt": receipt,
            "sourceUrl": f"https://example.org/research/{number + 1}",
            "storyGroupId": None,
            "textSha256": hashlib.sha256((title + "\nSource-only detail").encode()).hexdigest(),
        })
        items.append(item)
    original["asOf"] = "2026-11-12T23:00:00+00:00"
    original["items"] = items
    news_path = tmp_path / "news.json"
    news_path.write_text(json.dumps(original), encoding="utf-8")
    human = prepare_review(news_path)
    for number, row in enumerate(human["judgments"]):
        row.update({
            "importance": "IMPORTANT" if number % 2 == 0 else "UNIMPORTANT",
            "sourceAlignment": "SUPPORTED",
            "reviewer": "fixture-human",
            "reviewedAt": (
                "2026-11-02T00:00:00+00:00" if number < 6
                else "2026-11-12T00:00:00+00:00"
            ),
            "reason": "Independently reviewed news importance in source context.",
        })
    review_path = tmp_path / "human.json"
    review_path.write_text(json.dumps(human), encoding="utf-8")
    plan = plan_news_holdout(
        news_path, train_end=TRAIN_END,
        validation_end=VALIDATION_END, review_path=review_path,
    )
    assert plan["counts"] == {"TRAIN": 6, "VALIDATION": 4, "TEST": 2}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return news_path, plan_path, review_path, tmp_path / "research.sqlite3"


def step(paths: tuple[Path, Path, Path, Path]) -> dict[str, object]:
    return research_news_once(
        cohort_path=paths[0], plan_path=paths[1],
        review_path=paths[2], journal_path=paths[3],
    )


def test_repeated_without_new_news_researches_finite_alphas(tmp_path: Path) -> None:
    paths = files(tmp_path)
    original = [path.read_bytes() for path in paths[:3]]
    alphas = []
    for _ in range(5):
        result = step(paths)
        assert result["status"] == "DESCRIPTIVE_RESEARCH_ONLY"
        assert result["trainingArticles"] == 6
        assert result["validationArticles"] == 4
        assert result["testPartitionEvaluated"] is False
        assert result["modelPersistedOrPromoted"] is False
        assert result["modelImprovementValidated"] is False
        assert result["stockPricesUsed"] is False
        assert result["candidateValidationLogLoss"] >= 0
        alphas.append(result["smoothingAlpha"])
    assert alphas == [0.25, 0.5, 1.0, 2.0, 4.0]
    exhausted = step(paths)
    assert exhausted["status"] == "EXPERIMENT_GRID_EXHAUSTED"
    assert exhausted["modelTrained"] is False
    audit = audit_research_journal(paths[3])
    assert audit["totalIterations"] == 5
    assert [p.read_bytes() for p in paths[:3]] == original


def test_unreviewed_and_retroactive_train_gold_blocks_training(
    tmp_path: Path,
) -> None:
    paths = files(tmp_path)
    review = json.loads(paths[2].read_text())
    for number, row in enumerate(review["judgments"][:6]):
        if number % 2 == 0:
            for key in ("importance", "sourceAlignment", "reviewer",
                        "reviewedAt", "reason"):
                row[key] = None
        else:
            row["reviewedAt"] = "2026-11-04T00:00:00+00:00"
    paths[2].write_text(json.dumps(review))
    plan = plan_news_holdout(
        paths[0], train_end=TRAIN_END,
        validation_end=VALIDATION_END, review_path=paths[2],
    )
    paths[1].write_text(json.dumps(plan))
    report = step(paths)
    assert report["status"] == "BLOCKED_INSUFFICIENT_HUMAN_GOLD"
    assert report["trainingArticles"] == 0
    assert report["modelTrained"] is False
    assert audit_research_journal(paths[3])["totalIterations"] == 0


def test_test_labels_cannot_alter_news_validation_metrics(tmp_path: Path) -> None:
    paths = files(tmp_path)
    first = step(paths)
    review = json.loads(paths[2].read_text())
    for row in review["judgments"][10:]:
        row["importance"] = (
            "UNIMPORTANT" if row["importance"] == "IMPORTANT" else "IMPORTANT"
        )
    paths[2].write_text(json.dumps(review))
    plan = plan_news_holdout(
        paths[0], train_end=TRAIN_END,
        validation_end=VALIDATION_END, review_path=paths[2],
    )
    paths[1].write_text(json.dumps(plan))
    second = step(paths)
    assert second["status"] == "DESCRIPTIVE_RESEARCH_ONLY"
    assert second["candidateValidationLogLoss"] == first["candidateValidationLogLoss"]
    assert second["trainingArticles"] == first["trainingArticles"]
    assert second["validationArticles"] == first["validationArticles"]


def test_unpinned_holdout_and_missing_review_gate_fail_closed(tmp_path: Path) -> None:
    paths = files(tmp_path)
    plan = json.loads(paths[1].read_text())
    plan["humanReviewTimeGateApplied"] = False
    paths[1].write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="time-gated"):
        step(paths)
    assert not paths[3].exists()


def test_no_source_gold_tampering_or_market_inputs(tmp_path: Path) -> None:
    paths = files(tmp_path)
    doc = json.loads(paths[2].read_text())
    doc["judgments"][0]["textSha256"] = "0" * 64
    paths[2].write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="provenance"):
        step(paths)
    assert not paths[3].exists()
