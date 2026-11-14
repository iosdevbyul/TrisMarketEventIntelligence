"""News-only probability evaluation must not infer labels or claim improvements."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_historical_news_cohort import seeded

from tris_market_event_intelligence.historical_news_cohort import historical_news_cohort
from tris_market_event_intelligence.news_importance_evaluation import (
    evaluate_importance_predictions,
)
from tris_market_event_intelligence.news_importance_review import prepare_review


def prepared(tmp_path: Path) -> tuple[Path, Path, Path]:
    db = seeded(tmp_path)
    cohort = historical_news_cohort(
        source_db=db, as_of=datetime(2026, 11, 1, 13, tzinfo=UTC),
    )
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
    review = prepare_review(cohort_path)
    for i, row in enumerate(review["judgments"]):
        row["importance"] = ("IMPORTANT", "UNIMPORTANT", "UNCERTAIN")[i]
        row["sourceAlignment"] = "SUPPORTED"
        row["reviewer"] = "human-reviewer"
        row["reviewedAt"] = "2026-11-02T00:00:00+00:00"
        row["reason"] = "Reviewed the news context and independently assessed importance."
    review_path = tmp_path / "human.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    predictions = {
        "schemaVersion": "news_importance_probabilities_v1",
        "cohortSha256": review["cohortSha256"],
        "modelId": "fixture-model",
        "promptVersion": "fixture-v1",
        "generatedAt": "2026-11-01T13:00:00+00:00",
        "predictions": [{
            "articleId": item["articleId"],
            "textSha256": item["textSha256"],
            "probabilities": scores,
        } for item, scores in zip(cohort["items"], [
            {"IMPORTANT": 0.8, "UNIMPORTANT": 0.1, "UNCERTAIN": 0.1},
            {"IMPORTANT": 0.1, "UNIMPORTANT": 0.8, "UNCERTAIN": 0.1},
            {"IMPORTANT": 0.1, "UNIMPORTANT": 0.1, "UNCERTAIN": 0.8},
        ], strict=True)],
    }
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    return cohort_path, review_path, predictions_path


def score(paths: tuple[Path, Path, Path]) -> dict[str, object]:
    cohort, review, predictions = paths
    return evaluate_importance_predictions(
        cohort_path=cohort, review_path=review, predictions_path=predictions,
    )


def test_multiclass_importance_probability_scoring_is_descriptive(tmp_path: Path) -> None:
    files = prepared(tmp_path)
    before = [path.read_bytes() for path in files]
    result = score(files)
    assert result["status"] == "INSUFFICIENT_HUMAN_GOLD"
    assert result["articlesInOriginalCohort"] == 3
    assert result["humanReviewedArticles"] == 3
    assert result["scoredArticles"] == 2
    assert result["humanUncertainExcludedFromScoring"] == 1
    assert result["model"]["accuracy"] == 1.0
    assert result["model"]["importantPrecision"] == 1.0
    assert result["model"]["importantRecall"] == 1.0
    assert result["model"]["falseImportantRateOnUnimportant"] == 0.0
    assert result["model"]["multiclassBrier"] < result["alwaysUnimportantBaseline"][
        "multiclassBrier"
    ]
    assert result["modelPerformanceImprovementValidated"] is False
    assert result["trainingDatasetEligible"] is False
    assert result["stockMarketDataUsed"] is False
    assert [path.read_bytes() for path in files] == before


def test_missing_human_gold_is_not_implicitly_negative(tmp_path: Path) -> None:
    files = prepared(tmp_path)
    review = json.loads(files[1].read_text())
    for key in ("importance", "sourceAlignment", "reviewer", "reviewedAt", "reason"):
        review["judgments"][1][key] = None
    files[1].write_text(json.dumps(review))
    result = score(files)
    assert result["scoredArticles"] == 1
    assert result["unreviewedArticles"] == 1
    assert result["model"]["falseImportantRateOnUnimportant"] is None


def test_predictions_after_review_do_not_silently_count(tmp_path: Path) -> None:
    files = prepared(tmp_path)
    predictions = json.loads(files[2].read_text())
    predictions["generatedAt"] = "2026-11-03T00:00:00+00:00"
    files[2].write_text(json.dumps(predictions))
    result = score(files)
    assert result["scoredArticles"] == 0
    assert result["predictionsLaterThanReviewExcluded"] == 3
    assert result["model"]["accuracy"] is None


@pytest.mark.parametrize("alter", [
    lambda data: data["predictions"].pop(),
    lambda data: data["predictions"][1].update({"articleId": 1}),
    lambda data: data["predictions"][0].update({"textSha256": "0" * 64}),
    lambda data: data["predictions"][0]["probabilities"].update({"IMPORTANT": 1.1}),
    lambda data: data["predictions"][0]["probabilities"].update({"IMPORTANT": -0.1}),
    lambda data: data["predictions"][0]["probabilities"].update({"IMPORTANT": True}),
    lambda data: data["predictions"][0]["probabilities"].update({"IMPORTANT": "0.8"}),
    lambda data: data["predictions"][0]["probabilities"].update({"EXTRA": 0.0}),
    lambda data: data.update({"generatedAt": "2026-11-01T12:00:00+00:00"}),
    lambda data: data.update({"cohortSha256": "0" * 64}),
    lambda data: data.update({"direction": "POSITIVE"}),
])
def test_invalid_or_leaky_predictions_fail_closed(tmp_path: Path, alter) -> None:
    files = prepared(tmp_path)
    data = json.loads(files[2].read_text())
    alter(data)
    files[2].write_text(json.dumps(data))
    with pytest.raises((ValueError, TypeError)):
        score(files)


def test_tampered_review_or_article_hash_fails(tmp_path: Path) -> None:
    files = prepared(tmp_path)
    data = json.loads(files[1].read_text())
    data["judgments"][0]["sourceUrl"] = "https://attacker.example/not-source"
    files[1].write_text(json.dumps(data))
    with pytest.raises(ValueError, match="provenance"):
        score(files)
