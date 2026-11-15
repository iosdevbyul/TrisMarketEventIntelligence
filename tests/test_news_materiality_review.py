"""Magnitude must be independent of price direction and human labels."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_news_research_worker import files

from tris_market_event_intelligence.news_materiality_hypothesis import (
    NewsMaterialityHypothesis,
)
from tris_market_event_intelligence.news_materiality_review import (
    evaluate_materiality_validation,
    prepare_materiality_review,
)


def hypothesis(article, magnitude, *, direction="UNCERTAIN"):
    return {
        "article_id": article["articleId"],
        "text_sha256": article["textSha256"],
        "entity": "Illustrative public company",
        "event_summary": "Reported event involving senior leadership",
        "fact_status": "REPORTED",
        "potential_magnitude": magnitude,
        "direction": direction,
        "evidence_excerpt": article["title"],
        "conditional_mechanism": (
            "If the report is substantiated, management continuity may change."
        ),
        "verification_needed": ["Independent source confirmation"],
        "magnitude_basis": "HUMAN_OR_MODEL_HYPOTHESIS_UNCALIBRATED",
    }


def _inputs(tmp_path: Path):
    paths = files(tmp_path)
    cohort = json.loads(paths[0].read_text())
    review = prepare_materiality_review(paths[0])
    for row in review["judgments"]:
        row.update({
            "potentialMagnitude": "CRITICAL" if row["articleId"] % 2 else "LOW",
            "reviewer": "fixture-editor",
            "reviewedAt": "2026-11-13T00:00:00+00:00",
            "reason": "Hypothetical editorial assessment of organizational consequences.",
        })
    review_path = tmp_path / "magnitude_human.json"
    review_path.write_text(json.dumps(review))
    predicted = {
        "schemaVersion": "news_materiality_predictions_v1",
        "cohortSha256": review["cohortSha256"],
        "modelId": "fixture-research",
        "promptVersion": "magnitude-rubric-v1",
        "hypotheses": [
            hypothesis(item, "CRITICAL" if item["articleId"] % 2 else "LOW")
            for item in cohort["items"]
        ],
    }
    predictions_path = tmp_path / "magnitude_predicted.json"
    predictions_path.write_text(json.dumps(predicted))
    return paths, review_path, predictions_path


def _evaluate(paths, reviews, predictions):
    return evaluate_materiality_validation(
        cohort_path=paths[0], plan_path=paths[1],
        reviews_path=reviews, predictions_path=predictions,
    )


def test_direction_unknown_can_have_critical_magnitude() -> None:
    model = NewsMaterialityHypothesis.model_validate(
        hypothesis({
            "articleId": 5, "textSha256": "a" * 64,
            "title": "Senior executive confirms an unexpected resignation",
        }, "CRITICAL")
    )
    assert model.direction == "UNCERTAIN"
    assert model.potential_magnitude == "CRITICAL"
    assert model.market_prices_used is False
    assert "intensity_score" not in model.model_dump()


@pytest.mark.parametrize("change", [
    {"market_prices_used": True},
    {"price_prediction": True},
    {"predicted_return": 0.2},
    {"independent_human_gold": True},
    {"magnitude_basis": "NOT_ASSESSED"},
])
def test_no_price_predictions_or_fake_gold(change) -> None:
    row = hypothesis({
        "articleId": 5, "textSha256": "b" * 64,
        "title": "Senior executive confirms an unexpected resignation",
    }, "CRITICAL")
    row.update(change)
    with pytest.raises(ValidationError):
        NewsMaterialityHypothesis.model_validate(row)


def test_descriptive_validation_scores_only_validation_not_test(tmp_path: Path) -> None:
    paths, human, predicted = _inputs(tmp_path)
    original = [p.read_bytes() for p in paths[:3]]
    result = _evaluate(paths, human, predicted)
    assert result["validationArticles"] == 4
    assert result["scoredValidationArticles"] == 4
    assert result["exactBandAgreement"] == 1.0
    assert result["meanOrdinalBandError"] == 0.0
    assert result["testGoldScored"] is False
    assert result["modelImprovementValidated"] is False
    assert result["status"] == "INSUFFICIENT_REVIEWED_VALIDATION"
    modified = json.loads(human.read_text())
    for row in modified["judgments"][10:]:
        row["potentialMagnitude"] = "HIGH"
    human.write_text(json.dumps(modified))
    replay = _evaluate(paths, human, predicted)
    assert replay["exactBandAgreement"] == result["exactBandAgreement"]
    assert [p.read_bytes() for p in paths[:3]] == original


def test_source_evidence_hallucination_rejected(tmp_path: Path) -> None:
    paths, human, predicted = _inputs(tmp_path)
    document = json.loads(predicted.read_text())
    document["hypotheses"][0]["evidence_excerpt"] = (
        "This exact fabricated evidence string was never published in source"
    )
    predicted.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="missing from original"):
        _evaluate(paths, human, predicted)


def test_review_and_partition_provenance_fail_closed(tmp_path: Path) -> None:
    paths, human, predicted = _inputs(tmp_path)
    review = json.loads(human.read_text())
    review["judgments"][1]["textSha256"] = "0" * 64
    human.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="provenance"):
        _evaluate(paths, human, predicted)
    human.write_text(json.dumps(prepare_materiality_review(paths[0])))
    plan = json.loads(paths[1].read_text())
    plan["humanReviewTimeGateApplied"] = False
    paths[1].write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="holdout"):
        _evaluate(paths, human, predicted)


def test_incomplete_human_review_never_generates_gold(tmp_path: Path) -> None:
    paths, human, predicted = _inputs(tmp_path)
    human.write_text(json.dumps(prepare_materiality_review(paths[0])))
    output = _evaluate(paths, human, predicted)
    assert output["scoredValidationArticles"] == 0
    assert output["exactBandAgreement"] is None
    assert output["promotionAllowed"] is False
