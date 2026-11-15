"""A model-selected NEWS entity pair never becomes an independent fact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_news_research_worker import files

from tris_market_event_intelligence.news_materiality_hypothesis import (
    NewsMaterialityHypothesis,
)
from tris_market_event_intelligence.news_materiality_local import (
    _story_transitions,
)
from tris_market_event_intelligence.news_materiality_transition_review import (
    evaluate_transition_review,
    prepare_transition_review,
)


def _fixture(tmp_path: Path):
    paths = files(tmp_path)
    doc = json.loads(paths[0].read_text())
    models = []
    for article in doc["items"]:
        models.append(NewsMaterialityHypothesis(
            article_id=article["articleId"],
            text_sha256=article["textSha256"],
            entity=("Major" if article["articleId"] <= 10 else "Routine"),
            event_summary="Important hypothetical event affects corporate organization",
            fact_status="REPORTED",
            potential_magnitude=(
                "LOW" if article["articleId"] % 2 == 0 else "CRITICAL"
            ),
            direction="UNCERTAIN",
            evidence_excerpt=article["title"],
            conditional_mechanism=(
                "If the reported event proves true, the organization may "
                "need to change its governance processes."
            ),
            verification_needed=("Confirm reporting",),
            magnitude_basis="HUMAN_OR_MODEL_HYPOTHESIS_UNCALIBRATED",
        ))
    # Entity strings must really appear in source title, so use a shared
    # word found in all four VALIDATION fixture headlines.
    for i in range(6, 10):
        models[i] = models[i].model_copy(update={"entity": "regulatory" if (
            "regulatory" in doc["items"][i]["title"]
        ) else "Routine"})
    # For TRAIN we keep source-grounded entity strings, too.
    for i, item in enumerate(doc["items"]):
        if i < 6 or i >= 10:
            models[i] = models[i].model_copy(update={
                "entity": "regulatory" if "regulatory" in item["title"] else "Routine"
            })
    result = {
        "schemaVersion": "news_materiality_local_research_v1",
        "status": "RESEARCH_HYPOTHESES_UNVERIFIED",
        "cohortSha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        "modelId": "test:news",
        "promptVersion": "test-v1",
        "result": {
            "schemaVersion": "news_materiality_predictions_v1",
            "cohortSha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
            "modelId": "test:news", "promptVersion": "test-v1",
            "hypotheses": [m.model_dump() for m in models],
        },
        "transitions": _story_transitions(doc["items"], models),
        "transitionsAreIndependentFacts": False,
        "stockPricesUsed": False, "humanGoldUsed": False,
        "promotionAllowed": False,
    }
    report = tmp_path / "model-report.json"
    report.write_text(json.dumps(result))
    return paths, report


def _prepare(paths, report):
    return prepare_transition_review(
        cohort_path=paths[0], holdout_path=paths[1], report_path=report,
    )


def _evaluate(paths, report, review):
    return evaluate_transition_review(
        cohort_path=paths[0], holdout_path=paths[1],
        report_path=report, human_review_path=review,
    )


def test_only_validation_pairs_in_blind_packet(tmp_path: Path):
    paths, report = _fixture(tmp_path)
    template = _prepare(paths, report)
    assert len(template["judgments"]) == 2
    assert all(7 <= j["earlierArticleId"] <= 10 for j in template["judgments"])
    assert all(7 <= j["laterArticleId"] <= 10 for j in template["judgments"])
    assert "earlierMagnitude" not in json.dumps(template)
    assert "ordinalDifference" not in json.dumps(template)
    assert template["modelMagnitudesExposedInTemplate"] is False
    assert not paths[3].exists()


def test_human_same_event_change_is_descriptive_not_validated(tmp_path: Path):
    paths, report = _fixture(tmp_path)
    template = _prepare(paths, report)
    for j in template["judgments"]:
        j.update({
            "relation": "SAME_EVENT",
            "humanChange": "INCREASE",
            "reviewer": "synthetic-editor",
            "reviewedAt": "2026-11-13T00:00:00+00:00",
            "reason": "Hypothetical source comparison with independent editorial reasoning.",
        })
    review = tmp_path / "paired-review.json"
    review.write_text(json.dumps(template))
    result = _evaluate(paths, report, review)
    assert result["scoredSameEventPairs"] == 2
    assert 0 <= result["sameEventDirectionAgreement"] <= 1
    assert result["status"] == "INSUFFICIENT_HUMAN_TRANSITION_GOLD"
    assert result["testGoldScored"] is False
    assert result["modelImprovementValidated"] is False
    assert result["promotionAllowed"] is False


def test_different_event_not_counted_as_change(tmp_path: Path):
    paths, report = _fixture(tmp_path)
    template = _prepare(paths, report)
    j = template["judgments"][0]
    j.update({
        "relation": "DIFFERENT_EVENT",
        "humanChange": "UNASSESSABLE",
        "reviewer": "synthetic-editor",
        "reviewedAt": "2026-11-13T00:00:00+00:00",
        "reason": "These two stories share an entity but not an underlying event.",
    })
    review = tmp_path / "review.json"
    review.write_text(json.dumps(template))
    result = _evaluate(paths, report, review)
    assert result["differentEvents"] == 1
    assert result["scoredSameEventPairs"] == 0
    assert result["unreviewedPairs"] == 1
    template["judgments"][0]["humanChange"] = "INCREASE"
    review.write_text(json.dumps(template))
    with pytest.raises(ValueError, match="Different or uncertain"):
        _evaluate(paths, report, review)


def test_bad_model_transition_or_extra_human_pairs_refused(tmp_path: Path):
    paths, report = _fixture(tmp_path)
    template = _prepare(paths, report)
    review = tmp_path / "review.json"
    template["judgments"].append(dict(template["judgments"][0]))
    review.write_text(json.dumps(template))
    with pytest.raises(ValueError, match="complete selected pair set"):
        _evaluate(paths, report, review)
    original = json.loads(report.read_text())
    original["transitions"][0]["ordinalDifference"] = 99
    report.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="Unreplayable"):
        _prepare(paths, report)


def test_review_pinned_to_model_cohort_and_holdout(tmp_path: Path):
    paths, report = _fixture(tmp_path)
    template = _prepare(paths, report)
    template["holdoutSha256"] = "0" * 64
    review = tmp_path / "review.json"
    review.write_text(json.dumps(template))
    with pytest.raises(ValueError, match="Invalid pinned"):
        _evaluate(paths, report, review)
    template["holdoutSha256"] = hashlib.sha256(paths[1].read_bytes()).hexdigest()
    template["judgments"][0]["reviewedAt"] = "2026-11-13T00:00:00+00:00"
    review.write_text(json.dumps(template))
    with pytest.raises(ValueError, match="Unsupported human"):
        _evaluate(paths, report, review)


def test_empty_entity_pairs_returns_no_gold_not_success(tmp_path: Path):
    paths, report = _fixture(tmp_path)
    original = json.loads(report.read_text())
    original["transitions"] = []
    report.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="Unreplayable"):
        _prepare(paths, report)
