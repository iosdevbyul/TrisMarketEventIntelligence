"""NEWS-only magnitude rubric: human labels and descriptive VALIDATION evaluation.

A headline about private conduct, an espionage allegation, and confirmed
leadership departure may have different conditional materiality even when
future market direction is uncertain. No price outcome supplies truth labels.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from tris_market_event_intelligence.news_importance_review import (
    _cohort,
    _instant,
    _read,
)
from tris_market_event_intelligence.news_materiality_hypothesis import (
    BANDS,
    NewsMaterialityHypothesis,
)

REVIEW_SCHEMA = "news_materiality_human_review_v1"
PREDICTION_SCHEMA = "news_materiality_predictions_v1"
MIN_DESCRIPTIVE_VALIDATION = 100


def prepare_materiality_review(cohort_path: Path) -> dict[str, object]:
    raw, cohort = _cohort(cohort_path)
    return {
        "schemaVersion": REVIEW_SCHEMA,
        "cohortSha256": hashlib.sha256(raw).hexdigest(),
        "judgments": [{
            "articleId": item["articleId"],
            "textSha256": item["textSha256"],
            "potentialMagnitude": None,
            "reviewer": None,
            "reviewedAt": None,
            "reason": None,
        } for item in cohort["items"]],
        "generatedHumanLabels": 0,
        "stockPricesUsed": False,
        "humanGoldIndependentlyAudited": False,
    }


def _validated_reviews(cohort: dict[str, object], raw_cohort: bytes,
                       reviews_path: Path) -> dict[int, str]:
    data = json.loads(_read(reviews_path))
    if not isinstance(data, dict) or set(data) != {
        "schemaVersion", "cohortSha256", "judgments", "generatedHumanLabels",
        "stockPricesUsed", "humanGoldIndependentlyAudited",
    } or data["schemaVersion"] != REVIEW_SCHEMA or (
        data["cohortSha256"] != hashlib.sha256(raw_cohort).hexdigest()
        or data["generatedHumanLabels"] != 0
        or data["stockPricesUsed"] is not False
        or data["humanGoldIndependentlyAudited"] is not False
    ):
        raise ValueError("Invalid source-pinned NEWS-only human magnitude review")
    entries = data["judgments"]
    articles = {item["articleId"]: item for item in cohort["items"]}
    if not isinstance(entries, list) or len(entries) != len(articles):
        raise ValueError("Magnitude review must preserve every article")
    seen: set[int] = set()
    labels: dict[int, str] = {}
    for row in entries:
        if not isinstance(row, dict) or set(row) != {
            "articleId", "textSha256", "potentialMagnitude", "reviewer",
            "reviewedAt", "reason",
        }:
            raise ValueError("Malformed magnitude human judgment")
        article_id = row["articleId"]
        if type(article_id) is not int or article_id not in articles or article_id in seen:
            raise ValueError("Unknown or repeated magnitude review article")
        seen.add(article_id)
        article = articles[article_id]
        if row["textSha256"] != article["textSha256"]:
            raise ValueError("Magnitude review source provenance mismatch")
        values = [row[key] for key in (
            "potentialMagnitude", "reviewer", "reviewedAt", "reason",
        )]
        if all(v is None for v in values):
            continue
        if row["potentialMagnitude"] not in BANDS and (
            row["potentialMagnitude"] != "UNASSESSABLE"
        ):
            raise ValueError("Unsupported human materiality band")
        if (not isinstance(row["reviewer"], str)
            or not 1 <= len(row["reviewer"].strip()) <= 100
            or not isinstance(row["reason"], str)
            or not 20 <= len(row["reason"].strip()) <= 2000
            or _instant(row["reviewedAt"]) < _instant(article["firstObservedAt"])):
            raise ValueError("Complete timestamped human magnitude assessment required")
        labels[article_id] = row["potentialMagnitude"]
    return labels


def _validation_ids(plan_path: Path, raw_cohort: bytes,
                    cohort: dict[str, object]) -> set[int]:
    doc = json.loads(_read(plan_path))
    if not isinstance(doc, dict) or doc.get("schemaVersion") != (
        "news_holdout_plan_v1"
    ) or doc.get("status") != "PLANNED_NOT_AUDITED" or (
        doc.get("cohortSha256") != hashlib.sha256(raw_cohort).hexdigest()
        or doc.get("stockPricesUsed") is not False
        or doc.get("trainingDatasetEligible") is not False
        or doc.get("humanReviewTimeGateApplied") is not True
    ):
        raise ValueError("Pinned NEWS temporal holdout required")
    articles = {item["articleId"]: item for item in cohort["items"]}
    items = doc.get("items")
    if not isinstance(items, list) or len(items) != len(articles):
        raise ValueError("Incomplete NEWS holdout")
    seen: set[int] = set()
    validation: set[int] = set()
    counts: Counter[str] = Counter()
    for row in items:
        if not isinstance(row, dict):
            raise TypeError("Malformed NEWS holdout row")
        article_id = row.get("articleId")
        if type(article_id) is not int or article_id not in articles or article_id in seen:
            raise ValueError("Repeated or unknown NEWS holdout article")
        seen.add(article_id)
        article = articles[article_id]
        if row.get("textSha256") != article["textSha256"] or (
            _instant(row.get("firstObservedAt"))
            != _instant(article["firstObservedAt"])
        ):
            raise ValueError("Holdout article provenance changed")
        split = row.get("split")
        if split not in {"TRAIN", "VALIDATION", "TEST", "EMBARGO", "QUARANTINED"}:
            raise ValueError("Unknown NEWS research partition")
        counts[split] += 1
        if split == "VALIDATION":
            validation.add(article_id)
    if dict(counts) != doc.get("counts"):
        raise ValueError("NEWS holdout counts changed")
    return validation


def evaluate_materiality_validation(*, cohort_path: Path, plan_path: Path,
                                    reviews_path: Path,
                                    predictions_path: Path) -> dict[str, object]:
    """Descriptive VAL only. Never score reserved TEST or pick a winner."""
    raw, cohort = _cohort(cohort_path)
    validation_ids = _validation_ids(plan_path, raw, cohort)
    labels = _validated_reviews(cohort, raw, reviews_path)
    payload = json.loads(_read(predictions_path))
    if not isinstance(payload, dict) or set(payload) != {
        "schemaVersion", "cohortSha256", "modelId", "promptVersion",
        "hypotheses",
    } or payload["schemaVersion"] != PREDICTION_SCHEMA or (
        payload["cohortSha256"] != hashlib.sha256(raw).hexdigest()
        or not isinstance(payload["modelId"], str)
        or not 1 <= len(payload["modelId"].strip()) <= 120
        or not isinstance(payload["promptVersion"], str)
        or not 1 <= len(payload["promptVersion"].strip()) <= 80
    ):
        raise ValueError("Invalid pinned NEWS materiality predictions")
    items = payload["hypotheses"]
    articles = {row["articleId"]: row for row in cohort["items"]}
    if not isinstance(items, list) or len(items) != len(articles):
        raise ValueError("Every original news article needs a hypothesis")
    predictions = {}
    for item in items:
        model = NewsMaterialityHypothesis.model_validate(item)
        if model.article_id not in articles or model.article_id in predictions:
            raise ValueError("Unknown or duplicate news hypothesis")
        article = articles[model.article_id]
        if model.text_sha256 != article["textSha256"]:
            raise ValueError("News hypothesis provenance mismatch")
        original_text = article["title"] + "\n" + (article.get("summary") or "")
        if model.evidence_excerpt not in original_text:
            raise ValueError("Claim evidence excerpt missing from original news")
        predictions[model.article_id] = model
    total_errors = 0
    counts: Counter[str] = Counter()
    absolute_errors = []
    evaluated = 0
    abstentions = 0
    for article_id in sorted(validation_ids):
        human = labels.get(article_id)
        if human is None or human == "UNASSESSABLE":
            continue
        predicted = predictions[article_id].potential_magnitude
        counts[human] += 1
        if predicted == "UNASSESSED":
            abstentions += 1
            continue
        gap = abs(BANDS.index(predicted) - BANDS.index(human))
        absolute_errors.append(gap)
        total_errors += int(gap != 0)
        evaluated += 1
    return {
        "schemaVersion": "news_materiality_validation_evaluation_v1",
        "status": (
            "DESCRIPTIVE_ONLY_NOT_INDEPENDENT_VALIDATION"
            if evaluated >= MIN_DESCRIPTIVE_VALIDATION
            else "INSUFFICIENT_REVIEWED_VALIDATION"
        ),
        "cohortSha256": hashlib.sha256(raw).hexdigest(),
        "planSha256": hashlib.sha256(_read(plan_path)).hexdigest(),
        "reviewsSha256": hashlib.sha256(_read(reviews_path)).hexdigest(),
        "predictionsSha256": hashlib.sha256(_read(predictions_path)).hexdigest(),
        "modelId": payload["modelId"],
        "promptVersion": payload["promptVersion"],
        "validationArticles": len(validation_ids),
        "validationHumanBandCounts": dict(sorted(counts.items())),
        "scoredValidationArticles": evaluated,
        "unassessedValidationPredictions": abstentions,
        "exactBandAgreement": 1 - total_errors / evaluated if evaluated else None,
        "meanOrdinalBandError": (
            sum(absolute_errors) / evaluated if evaluated else None
        ),
        "predictionRationalesIndependentlyVerified": False,
        "humanReviewerAuthenticated": False,
        "humanGoldIndependentlyAudited": False,
        "crossStoryLeakageIndependentlyCleared": False,
        "testGoldScored": False,
        "modelImprovementValidated": False,
        "promotionAllowed": False,
        "stockPricesUsed": False,
        "productionDbModified": False,
    }
