"""Conservative news-only importance probability evaluation against human labels.

Descriptive scores require independent human judgments. They do not evaluate
stock movement, conditional market-pressure direction or the actual model
quality in the absence of validated human gold and a protected holdout.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from tris_market_event_intelligence.news_importance_review import (
    IMPORTANCE,
    _cohort,
    _instant,
    _read,
    score_review,
)

SCHEMA = "news_importance_probabilities_v1"
LABELS = ("IMPORTANT", "UNIMPORTANT", "UNCERTAIN")
MIN_DESCRIPTIVE_REVIEWS = 100


def _probabilities(value: object) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != IMPORTANCE:
        raise ValueError("Every importance class needs an explicit probability")
    if any(
        type(value[label]) not in {float, int}
        or not math.isfinite(value[label])
        or not 0.0 <= value[label] <= 1.0
        for label in LABELS
    ) or not math.isclose(sum(value.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Importance probabilities must be finite and sum to one")
    return {label: float(value[label]) for label in LABELS}


def _metrics(
    predictions: list[tuple[str, dict[str, float]]],
) -> dict[str, object]:
    if not predictions:
        return {
            "n": 0, "accuracy": None, "importantPrecision": None,
            "importantRecall": None, "macroF1": None,
            "multiclassBrier": None, "meanLogLoss": None,
            "falseImportantRateOnUnimportant": None,
        }
    count = len(predictions)
    confusion: Counter[str] = Counter()
    brier = logloss = correct = 0.0
    for true, probabilities in predictions:
        predicted = max(LABELS, key=lambda label: probabilities[label])
        confusion[f"{true}/{predicted}"] += 1
        correct += predicted == true
        brier += sum(
            (probabilities[label] - float(label == true)) ** 2 for label in LABELS
        ) / len(LABELS)
        logloss -= math.log(max(probabilities[true], 1e-15))
    tp = confusion["IMPORTANT/IMPORTANT"]
    fp = sum(
        confusion[f"{label}/IMPORTANT"] for label in LABELS if label != "IMPORTANT"
    )
    fn = sum(
        confusion[f"IMPORTANT/{label}"] for label in LABELS if label != "IMPORTANT"
    )
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1s = []
    for label in LABELS:
        true_positive = confusion[f"{label}/{label}"]
        pred_total = sum(confusion[f"{other}/{label}"] for other in LABELS)
        true_total = sum(confusion[f"{label}/{other}"] for other in LABELS)
        f1s.append(
            2 * true_positive / (pred_total + true_total)
            if pred_total + true_total else 0.0
        )
    unimportant = sum(
        confusion[f"UNIMPORTANT/{label}"] for label in LABELS
    )
    return {
        "n": count,
        "accuracy": correct / count,
        "importantPrecision": precision,
        "importantRecall": recall,
        "macroF1": sum(f1s) / len(f1s),
        "multiclassBrier": brier / count,
        "meanLogLoss": logloss / count,
        "falseImportantRateOnUnimportant": (
            confusion["UNIMPORTANT/IMPORTANT"] / unimportant
            if unimportant else None
        ),
        "confusion": dict(sorted(confusion.items())),
    }


def evaluate_importance_predictions(
    *, cohort_path: Path, review_path: Path, predictions_path: Path,
) -> dict[str, object]:
    """Score *all* aligned predictions; no inferred labels and no silent drops."""
    cohort_raw, cohort = _cohort(cohort_path)
    review_summary = score_review(cohort_path, review_path)
    review_raw = _read(review_path)
    reviews = json.loads(review_raw)
    predicted_raw = _read(predictions_path)
    model = json.loads(predicted_raw)
    if not isinstance(model, dict) or set(model) != {
        "schemaVersion", "cohortSha256", "modelId", "promptVersion",
        "generatedAt", "predictions",
    } or model["schemaVersion"] != SCHEMA or (
        model["cohortSha256"] != hashlib.sha256(cohort_raw).hexdigest()
    ):
        raise ValueError("Model prediction document is not pinned to original cohort")
    if not isinstance(model["modelId"], str) or not 1 <= len(
        model["modelId"].strip()
    ) <= 100 or not isinstance(model["promptVersion"], str) or not 1 <= len(
        model["promptVersion"].strip()
    ) <= 100:
        raise ValueError("Explicit model identity and prompt version required")
    generated_at = _instant(model["generatedAt"])
    candidate_rows = model["predictions"]
    if not isinstance(candidate_rows, list) or len(candidate_rows) != len(cohort["items"]):
        raise ValueError("Every original article requires a model output")
    articles = {item["articleId"]: item for item in cohort["items"]}
    labels = {item["articleId"]: item for item in reviews["judgments"]}
    mapped: dict[int, dict[str, float]] = {}
    for entry in candidate_rows:
        if not isinstance(entry, dict) or set(entry) != {
            "articleId", "textSha256", "probabilities",
        }:
            raise ValueError("Prediction must contain only news-importance fields")
        article_id = entry["articleId"]
        if type(article_id) is not int or article_id not in articles or (
            article_id in mapped
        ):
            raise ValueError("Unknown or duplicate prediction article")
        article = articles[article_id]
        if entry["textSha256"] != article["textSha256"]:
            raise ValueError("Prediction text provenance mismatch")
        if generated_at < _instant(article["firstObservedAt"]):
            raise ValueError("Prediction generated before original news was available")
        mapped[article_id] = _probabilities(entry["probabilities"])
    if set(mapped) != set(articles):
        raise ValueError("Predictions exclude original unclassified news")

    graded: list[tuple[str, dict[str, float]]] = []
    baseline: list[tuple[str, dict[str, float]]] = []
    uncertain_review = unreviewed = 0
    review_after_prediction = 0
    for article_id, article in articles.items():
        review = labels[article_id]
        if review["importance"] is None:
            unreviewed += 1
            continue
        if generated_at > _instant(review["reviewedAt"]):
            # A posteriori model runs can still be evaluated on a protected
            # gold set, but we cannot verify they never trained on these labels.
            review_after_prediction += 1
            continue
        if review["importance"] == "UNCERTAIN":
            uncertain_review += 1
            continue
        graded.append((review["importance"], mapped[article_id]))
        baseline.append((review["importance"], {
            "IMPORTANT": 0.0,
            "UNIMPORTANT": 1.0,
            "UNCERTAIN": 0.0,
        }))
    model_metrics = _metrics(graded)
    baseline_metrics = _metrics(baseline)
    sufficient = len(graded) >= MIN_DESCRIPTIVE_REVIEWS
    return {
        "schemaVersion": "news_importance_evaluation_v1",
        "status": (
            "DESCRIPTIVE_ONLY_NOT_VALIDATED" if sufficient
            else "INSUFFICIENT_HUMAN_GOLD"
        ),
        "cohortSha256": hashlib.sha256(cohort_raw).hexdigest(),
        "reviewSha256": hashlib.sha256(review_raw).hexdigest(),
        "predictionsSha256": hashlib.sha256(predicted_raw).hexdigest(),
        "modelId": model["modelId"],
        "promptVersion": model["promptVersion"],
        "articlesInOriginalCohort": len(articles),
        "humanReviewedArticles": review_summary["humanReviewedArticles"],
        "scoredArticles": len(graded),
        "unreviewedArticles": unreviewed,
        "humanUncertainExcludedFromScoring": uncertain_review,
        "predictionsLaterThanReviewExcluded": review_after_prediction,
        "requiredDescriptiveReviews": MIN_DESCRIPTIVE_REVIEWS,
        "model": model_metrics,
        "alwaysUnimportantBaseline": baseline_metrics,
        "newsOnly": True,
        "stockMarketDataUsed": False,
        "humanGoldIndependentlyAudited": False,
        "protectedHoldoutVerified": False,
        "modelPerformanceImprovementValidated": False,
        "trainingDatasetEligible": False,
        "productionDbModified": False,
    }
