"""Blind human review of NEWS-only, within-VALIDATION entity-pair transitions.

Pairs are hypotheses proposed by one model, not verified causal/story links.
This module never evaluates TEST and never promotes a model.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from tris_market_event_intelligence.news_importance_review import _cohort, _instant, _read
from tris_market_event_intelligence.news_materiality_hypothesis import BANDS
from tris_market_event_intelligence.news_materiality_local import (
    _parse_model_hypothesis,
    _story_transitions,
)
from tris_market_event_intelligence.news_materiality_review import (
    PREDICTION_SCHEMA,
    _validation_ids,
)

SCHEMA = "news_materiality_transition_human_review_v1"
RELATIONS = frozenset({"SAME_EVENT", "DIFFERENT_EVENT", "UNCERTAIN"})
CHANGES = frozenset({"INCREASE", "DECREASE", "UNCHANGED", "UNASSESSABLE"})
MIN_DESCRIPTIVE_PAIRS = 20


def _pinned_pairs(*, cohort_path: Path, holdout_path: Path,
                  report_path: Path) -> tuple[dict[str, str], list[dict[str, object]]]:
    raw, cohort = _cohort(cohort_path)
    val = _validation_ids(holdout_path, raw, cohort)
    raw_report = _read(report_path)
    report = json.loads(raw_report)
    if not isinstance(report, dict) or report.get("schemaVersion") != (
        "news_materiality_local_research_v1"
    ) or report.get("status") != "RESEARCH_HYPOTHESES_UNVERIFIED" or (
        report.get("cohortSha256") != hashlib.sha256(raw).hexdigest()
        or report.get("stockPricesUsed") is not False
        or report.get("humanGoldUsed") is not False
        or report.get("promotionAllowed") is not False
    ):
        raise ValueError("Complete source-pinned NEWS research report required")
    result = report.get("result")
    if not isinstance(result, dict) or set(result) != {
        "schemaVersion", "cohortSha256", "modelId", "promptVersion", "hypotheses",
    } or result["schemaVersion"] != PREDICTION_SCHEMA or (
        result["cohortSha256"] != hashlib.sha256(raw).hexdigest()
        or result["modelId"] != report.get("modelId")
        or result["promptVersion"] != report.get("promptVersion")
    ):
        raise ValueError("Pinned materiality predictions required")
    articles = {a["articleId"]: a for a in cohort["items"]}
    hypotheses = result["hypotheses"]
    if not isinstance(hypotheses, list) or len(hypotheses) != len(articles):
        raise ValueError("All original news articles require predictions")
    seen = set()
    parsed = []
    for item in hypotheses:
        if not isinstance(item, dict):
            raise TypeError("Invalid NEWS hypothesis row")
        article_id = item.get("article_id")
        if type(article_id) is not int or article_id not in articles or article_id in seen:
            raise ValueError("Duplicate or unknown NEWS hypothesis")
        seen.add(article_id)
        parsed.append(_parse_model_hypothesis(articles[article_id], json.dumps(item)))
    expected = _story_transitions(list(articles.values()), parsed)
    if report.get("transitions") != expected or report.get(
        "transitionsAreIndependentFacts"
    ) is not False:
        raise ValueError("Unreplayable or falsely certified transition report")
    # Both articles must be in the protected VALIDATION slice; never
    # generate packets for TRAIN or TEST even if the model has predictions.
    pairs = [t for t in expected if (
        t["earlierArticleId"] in val and t["laterArticleId"] in val
    )]
    return {
        "cohortSha256": hashlib.sha256(raw).hexdigest(),
        "holdoutSha256": hashlib.sha256(_read(holdout_path)).hexdigest(),
        "reportSha256": hashlib.sha256(raw_report).hexdigest(),
    }, pairs


def prepare_transition_review(*, cohort_path: Path, holdout_path: Path,
                              report_path: Path) -> dict[str, object]:
    hashes, pairs = _pinned_pairs(
        cohort_path=cohort_path, holdout_path=holdout_path,
        report_path=report_path,
    )
    return {
        "schemaVersion": SCHEMA, **hashes,
        "judgments": [{
            "earlierArticleId": p["earlierArticleId"],
            "laterArticleId": p["laterArticleId"],
            "relation": None, "humanChange": None,
            "reviewer": None, "reviewedAt": None, "reason": None,
        } for p in pairs],
        "modelMagnitudesExposedInTemplate": False,
        "generatedHumanLabels": 0,
        "stockPricesUsed": False,
        "humanGoldIndependentlyAudited": False,
    }


def evaluate_transition_review(*, cohort_path: Path, holdout_path: Path,
                               report_path: Path,
                               human_review_path: Path) -> dict[str, object]:
    hashes, pairs = _pinned_pairs(
        cohort_path=cohort_path, holdout_path=holdout_path,
        report_path=report_path,
    )
    reviewed = json.loads(_read(human_review_path))
    if not isinstance(reviewed, dict) or set(reviewed) != {
        "schemaVersion", "cohortSha256", "holdoutSha256", "reportSha256",
        "judgments", "modelMagnitudesExposedInTemplate",
        "generatedHumanLabels", "stockPricesUsed",
        "humanGoldIndependentlyAudited",
    } or reviewed["schemaVersion"] != SCHEMA or any(
        reviewed.get(k) != v for k, v in hashes.items()
    ) or reviewed["modelMagnitudesExposedInTemplate"] is not False or (
        reviewed["generatedHumanLabels"] != 0
        or reviewed["stockPricesUsed"] is not False
        or reviewed["humanGoldIndependentlyAudited"] is not False
    ):
        raise ValueError("Invalid pinned blind transition human review")
    judges = reviewed["judgments"]
    if not isinstance(judges, list) or len(judges) != len(pairs):
        raise ValueError("Human review must preserve complete selected pair set")
    by_id = {a["articleId"]: a for a in _cohort(cohort_path)[1]["items"]}
    original = {(p["earlierArticleId"], p["laterArticleId"]): p for p in pairs}
    seen = set()
    counts: Counter[str] = Counter()
    scored = agreement = ordinal_errors = 0
    unreviewed = uncertain = unrelated = unassessable = 0
    for row in judges:
        if not isinstance(row, dict) or set(row) != {
            "earlierArticleId", "laterArticleId", "relation", "humanChange",
            "reviewer", "reviewedAt", "reason",
        }:
            raise ValueError("Invalid transition review entry")
        key = row["earlierArticleId"], row["laterArticleId"]
        if key not in original or key in seen:
            raise ValueError("Human review changed or duplicated transition identity")
        seen.add(key)
        fields = [row[k] for k in (
            "relation", "humanChange", "reviewer", "reviewedAt", "reason",
        )]
        if all(value is None for value in fields):
            unreviewed += 1
            continue
        if row["relation"] not in RELATIONS or row["humanChange"] not in CHANGES:
            raise ValueError("Unsupported human relation or impact change")
        if row["relation"] != "SAME_EVENT" and row["humanChange"] != "UNASSESSABLE":
            raise ValueError("Different or uncertain events cannot declare impact change")
        if not isinstance(row["reviewer"], str) or not 1 <= len(
            row["reviewer"].strip()
        ) <= 100 or not isinstance(row["reason"], str) or not 20 <= len(
            row["reason"].strip()
        ) <= 2000 or _instant(row["reviewedAt"]) < _instant(
            by_id[key[1]]["firstObservedAt"]
        ):
            raise ValueError("Complete human review after both news observations required")
        counts[row["relation"]] += 1
        if row["relation"] == "UNCERTAIN":
            uncertain += 1
            continue
        if row["relation"] == "DIFFERENT_EVENT":
            unrelated += 1
            continue
        if row["humanChange"] == "UNASSESSABLE":
            unassessable += 1
            continue
        delta = original[key]["ordinalDifference"]
        if delta is None:
            unassessable += 1
            continue
        expected_change = "INCREASE" if delta > 0 else (
            "DECREASE" if delta < 0 else "UNCHANGED"
        )
        scored += 1
        agreement += expected_change == row["humanChange"]
        expected_sign = {"DECREASE": -1, "UNCHANGED": 0, "INCREASE": 1}
        ordinal_errors += abs(expected_sign[expected_change] -
                              expected_sign[row["humanChange"]])
    return {
        "schemaVersion": "news_materiality_transition_review_audit_v1",
        "status": (
            "DESCRIPTIVE_ONLY_NOT_INDEPENDENT_VALIDATION"
            if scored >= MIN_DESCRIPTIVE_PAIRS
            else "INSUFFICIENT_HUMAN_TRANSITION_GOLD"
        ),
        **hashes,
        "selectedPairsWithinValidation": len(pairs),
        "humanReviewedPairs": len(pairs) - unreviewed,
        "unreviewedPairs": unreviewed,
        "differentEvents": unrelated,
        "uncertainEventIdentity": uncertain,
        "unassessableImpactChange": unassessable,
        "scoredSameEventPairs": scored,
        "sameEventDirectionAgreement": agreement / scored if scored else None,
        "meanDirectionOrdinalError": ordinal_errors / scored if scored else None,
        "humanRelationCounts": dict(sorted(counts.items())),
        "pairsChosenByModelEntityNotIndependentSampling": True,
        "modelMagnitudesExposedInTemplate": False,
        "testGoldScored": False,
        "humanGoldIndependentlyAudited": False,
        "modelImprovementValidated": False,
        "promotionAllowed": False,
        "stockPricesUsed": False,
        "productionDbModified": False,
    }
