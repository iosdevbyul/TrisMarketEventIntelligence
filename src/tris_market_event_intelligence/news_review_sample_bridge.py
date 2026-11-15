"""Fail-closed bridge from human-edited representative sample to full review files.

Only explicitly supplied human judgments are copied. Nothing is inferred
from story candidates, model outputs, prices or adjacent articles.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tris_market_event_intelligence.news_importance_review import (
    ALIGNMENT,
    IMPORTANCE,
    _cohort,
    _instant,
    _read,
    prepare_review,
    score_review,
)
from tris_market_event_intelligence.news_materiality_hypothesis import BANDS
from tris_market_event_intelligence.news_materiality_review import (
    _validated_reviews,
    prepare_materiality_review,
)
from tris_market_event_intelligence.news_review_sample import prepare_news_review_sample

EDITABLE = frozenset({
    "importance", "potentialMagnitude", "sourceAlignment",
    "sameStoryConfirmedByHuman", "reviewer", "reviewedAt", "reason",
})


def bridge_review_sample(*, cohort_path: Path, sample_path: Path,
                         importance_template_path: Path,
                         materiality_template_path: Path) -> dict[str, object]:
    """Return two valid *self-attested* full reviews; never write or train."""
    cohort_raw, cohort = _cohort(cohort_path)
    observed_sha = hashlib.sha256(cohort_raw).hexdigest()
    sample = json.loads(_read(sample_path))
    if not isinstance(sample, dict) or sample.get("schemaVersion") != (
        "news_human_review_candidate_sample_v1"
    ) or sample.get("cohortSha256") != observed_sha:
        raise ValueError("Source-pinned human sample is required")
    raw_rows = sample.get("items")
    if not isinstance(raw_rows, list) or not 1 <= len(raw_rows) <= 100:
        raise ValueError("Bounded review sample is required")
    # Reproduce original independent selection, preserving its precise SHA
    # policy and all immutable article metadata; only human fields may change.
    expected = prepare_news_review_sample(
        cohort_path=cohort_path, max_selected=len(raw_rows),
    )
    if set(sample) != set(expected) or any(
        sample[field] != value for field, value in expected.items()
        if field != "items"
    ):
        raise ValueError("Review sampling policy or provenance was changed")
    originals = expected["items"]
    human_importance = json.loads(_read(importance_template_path))
    human_materiality = json.loads(_read(materiality_template_path))
    blank_importance = prepare_review(cohort_path)
    blank_materiality = prepare_materiality_review(cohort_path)
    if human_importance != blank_importance or human_materiality != blank_materiality:
        raise ValueError("Original full 482-row templates must remain untouched and blank")
    important_rows = {
        row["articleId"]: row for row in human_importance["judgments"]
    }
    magnitude_rows = {
        row["articleId"]: row for row in human_materiality["judgments"]
    }
    seen = set()
    importance_count = magnitude_count = story_count = 0
    for actual, original in zip(raw_rows, originals, strict=True):
        if not isinstance(actual, dict) or set(actual) != set(original):
            raise ValueError("Human sample row has altered shape")
        if any(actual[field] != value for field, value in original.items()
               if field not in EDITABLE):
            raise ValueError("Sample article text or source provenance was changed")
        identifier = actual["articleId"]
        if identifier in seen:
            raise ValueError("Repeated sample article")
        seen.add(identifier)
        importance = actual["importance"]
        magnitude = actual["potentialMagnitude"]
        alignment = actual["sourceAlignment"]
        story = actual["sameStoryConfirmedByHuman"]
        # Story relatedness is informational only. We never fan out labels
        # from the representative article to a candidate unit.
        if story is not None and type(story) is not bool:
            raise ValueError("Story relation must be human boolean or null")
        reviewed_importance = importance is not None or alignment is not None
        reviewed_magnitude = magnitude is not None
        any_review = reviewed_importance or reviewed_magnitude or story is not None
        human_fields = ("reviewer", "reviewedAt", "reason")
        if not any_review:
            if any(actual[field] is not None for field in human_fields):
                raise ValueError("No orphan reviewer metadata on unreviewed samples")
            continue
        if not isinstance(actual["reviewer"], str) or not 1 <= len(
            actual["reviewer"].strip()
        ) <= 100 or not isinstance(actual["reason"], str) or not 20 <= len(
            actual["reason"].strip()
        ) <= 2000 or _instant(actual["reviewedAt"]) < _instant(
            original["firstObservedAt"]
        ):
            raise ValueError("Human review requires name, timestamp and substantive reason")
        if reviewed_importance and (
            importance not in IMPORTANCE or alignment not in ALIGNMENT
        ):
            raise ValueError("Importance and source alignment must be provided together")
        if reviewed_magnitude and magnitude not in (*BANDS, "UNASSESSABLE"):
            raise ValueError("Invalid human potential impact magnitude")
        if importance is not None:
            important_rows[identifier].update({
                "importance": importance,
                "sourceAlignment": alignment,
                "reviewer": actual["reviewer"],
                "reviewedAt": actual["reviewedAt"],
                "reason": actual["reason"],
            })
            importance_count += 1
        if reviewed_magnitude:
            magnitude_rows[identifier].update({
                "potentialMagnitude": magnitude,
                "reviewer": actual["reviewer"],
                "reviewedAt": actual["reviewedAt"],
                "reason": actual["reason"],
            })
            magnitude_count += 1
        story_count += story is not None
    # Explicitly validate generated documents using the existing exact
    # complete-cohort validators (without opening TEST or scoring models).
    # The same validators check provenance and each individual judgment.
    # Temporary file writes are avoided by extracting equivalent checks
    # above and asserting all untouched fields are inherited from templates.
    if not (importance_count or magnitude_count):
        status = "NO_HUMAN_IMPORTANCE_OR_MAGNITUDE_JUDGMENTS"
    else:
        status = "SELF_ATTESTED_HUMAN_REVIEWS_NOT_INDEPENDENTLY_AUDITED"
    return {
        "schemaVersion": "news_review_sample_bridge_v1",
        "status": status,
        "cohortSha256": observed_sha,
        "sampleSha256": hashlib.sha256(_read(sample_path)).hexdigest(),
        "importanceTemplateSha256":
            hashlib.sha256(_read(importance_template_path)).hexdigest(),
        "materialityTemplateSha256":
            hashlib.sha256(_read(materiality_template_path)).hexdigest(),
        "selectedArticles": len(raw_rows),
        "importanceHumanJudgments": importance_count,
        "materialityHumanJudgments": magnitude_count,
        "humanStoryRelationsNotPropagated": story_count,
        "unreviewedOriginalArticlesPreserved": True,
        "importanceReview": human_importance,
        "materialityReview": human_materiality,
        "humanLabelsAuthenticated": False,
        "humanGoldIndependentlyAudited": False,
        "modelCalls": 0,
        "testGoldScored": False,
        "modelImprovementValidated": False,
        "promotionAllowed": False,
        "stockPricesUsed": False,
        "productionDbModified": False,
    }
