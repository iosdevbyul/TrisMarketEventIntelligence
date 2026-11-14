"""Fail-closed chronological news research splits without price data.

Never assume unique articles are independent news stories. Explicit story
groups, identical captured text and exact canonicalized source URL unify
samples. Ambiguous cross-window components are quarantined, not reassigned.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from tris_market_event_intelligence.news_importance_review import (
    _cohort,
    _instant,
    _read,
    score_review,
)

_MAX_PURGE_DAYS = 30


def _link_key(row: dict[str, object]) -> tuple[str, ...]:
    from tris_market_event_intelligence.event_ticker_screen import source_identity

    original_url = row["sourceUrl"]
    canonical = source_identity(original_url)
    if canonical is None:
        raise ValueError("Article source URL cannot be canonicalized")
    return (
        f"url:{canonical}",
        f"text:{row['textSha256']}",
        *([f"story:{row['storyGroupId']}"] if row.get("storyGroupId") is not None else []),
    )


def split_news_chronologically(
    *, cohort_path: Path, review_path: Path,
    train_end: datetime, validation_end: datetime,
    embargo_hours: int = 24,
) -> dict[str, object]:
    """Produce a reproducible *research partition plan* without gold contents.

    This is not a certified protected holdout: provenance of group assignment,
    previously exposed labels and manually annotated story duplicates still
    need independent review.
    """
    if any(
        not isinstance(stamp, datetime)
        or stamp.tzinfo is None or stamp.utcoffset() is None
        for stamp in (train_end, validation_end)
    ):
        raise ValueError("Both split boundaries require explicit timezones")
    train_at = _instant(train_end.isoformat())
    validation_at = _instant(validation_end.isoformat())
    if train_at >= validation_at:
        raise ValueError("Validation boundary must be after training boundary")
    if type(embargo_hours) is not int or not 0 <= embargo_hours <= _MAX_PURGE_DAYS * 24:
        raise ValueError("Embargo must be between 0 and 720 hours")
    delta = timedelta(hours=embargo_hours)
    if train_at + delta >= validation_at - delta:
        raise ValueError("Split windows need room for the requested embargo")

    cohort_raw, cohort = _cohort(cohort_path)
    review_summary = score_review(cohort_path, review_path)
    review_raw = _read(review_path)
    reviews = json.loads(review_raw)
    by_id = {row["articleId"]: row for row in cohort["items"]}
    labeled = {row["articleId"]: row for row in reviews["judgments"]}
    if len(labeled) != len(by_id):
        raise ValueError("Review and cohort article IDs do not align")

    parents = {article_id: article_id for article_id in by_id}

    def find(article_id: int) -> int:
        while parents[article_id] != article_id:
            parents[article_id] = parents[parents[article_id]]
            article_id = parents[article_id]
        return article_id

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)

    index: dict[str, int] = {}
    for article_id, row in by_id.items():
        for key in _link_key(row):
            old = index.setdefault(key, article_id)
            union(old, article_id)
    members: dict[int, list[int]] = {}
    for article_id in by_id:
        members.setdefault(find(article_id), []).append(article_id)

    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    assignments: list[dict[str, object]] = []
    for group_root, ids in sorted(members.items()):
        ids.sort()
        times = [_instant(by_id[article_id]["firstObservedAt"]) for article_id in ids]
        buckets: set[str] = set()
        for observed in times:
            if observed <= train_at - delta:
                buckets.add("TRAIN")
            elif train_at - delta < observed <= train_at + delta:
                buckets.add("PURGED_TRAIN_BOUNDARY")
            elif observed <= validation_at - delta:
                buckets.add("VALIDATION")
            elif observed <= validation_at + delta:
                buckets.add("PURGED_VALIDATION_BOUNDARY")
            else:
                buckets.add("TEST")
        if len(buckets) != 1:
            split = "QUARANTINED"
            reason = "STORY_OR_DUPLICATE_CROSSES_SPLIT"
        else:
            category = next(iter(buckets))
            split = category if category in {"TRAIN", "VALIDATION", "TEST"} else "QUARANTINED"
            reason = (
                "ELIGIBLE_RESEARCH_PARTITION" if split != "QUARANTINED"
                else category
            )
        for article_id in ids:
            row = by_id[article_id]
            review = labeled[article_id]
            if review["importance"] is None:
                gold_status = "UNREVIEWED"
            elif review["importance"] == "UNCERTAIN":
                gold_status = "HUMAN_UNCERTAIN"
            elif split == "TRAIN" and _instant(review["reviewedAt"]) > train_at:
                gold_status = "REVIEW_NOT_AVAILABLE_BY_TRAIN_CUTOFF"
            elif split == "TRAIN":
                gold_status = "TRAIN_LABEL_AVAILABLE_BY_CUTOFF_UNVERIFIED"
            else:
                gold_status = "HELD_OUT_GOLD_DO_NOT_EXPOSE"
            counts[split] += 1
            reasons[reason] += 1
            assignments.append({
                "articleId": article_id,
                "firstObservedAt": row["firstObservedAt"],
                "groupRootArticleId": group_root,
                "groupSize": len(ids),
                "partition": split,
                "reason": reason,
                "labelAvailability": gold_status,
                "newsTextSha256": row["textSha256"],
            })
    assignments.sort(key=lambda entry: entry["articleId"])
    return {
        "schemaVersion": "news_chronological_partition_v1",
        "status": "RESEARCH_PARTITIONS_NOT_CERTIFIED",
        "cohortSha256": hashlib.sha256(cohort_raw).hexdigest(),
        "reviewSha256": hashlib.sha256(review_raw).hexdigest(),
        "trainEndUtc": train_at.isoformat(),
        "validationEndUtc": validation_at.isoformat(),
        "embargoHours": embargo_hours,
        "totalArticles": len(assignments),
        "humanReviewedArticles": review_summary["humanReviewedArticles"],
        "partitionCounts": dict(sorted(counts.items())),
        "exclusionReasons": dict(sorted(reasons.items())),
        "groupCount": len(members),
        "items": assignments,
        "humanGoldExposedInOutput": False,
        "nearDuplicateSemanticsIndependentlyVerified": False,
        "holdoutNeverUsedForTrainingVerified": False,
        "holdoutRepeatedTuningPrevented": False,
        "researchUseOnly": True,
        "marketPricesUsed": False,
        "trainingDatasetEligible": False,
        "productionDbModified": False,
    }
