"""Fail-closed temporal and duplicate-group splits for news-only experiments.

This does not split or score stock-price data. Unknown cross-publication
relationships remain an explicit source of uncertainty and are never certified.
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
from tris_market_event_intelligence.story_index import canonical_url


class _Groups:
    def __init__(self, ids: list[int]) -> None:
        self.parents = {value: value for value in ids}

    def find(self, value: int) -> int:
        parent = self.parents[value]
        if parent != value:
            self.parents[value] = self.find(parent)
        return self.parents[value]

    def join(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        self.parents[max(a, b)] = min(a, b)


def plan_news_holdout(
    cohort_path: Path, *, train_end: datetime, validation_end: datetime,
    embargo_hours: int = 24, review_path: Path | None = None,
) -> dict[str, object]:
    """Partition a full news cohort without exposing label content.

    Duplicate / story-shared groups that straddle windows are quarantined.
    The final TEST split is reserved: no evaluation loops may promote models
    using TEST; group leakage auditing is conservative, not independently
    verified near-duplicate detection.
    """
    train = _instant(train_end.isoformat())
    validation = _instant(validation_end.isoformat())
    if type(embargo_hours) is not int or not 1 <= embargo_hours <= 24 * 30:
        raise ValueError("Embargo must be 1..720 whole hours")
    embargo = timedelta(hours=embargo_hours)
    if not train + embargo < validation:
        raise ValueError("Validation boundary must follow training and embargo")
    cohort_raw, cohort = _cohort(cohort_path)
    review_labels: dict[int, dict[str, object]] | None = None
    review_sha: str | None = None
    if review_path is not None:
        # Reuse the full Learning-02 gold/provenance validation; do not
        # assume file contents are authenticated human truth.
        score_review(cohort_path, review_path)
        review_raw = _read(review_path)
        judgments = json.loads(review_raw)["judgments"]
        review_labels = {row["articleId"]: row for row in judgments}
        review_sha = hashlib.sha256(review_raw).hexdigest()
    rows = cohort["items"]
    ids = [item["articleId"] for item in rows]
    links = _Groups(ids)
    identities: dict[tuple[str, str], int] = {}
    for item in rows:
        article_id = item["articleId"]
        story = item.get("storyGroupId")
        if story is not None:
            if type(story) is not int or story < 1:
                raise ValueError("Invalid story group")
            keys = [("story", str(story))]
        else:
            keys = []
        identity = canonical_url(item["sourceUrl"])
        keys.extend((("url", identity), ("text", item["textSha256"])))
        for key in keys:
            if key in identities:
                links.join(article_id, identities[key])
            else:
                identities[key] = article_id

    assigned: dict[int, str] = {}
    for item in rows:
        instant = _instant(item["firstObservedAt"])
        article_id = item["articleId"]
        if instant <= train:
            assigned[article_id] = "TRAIN"
        elif instant <= train + embargo:
            assigned[article_id] = "EMBARGO"
        elif instant <= validation:
            assigned[article_id] = "VALIDATION"
        elif instant <= validation + embargo:
            assigned[article_id] = "EMBARGO"
        else:
            assigned[article_id] = "TEST"

    by_group: dict[int, set[int]] = {}
    for article_id in ids:
        by_group.setdefault(links.find(article_id), set()).add(article_id)
    quarantine: dict[int, str] = {}
    for group_ids in by_group.values():
        windows = {assigned[article_id] for article_id in group_ids}
        if len(windows) > 1:
            reason = (
                "DUPLICATE_GROUP_CROSSES_SPLITS"
                if len(windows - {"EMBARGO"}) > 1
                else "DUPLICATE_GROUP_TOUCHES_EMBARGO"
            )
            quarantine.update(dict.fromkeys(group_ids, reason))
    items = []
    counts: Counter[str] = Counter()
    for item in rows:
        article_id = item["articleId"]
        status = (
            "QUARANTINED" if article_id in quarantine
            else assigned[article_id]
        )
        counts[status] += 1
        # Label visibility is computed only for the TRAIN split; later
        # reviews cannot be retroactively supplied to training. Validation
        # and TEST never expose human labels or reasons in this plan.
        eligibility = None
        if review_labels is not None:
            human = review_labels[article_id]
            if status != "TRAIN":
                eligibility = "NOT_TRAIN_LABEL"
            elif human["importance"] is None:
                eligibility = "UNREVIEWED"
            elif human["importance"] == "UNCERTAIN":
                eligibility = "HUMAN_UNCERTAIN"
            elif _instant(human["reviewedAt"]) > train:
                eligibility = "REVIEWED_AFTER_TRAIN_END"
            else:
                eligibility = "LABEL_AVAILABLE_BY_TRAIN_END_NOT_AUDITED"
        entry = {
            "articleId": article_id,
            "textSha256": item["textSha256"],
            "firstObservedAt": item["firstObservedAt"],
            "split": status,
            "exclusionReason": quarantine.get(article_id)
            if status == "QUARANTINED" else (
                "TEMPORAL_EMBARGO" if status == "EMBARGO" else None
            ),
        }
        if eligibility is not None:
            entry["trainGoldAvailability"] = eligibility
        items.append(entry)
    return {
        "schemaVersion": "news_holdout_plan_v1",
        "status": "PLANNED_NOT_AUDITED",
        "cohortSha256": hashlib.sha256(cohort_raw).hexdigest(),
        "trainEnd": train.isoformat(),
        "validationEnd": validation.isoformat(),
        "embargoHours": embargo_hours,
        "counts": dict(sorted(counts.items())),
        "groupsConsidered": len(by_group),
        "quarantinedArticles": len(quarantine),
        "items": items,
        "humanReviewSha256": review_sha,
        "humanReviewTimeGateApplied": review_labels is not None,
        "humanGoldValuesExposed": False,
        "trainingLabelsIndependentlyAudited": False,
        "testSetReservedForFinalIndependentEvaluation": True,
        "crossSplitExactIdentitiesQuarantined": True,
        "nearDuplicateDetectionComplete": False,
        "humanGoldIndependentlyVerified": False,
        "modelTrained": False,
        "modelPerformanceVerified": False,
        "trainingDatasetEligible": False,
        "stockPricesUsed": False,
        "sourceDbModified": False,
    }
