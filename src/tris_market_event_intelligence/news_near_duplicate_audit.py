"""Review-only cross-split near-duplicate news title candidate audit.

High lexical overlap is only an editorial REVIEW CANDIDATE; never an
automatically verified duplicate, approved split, or human news label.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

from tris_market_event_intelligence.news_importance_review import _cohort, _instant, _read
from tris_market_event_intelligence.story_index import WORD

MAX_CANDIDATES = 10000
MIN_SHARED_TOKENS = 5
MAX_AGE_DAYS = 30


def _tokens(title: str) -> frozenset[str]:
    if not isinstance(title, str):
        raise TypeError("News title must be text")
    tokens = WORD.findall(unicodedata.normalize("NFKC", title).casefold())
    return frozenset(token for token in tokens if len(token) >= 2)


def audit_news_near_duplicates(
    *, cohort_path: Path, holdout_path: Path,
    threshold: float = 0.78, max_days: int = MAX_AGE_DAYS,
) -> dict[str, object]:
    """Find potential *cross-partition* lexical repeats without editing splits.

    Existing holdout must be pinned to precisely this cohort. Check every
    cross-partition title pair in a bounded page. No market-data analysis.
    """
    if type(threshold) is not float or not 0.70 <= threshold <= 0.95:
        raise ValueError("Threshold must be a float between 0.70 and 0.95")
    if type(max_days) is not int or not 1 <= max_days <= MAX_AGE_DAYS:
        raise ValueError("Lookback days must be 1..30")
    cohort_raw, cohort = _cohort(cohort_path)
    holdout_raw = _read(holdout_path)
    holdout = json.loads(holdout_raw)
    if not isinstance(holdout, dict) or holdout.get("schemaVersion") != (
        "news_holdout_plan_v1"
    ) or holdout.get("status") != "PLANNED_NOT_AUDITED" or (
        holdout.get("cohortSha256") != hashlib.sha256(cohort_raw).hexdigest()
        or holdout.get("trainingDatasetEligible") is not False
        or holdout.get("stockPricesUsed") is not False
    ):
        raise ValueError("News holdout plan does not match original cohort")
    holdout_items = holdout.get("items")
    if not isinstance(holdout_items, list) or len(holdout_items) != len(cohort["items"]):
        raise ValueError("Holdout plan does not cover all articles")
    by_id = {row["articleId"]: row for row in cohort["items"]}
    partitions: dict[int, str] = {}
    counts: Counter[str] = Counter()
    for row in holdout_items:
        if not isinstance(row, dict):
            raise TypeError("Invalid holdout partition row")
        article_id = row.get("articleId")
        if type(article_id) is not int or article_id not in by_id or article_id in partitions:
            raise ValueError("Invalid or duplicate article in holdout")
        original = by_id[article_id]
        if row.get("textSha256") != original["textSha256"] or (
            _instant(row.get("firstObservedAt")) != _instant(original["firstObservedAt"])
        ):
            raise ValueError("News holdout provenance does not match article")
        category = row.get("split")
        if category not in {"TRAIN", "VALIDATION", "TEST", "EMBARGO", "QUARANTINED"}:
            raise ValueError("Invalid news partition")
        partitions[article_id] = category
        counts[category] += 1
    if dict(counts) != holdout.get("counts"):
        raise ValueError("Holdout partition counts changed")
    if set(partitions) != set(by_id):
        raise ValueError("Holdout partition missing an article")

    # Inverted token index avoids all-pairs scans for unrelated titles.
    postings: dict[str, set[int]] = defaultdict(set)
    representations: dict[int, frozenset[str]] = {}
    times = {}
    candidates: list[dict[str, object]] = []
    eligible = sorted(
        cohort["items"], key=lambda row: (_instant(row["firstObservedAt"]), row["articleId"])
    )
    for article in eligible:
        article_id = article["articleId"]
        words = _tokens(article["title"])
        instant = _instant(article["firstObservedAt"])
        if len(words) < MIN_SHARED_TOKENS or len("".join(words)) < 24:
            continue
        possibilities: Counter[int] = Counter()
        for word in words:
            possibilities.update(postings[word])
        for other, shared in possibilities.items():
            if shared < MIN_SHARED_TOKENS or partitions[other] == partitions[article_id]:
                continue
            if instant - times[other] > timedelta(days=max_days):
                continue
            previous = representations[other]
            ratio = shared / len(words | previous)
            if ratio < threshold:
                continue
            # True duplicate groups are already quarantined by exact keys;
            # this tool flags *additional* editorial review only.
            candidates.append({
                "earlierArticleId": other,
                "laterArticleId": article_id,
                "earlierPartition": partitions[other],
                "laterPartition": partitions[article_id],
                "earlierTextSha256": by_id[other]["textSha256"],
                "laterTextSha256": article["textSha256"],
                "titleTokenJaccard": round(ratio, 6),
                "reviewStatus": "HUMAN_REVIEW_REQUIRED",
                "confirmedSameStory": None,
            })
            if len(candidates) > MAX_CANDIDATES:
                raise ValueError("Candidate overflow; narrow cohort and repeat audit")
        representations[article_id] = words
        times[article_id] = instant
        for word in words:
            postings[word].add(article_id)
    candidates.sort(key=lambda row: (
        row["earlierArticleId"], row["laterArticleId"],
    ))
    return {
        "schemaVersion": "news_holdout_near_duplicate_audit_v1",
        "status": (
            "REVIEW_REQUIRED" if candidates else "NO_LEXICAL_CANDIDATES_NOT_VERIFIED"
        ),
        "cohortSha256": hashlib.sha256(cohort_raw).hexdigest(),
        "holdoutSha256": hashlib.sha256(holdout_raw).hexdigest(),
        "articlesChecked": len(by_id),
        "threshold": threshold,
        "maxAgeDays": max_days,
        "crossPartitionCandidates": len(candidates),
        "items": candidates,
        "candidateMatchesAreConfirmedDuplicates": False,
        "semanticOrTranslatedNearDuplicatesFullyDetected": False,
        "crossPageNearDuplicatesAudited": False,
        "trainingDatasetEligible": False,
        "humanGoldExposed": False,
        "stockPricesUsed": False,
        "productionDbModified": False,
    }
