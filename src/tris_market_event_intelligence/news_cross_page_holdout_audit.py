"""Bounded cross-page news leakage audit, without automatic story merging.

This checks complete, contiguous all-news cohort pages against existing
time-split plans and emits human-review candidates for cross-page repeats.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

from tris_market_event_intelligence.news_importance_review import _cohort, _instant, _read
from tris_market_event_intelligence.news_near_duplicate_audit import (
    MAX_CANDIDATES,
    MIN_SHARED_TOKENS,
    _tokens,
)
from tris_market_event_intelligence.story_index import canonical_url

MAX_PAGES = 16
MAX_NEWS = 20000
PARTITIONS = {"TRAIN", "VALIDATION", "TEST", "EMBARGO", "QUARANTINED"}


def audit_cross_page_news(
    *, pages: list[tuple[Path, Path]], threshold: float = 0.78, max_days: int = 30,
) -> dict[str, object]:
    """Validate consecutive pages and discover cross-page candidate leakage.

    Each page must have its own original pinned holdout plan using the same
    train/validation/embargo policy. This audit does not rewrite either plan.
    """
    if not isinstance(pages, list) or not 2 <= len(pages) <= MAX_PAGES:
        raise ValueError("Provide 2..16 cohort/holdout pairs")
    if type(threshold) is not float or not 0.70 <= threshold <= 0.95:
        raise ValueError("Invalid lexical threshold")
    if type(max_days) is not int or not 1 <= max_days <= 30:
        raise ValueError("Invalid observation window")
    records: list[dict[str, object]] = []
    page_fingerprints: list[dict[str, str]] = []
    partitions: dict[int, str] = {}
    policy = None
    source = cutoff = None
    cursor = 0
    all_ids: set[int] = set()
    for page_index, (cohort_path, plan_path) in enumerate(pages):
        cohort_raw, cohort = _cohort(cohort_path)
        plan_raw = _read(plan_path)
        plan = json.loads(plan_raw)
        if not isinstance(plan, dict) or plan.get("schemaVersion") != "news_holdout_plan_v1" or (
            plan.get("status") != "PLANNED_NOT_AUDITED"
            or plan.get("cohortSha256") != hashlib.sha256(cohort_raw).hexdigest()
            or plan.get("trainingDatasetEligible") is not False
            or plan.get("stockPricesUsed") is not False
        ):
            raise ValueError("Unpinned news-only page holdout")
        current_policy = (
            plan.get("trainEnd"), plan.get("validationEnd"), plan.get("embargoHours")
        )
        if policy is None:
            policy = current_policy
        elif current_policy != policy:
            raise ValueError("Inconsistent split boundaries between pages")
        if cohort.get("pageAfterId") != cursor or (
            page_index < len(pages) - 1 and cohort.get("hasMore") is not True
        ) or (page_index == len(pages) - 1 and cohort.get("hasMore") is not False):
            raise ValueError("Incomplete or nonconsecutive original news pagination")
        if source is None:
            source, cutoff = cohort.get("sourceDb"), cohort.get("asOf")
        elif (cohort.get("sourceDb"), cohort.get("asOf")) != (source, cutoff):
            raise ValueError("Pages refer to different news archives or cutoffs")
        rows = cohort["items"]
        if not rows or len(rows) > cohort.get("pageLimit", 0) or (
            len(rows) + len(records) > MAX_NEWS
        ):
            raise ValueError("Invalid or oversized original news pages")
        ids = [row["articleId"] for row in rows]
        if ids != sorted(ids) or ids[0] <= cursor or any(
            article_id in all_ids for article_id in ids
        ) or cohort.get("nextCursor") != ids[-1]:
            raise ValueError("Invalid, repeated, or unordered page cursor")
        all_ids.update(ids)
        cursor = ids[-1]
        by_id = {row["articleId"]: row for row in rows}
        planned = plan.get("items")
        if not isinstance(planned, list) or len(planned) != len(rows):
            raise ValueError("Holdout omits archived news")
        counts: Counter[str] = Counter()
        seen: set[int] = set()
        for item in planned:
            if not isinstance(item, dict):
                raise TypeError("Invalid holdout entry")
            article_id = item.get("articleId")
            if type(article_id) is not int or article_id not in by_id or article_id in seen:
                raise ValueError("Holdout article ID invalid or repeated")
            seen.add(article_id)
            original = by_id[article_id]
            if item.get("textSha256") != original["textSha256"] or (
                _instant(item.get("firstObservedAt"))
                != _instant(original["firstObservedAt"])
            ):
                raise ValueError("Holdout original article provenance mismatch")
            split = item.get("split")
            if split not in PARTITIONS:
                raise ValueError("Unrecognized holdout split")
            partitions[article_id] = split
            counts[split] += 1
        if dict(counts) != plan.get("counts"):
            raise ValueError("Holdout split counts do not match article list")
        records.extend({**row, "_page": page_index} for row in rows)
        page_fingerprints.append({
            "cohortSha256": hashlib.sha256(cohort_raw).hexdigest(),
            "holdoutSha256": hashlib.sha256(plan_raw).hexdigest(),
        })
    postings: dict[str, set[int]] = defaultdict(set)
    exact: dict[str, set[int]] = defaultdict(set)
    representations: dict[int, frozenset[str]] = {}
    timelines = {}
    lookup = {row["articleId"]: row for row in records}
    candidates: list[dict[str, object]] = []
    for record in sorted(records, key=lambda r: (_instant(r["firstObservedAt"]), r["articleId"])):
        article_id = record["articleId"]
        seen_at = _instant(record["firstObservedAt"])
        words = _tokens(record["title"])
        keys = (
            "url:" + canonical_url(record["sourceUrl"]),
            "sha:" + record["textSha256"],
        )
        if record.get("storyGroupId") is not None:
            keys += ("story:" + str(record["storyGroupId"]),)
        possible: set[int] = set()
        for key in keys:
            possible.update(exact[key])
        lexical: Counter[int] = Counter()
        if len(words) >= MIN_SHARED_TOKENS and len("".join(words)) >= 24:
            for word in words:
                lexical.update(postings[word])
            possible.update(
                prior for prior, shared in lexical.items()
                if shared >= MIN_SHARED_TOKENS
            )
        for prior in sorted(possible):
            before = lookup[prior]
            if before["_page"] == record["_page"] or (
                partitions[prior] == partitions[article_id]
            ):
                continue
            matched = any(prior in exact[key] for key in keys)
            delta = seen_at - timelines[prior]
            ratio = (
                len(words & representations[prior]) / len(words | representations[prior])
                if words | representations[prior] else 0.0
            )
            if not matched and (delta > timedelta(days=max_days) or ratio < threshold):
                continue
            candidates.append({
                "earlierArticleId": prior,
                "laterArticleId": article_id,
                "earlierPageIndex": before["_page"],
                "laterPageIndex": record["_page"],
                "earlierPartition": partitions[prior],
                "laterPartition": partitions[article_id],
                "relation": "EXACT_KEY" if matched else "LEXICAL_CANDIDATE",
                "titleTokenJaccard": round(ratio, 6),
                "reviewStatus": "HUMAN_REVIEW_REQUIRED",
                "confirmedSameStory": None,
            })
            if len(candidates) > MAX_CANDIDATES:
                raise ValueError("Too many cross-page candidates; review in smaller batches")
        for key in keys:
            exact[key].add(article_id)
        for word in words:
            postings[word].add(article_id)
        representations[article_id] = words
        timelines[article_id] = seen_at
    return {
        "schemaVersion": "news_cross_page_holdout_audit_v1",
        "status": "REVIEW_REQUIRED" if candidates else "NO_CANDIDATES_NOT_VERIFIED",
        "pageCount": len(pages),
        "articlesChecked": len(records),
        "pageFingerprints": page_fingerprints,
        "crossPageCandidates": len(candidates),
        "items": sorted(candidates, key=lambda r: (r["earlierArticleId"], r["laterArticleId"])),
        "continuousPaginationChecked": True,
        "candidateMatchesAreConfirmedDuplicates": False,
        "independentSemanticDuplicateVerification": False,
        "translationAndCorrectionLineageVerified": False,
        "holdoutCanBeCertified": False,
        "humanGoldExposed": False,
        "stockPricesUsed": False,
        "trainingDatasetEligible": False,
        "productionDbModified": False,
    }
