"""News-only, human-adjudicated article importance and source-context review.

All labels must be supplied by a person. They are self-attested, not verified
facts about a market and never training labels until independently audited.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "news_human_importance_review_v1"
IMPORTANCE = frozenset({"IMPORTANT", "UNIMPORTANT", "UNCERTAIN"})
ALIGNMENT = frozenset({"SUPPORTED", "CONTRADICTED", "UNRELATED", "INSUFFICIENT_CONTEXT"})
MAX_BYTES = 8_000_000


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("Existing bounded non-symlink review input required")
    data = path.read_bytes()
    if len(data) > MAX_BYTES:
        raise ValueError("Review input grew beyond limit")
    return data


def _instant(value: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError("Review timestamp must be a string")
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Invalid review timestamp") from exc
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Timezone-aware review timestamp required")
    return dt.astimezone(UTC)


def _cohort(path: Path) -> tuple[bytes, dict[str, object]]:
    raw = _read(path)
    doc = json.loads(raw)
    if not isinstance(doc, dict) or doc.get("schemaVersion") not in {
        "historical_news_cohort_v1", "historical_news_cohort_v2",
    } or doc.get("allNewsIncludedRegardlessOfExtraction") is not True or (
        doc.get("trainingDatasetEligible") is not False
        or doc.get("sourceDbModified") is not False
    ):
        raise ValueError("Expected read-only all-news cohort")
    cutoff = _instant(doc["asOf"])
    entries = doc.get("items")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 5000:
        raise ValueError("Human review needs 1..5000 historical articles")
    seen: set[int] = set()
    for row in entries:
        if not isinstance(row, dict):
            raise TypeError("Invalid archived article row")
        article_id = row.get("articleId")
        if type(article_id) is not int or article_id < 1 or article_id in seen:
            raise ValueError("Invalid or duplicate article ID")
        seen.add(article_id)
        if (
            not isinstance(row.get("textSha256"), str)
            or len(row["textSha256"]) != 64
            or any(x not in "0123456789abcdef" for x in row["textSha256"])
            or not isinstance(row.get("captureSha256"), str)
            or len(row["captureSha256"]) != 64
            or any(x not in "0123456789abcdef" for x in row["captureSha256"])
            or not isinstance(row.get("sourceUrl"), str)
            or not row["sourceUrl"].startswith(("https://", "http://"))
        ):
            raise ValueError("Archived article lacks pinned provenance")
        observed = _instant(row["firstObservedAt"])
        if observed > cutoff:
            raise ValueError("Article first observed after cohort cutoff")
        text = (row["title"] + "\n" + (row.get("summary") or "")).encode()
        if hashlib.sha256(text).hexdigest() != row["textSha256"]:
            raise ValueError("Archived article text hash mismatch")
        if row.get("importanceHumanGold") != "UNKNOWN" or (
            row.get("trainingEligible") is not False
        ):
            raise ValueError("Cohort must not contain prefilled training labels")
    return raw, doc


def prepare_review(cohort_path: Path) -> dict[str, object]:
    """Make blank, pin-checked review queue: no inferred importance."""
    raw, cohort = _cohort(cohort_path)
    return {
        "schemaVersion": SCHEMA,
        "cohortSha256": hashlib.sha256(raw).hexdigest(),
        "judgments": [{
            "articleId": row["articleId"],
            "source": row["source"],
            "sourceUrl": row["sourceUrl"],
            "textSha256": row["textSha256"],
            "captureSha256": row["captureSha256"],
            "firstObservedAt": row["firstObservedAt"],
            "storyGroupId": row.get("storyGroupId"),
            "importance": None,
            "sourceAlignment": None,
            "reviewer": None,
            "reviewedAt": None,
            "reason": None,
        } for row in cohort["items"]],
        "generatedLabels": 0,
        "reviewerAuthenticationVerified": False,
        "productionDbModified": False,
    }


def score_review(
    cohort_path: Path, reviews_path: Path,
) -> dict[str, object]:
    """Summarize independently entered news-only human judgments.

    This does NOT calculate model accuracy, market importance from prices,
    forecast confidence, or independent source authenticity.
    """
    cohort_raw, cohort = _cohort(cohort_path)
    doc = json.loads(_read(reviews_path))
    if not isinstance(doc, dict) or doc.get("schemaVersion") != SCHEMA or (
        doc.get("cohortSha256") != hashlib.sha256(cohort_raw).hexdigest()
    ):
        raise ValueError("Review document does not match immutable cohort")
    judgments = doc.get("judgments")
    if not isinstance(judgments, list) or len(judgments) != len(cohort["items"]):
        raise ValueError("Review must preserve every article including unimportant ones")
    by_article = {row["articleId"]: row for row in cohort["items"]}
    reviewed: Counter[str] = Counter()
    alignment: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = {}
    group_ids: set[int] = set()
    independent_groups: set[str] = set()
    missing = 0
    seen: set[int] = set()
    for row in judgments:
        if not isinstance(row, dict) or set(row) != {
            "articleId", "source", "sourceUrl", "textSha256", "captureSha256",
            "firstObservedAt", "storyGroupId", "importance", "sourceAlignment",
            "reviewer", "reviewedAt", "reason",
        }:
            raise ValueError("Review row has unexpected or missing fields")
        article_id = row["articleId"]
        if type(article_id) is not int or article_id not in by_article or article_id in seen:
            raise ValueError("Review article ID missing or duplicated")
        seen.add(article_id)
        original = by_article[article_id]
        pinned = ("source", "sourceUrl", "textSha256", "captureSha256",
                  "firstObservedAt", "storyGroupId")
        if any(row[key] != original.get(key) for key in pinned):
            raise ValueError("Review differs from archived article provenance")
        labels = ("importance", "sourceAlignment", "reviewer", "reviewedAt", "reason")
        if all(row[key] is None for key in labels):
            missing += 1
            continue
        if row["importance"] not in IMPORTANCE or row["sourceAlignment"] not in ALIGNMENT:
            raise ValueError("Complete human importance and source alignment required")
        if not isinstance(row["reviewer"], str) or not 1 <= len(
            row["reviewer"].strip()
        ) <= 100:
            raise ValueError("Nonempty human reviewer required")
        if not isinstance(row["reason"], str) or not 20 <= len(
            row["reason"].strip()
        ) <= 2000:
            raise ValueError("Substantive human reasoning required")
        if _instant(row["reviewedAt"]) < _instant(row["firstObservedAt"]):
            raise ValueError("Review cannot precede first observed source")
        reviewed[row["importance"]] += 1
        alignment[row["sourceAlignment"]] += 1
        by_source.setdefault(original["source"], Counter())[row["importance"]] += 1
        group = original.get("storyGroupId")
        if group is not None:
            if type(group) is not int or group < 1:
                raise ValueError("Invalid story grouping")
            group_ids.add(group)
            independent_groups.add(f"group:{group}")
        else:
            independent_groups.add(f"article:{article_id}")
    if len(seen) != len(by_article):
        raise ValueError("Missing original article in review")
    count = sum(reviewed.values())
    return {
        "schemaVersion": "news_human_importance_summary_v1",
        "status": "HUMAN_REVIEWS_PRESENT_UNVERIFIED" if count else "NO_HUMAN_REVIEWS",
        "cohortSha256": hashlib.sha256(cohort_raw).hexdigest(),
        "totalArticles": len(by_article),
        "humanReviewedArticles": count,
        "unreviewedArticles": missing,
        "importanceCounts": dict(sorted(reviewed.items())),
        "sourceAlignmentCounts": dict(sorted(alignment.items())),
        "importanceBySource": {
            source: dict(sorted(totals.items()))
            for source, totals in sorted(by_source.items())
        },
        "reviewedStoryGroupCount": len(group_ids),
        "potentialIndependentStoryUnits": len(independent_groups),
        "allNewsRetained": True,
        "reviewerAuthenticationVerified": False,
        "sourceFactsIndependentlyVerified": False,
        "humanLabelsIndependentlyAudited": False,
        "modelAccuracyVerified": False,
        "trainingDatasetEligible": False,
        "stockPricesUsed": False,
        "sourceDbModified": False,
    }
