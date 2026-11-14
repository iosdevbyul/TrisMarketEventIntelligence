"""Read-only historical article cohort export, including all unclassified news.

Extraction outcomes are not labels of market impact or importance. A missing
event does not mean that an article was unimportant. Reported news chronology
and source timestamps are not independent proofs of publication time.
"""
from __future__ import annotations

import hashlib
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path


def _timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError("Timestamp is not a string")
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Invalid article chronology") from exc
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Timezone-aware article chronology required")
    return dt.astimezone(UTC)


def _table_exists(db: sqlite3.Connection, table: str) -> bool:
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,),
    ).fetchone() is not None


def historical_news_cohort(
    *, source_db: Path, as_of: datetime, after_id: int = 0, limit: int = 1000,
) -> dict[str, object]:
    """Inventory and page every observed article, not just selected events.

    All article rows are read; the caller's cutoff is applied to first observed
    receipt in UTC. Event/claim processing counts are descriptive metadata and
    never inserted into contemporaneous training features.
    """
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("Timezone-aware as_of required")
    if type(after_id) is not int or after_id < 0:
        raise ValueError("Nonnegative article cursor required")
    if type(limit) is not int or not 1 <= limit <= 5000:
        raise ValueError("Article page limit must be 1..5000")
    if source_db.is_symlink() or not source_db.is_file():
        raise ValueError("Existing non-symlink production source DB required")
    cutoff = as_of.astimezone(UTC)
    uri = source_db.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=10)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        if not _table_exists(db, "articles") or not _table_exists(db, "captures"):
            raise ValueError("Initialized original capture and article tables required")
        has_story = _table_exists(db, "story_members")
        has_claims = _table_exists(db, "article_event_claims")
        has_events = _table_exists(db, "economic_events")
        fetched = db.execute(
            """SELECT a.id, a.source, a.external_id, a.url, a.title,
                      a.summary, a.published_at, a.detected_at,
                      c.sha256 AS capture_sha256, c.fetched_at, c.payload
               FROM articles a JOIN captures c ON c.id=a.capture_id
               ORDER BY a.id"""
        ).fetchall()
        valid = []
        source_counts: Counter[str] = Counter()
        exclusions: Counter[str] = Counter()
        for article in fetched:
            if not isinstance(article["payload"], bytes) or (
                hashlib.sha256(article["payload"]).hexdigest()
                != article["capture_sha256"]
            ):
                raise ValueError("Original RSS capture checksum mismatch")
            received = _timestamp(article["detected_at"])
            fetched_at = _timestamp(article["fetched_at"])
            if fetched_at > received:
                raise ValueError("Article detected before its capture was fetched")
            if received > cutoff:
                exclusions["AFTER_AS_OF"] += 1
                continue
            published = None
            if article["published_at"] is not None:
                try:
                    published = _timestamp(article["published_at"])
                except ValueError:
                    exclusions["PUBLISHED_AT_UNTRUSTED"] += 1
            valid.append((article, received, published))
            source_counts[article["source"]] += 1

        page = [(row, received, published) for row, received, published in valid
                if row["id"] > after_id][:limit + 1]
        has_more = len(page) > limit
        page = page[:limit]
        items = []
        for article, received, published in page:
            article_id = article["id"]
            if has_story:
                membership = db.execute(
                    "SELECT story_id FROM story_members WHERE article_id=?",
                    (article_id,),
                ).fetchone()
            else:
                membership = None
            if has_claims:
                claims = db.execute(
                    "SELECT processed_at FROM article_event_claims WHERE article_id=?",
                    (article_id,),
                ).fetchall()
                observed_claims = sum(_timestamp(c["processed_at"]) <= cutoff for c in claims)
            else:
                observed_claims = None
            if has_events:
                events = db.execute(
                    "SELECT processed_at FROM economic_events WHERE article_id=?",
                    (article_id,),
                ).fetchall()
                observed_events = sum(_timestamp(e["processed_at"]) <= cutoff for e in events)
            else:
                observed_events = None
            text = (article["title"] + "\n" + (article["summary"] or "")).encode("utf-8")
            items.append({
                "articleId": article_id,
                "source": article["source"],
                "externalId": article["external_id"],
                "sourceUrl": article["url"],
                "title": article["title"],
                "summary": article["summary"],
                "publishedAtReported": (
                    published.isoformat() if published is not None else None
                ),
                "firstObservedAt": received.isoformat(),
                "captureSha256": article["capture_sha256"],
                "captureChecksumVerified": True,
                "textSha256": hashlib.sha256(text).hexdigest(),
                "storyGroupId": membership["story_id"] if membership else None,
                "claimCountProcessedByCutoff": observed_claims,
                "eventCountProcessedByCutoff": observed_events,
                "importanceHumanGold": "UNKNOWN",
                "observedMarketReaction": "UNMEASURED",
                "negativeSampleConfirmed": False,
                "marketPriceLabel": None,
                "trainingEligible": False,
                "publishedTimeVerified": False,
            })
        return {
            "schemaVersion": "historical_news_cohort_v1",
            "status": "REVIEW_ONLY_NOT_TRAINING_READY",
            "asOf": cutoff.isoformat(),
            "sourceDb": str(source_db.resolve()),
            "totalObservedByCutoff": len(valid),
            "totalSourceArticles": len(fetched),
            "sourceCountsByCutoff": dict(sorted(source_counts.items())),
            "excludedCounts": dict(sorted(exclusions.items())),
            "pageAfterId": after_id,
            "pageLimit": limit,
            "items": items,
            "nextCursor": items[-1]["articleId"] if items else after_id,
            "hasMore": has_more,
            "allNewsIncludedRegardlessOfExtraction": True,
            "importanceGoldAvailable": False,
            "marketReactionLabelsAvailable": False,
            "negativeSamplesValidated": False,
            "trainingDatasetEligible": False,
            "sourceDbModified": False,
            "sourceAuthenticityIndependentlyVerified": False,
        }
