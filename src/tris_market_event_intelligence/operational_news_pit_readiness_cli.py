"""Read-only diagnostics from the existing real news.sqlite3 source ledger.

This audits real stored observations without inventing first-public timestamps,
issuer symbols, predictions, independently verified story families, or prices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.story_index import canonical_url

MAX_ARTICLES = 3000


def _stamp(raw: str | None) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        date = datetime.fromisoformat(raw)
        if date.utcoffset() is None:
            return None
        return date.astimezone(UTC)
    except (ValueError, TypeError, OverflowError):
        return None


def _tables(db: sqlite3.Connection) -> set[str]:
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def inspect_live_news(db_path: Path, *, limit: int = 1000) -> dict[str, object]:
    """Only inspect real DB facts. Existing production tables are untouched."""
    if not 1 <= limit <= MAX_ARTICLES:
        raise ValueError("Article audit bound must be 1..3000")
    if not db_path.is_file() or db_path.is_symlink():
        raise ValueError("Existing non-symlink news database required")
    from urllib.parse import quote

    db_uri = "file:" + quote(str(db_path.resolve()), safe="/") + "?mode=ro"
    with closing(sqlite3.connect(db_uri, uri=True, timeout=10)) as db:
        db.row_factory = sqlite3.Row
        tables = _tables(db)
        if not {"articles", "captures"}.issubset(tables):
            raise ValueError("News database is missing source tables")
        has_fulltext = "article_full_text" in tables
        has_story = "story_members" in tables
        has_automatic = "automatic_news_analyses" in tables
        fulltext_join = (
            "LEFT JOIN article_full_text f ON f.article_id=a.id"
            if has_fulltext else ""
        )
        story_join = (
            "LEFT JOIN story_members m ON m.article_id=a.id"
            if has_story else ""
        )
        analysis_join = (
            "LEFT JOIN (SELECT article_id, COUNT(*) AS models, "
            "SUM(CASE WHEN status='COMPLETED' THEN 1 ELSE 0 END) AS completed "
            "FROM automatic_news_analyses GROUP BY article_id) n ON n.article_id=a.id"
            if has_automatic else ""
        )
        fields = (
            ", f.status AS fulltext_status, f.observed_at AS text_observed, "
            "f.body, f.body_sha256, f.raw_payload, f.raw_sha256, f.sparse_pages"
            if has_fulltext else
            ", NULL AS fulltext_status, NULL AS text_observed, NULL AS body, "
            "NULL AS body_sha256, NULL AS raw_payload, NULL AS raw_sha256, "
            "NULL AS sparse_pages"
        )
        fields += (
            ", m.story_id, m.match_reason" if has_story
            else ", NULL AS story_id, NULL AS match_reason"
        )
        fields += (
            ", n.models AS model_rows, n.completed AS completed_models"
            if has_automatic else
            ", NULL AS model_rows, NULL AS completed_models"
        )
        total = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        rows = db.execute(
            "SELECT a.id, a.source, a.url, a.published_at, a.detected_at, "
            "c.fetched_at, c.sha256 AS feed_sha256, c.payload, c.source AS capture_source"
            + fields +
            " FROM articles a JOIN captures c ON c.id=a.capture_id "
            + fulltext_join + " " + story_join + " " + analysis_join +
            " ORDER BY a.id LIMIT ?", (limit,),
        ).fetchall()
        output = []
        family_seen: dict[int, int] = {}
        url_seen: dict[str, int] = {}
        counts: Counter[str] = Counter()
        for record in rows:
            article_id = record["id"]
            reasons = []
            published = _stamp(record["published_at"])
            detected = _stamp(record["detected_at"])
            fetched = _stamp(record["fetched_at"])
            if published is None:
                reasons.append("PUBLISHER_TIME_MISSING_OR_INVALID")
            if detected is None or fetched is None:
                reasons.append("COLLECTION_CLOCK_INVALID")
            if published and detected and published > detected:
                reasons.append("PUBLISHER_TIME_AFTER_DETECTION")
            if fetched and detected and detected < fetched:
                reasons.append("DETECTION_PRECEDES_FEED_CAPTURE")
            payload = record["payload"]
            if (not isinstance(payload, bytes)
                    or hashlib.sha256(payload).hexdigest() != record["feed_sha256"]
                    or record["capture_source"] != record["source"]):
                reasons.append("FEED_CAPTURE_INTEGRITY_FAILED")
            body = record["body"]
            if record["fulltext_status"] != "COMPLETE" or not isinstance(body, str):
                reasons.append("ORIGINAL_BODY_INCOMPLETE")
            elif (hashlib.sha256(body.encode()).hexdigest() != record["body_sha256"]
                  or record["sparse_pages"] not in (None, "[]")):
                reasons.append("BODY_INTEGRITY_OR_PDF_SPARSITY")
            raw = record["raw_payload"]
            if record["fulltext_status"] == "COMPLETE" and (
                not isinstance(raw, bytes)
                or hashlib.sha256(raw).hexdigest() != record["raw_sha256"]
            ):
                reasons.append("ORIGINAL_CAPTURE_INTEGRITY_FAILED")
            text_observed = _stamp(record["text_observed"])
            if record["fulltext_status"] == "COMPLETE" and text_observed is None:
                reasons.append("BODY_CAPTURE_TIME_INVALID")
            try:
                identity = canonical_url(record["url"])
            except ValueError:
                identity = None
                reasons.append("INVALID_ARTICLE_URL")
            if identity is not None and identity in url_seen:
                reasons.append("EXACT_URL_ALREADY_PRESENT")
            elif identity is not None:
                url_seen[identity] = article_id
            story = record["story_id"]
            if story is not None and story in family_seen:
                reasons.append("STORY_GROUP_HAS_EARLIER_ARTICLE")
            elif story is not None:
                family_seen[story] = article_id
            if story is None:
                reasons.append("STORY_GROUP_UNAVAILABLE")
            if not record["completed_models"]:
                reasons.append("NO_COMPLETED_AUTOMATIC_GEMMA_ANALYSIS")
            # Derived readiness is strictly diagnostic, not permission to
            # construct a fake PIT clock from article metadata.
            if reasons:
                for reason in set(reasons):
                    counts[reason] += 1
            output.append({
                "articleId": article_id, "source": record["source"],
                "publisherStatedAt": published.isoformat() if published else None,
                "feedDetectedAt": detected.isoformat() if detected else None,
                "feedCapturedAt": fetched.isoformat() if fetched else None,
                "originalBodyCapturedAt": (
                    text_observed.isoformat() if text_observed else None
                ),
                "storyCandidateId": story,
                "storyCandidateMatchReason": record["match_reason"],
                "completedModelRows": record["completed_models"] or 0,
                "sourceIntegrityReviewed": not any(
                    x in reasons for x in (
                        "FEED_CAPTURE_INTEGRITY_FAILED",
                        "BODY_INTEGRITY_OR_PDF_SPARSITY",
                        "ORIGINAL_CAPTURE_INTEGRITY_FAILED",
                    )
                ),
                "blockers": sorted(set(reasons)),
                "firstPublicTimeAuthenticated": False,
                "eventFamilyIndependentlyVerified": False,
                "symbolMappingIndependentlyVerified": False,
                "realPriceOutcomeVerified": False,
                "learningEligible": False,
            })
    return {
        "schemaVersion": "operational_news_pit_readiness_v1",
        "databaseModified": False, "scanned": len(output),
        "totalArticlesAtScan": total, "truncated": total > len(output),
        "blockerCounts": dict(sorted(counts.items())),
        "rows": output,
        "verifiedFirstPublicArticleCount": 0,
        "independentlyVerifiedMarketOutcomeCount": 0,
        "learningEligibleCount": 0,
        "note": (
            "Publisher publication times, RSS fetch times and story groups "
            "are observable evidence, but do not independently prove the "
            "earliest public event time or causal market response."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit real local news source readiness without modifying the database",
    )
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = inspect_live_news(args.db, limit=args.limit)
        if args.output.is_symlink():
            raise ValueError("Output symlink is prohibited")
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps({
            "scanned": report["scanned"], "totalArticles": report["totalArticlesAtScan"],
            "blockerCounts": report["blockerCounts"], "output": str(args.output),
            "productionDatabaseModified": False,
        }, ensure_ascii=False))
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"news-pit-readiness: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
