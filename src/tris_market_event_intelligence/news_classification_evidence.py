"""Read-only diagnosis of actually stored post-snapshot articles.

Preserve source/article/event identity. Canonical-URL groups are review-only
hints; never merge records or automatically approve a ticker association.
"""
from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.claim_extraction import EDUCATIONAL, extract_claims
from tris_market_event_intelligence.multi_events import (
    extract_news_event,
)
from tris_market_event_intelligence.story_index import canonical_url

MAX_ARTICLES = 50_000
MAX_DETAILS = 30


def _time(value: str) -> datetime:
    date = datetime.fromisoformat(value)
    if date.utcoffset() is None:
        raise ValueError("News timestamps must include a timezone")
    return date.astimezone(UTC)


def inspect_unclassified_news(
    news_db: Path, *, observed_at: datetime, as_of: datetime,
) -> dict[str, object]:
    """Diagnose current rule coverage, not assert historical extractor output.

    Distinguishes no lexical cues, excluded educational titles, and genuine
    extractor-version/rule disagreement. Includes only bounded snippets from
    the already stored RSS source; no remote article retrieval.
    """
    if not news_db.is_file() or news_db.is_symlink():
        raise ValueError("Existing regular non-symlink database required")
    if observed_at.utcoffset() is None or as_of.utcoffset() is None:
        raise ValueError("Timezone-aware evidence window required")
    start, stop = observed_at.astimezone(UTC), as_of.astimezone(UTC)
    if start > stop:
        raise ValueError("Invalid evidence interval")
    with closing(sqlite3.connect(
        news_db.resolve().as_uri() + "?mode=ro", uri=True, timeout=5,
    )) as db:
        db.execute("PRAGMA query_only=ON")
        tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {"articles", "economic_events", "extraction_runs"}
        if not required.issubset(tables):
            return {
                "status": "UNAVAILABLE_SCHEMA",
                "missing_tables": sorted(required - tables),
                "database_modified": False,
            }
        cols = {
            table: {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
            for table in required
        }
        if not {
            "id", "source", "url", "title", "summary", "detected_at",
        }.issubset(cols["articles"]) or not {
            "article_id", "extractor_version",
        }.issubset(cols["extraction_runs"]) or not {
            "article_id", "id",
        }.issubset(cols["economic_events"]):
            return {
                "status": "UNAVAILABLE_SCHEMA",
                "missing_columns": True,
                "database_modified": False,
            }
        count = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        if count > MAX_ARTICLES:
            return {
                "status": "BOUNDED_AUDIT_LIMIT", "articles_total": count,
                "database_modified": False,
            }
        articles = db.execute(
            """SELECT a.id, a.source, a.url, a.title, a.summary, a.detected_at,
                      EXISTS(SELECT 1 FROM extraction_runs r
                             WHERE r.article_id=a.id),
                      EXISTS(SELECT 1 FROM economic_events e
                             WHERE e.article_id=a.id)
               FROM articles a ORDER BY a.id"""
        ).fetchall()
    in_window = [row for row in articles if start <= _time(row[5]) <= stop]
    groups: dict[str, list[tuple]] = defaultdict(list)
    results: list[dict[str, object]] = []
    reasons: Counter[str] = Counter()
    for article_id, source, url, title, summary, detected, checkpoint, has_event in in_window:
        try:
            group_key = canonical_url(url)
        except ValueError:
            group_key = f"invalid-url:article:{article_id}"
        groups[group_key].append((
            article_id, source, title, summary, detected, bool(checkpoint), bool(has_event)
        ))
    duplicate_ids: dict[int, list[int]] = {}
    group_samples: list[dict[str, object]] = []
    for key, members in groups.items():
        sources = {member[1] for member in members}
        if len(members) < 2 or len(sources) < 2:
            continue
        ids = sorted(member[0] for member in members)
        for member in members:
            duplicate_ids[member[0]] = ids
        if len(group_samples) < MAX_DETAILS:
            group_samples.append({
                "canonical_url": key,
                "article_ids": ids[:30],
                "sources": sorted(sources),
                "requires_independent_identity_review": True,
                "automatically_merged": False,
            })
    for article_id, source, url, title, summary, detected, checkpoint, has_event in in_window:
        title = title or ""
        current = extract_news_event(source, title, summary)
        claims = extract_claims(source, title, summary)
        if not checkpoint:
            reason = "AWAITING_EXTRACTION"
        elif has_event:
            reason = "HAS_STORED_EVENT"
        elif current is not None:
            reason = "CURRENT_RULE_MATCH_BUT_NO_STORED_EVENT"
        elif claims:
            reason = "CLAIM_ONLY_NO_EVENT"
        elif EDUCATIONAL.search(title):
            reason = "EXPLANATORY_OR_EDUCATIONAL_HEADLINE"
        else:
            reason = "NO_SUPPORTED_EVENT_CUE"
        reasons[reason] += 1
        if len(results) < MAX_DETAILS:
            results.append({
                "article_id": article_id,
                "source": source,
                "url": url,
                "detected_at": _time(detected).isoformat(),
                "headline": title[:300],
                "summary_excerpt": (summary or "")[:300],
                "classification_reason": reason,
                "current_rule_classification": (
                    current.event.classification if current else None
                ),
                "current_rule_status": current.event.status if current else None,
                "claim_count": len(claims),
                "cross_source_same_url_article_ids": duplicate_ids.get(article_id, []),
                "automatic_event_created_by_audit": False,
                "automatic_ticker_approved": False,
            })
    return {
        "schema_version": "news_classification_evidence_v1",
        "status": "OBSERVED_READ_ONLY",
        "snapshot_observed_at": start.isoformat(),
        "evaluation_cutoff": stop.isoformat(),
        "articles_inspected": len(in_window),
        "reason_counts": dict(sorted(reasons.items())),
        "article_samples": results,
        "article_samples_truncated": len(in_window) > len(results),
        "cross_source_canonical_url_groups": sum(
            len({m[1] for m in members}) > 1 and len(members) > 1
            for members in groups.values()
        ),
        "cross_source_group_samples": group_samples,
        "unique_canonical_url_groups": len(groups),
        "unique_articles_verified": False,
        "cross_source_events_suppressed": 0,
        "automatic_ticker_approvals": 0,
        "rule_version_replay_not_performed": True,
        "database_modified": False,
        "network_requests": 0,
        "model_invoked": False,
        "note": (
            "Diagnostic reasons use the current rules against persisted RSS text; "
            "they do not reconstruct the historical extractor version or validate "
            "the full external article. Same canonical URL across sources is a "
            "review-only group, not authority to delete or merge article/event IDs."
        ),
    }
