"""Read-only source/article/event pipeline and duplicate-provenance audit.

Distinct article/event IDs are preserved; this never alters the operational DB.
"""
from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.story_index import canonical_url

MAX_AUDIT_ROWS = 50_000
MAX_SAMPLES = 20


def inspect_news_pipeline(
    db_path: Path, *, snapshot_observed_at: datetime, as_of: datetime,
) -> dict[str, object]:
    if not db_path.is_file() or db_path.is_symlink():
        raise ValueError("Existing non-symlink news DB required")
    if snapshot_observed_at.utcoffset() is None or as_of.utcoffset() is None:
        raise ValueError("Timezone-aware pipeline observation required")
    observed = snapshot_observed_at.astimezone(UTC)
    now = as_of.astimezone(UTC)
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro",
                                 uri=True, timeout=5)) as db:
        db.execute("PRAGMA query_only=ON")
        names = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        required = {"articles", "economic_events"}
        if not required.issubset(names):
            raise ValueError("Operational article/event tables not present")
        article_count = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        event_count = db.execute("SELECT COUNT(*) FROM economic_events").fetchone()[0]
        if article_count > MAX_AUDIT_ROWS or event_count > MAX_AUDIT_ROWS:
            raise ValueError("News pipeline audit exceeds bounded row limit")
        articles = db.execute(
            "SELECT id, source, url, title, detected_at FROM articles ORDER BY id"
        ).fetchall()
        events = db.execute(
            "SELECT id, article_id, extractor_version, source, source_url "
            "FROM economic_events ORDER BY id"
        ).fetchall()
        attempts: list[tuple] = []
        if "fetch_attempts" in names:
            attempts = db.execute(
                "SELECT source, status, inserted, completed_at "
                "FROM fetch_attempts ORDER BY id"
            ).fetchall()

    first_by_url: dict[tuple[str, str], int] = {}
    canonical_groups: dict[str, list[int]] = defaultdict(list)
    same_source_duplicates: list[dict[str, object]] = []
    first_seen_after = 0
    recent_articles_by_source: Counter[str] = Counter()
    for article_id, source, url, title, detected in articles:
        receipt = datetime.fromisoformat(detected)
        if receipt.utcoffset() is None:
            raise ValueError("Article receipt timestamp missing timezone")
        seen = receipt.astimezone(UTC)
        if observed <= seen <= now:
            first_seen_after += 1
            recent_articles_by_source[source] += 1
        key = (source, url)
        if key in first_by_url and len(same_source_duplicates) < MAX_SAMPLES:
            same_source_duplicates.append({
                "original_article_id": first_by_url[key],
                "duplicate_article_id": article_id,
                "source": source,
                "url": url,
            })
        else:
            first_by_url.setdefault(key, article_id)
        try:
            canonical_groups[canonical_url(url)].append(article_id)
        except ValueError:
            # Retain evidence for malformed old URLs; don't drop or rewrite.
            canonical_groups["unparseable-article-" + str(article_id)].append(article_id)
    source_url_counts = Counter((row[1], row[2]) for row in articles)
    repeated = sum(number - 1 for number in source_url_counts.values() if number > 1)
    cross_source_groups = sorted(
        (ids for ids in canonical_groups.values() if len(ids) > 1),
        key=lambda ids: (-len(ids), ids[0]),
    )
    event_by_article: Counter[int] = Counter(row[1] for row in events)
    event_duplicates = sum(count - 1 for count in event_by_article.values() if count > 1)
    latest_attempt: dict[str, dict[str, object]] = {}
    for source, status, inserted, completed_at in attempts:
        latest_attempt[source] = {
            "status": status, "inserted": inserted,
            "completed_at": completed_at,
        }
    return {
        "schema_version": "news_pipeline_evidence_v1",
        "status": "OBSERVED_READ_ONLY",
        "snapshot_observed_at": observed.isoformat(),
        "evaluation_cutoff": now.isoformat(),
        "articles_total": len(articles),
        "events_total": len(events),
        "articles_first_seen_after_snapshot": first_seen_after,
        "articles_first_seen_after_snapshot_by_source": dict(sorted(
            recent_articles_by_source.items()
        )),
        "same_source_exact_url_duplicate_articles": repeated,
        "same_source_duplicate_samples": same_source_duplicates,
        "canonical_url_multirecord_groups": len(cross_source_groups),
        "canonical_group_event_samples": [
            {
                "article_ids": ids[:30],
                "event_ids": [
                    event_id for event_id, article_id, *_ in events
                    if article_id in set(ids)
                ][:30],
            }
            for ids in cross_source_groups[:MAX_SAMPLES]
        ],
        "multiple_events_per_article": event_duplicates,
        "latest_fetch_attempts": dict(sorted(latest_attempt.items())),
        "fetch_attempts_available": "fetch_attempts" in names,
        "news_collection_active_verified": False,
        "new_event_after_snapshot_not_equivalent_to_new_article": True,
        "historical_duplicates_modified": False,
        "database_modified": False,
        "note": (
            "Existing event IDs may repeat the same story via different RSS GUIDs, "
            "sources, or extractor versions. URL groups are diagnostic, not proof "
            "that separate publications should be merged. No runtime status or "
            "missing news coverage is inferred from historical attempts."
        ),
    }
