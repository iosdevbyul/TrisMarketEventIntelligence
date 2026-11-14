"""Read-only, point-in-time operating evidence for each new-news pipeline stage.

An RSS poll, stored article, extractor checkpoint, economic event and
human-review proposal are distinct observations. Never infer missing
publisher content or verified investment impact from any of these.
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

MAX_ARTICLES = 50_000
MAX_SAMPLES = 20


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError("Stage evidence requires timezone-aware timestamps")
    return parsed.astimezone(UTC)


def inspect_news_stage_transitions(
    db_path: Path, *, snapshot_observed_at: datetime, as_of: datetime,
) -> dict[str, object]:
    """Summarize actual SQLite rows; no DDL, data writes or network access."""
    if db_path.is_symlink() or not db_path.is_file():
        raise ValueError("Existing nonsymlink news DB required")
    if snapshot_observed_at.utcoffset() is None or as_of.utcoffset() is None:
        raise ValueError("Timezone-aware stage boundaries required")
    since = snapshot_observed_at.astimezone(UTC)
    cutoff = as_of.astimezone(UTC)
    if cutoff < since:
        raise ValueError("Stage cutoff predates listing observation")
    with closing(sqlite3.connect(
        db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5,
    )) as db:
        db.execute("PRAGMA query_only=ON")
        tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        needed = {"articles", "economic_events", "extraction_runs"}
        if not needed.issubset(tables):
            return {
                "status": "UNAVAILABLE_SCHEMA",
                "missing_tables": sorted(needed - tables),
                "database_modified": False,
            }
        required = {
            "articles": {"id", "source", "url", "detected_at"},
            "economic_events": {"id", "article_id", "status"},
            "extraction_runs": {"article_id", "processed_at"},
        }
        for table, columns in required.items():
            present = {item[1] for item in db.execute(f"PRAGMA table_info({table})")}
            if not columns.issubset(present):
                return {
                    "status": "UNAVAILABLE_SCHEMA",
                    "missing_columns_table": table,
                    "database_modified": False,
                }
        count = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        if count > MAX_ARTICLES:
            return {
                "status": "BOUNDED_AUDIT_LIMIT",
                "total_articles": count,
                "max_articles": MAX_ARTICLES,
                "database_modified": False,
            }
        rows = db.execute(
            """SELECT a.id, a.source, a.url, a.detected_at,
                      EXISTS (SELECT 1 FROM extraction_runs r
                              WHERE r.article_id=a.id) AS checkpointed,
                      (SELECT MIN(r.processed_at) FROM extraction_runs r
                       WHERE r.article_id=a.id) AS first_processed,
                      (SELECT COUNT(*) FROM economic_events e
                       WHERE e.article_id=a.id) AS events,
                      (SELECT MIN(e.processed_at) FROM economic_events e
                       WHERE e.article_id=a.id) AS first_event_at
               FROM articles a ORDER BY a.id"""
        ).fetchall()
        # Most legacy fixtures have no event.processed_at; inspect this
        # explicitly instead of falling through to SQL errors.
    totals: Counter[str] = Counter()
    source_summary: dict[str, Counter[str]] = {}
    samples: list[dict[str, object]] = []
    event_delay_seconds: list[float] = []
    for article_id, source, url, received_text, checkpointed, processed_text, count_events, event_text in rows:
        received = _utc(received_text)
        if not since <= received <= cutoff:
            continue
        counts = source_summary.setdefault(source, Counter())
        totals["articles"] += 1
        counts["articles"] += 1
        if checkpointed:
            totals["checkpointed"] += 1
            counts["checkpointed"] += 1
        else:
            totals["pending_extraction"] += 1
            counts["pending_extraction"] += 1
        if count_events:
            totals["events"] += count_events
            counts["events"] += count_events
            totals["articles_with_events"] += 1
            counts["articles_with_events"] += 1
            if event_text:
                event_delay_seconds.append(
                    (_utc(event_text) - received).total_seconds()
                )
        elif checkpointed:
            totals["processed_without_event"] += 1
            counts["processed_without_event"] += 1
        if len(samples) < MAX_SAMPLES and (not checkpointed or not count_events):
            samples.append({
                "article_id": article_id,
                "source": source,
                "stage": (
                    "AWAITING_EVENT_EXTRACTION" if not checkpointed
                    else "EXTRACTED_NO_CLASSIFIED_EVENT"
                ),
                "detected_at": received.isoformat(),
                "url": url,
            })
    return {
        "schema_version": "news_stage_transitions_v1",
        "status": "OBSERVED_READ_ONLY",
        "snapshot_observed_at": since.isoformat(),
        "evaluation_cutoff": cutoff.isoformat(),
        "new_articles": totals["articles"],
        "extraction_checkpointed_articles": totals["checkpointed"],
        "pending_extraction_articles": totals["pending_extraction"],
        "processed_without_classified_event": totals["processed_without_event"],
        "articles_with_events": totals["articles_with_events"],
        "event_rows": totals["events"],
        "event_processing_delay_seconds": {
            "count": len(event_delay_seconds),
            "min": min(event_delay_seconds) if event_delay_seconds else None,
            "max": max(event_delay_seconds) if event_delay_seconds else None,
        },
        "by_source": {
            source: {
                key: count[key] for key in (
                    "articles", "checkpointed", "pending_extraction",
                    "processed_without_event", "articles_with_events", "events",
                )
            }
            for source, count in sorted(source_summary.items())
        },
        "unresolved_article_samples": samples,
        "ticker_candidates_measured_separately": True,
        "ticker_candidates_automatically_approved": 0,
        "automatic_backfill": False,
        "database_modified": False,
        "model_invoked": False,
        "note": (
            "Zero new articles with successful RSS polls is not proof of complete "
            "publisher coverage. A processed article with no event may be "
            "unclassified by design. Candidate counts come from the separately "
            "verified issuer proposal report, not this operational stage audit."
        ),
    }
