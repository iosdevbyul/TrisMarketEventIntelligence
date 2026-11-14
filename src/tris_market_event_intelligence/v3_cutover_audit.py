"""Read-only attribution of article outcomes after the pinned v3 cutover.

The report uses actual SQLite checkpoints and event rows, never infers
whether external publishers were fully covered or which ticker benefits.
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.story_index import canonical_url

MAX_ROWS = 50_000
MAX_SAMPLES = 25
V3 = "multi-v3"
V2 = "multi-v2"


def _utc(raw: str) -> datetime:
    stamp = datetime.fromisoformat(raw)
    if stamp.utcoffset() is None:
        raise ValueError("v3 audit requires timezone-aware timestamps")
    return stamp.astimezone(UTC)


def audit_v3_cutover(db_path: Path, *, as_of: datetime) -> dict[str, object]:
    """Observe the persisted v3 activation fence without creating one."""
    if db_path.is_symlink() or not db_path.is_file():
        raise ValueError("Existing regular news database required")
    if as_of.utcoffset() is None:
        raise ValueError("Audit cutoff must be timezone-aware")
    cutoff = as_of.astimezone(UTC)
    with closing(sqlite3.connect(
        db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5,
    )) as db:
        db.execute("PRAGMA query_only=ON")
        tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required = {
            "articles", "extraction_runs", "economic_events",
            "event_extractor_rollout",
        }
        if "event_extractor_rollout" not in tables:
            return {
                "status": "NOT_ACTIVATED",
                "cutover": None,
                "database_modified": False,
            }
        if not required.issubset(tables):
            return {
                "status": "UNAVAILABLE_SCHEMA",
                "missing_tables": sorted(required - tables),
                "database_modified": False,
            }
        expected = {
            "articles": {"id", "source", "url", "title", "detected_at"},
            "extraction_runs": {"article_id", "extractor_version"},
            "economic_events": {
                "id", "article_id", "extractor_version", "classification",
            },
            "event_extractor_rollout": {
                "extractor_version", "activated_at", "article_id_floor",
            },
        }
        for name, keys in expected.items():
            actual = {
                row[1] for row in db.execute(f"PRAGMA table_info({name})")
            }
            if not keys.issubset(actual):
                return {
                    "status": "UNAVAILABLE_SCHEMA",
                    "missing_columns_table": name,
                    "database_modified": False,
                }
        fence = db.execute(
            """SELECT activated_at, article_id_floor
               FROM event_extractor_rollout WHERE extractor_version=?""",
            (V3,),
        ).fetchone()
        if fence is None:
            return {
                "status": "NOT_ACTIVATED",
                "cutover": None,
                "database_modified": False,
            }
        since, floor = _utc(fence[0]), fence[1]
        if not isinstance(floor, int) or floor < 0 or since > cutoff:
            return {
                "status": "INVALID_CUTOVER",
                "database_modified": False,
            }
        total = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        if total > MAX_ROWS:
            return {
                "status": "BOUNDED_AUDIT_LIMIT",
                "total_articles": total, "max_articles": MAX_ROWS,
                "database_modified": False,
            }
        rows = db.execute(
            """SELECT a.id, a.source, a.url, a.title, a.detected_at,
                      EXISTS(SELECT 1 FROM extraction_runs r
                             WHERE r.article_id=a.id AND r.extractor_version=?),
                      EXISTS(SELECT 1 FROM extraction_runs r
                             WHERE r.article_id=a.id AND r.extractor_version=?),
                      (SELECT COUNT(*) FROM economic_events e
                       WHERE e.article_id=a.id AND e.extractor_version=?),
                      (SELECT COUNT(*) FROM economic_events e
                       WHERE e.article_id=a.id AND e.extractor_version=?)
               FROM articles a ORDER BY a.id""",
            (V3, V2, V3, V2),
        ).fetchall()
    earliest: dict[tuple[str, str], int] = {}
    for ident, _, url, title, *_ in rows:
        try:
            key = (canonical_url(url), title.strip().casefold())
        except (ValueError, TypeError, AttributeError):
            key = (f"invalid:{ident}", "")
        earliest.setdefault(key, ident)
    counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = {}
    samples: list[dict[str, object]] = []
    for ident, source, url, title, detected, v3_run, v2_run, v3_events, v2_events in rows:
        received = _utc(detected)
        if ident <= floor or received < since or received > cutoff:
            continue
        try:
            key = (canonical_url(url), title.strip().casefold())
        except (ValueError, TypeError, AttributeError):
            key = (f"invalid:{ident}", "")
        rep = earliest[key]
        if v3_events:
            stage = "V3_EVENT_WRITTEN"
        elif v3_run and ident != rep:
            stage = "V3_DUPLICATE_CHECKPOINTED"
        elif v3_run:
            stage = "V3_CLASSIFIED_NO_EVENT"
        elif v2_run:
            stage = "V2_PROCESSED_FIRST"
        else:
            stage = "AWAITING_EXTRACTION"
        counts[stage] += 1
        sources = source_counts.setdefault(source, Counter())
        sources[stage] += 1
        if v3_events and v2_events:
            counts["MULTIVERSION_EVENT_ANOMALY"] += 1
        if v3_run and v2_run:
            counts["MULTIVERSION_CHECKPOINT_ANOMALY"] += 1
        counts["v3_event_rows"] += v3_events
        counts["v2_event_rows"] += v2_events
        if len(samples) < MAX_SAMPLES:
            samples.append({
                "article_id": ident,
                "source": source,
                "detected_at": received.isoformat(),
                "stage": stage,
                "representative_article_id": rep,
                "v3_event_rows": v3_events,
                "v2_event_rows": v2_events,
            })
    stage_names = (
        "V3_EVENT_WRITTEN", "V3_DUPLICATE_CHECKPOINTED",
        "V3_CLASSIFIED_NO_EVENT", "V2_PROCESSED_FIRST", "AWAITING_EXTRACTION",
    )
    return {
        "schema_version": "v3_cutover_audit_v1",
        "status": "OBSERVED_READ_ONLY",
        "cutover": {
            "activated_at": since.isoformat(),
            "article_id_floor": floor,
        },
        "evaluation_cutoff": cutoff.isoformat(),
        "post_cutover_articles": sum(counts[k] for k in stage_names),
        "stage_counts": {name: counts[name] for name in stage_names},
        "v3_event_rows": counts["v3_event_rows"],
        "v2_event_rows": counts["v2_event_rows"],
        "multiversion_event_anomalies": counts["MULTIVERSION_EVENT_ANOMALY"],
        "multiversion_checkpoint_anomalies": counts[
            "MULTIVERSION_CHECKPOINT_ANOMALY"
        ],
        "by_source": {
            source: {name: source_count[name] for name in stage_names}
            for source, source_count in sorted(source_counts.items())
        },
        "article_samples": samples,
        "samples_truncated": sum(counts[k] for k in stage_names) > len(samples),
        "pending_is_not_a_verified_failure": True,
        "source_coverage_verified": False,
        "ticker_auto_approvals": 0,
        "database_modified": False,
        "model_invoked": False,
        "note": (
            "Article counts reflect stored RSS rows, not publisher coverage. "
            "v3 duplicate checkpoints indicate the existing exact URL and "
            "headline representative rule, not independent article authenticity. "
            "v2-first checkpoints are intentionally not replayed by v3."
        ),
    }
