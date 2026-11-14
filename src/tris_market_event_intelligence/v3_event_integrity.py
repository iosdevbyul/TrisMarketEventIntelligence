"""Read-only evidence integrity checks for persisted post-cutover v3 events.

No extraction, replay, source fetch or market-impact judgment is performed.
A checksum only verifies stored provenance consistency, not the publisher.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

MAX_EVENTS = 50_000
MAX_SAMPLES = 20
VERSION = "multi-v3"


def _utc(raw: str) -> datetime:
    value = datetime.fromisoformat(raw)
    if value.utcoffset() is None:
        raise ValueError("Event evidence requires timezone-aware timestamps")
    return value.astimezone(UTC)


def inspect_v3_event_integrity(db_path: Path, *, as_of: datetime) -> dict[str, object]:
    """Inspect actual v3 writes since the persisted activation fence."""
    if db_path.is_symlink() or not db_path.is_file():
        raise ValueError("Existing non-symlink news DB required")
    if as_of.utcoffset() is None:
        raise ValueError("Timezone-aware as_of required")
    end = as_of.astimezone(UTC)
    with closing(sqlite3.connect(
        db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5,
    )) as db:
        db.execute("PRAGMA query_only=ON")
        tables = {
            r[0] for r in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        needed = {
            "articles", "captures", "economic_events", "event_metadata",
            "extraction_runs", "event_extractor_rollout",
        }
        if "event_extractor_rollout" not in tables:
            return {
                "status": "NOT_ACTIVATED",
                "database_modified": False,
            }
        if not needed.issubset(tables):
            return {
                "status": "UNAVAILABLE_SCHEMA",
                "missing_tables": sorted(needed - tables),
                "database_modified": False,
            }
        required = {
            "articles": {"id", "source", "url", "detected_at", "capture_id"},
            "captures": {"id", "sha256", "payload"},
            "economic_events": {
                "id", "article_id", "extractor_version", "status",
                "source", "source_url", "detected_at", "processed_at",
                "capture_sha256", "evidence",
            },
            "event_metadata": {"event_id", "claim_kind", "tags_json"},
            "extraction_runs": {"article_id", "extractor_version", "processed_at"},
            "event_extractor_rollout": {
                "extractor_version", "activated_at", "article_id_floor",
            },
        }
        for table, columns in required.items():
            found = {r[1] for r in db.execute(f"PRAGMA table_info({table})")}
            if not columns.issubset(found):
                return {
                    "status": "UNAVAILABLE_SCHEMA",
                    "missing_columns_table": table,
                    "database_modified": False,
                }
        fence = db.execute(
            """SELECT activated_at, article_id_floor
               FROM event_extractor_rollout WHERE extractor_version=?""",
            (VERSION,),
        ).fetchone()
        if fence is None:
            return {
                "status": "NOT_ACTIVATED",
                "database_modified": False,
            }
        activated = _utc(fence[0])
        floor = fence[1]
        if type(floor) is not int or floor < 0 or activated > end:
            return {"status": "INVALID_CUTOVER", "database_modified": False}
        count = db.execute(
            "SELECT COUNT(*) FROM economic_events WHERE extractor_version=?",
            (VERSION,),
        ).fetchone()[0]
        if count > MAX_EVENTS:
            return {
                "status": "BOUNDED_AUDIT_LIMIT",
                "v3_events_total": count,
                "max_events": MAX_EVENTS,
                "database_modified": False,
            }
        rows = db.execute(
            """SELECT e.id, e.article_id, e.status,
                      e.source, e.source_url, e.detected_at, e.processed_at,
                      e.capture_sha256, e.evidence,
                      a.source, a.url, a.detected_at,
                      c.sha256, c.payload,
                      m.event_id, m.tags_json,
                      r.processed_at
               FROM economic_events e
               LEFT JOIN articles a ON a.id=e.article_id
               LEFT JOIN captures c ON c.id=a.capture_id
               LEFT JOIN event_metadata m ON m.event_id=e.id
               LEFT JOIN extraction_runs r ON
                   r.article_id=e.article_id AND r.extractor_version=?
               WHERE e.extractor_version=?
               ORDER BY e.id""",
            (VERSION, VERSION),
        ).fetchall()
    counts: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = {}
    examples: list[dict[str, object]] = []
    delays: list[float] = []
    for (
        event_id, article_id, status, source, source_url, detected, processed,
        stored_hash, evidence, article_source, article_url, article_detected,
        capture_hash, payload, metadata_id, tags_json, checkpoint,
    ) in rows:
        if article_id <= floor or article_detected is None:
            counts["OUTSIDE_FENCE_OR_MISSING_ARTICLE"] += 1
            continue
        received = _utc(article_detected)
        if not activated <= received <= end:
            counts["OUTSIDE_FENCE_OR_MISSING_ARTICLE"] += 1
            continue
        counts["events"] += 1
        statuses[status] += 1
        issues: list[str] = []
        if checkpoint is None:
            issues.append("MISSING_V3_CHECKPOINT")
        if metadata_id is None:
            issues.append("MISSING_EVENT_METADATA")
        else:
            try:
                if not isinstance(json.loads(tags_json), list):
                    issues.append("INVALID_EVENT_TAGS")
            except (ValueError, TypeError):
                issues.append("INVALID_EVENT_TAGS")
        if (
            source != article_source or source_url != article_url
            or detected != article_detected
        ):
            issues.append("ARTICLE_EVENT_PROVENANCE_MISMATCH")
        if (
            not isinstance(payload, bytes)
            or not isinstance(capture_hash, str)
            or not isinstance(stored_hash, str)
            or hashlib.sha256(payload).hexdigest() != capture_hash
            or stored_hash != capture_hash
        ):
            issues.append("CAPTURE_SHA256_MISMATCH")
        try:
            created = _utc(processed)
            if created < received or created > end:
                issues.append("INVALID_PROCESSING_TIME")
            else:
                delays.append((created - received).total_seconds())
            if checkpoint is not None and _utc(checkpoint) != created:
                issues.append("CHECKPOINT_EVENT_TIME_MISMATCH")
        except (ValueError, TypeError):
            issues.append("INVALID_PROCESSING_TIME")
        if not isinstance(evidence, str) or not evidence.strip():
            issues.append("EMPTY_EVENT_EVIDENCE")
        for issue in issues:
            counts[issue] += 1
        if not issues:
            counts["consistent_events"] += 1
        if source not in by_source:
            by_source[source] = Counter()
        by_source[source]["events"] += 1
        by_source[source]["integrity_issues"] += bool(issues)
        if len(examples) < MAX_SAMPLES:
            examples.append({
                "event_id": event_id,
                "article_id": article_id,
                "source": source,
                "status": status,
                "integrity_issues": issues,
                "unverified_market_impact": True,
            })
    issues_total = counts["events"] - counts["consistent_events"]
    return {
        "schema_version": "v3_event_integrity_v1",
        "status": (
            "NO_POST_CUTOVER_V3_EVENTS" if counts["events"] == 0 else
            "INTEGRITY_ISSUES" if issues_total or counts["OUTSIDE_FENCE_OR_MISSING_ARTICLE"]
            else "OBSERVED_STORED_EVIDENCE"
        ),
        "activated_at": activated.isoformat(),
        "article_id_floor": floor,
        "evaluation_cutoff": end.isoformat(),
        "post_cutover_v3_event_rows": counts["events"],
        "consistent_event_rows": counts["consistent_events"],
        "event_rows_with_integrity_issues": issues_total,
        "outside_fence_or_missing_article": counts["OUTSIDE_FENCE_OR_MISSING_ARTICLE"],
        "issue_counts": {
            name: counts[name] for name in (
                "MISSING_V3_CHECKPOINT", "MISSING_EVENT_METADATA",
                "INVALID_EVENT_TAGS", "ARTICLE_EVENT_PROVENANCE_MISMATCH",
                "CAPTURE_SHA256_MISMATCH", "INVALID_PROCESSING_TIME",
                "CHECKPOINT_EVENT_TIME_MISMATCH", "EMPTY_EVENT_EVIDENCE",
            )
        },
        "event_status_counts": dict(sorted(statuses.items())),
        "by_source": {
            name: {"events": c["events"], "integrity_issues": c["integrity_issues"]}
            for name, c in sorted(by_source.items())
        },
        "receipt_to_event_seconds": {
            "count": len(delays),
            "minimum": min(delays) if delays else None,
            "maximum": max(delays) if delays else None,
        },
        "event_samples": examples,
        "samples_truncated": counts["events"] > len(examples),
        "source_authenticity_verified": False,
        "event_classification_accuracy_verified": False,
        "investment_impact_verified": False,
        "database_modified": False,
        "model_invoked": False,
        "note": (
            "Checks only SQLite linkage, timestamps, stored RSS payload hashes "
            "and event metadata. Consistency does not establish truthful "
            "publisher content or correct economic classification."
        ),
    }
