"""Explicit, bounded post-listing news event materialization and validation.

This is deliberately NOT a cron/launchd change: operator opt-in is required.
"""
from __future__ import annotations

import fcntl
import os
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.event_store import EventProcessor
from tris_market_event_intelligence.listing_news_validation import (
    _verified_bundle,
    validate_existing_listing_news,
)

MAX_MATERIALIZATION_BATCH = 500


def materialize_post_snapshot_events(
    bundle: Path, news_db: Path, output_dir: Path, *,
    batch_size: int = MAX_MATERIALIZATION_BATCH,
    as_of: datetime | None = None,
) -> dict[str, object]:
    """Operator-authorized writes are only append-only event/checkpoint rows.

    The original collector / pipeline lock prevents overlapping integrated
    model passes; the method never calls the network, Ollama or the collector.
    """
    if not 1 <= batch_size <= MAX_MATERIALIZATION_BATCH:
        raise ValueError("batch_size must be between 1 and 500")
    if news_db.is_symlink() or not news_db.is_file():
        raise ValueError("Existing nonsymlink news DB required")
    _, issuers, first_seen, _ = _verified_bundle(bundle)
    if as_of is not None and as_of.utcoffset() is None:
        raise ValueError("Evaluation cutoff must have a timezone")
    now = (as_of or datetime.now(UTC)).astimezone(UTC)
    if now < first_seen:
        raise ValueError("Evaluation precedes source acquisition")
    if output_dir.is_symlink() or output_dir.exists() or (
        not output_dir.parent.is_dir() or output_dir.parent.is_symlink()
    ):
        raise ValueError("Result must be a new directory inside an existing real folder")
    lock_path = news_db.with_name(news_db.name + ".pipeline.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Integrated extraction is already running") from exc
        # Preflight before any DB mutation. Missing operational article schema
        # cannot be interpreted as zero pending.
        with sqlite3.connect(news_db.resolve().as_uri() + "?mode=ro", uri=True) as db:
            db.execute("PRAGMA query_only=ON")
            present = {r[0] for r in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            if not {"articles", "captures"}.issubset(present):
                raise ValueError("Operational articles and captures required")
            columns = {r[1] for r in db.execute("PRAGMA table_info(articles)")}
            if not {"id", "detected_at", "capture_id", "title", "source"}.issubset(columns):
                raise ValueError("Incomplete article schema; no events materialized")
        extraction = EventProcessor(news_db).process_pending(
            batch_size=batch_size,
            received_since=first_seen, received_until=now,
        )
        # The write is committed before this independent read-only report.
        # A failed report does NOT undo append-only event records; retrying the
        # bounded processor is safe due to extraction_run idempotency.
        validation = validate_existing_listing_news(
            bundle, news_db, output_dir, as_of=now, post_snapshot_only=True,
        )
    return {
        "schema_version": "post_snapshot_materialization_v1",
        "status": "EXTRACTED_AND_VALIDATED",
        "source_first_observed_at": first_seen.isoformat(),
        "evaluation_cutoff": now.isoformat(),
        "issuer_rows": len(issuers),
        "event_extraction": asdict(extraction),
        "batch_limit": batch_size,
        "more_pending_articles_not_excluded": extraction.processed == batch_size,
        "validation_summary": str(output_dir / "summary.json"),
        "post_snapshot_events_inspected": validation["events_inspected"],
        "ticker_candidates": validation["candidate_count"],
        "diagnostic_alias_mentions": validation["diagnostic_alias_mentions"],
        "approval_automated": False,
        "collector_invoked": False,
        "model_invoked": False,
        "launchagent_modified": False,
        "historical_articles_modified": False,
        "new_event_records_may_have_been_written": extraction.events > 0,
    }
