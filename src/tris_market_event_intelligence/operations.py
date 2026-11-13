"""Stage 6: bounded operational RSS cycle and atomic local status checkpoints.

No service activation, portal HTML crawling, API credentials, trading, or AI calls.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path

import httpx

from tris_market_event_intelligence.claim_store import ClaimProcessor
from tris_market_event_intelligence.collector import NewsStore, OfficialNewsCollector, utc_now
from tris_market_event_intelligence.operational_history import history_path_for, record_cycle
from tris_market_event_intelligence.source_catalog import DEFAULT_FEEDS
from tris_market_event_intelligence.story_index import StoryIndex

# Every default was successfully requested by the bounded public network probe.
# Exclude known HTTP 403 BLS and rate-limited GDELT from automatic operation.
OPERATIONAL_SOURCES = (
    "fed_monetary", "fed_all",
    "bok_all", "bok_statistics", "bok_corrections",
    "bbc_business", "bbc_technology", "bbc_world",
)


def _write_status(path: Path, content: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=".market-event-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            os.fchmod(f.fileno(), 0o600)
            json.dump(content, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(filename, path)
    finally:
        if os.path.exists(filename):
            os.unlink(filename)


def run_cycle(
    db_path: Path, *, status_path: Path | None = None,
    selected_sources: tuple[str, ...] = OPERATIONAL_SOURCES,
    client: httpx.Client | None = None,
) -> dict[str, object]:
    """One finite cycle. Reject overlapping invocations before any network calls.

    A partial provider outage is degraded, not successful operation. Failure to
    collect any source is reported as failure; existing DB evidence is untouched.
    """
    if not selected_sources or len(set(selected_sources)) != len(selected_sources):
        raise ValueError("Select one or more distinct RSS sources")
    unknown = set(selected_sources).difference(DEFAULT_FEEDS)
    if unknown:
        raise ValueError(f"Unknown sources: {', '.join(sorted(unknown))}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = db_path.with_name(db_path.name + ".cycle.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "status": "already_running", "source_results": {},
                "new_articles": 0, "new_claims": 0,
                "quality_validated": False, "mac_runtime_verified": False,
            }
        started_at = utc_now()
        try:
            store = NewsStore(db_path)
            store.initialize()
            before = store.stats()["articles"]
            # Network failures on one source must not abort claim processing.
            results = OfficialNewsCollector(
                store, client=client,
                feeds={name: DEFAULT_FEEDS[name] for name in selected_sources},
            ).collect_once()
            claims = ClaimProcessor(db_path)
            stories = StoryIndex(db_path)
            claim_total = grouped_total = 0
            while True:
                batch = claims.process_pending()
                claim_total += batch.claims
                if batch.processed < 1000:
                    break
            while True:
                batch = stories.index_pending()
                grouped_total += batch.indexed
                if batch.indexed < 1000:
                    break
            ok = sum(x["status"] in {"ok", "not_modified"} for x in results.values())
            status = "failed" if ok == 0 else "healthy" if ok == len(results) else "degraded"
            outcome: dict[str, object] = {
                "status": status,
                "started_at": started_at,
                "finished_at": utc_now(),
                "source_results": results,
                "successful_sources": ok,
                "total_sources": len(results),
                "new_articles": store.stats()["articles"] - before,
                "new_claims": claim_total,
                "indexed_articles": grouped_total,
                # Must remain false until on-device validation and human audit.
                "quality_validated": False,
                "mac_runtime_verified": False,
                "ai_used": False,
                "scraping_used": False,
            }
    
        except Exception as exc:  # noqa: BLE001 - record uncaught cycle failures safely
            # Do not leave a stale healthy checkpoint after an unexpected failure.
            # Exception messages can contain URLs, secrets or article contents:
            # persist only a bounded exception class and never an exception string.
            outcome = {
                "status": "failed",
                "started_at": started_at,
                "finished_at": utc_now(),
                "error_code": type(exc).__name__,
                "source_results": {},
                "successful_sources": 0,
                "total_sources": len(selected_sources),
                "new_articles": 0,
                "new_claims": 0,
                "indexed_articles": 0,
                "quality_validated": False,
                "mac_runtime_verified": False,
                "ai_used": False,
                "scraping_used": False,
            }
            if status_path:
                _write_status(status_path, outcome)
                record_cycle(history_path_for(status_path), outcome)
            return outcome
        if status_path:
            _write_status(status_path, outcome)
            record_cycle(history_path_for(status_path), outcome)
        return outcome
