"""One bounded news -> extraction -> optional local Gemma operational pass.

No automatic model calls: the owner must explicitly opt in. Reuses the
existing RSS cycle, extraction/impact processors, and source freshness
gate. This does not enable launchd or authenticate market news.
"""
from __future__ import annotations

import fcntl
import os
import sqlite3
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import httpx

from tris_market_event_intelligence.collector import utc_now
from tris_market_event_intelligence.event_store import EventProcessor
from tris_market_event_intelligence.exchange_sessions import SessionCalendar
from tris_market_event_intelligence.impact import ImpactProvider
from tris_market_event_intelligence.impact_store import ImpactProcessor
from tris_market_event_intelligence.listing_news_validation import (
    _verified_bundle,
    validate_existing_listing_news,
)
from tris_market_event_intelligence.operations import (
    OPERATIONAL_SOURCES,
    _write_status,
    run_cycle,
)
from tris_market_event_intelligence.session_operations import market_session_plan
from tris_market_event_intelligence.source_health import audit_source_health

MAX_EVENT_EXTRACTION = 1000
MAX_LOCAL_ANALYSES = 3


def _articles_before(path: Path) -> int:
    if not path.is_file():
        return 0
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        if not db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='articles'"
        ).fetchone():
            return 0
        return int(db.execute("SELECT COALESCE(MAX(id),0) FROM articles").fetchone()[0])


def _fresh_event_ids(path: Path, first_article_id: int, sources: tuple[str, ...]) -> tuple[int, ...]:
    """Find eligible news from THIS run, not an old unprocessed research backlog."""
    placeholders = ",".join("?" for _ in sources)
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        result = db.execute(
            f"""SELECT e.id FROM economic_events e
                JOIN event_metadata m ON m.event_id=e.id
                WHERE e.article_id > ?
                  AND e.source IN ({placeholders})
                  AND (
                    (e.status='VERIFIED' AND e.classification='MONETARY_POLICY')
                    OR (
                      e.status IN ('CANDIDATE','UNVERIFIED')
                      AND e.classification != 'MONETARY_POLICY'
                      AND m.claim_kind='REPORTED'
                    )
                  )
                ORDER BY e.id ASC LIMIT ?""",
            (first_article_id, *sources, MAX_LOCAL_ANALYSES),
        ).fetchall()
    return tuple(int(row[0]) for row in result)


def run_integrated_pass(
    db_path: Path,
    *,
    selected_sources: tuple[str, ...] = OPERATIONAL_SOURCES,
    status_path: Path | None = None,
    client: httpx.Client | None = None,
    model: ImpactProvider | None = None,
    enable_local_analysis: bool = False,
    live_freshness_seconds: int = 900,
    source_success_seconds: int = 900,
    max_event_extraction: int = MAX_EVENT_EXTRACTION,
    session_calendar: SessionCalendar | None = None,
    target_markets: tuple[str, ...] = ("XKRX", "XNYS", "XNAS"),
    listing_bundle: Path | None = None,
    review_output_root: Path | None = None,
) -> dict[str, object]:
    """Finite pass with separate pipeline lock; defaults to *zero* LLM calls.

    The normal collector retains its original lock and status contract.
    A pipeline lock spans collection, extraction and bounded local inference
    so a second integrated invocation cannot duplicate model work.
    """
    if not selected_sources or len(set(selected_sources)) != len(selected_sources):
        raise ValueError("Distinct RSS sources are required")
    if enable_local_analysis and model is None:
        raise ValueError("Explicit local model required when analysis is enabled")
    if model is not None and not enable_local_analysis:
        raise ValueError("Do not supply a model without explicit analysis opt-in")
    if not 1 <= live_freshness_seconds <= 86400:
        raise ValueError("Freshness budget must be 1..86400 seconds")
    if not 1 <= source_success_seconds <= 86400:
        raise ValueError("Source poll age must be 1..86400 seconds")
    if not 1 <= max_event_extraction <= MAX_EVENT_EXTRACTION:
        raise ValueError("Extraction batch must be 1..1000")
    if (listing_bundle is None) != (review_output_root is None):
        raise ValueError("Ticker review requires both listing bundle and output folder")
    if listing_bundle is not None and (enable_local_analysis or model is not None):
        raise ValueError("Ticker review pass must not invoke a local model")
    first_seen = None
    if listing_bundle is not None:
        # Validate the pinned source and write destination before network or DB
        # work. Never enable lookahead by silently changing source observation.
        _, _, first_seen, _ = _verified_bundle(listing_bundle)
        assert review_output_root is not None
        if (review_output_root.is_symlink() or not review_output_root.is_dir()
                or not review_output_root.is_absolute()):
            raise ValueError("Existing absolute non-symlink private review folder required")
    if session_calendar is not None:
        plan = market_session_plan(
            session_calendar, now=datetime.now(UTC),
            markets=target_markets, sources=selected_sources,
        )
        if plan["decision"] != "RUN":
            return {
                "schema_version": "integrated_news_pass_v1",
                "status": (
                    "session_skipped" if plan["decision"] == "SKIP" else "blocked"
                ),
                "session_plan": plan,
                "collection": None, "event_extraction": None,
                "source_health": None, "model_invoked": False,
                "network_requests": 0, "mac_runtime_verified": False,
                "market_coverage_verified": False,
                "prediction_model_trained": False,
            }
    db_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_lock = db_path.with_name(db_path.name + ".pipeline.lock")
    fd = os.open(pipeline_lock, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "schema_version": "integrated_news_pass_v1", "status": "already_running",
                "network_requests_verified": False, "model_invoked": False,
                "mac_runtime_verified": False,
            }
        began = utc_now()
        before = _articles_before(db_path)
        # Let the original collector own its status/history checkpoint.
        collector = run_cycle(
            db_path, selected_sources=selected_sources, client=client,
        )
        report: dict[str, object] = {
            "schema_version": "integrated_news_pass_v1",
            "session_plan": plan if session_calendar is not None else None,
            "started_at": began,
            "status": collector["status"],
            "collection": collector,
            "event_extraction": {"processed": 0, "events": 0, "verified": 0},
            "source_health": None,
            "analysis": {
                "enabled": enable_local_analysis,
                "selected_event_ids": [],
                "processed": 0, "impacts": 0, "failed": 0, "deferred": 0,
            },
            "model_invoked": False,
            "mac_runtime_verified": False,
            "market_coverage_verified": False,
            "prediction_model_trained": False,
            "ticker_review": {
                "enabled": listing_bundle is not None,
                "status": "NOT_REQUESTED" if listing_bundle is None else "PENDING",
                "candidates": 0,
                "automatic_approvals": 0,
            },
        }
        try:
            if collector["status"] in {"failed", "already_running"}:
                report["status"] = "blocked"
            else:
                processor = EventProcessor(db_path)
                # Intentional bounded pass; never silently drain unlimited backlog.
                extraction = processor.process_pending(
                    batch_size=max_event_extraction,
                    received_since=first_seen,
                )
                report["event_extraction"] = asdict(extraction)
                report["source_health"] = audit_source_health(
                    db_path, expected_sources=selected_sources,
                    now=datetime.now(UTC),
                    max_success_age_seconds=source_success_seconds,
                )
                if listing_bundle is not None:
                    assert review_output_root is not None
                    # All writer calls use the same .pipeline.lock. The
                    # validation is independently read-only and create-only.
                    folder = review_output_root / (
                        f"integrated-{datetime.now(UTC):%Y%m%dT%H%M%S}-"
                        f"{os.getpid()}-{time.time_ns()}"
                    )
                    review = validate_existing_listing_news(
                        listing_bundle, db_path, folder,
                        as_of=datetime.now(UTC), post_snapshot_only=True,
                    )
                    report["ticker_review"] = {
                        "enabled": True,
                        "status": "REVIEW_REQUIRED" if review["candidate_count"]
                                  else "NO_CANDIDATES",
                        "summary_path": str(folder / "summary.json"),
                        "source_observed_at": first_seen.isoformat(),
                        "events_inspected": review["events_inspected"],
                        "candidates": review["candidate_count"],
                        "alias_hints": review["diagnostic_alias_mentions"],
                        "has_more_events": review["has_more_events"],
                        "automatic_approvals": 0,
                        "human_review_required": True,
                        "historical_listing_verified": False,
                    }
                if enable_local_analysis:
                    event_ids = _fresh_event_ids(db_path, before, selected_sources)
                    report["analysis"]["selected_event_ids"] = list(event_ids)
                    if event_ids:
                        impact = ImpactProcessor(db_path, model).process_pending(
                            batch_size=len(event_ids), event_ids=event_ids,
                            live_freshness_seconds=live_freshness_seconds,
                            source_success_seconds=source_success_seconds,
                        )
                        report["analysis"].update(asdict(impact))
                        report["model_invoked"] = impact.processed + impact.failed > 0
                if collector["status"] == "degraded" or (
                    report["source_health"]["status"] != "OK"
                ):
                    report["status"] = "degraded"
                if report["analysis"]["failed"]:
                    report["status"] = "degraded"
                if report["analysis"]["deferred"]:
                    report["status"] = "degraded"
                if report["analysis"].get("already_running"):
                    report["status"] = "blocked"
        except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
            report["status"] = "failed"
            report["error_code"] = type(exc).__name__
        report["finished_at"] = utc_now()
        if status_path is not None:
            _write_status(status_path, report)
        return report
