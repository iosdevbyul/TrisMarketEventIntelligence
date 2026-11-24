"""Automatic news collection -> verified original article -> Gemma reasoning.

Opt-in foreground worker. A single nonblocking process lock prevents duplicate
AI jobs; a durable SQLite ledger enables crash recovery and visibility.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sqlite3
import time
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from tris_market_event_intelligence.automatic_forecast_precommitment import (
    initialize as initialize_forecasts,
)
from tris_market_event_intelligence.automatic_forecast_precommitment import (
    inspect_commitments,
    record_new_analysis,
)
from tris_market_event_intelligence.event_aware_forecast import ask_dual, prior_context
from tris_market_event_intelligence.gemma_original_article_batch_cli import (
    MODEL_PATTERN,
    ask_gemma,
    result_error,
)
from tris_market_event_intelligence.market_outcome_validation_cli import _mappings
from tris_market_event_intelligence.news_content_quality import classify
from tris_market_event_intelligence.news_semantic_consistency import compare_analyses
from tris_market_event_intelligence.operations import OPERATIONAL_SOURCES, run_cycle

SCHEMA = """
CREATE TABLE IF NOT EXISTS automatic_news_analyses (
 article_id INTEGER NOT NULL,
 body_sha256 TEXT NOT NULL,
 model TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('COMPLETED','FAILED')),
 processed_at TEXT NOT NULL,
 attempts INTEGER NOT NULL DEFAULT 0,
 error_code TEXT,
 analysis_json TEXT,
 PRIMARY KEY(article_id,body_sha256,model)
);
"""


def now() -> str:
    return datetime.now(UTC).isoformat()


def initialize(db: sqlite3.Connection) -> None:
    db.execute(SCHEMA)


def pending(db: sqlite3.Connection, model: str, limit: int,
            retry_minutes: int, *, fresh_only_minutes: int | None = None,
            observed_now: datetime | None = None, sec_only: bool = False) -> tuple[list[sqlite3.Row], dict[str, int]]:
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    if not {"articles", "article_full_text"}.issubset(tables):
        return [], {"eligible": 0, "missingFullText": 0}
    scope = "a.source GLOB 'sec_8k_[0-9]*'" if sec_only else "1=1"
    missing = db.execute(
        f"""SELECT COUNT(*) FROM articles a LEFT JOIN article_full_text f
           ON a.id=f.article_id
           WHERE {scope} AND (f.article_id IS NULL OR f.status!='COMPLETE'
              OR f.body IS NULL)"""
    ).fetchone()[0]
    fulltext_columns = {
        column[1] for column in db.execute("PRAGMA table_info(article_full_text)")
    }
    body_observed_expr = (
        "f.observed_at AS body_observed_at"
        if "observed_at" in fulltext_columns
        else "NULL AS body_observed_at"
    )
    format_expr = "f.format" if "format" in fulltext_columns else "NULL"
    error_expr = "f.error_code" if "error_code" in fulltext_columns else "NULL"
    sec_guard = (
        "AND f.format IN ('SEC_BUNDLE','SEC_BUNDLE_PARTIAL')"
        if sec_only and "format" in fulltext_columns else ""
    )
    # Forward capture is ordered by the *actual detected clock*, not IDs.
    # The older historical backlog may exceed the 2000-row query bound.
    order = (
        "a.detected_at DESC, a.id DESC"
        if fresh_only_minutes is not None else "a.id"
    )
    candidates = db.execute(
        f"""SELECT a.id, a.source, a.title, a.url, a.published_at,
                  a.detected_at, f.body, f.body_sha256, f.sparse_pages,
                  {body_observed_expr}, {format_expr} AS content_format,
                  {error_expr} AS content_error,
                  l.status AS prior_status, l.processed_at AS prior_at,
                  l.attempts AS prior_attempts
           FROM articles a JOIN article_full_text f ON f.article_id=a.id
           LEFT JOIN automatic_news_analyses l ON
                 l.article_id=a.id AND l.body_sha256=f.body_sha256 AND l.model=?
           WHERE f.status='COMPLETE' AND f.body IS NOT NULL AND {scope} {sec_guard}
             AND (l.status IS NULL OR l.status='FAILED')
           ORDER BY {order} LIMIT 2000""", (model,)
    ).fetchall()
    current = observed_now or datetime.now(UTC)
    cutoff = current - timedelta(minutes=retry_minutes)
    fresh_cutoff = (
        current - timedelta(minutes=fresh_only_minutes)
        if fresh_only_minutes is not None else None
    )
    blocked_fresh = 0
    blocked_quality = 0
    ready_new = []
    ready_retry = []
    for row in candidates:
        if fresh_cutoff is not None:
            try:
                published = datetime.fromisoformat(row["published_at"])
                detected = datetime.fromisoformat(row["detected_at"])
                observed = datetime.fromisoformat(row["body_observed_at"])
                # A recent detection is NOT a newly published historical story:
                # the release, detection and original body must all be recent.
                fresh = (
                    all(t.tzinfo is not None and t.utcoffset() is not None
                        for t in (published, detected, observed))
                    and fresh_cutoff <= published <= detected <= observed <= current
                    and observed - published <= timedelta(minutes=fresh_only_minutes)
                )
            except (TypeError, ValueError, OverflowError):
                fresh = False
            if not fresh:
                blocked_fresh += 1
                continue
        body = row["body"]
        digest = row["body_sha256"]
        if not isinstance(body, str) or not isinstance(digest, str):
            continue
        if hashlib.sha256(body.encode()).hexdigest() != digest:
            continue
        if classify(body, url=row["url"], title=row["title"],
                    sparse=row["sparse_pages"]) != "TEXT_CANDIDATE":
            blocked_quality += 1
            continue
        if row["prior_status"] == "FAILED":
            if row["prior_attempts"] >= 3:
                continue
            try:
                if datetime.fromisoformat(row["prior_at"]) > cutoff:
                    continue
            except (TypeError, ValueError):
                continue
            ready_retry.append(row)
        else:
            ready_new.append(row)
    if fresh_only_minutes is not None:
        # Fresh-only mode does not borrow spare slots from past article IDs.
        chosen = (ready_new + ready_retry)[:limit]
        return chosen, {
            "eligible": len(candidates), "missingFullText": missing,
            "readyNew": len(ready_new), "readyRetry": len(ready_retry),
            "contentQualityRejected": blocked_quality,
            "freshOnlyMinutes": fresh_only_minutes,
            "freshClockRejected": blocked_fresh,
            "freshAdmissible": len(ready_new) + len(ready_retry),
            "noHistoricalBackfill": True,
            "freshQueryLimitReached": len(candidates) == 2000,
        }
    # A large historical backlog must not prevent newly collected news from
    # receiving timely analysis. Reserve capacity for old news and due retries.
    newest_slots = (limit + 2) // 3
    oldest_slots = (limit + 1) // 3
    retry_slots = limit - newest_slots - oldest_slots
    selected = (
        list(reversed(ready_new[-newest_slots:]))
        + ready_new[:oldest_slots]
        + ready_retry[:retry_slots]
    )
    # For small backlogs the two new-news queues may overlap.
    unique = {row["id"]: row for row in selected}
    for row in list(reversed(ready_new)) + ready_retry:
        if len(unique) >= limit:
            break
        unique.setdefault(row["id"], row)
    return list(unique.values())[:limit], {
        "eligible": len(candidates), "missingFullText": missing,
        "readyNew": len(ready_new), "readyRetry": len(ready_retry),
        "contentQualityRejected": blocked_quality,
    }


def process(db_path: Path, *, model: str = "gemma4:e2b",
            limit: int = 10, retry_minutes: int = 30,
            collect: bool = True, client: httpx.Client | None = None,
            verification_runs: int = 1, seal_forecasts: bool = False,
            mappings_path: Path | None = None,
            fresh_only_minutes: int | None = None,
            event_aware: bool = False, sec_only: bool = False) -> dict:
    """One bounded pass. Failed results are retryable; completed ones immutable."""
    if sec_only and collect:
        raise ValueError("SEC-only analysis requires explicit collection or skip-collection")
    if event_aware and (not seal_forecasts or fresh_only_minutes is None):
        raise ValueError("Event-aware forecasts require fresh-only sealing")
    if event_aware and verification_runs != 1:
        raise ValueError("Event-aware capture requires one verification run")
    if fresh_only_minutes is not None and not seal_forecasts:
        raise ValueError("Fresh-only capture requires explicit forecast sealing")
    if fresh_only_minutes is not None and not 1 <= fresh_only_minutes <= 1440:
        raise ValueError("fresh_only_minutes must be 1..1440")
    if seal_forecasts and mappings_path is None:
        raise ValueError("Explicit reviewed market mappings required to precommit")
    reviewed_mappings = _mappings(mappings_path) if seal_forecasts and mappings_path else {}
    if not MODEL_PATTERN.fullmatch(model):
        raise ValueError("Invalid Ollama model")
    if verification_runs not in (1, 2):
        raise ValueError("verification_runs must be 1 or 2")
    if not 1 <= limit <= 100 or not 1 <= retry_minutes <= 1440:
        raise ValueError("Invalid bounded run settings")
    if not db_path.is_file() or db_path.is_symlink():
        raise ValueError("Existing non-symlink news database required")
    lock_path = db_path.with_name(db_path.name + ".auto-gemma.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "ALREADY_RUNNING", "modelInvoked": False}
        collection = run_cycle(db_path, selected_sources=OPERATIONAL_SOURCES,
                               enable_fulltext=True) if collect else None
        with closing(sqlite3.connect(db_path, timeout=30)) as db:
            db.execute("PRAGMA busy_timeout=30000")
            initialize(db)
            if seal_forecasts:
                initialize_forecasts(db)
            db.commit()
            rows, coverage = pending(
                db, model, limit, retry_minutes,
                fresh_only_minutes=fresh_only_minutes,
                observed_now=datetime.now(UTC), sec_only=sec_only,
            )
            owned = client is None
            session = client or httpx.Client(
                timeout=httpx.Timeout(300), trust_env=False
            )
            completed = failed = 0
            commitment_counts: dict[str, int] = {}
            results = []
            try:
                for row in rows:
                    body = row["body"]
                    request = {
                        "article": {
                            "articleId": row["id"], "source": row["source"],
                            "title": row["title"], "url": row["url"],
                            "publishedAt": row["published_at"],
                            "detectedAt": row["detected_at"],
                            "bodySha256": row["body_sha256"],
                        },
                        "body": body[:12000],
                        "bodyTruncated": len(body) > 12000,
                    }
                    if row["source"].startswith("sec_8k_"):
                        request["article"]["contentScope"] = (
                            "PRIMARY_WITH_PARTIAL_EXHIBITS"
                            if row["content_format"] == "SEC_BUNDLE_PARTIAL"
                            else "CAPTURED_FILING_DOCUMENTS"
                        )
                        request["article"]["missingEvidence"] = row["content_error"]
                    try:
                        analysis = (
                            ask_dual(session, model, request, prior_context(db, request["article"]))
                            if event_aware else ask_gemma(session, model, request)
                        )
                        if verification_runs == 2:
                            second = ask_gemma(session, model, request)
                            analysis["_repeatability"] = compare_analyses(analysis, second)
                        else:
                            analysis["_repeatability"] = {
                                "status": "SINGLE_RUN_UNVERIFIED",
                                "runs": 1,
                                "semanticCorrectnessVerified": False,
                                "priceDataUsed": False,
                            }
                        if row["source"].startswith("sec_8k_"):
                            analysis["_sourceScope"] = {
                                "scope": request["article"]["contentScope"],
                                "missingEvidence": row["content_error"],
                                "bodyTruncated": request["bodyTruncated"],
                                "historicalResearchOnly": not seal_forecasts,
                            }
                        status, error_code = "COMPLETED", None
                        encoded = json.dumps(analysis, ensure_ascii=False)
                        completed += 1
                    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                        status, error_code = "FAILED", result_error(exc)
                        encoded = None
                        failed += 1
                    analysis_time = now()
                    with db:
                        inserted = db.execute(
                            """INSERT INTO automatic_news_analyses
                               (article_id,body_sha256,model,status,processed_at,
                                attempts,error_code,analysis_json)
                               VALUES (?,?,?,?,?,1,?,?)
                               ON CONFLICT(article_id,body_sha256,model) DO UPDATE SET
                                status=excluded.status,
                                processed_at=excluded.processed_at,
                                attempts=automatic_news_analyses.attempts+1,
                                error_code=excluded.error_code,
                                analysis_json=excluded.analysis_json
                               WHERE automatic_news_analyses.status='FAILED'""",
                            (row["id"], row["body_sha256"], model, status,
                             analysis_time, error_code, encoded),
                        )
                        if (seal_forecasts and status == "COMPLETED"
                                and inserted.rowcount == 1):
                            forecast_counts = record_new_analysis(
                                db,
                                article={**request["article"],
                                         "bodyObservedAt": row["body_observed_at"]},
                                analysis=analysis, model=model,
                                body_sha256=row["body_sha256"],
                                analyzed_at=analysis_time,
                                mappings=reviewed_mappings,
                            )
                            for eligibility, count in forecast_counts.items():
                                commitment_counts[eligibility] = (
                                    commitment_counts.get(eligibility, 0) + count
                                )
                    results.append({"articleId": row["id"], "status": status,
                                    "errorCode": error_code})
            finally:
                if owned:
                    session.close()
            commitment_audit = inspect_commitments(db) if seal_forecasts else None
    collection_status = collection.get("status") if collection is not None else None
    return {
        "status": (
            "PARTIAL_FAILURE" if failed else
            "DEGRADED_COLLECTION" if collection_status in {"degraded", "failed"} else
            "COMPLETED"
        ),
        "collection": collection, "collectionStatus": collection_status,
        "collectionHealthy": collection_status == "healthy" if collect else None,
        "model": model, "completed": completed, "failed": failed,
        "coverage": coverage, "results": results,
        "verificationRuns": verification_runs,
        "marketPricesAnalyzed": False,
        "forecastPrecommitmentsEnabled": seal_forecasts,
        "freshOnlyCaptureEnabled": fresh_only_minutes is not None,
        "noHistoricalPredictionBackfill": fresh_only_minutes is not None,
        "newForecastCommitments": commitment_counts if seal_forecasts else {},
        "forecastCommitmentAudit": commitment_audit,
        "forecastFirstPublicAvailabilityIndependentlyVerified": False,
        "forecastExternalTimestampAuthenticated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Opt-in automatic news + Gemma pipeline")
    base = Path.home() / "Library/Application Support/TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--retry-minutes", type=int, default=30)
    parser.add_argument("--interval", type=int, default=900,
                        help="Seconds between passes when --watch is selected")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--verification-runs", type=int, choices=(1, 2), default=1,
                        help="1=unverified single analysis; 2=compare independent model calls")
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--sec-only", action="store_true")
    parser.add_argument(
        "--fresh-only-minutes", type=int,
        help="With --seal-forecasts, process only genuinely recent first-published, "
             "detected and full-body-observed articles, never old backlog",
    )
    parser.add_argument(
        "--seal-forecasts", action="store_true",
        help="Opt in to immutable local forecast commitments on NEW model completions",
    )
    parser.add_argument(
        "--mappings", type=Path,
        help="Required source-reviewed precommitted target map when --seal-forecasts",
    )
    args = parser.parse_args()
    if not 300 <= args.interval <= 86400:
        parser.error("interval must be 300..86400 seconds")
    while True:
        try:
            result = process(args.db.expanduser(), model=args.model,
                             limit=args.limit, retry_minutes=args.retry_minutes,
                             collect=not args.skip_collection,
                             verification_runs=args.verification_runs,
                             seal_forecasts=args.seal_forecasts,
                             mappings_path=args.mappings,
                             fresh_only_minutes=args.fresh_only_minutes, sec_only=args.sec_only)
        except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
            result = {"status": "ERROR", "errorCode": type(exc).__name__}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not args.watch:
            if result["status"] not in {"COMPLETED", "ALREADY_RUNNING"}:
                raise SystemExit(2)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
