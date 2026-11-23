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

from tris_market_event_intelligence.gemma_original_article_batch_cli import (
    MODEL_PATTERN,
    ask_gemma,
    result_error,
)
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
            retry_minutes: int) -> tuple[list[sqlite3.Row], dict[str, int]]:
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    if not {"articles", "article_full_text"}.issubset(tables):
        return [], {"eligible": 0, "missingFullText": 0}
    missing = db.execute(
        """SELECT COUNT(*) FROM articles a LEFT JOIN article_full_text f
           ON a.id=f.article_id
           WHERE f.article_id IS NULL OR f.status!='COMPLETE'
              OR f.body IS NULL"""
    ).fetchone()[0]
    candidates = db.execute(
        """SELECT a.id, a.source, a.title, a.url, a.published_at,
                  a.detected_at, f.body, f.body_sha256, f.sparse_pages,
                  l.status AS prior_status, l.processed_at AS prior_at,
                  l.attempts AS prior_attempts
           FROM articles a JOIN article_full_text f ON f.article_id=a.id
           LEFT JOIN automatic_news_analyses l ON
                 l.article_id=a.id AND l.body_sha256=f.body_sha256 AND l.model=?
           WHERE f.status='COMPLETE' AND f.body IS NOT NULL
             AND (l.status IS NULL OR l.status='FAILED')
           ORDER BY a.id LIMIT 2000""", (model,)
    ).fetchall()
    cutoff = datetime.now(UTC) - timedelta(minutes=retry_minutes)
    ready_new = []
    ready_retry = []
    for row in candidates:
        body = row["body"]
        digest = row["body_sha256"]
        if not isinstance(body, str) or not isinstance(digest, str):
            continue
        if hashlib.sha256(body.encode()).hexdigest() != digest:
            continue
        if row["sparse_pages"] not in (None, "[]"):
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
    }


def process(db_path: Path, *, model: str = "gemma4:e2b",
            limit: int = 10, retry_minutes: int = 30,
            collect: bool = True, client: httpx.Client | None = None,
            verification_runs: int = 1) -> dict:
    """One bounded pass. Failed results are retryable; completed ones immutable."""
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
            db.commit()
            rows, coverage = pending(db, model, limit, retry_minutes)
            owned = client is None
            session = client or httpx.Client(
                timeout=httpx.Timeout(300), trust_env=False
            )
            completed = failed = 0
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
                    try:
                        analysis = ask_gemma(session, model, request)
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
                        status, error_code = "COMPLETED", None
                        encoded = json.dumps(analysis, ensure_ascii=False)
                        completed += 1
                    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                        status, error_code = "FAILED", result_error(exc)
                        encoded = None
                        failed += 1
                    with db:
                        db.execute(
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
                            (row["id"], row["body_sha256"], model, status, now(),
                             error_code, encoded),
                        )
                    results.append({"articleId": row["id"], "status": status,
                                    "errorCode": error_code})
            finally:
                if owned:
                    session.close()
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
    args = parser.parse_args()
    if not 300 <= args.interval <= 86400:
        parser.error("interval must be 300..86400 seconds")
    while True:
        try:
            result = process(args.db.expanduser(), model=args.model,
                             limit=args.limit, retry_minutes=args.retry_minutes,
                             collect=not args.skip_collection,
                             verification_runs=args.verification_runs)
        except (OSError, sqlite3.Error, ValueError) as exc:
            result = {"status": "ERROR", "errorCode": type(exc).__name__}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not args.watch:
            if result["status"] not in {"COMPLETED", "ALREADY_RUNNING"}:
                raise SystemExit(2)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
