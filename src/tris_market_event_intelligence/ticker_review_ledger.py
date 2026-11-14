"""Isolated append-only SQLite ledger for validated post-hoc ticker reviews.

Never opens the live news SQLite. A review is NOT an EventTickerLink:
no news-time linkage evidence, financial labels or training-ready export.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.ticker_human_review import review_ticker_candidates

SCHEMA = "ticker_review_ledger_v1"
MAX_ROWS = 1000


def _payload(item: dict[str, object]) -> str:
    return json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _digest(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _location(path: Path) -> Path:
    path = path.expanduser()
    if path.is_symlink() or not path.parent.is_dir():
        raise ValueError("Ledger must use a non-symlink path within an existing folder")
    return path.resolve(strict=False)


def _read_connection(path: Path) -> sqlite3.Connection:
    location = _location(path)
    if not location.is_file():
        raise ValueError("An existing initialized review ledger is required")
    db = sqlite3.connect(location.as_uri() + "?mode=ro", uri=True, timeout=10)
    try:
        db.execute("PRAGMA query_only=ON")
        row = db.execute(
            "SELECT value FROM ledger_meta WHERE key='schema_version'"
        ).fetchone()
        if row is None or row[0] != SCHEMA:
            raise ValueError("Refusing a database without the correct ledger identity")
        return db
    except (sqlite3.Error, ValueError):
        db.close()
        raise ValueError("Refusing a database without the correct ledger identity") from None


def _initialize(db: sqlite3.Connection) -> None:
    tables = {
        row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if tables:
        if "ledger_meta" not in tables:
            raise ValueError("Refusing an existing non-review database")
        row = db.execute(
            "SELECT value FROM ledger_meta WHERE key='schema_version'"
        ).fetchone()
        if row is None or row[0] != SCHEMA:
            raise ValueError("Refusing mismatched review database")
    else:
        db.executescript(
            """
            CREATE TABLE ledger_meta (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            );
            INSERT INTO ledger_meta(key,value)
                VALUES ('schema_version','ticker_review_ledger_v1');
            CREATE TABLE review_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                proposal_sha256 TEXT NOT NULL,
                event_id INTEGER NOT NULL,
                market TEXT NOT NULL,
                symbol TEXT NOT NULL,
                decision TEXT NOT NULL CHECK(decision IN ('APPROVE','REJECT')),
                reviewed_at TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                record_sha256 TEXT NOT NULL,
                record_json TEXT NOT NULL,
                UNIQUE(proposal_sha256,event_id,market,symbol)
            );
            CREATE INDEX idx_review_decisions_event
                ON review_decisions(event_id,market,symbol);
            """
        )
    if "review_decisions" not in {
        row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }:
        raise ValueError("Incomplete review ledger schema")


def import_review_decisions(
    path: Path, proposals_path: Path, decisions_path: Path, *,
    as_of: datetime | None = None,
) -> dict[str, object]:
    """Explicit import only; atomically write validated decisions to isolated DB.

    Identical replay is idempotent, conflicting updates fail closed. To
    reconsider a decision, produce a fresh reviewed proposal bundle.
    """
    audit = review_ticker_candidates(proposals_path, decisions_path, as_of=as_of)
    location = _location(path)
    if location in {
        proposals_path.expanduser().resolve(), decisions_path.expanduser().resolve()
    }:
        raise ValueError("Review DB cannot overwrite the review evidence inputs")
    if location.exists() and not location.is_file():
        raise ValueError("Review ledger must be a regular file")
    if location.exists():
        # Validate identity via a strictly read-only connection BEFORE any write.
        with _read_connection(location):
            pass
    db = sqlite3.connect(location, timeout=30)
    try:
        db.execute("PRAGMA busy_timeout=30000")
        _initialize(db)
        inserted = 0
        replayed = 0
        recorded_at = datetime.now(UTC).isoformat()
        with db:
            for item in audit["items"]:
                key = (
                    item["proposal_sha256"], item["event_id"],
                    item["market"], item["symbol"],
                )
                raw = _payload(item)
                sha = _digest(raw)
                existing = db.execute(
                    """SELECT record_sha256,record_json FROM review_decisions
                       WHERE proposal_sha256=? AND event_id=? AND market=? AND symbol=?""",
                    key,
                ).fetchone()
                if existing is not None:
                    if existing != (sha, raw):
                        raise ValueError("Existing review decision is immutable")
                    replayed += 1
                    continue
                db.execute(
                    """INSERT INTO review_decisions (
                        proposal_sha256,event_id,market,symbol,
                        decision,reviewed_at,recorded_at,record_sha256,record_json
                    ) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        *key, item["decision"], item["reviewed_at"],
                        recorded_at, sha, raw,
                    ),
                )
                inserted += 1
    finally:
        db.close()
    return {
        "schema_version": SCHEMA,
        "status": "STORED_POST_HOC_RESEARCH_ONLY",
        "inserted": inserted, "identical_replays": replayed,
        "source_proposal_sha256": audit["proposal_sha256"],
        "news_database_modified": False,
        "point_in_time_training_eligible": False,
        "approved_ticker_links_created": False,
    }


def list_review_decisions(
    path: Path, *, event_id: int | None = None, limit: int = 100,
) -> dict[str, object]:
    """Read-only paginated-style recent review listing with integrity checks."""
    if type(limit) is not int or not 1 <= limit <= MAX_ROWS:
        raise ValueError("Limit must be 1..1000")
    if event_id is not None and (type(event_id) is not int or event_id < 1):
        raise ValueError("Event ID must be positive")
    with _read_connection(path) as db:
        if event_id is None:
            rows = db.execute(
                """SELECT id,record_sha256,record_json
                   FROM review_decisions ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        else:
            rows = db.execute(
                """SELECT id,record_sha256,record_json
                   FROM review_decisions WHERE event_id=?
                   ORDER BY id DESC LIMIT ?""",
                (event_id, limit),
            ).fetchall()
    items: list[dict[str, object]] = []
    for row_id, stored_sha, raw in rows:
        if _digest(raw) != stored_sha:
            raise ValueError("Stored review record integrity mismatch")
        data = json.loads(raw)
        if data.get("eligible_for_point_in_time_training") is not False:
            raise ValueError("Review record unexpectedly claims training eligibility")
        items.append({"ledger_id": row_id, **data})
    return {
        "schema_version": SCHEMA,
        "status": "POST_HOC_REVIEWS_ONLY",
        "items": items, "returned": len(items),
        "point_in_time_training_eligible": False,
        "news_database_modified": False,
        "approved_ticker_links_created": False,
    }
