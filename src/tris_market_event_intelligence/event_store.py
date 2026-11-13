"""Append-only event materialization from collector articles."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from tris_market_event_intelligence.collector import NewsStore, utc_now
from tris_market_event_intelligence.events import EXTRACTOR_VERSION, extract_policy_event


@dataclass(frozen=True)
class ProcessingStats:
    processed: int
    events: int
    verified: int


class EventProcessor:
    def __init__(self, database: Path) -> None:
        self.store = NewsStore(database)

    def initialize(self) -> None:
        self.store.initialize()
        with self.store.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS extraction_runs (
                    article_id INTEGER NOT NULL REFERENCES articles(id),
                    extractor_version TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    PRIMARY KEY(article_id, extractor_version)
                );
                CREATE TABLE IF NOT EXISTS economic_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id INTEGER NOT NULL REFERENCES articles(id),
                    extractor_version TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('CANDIDATE', 'VERIFIED', 'UNVERIFIED')),
                    direction TEXT,
                    change_basis_points INTEGER,
                    evidence TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    published_at TEXT,
                    detected_at TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    capture_sha256 TEXT NOT NULL,
                    UNIQUE(article_id, extractor_version)
                );
                """
            )

    def process_pending(self, batch_size: int = 1000) -> ProcessingStats:
        if not 1 <= batch_size <= 10000:
            raise ValueError("batch_size must be 1..10000")
        self.initialize()
        processed = 0
        created = 0
        verified = 0
        # Reprocess only previously unseen (article_id, extractor_version) pairs.
        with self.store.connect() as db:
            rows = db.execute(
                """
                SELECT a.id, a.source, a.url, a.title, a.summary, a.published_at,
                       a.detected_at, c.sha256
                FROM articles a
                JOIN captures c ON a.capture_id = c.id
                WHERE NOT EXISTS (
                    SELECT 1 FROM extraction_runs r
                    WHERE r.article_id = a.id AND r.extractor_version = ?
                )
                ORDER BY a.id ASC LIMIT ?
                """,
                (EXTRACTOR_VERSION, batch_size),
            ).fetchall()
            for article_id, source, url, title, summary, published, detected, sha in rows:
                event = extract_policy_event(source, title, summary)
                now = utc_now()
                # Per-article transaction ensures a processing checkpoint cannot
                # advance if writing the corresponding event fails.
                with db:
                    if event is not None:
                        cursor = db.execute(
                            """
                            INSERT OR IGNORE INTO economic_events (
                                article_id, extractor_version, classification, status,
                                direction, change_basis_points, evidence, source, source_url,
                                published_at, detected_at, processed_at, capture_sha256
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (article_id, event.extractor_version, event.classification,
                             event.status, event.direction, event.change_basis_points,
                             event.evidence, source, url, published, detected, now, sha),
                        )
                        created += cursor.rowcount
                        if event.status == "VERIFIED" and cursor.rowcount:
                            verified += 1
                    db.execute(
                        "INSERT OR IGNORE INTO extraction_runs VALUES (?, ?, ?)",
                        (article_id, EXTRACTOR_VERSION, now),
                    )
                    processed += 1
        return ProcessingStats(processed, created, verified)

    def list_events(self, status: str | None = None) -> list[dict[str, object]]:
        self.initialize()
        with self.store.connect() as db:
            db.row_factory = sqlite3.Row
            if status is not None:
                rows = db.execute(
                    "SELECT * FROM economic_events WHERE status = ? ORDER BY id", (status,)
                ).fetchall()
            else:
                rows = db.execute("SELECT * FROM economic_events ORDER BY id").fetchall()
        return [dict(row) for row in rows]
