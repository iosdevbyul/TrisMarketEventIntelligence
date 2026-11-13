"""Append-only per-article claim extraction with atomic checkpoints."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from tris_market_event_intelligence.claim_extraction import CLAIM_VERSION, extract_claims
from tris_market_event_intelligence.collector import NewsStore, utc_now


@dataclass(frozen=True)
class ClaimStats:
    processed: int
    claims: int


class ClaimProcessor:
    def __init__(self, database: Path) -> None:
        self.store = NewsStore(database)

    def initialize(self) -> None:
        self.store.initialize()
        with self.store.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS claim_extraction_runs (
                    article_id INTEGER NOT NULL REFERENCES articles(id),
                    extractor_version TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    PRIMARY KEY (article_id, extractor_version)
                );
                CREATE TABLE IF NOT EXISTS article_event_claims (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id INTEGER NOT NULL REFERENCES articles(id),
                    extractor_version TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    classification TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    claim_kind TEXT NOT NULL,
                    polarity TEXT NOT NULL,
                    temporal_context TEXT NOT NULL,
                    status TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    verification_note TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    published_at TEXT,
                    detected_at TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    capture_sha256 TEXT NOT NULL,
                    UNIQUE(article_id, extractor_version, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_claim_article
                    ON article_event_claims(article_id, extractor_version);
                """
            )

    def process_pending(self, batch_size: int = 1000) -> ClaimStats:
        if not 1 <= batch_size <= 10000:
            raise ValueError("batch_size must be 1..10000")
        self.initialize()
        processed = created = 0
        with self.store.connect() as db:
            articles = db.execute(
                """
                SELECT a.id, a.source, a.url, a.title, a.summary, a.published_at,
                       a.detected_at, c.sha256
                FROM articles a JOIN captures c ON a.capture_id=c.id
                WHERE NOT EXISTS (
                    SELECT 1 FROM claim_extraction_runs r
                    WHERE r.article_id=a.id AND r.extractor_version=?
                )
                ORDER BY a.id LIMIT ?
                """,
                (CLAIM_VERSION, batch_size),
            ).fetchall()
            for article_id, source, url, title, summary, published, detected, sha in articles:
                claims = extract_claims(source, title, summary)
                now = utc_now()
                with db:
                    for ordinal, claim in enumerate(claims):
                        db.execute(
                            """
                            INSERT INTO article_event_claims (
                                article_id, extractor_version, ordinal,
                                classification, tag, claim_kind, polarity,
                                temporal_context, status, evidence, verification_note,
                                source, source_url, published_at, detected_at,
                                processed_at, capture_sha256
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (article_id, CLAIM_VERSION, ordinal,
                             claim.classification, claim.tag, claim.claim_kind,
                             claim.polarity, claim.temporal_context, claim.status,
                             claim.evidence, claim.verification_note, source, url,
                             published, detected, now, sha),
                        )
                        created += 1
                    db.execute(
                        "INSERT INTO claim_extraction_runs VALUES (?, ?, ?)",
                        (article_id, CLAIM_VERSION, now),
                    )
                processed += 1
        return ClaimStats(processed, created)

    def list_claims(self) -> list[dict[str, object]]:
        self.initialize()
        with self.store.connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM article_event_claims ORDER BY id").fetchall()
        return [dict(row) for row in rows]
