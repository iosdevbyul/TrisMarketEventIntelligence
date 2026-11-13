"""Idempotent bridge from article event claims into the existing impact pipeline.

This does not perform verification or invoke the model. It preserves original
claim state, evidence and provenance, including multiple claims per article.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tris_market_event_intelligence.claim_store import ClaimProcessor
from tris_market_event_intelligence.collector import utc_now
from tris_market_event_intelligence.event_store import EventProcessor

BRIDGE_VERSION = "claim-bridge-v1"


@dataclass(frozen=True)
class BridgeStats:
    processed: int
    created: int


class ClaimEventBridge:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.claims = ClaimProcessor(db_path)
        self.events = EventProcessor(db_path)

    def initialize(self) -> None:
        self.claims.initialize()
        self.events.initialize()
        with self.events.store.connect() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS claim_event_links (
                    claim_id INTEGER PRIMARY KEY REFERENCES article_event_claims(id),
                    event_id INTEGER NOT NULL UNIQUE REFERENCES economic_events(id),
                    bridge_version TEXT NOT NULL,
                    linked_at TEXT NOT NULL
                )"""
            )

    def process_pending(self, batch_size: int = 100) -> BridgeStats:
        """Commit one claim and link atomically; replay never duplicates events.

        A claim may represent a forecast, denial or historical reference;
        link it without silently changing its kind or verification status.
        The impact processor independently applies its eligibility rules.
        """
        if not 1 <= batch_size <= 1000:
            raise ValueError("batch_size must be 1..1000")
        self.initialize()
        with self.events.store.connect() as db:
            rows = db.execute(
                """SELECT c.id, c.article_id, c.classification, c.status, c.evidence,
                          c.source, c.source_url, c.published_at, c.detected_at,
                          c.capture_sha256, c.claim_kind, c.tag, c.verification_note
                   FROM article_event_claims c
                   WHERE NOT EXISTS (
                       SELECT 1 FROM claim_event_links l WHERE l.claim_id=c.id
                   )
                   ORDER BY c.id ASC LIMIT ?""",
                (batch_size,),
            ).fetchall()
        processed = created = 0
        for row in rows:
            (claim_id, article_id, classification, status, evidence, source,
             url, published, detected, sha, kind, tag, note) = row
            # The bridge cannot confer stronger certainty than original extraction.
            if status not in {"VERIFIED", "CANDIDATE", "UNVERIFIED"}:
                raise ValueError("Unsupported source claim status")
            version = f"{BRIDGE_VERSION}:{claim_id}"
            now = utc_now()
            with self.events.store.connect() as db, db:
                cursor = db.execute(
                    """INSERT OR IGNORE INTO economic_events (
                        article_id, extractor_version, classification, status,
                        direction, change_basis_points, evidence, source, source_url,
                        published_at, detected_at, processed_at, capture_sha256
                    ) VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, ?)""",
                    (article_id, version, classification, status, evidence, source,
                     url, published, detected, now, sha),
                )
                if cursor.rowcount:
                    event_id = cursor.lastrowid
                    db.execute(
                        """INSERT INTO event_metadata (
                            event_id, claim_kind, tags_json, verification_note
                        ) VALUES (?, ?, ?, ?)""",
                        (event_id, kind, json.dumps([tag], ensure_ascii=False), note),
                    )
                    created += 1
                else:
                    event_id = db.execute(
                        "SELECT id FROM economic_events "
                        "WHERE article_id=? AND extractor_version=?",
                        (article_id, version),
                    ).fetchone()[0]
                db.execute(
                    "INSERT OR IGNORE INTO claim_event_links "
                    "(claim_id, event_id, bridge_version, linked_at) "
                    "VALUES (?, ?, ?, ?)",
                    (claim_id, event_id, BRIDGE_VERSION, now),
                )
            processed += 1
        return BridgeStats(processed, created)
