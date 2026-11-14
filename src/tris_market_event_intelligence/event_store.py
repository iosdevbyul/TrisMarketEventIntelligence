"""Append-only event materialization from collector articles."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.collector import NewsStore, utc_now
from tris_market_event_intelligence.multi_events import (
    EXTRACTOR_VERSION,
    extract_news_event,
)
from tris_market_event_intelligence.story_index import canonical_url
from tris_market_event_intelligence.versioned_event_preview import (
    preview_news_event_v3,
)

V3_PRODUCTION_VERSION = "multi-v3"


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
                CREATE TABLE IF NOT EXISTS event_metadata (
                    event_id INTEGER PRIMARY KEY REFERENCES economic_events(id),
                    claim_kind TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    verification_note TEXT NOT NULL
                );
                """
            )

    def activate_v3_rollout(self) -> tuple[datetime, int, bool]:
        """Pin activation time and existing article high-water mark once.

        The returned immutable fence is never inferred from historical news
        or the listing snapshot. A separate operator opt-in is mandatory.
        """
        self.initialize()
        with self.store.connect() as db, db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS event_extractor_rollout (
                    extractor_version TEXT PRIMARY KEY,
                    activated_at TEXT NOT NULL,
                    article_id_floor INTEGER NOT NULL CHECK(article_id_floor >= 0)
                )"""
            )
            activated = datetime.now(UTC)
            floor = int(db.execute(
                "SELECT COALESCE(MAX(id), 0) FROM articles"
            ).fetchone()[0])
            inserted = db.execute(
                """INSERT OR IGNORE INTO event_extractor_rollout
                   (extractor_version, activated_at, article_id_floor)
                   VALUES (?, ?, ?)""",
                (V3_PRODUCTION_VERSION, activated.isoformat(), floor),
            ).rowcount
            row = db.execute(
                """SELECT activated_at, article_id_floor
                   FROM event_extractor_rollout
                   WHERE extractor_version = ?""",
                (V3_PRODUCTION_VERSION,),
            ).fetchone()
        if row is None:
            raise ValueError("Missing v3 rollout fence")
        try:
            cutoff = datetime.fromisoformat(row[0])
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid v3 rollout timestamp") from exc
        if cutoff.utcoffset() is None or cutoff > datetime.now(UTC):
            raise ValueError("Invalid v3 rollout cutoff")
        if not isinstance(row[1], int) or row[1] < 0 or row[1] > floor:
            raise ValueError("Invalid v3 rollout article floor")
        return cutoff.astimezone(UTC), row[1], bool(inserted)

    def process_pending(
        self, batch_size: int = 1000, *,
        received_since: datetime | None = None,
        received_until: datetime | None = None,
        extractor_version: str = EXTRACTOR_VERSION,
        after_article_id: int | None = None,
    ) -> ProcessingStats:
        if not 1 <= batch_size <= 10000:
            raise ValueError("batch_size must be 1..10000")
        if extractor_version not in {EXTRACTOR_VERSION, V3_PRODUCTION_VERSION}:
            raise ValueError("Unknown extractor version")
        if extractor_version == V3_PRODUCTION_VERSION and (
            after_article_id is None or after_article_id < 0
            or received_since is None
        ):
            raise ValueError("v3 requires explicit current-run article floor and receipt cutoff")
        if extractor_version == EXTRACTOR_VERSION and after_article_id is not None:
            raise ValueError("Legacy v2 does not use a fresh-only article floor")
        for bound in (received_since, received_until):
            if bound is not None and bound.utcoffset() is None:
                raise ValueError("Event extraction windows require timezone-aware datetimes")
        if (
            received_since is not None and received_until is not None
            and received_since > received_until
        ):
            raise ValueError("Event extraction window is reversed")
        self.initialize()
        processed = 0
        created = 0
        verified = 0
        with self.store.connect() as db:
            # v3 never replays an article already checkpointed by ANY version,
            # and never processes articles present before this integrated run.
            # No old v2 event receives a second derived row.
            representative: dict[tuple[str, str], int] = {}
            if extractor_version == V3_PRODUCTION_VERSION:
                existing = db.execute(
                    "SELECT id, url, title FROM articles ORDER BY id"
                ).fetchall()
                if len(existing) > 50000:
                    raise ValueError("v3 representative scan exceeds safe row bound")
                for existing_id, existing_url, existing_title in existing:
                    try:
                        key = (canonical_url(existing_url),
                               existing_title.strip().casefold())
                    except ValueError:
                        key = (f"invalid:{existing_id}", "")
                    representative.setdefault(key, existing_id)
            rows = db.execute(
                """
                SELECT a.id, a.source, a.url, a.title, a.summary, a.published_at,
                       a.detected_at, c.sha256
                FROM articles a
                JOIN captures c ON a.capture_id = c.id
                WHERE NOT EXISTS (
                    SELECT 1 FROM extraction_runs r
                    WHERE r.article_id = a.id
                      AND (? = ? OR r.extractor_version = ?)
                )
                  AND (? IS NULL OR a.id > ?)
                  AND (? IS NULL OR julianday(a.detected_at) >= julianday(?))
                  AND (? IS NULL OR julianday(a.detected_at) <= julianday(?))
                ORDER BY a.id ASC LIMIT ?
                """,
                (
                    extractor_version,
                    V3_PRODUCTION_VERSION,
                    extractor_version,
                    after_article_id,
                    after_article_id,
                    received_since.isoformat() if received_since else None,
                    received_since.isoformat() if received_since else None,
                    received_until.isoformat() if received_until else None,
                    received_until.isoformat() if received_until else None,
                    batch_size,
                ),
            ).fetchall()
            for article_id, source, url, title, summary, published, detected, sha in rows:
                if extractor_version == V3_PRODUCTION_VERSION:
                    try:
                        key = (canonical_url(url), title.strip().casefold())
                    except ValueError:
                        key = (f"invalid:{article_id}", "")
                    is_representative = representative.get(key) == article_id
                    classification = (
                        preview_news_event_v3(source, title, summary)
                        if is_representative else None
                    )
                    if classification is not None:
                        classification = replace(
                            classification, event=replace(
                                classification.event,
                                extractor_version=V3_PRODUCTION_VERSION,
                            ),
                        )
                else:
                    classification = extract_news_event(source, title, summary)
                event = classification.event if classification else None
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
                        if cursor.rowcount and classification is not None:
                            db.execute(
                                """INSERT INTO event_metadata (
                                    event_id, claim_kind, tags_json, verification_note
                                ) VALUES (?, ?, ?, ?)""",
                                (cursor.lastrowid, classification.claim_kind,
                                 json.dumps(classification.tags, ensure_ascii=False),
                                 classification.verification_note),
                            )
                        created += cursor.rowcount
                        if event.status == "VERIFIED" and cursor.rowcount:
                            verified += 1
                    db.execute(
                        "INSERT OR IGNORE INTO extraction_runs VALUES (?, ?, ?)",
                        (article_id, extractor_version, now),
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
