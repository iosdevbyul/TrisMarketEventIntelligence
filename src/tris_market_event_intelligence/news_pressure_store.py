"""Explicit, append-only persistence of evidence-bound news pressure signals.

This is an opt-in write interface. It never starts a model or modifies the
existing market_impacts checkpoint and does not predict stock prices.
"""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.news_impact_signal import NewsImpactSignal

SCHEMA = "news_impact_signal_v1"


class NewsPressureStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("Existing non-symlink news database required")
        mode = "ro" if readonly else "rw"
        db = sqlite3.connect(self.path.resolve().as_uri() + f"?mode={mode}",
                             uri=True, timeout=10)
        db.execute("PRAGMA foreign_keys=ON")
        if readonly:
            db.execute("PRAGMA query_only=ON")
        return db

    def initialize(self) -> None:
        """Explicit opt-in migration; never run from the existing 15m job."""
        with closing(self._connect()) as db, db:
            required = {"economic_events", "market_impacts"}
            found = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            if not required.issubset(found):
                raise ValueError("Expected initialized news and impact tables")
            db.execute(
                """CREATE TABLE IF NOT EXISTS news_pressure_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    impact_id INTEGER NOT NULL REFERENCES market_impacts(id),
                    event_id INTEGER NOT NULL REFERENCES economic_events(id),
                    pressure_prompt_version TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    UNIQUE(impact_id, pressure_prompt_version)
                )"""
            )

    def save(self, *, impact_id: int, signal: NewsImpactSignal) -> dict[str, object]:
        """Accept only signals linked to an existing, matching impact row."""
        if type(impact_id) is not int or impact_id < 1:
            raise ValueError("Positive source impact ID required")
        signal = NewsImpactSignal.model_validate(signal)
        if signal.prompt_version != "news-pressure-intensity-v1":
            raise ValueError("Only opt-in news-pressure assessment versions allowed")
        raw = signal.model_dump_json()
        fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        with closing(self._connect()) as db, db:
            row = db.execute(
                """SELECT i.event_id, i.market, i.sector, i.direction, i.horizon,
                          i.uncertainty, i.evidence_excerpt, e.source_url
                   FROM market_impacts i
                   JOIN economic_events e ON e.id=i.event_id
                   WHERE i.id=?""", (impact_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Source impact not found")
            (
                event_id, market, sector, direction, horizon, uncertainty,
                excerpt, source_url,
            ) = row
            if (
                signal.event_id != event_id or signal.market != market
                or signal.sector != sector or signal.direction != direction
                or signal.horizon != horizon or signal.uncertainty != uncertainty
                or signal.evidence_excerpt != excerpt or signal.source_url != source_url
            ):
                raise ValueError("Signal does not match persisted impact provenance")
            previous = db.execute(
                """SELECT id, payload_sha256, payload_json
                   FROM news_pressure_signals
                   WHERE impact_id=? AND pressure_prompt_version=?""",
                (impact_id, signal.prompt_version),
            ).fetchone()
            if previous is not None:
                if previous[1:] != (fingerprint, raw):
                    raise ValueError("Immutable news pressure assessment conflict")
                return {"id": previous[0], "inserted": False,
                        "payload_sha256": fingerprint}
            cursor = db.execute(
                """INSERT INTO news_pressure_signals(
                    impact_id, event_id, pressure_prompt_version,
                    payload_json, payload_sha256
                ) VALUES (?, ?, ?, ?, ?)""",
                (impact_id, event_id, signal.prompt_version, raw, fingerprint),
            )
            return {"id": cursor.lastrowid, "inserted": True,
                    "payload_sha256": fingerprint}

    def list_signals(self, *, after_id: int = 0, limit: int = 100) -> dict[str, object]:
        if type(after_id) is not int or after_id < 0 or (
            type(limit) is not int or not 1 <= limit <= 500
        ):
            raise ValueError("Invalid cursor or limit")
        with closing(self._connect(readonly=True)) as db:
            rows = db.execute(
                """SELECT id, impact_id, event_id, payload_json, payload_sha256
                   FROM news_pressure_signals WHERE id > ?
                   ORDER BY id LIMIT ?""", (after_id, limit + 1),
            ).fetchall()
        items = []
        for row_id, impact_id, event_id, raw, stored_sha in rows[:limit]:
            if hashlib.sha256(raw.encode("utf-8")).hexdigest() != stored_sha:
                raise ValueError("News pressure record integrity mismatch")
            signal = NewsImpactSignal.model_validate_json(raw)
            if signal.event_id != event_id:
                raise ValueError("News pressure record event mismatch")
            items.append({
                "id": row_id, "sourceImpactId": impact_id,
                **signal.model_dump(mode="json"),
            })
        return {
            "schemaVersion": SCHEMA,
            "items": items,
            "nextCursor": items[-1]["id"] if items else after_id,
            "hasMore": len(rows) > limit,
        }
