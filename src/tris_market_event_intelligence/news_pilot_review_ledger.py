"""Manual, append-only editorial decisions for isolated news pilot evidence.

Approval is a human *research review* only. Never publishes a signal, verifies
the reported event externally, edits the source DB, or invokes an AI model.
"""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.news_impact_signal import NewsImpactSignal

SCHEMA = "news_pilot_review_ledger_v1"
DECISIONS = {"APPROVED_FOR_RESEARCH", "REJECTED"}
TABLE = """CREATE TABLE IF NOT EXISTS news_pilot_review_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evidence_id INTEGER NOT NULL UNIQUE REFERENCES news_pilot_evidence(id),
    signal_sha256 TEXT NOT NULL,
    source_evidence_sha256 TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('APPROVED_FOR_RESEARCH','REJECTED')),
    reviewer TEXT NOT NULL,
    reason TEXT NOT NULL,
    reviewed_at TEXT NOT NULL
)"""


class NewsPilotReviewLedger:
    """Uses only the separate journal for writes and the source DB in read-only mode."""

    def __init__(self, *, source_db: Path, journal_db: Path) -> None:
        self.source_db = source_db.expanduser()
        self.journal_db = journal_db.expanduser()
        if self.source_db.resolve() == self.journal_db.resolve():
            raise ValueError("Review journal cannot be the production source database")

    def _journal(self, *, readonly: bool) -> sqlite3.Connection:
        path = self.journal_db
        if path.is_symlink() or not path.is_file():
            raise ValueError("Existing non-symlink evidence journal required")
        if readonly:
            uri = path.resolve().as_uri() + "?mode=ro"
        else:
            uri = path.resolve().as_uri() + "?mode=rw"
        db = sqlite3.connect(uri, uri=True, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        if readonly:
            db.execute("PRAGMA query_only=ON")
        required = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "news_pilot_evidence" not in required:
            db.close()
            raise ValueError("Not an initialized news evidence journal")
        return db

    def _verify(self, row: sqlite3.Row) -> NewsImpactSignal:
        raw = row["signal_json"]
        signal_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if signal_sha != row["signal_sha256"]:
            raise ValueError("Journal signal integrity mismatch")
        signal = NewsImpactSignal.model_validate_json(raw)
        if signal.event_id != row["event_id"] or signal.evidence_assessment is None:
            raise ValueError("Missing signal provenance")
        if (
            self.source_db.is_symlink()
            or not self.source_db.is_file()
        ):
            raise ValueError("Existing non-symlink original source database required")
        uri = self.source_db.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=5)) as src:
            src.execute("PRAGMA query_only=ON")
            source = src.execute(
                "SELECT evidence, source_url, status FROM economic_events WHERE id=?",
                (row["event_id"],),
            ).fetchone()
        if source is None:
            raise ValueError("Original source event no longer exists")
        evidence, url, status = source
        if (
            hashlib.sha256(evidence.encode("utf-8")).hexdigest()
            != row["source_evidence_sha256"]
            or signal.evidence_excerpt not in evidence
            or signal.source_url != url
            or signal.evidence_assessment.fact_status != status
        ):
            raise ValueError("Original event evidence changed; review prohibited")
        return signal

    def list_entries(self, *, limit: int = 100) -> dict[str, object]:
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("Review list limit must be 1..500")
        with closing(self._journal(readonly=True)) as db:
            has_decisions = bool(db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='news_pilot_review_decisions'"
            ).fetchone())
            base = """SELECT e.id, e.event_id, e.report_sha256,
                       e.signal_json, e.signal_sha256, e.source_evidence_sha256"""
            if has_decisions:
                rows = db.execute(base + """,
                       d.decision, d.reviewer, d.reason, d.reviewed_at
                       FROM news_pilot_evidence e
                       LEFT JOIN news_pilot_review_decisions d ON d.evidence_id=e.id
                       ORDER BY e.id LIMIT ?""", (limit,)).fetchall()
            else:
                rows = db.execute(base + """
                       FROM news_pilot_evidence e
                       ORDER BY e.id LIMIT ?""", (limit,)).fetchall()
            items = []
            for row in rows:
                signal = self._verify(row)
                item = {
                    "id": row["id"],
                    "event_id": row["event_id"],
                    "report_sha256": row["report_sha256"],
                    "review_status": (
                        row["decision"] if has_decisions and row["decision"]
                        else "REQUIRES_HUMAN_REVIEW"
                    ),
                    "direction": signal.direction,
                    "intensity": signal.intensity,
                    "intensity_score": signal.intensity_score,
                    "uncertainty": signal.uncertainty,
                    "fact_status": signal.evidence_assessment.fact_status,
                    "source_url": signal.source_url,
                    "published": False,
                }
                if has_decisions and row["decision"]:
                    item["reviewer"] = row["reviewer"]
                    item["reason"] = row["reason"]
                    item["reviewed_at"] = row["reviewed_at"]
                items.append(item)
        return {"schemaVersion": SCHEMA, "items": items, "productionDbModified": False}

    def decide(
        self, *, evidence_id: int, decision: str,
        reviewer: str, reason: str, commit: bool = False,
    ) -> dict[str, object]:
        if type(evidence_id) is not int or evidence_id < 1:
            raise ValueError("Positive evidence ID required")
        if decision not in DECISIONS:
            raise ValueError("Unsupported editorial decision")
        if (
            not reviewer.strip() or len(reviewer) > 100
            or not reason.strip() or not 15 <= len(reason.strip()) <= 1000
        ):
            raise ValueError("Reviewer and substantive review reason required")
        reviewer, reason = reviewer.strip(), reason.strip()
        with closing(self._journal(readonly=not commit)) as db:
            if commit:
                db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(
                    "SELECT * FROM news_pilot_evidence WHERE id=?",
                    (evidence_id,),
                ).fetchone()
                if row is None:
                    raise ValueError("Evidence entry not found")
                signal = self._verify(row)
                has_decisions = bool(db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' "
                    "AND name='news_pilot_review_decisions'"
                ).fetchone())
                existing = (
                    db.execute(
                        "SELECT decision, reviewer, reason, reviewed_at "
                        "FROM news_pilot_review_decisions WHERE evidence_id=?",
                        (evidence_id,),
                    ).fetchone() if has_decisions else None
                )
                if existing:
                    if (existing["decision"], existing["reviewer"], existing["reason"]) != (
                        decision, reviewer, reason
                    ):
                        raise ValueError("Immutable review decision already recorded")
                    return {
                        "status": "ALREADY_REVIEWED",
                        "id": evidence_id,
                        "decision": decision,
                        "reviewed_at": existing["reviewed_at"],
                        "published": False,
                        "productionDbModified": False,
                    }
                if not commit:
                    return {
                        "status": "REVIEW_PREVIEW_ONLY",
                        "id": evidence_id,
                        "event_id": signal.event_id,
                        "decision": decision,
                        "fact_status": signal.evidence_assessment.fact_status,
                        "reviewer": reviewer,
                        "reason": reason,
                        "published": False,
                        "journalModified": False,
                        "productionDbModified": False,
                    }
                db.execute(TABLE)
                now = datetime.now(UTC).isoformat()
                db.execute(
                    """INSERT INTO news_pilot_review_decisions (
                        evidence_id, signal_sha256, source_evidence_sha256,
                        decision, reviewer, reason, reviewed_at
                    ) VALUES (?,?,?,?,?,?,?)""",
                    (
                        evidence_id, row["signal_sha256"],
                        row["source_evidence_sha256"], decision,
                        reviewer, reason, now,
                    ),
                )
                db.commit()
            except Exception:
                if commit:
                    db.rollback()
                raise
        return {
            "status": "REVIEW_RECORDED",
            "id": evidence_id,
            "decision": decision,
            "reviewed_at": now,
            "published": False,
            "productionDbModified": False,
        }
