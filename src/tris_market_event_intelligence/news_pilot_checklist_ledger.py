"""Manual checklist evidence trail for isolated news review only.

Entries are self-attested notes about material to check, not independent
verification. Never fetches URLs, publishes, or touches the operational DB.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from tris_market_event_intelligence.news_pilot_review_ledger import (
    NewsPilotReviewLedger,
)

STATES = frozenset({"SOURCE_DOCUMENTED", "NOT_CONFIRMED", "CONFLICTING_SOURCE"})
TABLE = """CREATE TABLE IF NOT EXISTS news_pilot_checklist_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    evidence_id INTEGER NOT NULL REFERENCES news_pilot_evidence(id),
    check_code TEXT NOT NULL,
    review_status TEXT NOT NULL CHECK(review_status IN (
        'SOURCE_DOCUMENTED', 'NOT_CONFIRMED', 'CONFLICTING_SOURCE'
    )),
    source_url TEXT,
    source_excerpt TEXT,
    reviewer TEXT NOT NULL,
    note TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    signal_sha256 TEXT NOT NULL,
    source_evidence_sha256 TEXT NOT NULL,
    UNIQUE(evidence_id, check_code)
)"""


def _source_url(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 1000:
        raise ValueError("Evidence source URL invalid")
    candidate = urlsplit(value)
    if (
        candidate.scheme != "https"
        or not candidate.hostname
        or candidate.username is not None
        or candidate.password is not None
    ):
        raise ValueError("Evidence source must be an HTTPS URL without credentials")
    return value


def validate_checklist_row(
    row: sqlite3.Row, *, signal_sha256: str,
    source_evidence_sha256: str, allowed_checks: tuple[str, ...],
) -> None:
    """Fail closed on malformed, orphaned, or mismatched review notes."""
    if (
        row["check_code"] not in allowed_checks
        or row["review_status"] not in STATES
        or row["signal_sha256"] != signal_sha256
        or row["source_evidence_sha256"] != source_evidence_sha256
        or not isinstance(row["reviewer"], str)
        or not 0 < len(row["reviewer"].strip()) <= 100
        or not isinstance(row["note"], str)
        or not 15 <= len(row["note"].strip()) <= 1000
    ):
        raise ValueError("Invalid independent-check review provenance")
    try:
        date = datetime.fromisoformat(row["recorded_at"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid checklist review timestamp") from exc
    if date.tzinfo is None:
        raise ValueError("Checklist review timestamp must be timezone aware")
    url = _source_url(row["source_url"])
    excerpt = row["source_excerpt"]
    if row["review_status"] in {"SOURCE_DOCUMENTED", "CONFLICTING_SOURCE"}:
        if not url or not isinstance(excerpt, str) or not 8 <= len(excerpt.strip()) <= 1000:
            raise ValueError("Source documentation requires URL and cited excerpt")
    elif url is not None or excerpt is not None:
        raise ValueError("NOT_CONFIRMED must not imply evidence sources")


class NewsPilotChecklistLedger:
    """Human attestation records in the separate journal, never production."""

    def __init__(self, *, source_db: Path, journal_db: Path) -> None:
        self.ledger = NewsPilotReviewLedger(
            source_db=source_db, journal_db=journal_db,
        )

    def record(
        self, *, evidence_id: int, check_code: str, review_status: str,
        reviewer: str, note: str, source_url: str | None = None,
        source_excerpt: str | None = None, commit: bool = False,
    ) -> dict[str, object]:
        if type(evidence_id) is not int or evidence_id <= 0:
            raise ValueError("Positive journal evidence ID required")
        if review_status not in STATES:
            raise ValueError("Unknown checklist review status")
        if not isinstance(reviewer, str) or not isinstance(note, str):
            raise ValueError("Reviewer and note required")
        reviewer, note = reviewer.strip(), note.strip()
        if not 0 < len(reviewer) <= 100 or not 15 <= len(note) <= 1000:
            raise ValueError("Reviewer or note invalid")
        url = _source_url(source_url)
        if review_status in {"SOURCE_DOCUMENTED", "CONFLICTING_SOURCE"}:
            if (
                not url or not isinstance(source_excerpt, str)
                or not 8 <= len(source_excerpt.strip()) <= 1000
            ):
                raise ValueError("Documented source requires HTTPS URL and source excerpt")
            source_excerpt = source_excerpt.strip()
        elif url is not None or source_excerpt is not None:
            raise ValueError("NOT_CONFIRMED cannot have a cited source")

        with closing(self.ledger._journal(readonly=not commit)) as db:
            if commit:
                db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(
                    "SELECT * FROM news_pilot_evidence WHERE id=?",
                    (evidence_id,),
                ).fetchone()
                if row is None:
                    raise ValueError("Journal evidence entry not found")
                signal = self.ledger._verify(row)
                checks = signal.evidence_assessment.verification_needed
                if check_code not in checks:
                    raise ValueError("Unknown check code for this news event")
                exists = db.execute(
                    """SELECT 1 FROM sqlite_master WHERE type='table'
                       AND name='news_pilot_checklist_reviews'"""
                ).fetchone()
                existing = (
                    db.execute(
                        """SELECT * FROM news_pilot_checklist_reviews
                           WHERE evidence_id=? AND check_code=?""",
                        (evidence_id, check_code),
                    ).fetchone()
                    if exists else None
                )
                if existing:
                    validate_checklist_row(
                        existing, signal_sha256=row["signal_sha256"],
                        source_evidence_sha256=row["source_evidence_sha256"],
                        allowed_checks=checks,
                    )
                    if (
                        existing["review_status"], existing["reviewer"],
                        existing["note"], existing["source_url"],
                        existing["source_excerpt"],
                    ) != (review_status, reviewer, note, url, source_excerpt):
                        raise ValueError("Immutable checklist review conflict")
                    return {
                        "status": "ALREADY_RECORDED", "journalEvidenceId": evidence_id,
                        "checkCode": check_code, "reviewStatus": review_status,
                        "published": False, "productionDbModified": False,
                    }
                if not commit:
                    return {
                        "status": "CHECK_PREVIEW_ONLY", "journalEvidenceId": evidence_id,
                        "eventId": signal.event_id, "checkCode": check_code,
                        "reviewStatus": review_status, "sourceUrl": url,
                        "sourceExcerpt": source_excerpt, "reviewer": reviewer,
                        "note": note, "independentlyVerified": False,
                        "published": False, "journalModified": False,
                        "productionDbModified": False,
                    }
                db.execute(TABLE)
                timestamp = datetime.now(UTC).isoformat()
                db.execute(
                    """INSERT INTO news_pilot_checklist_reviews (
                        evidence_id,check_code,review_status,source_url,
                        source_excerpt,reviewer,note,recorded_at,
                        signal_sha256,source_evidence_sha256
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        evidence_id, check_code, review_status, url,
                        source_excerpt, reviewer, note, timestamp,
                        row["signal_sha256"], row["source_evidence_sha256"],
                    ),
                )
                db.commit()
            except Exception:
                if commit:
                    db.rollback()
                raise
        return {
            "status": "CHECK_RECORDED", "journalEvidenceId": evidence_id,
            "checkCode": check_code, "reviewStatus": review_status,
            "recordedAt": timestamp, "independentlyVerified": False,
            "published": False, "productionDbModified": False,
        }
