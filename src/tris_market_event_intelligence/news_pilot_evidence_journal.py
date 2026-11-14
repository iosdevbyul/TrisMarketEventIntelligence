"""Opt-in isolated journal for sourced two-stage news-only pilot results.

The production news database is always opened read-only. Successful pilot
outputs are archived in a completely separate SQLite file, never rewritten
into market_impacts or production checkpoints. No inference or price feeds.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from pydantic import ValidationError

from tris_market_event_intelligence.news_impact_signal import NewsImpactSignal

JOURNAL_SCHEMA = "news_pilot_evidence_journal_v1"
REPORT_SCHEMA = "news_two_stage_pilot_v1"
PROMPT_VERSION = "news-pressure-intensity-v1"


def _fingerprint(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_report(path: Path) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Existing non-symlink pilot report required")
    raw = path.read_bytes()
    if not raw or len(raw) > 2_000_000:
        raise ValueError("Pilot report size invalid")
    report = json.loads(raw)
    if not isinstance(report, dict) or report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError("Unsupported pilot report schema")
    return report, _fingerprint(raw)


def _source_connection(path: Path) -> sqlite3.Connection:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Existing non-symlink source DB required")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    return db


def validate_pilot_report(
    *, source_db: Path, report_path: Path,
) -> dict[str, object]:
    """Validate stored news provenance, never claim external fact verification."""
    report, report_sha = _read_report(report_path)
    items = report.get("items")
    if (
        not isinstance(items, list) or not 1 <= len(items) <= 3
        or report.get("selected_events") != len(items)
        or report.get("direction_model_calls") != len(items)
        or report.get("stock_returns_predicted") is not False
        or report.get("production_database_modified") is not False
        or report.get("operational_quality_verified") is not False
        or not isinstance(report.get("model_id"), str)
    ):
        raise ValueError("Incomplete or inconsistent pilot run metadata")
    accepted: list[dict[str, object]] = []
    seen: set[int] = set()
    with closing(_source_connection(source_db)) as db:
        for item in items:
            if not isinstance(item, dict) or type(item.get("event_id")) is not int:
                raise ValueError("Pilot event must be identified")
            event_id = item["event_id"]
            if event_id < 1 or event_id in seen:
                raise ValueError("Duplicate or invalid event ID")
            seen.add(event_id)
            row = db.execute(
                """SELECT e.id, e.classification, e.status, e.source_url,
                          e.evidence, m.claim_kind
                   FROM economic_events e
                   LEFT JOIN event_metadata m ON m.event_id=e.id
                   WHERE e.id=?""",
                (event_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Pilot event missing from source database")
            if (
                item.get("classification") != row["classification"]
                or item.get("event_status") != row["status"]
                or item.get("source_url") != row["source_url"]
            ):
                raise ValueError("Pilot event metadata mismatches saved source")
            if item.get("status") != "TWO_STAGE_OUTPUT_VALIDATED":
                if item.get("signal") is not None:
                    raise ValueError("Failed pilot entry cannot contain a signal")
                continue
            try:
                signal = NewsImpactSignal.model_validate(item["signal"])
            except (KeyError, ValueError, ValidationError) as exc:
                raise ValueError("Invalid pilot news signal") from exc
            assessment = signal.evidence_assessment
            if (
                signal.event_id != event_id
                or signal.source_url != row["source_url"]
                or signal.model_id != report["model_id"]
                or signal.prompt_version != PROMPT_VERSION
                or signal.direction == "UNCERTAIN"
                or assessment is None
                or assessment.fact_status != row["status"]
                or signal.evidence_excerpt not in row["evidence"]
                or assessment.reported_fact != signal.evidence_excerpt
                or not signal.rationale.startswith(
                    assessment.conditional_mechanism + " News pressure: "
                )
                or (
                    row["status"] != "VERIFIED" and signal.uncertainty != "HIGH"
                )
                or (
                    row["classification"] == "SERVICE_INCIDENT"
                    and row["status"] != "VERIFIED"
                    and signal.intensity == "HIGH"
                )
            ):
                raise ValueError("Pilot signal provenance or evidence mismatch")
            accepted.append({
                "event_id": event_id,
                "signal": signal.model_dump(mode="json"),
                "source_evidence_sha256": _fingerprint(
                    row["evidence"].encode("utf-8")
                ),
            })
    if report.get("strength_model_calls") != len(accepted) + sum(
        item.get("status") == "REJECTED_OR_FAILED"
        and item.get("failed_stage") == "INTENSITY" for item in items
    ):
        raise ValueError("Strength inference call accounting mismatch")
    return {
        "schemaVersion": JOURNAL_SCHEMA,
        "reportSha256": report_sha,
        "sourceDb": str(source_db.resolve()),
        "reviewStatus": "REQUIRES_HUMAN_REVIEW",
        "accepted": accepted,
        "acceptedCount": len(accepted),
        "rejectedOrUnassessedCount": len(items) - len(accepted),
        "productionDbModified": False,
    }


class NewsPilotEvidenceJournal:
    """Separate opt-in journal. This class cannot write into source_db."""

    def __init__(self, *, source_db: Path, journal_db: Path) -> None:
        if source_db.expanduser().resolve() == journal_db.expanduser().resolve():
            raise ValueError("Journal must be separate from production source DB")
        self.source_db = source_db.expanduser()
        self.journal_db = journal_db.expanduser()

    def import_report(
        self, report_path: Path, *, commit: bool = False,
    ) -> dict[str, object]:
        checked = validate_pilot_report(
            source_db=self.source_db, report_path=report_path.expanduser(),
        )
        if not commit:
            return {
                **{k: v for k, v in checked.items() if k != "accepted"},
                "status": "VALIDATED_PREVIEW_ONLY",
                "journalModified": False,
            }
        if self.journal_db.is_symlink():
            raise ValueError("Symlink journal prohibited")
        if not self.journal_db.parent.is_dir():
            raise ValueError("Existing journal parent directory required")
        if self.journal_db.exists() and not self.journal_db.is_file():
            raise ValueError("Journal path must be a regular file")
        # Exclusive create, so an accidental path to an existing unrelated
        # database is never silently modified.
        created = False
        if not self.journal_db.exists():
            descriptor = os.open(
                self.journal_db, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            os.close(descriptor)
            created = True
        try:
            with closing(sqlite3.connect(
                self.journal_db.resolve().as_uri() + "?mode=rw",
                uri=True, timeout=10,
            )) as db, db:
                db.execute("PRAGMA foreign_keys=ON")
                tables = {
                    row[0] for row in db.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                if tables and "news_pilot_evidence" not in tables:
                    raise ValueError("Journal points to an unrelated database")
                db.execute(
                    """CREATE TABLE IF NOT EXISTS news_pilot_evidence (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        report_sha256 TEXT NOT NULL,
                        event_id INTEGER NOT NULL,
                        source_evidence_sha256 TEXT NOT NULL,
                        signal_json TEXT NOT NULL,
                        signal_sha256 TEXT NOT NULL,
                        review_status TEXT NOT NULL
                            CHECK(review_status='REQUIRES_HUMAN_REVIEW'),
                        UNIQUE(report_sha256, event_id)
                    )"""
                )
                inserted = 0
                for entry in checked["accepted"]:
                    raw = json.dumps(
                        entry["signal"], sort_keys=True,
                        ensure_ascii=False, separators=(",", ":"),
                    )
                    existing = db.execute(
                        """SELECT signal_sha256, signal_json,
                                  source_evidence_sha256
                           FROM news_pilot_evidence
                           WHERE report_sha256=? AND event_id=?""",
                        (checked["reportSha256"], entry["event_id"]),
                    ).fetchone()
                    fingerprint = _fingerprint(raw.encode("utf-8"))
                    if existing:
                        if existing != (
                            fingerprint, raw, entry["source_evidence_sha256"]
                        ):
                            raise ValueError("Immutable evidence journal conflict")
                        continue
                    db.execute(
                        """INSERT INTO news_pilot_evidence(
                           report_sha256, event_id, source_evidence_sha256,
                           signal_json, signal_sha256, review_status
                        ) VALUES (?,?,?,?,?,?)""",
                        (checked["reportSha256"], entry["event_id"],
                         entry["source_evidence_sha256"], raw, fingerprint,
                         "REQUIRES_HUMAN_REVIEW"),
                    )
                    inserted += 1
        except Exception:
            if created and self.journal_db.exists():
                self.journal_db.unlink()
            raise
        return {
            "schemaVersion": JOURNAL_SCHEMA,
            "status": "ARCHIVED_FOR_REVIEW",
            "reportSha256": checked["reportSha256"],
            "acceptedCount": checked["acceptedCount"],
            "inserted": inserted,
            "reviewStatus": "REQUIRES_HUMAN_REVIEW",
            "productionDbModified": False,
        }
