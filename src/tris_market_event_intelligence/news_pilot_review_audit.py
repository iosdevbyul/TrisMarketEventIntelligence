"""Read-only audit and status-filtered queue for the isolated news research journal.

This audits local consistency, not cryptographic authenticity, external truth,
or authority to publish. It never modifies either SQLite database.
"""
from __future__ import annotations

import hashlib
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Literal

from tris_market_event_intelligence.news_pilot_review_ledger import (
    DECISIONS,
    NewsPilotReviewLedger,
)

STATES = frozenset({"REQUIRES_HUMAN_REVIEW", *DECISIONS})
AuditStatus = Literal["REQUIRES_HUMAN_REVIEW", "APPROVED_FOR_RESEARCH", "REJECTED"]


class NewsPilotReviewAuditor:
    def __init__(self, *, source_db: Path, journal_db: Path) -> None:
        self.ledger = NewsPilotReviewLedger(source_db=source_db, journal_db=journal_db)

    def audit(
        self, *, status: AuditStatus | None = None, after_id: int = 0,
        limit: int = 100,
    ) -> dict[str, object]:
        """Audit every archived row before returning any subset.

        Fail closed if there is orphaned review data, changed evidence,
        inconsistent review hashes, invalid review metadata, or extra
        decision rows. Counts cover all rows, not only the paged subset.
        """
        if status is not None and status not in STATES:
            raise ValueError("Unsupported audit status filter")
        if type(after_id) is not int or after_id < 0:
            raise ValueError("Audit cursor must be nonnegative")
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("Audit limit must be 1..500")

        with closing(self.ledger._journal(readonly=True)) as db:
            has_reviews = db.execute(
                """SELECT 1 FROM sqlite_master
                   WHERE type='table' AND name='news_pilot_review_decisions'"""
            ).fetchone() is not None
            if has_reviews:
                # Verify the review table schema is not being used to hide
                # decisions that no longer point to source evidence.
                bad = db.execute(
                    """SELECT COUNT(*) FROM news_pilot_review_decisions d
                       LEFT JOIN news_pilot_evidence e ON e.id=d.evidence_id
                       WHERE e.id IS NULL"""
                ).fetchone()[0]
                if bad:
                    raise ValueError("Orphaned research review decisions")
                rows = db.execute(
                    """SELECT e.*, d.id AS review_id, d.decision,
                              d.reviewer, d.reason, d.reviewed_at,
                              d.signal_sha256 AS review_signal_sha256,
                              d.source_evidence_sha256 AS review_source_sha256
                       FROM news_pilot_evidence e
                       LEFT JOIN news_pilot_review_decisions d
                       ON d.evidence_id=e.id ORDER BY e.id"""
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM news_pilot_evidence ORDER BY id"
                ).fetchall()

            totals: Counter[str] = Counter({state: 0 for state in STATES})
            items: list[dict[str, object]] = []
            review_count = 0
            for row in rows:
                signal = self.ledger._verify(row)
                assessment = signal.evidence_assessment
                if row["review_status"] != "REQUIRES_HUMAN_REVIEW":
                    raise ValueError("Original review state was modified")
                resolved = "REQUIRES_HUMAN_REVIEW"
                if has_reviews and row["review_id"] is not None:
                    resolved = row["decision"]
                    if resolved not in DECISIONS:
                        raise ValueError("Unknown research review decision")
                    if (
                        row["review_signal_sha256"] != row["signal_sha256"]
                        or row["review_source_sha256"]
                        != row["source_evidence_sha256"]
                    ):
                        raise ValueError("Review decision evidence hash mismatch")
                    if (
                        not isinstance(row["reviewer"], str)
                        or not 0 < len(row["reviewer"].strip()) <= 100
                        or not isinstance(row["reason"], str)
                        or not 15 <= len(row["reason"].strip()) <= 1000
                    ):
                        raise ValueError("Incomplete editorial review metadata")
                    try:
                        dt = datetime.fromisoformat(row["reviewed_at"])
                    except (TypeError, ValueError) as exc:
                        raise ValueError("Invalid review timestamp") from exc
                    if dt.tzinfo is None:
                        raise ValueError("Missing review timestamp timezone")
                    review_count += 1
                totals[resolved] += 1
                if row["id"] <= after_id or (status is not None and resolved != status):
                    continue
                entry: dict[str, object] = {
                    "id": row["id"],
                    "event_id": row["event_id"],
                    "review_status": resolved,
                    "direction": signal.direction,
                    "intensity": signal.intensity,
                    "intensity_score": signal.intensity_score,
                    "uncertainty": signal.uncertainty,
                    "fact_status": assessment.fact_status,
                    "strength_basis": assessment.strength_basis,
                    "source_url": signal.source_url,
                    "signal_sha256": row["signal_sha256"],
                    "report_sha256": row["report_sha256"],
                    "published": False,
                }
                if resolved in DECISIONS:
                    entry["reviewer"] = row["reviewer"]
                    entry["reviewed_at"] = row["reviewed_at"]
                    entry["review_reason"] = row["reason"]
                items.append(entry)

            if has_reviews:
                count = db.execute(
                    "SELECT COUNT(*) FROM news_pilot_review_decisions"
                ).fetchone()[0]
                if count != review_count:
                    raise ValueError("Duplicate or unexplained review ledger rows")
            # Integrity of content hashes is a local consistency check; an
            # actor who can modify both payload and hashes could still forge.
            digest = hashlib.sha256()
            for row in rows:
                digest.update(
                    f"{row['id']}:{row['signal_sha256']}:"
                    f"{row['source_evidence_sha256']}\n".encode()
                )
            filtered = items[:limit]
            return {
                "schemaVersion": "news_pilot_review_audit_v1",
                "status": "LOCAL_CONSISTENCY_VERIFIED",
                "statusFilter": status or "ALL",
                "counts": dict(sorted(totals.items())),
                "totalEvidence": len(rows),
                "reviewDecisions": review_count,
                "items": filtered,
                "nextCursor": filtered[-1]["id"] if filtered else after_id,
                "hasMore": len(items) > limit,
                "localEvidenceDigestSha256": digest.hexdigest(),
                "reportIsCryptographicallyAuthentic": False,
                "independentSourceVerification": False,
                "published": False,
                "productionDbModified": False,
                "journalModified": False,
            }
