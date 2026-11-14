"""Read-only research review packet showing the complete provenance and caveats.

The source is reported text; a validated quotation does NOT independently
verify the story, market transmission, or numeric pressure magnitude.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.impact_semantic_review import semantic_review_packet
from tris_market_event_intelligence.news_pilot_checklist_ledger import (
    validate_checklist_row,
)
from tris_market_event_intelligence.news_pilot_review_audit import NewsPilotReviewAuditor
from tris_market_event_intelligence.news_pilot_review_ledger import NewsPilotReviewLedger


class NewsPilotReviewPacket:
    def __init__(self, *, source_db: Path, journal_db: Path) -> None:
        self.ledger = NewsPilotReviewLedger(
            source_db=source_db, journal_db=journal_db,
        )
        self.auditor = NewsPilotReviewAuditor(
            source_db=source_db, journal_db=journal_db,
        )

    def build(self, *, evidence_id: int) -> dict[str, object]:
        if type(evidence_id) is not int or evidence_id < 1:
            raise ValueError("Positive journal evidence ID required")

        # Audit ALL entries, not just the requested item. A broken history
        # elsewhere must not appear safe because a user selected this ID.
        audit = self.auditor.audit(after_id=0, limit=500)
        if audit["status"] != "LOCAL_CONSISTENCY_VERIFIED":
            raise ValueError("Review journal audit failed")
        with closing(self.ledger._journal(readonly=True)) as db:
            row = db.execute(
                "SELECT * FROM news_pilot_evidence WHERE id=?",
                (evidence_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Journal evidence entry not found")
            signal = self.ledger._verify(row)
            has_decisions = db.execute(
                """SELECT 1 FROM sqlite_master WHERE type='table'
                   AND name='news_pilot_review_decisions'"""
            ).fetchone() is not None
            decision = (
                db.execute(
                    """SELECT decision, reviewer, reason, reviewed_at,
                              signal_sha256, source_evidence_sha256
                       FROM news_pilot_review_decisions
                       WHERE evidence_id=?""",
                    (evidence_id,),
                ).fetchone() if has_decisions else None
            )
            if decision and (
                decision["signal_sha256"] != row["signal_sha256"]
                or decision["source_evidence_sha256"] != row["source_evidence_sha256"]
            ):
                raise ValueError("Review decision provenance mismatch")

        with closing(self.ledger._journal(readonly=True)) as db:
            has_checklist = db.execute(
                """SELECT 1 FROM sqlite_master WHERE type='table'
                   AND name='news_pilot_checklist_reviews'"""
            ).fetchone() is not None
            recorded = (
                db.execute(
                    """SELECT * FROM news_pilot_checklist_reviews
                       WHERE evidence_id=?""", (evidence_id,),
                ).fetchall() if has_checklist else []
            )
            reviewed_checks: dict[str, dict[str, object]] = {}
            for entry in recorded:
                validate_checklist_row(
                    entry, signal_sha256=row["signal_sha256"],
                    source_evidence_sha256=row["source_evidence_sha256"],
                    allowed_checks=signal.evidence_assessment.verification_needed,
                )
                reviewed_checks[entry["check_code"]] = {
                    "reviewStatus": entry["review_status"],
                    "reviewer": entry["reviewer"],
                    "note": entry["note"],
                    "sourceUrl": entry["source_url"],
                    "sourceExcerpt": entry["source_excerpt"],
                    "recordedAt": entry["recorded_at"],
                    "independentlyVerified": False,
                }

        source_uri = self.ledger.source_db.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(
            source_uri, uri=True, timeout=5,
        )) as src:
            src.execute("PRAGMA query_only=ON")
            source = src.execute(
                """SELECT evidence, source_url, status, classification
                   FROM economic_events WHERE id=?""",
                (signal.event_id,),
            ).fetchone()
        if source is None:
            raise ValueError("Original source event missing")
        evidence, url, fact_status, classification = source
        assessment = signal.evidence_assessment
        if signal.source_url != url or assessment.fact_status != fact_status:
            raise ValueError("Source and signal mismatch")
        if assessment.reported_fact not in evidence:
            raise ValueError("Source quotation not in original evidence")
        selected = signal.evidence_excerpt
        offset = evidence.find(selected)
        if offset < 0:
            raise ValueError("Evidence quote not in original source")
        # Only a bounded source excerpt around the literal quote is displayed,
        # so a large original report does not flood a terminal or audit packet.
        before = max(0, offset - 300)
        after = min(len(evidence), offset + len(selected) + 300)
        review_context = evidence[before:after]
        semantic_review = semantic_review_packet(
            source_text=review_context,
            evidence_excerpt=selected,
            rationale=assessment.conditional_mechanism,
        )
        return {
            "schemaVersion": "news_pilot_review_packet_v1",
            "auditStatus": audit["status"],
            "journalEvidenceId": evidence_id,
            "eventId": signal.event_id,
            "reviewStatus": decision["decision"] if decision else "REQUIRES_HUMAN_REVIEW",
            "source": {
                "url": url,
                "classification": classification,
                "factStatus": fact_status,
                "reportedEvidenceContext": review_context,
                "contextTruncated": before > 0 or after < len(evidence),
                "literalEvidenceQuote": selected,
            },
            "hypothesis": {
                "market": signal.market,
                "sector": signal.sector,
                "direction": signal.direction,
                "horizon": signal.horizon,
                "conditionalMechanism": assessment.conditional_mechanism,
                "modelRationale": signal.rationale,
                "intensity": signal.intensity,
                "intensityScore": signal.intensity_score,
                "uncertainty": signal.uncertainty,
                "strengthBasis": assessment.strength_basis,
                "scoreIsCalibratedToReturns": False,
            },
            "semanticContextReview": {
                **semantic_review,
                "scope": "BOUNDED_ORIGINAL_SOURCE_CONTEXT",
                "reviewTarget": "CONDITIONAL_MECHANISM",
                "sourceContextTruncated": before > 0 or after < len(evidence),
                "published": False,
            },
            "independentVerificationChecklist": [
                {"check": check, "verified": False, "manualEvidence": reviewed_checks.get(check)}
                for check in assessment.verification_needed
            ],
            "reviewDecision": (
                {
                    "decision": decision["decision"],
                    "reviewer": decision["reviewer"],
                    "reason": decision["reason"],
                    "reviewedAt": decision["reviewed_at"],
                } if decision else None
            ),
            "provenance": {
                "reportSha256": row["report_sha256"],
                "signalSha256": row["signal_sha256"],
                "sourceEvidenceSha256": row["source_evidence_sha256"],
            },
            "independentSourceVerification": False,
            "researchApprovalIsPublication": False,
            "published": False,
            "productionDbModified": False,
            "journalModified": False,
        }
