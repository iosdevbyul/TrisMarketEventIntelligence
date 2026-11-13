"""Explicit, append-only review decisions with cited local article evidence.

No automated verification of underlying news claims is implied.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from tris_market_event_intelligence.claim_extraction import extract_claims
from tris_market_event_intelligence.claim_store import ClaimProcessor
from tris_market_event_intelligence.collector import utc_now

REVIEW_STATES = frozenset({
    "UNDER_REVIEW", "CORROBORATED", "DISPUTED", "RETRACTED", "CONFIRMED",
})
OFFICIAL_HOSTS: dict[str, str] = {
    "fed_monetary": "federalreserve.gov",
    "fed_all": "federalreserve.gov",
    "bok_monetary": "bok.or.kr",
    "bls_cpi": "bls.gov",
    "bls_ppi": "bls.gov",
    "bls_employment": "bls.gov",
    "sec_press": "sec.gov",
}
MAX_NOTE = 1000
MAX_QUOTE = 1000


class ReviewConflict(ValueError):
    """The caller's expected revision is stale."""


@dataclass(frozen=True)
class ReviewDecision:
    revision_id: int
    claim_id: int
    status: str
    prior_status: str
    evidence_article_id: int
    reviewer: str


class ClaimReviewManager:
    def __init__(self, database: Path) -> None:
        self.claims = ClaimProcessor(database)

    def initialize(self) -> None:
        self.claims.initialize()
        with self.claims.store.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS claim_review_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_id TEXT NOT NULL UNIQUE,
                    claim_id INTEGER NOT NULL REFERENCES article_event_claims(id),
                    revision INTEGER NOT NULL,
                    prior_status TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN (
                        'UNDER_REVIEW', 'CORROBORATED', 'DISPUTED',
                        'RETRACTED', 'CONFIRMED'
                    )),
                    evidence_article_id INTEGER NOT NULL REFERENCES articles(id),
                    evidence_quote TEXT NOT NULL,
                    evidence_kind TEXT NOT NULL CHECK(evidence_kind IN (
                        'SUPPORTING_REPORT', 'PRIMARY_DOCUMENT',
                        'CORRECTION', 'DENIAL'
                    )),
                    reviewer TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(claim_id, revision)
                );
                CREATE INDEX IF NOT EXISTS idx_review_claim
                    ON claim_review_history(claim_id, id);
                """
            )

    def record(
        self, *, claim_id: int, status: str, evidence_article_id: int,
        evidence_quote: str, evidence_kind: str, reason: str,
        expected_revision: int, reviewer: str = "local_operator",
        operation_id: str | None = None,
    ) -> ReviewDecision:
        """Persist a human decision with a literal excerpt from a stored article.

        Multiple publishers or an official feed do not automatically change status.
        CONFIRMED is possible only on explicit human review of an indexed primary
        document from a fixed, official-source-domain pair.
        """
        if status not in REVIEW_STATES:
            raise ValueError("Unknown review status")
        if evidence_kind not in {
            "SUPPORTING_REPORT", "PRIMARY_DOCUMENT", "CORRECTION", "DENIAL"
        }:
            raise ValueError("Unknown evidence kind")
        if not isinstance(claim_id, int) or claim_id < 1:
            raise ValueError("Invalid claim ID")
        if not isinstance(evidence_article_id, int) or evidence_article_id < 1:
            raise ValueError("Invalid evidence article ID")
        if not isinstance(expected_revision, int) or expected_revision < 0:
            raise ValueError("Invalid expected revision")
        if not 1 <= len(evidence_quote.strip()) <= MAX_QUOTE:
            raise ValueError("Evidence quote must be 1..1000 characters")
        if not 5 <= len(reason.strip()) <= MAX_NOTE:
            raise ValueError("Reason must be 5..1000 characters")
        if not 1 <= len(reviewer.strip()) <= 80:
            raise ValueError("Invalid reviewer")
        if status == "CONFIRMED" and evidence_kind != "PRIMARY_DOCUMENT":
            raise ValueError("Confirmation requires primary-document evidence")
        if status == "RETRACTED" and evidence_kind != "CORRECTION":
            raise ValueError("Retraction requires a correction record")
        if status == "DISPUTED" and evidence_kind not in {"DENIAL", "CORRECTION"}:
            raise ValueError("Dispute requires denial or correction evidence")
        op_id = operation_id or str(uuid.uuid4())
        if not 1 <= len(op_id) <= 128:
            raise ValueError("Invalid operation ID")

        self.initialize()
        with self.claims.store.connect() as db:
            # Prevent concurrent reviewers from racing on the same claim.
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT id, claim_id, status, prior_status, evidence_article_id, reviewer "
                "FROM claim_review_history WHERE operation_id=?",
                (op_id,),
            ).fetchone()
            if existing:
                if existing[1] != claim_id or existing[2] != status or (
                    existing[4] != evidence_article_id
                ):
                    raise ValueError("Operation ID already used for another decision")
                return ReviewDecision(
                    existing[0], existing[1], existing[2], existing[3],
                    existing[4], existing[5],
                )
            claim = db.execute(
                "SELECT status, classification, source FROM article_event_claims WHERE id=?",
                (claim_id,)
            ).fetchone()
            if claim is None:
                raise ValueError("Claim ID not found")
            article = db.execute(
                "SELECT source, url, title, summary FROM articles WHERE id=?",
                (evidence_article_id,),
            ).fetchone()
            if article is None:
                raise ValueError("Evidence article ID not found")
            source, url, title, summary = article
            if evidence_quote not in title and evidence_quote not in (summary or ""):
                raise ValueError("Quote must appear verbatim in the stored evidence article")
            if evidence_kind == "PRIMARY_DOCUMENT":
                expected_host = OFFICIAL_HOSTS.get(source)
                parsed = urlsplit(url)
                hostname = (parsed.hostname or "").lower()
                if (
                    expected_host is None or parsed.scheme != "https"
                    or (hostname != expected_host
                        and not hostname.endswith("." + expected_host))
                ):
                    raise ValueError("Primary evidence requires a known official source and URL")
                # Prevent a valid but unrelated official article from being used
                # to certify a different class of news claim.
                supported = extract_claims(source, title, summary)
                if not any(
                    evidence_quote in item.evidence
                    and item.classification == claim[1]
                    and item.claim_kind == "REPORTED"
                    and item.polarity == "AFFIRMED"
                    for item in supported
                ):
                    raise ValueError("Primary evidence does not corroborate the claim type")
            previous = db.execute(
                "SELECT revision, status FROM claim_review_history "
                "WHERE claim_id=? ORDER BY revision DESC LIMIT 1",
                (claim_id,),
            ).fetchone()
            revision = previous[0] if previous else 0
            prior_status = previous[1] if previous else claim[0]
            if expected_revision != revision:
                raise ReviewConflict(
                    f"Stale revision: expected {expected_revision}, actual {revision}"
                )
            cursor = db.execute(
                """
                INSERT INTO claim_review_history (
                    operation_id, claim_id, revision, prior_status, status,
                    evidence_article_id, evidence_quote, evidence_kind,
                    reviewer, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (op_id, claim_id, revision + 1, prior_status, status,
                 evidence_article_id, evidence_quote, evidence_kind,
                 reviewer, reason, utc_now()),
            )
            return ReviewDecision(
                cursor.lastrowid, claim_id, status, prior_status,
                evidence_article_id, reviewer,
            )
