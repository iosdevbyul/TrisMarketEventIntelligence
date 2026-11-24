"""Opt-in immutable forecast precommitment recorded at *new* model completion.

A local SQLite record with triggers and hashes is not a trusted external
timestamp/signature. Never label it as independently proven prospective.
No historical backfill and absolutely no price lookup during commitment.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from tris_market_event_intelligence.event_aware_forecast import horizon_evidence, horizon_prediction
from tris_market_event_intelligence.market_outcome_validation_cli import HORIZONS, ZONES, _timestamp

SCHEMA = """
CREATE TABLE IF NOT EXISTS automatic_forecast_commitments (
  article_id INTEGER NOT NULL,
  body_sha256 TEXT NOT NULL,
  model TEXT NOT NULL,
  impact_index INTEGER NOT NULL,
  horizon_sessions INTEGER NOT NULL,
  sealed_at TEXT NOT NULL,
  publisher_stated_at TEXT,
  detected_at TEXT,
  original_body_observed_at TEXT,
  analyzed_at TEXT NOT NULL,
  target TEXT NOT NULL,
  direction TEXT NOT NULL,
  rationale TEXT NOT NULL,
  evidence_quote TEXT NOT NULL,
  market TEXT,
  symbol TEXT,
  benchmark TEXT,
  mapping_verified_at TEXT,
  market_cutoff_policy TEXT NOT NULL,
  eligibility TEXT NOT NULL,
  forecast_sha256 TEXT NOT NULL,
  previous_commitment_sha256 TEXT,
  first_public_availability_independently_verified INTEGER NOT NULL DEFAULT 0
    CHECK(first_public_availability_independently_verified=0),
  external_timestamp_authenticated INTEGER NOT NULL DEFAULT 0
    CHECK(external_timestamp_authenticated=0),
  PRIMARY KEY(article_id,body_sha256,model,impact_index,horizon_sessions)
);
CREATE TRIGGER IF NOT EXISTS no_forecast_commitment_update
BEFORE UPDATE ON automatic_forecast_commitments BEGIN
  SELECT RAISE(ABORT, 'FORECAST_PRECOMMITMENT_IMMUTABLE');
END;
CREATE TRIGGER IF NOT EXISTS no_forecast_commitment_delete
BEFORE DELETE ON automatic_forecast_commitments BEGIN
  SELECT RAISE(ABORT, 'FORECAST_PRECOMMITMENT_IMMUTABLE');
END;
"""

CUTOFF = "BEFORE_EXCHANGE_LOCAL_PUBLICATION_DATE_END; NEXT_SESSION_ENTRY_CLOSE"
ELIGIBLE = "CANDIDATE_ONLY_UNAUTHENTICATED_SOURCE_CLOCK"
UNMAPPED = "NO_PRECOMMITTED_INSTRUMENT"
LATE = "AFTER_CONSERVATIVE_LOCAL_CUTOFF"
INVALID = "UNVERIFIED_SOURCE_CLOCK"
BAD_MAPPING = "MAPPING_NOT_VERIFIED_AT_COMMITMENT"
DIRECTIONS = {"UP", "DOWN", "NEUTRAL", "UNCERTAIN"}


def initialize(db: sqlite3.Connection) -> None:
    db.executescript(SCHEMA)


def _time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return _timestamp(value).astimezone(UTC)
    except (ValueError, TypeError, OverflowError):
        return None


def _canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def record_new_analysis(
    db: sqlite3.Connection, *,
    article: dict, analysis: dict, model: str,
    body_sha256: str, analyzed_at: str, mappings: dict[str, dict],
) -> dict[str, int]:
    """Called in the SAME transaction as a genuinely new completed LLM row.

    Replays/retries never change a completed forecast. All model impacts are
    sealed as produced, including abstention and unscored/unmapped targets.
    Price bars are never read. Output may be candidate, not certified.
    """
    published = _time(article.get("publishedAt"))
    detected = _time(article.get("detectedAt"))
    analyzed = _time(analyzed_at)
    body_observed = _time(article.get("bodyObservedAt"))
    if analyzed is None:
        raise ValueError("Expected clock-stamped completed model inference")
    impacts = analysis.get("impacts")
    if not isinstance(impacts, list) or len(impacts) > 12:
        raise ValueError("Invalid bounded Gemma impacts for precommitment")
    counts: dict[str, int] = {}
    for index, impact in enumerate(impacts):
        target = impact.get("target")
        direction = impact.get("direction")
        reason = impact.get("reason")
        quote = impact.get("evidence_quote")
        if (not isinstance(target, str) or not target.strip()
                or direction not in DIRECTIONS
                or not isinstance(reason, str) or not reason.strip()
                or not isinstance(quote, str) or not quote.strip()):
            raise ValueError("Forecast impact lacks original target/direction/source rationale")
        mapping = mappings.get(target)
        if (published is None or detected is None or published > detected
                or detected > analyzed
                or (body_observed is not None and body_observed > analyzed)):
            status = INVALID
        elif mapping is None:
            status = UNMAPPED
        else:
            verified = _time(mapping["verifiedAt"])
            if verified is None or verified > analyzed:
                status = BAD_MAPPING
            elif analyzed.astimezone(ZoneInfo(ZONES[mapping["market"]])).date() > (
                published.astimezone(ZoneInfo(ZONES[mapping["market"]])).date()
            ):
                status = LATE
            else:
                status = ELIGIBLE
        for horizon in HORIZONS:
            direction, reason = horizon_prediction(analysis, impact, horizon)
            quote = horizon_evidence(analysis, impact, horizon)
            if direction not in DIRECTIONS or not isinstance(reason, str) or not reason.strip():
                raise ValueError("Invalid horizon forecast")
            last = db.execute(
                "SELECT forecast_sha256 FROM automatic_forecast_commitments "
                "ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            prior = last[0] if last else None
            data = {
                "articleId": article["articleId"],
                "bodySha256": body_sha256,
                "model": model, "impactIndex": index,
                "horizonSessions": horizon,
                "publisherStatedAt": article.get("publishedAt"),
                "detectedAt": article.get("detectedAt"),
                "originalBodyObservedAt": article.get("bodyObservedAt"),
                "analyzedAt": analyzed_at, "sealedAt": analyzed_at,
                "target": target, "direction": direction,
                "rationale": reason, "evidenceQuote": quote,
                "market": mapping["market"] if mapping else None,
                "symbol": mapping["symbol"] if mapping else None,
                "benchmark": mapping["benchmark"] if mapping else None,
                "mappingVerifiedAt": mapping["verifiedAt"] if mapping else None,
                "marketCutoffPolicy": CUTOFF,
                "eligibility": status,
                "previousCommitmentSha256": prior,
                "independentlyVerifiedFirstAvailability": False,
                "externallyAuthenticatedSeal": False,
            }
            digest = hashlib.sha256(_canonical(data).encode("utf-8")).hexdigest()
            db.execute(
                """INSERT INTO automatic_forecast_commitments
                (article_id,body_sha256,model,impact_index,horizon_sessions,
                 sealed_at,publisher_stated_at,detected_at,original_body_observed_at,
                 analyzed_at,target,direction,rationale,evidence_quote,
                 market,symbol,benchmark,mapping_verified_at,market_cutoff_policy,
                 eligibility,forecast_sha256,previous_commitment_sha256)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (article["articleId"], body_sha256, model, index, horizon,
                 analyzed_at, article.get("publishedAt"), article.get("detectedAt"),
                 article.get("bodyObservedAt"), analyzed_at, target, direction,
                 reason, quote, data["market"], data["symbol"],
                 data["benchmark"], data["mappingVerifiedAt"], CUTOFF,
                 status, digest, prior),
            )
            counts[status] = counts.get(status, 0) + 1
    return counts



def verify_chain(db: sqlite3.Connection) -> dict:
    """Reconstruct each local canonical record; detect accidental rewriting."""
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT * FROM automatic_forecast_commitments ORDER BY rowid ASC"
    ).fetchall()
    previous = None
    for row in rows:
        if row["previous_commitment_sha256"] != previous:
            raise ValueError("COMMITMENT_HASH_CHAIN_BROKEN")
        data = {
            "articleId": row["article_id"],
            "bodySha256": row["body_sha256"],
            "model": row["model"], "impactIndex": row["impact_index"],
            "horizonSessions": row["horizon_sessions"],
            "publisherStatedAt": row["publisher_stated_at"],
            "detectedAt": row["detected_at"],
            "originalBodyObservedAt": row["original_body_observed_at"],
            "analyzedAt": row["analyzed_at"], "sealedAt": row["sealed_at"],
            "target": row["target"], "direction": row["direction"],
            "rationale": row["rationale"], "evidenceQuote": row["evidence_quote"],
            "market": row["market"], "symbol": row["symbol"],
            "benchmark": row["benchmark"],
            "mappingVerifiedAt": row["mapping_verified_at"],
            "marketCutoffPolicy": row["market_cutoff_policy"],
            "eligibility": row["eligibility"],
            "previousCommitmentSha256": previous,
            "independentlyVerifiedFirstAvailability": False,
            "externallyAuthenticatedSeal": False,
        }
        digest = hashlib.sha256(_canonical(data).encode("utf-8")).hexdigest()
        if (row["forecast_sha256"] != digest
                or row["first_public_availability_independently_verified"] != 0
                or row["external_timestamp_authenticated"] != 0):
            raise ValueError("FORECAST_COMMITMENT_CONTENT_HASH_MISMATCH")
        previous = digest
    return {
        "integrity": "LOCAL_HASH_CHAIN_VALID",
        "records": len(rows), "chainHeadSha256": previous,
        "independentTimestampAuthority": False,
        "historicalTamperProof": False,
        "trainingEligible": False,
    }

def inspect_commitments(db: sqlite3.Connection) -> dict:
    """Read-only visibility of sealing and limited PIT eligibility status."""
    if not db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='automatic_forecast_commitments'"
    ).fetchone():
        return {
            "records": 0, "eligibleCandidateRecords": 0,
            "firstAvailabilityIndependentlyVerified": False,
            "externalTimestampAuthenticated": False,
        }
    values = db.execute(
        "SELECT eligibility, COUNT(*) FROM automatic_forecast_commitments "
        "GROUP BY eligibility"
    ).fetchall()
    counts = {name: count for name, count in values}
    return {
        "records": sum(counts.values()),
        "eligibleCandidateRecords": counts.get(ELIGIBLE, 0),
        "byEligibility": counts,
        "firstAvailabilityIndependentlyVerified": False,
        "externalTimestampAuthenticated": False,
        "prospectiveAccuracy": None,
        "trainingEligible": False,
    }
