"""Read-only reconciliation of a pinned proposal bundle with the isolated review ledger.

A post-hoc approval is never a point-in-time EventTickerLink or training label.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.event_ticker_proposals import MARKETS, SYMBOL, _when
from tris_market_event_intelligence.ticker_human_review import _read, _web_url
from tris_market_event_intelligence.ticker_review_ledger import _digest, _read_connection
from tris_market_event_intelligence.ticker_review_ledger_audit import audit_review_ledger

MAX_CANDIDATES = 1000
UNVERIFIED_FLAGS = (
    "registry_source_authenticated",
    "symbol_identity_independently_verified",
    "economic_relationship_verified",
    "point_in_time_registry_claims_verified",
    "training_dataset_eligible",
    "market_impact_predicted",
    "database_modified",
)


def reconcile_ticker_review_queue(
    proposals_path: Path, ledger_path: Path,
) -> dict[str, object]:
    """Check exact proposal bytes against immutable decisions without writing anything.

    The entire ledger is audited before reconciling. Reviews from other bundles
    never satisfy this bundle; a new bundle needs new explicit human decisions.
    """
    data = _read(proposals_path)
    fingerprint = hashlib.sha256(data).hexdigest()
    document = json.loads(data)
    if not isinstance(document, dict) or (
        document.get("schema_version") != "event_ticker_proposals_v1"
        or document.get("status") != "REVIEW_REQUIRED"
    ):
        raise ValueError("Unsupported proposal bundle")
    if any(document.get(flag) is not False for flag in UNVERIFIED_FLAGS):
        raise ValueError("Unverified proposal safety flags required")
    events = document.get("items")
    if not isinstance(events, list) or len(events) > MAX_CANDIDATES:
        raise ValueError("Proposal event count exceeds review bounds")

    candidates: dict[tuple[int, str, str], dict[str, object]] = {}
    for event in events:
        if not isinstance(event, dict) or (
            type(event.get("event_id")) is not int or event["event_id"] < 1
        ):
            raise ValueError("Invalid proposal event ID")
        proposals = event.get("proposals")
        if not isinstance(proposals, list) or len(proposals) > MAX_CANDIDATES:
            raise ValueError("Invalid candidate list")
        for candidate in proposals:
            if not isinstance(candidate, dict) or (
                candidate.get("relationship") != "DIRECT_CANDIDATE"
                or candidate.get("human_review_required") is not True
            ):
                raise ValueError("Only explicit human-review issuer candidates allowed")
            market, symbol = candidate.get("market"), candidate.get("symbol")
            if not isinstance(market, str) or market not in MARKETS or (
                not isinstance(symbol, str) or not SYMBOL.fullmatch(symbol)
            ):
                raise ValueError("Invalid candidate exchange or symbol")
            key = (event["event_id"], market, symbol)
            if key in candidates:
                raise ValueError("Duplicate ticker candidate")
            received = _when(candidate["article_detected_at"])
            observed = _when(candidate["registry_evidence_seen_at"])
            if observed > received:
                raise ValueError("Future listing evidence is not admissible")
            _web_url(candidate["article_url"])
            if not isinstance(candidate.get("source_id"), str) or (
                not candidate["source_id"].strip()
            ):
                raise ValueError("Missing registry source provenance")
            candidates[key] = candidate
            if len(candidates) > MAX_CANDIDATES:
                raise ValueError("Too many ticker candidates")
    if _read(proposals_path) != data:
        raise ValueError("Proposal bundle changed during reconciliation")

    # Refuse a corrupt ledger even if the corrupt row falls outside this bundle.
    audit = audit_review_ledger(ledger_path)
    decisions: dict[tuple[int, str, str], tuple[str, int]] = {}
    db = _read_connection(ledger_path)
    try:
        rows = db.execute(
            """SELECT id,event_id,market,symbol,decision,record_sha256,record_json
               FROM review_decisions WHERE proposal_sha256=? ORDER BY id""",
            (fingerprint,),
        )
        for row_id, event_id, market, symbol, verdict, stored_hash, raw in rows:
            key = (event_id, market, symbol)
            if key not in candidates or key in decisions:
                raise ValueError("Unmatched or duplicated ledger decision in bundle")
            if _digest(raw) != stored_hash:
                raise ValueError("Review record hash mismatch")
            item = json.loads(raw)
            candidate = candidates[key]
            if item.get("article_url") != candidate["article_url"] or (
                _when(item["article_detected_at"])
                != _when(candidate["article_detected_at"])
            ) or item.get("registry_source_id") != candidate["source_id"]:
                raise ValueError("Stored review provenance diverges from proposal")
            if item.get("eligible_for_point_in_time_training") is not False:
                raise ValueError("Cannot use post-hoc review as training evidence")
            decisions[key] = (verdict, row_id)
    except (sqlite3.Error, TypeError, KeyError) as exc:
        raise ValueError("Unable to reconcile review ledger") from exc
    finally:
        db.close()

    items: list[dict[str, object]] = []
    counts = {"PENDING_HUMAN_REVIEW": 0, "APPROVED_POST_HOC": 0, "REJECTED": 0}
    for event_id, market, symbol in sorted(candidates):
        verdict = decisions.get((event_id, market, symbol))
        status = (
            "PENDING_HUMAN_REVIEW" if verdict is None
            else "APPROVED_POST_HOC" if verdict[0] == "APPROVE"
            else "REJECTED"
        )
        counts[status] += 1
        items.append({
            "event_id": event_id,
            "market": market,
            "symbol": symbol,
            "review_status": status,
            "ledger_id": verdict[1] if verdict is not None else None,
        })
    return {
        "schema_version": "ticker_review_queue_v1",
        "status": "RESEARCH_ONLY_REVIEW_QUEUE",
        "proposal_sha256": fingerprint,
        "candidates_total": len(candidates),
        "pending": counts["PENDING_HUMAN_REVIEW"],
        "approved_post_hoc": counts["APPROVED_POST_HOC"],
        "rejected": counts["REJECTED"],
        "ledger_records_audited": audit["records_checked"],
        "items": items,
        "human_review_required": True,
        "reviewer_identity_authenticated": False,
        "external_evidence_authenticated": False,
        "point_in_time_training_eligible": False,
        "approved_ticker_links_created": False,
        "database_modified": False,
    }
