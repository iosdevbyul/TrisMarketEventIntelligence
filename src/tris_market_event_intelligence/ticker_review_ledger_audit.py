"""Read-only whole-ledger integrity audit for post-hoc ticker decisions.

This audit deliberately never exports approved EventTickerLinks or training data.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from tris_market_event_intelligence.ticker_review_ledger import (
    SCHEMA,
    _digest,
    _payload,
    _read_connection,
)

MAX_AUDIT_ROWS = 100_000
REQUIRED_FIELDS = {
    "event_id", "market", "symbol", "decision", "reviewed_at",
    "reviewer_id", "review_note", "evidence_url", "article_url",
    "article_detected_at", "registry_source_id", "proposal_sha256",
    "review_status", "eligible_for_point_in_time_training",
}


def audit_review_ledger(path: Path) -> dict[str, object]:
    """Verify every stored row and its indexed columns; fail closed on corruption.

    No implicit ledger creation, repair, SQLite writes or sample-based success.
    The reviewer's identity and third-party evidence are NOT authenticated.
    """
    db = _read_connection(path)
    try:
        result = db.execute("PRAGMA integrity_check").fetchall()
        if result != [("ok",)]:
            raise ValueError("SQLite review ledger integrity failure")
        count = db.execute("SELECT COUNT(*) FROM review_decisions").fetchone()[0]
        if count > MAX_AUDIT_ROWS:
            raise ValueError("Review ledger exceeds bounded full-audit limit")
        approved = rejected = 0
        last_id = 0
        cursor = db.execute(
            """SELECT id,proposal_sha256,event_id,market,symbol,decision,
                      reviewed_at,record_sha256,record_json
               FROM review_decisions ORDER BY id ASC"""
        )
        for row in cursor:
            (record_id, proposal, event_id, market, symbol, decision,
             reviewed_at, stored_sha, raw) = row
            if type(raw) is not str or _digest(raw) != stored_sha:
                raise ValueError("Stored review record integrity mismatch")
            try:
                data = json.loads(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError("Malformed stored review record") from exc
            if not isinstance(data, dict) or not REQUIRED_FIELDS.issubset(data):
                raise ValueError("Incomplete stored review record")
            if raw != _payload(data):
                raise ValueError("Noncanonical stored review record")
            if type(record_id) is not int or record_id <= last_id:
                raise ValueError("Unexpected ledger record ordering")
            last_id = record_id
            if any((
                data["proposal_sha256"] != proposal,
                data["event_id"] != event_id or type(event_id) is not int or event_id < 1,
                data["market"] != market,
                data["symbol"] != symbol,
                data["decision"] != decision,
                data["reviewed_at"] != reviewed_at,
            )):
                raise ValueError("Review record/index mismatch")
            if not isinstance(proposal, str) or len(proposal) != 64 or any(
                c not in "0123456789abcdef" for c in proposal
            ):
                raise ValueError("Invalid proposal digest")
            if not isinstance(market, str) or market not in {"XKRX", "XNYS", "XNAS"}:
                raise ValueError("Invalid review market")
            if not isinstance(symbol, str) or not symbol:
                raise ValueError("Invalid review ticker")
            if decision not in {"APPROVE", "REJECT"}:
                raise ValueError("Invalid review decision")
            expected = "POST_HOC_RESEARCH_ASSOCIATION" if decision == "APPROVE" else "REJECTED"
            if data["review_status"] != expected:
                raise ValueError("Review status conflicts with decision")
            if data["eligible_for_point_in_time_training"] is not False:
                raise ValueError("Review record unexpectedly claims training eligibility")
            if not all(isinstance(data[key], str) and data[key] for key in (
                "reviewer_id", "review_note", "evidence_url", "article_url",
                "article_detected_at", "registry_source_id",
            )):
                raise ValueError("Incomplete review provenance")
            for key in ("reviewed_at", "article_detected_at"):
                value = datetime.fromisoformat(data[key])
                if value.utcoffset() is None:
                    raise ValueError("Review evidence timestamp lacks timezone")
            if decision == "APPROVE":
                approved += 1
            else:
                rejected += 1
        if approved + rejected != count:
            raise ValueError("Review ledger row count mismatch")
    except (sqlite3.Error, TypeError, KeyError, ValueError) as exc:
        raise ValueError("Review ledger full integrity audit failed") from exc
    finally:
        db.close()
    return {
        "schema_version": SCHEMA,
        "status": "VERIFIED_POST_HOC_RECORD_INTEGRITY_ONLY",
        "records_checked": count,
        "approved_post_hoc": approved,
        "rejected": rejected,
        "all_records_checked": True,
        "reviewer_identity_authenticated": False,
        "external_evidence_authenticated": False,
        "point_in_time_training_eligible": False,
        "approved_ticker_links_created": False,
        "database_modified": False,
    }
