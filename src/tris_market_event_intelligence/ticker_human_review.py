"""Offline, nonmutating review of event-to-listed-issuer proposals.

A human approval attests to a proposed association, not issuer authenticity,
historical knowledge, causation, or eligibility for minute-return training.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from tris_market_event_intelligence.event_ticker_proposals import _when

MAX_INPUT_BYTES = 2_000_000
MAX_DECISIONS = 1000
COLUMNS = {
    "event_id", "market", "symbol", "decision", "reviewed_at",
    "reviewer_id", "evidence_url", "review_note", "proposal_sha256",
}
DECISIONS = {"APPROVE", "REJECT"}
HEX = set("0123456789abcdef")


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("Existing bounded non-symlink review inputs required")
    content = path.read_bytes()
    if len(content) > MAX_INPUT_BYTES:
        raise ValueError("Review input exceeded limit during reading")
    return content


def _text(value: object, *, max_size: int = 200) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_size:
        raise ValueError("Required bounded text missing")
    if any(char in value for char in "\r\n\0"):
        raise ValueError("Review text must be single-line")
    return value.strip()


def _web_url(value: object) -> str:
    url = _text(value, max_size=1000)
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Review evidence requires a public HTTPS URL without credentials")
    if parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Localhost evidence is not acceptable")
    return url


def review_ticker_candidates(
    proposals_path: Path, decisions_path: Path, *, as_of: datetime | None = None,
) -> dict[str, object]:
    """Validate human decisions against *exact* pinned proposal bytes.

    This function deliberately does not return a training-compatible
    EventTickerLink schema or write to any DB or file.
    """
    clock = as_of or datetime.now(UTC)
    if clock.utcoffset() is None:
        raise ValueError("Clock must contain timezone offset")
    proposals_bytes = _read(proposals_path)
    digest = hashlib.sha256(proposals_bytes).hexdigest()
    document = json.loads(proposals_bytes)
    if not isinstance(document, dict) or document.get("schema_version") != (
        "event_ticker_proposals_v1"
    ) or document.get("status") != "REVIEW_REQUIRED":
        raise ValueError("Unsupported proposal format")
    # Refuse externally asserted provenance or training-ready flags.
    if any(document.get(key) is not False for key in (
        "registry_source_authenticated", "symbol_identity_independently_verified",
        "economic_relationship_verified", "point_in_time_registry_claims_verified",
        "training_dataset_eligible", "market_impact_predicted", "database_modified",
    )):
        raise ValueError("Proposal review expects conservative unverified flags")
    items = document.get("items")
    if not isinstance(items, list) or len(items) > MAX_DECISIONS:
        raise ValueError("Invalid or oversized candidate list")
    available: dict[tuple[int, str, str], dict[str, object]] = {}
    for item in items:
        if not isinstance(item, dict) or type(item.get("event_id")) is not int:
            raise ValueError("Malformed proposal item")
        proposals = item.get("proposals")
        if not isinstance(proposals, list) or len(proposals) > MAX_DECISIONS:
            raise ValueError("Malformed item candidates")
        for candidate in proposals:
            if not isinstance(candidate, dict) or candidate.get("relationship") != (
                "DIRECT_CANDIDATE"
            ) or candidate.get("human_review_required") is not True:
                raise ValueError("Only manual direct candidate proposals can be reviewed")
            key = item["event_id"], candidate.get("market"), candidate.get("symbol")
            if key in available:
                raise ValueError("Duplicate proposed market-symbol-event key")
            if not all(isinstance(key_part, str) and key_part for key_part in key[1:]):
                raise ValueError("Invalid candidate market or symbol")
            received = _when(candidate["article_detected_at"])
            evidence_seen = _when(candidate["registry_evidence_seen_at"])
            if evidence_seen > received:
                raise ValueError("Candidate includes future registry evidence")
            _web_url(candidate["article_url"])
            _text(candidate["source_id"])
            available[key] = candidate
    decisions_bytes = _read(decisions_path)
    text = decisions_bytes.decode("utf-8-sig")
    import io

    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None or set(reader.fieldnames) != COLUMNS or (
        len(reader.fieldnames) != len(COLUMNS)
    ):
        raise ValueError("Review decision CSV contract mismatch")
    checked: set[tuple[int, str, str]] = set()
    reviewed: list[dict[str, object]] = []
    for index, decision in enumerate(reader, start=1):
        if index > MAX_DECISIONS:
            raise ValueError("Decision count exceeded limit")
        if None in decision or any(x is None for x in decision.values()):
            raise ValueError("Malformed review decision row")
        try:
            id_string = _text(decision["event_id"])
            event_id = int(id_string)
            if event_id < 1 or str(event_id) != id_string:
                raise ValueError("Noncanonical event ID")
            key = event_id, _text(decision["market"]), _text(decision["symbol"])
            if key not in available or key in checked:
                raise ValueError("Unknown or duplicate candidate decision")
            if decision["proposal_sha256"] != digest:
                raise ValueError("Decision does not match pinned proposal SHA-256")
            verdict = decision["decision"]
            if verdict not in DECISIONS:
                raise ValueError("Explicit APPROVE/REJECT decision required")
            reviewer = _text(decision["reviewer_id"], max_size=100)
            note = _text(decision["review_note"], max_size=1000)
            url = _web_url(decision["evidence_url"])
            reviewed_at = _when(decision["reviewed_at"])
            if reviewed_at > clock.astimezone(UTC):
                raise ValueError("Decision timestamp is in the future")
            candidate = available[key]
            received = _when(candidate["article_detected_at"])
            if reviewed_at < received:
                raise ValueError("Review cannot predate article receipt")
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError(f"Invalid review decision row {index}") from exc
        checked.add(key)
        reviewed.append({
            "event_id": event_id, "market": key[1], "symbol": key[2],
            "decision": verdict, "reviewed_at": reviewed_at.isoformat(),
            "reviewer_id": reviewer, "review_note": note, "evidence_url": url,
            "article_url": candidate["article_url"],
            "article_detected_at": received.isoformat(),
            "registry_source_id": candidate["source_id"],
            "proposal_sha256": digest,
            "review_status": "POST_HOC_RESEARCH_ASSOCIATION" if verdict == "APPROVE"
            else "REJECTED",
            "eligible_for_point_in_time_training": False,
        })
    if not reviewed:
        raise ValueError("At least one explicit review decision is required")
    if _read(proposals_path) != proposals_bytes or _read(decisions_path) != decisions_bytes:
        raise ValueError("Review evidence files changed during validation")
    return {
        "schema_version": "event_ticker_human_review_v1",
        "status": "REVIEWED_NOT_TRAINING_READY",
        "proposal_sha256": digest,
        "decisions": len(reviewed),
        "approved_for_post_hoc_review": sum(
            x["decision"] == "APPROVE" for x in reviewed
        ),
        "rejected": sum(x["decision"] == "REJECT" for x in reviewed),
        "unreviewed_candidates": len(available) - len(reviewed),
        "items": reviewed,
        "actual_reviewer_identity_authenticated": False,
        "external_issuer_evidence_authenticated": False,
        "training_dataset_eligible": False,
        "auto_approved": False,
        "existing_database_modified": False,
        "note": (
            "Approvals are post-receipt human research judgments, not time-of-news "
            "knowledge; no EventTickerLink or return label can be exported."
        ),
    }
