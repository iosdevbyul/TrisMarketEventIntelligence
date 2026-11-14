"""Separate diagnostic-only issuer alias hints; never create approved ticker links.

The source snapshot contains *names*, not reviewed aliases. Corporate suffix
stripping is only a reproducible hypothesis for an operator to examine.
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

from tris_market_event_intelligence.event_ticker_proposals import (
    ListedIssuer,
    _matches,
    _when,
)

# Conservative suffixes: do not remove Holdings, Group, Financial etc., and
# never interpret the resulting bare name as an authenticated legal identity.
LEGAL_ENDINGS = re.compile(
    r"(?:\s+(?:Corporation|Corp\.?|Incorporated|Inc\.?|Limited|Ltd\.?|PLC))$",
    re.IGNORECASE,
)
MIN_ALIAS_LENGTH = 6
MAX_HINTS = 100


def derived_alias_candidates(
    issuers: tuple[ListedIssuer, ...],
) -> dict[str, list[ListedIssuer]]:
    """Only propose reviewable nontrivial legal-suffix variants."""
    candidates: dict[str, list[ListedIssuer]] = {}
    for issuer in issuers:
        name = issuer.canonical_name.strip()
        stripped = LEGAL_ENDINGS.sub("", name).strip().rstrip(",").strip()
        if (
            stripped.casefold() == name.casefold()
            or len(stripped) < MIN_ALIAS_LENGTH
            or not any(letter.isalpha() for letter in stripped)
            or stripped.casefold() in {"common", "global", "national", "united"}
        ):
            continue
        candidates.setdefault(stripped.casefold(), []).append(issuer)
    return candidates


def inspect_alias_coverage(
    news_db: Path, issuers: tuple[ListedIssuer, ...],
    event_ids: list[int], *, first_seen: datetime, as_of: datetime,
) -> dict[str, object]:
    """Read only title/receipt data from the exact existing event page.

    Output is explicitly NOT proposals JSON and cannot be fed into the
    human-review approval CLI or market-return model.
    """
    if not event_ids:
        return {
            "counts": {}, "hints": [], "hints_total": 0,
            "hints_truncated": False, "derived_aliases_checked": 0,
            "diagnostic_only": True, "automatic_links_created": False,
            "database_modified": False,
        }
    if len(event_ids) > 500 or len(event_ids) != len(set(event_ids)) or (
        any(type(value) is not int or value <= 0 for value in event_ids)
    ):
        raise ValueError("Diagnostic page requires 1..500 unique event IDs")
    mapping = derived_alias_candidates(issuers)
    if not mapping:
        return {
            "counts": {}, "hints": [], "hints_total": 0,
            "hints_truncated": False, "derived_aliases_checked": 0,
            "diagnostic_only": True, "automatic_links_created": False,
            "database_modified": False,
        }
    uri = news_db.resolve().as_uri() + "?mode=ro"
    placeholders = ",".join("?" for _ in event_ids)
    with sqlite3.connect(uri, uri=True, timeout=5) as db:
        db.execute("PRAGMA query_only=ON")
        rows = db.execute(
            f"""SELECT e.id, a.title, a.detected_at, e.source_url
                 FROM economic_events e
                 JOIN articles a ON a.id = e.article_id
                 WHERE e.id IN ({placeholders}) ORDER BY e.id""",
            event_ids,
        ).fetchall()
    if len(rows) != len(event_ids):
        raise ValueError("Event page changed during alias diagnostics")

    counts: Counter[str] = Counter()
    hints: list[dict[str, object]] = []
    matched_total = 0
    for event_id, title, detected, article_url in rows:
        if not isinstance(title, str):
            title = ""
        received = _when(detected)
        for alias, matches in mapping.items():
            if not _matches(title, alias):
                continue
            matched_total += 1
            distinct = {(i.market, i.symbol) for i in matches}
            period = "PRE_OBSERVATION" if received < first_seen else "POST_OBSERVATION"
            if received > as_of:
                status = "FUTURE_NEWS_TIMESTAMP"
            elif len(distinct) > 1:
                status = "AMBIGUOUS_IDENTITY"
            elif period == "PRE_OBSERVATION":
                status = "PRE_OBSERVATION_ONLY"
            else:
                status = "NEEDS_HUMAN_ALIAS_REVIEW"
            counts[status] += 1
            if len(hints) >= MAX_HINTS:
                continue
            hints.append({
                "event_id": event_id,
                "article_detected_at": received.isoformat(),
                "article_url": article_url,
                "headline": title[:500],
                "candidate_alias": alias,
                "possible_issuers": [
                    {
                        "market": market, "symbol": symbol,
                        "canonical_name": next(
                            i.canonical_name for i in matches
                            if (i.market, i.symbol) == (market, symbol)
                        ),
                    }
                    for market, symbol in sorted(distinct)
                ],
                "diagnostic_status": status,
                "source_observation_at": first_seen.isoformat(),
                "human_review_required": True,
                "approved_ticker_link": False,
                "historical_listing_verified": False,
                "training_dataset_eligible": False,
            })
    return {
        "counts": dict(sorted(counts.items())),
        "hints": hints,
        "hints_total": matched_total,
        "hints_truncated": matched_total > len(hints),
        "derived_aliases_checked": len(mapping),
        "diagnostic_only": True,
        "automatic_links_created": False,
        "database_modified": False,
    }
