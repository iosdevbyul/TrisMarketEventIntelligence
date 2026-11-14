"""Read-only point-in-time ticker link *proposals* from reviewed CSV metadata.

A matching name is neither proof of issuer identity nor causal stock impact.
This module does not create training labels or mutate SQLite.
"""
from __future__ import annotations

import csv
import itertools
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

MARKETS = {"XKRX", "XNYS", "XNAS"}
MAX_REGISTRY_BYTES = 2_000_000
MAX_ROWS = 10_000
MAX_EVENTS = 1_000
FIELDS = {
    "market", "symbol", "canonical_name", "aliases", "valid_from",
    "valid_until", "evidence_seen_at", "source_id",
}
SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,19}\Z")
ALPHA = re.compile(r"[A-Za-z0-9]")
CJK = re.compile(r"[\u3131-\u318e\uac00-\ud7a3]")
WORD = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True)
class ListedIssuer:
    market: str
    symbol: str
    canonical_name: str
    aliases: tuple[str, ...]
    valid_from: datetime
    valid_until: datetime | None
    evidence_seen_at: datetime
    source_id: str


def _when(raw: str) -> datetime:
    value = datetime.fromisoformat(raw)
    if value.utcoffset() is None:
        raise ValueError("Timezone-aware evidence and validity timestamps required")
    return value.astimezone(UTC)


def _names(name: str, aliases: str) -> tuple[str, ...]:
    values = [name, *(s.strip() for s in aliases.split("|") if s.strip())]
    cleaned: list[str] = []
    for value in values:
        value = value.strip()
        if not value or len(value) > 120 or any(c in value for c in "\n\r\t"):
            raise ValueError("Issuer name must be bounded nonempty text")
        if not (CJK.search(value) or ALPHA.search(value)):
            raise ValueError("Issuer name needs letters or digits")
        if not CJK.search(value) and len(value) < 4:
            raise ValueError("Short Latin names are ambiguous without independent identity review")
        if CJK.search(value) and len(value) < 2:
            raise ValueError("Single-character names are ambiguous")
        if value.casefold() not in {s.casefold() for s in cleaned}:
            cleaned.append(value)
    return tuple(cleaned)


def load_reviewed_issuers(path: Path) -> tuple[ListedIssuer, ...]:
    """Parse operator-supplied bounded registry, with no network or SQLite writes."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_REGISTRY_BYTES:
        raise ValueError("Bounded existing non-symlink CSV listing registry required")
    issuers: list[ListedIssuer] = []
    seen: set[tuple[str, str, datetime]] = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(FIELDS) or (
            set(reader.fieldnames) != FIELDS
        ):
            raise ValueError("Listing CSV has an invalid column contract")
        for index, row in enumerate(reader, start=1):
            if index > MAX_ROWS:
                raise ValueError("Listing CSV exceeds row limit")
            try:
                market, symbol = row["market"].strip(), row["symbol"].strip()
                if market not in MARKETS or not SYMBOL.fullmatch(symbol):
                    raise ValueError("Unsupported exchange or ticker symbol")
                started = _when(row["valid_from"])
                ended = _when(row["valid_until"]) if row["valid_until"] else None
                observed = _when(row["evidence_seen_at"])
                if ended is not None and ended <= started:
                    raise ValueError("Invalid listing validity interval")
                source = row["source_id"].strip()
                if not source or len(source) > 200:
                    raise ValueError("Reviewed source identifier required")
                names = _names(row["canonical_name"], row["aliases"])
                key = market, symbol, started
                if key in seen:
                    raise ValueError("Duplicate ticker validity row")
                seen.add(key)
                issuers.append(ListedIssuer(
                    market, symbol, names[0], names, started, ended, observed, source,
                ))
            except (AttributeError, TypeError, KeyError, ValueError) as exc:
                raise ValueError(f"Invalid listing registry row {index}") from exc
    if not issuers:
        raise ValueError("Listing registry has no issuers")
    # A registry with contradictory overlapping windows cannot be trusted.
    by_symbol: dict[tuple[str, str], list[ListedIssuer]] = {}
    for issuer in issuers:
        by_symbol.setdefault((issuer.market, issuer.symbol), []).append(issuer)
    for versions in by_symbol.values():
        ordered = sorted(versions, key=lambda x: x.valid_from)
        for old, new in itertools.pairwise(ordered):
            if old.valid_until is None or old.valid_until > new.valid_from:
                raise ValueError("Overlapping listing validity for one ticker")
    return tuple(issuers)


def _matches(text: str, name: str) -> bool:
    """Require token boundaries for Latin names; Korean names use exact substring."""
    haystack = text.casefold()
    needle = name.casefold()
    begin = 0
    while (at := haystack.find(needle, begin)) >= 0:
        end = at + len(needle)
        left = at == 0 or not WORD.fullmatch(haystack[at - 1])
        right = end == len(haystack) or not WORD.fullmatch(haystack[end])
        if left and right:
            return True
        begin = at + 1
    return False


def propose_event_tickers(
    db_path: Path, issuers: tuple[ListedIssuer, ...], *,
    limit: int = 100, as_of: datetime | None = None,
    after_event_id: int | None = None,
) -> dict[str, object]:
    """Only propose direct issuer-name mentions from *existing* extracted events.

    A registry row is eligible only when its evidence was available when
    the article was locally received; future listing knowledge is excluded.
    """
    if not 1 <= limit <= MAX_EVENTS:
        raise ValueError("Limit must be 1..1000")
    if after_event_id is not None and (type(after_event_id) is not int or after_event_id < 0):
        raise ValueError("after_event_id must be a nonnegative integer")
    now = as_of or datetime.now(UTC)
    if now.utcoffset() is None:
        raise ValueError("as_of must be timezone aware")
    if not db_path.is_file() or db_path.is_symlink():
        raise ValueError("Existing nonsymlink news database required")
    uri = db_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        if after_event_id is None:
            rows = db.execute(
                """SELECT e.id, a.title, a.summary, a.detected_at, e.source, e.source_url
                   FROM economic_events e JOIN articles a ON e.article_id=a.id
                   WHERE e.status IN ('VERIFIED','CANDIDATE','UNVERIFIED')
                   ORDER BY e.id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            has_more = False  # Legacy preview: only latest requested events.
        else:
            # Explicit forward scan: avoid silently omitting new events after
            # the latest 1,000 and permit deterministic bounded pagination.
            fetched = db.execute(
                """SELECT e.id, a.title, a.summary, a.detected_at, e.source, e.source_url
                   FROM economic_events e JOIN articles a ON e.article_id=a.id
                   WHERE e.status IN ('VERIFIED','CANDIDATE','UNVERIFIED')
                   AND e.id > ?
                   ORDER BY e.id ASC LIMIT ?""",
                (after_event_id, limit + 1),
            ).fetchall()
            has_more = len(fetched) > limit
            rows = fetched[:limit]
    results: list[dict[str, object]] = []
    for event_id, title, summary, detected, source, url in rows:
        received = _when(detected)
        # Use headline only: syndicated/HTML summaries are noisy; issuer-only,
        # non-instrument words (e.g. an ordinary word or ticker) are excluded.
        text = title or ""
        if received > now.astimezone(UTC):
            results.append({"event_id": event_id, "status": "FUTURE_NEWS_TIMESTAMP",
                            "article_detected_at": received.isoformat(), "proposals": []})
            continue
        names_to_issuers: dict[str, set[tuple[str, str]]] = {}
        eligible: dict[tuple[str, str], tuple[ListedIssuer, set[str]]] = {}
        future_evidence = False
        for issuer in issuers:
            matched = {name for name in issuer.aliases if _matches(text, name)}
            if not matched:
                continue
            if issuer.evidence_seen_at > received:
                future_evidence = True
                continue
            if not (issuer.valid_from <= received and (
                issuer.valid_until is None or received < issuer.valid_until
            )):
                continue
            key = issuer.market, issuer.symbol
            prior = eligible.get(key)
            if prior is None:
                eligible[key] = (issuer, set(matched))
            else:
                prior[1].update(matched)
            for name in matched:
                names_to_issuers.setdefault(name.casefold(), set()).add(key)
        ambiguous = {name for name, keys in names_to_issuers.items() if len(keys) > 1}
        proposals = []
        for (market, symbol), (issuer, found) in sorted(eligible.items()):
            if any(name.casefold() in ambiguous for name in found):
                continue
            proposals.append({
                "market": market, "symbol": symbol, "relationship": "DIRECT_CANDIDATE",
                "matched_names": sorted(found), "source_id": issuer.source_id,
                "registry_evidence_seen_at": issuer.evidence_seen_at.isoformat(),
                "article_detected_at": received.isoformat(),
                "article_source": source, "article_url": url,
                "human_review_required": True,
            })
        status = (
            "AMBIGUOUS_ALIAS" if ambiguous
            else "PROPOSED" if proposals
            else "FUTURE_REGISTRY_EVIDENCE" if future_evidence
            else "NO_VERIFIED_NAME_MATCH"
        )
        results.append({"event_id": event_id, "status": status,
                        "article_detected_at": received.isoformat(), "proposals": proposals})
    return {
        "schema_version": "event_ticker_proposals_v1",
        "status": "REVIEW_REQUIRED",
        "events_inspected": len(rows),
        "after_event_id": after_event_id,
        "next_cursor": rows[-1][0] if after_event_id is not None and rows else after_event_id,
        "has_more": has_more,
        "proposals": sum(len(r["proposals"]) for r in results),
        "items": results,
        "registry_source_authenticated": False,
        "symbol_identity_independently_verified": False,
        "economic_relationship_verified": False,
        "point_in_time_registry_claims_verified": False,
        "training_dataset_eligible": False,
        "market_impact_predicted": False,
        "database_modified": False,
        "note": "Only operator-supplied company name mentions in headlines. No auto-approved ticker links.",
    }
