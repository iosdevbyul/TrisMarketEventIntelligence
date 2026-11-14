"""Point-in-time linked news and overlapping-event quality screening.

Explicit symbol mappings are trusted only if observed before event availability.
Duplicate source stories and overlapping horizon exposure cannot silently become
independent training observations. This module never claims causal attribution.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from tris_market_event_intelligence.exchange_sessions import utc
from tris_market_event_intelligence.intraday_event_study import (
    IntradayNews,
    IntradayResult,
)

_TRACKING = {"fbclid", "gclid", "at_medium", "at_campaign"}


@dataclass(frozen=True)
class EventTickerLink:
    event_id: int
    market: str
    symbol: str
    benchmark_symbol: str
    source_url: str
    evidence_seen_at: datetime
    relationship: str

    def __post_init__(self) -> None:
        if self.event_id < 1 or not self.market.strip():
            raise ValueError("Event link requires valid ID and market")
        if not self.symbol.strip() or not self.benchmark_symbol.strip():
            raise ValueError("Event link requires symbol and benchmark")
        if self.relationship not in {"DIRECT", "SECTOR", "MARKET"}:
            raise ValueError("Unknown event-to-ticker relationship")
        utc(self.evidence_seen_at)
        if source_identity(self.source_url) is None:
            raise ValueError("Event link requires an absolute article URL")


def source_identity(url: str) -> str | None:
    """Conservatively remove tracking query keys but preserve article IDs."""
    parsed = urlsplit(url.strip())
    if parsed.scheme.lower() not in {"https", "http"} or not parsed.hostname:
        return None
    query = [
        (key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING
    ]
    return urlunsplit((
        parsed.scheme.lower(), parsed.netloc.lower(),
        parsed.path.rstrip("/") or "/", urlencode(sorted(query)), "",
    ))


def screen_intraday_links(
    events: Iterable[IntradayNews],
    links: Iterable[EventTickerLink],
    results: Iterable[IntradayResult],
    *,
    horizons: tuple[int, ...],
) -> tuple[list[IntradayResult], list[dict[str, object]]]:
    """Reject future-linked/duplicate samples and censor contaminated horizons.

    A later event on the same symbol censors the *earlier* sample's labels,
    without removing the later event. This does not guarantee causal isolation.
    """
    news = list(events)
    calculated = list(results)
    given = list(links)
    if not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("Expected positive minute horizons")
    by_event: dict[tuple[int, str, str], EventTickerLink] = {}
    for link in given:
        key = (link.event_id, link.market, link.symbol)
        if key in by_event:
            raise ValueError("Duplicate event-to-ticker link")
        by_event[key] = link
    keys = {(e.event_id, e.market, e.symbol) for e in news}
    if len(keys) != len(news) or len(calculated) != len(news):
        raise ValueError("News and study results must contain unique aligned events")
    if set(by_event) != keys:
        raise ValueError("Exactly one explicit source link per event-market-symbol required")
    for e, r in zip(news, calculated, strict=True):
        if (e.event_id, e.market, e.symbol) != (r.event_id, r.market, r.symbol):
            raise ValueError("Event study result order does not match news")
    duplicate: set[tuple[str, str, str]] = set()
    accepted: list[IntradayNews] = []
    excluded: dict[tuple[int, str, str], str] = {}
    # First available information wins; never prefer later knowledge.
    for e in sorted(news, key=lambda x: (utc(x.available_at), x.event_id)):
        key = (e.event_id, e.market, e.symbol)
        link = by_event[key]
        if link.benchmark_symbol != e.benchmark_symbol:
            excluded[key] = "BENCHMARK_LINK_MISMATCH"
            continue
        if utc(link.evidence_seen_at) > utc(e.available_at):
            excluded[key] = "LINK_KNOWN_AFTER_AVAILABILITY"
            continue
        identity = (e.market, e.symbol, source_identity(link.source_url))
        if identity in duplicate:
            excluded[key] = "DUPLICATE_SOURCE_EVENT"
            continue
        duplicate.add(identity)
        accepted.append(e)
    by_symbol: dict[tuple[str, str], list[datetime]] = {}
    for e in accepted:
        by_symbol.setdefault((e.market, e.symbol), []).append(utc(e.available_at))
    for times in by_symbol.values():
        times.sort()
    approved: list[IntradayResult] = []
    audit: list[dict[str, object]] = []
    for e, r in zip(news, calculated, strict=True):
        key = (e.event_id, e.market, e.symbol)
        reason = excluded.get(key)
        if reason is not None:
            approved.append(replace(
                r, status=reason,
                labels=tuple(replace(label, status=reason, stock_return=None,
                                     benchmark_return=None, excess_return=None)
                             for label in r.labels),
            ))
            audit.append({"event_id": e.event_id, "market": e.market,
                          "symbol": e.symbol, "status": reason, "censored_horizons": []})
            continue
        event_time = utc(e.available_at)
        following = next((
            t for t in by_symbol[(e.market, e.symbol)] if t > event_time
        ), None)
        # Simultaneously available distinct sources cannot be causally isolated.
        simultaneous = sum(
            utc(other.available_at) == event_time
            for other in accepted if other.market == e.market and other.symbol == e.symbol
        ) > 1
        censored: list[int] = []
        labels = []
        for label in r.labels:
            if label.status != "COMPLETED":
                labels.append(label)
                continue
            # Use the actual observable entry close, not published_at. A later
            # independent news item at any point up to exit contaminates a label.
            exit_at = utc(datetime.fromisoformat(label.exit_at))
            if simultaneous or (following is not None and following <= exit_at):
                censored.append(label.horizon_minutes)
                labels.append(replace(
                    label, status="OVERLAPPING_NEWS", stock_return=None,
                    benchmark_return=None, excess_return=None,
                ))
            else:
                labels.append(label)
        status = (
            "PARTIAL" if any(label.status != "COMPLETED" for label in labels)
            else "COMPLETED"
        ) if r.status in {"COMPLETED", "PARTIAL"} else r.status
        approved.append(replace(r, status=status, labels=tuple(labels)))
        audit.append({
            "event_id": e.event_id, "market": e.market, "symbol": e.symbol,
            "status": "SCREENED", "censored_horizons": censored,
        })
    return approved, audit
