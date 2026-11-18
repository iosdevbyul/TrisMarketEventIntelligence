"""Read-only bridge from existing RSS/economic-event SQLite into pressure packets.

No news or indicator facts are fabricated. Unreviewed source timestamps remain
unverified, and RSS summaries are not treated as authenticated article bodies.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.events import clean_text
from tris_market_event_intelligence.market_indicator_releases import IndicatorReleaseStore
from tris_market_event_intelligence.market_evidence_pressure import (
    MarketEvidencePacket,
)


def _date(raw: str | None) -> str:
    if not raw:
        raise ValueError("MARKET_SOURCE_PUBLICATION_DATE_MISSING")
    stamp = datetime.fromisoformat(raw)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("MARKET_SOURCE_PUBLICATION_ZONE_MISSING")
    return stamp.astimezone(UTC).date().isoformat()


def make_packet_from_db(
    db: Path, *, event_id: str, subject: str, as_of: str,
    article_ids: tuple[int, ...] = (), economic_event_ids: tuple[int, ...] = (),
    indicator_release_ids: tuple[int, ...] = (),
) -> MarketEvidencePacket:
    if (
        not article_ids and not economic_event_ids and not indicator_release_ids
        or len(article_ids) + len(economic_event_ids) + 2 * len(indicator_release_ids) > 24
        or any(type(i) is not int or i <= 0 for i in (
            *article_ids, *economic_event_ids, *indicator_release_ids
        ))
        or len(set(article_ids)) != len(article_ids)
        or len(set(economic_event_ids)) != len(economic_event_ids)
        or len(set(indicator_release_ids)) != len(indicator_release_ids)
    ):
        raise ValueError("MARKET_SOURCE_SELECTION_INVALID")
    if not db.is_file():
        raise ValueError("MARKET_SOURCE_DB_MISSING")
    evidences = []
    with sqlite3.connect("file:" + str(db.resolve()) + "?mode=ro", uri=True) as connection:
        for identifier in article_ids:
            row = connection.execute(
                "SELECT source, url, title, summary, published_at "
                "FROM articles WHERE id=?", (identifier,),
            ).fetchone()
            if row is None:
                raise ValueError("MARKET_SOURCE_ARTICLE_NOT_FOUND")
            name, url, title, summary, published = row
            excerpt = clean_text(" ".join(x for x in (title, summary) if x))[:1400]
            evidences.append({
                "sourceId": "article-" + str(identifier),
                "kind": "NEWS",
                "publishedDate": _date(published),
                "sourceName": name,
                "sourceUrl": url,
                "excerpt": excerpt,
                "verification": "REPORTED_UNVERIFIED",
            })
        for identifier in economic_event_ids:
            row = connection.execute(
                "SELECT classification, evidence, source, source_url, "
                "published_at, status FROM economic_events WHERE id=?",
                (identifier,),
            ).fetchone()
            if row is None:
                raise ValueError("MARKET_SOURCE_ECONOMIC_EVENT_NOT_FOUND")
            classification, excerpt, source, url, published, status = row
            text = clean_text(excerpt or "")[:1400]
            is_indicator = classification in {"ECONOMIC_INDICATOR", "MONETARY_POLICY"}
            evidences.append({
                "sourceId": "economic-event-" + str(identifier),
                "kind": "ECONOMIC_INDICATOR" if is_indicator else "NEWS",
                "publishedDate": _date(published),
                "sourceName": source,
                "sourceUrl": url,
                "excerpt": text,
                "verification": (
                    "VERIFIED_SOURCE"
                    if status == "VERIFIED" and classification == "MONETARY_POLICY"
                    else "REPORTED_UNVERIFIED"
                ),
                **({"indicatorName": classification} if is_indicator else {}),
            })
    for identifier in indicator_release_ids:
        release = IndicatorReleaseStore(db).read(identifier)
        actual = {
            "sourceId": f"indicator-release-{identifier}-actual",
            "kind": "ECONOMIC_INDICATOR",
            "publishedDate": release.publishedAt.astimezone(UTC).date().isoformat(),
            "sourceName": release.actualSourceName,
            "sourceUrl": release.actualSourceUrl,
            "excerpt": release.actualExcerpt,
            "verification": "REPORTED_UNVERIFIED",
            "indicatorName": release.indicatorName,
            "indicatorPeriod": release.period,
            "indicatorValue": release.actual,
            "indicatorPrevious": release.previous,
        }
        evidences.append(actual)
        if release.consensus is not None:
            evidences.append({
                "sourceId": f"indicator-release-{identifier}-consensus",
                "kind": "ECONOMIC_INDICATOR",
                "publishedDate": release.consensusPublishedAt.astimezone(
                    UTC,
                ).date().isoformat(),
                "sourceName": release.consensusSourceName,
                "sourceUrl": release.consensusSourceUrl,
                "excerpt": release.consensusExcerpt,
                "verification": "REPORTED_UNVERIFIED",
                "indicatorName": release.indicatorName,
                "indicatorPeriod": release.period,
                "expectation": release.consensus,
            })
    return MarketEvidencePacket.model_validate({
        "eventId": event_id,
        "subject": subject,
        "asOfDateExclusive": as_of,
        "evidences": evidences,
    })
