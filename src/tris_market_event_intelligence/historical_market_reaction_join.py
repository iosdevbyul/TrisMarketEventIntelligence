"""Research-only historical news-to-observed-minute reaction matching.

Never infer importance, causality or zero market impact from an unlinked item.
Uses the existing event study, point-in-time link screening and market CSV audit.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.event_ticker_screen import (
    EventTickerLink,
    screen_intraday_links,
    source_identity,
)
from tris_market_event_intelligence.intraday_event_study import (
    IntradayNews,
    MinuteClose,
    study_intraday,
)
from tris_market_event_intelligence.intraday_event_study_cli import (
    _calendar_csv,
    _csv_rows,
)
from tris_market_event_intelligence.market_minute_source import (
    audit_market_source,
)

_MAX_COHORT_BYTES = 8_000_000
_MAX_LINK_BYTES = 1_000_000


def _read(path: Path, max_bytes: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError("Existing bounded non-symlink research input required")
    data = path.read_bytes()
    if len(data) > max_bytes:
        raise ValueError("Research input grew during inspection")
    return data


def _utc(stamp: str) -> datetime:
    if not isinstance(stamp, str):
        raise TypeError("Timestamp must be text")
    try:
        dt = datetime.fromisoformat(stamp)
    except ValueError as exc:
        raise ValueError("Invalid timestamp") from exc
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Timestamp must include a timezone")
    return dt.astimezone(UTC)


def join_historical_market_reactions(
    *, cohort_path: Path, links_path: Path, minute_manifest: Path,
    sessions_path: Path, halts_path: Path | None = None,
) -> dict[str, object]:
    """Return observed historical excess returns, with explicit unmeasured cases.

    All price data and mapping assertions are operator-supplied, hash pinned
    but NOT authenticated. Only the existing screened event-study engine
    calculates prices or labels. Never produce train-ready labels here.
    """
    cohort_raw = _read(cohort_path, _MAX_COHORT_BYTES)
    cohort = json.loads(cohort_raw)
    if not isinstance(cohort, dict) or (
        cohort.get("schemaVersion") != "historical_news_cohort_v1"
        or cohort.get("status") != "REVIEW_ONLY_NOT_TRAINING_READY"
        or cohort.get("allNewsIncludedRegardlessOfExtraction") is not True
        or cohort.get("trainingDatasetEligible") is not False
        or cohort.get("sourceDbModified") is not False
    ):
        raise ValueError("Expected the original all-news research cohort")
    rows = cohort.get("items")
    if not isinstance(rows, list) or len(rows) > 5000:
        raise ValueError("Bounded original cohort page required")
    cutoff = _utc(cohort["asOf"])
    observed: dict[int, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("Invalid article row")
        article_id = row.get("articleId")
        if type(article_id) is not int or article_id < 1 or article_id in observed:
            raise ValueError("Duplicate or invalid article ID")
        if (
            row.get("captureChecksumVerified") is not True
            or row.get("importanceHumanGold") != "UNKNOWN"
            or row.get("observedMarketReaction") != "UNMEASURED"
            or row.get("trainingEligible") is not False
            or row.get("negativeSampleConfirmed") is not False
            or row.get("marketPriceLabel") is not None
        ):
            raise ValueError("Cohort must not contain inferred outcome labels")
        if not isinstance(row.get("sourceUrl"), str) or (
            source_identity(row["sourceUrl"]) is None
        ):
            raise ValueError("Article requires a valid original source URL")
        if _utc(row["firstObservedAt"]) > cutoff:
            raise ValueError("News was not known by research cutoff")
        observed[article_id] = row

    # Audit the manifest *before* reading any price CSV. This is an integrity
    # check for supplied files, not a provider-licence or authenticity check.
    source_audit = audit_market_source(minute_manifest)
    if any(
        source["status"] != "SCHEMA_COMPATIBLE_SOURCE_UNVERIFIED"
        for source in source_audit["sources"]
    ):
        raise ValueError("Only schema-compatible adjusted bars may be compared")
    manifest_raw = _read(minute_manifest, 65536)
    manifest = json.loads(manifest_raw)
    bars: list[MinuteClose] = []
    for entry in manifest["sources"]:
        price_path = minute_manifest.parent / entry["filename"]
        raw = _read(price_path, 32 * 1024 * 1024)
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Minute price file changed after audit")
        price_rows = _csv_rows(price_path, {
            "market", "symbol", "close_at", "adjusted_close", "adjustment",
            "instrument_kind", "provider", "observed_at",
        })
        for row in price_rows:
            if _utc(row["observed_at"]) <= cutoff:
                bars.append(MinuteClose(
                    market=row["market"], symbol=row["symbol"],
                    close_at=_utc(row["close_at"]),
                    adjusted_close=float(row["adjusted_close"]),
                ))
        if hashlib.sha256(_read(price_path, 32 * 1024 * 1024)).hexdigest() != (
            entry["sha256"]
        ):
            raise ValueError("Minute price file changed during comparison")
    if _read(minute_manifest, 65536) != manifest_raw:
        raise ValueError("Market manifest changed during comparison")

    _read(links_path, _MAX_LINK_BYTES)
    link_rows = _csv_rows(links_path, {
        "event_id", "market", "symbol", "benchmark_symbol",
        "source_url", "evidence_seen_at", "relationship",
    })
    if len(link_rows) > 5000:
        raise ValueError("Too many proposed article-to-market links")
    calendar = _calendar_csv(sessions_path, halts_path)
    if calendar is None:
        raise ValueError("Explicit exchange calendar required")
    event_news: list[IntradayNews] = []
    links: list[EventTickerLink] = []
    blocked: Counter[str] = Counter()
    matched: set[int] = set()
    for item in link_rows:
        article_id = int(item["event_id"])
        article = observed.get(article_id)
        if article is None:
            raise ValueError("Point-in-time link references an absent cohort article")
        if source_identity(item["source_url"]) != source_identity(article["sourceUrl"]):
            raise ValueError("Link URL is not the archived article source")
        seen = _utc(item["evidence_seen_at"])
        first = _utc(article["firstObservedAt"])
        if seen > first:
            blocked["LINK_KNOWN_AFTER_NEWS"] += 1
            continue
        link = EventTickerLink(
            event_id=article_id, market=item["market"],
            symbol=item["symbol"], benchmark_symbol=item["benchmark_symbol"],
            source_url=item["source_url"], evidence_seen_at=seen,
            relationship=item["relationship"],
        )
        # A reported publication time is NOT independently authenticated.
        # Use actual first receipt as the conservative news availability.
        event_news.append(IntradayNews(
            event_id=article_id, market=link.market, symbol=link.symbol,
            benchmark_symbol=link.benchmark_symbol, published_at=first,
            received_at=first, available_at=first,
        ))
        links.append(link)
        matched.add(article_id)

    outcomes = study_intraday(event_news, bars, calendar=calendar)
    screened, screening = screen_intraday_links(
        event_news, links, outcomes, horizons=(1, 5, 15, 30, 60),
    )
    labels_by_article: dict[int, list[dict[str, object]]] = {}
    counts: Counter[str] = Counter()
    for result in screened:
        labels = []
        for item in result.labels:
            if item.status == "COMPLETED":
                counts["OBSERVED_EXCESS_RETURN_UNVERIFIED"] += 1
            else:
                counts[item.status] += 1
            labels.append({
                "horizonMinutes": item.horizon_minutes,
                "status": item.status,
                "excessReturn": item.excess_return,
                "stockReturn": item.stock_return,
                "benchmarkReturn": item.benchmark_return,
            })
        labels_by_article.setdefault(result.event_id, []).append({
            "market": result.market, "symbol": result.symbol,
            "benchmarkSymbol": result.benchmark_symbol,
            "entryAt": result.entry_at, "status": result.status,
            "labels": labels,
        })
    results = []
    for article_id, row in observed.items():
        observations = labels_by_article.get(article_id, [])
        results.append({
            "articleId": article_id,
            "sourceUrl": row["sourceUrl"],
            "firstObservedAt": row["firstObservedAt"],
            "marketObservations": observations,
            "marketReactionStatus": (
                "OBSERVED_RESEARCH_ONLY" if any(
                    label["status"] == "COMPLETED"
                    for event in observations for label in event["labels"]
                ) else "UNMEASURED"
            ),
            "importanceHumanGold": "UNKNOWN",
            "unmatchedIsUnimportant": False,
            "trainingEligible": False,
        })
        if not observations:
            counts["ARTICLES_WITHOUT_MEASURED_LINK"] += 1
    return {
        "schemaVersion": "historical_news_market_reaction_join_v1",
        "status": "OBSERVED_RESEARCH_ONLY_SOURCE_UNVERIFIED",
        "asOf": cutoff.isoformat(),
        "cohortSha256": hashlib.sha256(cohort_raw).hexdigest(),
        "minuteManifestSha256": hashlib.sha256(manifest_raw).hexdigest(),
        "articlesInCohortPage": len(rows),
        "articlesWithPointInTimeLinks": len(matched),
        "blockedLinks": dict(sorted(blocked.items())),
        "labelCounts": dict(sorted(counts.items())),
        "screening": screening,
        "items": results,
        "priceSourceAuthenticated": False,
        "adjustmentIndependentlyVerified": False,
        "exchangeCalendarVerified": False,
        "linkIdentityIndependentlyVerified": False,
        "causalEffectEstimated": False,
        "newsImportanceMeasured": False,
        "trainingDatasetEligible": False,
        "productionDbModified": False,
        "marketDbModified": False,
        "stockPredictionProduced": False,
    }
