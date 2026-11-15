"""Offline historical price-shock research: NEWS + macro, no inferred causality.

Supplied historical files are untrusted until provider/licence provenance is reviewed.
This is an opt-in RESEARCH packet, never prediction gold or live trading input.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from math import isfinite
from pathlib import Path
from zoneinfo import ZoneInfo

from tris_market_event_intelligence.news_importance_review import _cohort

MAX_FILE = 8_000_000
MAX_DAYS = 5000
MAX_MACRO = 20000
MAX_NEWS = 10000
NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class DayBar:
    day: date
    adjusted_close: float
    volume: int


@dataclass(frozen=True)
class MacroRelease:
    series_id: str
    observation_date: date
    vintage_date: date
    release_at: datetime
    available_at: datetime
    value: float
    source: str


def _checked_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_FILE:
        raise ValueError("Existing bounded non-symlink research source required")
    raw = path.read_bytes()
    if len(raw) > MAX_FILE:
        raise ValueError("Research source file grew beyond size bound")
    return raw


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _csv_rows(raw: bytes, columns: set[str]) -> list[dict[str, str]]:
    import io

    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if reader.fieldnames is None or set(reader.fieldnames) != columns:
        raise ValueError("Research CSV columns do not match required contract")
    return list(reader)


def read_daily_prices(path: Path) -> tuple[list[DayBar], str]:
    """Require provider-adjusted close; metadata alone cannot verify authenticity."""
    raw = _checked_bytes(path)
    rows = _csv_rows(raw, {"date", "adjusted_close", "volume"})
    if not 4 <= len(rows) <= MAX_DAYS:
        raise ValueError("Expected 4..5000 price sessions")
    result: list[DayBar] = []
    previous: date | None = None
    for row in rows:
        day = date.fromisoformat(row["date"])
        price = float(row["adjusted_close"])
        volume_text = row["volume"]
        if (not isfinite(price) or price <= 0 or not volume_text.isdecimal()
                or day.weekday() > 4):
            raise ValueError("Invalid adjusted price/volume/session")
        volume = int(volume_text)
        if previous is not None and day <= previous:
            raise ValueError("Daily sessions must be strictly increasing and unique")
        result.append(DayBar(day, price, volume))
        previous = day
    return result, _sha(raw)


def read_macro_releases(path: Path) -> tuple[list[MacroRelease], str]:
    """Use recorded *release* and source-observed availability, never latest revisions."""
    raw = _checked_bytes(path)
    rows = _csv_rows(raw, {
        "series_id", "observation_date", "vintage_date", "release_at",
        "available_at", "value", "source",
    })
    if len(rows) > MAX_MACRO:
        raise ValueError("Macro research file too large")
    result: list[MacroRelease] = []
    unique: set[tuple[str, date, date]] = set()
    for row in rows:
        key = row["series_id"].strip()
        source = row["source"].strip()
        observed = date.fromisoformat(row["observation_date"])
        vintage = date.fromisoformat(row["vintage_date"])
        released = datetime.fromisoformat(row["release_at"])
        available = datetime.fromisoformat(row["available_at"])
        value = float(row["value"])
        if (not key or len(key) > 64 or not source or len(source) > 120
                or not isfinite(value) or released.utcoffset() is None
                or available.utcoffset() is None):
            raise ValueError("Invalid macro source, value or timestamp")
        released = released.astimezone(UTC)
        available = available.astimezone(UTC)
        if (released > available or vintage > released.date()
                or observed > released.date()):
            raise ValueError("Future macro revision or observation leakage")
        identity = (key, observed, vintage)
        if identity in unique:
            raise ValueError("Duplicated macro vintage")
        unique.add(identity)
        result.append(MacroRelease(
            key, observed, vintage, released, available, value, source,
        ))
    return result, _sha(raw)


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("Point-in-time NEWS timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def _context(
    *, day: date, articles: list[dict[str, object]],
    macro: list[MacroRelease], news_limit: int = 30,
) -> dict[str, object]:
    """Only source available by the exact stated cutoff can enter as-of features."""
    open_at = datetime.combine(day, time(9, 30), tzinfo=NY).astimezone(UTC)
    close_at = datetime.combine(day, time(16, 0), tzinfo=NY).astimezone(UTC)
    lookback = open_at - timedelta(days=3)
    post_end = close_at + timedelta(days=1)
    before: list[dict[str, object]] = []
    after: list[dict[str, object]] = []
    for article in articles:
        first_seen = _timestamp(str(article["firstObservedAt"]))
        if first_seen < lookback or first_seen > post_end:
            continue
        item = {
            "articleId": article["articleId"], "textSha256": article["textSha256"],
            "title": article["title"], "source": article["source"],
            "firstObservedAt": article["firstObservedAt"],
            "sourceUrl": article["sourceUrl"],
            "timing": "AVAILABLE_PRE_OPEN" if first_seen <= open_at else "POST_OPEN",
            "issuerAssociationVerified": False,
        }
        (before if first_seen <= open_at else after).append(item)
    before.sort(key=lambda x: (x["firstObservedAt"], x["articleId"]))
    after.sort(key=lambda x: (x["firstObservedAt"], x["articleId"]))
    # For each observation period, choose the latest available vintage as of the cutoff.
    available: dict[tuple[str, date], MacroRelease] = {}
    for release in macro:
        if release.available_at > open_at:
            continue
        key = release.series_id, release.observation_date
        old = available.get(key)
        if old is None or (release.available_at, release.vintage_date) > (
            old.available_at, old.vintage_date
        ):
            available[key] = release
    macro_points = [
        {
            "seriesId": r.series_id, "observationDate": r.observation_date.isoformat(),
            "vintageDate": r.vintage_date.isoformat(),
            "releaseAt": r.release_at.isoformat(),
            "availableAt": r.available_at.isoformat(),
            "value": r.value, "source": r.source,
        }
        for r in sorted(available.values(), key=lambda x: (
            x.series_id, x.observation_date
        ))
    ]
    return {
        "asOf": open_at.isoformat(),
        "newsAvailableBeforeOpen": before[:news_limit],
        "newsAvailableBeforeOpenTotal": len(before),
        "newsFirstObservedAfterOpen": after[:news_limit],
        "newsFirstObservedAfterOpenTotal": len(after),
        "macroPointInTime": macro_points,
        "macroObservationCount": len(macro_points),
        "newsIssuerLinksVerified": False,
        "intradayCausalityEstablished": False,
        "oldArticlesWithoutOriginalObservationHistoryExcluded": True,
    }


def build_historical_shock_research(
    *, symbol: str, benchmark_symbol: str, stock_path: Path,
    benchmark_path: Path, news_cohort_path: Path, macro_path: Path,
    top: int = 100, controls: int = 100,
) -> dict[str, object]:
    """Select historical shock days and small-move controls for *research*.

    Excludes stock market holidays via joined benchmark sessions; does not
    extrapolate missing prices. Current news cohort cannot prove a decade
    of event-time source availability.
    """
    if (not symbol or not benchmark_symbol or symbol == benchmark_symbol
            or len(symbol) > 24 or len(benchmark_symbol) > 24):
        raise ValueError("Two distinct explicit bounded symbols required")
    if (type(top) is not int or type(controls) is not int
            or not 1 <= top <= 100 or not 1 <= controls <= 100):
        raise ValueError("Research groups must be 1..100 days")
    stock, stock_sha = read_daily_prices(stock_path)
    benchmark, benchmark_sha = read_daily_prices(benchmark_path)
    macro, macro_sha = read_macro_releases(macro_path)
    news_raw, cohort = _cohort(news_cohort_path)
    articles = cohort["items"]
    if len(articles) > MAX_NEWS:
        raise ValueError("Bounded NEWS cohort required")
    base = {bar.day: bar for bar in benchmark}
    # Adjacent daily closes only: no returns across missing stock observations.
    candidate: list[dict[str, object]] = []
    for previous, current in zip(stock, stock[1:]):
        if previous.day not in base or current.day not in base:
            continue
        sr = current.adjusted_close / previous.adjusted_close - 1
        mr = base[current.day].adjusted_close / base[previous.day].adjusted_close - 1
        candidate.append({
            "date": current.day.isoformat(), "stockReturn": sr,
            "benchmarkReturn": mr, "excessReturn": sr - mr,
            "absoluteExcessReturn": abs(sr - mr),
            "stockVolume": current.volume,
        })
    if len(candidate) < top + controls:
        raise ValueError("Insufficient common trading sessions for two research groups")
    high = sorted(candidate, key=lambda x: (
        -x["absoluteExcessReturn"], x["date"],
    ))[:top]
    high_days = {x["date"] for x in high}
    rest = [x for x in candidate if x["date"] not in high_days]
    low = sorted(rest, key=lambda x: (
        x["absoluteExcessReturn"], x["date"],
    ))[:controls]
    def package(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return [{
            **row,
            "context": _context(
                day=date.fromisoformat(str(row["date"])),
                articles=articles, macro=macro,
            ),
            "retrospectiveEventCandidateOnly": True,
            "causalNewsAttributionVerified": False,
            "prospectivePrediction": False,
        } for row in rows]
    counts = Counter(_timestamp(str(a["firstObservedAt"])).date().isoformat()
                     for a in articles)
    return {
        "schemaVersion": "historical_news_macro_shock_research_v1",
        "status": "RESEARCH_PACKET_UNVERIFIED",
        "symbol": symbol, "benchmarkSymbol": benchmark_symbol,
        "topCount": top, "controlCount": controls,
        "stockPriceSha256": stock_sha,
        "benchmarkPriceSha256": benchmark_sha,
        "macroSha256": macro_sha, "newsCohortSha256": _sha(news_raw),
        "priceSourceAuthenticityVerified": False,
        "priceAdjustmentsExternallyVerified": False,
        "macroSourceAuthenticityVerified": False,
        "newsAndMacroCompleteForTenYears": False,
        "historicalNewsObservationDays": len(counts),
        "newsObservedDatesUTC": sorted(counts),
        "rankingIsRetrospectiveNotProspective": True,
        "controlSampling": "smallest-absolute-benchmark-excess-return",
        "calibrationOrPredictiveImprovementMeasured": False,
        "forwardPredictionSealed": False,
        "humanGoldCreated": False, "productionDbModified": False,
        "llmCalls": 0, "modelPromotionAllowed": False,
        "topEvents": package(high), "controlEvents": package(low),
    }
