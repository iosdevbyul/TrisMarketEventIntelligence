"""Past-only market trend forecasts and predeclared news-effect combination.

The price forecast is a simple diagnostic benchmark, NOT a trained financial
model and NOT evidence of event causation. Current release and later sessions
cannot enter the feature window. Keep the predicted event-price direction
distinct from the model's qualitative event-pressure direction.
"""
from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from tris_market_event_intelligence.sequential_market_learning_cli import (
    asset_market,
    yahoo_bars,
)

DIRECTIONS = ("NEGATIVE", "NEUTRAL", "POSITIVE")
SCORES = {"NEGATIVE": -1, "NEUTRAL": 0, "POSITIVE": 1}
SEOUL = ZoneInfo("Asia/Seoul")
NEW_YORK = ZoneInfo("America/New_York")


def historical_sessions(
    client: httpx.Client, symbol: str, release: datetime,
    *, lookback_days: int = 45,
) -> list[tuple[datetime, float]]:
    """Historical sessions only; exclude same-day close even for midday news."""
    if release.utcoffset() is None:
        raise ValueError("Release timestamp requires a timezone")
    zone = SEOUL if asset_market(symbol) == "KR" else NEW_YORK
    close = time(15, 30) if asset_market(symbol) == "KR" else time(16)
    # Fetching a future-looking Yahoo range is not allowed even if filtered.
    # yahoo_bars() requests future data, so we do not call it here.
    # The supplied client may enforce an earlier end timestamp.
    import urllib.parse

    url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
           + urllib.parse.quote(symbol, safe=""))
    response = client.get(
        url,
        params={
            "period1": int((release - timedelta(days=lookback_days)).timestamp()),
            "period2": int(release.timestamp()),
            "interval": "1d",
        },
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    chart = response.json()["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError("No historical price bars")
    record = chart["result"][0]
    from tris_market_event_intelligence.sequential_market_learning_cli import (
        verify_market_metadata,
    )

    verify_market_metadata(symbol, record.get("meta") or {})
    close_bars = record["indicators"]["quote"][0]["close"]
    sessions: dict[str, tuple[datetime, float]] = {}
    for timestamp, value in zip(record.get("timestamp") or [], close_bars, strict=False):
        if value is None or float(value) <= 0:
            continue
        observed = datetime.fromtimestamp(timestamp, UTC)
        local_date = observed.astimezone(zone).date()
        completed = datetime.combine(local_date, close, zone).astimezone(UTC)
        if completed >= release:
            continue
        sessions[local_date.isoformat()] = completed, float(value)
    return sorted(sessions.values())


def trend_forecast(
    client: httpx.Client, symbol: str, release: datetime,
    *, sessions: int = 3,
) -> dict[str, Any]:
    """Past-only 3-session trend; score is an explicit, simple baseline."""
    try:
        candles = historical_sessions(client, symbol, release)
    except (ValueError, KeyError, TypeError, IndexError, httpx.HTTPError) as exc:
        return {"status": "UNAVAILABLE", "reasonCode": type(exc).__name__,
                "symbol": symbol}
    if len(candles) < sessions + 1:
        return {"status": "UNAVAILABLE", "reasonCode": "INSUFFICIENT_PRIOR_SESSIONS",
                "symbol": symbol, "priorSessions": len(candles)}
    before, after = candles[-sessions-1], candles[-1]
    change = after[1] / before[1] - 1
    direction = ("POSITIVE" if change > 0.005 else
                 "NEGATIVE" if change < -0.005 else "NEUTRAL")
    return {
        "status": "AVAILABLE", "symbol": symbol, "direction": direction,
        "priorReturn": change, "priorSessions": sessions,
        "firstPriorClose": before[0].isoformat(),
        "lastPriorClose": after[0].isoformat(),
        "strictlyPreRelease": after[0] < release,
        "rule": "three_prior_session_return_threshold_0.5pct",
        "pointInTimeCertified": False,
    }


def combine_pressure(
    baseline_direction: str | None, news_direction: str | None,
) -> dict[str, Any]:
    """A fully specified additive counterfactual, without fitting to holdout.

    A neutral news pressure means *no additional pressure*, not a prediction
    that the security price has zero movement.
    """
    if baseline_direction not in SCORES or news_direction not in SCORES:
        return {"status": "ABSTAIN", "direction": None}
    combined = SCORES[baseline_direction] + SCORES[news_direction]
    if combined > 0:
        final = "POSITIVE"
    elif combined < 0:
        final = "NEGATIVE"
    else:
        final = "NEUTRAL"
    return {
        "status": "AVAILABLE",
        "direction": final,
        "baselineDirection": baseline_direction,
        "newsPressureDirection": news_direction,
        "combinationRule": "sum_signed_directions_clipped_to_sign_v1",
        "noPostReleasePriceInput": True,
        "causalAttributionEstablished": False,
    }


def annotate_price_study(
    prediction: dict[str, Any], evaluation: dict[str, Any],
    client: httpx.Client, release: datetime,
) -> dict[str, Any]:
    """Record fixed baseline/news/combined forecasts before checking outcomes.

    Caller must calculate forecasts first in its own prediction phase for
    genuine temporal replay. This helper is suitable for research diagnostics
    only and never trains or certifies PIT evidence.
    """
    from copy import deepcopy

    out = deepcopy(evaluation)
    cached: dict[str, dict] = {}
    for item in out.get("impacts", []):
        symbol = item.get("priceProxy")
        if not symbol:
            continue
        if symbol not in cached:
            cached[symbol] = trend_forecast(client, symbol, release)
        baseline = cached[symbol]
        news = item.get("direction")
        paired = combine_pressure(
            baseline.get("direction") if baseline["status"] == "AVAILABLE" else None,
            news,
        )
        item["priorOnlyBaseline"] = baseline
        item["baselinePlusNews"] = paired
        actual = item.get("actualDirection")
        if item.get("status") in {"CORRECT", "WRONG"} and actual in SCORES:
            item["baselinePriceMatch"] = (
                baseline["direction"] == actual
                if baseline["status"] == "AVAILABLE" else None
            )
            item["combinedPriceMatch"] = (
                paired["direction"] == actual if paired["status"] == "AVAILABLE"
                else None
            )
        else:
            item["baselinePriceMatch"] = None
            item["combinedPriceMatch"] = None
    out["priceStudyContract"] = "past_only_baseline_plus_news_v1"
    out["priceStudyLimitation"] = (
        "Baseline and combined forecasts are research diagnostics only; "
        "price-source publication times are not independently verified."
    )
    return out
