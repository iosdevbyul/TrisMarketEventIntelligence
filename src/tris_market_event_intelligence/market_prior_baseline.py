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
    from urllib.parse import quote

    url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
           + quote(symbol, safe=""))
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


def gated_news_adjustment(
    baseline_direction: str | None, news_direction: str | None,
    surprise: dict[str, Any] | None, market_context: dict[str, Any] | None,
    release: datetime,
) -> dict[str, Any]:
    """Require independently timestamped expectations before overriding a trend.

    Model-authored surpriseAssessment is a *claim*, not proof that an outcome
    differed from the market's pre-release expectations. Without a verified
    before-release expectation feed the safe price forecast is unchanged.
    """
    if baseline_direction not in SCORES:
        return {"status": "ABSTAIN", "direction": None,
                "reasonCode": "PRICE_BASELINE_UNAVAILABLE"}
    held = {
        "status": "BASELINE_RETAINED", "direction": baseline_direction,
        "newsPressureDirection": news_direction,
        "newsAdjusted": False, "reasonCode": "NEWS_EVIDENCE_UNVERIFIED",
        "causalAttributionEstablished": False,
    }
    if news_direction not in {"POSITIVE", "NEGATIVE"}:
        held["reasonCode"] = "NO_DIRECTIONAL_NEWS_SURPRISE"
        return held
    surprise = surprise or {}
    if surprise.get("novelty") != "NEW":
        held["reasonCode"] = "NOVELTY_NOT_ESTABLISHED"
        return held
    if surprise.get("expectationStatus") not in {"ABOVE", "BELOW"}:
        held["reasonCode"] = "EXPECTATION_SURPRISE_NOT_ESTABLISHED"
        return held
    # The expectation source is separate from the Gemma-authored assessment.
    # UNKNOWN is the current default; do not promote a self-assessed surprise.
    consensus = (market_context or {}).get("consensus") or {}
    if consensus.get("status") != "VERIFIED":
        held["reasonCode"] = "NO_INDEPENDENT_PRE_RELEASE_CONSENSUS"
        return held
    published = consensus.get("publishedAt")
    try:
        source_time = datetime.fromisoformat(published)
        if source_time.utcoffset() is None or source_time >= release:
            held["reasonCode"] = "CONSENSUS_NOT_PRIOR_TO_RELEASE"
            return held
    except (TypeError, ValueError):
        held["reasonCode"] = "CONSENSUS_TIMESTAMP_UNVERIFIED"
        return held
    # Without audited evidence connecting the *specific issuer/security*
    # and surprise, even verified market-wide estimates cannot authorize an
    # individual asset override.
    if not consensus.get("issuerLinkVerified", False):
        held["reasonCode"] = "TARGET_SPECIFIC_EXPOSURE_UNVERIFIED"
        return held
    changed = combine_pressure(baseline_direction, news_direction)
    return {
        **changed,
        "status": "NEWS_ADJUSTED",
        "newsAdjusted": changed.get("direction") != baseline_direction,
        "reasonCode": "INDEPENDENT_TIMED_SURPRISE_CONFIRMED",
        "causalAttributionEstablished": False,
    }


def forecast_targets(
    prediction: dict[str, Any], client: httpx.Client, release: datetime,
    *, market_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Freeze baseline and combined decisions before any outcome is fetched."""
    cache: dict[str, dict] = {}
    forecasts = []
    for row in prediction.get("impacts", []):
        symbol = row.get("priceProxy")
        if not symbol:
            forecasts.append({"status": "UNAVAILABLE", "reasonCode": "NO_PRICE_TARGET"})
            continue
        if symbol not in cache:
            cache[symbol] = trend_forecast(client, symbol, release)
        base = cache[symbol]
        forecasts.append({
            "gatedNewsAdjustment": gated_news_adjustment(
                base.get("direction") if base["status"] == "AVAILABLE" else None,
                row.get("direction"), prediction.get("surpriseAssessment"),
                market_context, release,
            ),
            "priorOnlyBaseline": base,
            "baselinePlusNews": combine_pressure(
                base.get("direction") if base["status"] == "AVAILABLE" else None,
                row.get("direction"),
            ),
        })
    return forecasts


def annotate_price_study(
    evaluation: dict[str, Any], precommitted: list[dict[str, Any]],
) -> dict[str, Any]:
    """Join previously calculated forecasts to realized outcomes by position."""
    from copy import deepcopy

    out = deepcopy(evaluation)
    if len(out.get("impacts", [])) != len(precommitted):
        raise ValueError("Forecast/outcome target count mismatch")
    for row, projected in zip(out["impacts"], precommitted, strict=True):
        base = projected.get("priorOnlyBaseline", {"status": "UNAVAILABLE"})
        combined = projected.get("baselinePlusNews", {"status": "ABSTAIN"})
        gated = projected.get("gatedNewsAdjustment", {"status": "ABSTAIN"})
        row["priorOnlyBaseline"] = base
        row["baselinePlusNews"] = combined
        row["gatedNewsAdjustment"] = gated
        actual = row.get("actualDirection")
        if row.get("status") in {"CORRECT", "WRONG"} and actual in SCORES:
            row["baselinePriceMatch"] = (
                base["direction"] == actual if base["status"] == "AVAILABLE"
                else None
            )
            row["combinedPriceMatch"] = (
                combined["direction"] == actual if combined["status"] == "AVAILABLE"
                else None
            )
            row["gatedPriceMatch"] = (
                gated["direction"] == actual if gated.get("direction") in SCORES
                else None
            )
        else:
            row["baselinePriceMatch"] = None
            row["combinedPriceMatch"] = None
            row["gatedPriceMatch"] = None
    out["priceStudyContract"] = "past_only_baseline_plus_gated_news_v2"
    out["priceStudyLimitation"] = (
        "Baseline and combined decisions were computed before the outcome "
        "fetch, but retrospective source data are not independently PIT-certified."
    )
    return out
