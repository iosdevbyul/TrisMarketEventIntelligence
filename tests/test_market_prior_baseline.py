"""Past-only baseline regression, including the no-forward-prices invariant."""
from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from tris_market_event_intelligence.market_prior_baseline import (
    annotate_price_study,
    combine_pressure,
    forecast_targets,
    historical_sessions,
    trend_forecast,
)


def response(symbol: str, *, market: str, release: datetime) -> dict:
    # Yahoo daily bar timestamp is not necessarily the actual close time.
    # Historical sessions code reconstructs exchange-local close for filtering.
    dates = [
        datetime(2026, 11, 10, 14, tzinfo=UTC),
        datetime(2026, 11, 11, 14, tzinfo=UTC),
        datetime(2026, 11, 12, 14, tzinfo=UTC),
        datetime(2026, 11, 13, 14, tzinfo=UTC),
        datetime(2026, 11, 16, 14, tzinfo=UTC),
    ]
    return {
        "chart": {"error": None, "result": [{
            "meta": {"symbol": symbol, "currency": "USD" if market == "US" else "KRW",
                     "exchangeName": "NMS" if market == "US" else "KSC"},
            "timestamp": [int(d.timestamp()) for d in dates],
            "indicators": {"quote": [{"close": [100, 98, 96, 94, 130]}]},
        }]},
    }


class PriceClient:
    def __init__(self):
        self.calls = []

    def get(self, url, *, params, headers):
        self.calls.append((url, dict(params)))
        symbol = url.rsplit("/", 1)[-1]
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json=response(symbol, market="KR" if symbol.endswith(".KS") else "US",
                          release=datetime(2026, 11, 16, 13, tzinfo=UTC)),
        )


def test_baseline_excludes_later_market_close_even_if_provider_has_it():
    release = datetime(2026, 11, 16, 13, tzinfo=UTC)
    client = PriceClient()
    candles = historical_sessions(client, "NVDA", release)
    assert len(candles) == 4
    assert all(ts < release for ts, price in candles)
    assert candles[-1][1] == 94
    assert client.calls[0][1]["period2"] == int(release.timestamp())
    predicted = trend_forecast(client, "NVDA", release)
    assert predicted["status"] == "AVAILABLE"
    assert predicted["direction"] == "NEGATIVE"
    assert predicted["strictlyPreRelease"] is True


def test_additive_news_pressure_is_not_false_flat_price_prediction():
    assert combine_pressure("NEGATIVE", "NEUTRAL")["direction"] == "NEGATIVE"
    assert combine_pressure("NEGATIVE", "POSITIVE")["direction"] == "NEUTRAL"
    assert combine_pressure("POSITIVE", "POSITIVE")["direction"] == "POSITIVE"
    assert combine_pressure(None, "POSITIVE")["status"] == "ABSTAIN"


def test_forecast_sealed_before_outcome_and_compared_paired(monkeypatch):
    import tris_market_event_intelligence.market_prior_baseline as baseline

    calls = []
    def predictable(client, symbol, release):
        calls.append((symbol, release))
        return {"status": "AVAILABLE", "direction": "NEGATIVE",
                "symbol": symbol, "strictlyPreRelease": True}

    monkeypatch.setattr(baseline, "trend_forecast", predictable)
    prediction = {"impacts": [
        {"priceProxy": "NVDA", "direction": "NEUTRAL"},
        {"priceProxy": "NVDA", "direction": "POSITIVE"},
    ]}
    release = datetime(2026, 11, 16, 13, tzinfo=UTC)
    prior = forecast_targets(prediction, object(), release)
    assert len(calls) == 1
    assert prior[0]["baselinePlusNews"]["direction"] == "NEGATIVE"
    assert prior[1]["baselinePlusNews"]["direction"] == "NEUTRAL"
    evaluation = {"impacts": [
        {"status": "WRONG", "actualDirection": "NEGATIVE"},
        {"status": "WRONG", "actualDirection": "NEGATIVE"},
    ]}
    adjusted = annotate_price_study(evaluation, prior)
    assert [r["baselinePriceMatch"] for r in adjusted["impacts"]] == [True, True]
    assert [r["combinedPriceMatch"] for r in adjusted["impacts"]] == [True, False]
    assert "baselinePriceMatch" not in evaluation["impacts"][0]
    with pytest.raises(ValueError, match="count mismatch"):
        annotate_price_study({"impacts": []}, prior)


def test_korea_zone_excludes_same_day_close_at_release(monkeypatch):
    release = datetime(2026, 11, 16, 2, tzinfo=UTC)
    client = PriceClient()
    candles = historical_sessions(client, "069500.KS", release)
    assert candles[-1][1] == 94
    assert len(candles) == 4
    assert all(when < release for when, _ in candles)
