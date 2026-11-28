"""Offline source-adapter tests: provider errors must not fabricate market prices."""
import csv
from datetime import date
from pathlib import Path

import pytest

from tris_market_event_intelligence.market_price_collection_cli import collect_prices


class FakeSeries:
    def __init__(self, data):
        self._data = data

    def items(self):
        return self._data.items()


class FakeFrame:
    def __init__(self, values):
        self.values = values
        self.empty = not values

    def __contains__(self, item):
        return item == "Close"

    def __getitem__(self, item):
        assert item == "Close"
        return FakeSeries(self.values)


class FakeTicker:
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, **kwargs):
        assert kwargs["auto_adjust"] is True
        assert kwargs["interval"] == "1d"
        assert kwargs["start"] == "2026-01-01"
        assert kwargs["end"] == "2026-01-04"
        if self.symbol == "SPY":
            return FakeFrame({date(2026, 1, 2): 100.0})
        return FakeFrame({date(2026, 1, 2): 200.0})


class FakeClient:
    def Ticker(self, symbol):
        return FakeTicker(symbol)


def test_collect_only_actual_source_rows(tmp_path: Path):
    mapping = tmp_path / "mapping.csv"
    with mapping.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["target", "market", "symbol", "benchmark", "verified_at"])
        writer.writerow(["Bank X", "XNAS", "BNS", "SPY", "2026-11-21T00:00:00+00:00"])
    rows, manifest = collect_prices(
        mapping, start=date(2026, 1, 1), end=date(2026, 1, 4),
        client=FakeClient(),
    )
    assert len(rows) == 2
    assert [row["symbol"] for row in rows] == ["BNS", "SPY"]
    assert rows[0]["adjusted_close"] == "200"
    assert manifest["pointInTimeArchive"] is False
    assert manifest["priceRows"] == 2


def test_missing_provider_series_is_marked_missing(tmp_path: Path):
    class EmptyTicker:
        def history(self, **kwargs):
            return FakeFrame({})

    class EmptyClient:
        def Ticker(self, symbol):
            return EmptyTicker()

    mapping = tmp_path / "mapping.csv"
    with mapping.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["target", "market", "symbol", "benchmark", "verified_at"])
        writer.writerow(["Bank X", "XNYS", "BNS", "SPY", "2026-11-21T00:00:00+00:00"])
    rows, manifest = collect_prices(
        mapping, start=date(2026, 1, 1), end=date(2026, 1, 4),
        client=EmptyClient(),
    )
    assert rows == []
    assert manifest["missingSymbols"] == ["BNS", "SPY"]


def test_invalid_interval_rejected(tmp_path: Path):
    mapping = tmp_path / "nonexistent.csv"
    with pytest.raises(ValueError, match="start < end"):
        collect_prices(
            mapping, start=date(2026, 1, 4), end=date(2026, 1, 1),
            client=FakeClient(),
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), 0.0, -1.0, None])
def test_invalid_series_is_atomic_and_other_symbols_continue(tmp_path: Path, invalid):
    class InvalidTicker(FakeTicker):
        def history(self, **kwargs):
            if self.symbol == "BNS":
                return FakeFrame({date(2026, 1, 2): 200.0, date(2026, 1, 3): invalid})
            return super().history(**kwargs)

    class InvalidClient:
        def Ticker(self, symbol):
            return InvalidTicker(symbol)

    mapping = tmp_path / "mapping.csv"
    with mapping.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["target", "market", "symbol", "benchmark", "verified_at"])
        writer.writerow(["Bank X", "XNYS", "BNS", "SPY", "2026-11-21T00:00:00+00:00"])
    rows, manifest = collect_prices(
        mapping, start=date(2026, 1, 1), end=date(2026, 1, 4),
        client=InvalidClient(),
    )
    assert [row["symbol"] for row in rows] == ["SPY"]
    assert manifest["missingSymbols"] == ["BNS"]
    assert manifest["rejectedSeries"][0]["date"] == "2026-01-03"
