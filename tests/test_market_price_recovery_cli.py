"""Ensure recovery preserves real observations and never manufactures prices."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tris_market_event_intelligence.market_price_recovery_cli import recover_prices


def _csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    primary = tmp_path / "new.csv"
    fallback = tmp_path / "old.csv"
    mappings = tmp_path / "mappings.csv"
    fields = ["symbol", "date", "adjusted_close", "source"]
    _csv(primary, fields, [
        ["SPY", "2026-11-19", "100", "yfinance"], 
        ["SPY", "2026-11-20", "101", "yfinance"],
    ])
    _csv(fallback, fields, [
        ["BNS", "2026-11-19", "77", "yfinance"],
        ["SPY", "2026-11-19", "999", "old-revision"],
    ])
    _csv(mappings, ["target", "market", "symbol", "benchmark", "verified_at"], [
        ["BNS bank", "XNYS", "BNS", "SPY", "2026-11-22T00:00:00+00:00"],
    ])
    return primary, fallback, mappings


def test_missing_instrument_recovers_only_from_older_real_rows(tmp_path: Path) -> None:
    primary, fallback, mappings = _fixture(tmp_path)
    rows, manifest = recover_prices(primary, fallback, mappings)
    assert len(rows) == 3
    assert [row["adjusted_close"] for row in rows if row["symbol"] == "BNS"] == ["77"]
    assert [row["adjusted_close"] for row in rows if row["symbol"] == "SPY"] == [
        "100", "101",
    ]
    assert manifest["symbolOrigin"] == {"BNS": "fallback", "SPY": "primary"}
    assert manifest["recoveredSymbols"] == ["BNS"]
    assert manifest["missingSymbols"] == []
    assert manifest["completeDateCoverageGuaranteed"] is False


def test_missing_both_sources_not_synthesized(tmp_path: Path) -> None:
    primary, fallback, mappings = _fixture(tmp_path)
    _csv(fallback, ["symbol", "date", "adjusted_close", "source"], [
        ["SPY", "2026-11-19", "99", "yfinance"],
    ])
    rows, manifest = recover_prices(primary, fallback, mappings)
    assert len(rows) == 2
    assert manifest["missingSymbols"] == ["BNS"]
    assert not manifest["completeInstrumentCoverage"]


def test_duplicate_observation_rejected(tmp_path: Path) -> None:
    primary, fallback, mappings = _fixture(tmp_path)
    _csv(fallback, ["symbol", "date", "adjusted_close", "source"], [
        ["BNS", "2026-11-19", "77", "yfinance"],
        ["BNS", "2026-11-19", "77", "yfinance"],
    ])
    with pytest.raises(ValueError, match="Duplicate"):
        recover_prices(primary, fallback, mappings)
