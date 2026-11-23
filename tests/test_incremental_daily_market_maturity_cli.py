"""Offline tests for incremental actual-price source extension.

All prices in this test are synthetic; provider calls are mocked.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from tris_market_event_intelligence.incremental_daily_market_maturity_cli import (
    extend_observed_daily,
    refresh_and_recheck,
)


def _csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    old = tmp_path / "old-prices.csv"
    _csv(old, ["symbol", "date", "adjusted_close", "source"], [
        ["NVDA", "2026-11-19", "100", "Yahoo Finance via yfinance; daily auto_adjust=True"],
        ["NVDA", "2026-11-20", "105", "Yahoo Finance via yfinance; daily auto_adjust=True"],
        ["SPY", "2026-11-19", "400", "Yahoo Finance via yfinance; daily auto_adjust=True"],
        ["SPY", "2026-11-20", "401", "Yahoo Finance via yfinance; daily auto_adjust=True"],
        ["EXTRA", "2026-11-20", "42", "synthetic older untouched source"],
    ])
    mapping = tmp_path / "mapping.csv"
    _csv(mapping, ["target", "market", "symbol", "benchmark", "verified_at"], [
        ["Nvidia", "XNAS", "NVDA", "SPY", "2026-11-23T01:00:00+00:00"],
        ["Other", "XNAS", "EXTRA", "SPY", "2026-11-20T01:00:00+00:00"],
    ])
    targets = tmp_path / "quotes.csv"
    _csv(targets, ["target", "market", "symbol", "benchmark", "verified_at"], [
        ["Nvidia", "XNAS", "NVDA", "SPY", "2026-11-23T01:00:00+00:00"],
    ])
    return old, mapping, targets


def quotes(*, altered: bool = False, include_future: bool = True) -> list[dict]:
    return [
        {"symbol": symbol, "date": day, "adjusted_close": str(amount),
         "source": "Yahoo Finance via yfinance; daily auto_adjust=True"}
        for symbol, anchor, newer in (
            ("NVDA", 95 if altered else 105, 110),
            ("SPY", 401, 402),
        )
        for day, amount in (
            [("2026-11-20", anchor)]
            + ([("2026-11-23", newer)] if include_future else [])
        )
    ]


def test_only_appends_real_post_anchor_prices_preserves_prior_symbols(
    tmp_path: Path,
) -> None:
    old, _, _ = setup(tmp_path)
    original_sha = hashlib.sha256(old.read_bytes()).hexdigest()
    rows, result = extend_observed_daily(old, quotes(), {"NVDA", "SPY"})
    assert hashlib.sha256(old.read_bytes()).hexdigest() == original_sha
    assert result["newRowsBySymbol"] == {"NVDA": 1, "SPY": 1}
    assert result["newPriceRows"] == 2
    assert result["existingPriceRows"] == 5
    assert result["resultPriceRows"] == 7
    assert len([x for x in rows if x["symbol"] == "EXTRA"]) == 1
    assert [x["date"] for x in rows if x["symbol"] == "NVDA"] == [
        "2026-11-19", "2026-11-20", "2026-11-23",
    ]


def test_adjusted_close_restatement_must_not_silently_splice(
    tmp_path: Path,
) -> None:
    old, _, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="revised"):
        extend_observed_daily(old, quotes(altered=True), {"NVDA", "SPY"})


def test_missing_price_anchor_and_missing_symbols_are_rejected(tmp_path: Path) -> None:
    old, _, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="anchor"):
        extend_observed_daily(
            old,
            [x for x in quotes() if x["date"] == "2026-11-23"],
            {"NVDA", "SPY"},
        )
    with pytest.raises(ValueError, match="fresh provider series"):
        extend_observed_daily(
            old, [x for x in quotes() if x["symbol"] == "NVDA"],
            {"NVDA", "SPY"},
        )
    with pytest.raises(ValueError, match="existing observed anchor"):
        extend_observed_daily(old, quotes(), {"MISSING"})


def test_idempotent_no_new_trading_bars_requires_no_fake_outcome(
    tmp_path: Path,
) -> None:
    old, _, _ = setup(tmp_path)
    rows, report = extend_observed_daily(
        old, quotes(include_future=False), {"NVDA", "SPY"},
    )
    assert report["newPriceRows"] == 0
    assert len(rows) == 5


def test_excludes_unfinished_market_day_and_bad_quote_scope(tmp_path: Path) -> None:
    old, mapping, targets = setup(tmp_path)
    previous = tmp_path / "old-outcome.json"
    previous.write_text(json.dumps({
        "schemaVersion": "automatic_gemma_daily_outcomes_v1",
        "sourceType": "REAL_AUTOMATIC_SQLITE_GEMMA_LEDGER",
        "researchOnly": True,
        "trainingEligible": False, "prospectivePredictionAccuracy": None,
    }))
    with pytest.raises(ValueError, match="unfinished"):
        refresh_and_recheck(
            previous, tmp_path / "news.sqlite3", mapping, old, targets,
            tmp_path / "not-created", model="gemma4:e2b",
            end=date(2099, 11, 1), client=object(),
        )


def test_real_price_refresh_rechecks_only_source_identical_cohort(
    tmp_path: Path,
) -> None:
    old, mapping, targets = setup(tmp_path)
    previous = tmp_path / "previous.json"
    source = {
        "articleId": 20975, "impactIndex": 0,
        "predictionSourceSha256": "e" * 64, "target": "Nvidia",
        "prediction": "UP", "publishedAt": "2026-11-23T00:08:00+00:00",
        "predictionSealedBeforeOutcome": False, "status": "NO_ENTRY_SESSION",
        "sourceAuditStatus": "STORED_ORIGINAL_SHA_MATCHED",
        "horizons": [],
    }
    previous.write_text(json.dumps({
        "schemaVersion": "automatic_gemma_daily_outcomes_v1",
        "sourceType": "REAL_AUTOMATIC_SQLITE_GEMMA_LEDGER",
        "researchOnly": True, "trainingEligible": False,
        "prospectivePredictionAccuracy": None,
        "impactsInspected": 1, "items": [source],
    }))
    new = {
        "schemaVersion": "automatic_gemma_daily_outcomes_v1",
        "sourceType": "REAL_AUTOMATIC_SQLITE_GEMMA_LEDGER",
        "researchOnly": True, "trainingEligible": False,
        "prospectivePredictionAccuracy": None,
        "impactsInspected": 2,
        "items": [
            {**source, "status": "PARTIAL", "entryDay": "2026-11-23",
             "horizons": [
                 {"sessions": 1, "verdict": "CORRECT", "dataStatus": "COMPLETED"},
                 {"sessions": 5, "verdict": "UNAVAILABLE", "dataStatus": "INSUFFICIENT_FUTURE"},
             ]},
            {**source, "articleId": 20976, "status": "UNMAPPED"},
        ],
    }
    with patch(
        "tris_market_event_intelligence.incremental_daily_market_maturity_cli.collect_prices"
    ) as collection, patch(
        "tris_market_event_intelligence.incremental_daily_market_maturity_cli.evaluate_automatic_outcomes"
    ) as evaluator:
        collection.return_value = (
            quotes(), {"missingSymbols": [], "provider": "synthetic fixture"},
        )
        evaluator.return_value = new
        directory = tmp_path / "incremental-output"
        result = refresh_and_recheck(
            previous, tmp_path / "news.sqlite3", mapping, old, targets,
            directory, model="gemma4:e2b", end=date(2026, 11, 24),
            client=object(),
        )
    assert result["newPriceRows"] == 2
    assert result["commonIdenticalSourceImpacts"] == 1
    assert result["newSourceImpacts"] == 1
    assert result["sameSourceHorizonScores"]["1d"]["newlyScored"] == 1
    assert result["sameSourceHorizonScores"]["5d"]["newlyScored"] == 0
    assert (directory / "expanded-adjusted-daily.csv").exists()
    assert (directory / "same-cohort-coverage.json").exists()
    with pytest.raises(ValueError, match="overwrite"):
        refresh_and_recheck(
            previous, tmp_path / "news.sqlite3", mapping, old, targets,
            directory, model="gemma4:e2b", end=date(2026, 11, 24),
            client=object(),
        )
