"""Synthetic data regression tests for exact real-issuer market coverage."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from tris_market_event_intelligence import listed_equity_price_coverage_cli as coverage
from tris_market_event_intelligence.listed_equity_price_coverage_cli import (
    prepare_direct_equities,
    run,
)


def csv_rows(path: Path, fields: list[str], values: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(values)


def fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    outcomes = tmp_path / "actual-shaped-automatic-outcomes.json"
    items = [
        {
            "articleId": 20975, "target": "Nvidia", "status": "UNMAPPED",
            "sourceAuditStatus": "STORED_ORIGINAL_SHA_MATCHED",
            "publishedAt": "2026-11-20T16:00:00+00:00",
            "prediction": "UP", "predictionSealedBeforeOutcome": False,
        },
        {
            "articleId": 20989, "target": "Capita", "status": "UNMAPPED",
            "sourceAuditStatus": "STORED_ORIGINAL_SHA_MATCHED",
            "publishedAt": "2026-11-23T07:00:00+00:00",
            "prediction": "DOWN", "predictionSealedBeforeOutcome": False,
        },
        {
            "articleId": 20990, "target": "South East Water", "status": "UNMAPPED",
            "sourceAuditStatus": "STORED_ORIGINAL_SHA_MATCHED",
            "publishedAt": "2026-11-23T08:00:00+00:00",
            "prediction": "DOWN", "predictionSealedBeforeOutcome": False,
        },
        {
            "articleId": 20975, "target": "Anthropic", "status": "UNMAPPED",
            "sourceAuditStatus": "STORED_ORIGINAL_SHA_MATCHED",
            "publishedAt": "2026-11-20T16:00:00+00:00",
            "prediction": "UNCERTAIN", "predictionSealedBeforeOutcome": False,
        },
    ]
    for i, row in enumerate(items):
        row["impactIndex"] = i
        row["predictionSourceSha256"] = hashlib.sha256(
            f"synthetic-original-{row['articleId']}".encode()
        ).hexdigest()
    outcomes.write_text(json.dumps({
        "schemaVersion": "automatic_gemma_daily_outcomes_v1",
        "sourceType": "REAL_AUTOMATIC_SQLITE_GEMMA_LEDGER",
        "researchOnly": True, "trainingEligible": False,
        "prospectivePredictionAccuracy": None,
        "impactsInspected": len(items), "items": items,
        "statusCounts": {"UNMAPPED": len(items), "COMPLETED": 0},
        "metrics": {"1d": {"denominator": 0}},
    }), encoding="utf-8")
    mapping = tmp_path / "previous-reviewed.csv"
    csv_rows(mapping, [
        "target", "market", "symbol", "benchmark", "verified_at",
    ], [[
        "Original issuer", "XNAS", "EXMP", "SPY", "2026-11-19T00:00:00+00:00",
    ]])
    prices = tmp_path / "prior-adjusted.csv"
    csv_rows(prices, ["symbol", "date", "adjusted_close", "source"], [
        ["EXMP", "2026-11-20", "100", "synthetic prior quote"],
        ["SPY", "2026-11-20", "400", "synthetic prior quote"],
    ])
    return outcomes, mapping, prices


def test_real_named_up_down_issuers_only_no_private_sector_or_venue(
    tmp_path: Path,
) -> None:
    outcomes, mapping, prices = fixtures(tmp_path)
    old_source = hashlib.sha256(outcomes.read_bytes()).hexdigest()
    mapped, new, report = prepare_direct_equities(
        outcomes, mapping, prices,
        verified_at=datetime(2026, 11, 23, 13, 0, tzinfo=UTC),
    )
    assert hashlib.sha256(outcomes.read_bytes()).hexdigest() == old_source
    assert report["sourceImpactCount"] == 4
    assert report["newDirectEquities"] == 2
    assert {row["target"] for row in new} == {"Nvidia", "Capita"}
    assert {row["symbol"] for row in new} == {"NVDA", "CPI.L"}
    assert {row["market"] for row in new} == {"XNAS", "XLON"}
    assert report["quoteSymbols"] == ["CPI.L", "ISF.L", "NVDA", "SPY"]
    assert report["newRealPricesFetched"] is False
    assert len(mapped) == 3
    assert report["trainingEligible"] is False
    assert report["prospectivePredictionAccuracy"] is None


def test_dont_infer_instrument_when_named_company_had_no_valid_direction(
    tmp_path: Path,
) -> None:
    outcomes, mapping, prices = fixtures(tmp_path)
    doc = json.loads(outcomes.read_text(encoding="utf-8"))
    doc["items"][0]["prediction"] = "UNCERTAIN"
    doc["items"][1]["sourceAuditStatus"] = "NEWS_ORIGINAL_INCOMPLETE_OR_DUPLICATE"
    outcomes.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="No source-admitted"):
        prepare_direct_equities(
            outcomes, mapping, prices, verified_at=datetime.now(UTC),
        )


def test_no_provider_call_or_db_write_in_planning_mode(tmp_path: Path) -> None:
    outcomes, mappings, prices = fixtures(tmp_path)
    db = tmp_path / "news.sqlite3"
    db.write_bytes(b"fake db deliberately unused for plan")
    original = db.read_bytes()
    folder = tmp_path / "new-direct-research"
    with patch(
        "tris_market_event_intelligence.listed_equity_price_coverage_cli.collect_prices"
    ) as collect:
        report = run(
            outcomes, db, mappings, prices, folder, model="gemma4:e2b",
        )
    collect.assert_not_called()
    assert db.read_bytes() == original
    assert report["newRealPricesFetched"] is False
    assert (folder / "coverage-plan.json").is_file()
    assert (folder / "new-listed-quote-targets.csv").is_file()
    with pytest.raises(ValueError, match="already exists"):
        run(outcomes, db, mappings, prices, folder, model="gemma4:e2b")


def test_real_quote_snapshot_recovers_complete_instrument_without_date_splice(
    tmp_path: Path,
) -> None:
    outcomes, mappings, prices = fixtures(tmp_path)
    db = tmp_path / "news.sqlite3"
    db.write_bytes(b"synthetic source for mocked evaluation")
    with patch.object(coverage, "collect_prices") as collect, patch.object(
        coverage, "evaluate_automatic_outcomes"
    ) as evaluate, patch.object(coverage.importlib, "import_module") as yf:
        collect.return_value = ([
            {
                "symbol": symbol, "date": day, "adjusted_close": str(price),
                "source": "synthetic adjusted fixture",
            }
            for symbol, price in (
                ("CPI.L", 80), ("ISF.L", 800), ("NVDA", 200), ("SPY", 450)
            )
            for day in ("2026-11-19", "2026-11-20")
        ], {
            "missingSymbols": [], "researchOnly": True,
            "pointInTimeArchive": False,
        })
        reevaluated = json.loads(outcomes.read_text(encoding="utf-8"))
        reevaluated["items"][0]["status"] = "COMPLETED"
        reevaluated["items"][0]["horizons"] = [
            {"sessions": horizon, "verdict": "CORRECT", "dataStatus": "COMPLETED"}
            for horizon in (1, 5)
        ]
        evaluate.return_value = reevaluated
        folder = tmp_path / "new-research"
        r = run(
            outcomes, db, mappings, prices, folder, model="gemma4:e2b",
            fetch=True, start=date(2026, 11, 19),
            end=date(2026, 11, 22),
        )
    assert yf.called
    assert r["newRealPricesFetched"] is True
    assert r["newDirectionScoresObserved"] == 1
    assert r["commonIdenticalSourceImpacts"] == 4
    assert r["cohortsIdentical"] is True
    assert r["quoteRows"] == 8
    assert (folder / "expanded-adjusted-daily.csv").is_file()
    with (folder / "expanded-adjusted-daily.csv").open(newline="") as stream:
        actual = list(csv.DictReader(stream))
    assert sum(row["symbol"] == "EXMP" for row in actual) == 1
    assert sum(row["symbol"] == "SPY" for row in actual) == 2
    assert (folder / "automatic-observed-outcomes.json").is_file()
    assert (folder / "expanded-adjusted-daily.csv.manifest.json").is_file()


def test_current_unfinished_utc_daily_bar_cannot_be_fetched(
    tmp_path: Path,
) -> None:
    outcomes, mappings, prices = fixtures(tmp_path)
    with pytest.raises(ValueError, match="unfinished"):
        run(
            outcomes, tmp_path / "news.sqlite3", mappings, prices,
            tmp_path / "must-not-create", model="gemma4:e2b",
            fetch=True, start=date(2026, 11, 1), end=date(2099, 1, 1),
        )
    assert not (tmp_path / "must-not-create").exists()
