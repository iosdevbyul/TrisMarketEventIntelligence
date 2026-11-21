"""Independent absolute/relative labels and explicit proxy mapping boundaries."""
from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_market_outcome_validation_cli import _fixture

from tris_market_event_intelligence.market_outcome_validation_cli import evaluate
from tris_market_event_intelligence.market_proxy_cohort_cli import prepare_cohort


def test_market_relative_verdict_and_unmapped_denominator(tmp_path: Path) -> None:
    folder, mappings, prices = _fixture(tmp_path)
    with prices.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["symbol", "date", "adjusted_close", "source"])
        for date, equity, benchmark in [
            ("2026-01-06", 100, 200),
            ("2026-01-07", 99, 195),
            ("2026-01-08", 99, 195),
            ("2026-01-09", 98, 194),
            ("2026-01-12", 97, 193),
            ("2026-01-13", 96, 192),
        ]:
            writer.writerow(["EXMP", date, equity, "test"])
            writer.writerow(["SPY", date, benchmark, "test"])
    report = evaluate(folder, mappings, prices, price_source_description="test data")
    target = report["items"][0]
    first = target["horizons"][0]
    assert first["verdict"] == "INCORRECT"
    assert first["relativeVerdict"] == "CORRECT"
    assert first["actualDirection"] == "DOWN"
    assert first["relativeActualDirection"] == "UP"
    assert report["metrics"]["1d"]["incorrect"] == 1
    assert report["metrics"]["1d"]["relative"]["correct"] == 1
    assert report["metrics"]["1d"]["unavailable"] == 1
    assert report["metrics"]["1d"]["totalImpacts"] == 2
    assert report["metrics"]["1d"]["coverage"] == pytest.approx(0.5)


def test_source_backed_sector_proxies_only_and_omission_audit(tmp_path: Path) -> None:
    folder, _, _ = _fixture(tmp_path)
    article_file = folder / "1.json"
    article = json.loads(article_file.read_text())
    article["analysis"]["impacts"].extend([
        {
            "target": "Technology Sector", "direction": "UP",
            "reason": "Research", "evidence_quote": "E",
        },
        {
            "target": "Inflation", "direction": "UP",
            "reason": "Not an equity", "evidence_quote": "E",
        },
    ])
    article_file.write_text(json.dumps(article))
    mappings, audit = prepare_cohort(
        folder, verified_at=datetime(2026, 11, 21, tzinfo=UTC),
    )
    assert len(mappings) == 1
    assert mappings[0]["symbol"] == "XLK"
    assert mappings[0]["mapping_kind"] == "SECTOR_ETF_PROXY"
    assert mappings[0]["proxy_limitation"]
    assert mappings[0]["evidence_url"].startswith("https://")
    assert audit["mappedImpacts"] == 1
    assert audit["impactsInspected"] == 4
    assert {entry["target"] for entry in audit["omitted"]} == {
        "Example Corp", "Unknown", "Inflation",
    }


def test_sector_proxy_requires_documented_limitation(tmp_path: Path) -> None:
    folder, mappings, prices = _fixture(tmp_path)
    with mappings.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "target", "market", "symbol", "benchmark", "verified_at",
            "mapping_kind", "proxy_limitation", "evidence_url",
        ])
        writer.writerow([
            "Example Corp", "XNYS", "XLK", "SPY",
            "2026-11-21T00:00:00+00:00", "SECTOR_ETF_PROXY", "", "",
        ])
    with pytest.raises(ValueError, match="requires limitation"):
        evaluate(folder, mappings, prices, price_source_description="test data")
