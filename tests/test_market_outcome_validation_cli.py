"""Regression tests for the Gemma-to-real-price outcome evaluation bridge."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.market_outcome_validation_cli import evaluate


def _csv(path: Path, fields: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def _fixture(tmp_path: Path):
    folder = tmp_path / "gemma"
    folder.mkdir()
    body = {
        "schemaVersion": "gemma_original_article_review_v2",
        "model": "gemma4:e2b",
        "processedAt": "2026-01-05T20:00:00+00:00",
        "article": {
            "articleId": 1, "title": "Example", "publishedAt": "2026-01-05T15:00:00+00:00",
        },
        "analysis": {
            "impacts": [{
                "target": "Example Corp", "direction": "UP",
                "reason": "Demand expected to increase", "evidence_quote": "Original text",
            }, {
                "target": "Unknown", "direction": "DOWN",
                "reason": "Possible risk", "evidence_quote": "Original text",
            }],
        },
        "predictionSealedBeforeOutcome": False,
    }
    (folder / "1.json").write_text(json.dumps(body), encoding="utf-8")
    mappings = tmp_path / "mappings.csv"
    _csv(mappings, ["target", "market", "symbol", "benchmark", "verified_at"], [
        ["Example Corp", "XNAS", "EXMP", "SPY", "2026-01-01T00:00:00+00:00"],
    ])
    prices = tmp_path / "prices.csv"
    _csv(prices, ["symbol", "date", "adjusted_close", "source"], [
        ["EXMP", "2026-01-06", "100", "test sourced fixture"],
        ["EXMP", "2026-01-07", "103", "test sourced fixture"],
        ["EXMP", "2026-01-08", "104", "test sourced fixture"],
        ["EXMP", "2026-01-09", "105", "test sourced fixture"],
        ["EXMP", "2026-01-12", "106", "test sourced fixture"],
        ["EXMP", "2026-01-13", "107", "test sourced fixture"],
        ["SPY", "2026-01-06", "200", "test sourced fixture"],
        ["SPY", "2026-01-07", "201", "test sourced fixture"],
        ["SPY", "2026-01-08", "202", "test sourced fixture"],
        ["SPY", "2026-01-09", "203", "test sourced fixture"],
        ["SPY", "2026-01-12", "204", "test sourced fixture"],
        ["SPY", "2026-01-13", "205", "test sourced fixture"],
    ])
    return folder, mappings, prices


def test_predictions_join_to_prices_and_preserve_unmapped(tmp_path):
    folder, mappings, prices = _fixture(tmp_path)
    result = evaluate(folder, mappings, prices, price_source_description="fixture")
    assert result["articlesInspected"] == 1
    assert result["impactsInspected"] == 2
    assert result["metrics"]["1d"]["correct"] == 1
    assert result["metrics"]["5d"]["correct"] == 1
    assert result["metrics"]["1d"]["accuracy"] == 1.0
    item, unmapped = result["items"]
    assert item["entryDay"] == "2026-01-06"
    assert item["evaluationClass"] == "RETROSPECTIVE_UNSEALED"
    assert not item["predictionSealedBeforeOutcome"]
    assert item["horizons"][0]["stockReturn"] == pytest.approx(0.03)
    assert item["horizons"][0]["benchmarkReturn"] == pytest.approx(0.005)
    assert item["horizons"][0]["excessReturn"] == pytest.approx(0.025)
    assert item["predictionSourceSha256"] == hashlib.sha256(
        (folder / "1.json").read_bytes()
    ).hexdigest()
    assert unmapped["status"] == "UNMAPPED"


def test_later_verified_mapping_is_explicitly_retrospective(tmp_path):
    folder, mappings, prices = _fixture(tmp_path)
    _csv(mappings, ["target", "market", "symbol", "benchmark", "verified_at"], [
        ["Example Corp", "XNAS", "EXMP", "SPY", "2026-02-01T00:00:00+00:00"],
    ])
    result = evaluate(folder, mappings, prices, price_source_description="fixture")
    assert result["items"][0]["status"] == "COMPLETED"
    assert result["items"][0]["mappingPointInTime"] is False
    assert result["items"][0]["evaluationClass"] == "RETROSPECTIVE_POST_EVENT_MAPPING"
    assert result["metrics"]["1d"]["accuracy"] == 1.0


def test_prices_must_be_provenanced_and_positive(tmp_path):
    folder, mappings, prices = _fixture(tmp_path)
    _csv(prices, ["symbol", "date", "adjusted_close", "source"], [
        ["EXMP", "2026-01-06", "0", ""],
    ])
    with pytest.raises(ValueError):
        evaluate(folder, mappings, prices, price_source_description="fixture")


def test_duplicate_article_ids_rejected(tmp_path):
    folder, mappings, prices = _fixture(tmp_path)
    (folder / "duplicate.json").write_bytes((folder / "1.json").read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate(folder, mappings, prices, price_source_description="fixture")
