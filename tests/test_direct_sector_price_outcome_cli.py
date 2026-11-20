"""Actual adjusted-close retrospective comparison with explicit sector mapping."""
import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.direct_sector_price_outcome_cli import (
    evaluate_sector_prices,
    read_manifest,
)


def price_file(path: Path, prices: list[int]) -> None:
    days = [
        "2026-08-13", "2026-08-14", "2026-08-17",
        "2026-08-18", "2026-08-19", "2026-08-20",
        "2026-08-21",
    ]
    data = "date,adjusted_close,volume\n"
    data += "".join(f"{date},{price},1000\n" for date, price in zip(days, prices))
    path.write_text(data, encoding="utf-8")


def fixtures(tmp_path: Path):
    price_file(tmp_path / "sector.csv", [100, 101, 110, 112, 113, 114, 116])
    price_file(tmp_path / "market.csv", [100, 101, 102, 103, 104, 105, 106])
    config = {
        "schemaVersion": "sector_price_manifest_v1",
        "market": "XNAS",
        "benchmark": {"symbol": "MARKET", "csv": "market.csv"},
        "sectors": {"SEMICONDUCTORS": {"symbol": "SECTOR", "csv": "sector.csv"}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    parsed, series = read_manifest(path)
    return parsed, series


def prediction(timestamp="2026-08-14T17:00:00-04:00", direction="POSITIVE"):
    return {
        "schemaVersion": "direct_news_sector_impact_v1",
        "originalHoldoutSha256": "sealed",
        "priceDataAccessed": False,
        "originalPredictionsModified": False,
        "articles": [{
            "articleId": 117,
            "publishedAtReported": timestamp,
            "sectors": [{
                "sector": "SEMICONDUCTORS", "direction": direction,
                "sourceQuote": "반도체 경기 호황과 파급효과 확산",
            }],
        }],
    }


def test_real_price_comparison_uses_previous_close_and_benchmark(tmp_path):
    manifest, series = fixtures(tmp_path)
    report = evaluate_sector_prices(prediction(), manifest, series)
    assert report["statusCounts"] == {"RETROSPECTIVE_RETURNS_COMPUTED": 1}
    row = report["records"][0]
    h1 = row["outcomes"]["1"]
    assert h1["baselineCloseSession"] == "2026-08-14"
    assert h1["outcomeCloseSession"] == "2026-08-17"
    assert h1["sectorReturn"] == pytest.approx(110 / 101 - 1)
    assert h1["benchmarkReturn"] == pytest.approx(102 / 101 - 1)
    assert h1["directionMatched"] is True
    assert report["horizonComparisonCounts"]["3"]["matched"] == 1
    assert report["retrospectiveEvaluationOnly"] is True
    assert report["pointInTimeCertified"] is False
    assert report["causalImpactVerified"] is False


def test_negative_hypothesis_is_diagnostic_not_expost_label(tmp_path):
    manifest, series = fixtures(tmp_path)
    report = evaluate_sector_prices(prediction(direction="NEGATIVE"), manifest, series)
    assert report["records"][0]["outcomes"]["1"]["directionMatched"] is False
    assert report["horizonComparisonCounts"]["1"] == {"observed": 1, "matched": 0}
    assert report["originalPredictionSealedBeforePricesIndependentlyVerified"] is False


def test_missing_publication_clock_and_mapping_cannot_invent_returns(tmp_path):
    manifest, series = fixtures(tmp_path)
    missing_date = evaluate_sector_prices(prediction(timestamp=None), manifest, series)
    assert missing_date["statusCounts"] == {"PUBLISHER_CLOCK_NOT_PROVIDED": 1}
    absent_sector = prediction()
    absent_sector["articles"][0]["sectors"][0]["sector"] = "ENERGY"
    report = evaluate_sector_prices(absent_sector, manifest, series)
    assert report["statusCounts"] == {"SECTOR_PRICE_SERIES_NOT_CONFIGURED": 1}
    assert report["horizonComparisonCounts"]["1"]["observed"] == 0


def test_non_directional_hypothesis_excluded_from_backtest(tmp_path):
    manifest, series = fixtures(tmp_path)
    report = evaluate_sector_prices(prediction(direction="UNKNOWN"), manifest, series)
    assert report["statusCounts"] == {"NON_DIRECTIONAL_HYPOTHESIS": 1}
    assert report["horizonComparisonCounts"]["1"]["observed"] == 0


def test_unpinned_forecast_contract_rejected(tmp_path):
    manifest, series = fixtures(tmp_path)
    invalid = prediction()
    invalid["priceDataAccessed"] = True
    with pytest.raises(ValueError, match="DIRECT_ANALYSIS_CONTRACT_INVALID"):
        evaluate_sector_prices(invalid, manifest, series)


def test_in_session_article_marks_baseline_as_after_release(tmp_path):
    manifest, series = fixtures(tmp_path)
    report = evaluate_sector_prices(
        prediction(timestamp="2026-08-14T12:00:00-04:00"), manifest, series,
    )
    assert report["records"][0]["baselineCloseMayPostdatePublication"] is True


def test_after_close_article_baseline_does_not_postdate_release(tmp_path):
    manifest, series = fixtures(tmp_path)
    report = evaluate_sector_prices(prediction(), manifest, series)
    assert report["records"][0]["baselineCloseMayPostdatePublication"] is False


def test_reject_unknown_market_and_fabricated_sector_mapping(tmp_path):
    fixtures(tmp_path)
    manifest_file = tmp_path / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["sectors"] = {"FANTASY": {"symbol": "FAKE", "csv": "sector.csv"}}
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="PRICE_MANIFEST_MAPPING_INVALID"):
        read_manifest(manifest_file)
