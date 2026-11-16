"""All-session SEC-evidence research records are not NEWS-negative controls."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.historical_sec_event_universe_cli import (
    build_sec_event_universe,
)


def _source_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    stock = tmp_path / "nvda.csv"
    bench = tmp_path / "spy.csv"
    stock.write_text(
        "date,adjusted_close,volume\n"
        "2018-11-13,12,100\n2018-11-14,11,100\n"
        "2018-11-15,10,100\n2018-11-16,8,100\n"
        "2018-11-19,9,100\n2018-11-20,8.5,100\n"
        "2018-11-21,9.5,100\n"
    )
    bench.write_text(
        "date,adjusted_close,volume\n"
        "2018-11-13,100,100\n2018-11-14,101,100\n"
        "2018-11-15,100,100\n2018-11-16,99,100\n"
        "2018-11-19,100,100\n2018-11-20,101,100\n"
        "2018-11-21,102,100\n"
    )
    from tris_market_event_intelligence.historical_shock_research import read_daily_prices

    _, stock_sha = read_daily_prices(stock)
    _, market_sha = read_daily_prices(bench)
    event = tmp_path / "sec-event.json"
    event.write_text(json.dumps({
        "schemaVersion": "historical_sec_earnings_price_event_v1",
        "status": "SEC_FILING_PREOPEN_PRICE_OUTCOME_OBSERVED_RETROSPECTIVE",
        "ticker": "NVDA",
        "benchmark": "SPY",
        "eventId": "NVDA-FY2019-Q3-2018-11-15",
        "researchMode": "RETROSPECTIVE_OUTCOME_ALREADY_KNOWN",
        "stockPriceSha256": stock_sha,
        "benchmarkPriceSha256": market_sha,
        "secEvidenceSha256": "a" * 64,
        "corroborationSha256": "b" * 64,
        "decisionAt": "2018-11-16T14:30:00+00:00",
        "modelPredictionMadeBeforeOutcome": False,
        "trainingReady": False,
        "causalImpactVerified": False,
        "modelPromotionAllowed": False,
        "secAcceptanceDoesNotProvePublisherFirstRelease": True,
        "secAcceptanceDoesNotProveFirstMarketAvailability": True,
        "features": {
            "sourceType": "SEC_8K_WITH_ISSUER_EXHIBIT",
            "secAcceptedAt": "2018-11-15T21:21:48+00:00",
            "sourceAvailableByDecisionCutoff": True,
            "secAccession": "0001045810-18-000148",
            "filingUrl": (
                "https://www.sec.gov/Archives/edgar/data/1045810/"
                "0001045810-18-000148-index.htm"
            ),
            "exhibitUrl": (
                "https://www.sec.gov/Archives/edgar/data/1045810/"
                "000104581018000148/q3fy19pr.htm"
            ),
            "issuerFacts": [{
                "id": "revenue", "nature": "REPORTED_RESULT",
                "statement": "NVIDIA third-quarter revenue USD 3.18bn",
            }],
            "publisherFirstReleaseTimeVerified": False,
            "gkgIssuerLinksVerified": False,
            "previousSession": "2018-11-15",
            "previousStockAdjustedClose": 10,
            "previousBenchmarkAdjustedClose": 100,
        },
        "outcome": {
            "session": "2018-11-16",
            "returnWindow": "PREVIOUS_CLOSE_TO_EVENT_DAY_CLOSE",
            "stockReturn": -0.2,
            "benchmarkReturn": -0.01,
            "excessReturn": -0.19,
            "actualOutcomeUnavailableAtDecision": True,
            "notAnExecutablePreopenEntryReturn": True,
        },
    }))
    return stock, bench, event


def _run(tmp_path: Path) -> dict:
    stock, bench, event = _source_files(tmp_path)
    return build_sec_event_universe(
        stock_csv=stock, benchmark_csv=bench, event_paths=[event],
    )


def test_entire_universe_without_faking_news_negative_days(tmp_path: Path) -> None:
    report = _run(tmp_path)
    assert report["eventCount"] == 1
    assert report["sessionCount"] == 6
    assert report["sessionsWithPinnedSecEvent"] == 1
    assert report["sessionsWithoutPinnedSecEvent"] == 5
    assert report["nonEventSessionsAreTrueNoNewsControls"] is False
    assert report["trainingReady"] is False
    items = {row["session"]: row for row in report["records"]}
    event = items["2018-11-16"]
    assert event["features"]["pinnedSecEvents"][0]["secAcceptedAt"] == (
        "2018-11-15T21:21:48+00:00"
    )
    assert event["features"]["previousSession"] == "2018-11-15"
    assert event["outcome"]["horizonsTradingSessions"]["1"]["excessReturn"] == (
        pytest.approx(-0.19)
    )
    assert event["outcome"]["horizonsTradingSessions"]["3"]["targetSession"] == (
        "2018-11-20"
    )
    assert "excessReturn" not in json.dumps(event["features"])
    normal = items["2018-11-19"]
    assert normal["features"]["pinnedSecEvents"] == []
    assert normal["features"]["secEvidenceStatus"] == "NO_PINNED_SEC_EVIDENCE"
    assert normal["features"]["otherCompanyOrMacroNewsAbsenceVerified"] is False


def test_event_tampering_cannot_inject_outcome_as_features(tmp_path: Path) -> None:
    stock, bench, event = _source_files(tmp_path)
    packet = json.loads(event.read_text())
    packet["outcome"]["excessReturn"] = -0.01
    event.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="price truth mismatch"):
        build_sec_event_universe(
            stock_csv=stock, benchmark_csv=bench, event_paths=[event],
        )


def test_future_sec_clock_cannot_be_promoted(tmp_path: Path) -> None:
    stock, bench, event = _source_files(tmp_path)
    packet = json.loads(event.read_text())
    packet["features"]["secAcceptedAt"] = "2018-11-16T14:31:00+00:00"
    event.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="cutoff"):
        build_sec_event_universe(
            stock_csv=stock, benchmark_csv=bench, event_paths=[event],
        )


def test_rejects_duplicate_event_and_missing_price_corporate_action(tmp_path: Path) -> None:
    stock, bench, event = _source_files(tmp_path)
    with pytest.raises(ValueError, match="Duplicated"):
        build_sec_event_universe(
            stock_csv=stock, benchmark_csv=bench, event_paths=[event, event],
        )
    stock.write_text(stock.read_text().replace("2018-11-16,8,100", "2018-11-16,7,100"))
    with pytest.raises(ValueError, match="source lineage"):
        build_sec_event_universe(
            stock_csv=stock, benchmark_csv=bench, event_paths=[event],
        )


def test_missing_news_coverage_is_not_gold_and_source_unchanged(tmp_path: Path) -> None:
    stock, bench, event = _source_files(tmp_path)
    originals = [path.read_bytes() for path in (stock, bench, event)]
    report = build_sec_event_universe(
        stock_csv=stock, benchmark_csv=bench, event_paths=[event],
    )
    assert [path.read_bytes() for path in (stock, bench, event)] == originals
    assert report["actualModelSkillEstablished"] is False
    assert report["evidenceCoverageComplete"] is False
    assert report["forwardPredictionsSealed"] is False
    assert all(x["trainingEligible"] is False for x in report["records"])
