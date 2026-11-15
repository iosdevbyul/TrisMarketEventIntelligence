"""Historic shock/research data quality and PIT macro/news isolation."""
from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path

import pytest
from test_news_research_worker import files

from tris_market_event_intelligence.historical_shock_research import (
    build_historical_shock_research,
    read_daily_prices,
    read_macro_releases,
)


def _csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _setup(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    stock, benchmark, macro = (tmp_path / "stock.csv", tmp_path / "bench.csv",
                               tmp_path / "macro.csv")
    dates = []
    day = date(2026, 10, 1)
    while len(dates) < 45:
        if day.weekday() < 5:
            dates.append(day)
        day += timedelta(days=1)
    stock_rows = []
    bench_rows = []
    price = 100.0
    index_price = 100.0
    for i, stamp in enumerate(dates):
        price *= 1.25 if i == 14 else 1.01
        index_price *= 1.004
        stock_rows.append({
            "date": stamp.isoformat(), "adjusted_close": f"{price:.8f}",
            "volume": "120000",
        })
        bench_rows.append({
            "date": stamp.isoformat(), "adjusted_close": f"{index_price:.8f}",
            "volume": "100000",
        })
    fields = ["date", "adjusted_close", "volume"]
    _csv(stock, fields, stock_rows)
    _csv(benchmark, fields, bench_rows)
    _csv(macro, [
        "series_id", "observation_date", "vintage_date", "release_at",
        "available_at", "value", "source",
    ], [
        {
            "series_id": "CPI", "observation_date": "2026-09-01",
            "vintage_date": "2026-10-01",
            "release_at": "2026-10-01T12:30:00+00:00",
            "available_at": "2026-10-01T12:31:00+00:00",
            "value": "3.0", "source": "TEST",
        },
        {
            "series_id": "CPI", "observation_date": "2026-09-01",
            "vintage_date": "2026-11-01",
            "release_at": "2026-11-01T12:30:00+00:00",
            "available_at": "2026-11-01T12:31:00+00:00",
            "value": "2.9", "source": "TEST",
        },
        {
            "series_id": "JOBS", "observation_date": "2026-09-01",
            "vintage_date": "2026-10-15",
            "release_at": "2026-10-15T14:00:00+00:00",
            "available_at": "2026-10-15T14:01:00+00:00",
            "value": "42", "source": "TEST",
        },
    ])
    return stock, benchmark, macro, files(tmp_path)[0]


def test_historic_top_controls_and_macro_vintage_no_lookahead(tmp_path: Path) -> None:
    stock, bench, macro, news = _setup(tmp_path)
    before = [p.read_bytes() for p in (stock, bench, macro, news)]
    report = build_historical_shock_research(
        symbol="NVDA", benchmark_symbol="SPY", stock_path=stock,
        benchmark_path=bench, news_cohort_path=news, macro_path=macro,
        top=2, controls=2,
    )
    assert report["status"] == "RESEARCH_PACKET_UNVERIFIED"
    assert len(report["topEvents"]) == 2
    assert len(report["controlEvents"]) == 2
    assert not {x["date"] for x in report["topEvents"]} & {
        x["date"] for x in report["controlEvents"]
    }
    assert report["topEvents"][0]["date"] == "2026-10-21"
    pre = report["topEvents"][0]["context"]
    assert pre["newsAvailableBeforeOpen"] == []
    assert pre["macroPointInTime"][0]["seriesId"] == "CPI"
    assert pre["macroPointInTime"][0]["value"] == 3.0
    assert pre["macroPointInTime"][0]["vintageDate"] == "2026-10-01"
    assert report["llmCalls"] == 0
    assert report["modelPromotionAllowed"] is False
    assert report["newsAndMacroCompleteForTenYears"] is False
    assert [p.read_bytes() for p in (stock, bench, macro, news)] == before


def test_macro_future_release_and_duplicate_revision_rejected(tmp_path: Path) -> None:
    stock, _, macro, _ = _setup(tmp_path)
    rows = list(csv.DictReader(macro.read_text().splitlines()))
    rows[0]["available_at"] = "2026-09-01T00:00:00+00:00"
    _csv(macro, list(rows[0]), rows)
    with pytest.raises(ValueError, match="Future macro"):
        read_macro_releases(macro)
    rows[0]["available_at"] = "2026-10-01T12:31:00+00:00"
    rows.append(rows[0].copy())
    _csv(macro, list(rows[0]), rows)
    with pytest.raises(ValueError, match="Duplicated macro vintage"):
        read_macro_releases(macro)
    assert len(read_daily_prices(stock)[0]) == 45


def test_untrusted_price_files_fail_closed(tmp_path: Path) -> None:
    stock, _, _, _ = _setup(tmp_path)
    rows = list(csv.DictReader(stock.read_text().splitlines()))
    rows[2]["adjusted_close"] = "0"
    _csv(stock, list(rows[0]), rows)
    with pytest.raises(ValueError, match="Invalid adjusted"):
        read_daily_prices(stock)
    rows[2]["adjusted_close"] = "1.0"
    rows[3]["date"] = rows[2]["date"]
    _csv(stock, list(rows[0]), rows)
    with pytest.raises(ValueError, match="strictly increasing"):
        read_daily_prices(stock)


def test_historical_packet_does_not_claim_news_caused_price_move(tmp_path: Path) -> None:
    stock, bench, macro, news = _setup(tmp_path)
    doc = json.loads(news.read_text())
    item = doc["items"][0]
    # _cohort will enforce pinned text and as-of source integrity.
    assert item["firstObservedAt"] >= "2026-11-01"
    packet = build_historical_shock_research(
        symbol="NVDA", benchmark_symbol="SPY", stock_path=stock,
        benchmark_path=bench, news_cohort_path=news, macro_path=macro,
        top=1, controls=1,
    )
    assert packet["topEvents"][0]["causalNewsAttributionVerified"] is False
    assert packet["topEvents"][0]["context"]["newsIssuerLinksVerified"] is False
    assert packet["forwardPredictionSealed"] is False
    assert packet["humanGoldCreated"] is False
