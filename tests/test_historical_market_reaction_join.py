"""Historical market reaction joins never manufacture negative/important labels."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from tris_market_event_intelligence.historical_market_reaction_join import (
    join_historical_market_reactions,
)

RECEIVED = "2026-11-13T09:00:30+09:00"
CUTOFF = "2026-11-13T16:00:00+09:00"


def _csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def fixture(tmp_path: Path, *, link_at: str = "2026-11-13T09:00:20+09:00") -> dict:
    cohort = tmp_path / "cohort.json"
    items = [{
        "articleId": article_id,
        "sourceUrl": f"https://example.org/{article_id}",
        "firstObservedAt": RECEIVED,
        "captureChecksumVerified": True,
        "importanceHumanGold": "UNKNOWN",
        "observedMarketReaction": "UNMEASURED",
        "trainingEligible": False,
        "negativeSampleConfirmed": False,
        "marketPriceLabel": None,
    } for article_id in (1, 2)]
    cohort.write_text(json.dumps({
        "schemaVersion": "historical_news_cohort_v1",
        "status": "REVIEW_ONLY_NOT_TRAINING_READY",
        "allNewsIncludedRegardlessOfExtraction": True,
        "trainingDatasetEligible": False,
        "sourceDbModified": False,
        "asOf": CUTOFF,
        "items": items,
    }), encoding="utf-8")
    links = tmp_path / "links.csv"
    _csv(links, [
        "event_id", "market", "symbol", "benchmark_symbol",
        "source_url", "evidence_seen_at", "relationship",
    ], [["1", "XKRX", "005930", "KOSPI", "https://example.org/1",
         link_at, "DIRECT"]])
    sessions = tmp_path / "sessions.csv"
    _csv(sessions, ["market", "date", "status", "open_at", "close_at"], [
        ["XKRX", "2026-11-13", "OPEN", "2026-11-13T09:00:00+09:00",
         "2026-11-13T15:30:00+09:00"],
    ])
    sources = []
    for name, kind, symbol, prices in [
        ("stock.csv", "STOCK", "005930", [100, 101, 102]),
        ("benchmark.csv", "BENCHMARK", "KOSPI", [100, 100, 100]),
    ]:
        path = tmp_path / name
        _csv(path, [
            "market", "symbol", "close_at", "adjusted_close", "adjustment",
            "instrument_kind", "provider", "observed_at",
        ], [
            ["XKRX", symbol, f"2026-11-13T09:0{i}:00+09:00",
             str(price), "adjusted", kind, "fixture-only",
             f"2026-11-13T09:0{i}:10+09:00"]
            for i, price in enumerate(prices, start=1)
        ])
        sources.append({
            "filename": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "market": "XKRX", "provider": "fixture-only", "source_kind": kind,
        })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "minute_source_v1", "sources": sources,
    }), encoding="utf-8")
    return {
        "cohort_path": cohort, "links_path": links,
        "minute_manifest": manifest, "sessions_path": sessions,
    }


def test_observed_return_and_unimportant_news_not_assumed(tmp_path: Path) -> None:
    kwargs = fixture(tmp_path)
    originals = {k: path.read_bytes() for k, path in kwargs.items()}
    report = join_historical_market_reactions(**kwargs)
    assert report["articlesInCohortPage"] == 2
    assert report["articlesWithPointInTimeLinks"] == 1
    assert report["labelCounts"]["OBSERVED_EXCESS_RETURN_UNVERIFIED"] == 1
    assert report["labelCounts"]["ARTICLES_WITHOUT_MEASURED_LINK"] == 1
    first, second = report["items"]
    assert first["marketReactionStatus"] == "OBSERVED_RESEARCH_ONLY"
    labels = first["marketObservations"][0]["labels"]
    assert labels[0]["status"] == "COMPLETED"
    assert labels[0]["horizonMinutes"] == 1
    assert labels[0]["excessReturn"] == pytest.approx(102 / 101 - 1)
    assert any(label["status"] == "MISSING_MINUTE_BAR" for label in labels)
    assert second["marketObservations"] == []
    assert second["marketReactionStatus"] == "UNMEASURED"
    assert second["importanceHumanGold"] == "UNKNOWN"
    assert second["unmatchedIsUnimportant"] is False
    assert report["priceSourceAuthenticated"] is False
    assert report["trainingDatasetEligible"] is False
    assert all(path.read_bytes() == originals[k] for k, path in kwargs.items())


def test_late_link_is_blocked_not_backfilled(tmp_path: Path) -> None:
    kwargs = fixture(tmp_path, link_at="2026-11-13T09:01:00+09:00")
    result = join_historical_market_reactions(**kwargs)
    assert result["blockedLinks"] == {"LINK_KNOWN_AFTER_NEWS": 1}
    assert result["articlesWithPointInTimeLinks"] == 0
    assert result["labelCounts"]["ARTICLES_WITHOUT_MEASURED_LINK"] == 2
    assert all(row["marketReactionStatus"] == "UNMEASURED" for row in result["items"])


def test_link_to_different_article_is_rejected(tmp_path: Path) -> None:
    kwargs = fixture(tmp_path)
    links = kwargs["links_path"]
    links.write_text(links.read_text().replace("example.org/1", "example.org/2"))
    with pytest.raises(ValueError, match="Link URL"):
        join_historical_market_reactions(**kwargs)


def test_unknown_article_rejected(tmp_path: Path) -> None:
    kwargs = fixture(tmp_path)
    links = kwargs["links_path"]
    links.write_text(links.read_text().replace("1,XKRX", "9,XKRX"))
    with pytest.raises(ValueError, match="absent cohort"):
        join_historical_market_reactions(**kwargs)


def test_wrong_market_data_sha_fails_closed(tmp_path: Path) -> None:
    kwargs = fixture(tmp_path)
    stock = tmp_path / "stock.csv"
    stock.write_bytes(stock.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        join_historical_market_reactions(**kwargs)


def test_unsafe_source_and_outcome_labels_fail_closed(tmp_path: Path) -> None:
    kwargs = fixture(tmp_path)
    cohort = kwargs["cohort_path"]
    doc = json.loads(cohort.read_text())
    doc["items"][1]["importanceHumanGold"] = "UNIMPORTANT"
    cohort.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="inferred outcome labels"):
        join_historical_market_reactions(**kwargs)


def test_future_observed_bars_do_not_create_retrospective_label(tmp_path: Path) -> None:
    kwargs = fixture(tmp_path)
    cohort = kwargs["cohort_path"]
    doc = json.loads(cohort.read_text())
    doc["asOf"] = "2026-11-13T09:01:30+09:00"
    cohort.write_text(json.dumps(doc))
    report = join_historical_market_reactions(**kwargs)
    assert report["labelCounts"].get("OBSERVED_EXCESS_RETURN_UNVERIFIED", 0) == 0
    assert report["items"][0]["marketReactionStatus"] == "UNMEASURED"


def test_timezone_is_required_and_inputs_not_created(tmp_path: Path) -> None:
    kwargs = fixture(tmp_path)
    cohort = kwargs["cohort_path"]
    doc = json.loads(cohort.read_text())
    doc["asOf"] = "2026-11-13T16:00:00"
    cohort.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="timezone"):
        join_historical_market_reactions(**kwargs)
    assert not (tmp_path / "production.sqlite3").exists()
