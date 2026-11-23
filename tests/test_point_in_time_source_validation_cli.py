"""Source integration regression fixtures only; these are NOT real exchange quotes."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tris_market_event_intelligence.point_in_time_source_validation_cli import (
    evaluate_sources,
)

KST = ZoneInfo("Asia/Seoul")


def stamp(hour: int, minute: int) -> str:
    return datetime(2026, 11, 23, hour, minute, tzinfo=KST).isoformat()


def write_csv(path: Path, headers: list[str], values: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(values)


def experiment(tmp_path: Path, *,
               missing_benchmark: bool = False, halted: bool = False,
               post_seal_capture: bool = False,
               reprint: bool = False,
               unverified: bool = False) -> dict:
    pred = tmp_path / "predictions.csv"
    headers = [
        "article_id", "family_id", "published_at", "first_public_at",
        "detected_at", "novelty", "source_url", "source_verified",
        "target", "benchmark", "direction", "forecast_at", "horizon_end",
        "horizon_name", "mode", "model_version", "rule_version", "prediction_id",
        "market",
    ]
    write_csv(pred, headers, [{
        "article_id": 1, "family_id": "story-42",
        "published_at": stamp(14, 0), "first_public_at": stamp(14, 0),
        "detected_at": stamp(14, 1), "novelty": "REPRINT" if reprint else "NEW",
        "source_url": "https://example.com/a", "source_verified": "false" if unverified else "true",
        "target": "EQUITY", "benchmark": "INDEX", "direction": "DOWN",
        "forecast_at": stamp(14, 2), "horizon_end": stamp(15, 30),
        "horizon_name": "session-close", "mode": "HISTORICAL_REPLAY",
        "model_version": "Gemma-4", "rule_version": "none",
        "prediction_id": "p1", "market": "XKRX",
    }])
    sessions = tmp_path / "sessions.csv"
    write_csv(sessions, ["market", "date", "status", "open_at", "close_at"], [{
        "market": "XKRX", "date": "2026-11-23", "status": "OPEN",
        "open_at": stamp(9, 0), "close_at": stamp(15, 30),
    }])
    price_rows = []
    for symbol, levels in [
        ("EQUITY", [110, 109, 102]),
        ("INDEX", [100, 100, 99]),
    ]:
        if missing_benchmark and symbol == "INDEX":
            continue
        for at, price in zip([(13, 59), (14, 2), (15, 30)], levels):
            price_rows.append({
                "market": "XKRX", "symbol": symbol, "close_at": stamp(*at),
                "adjusted_close": str(price), "adjustment": "adjusted",
                "instrument_kind": "STOCK", "provider": "fixture-provider",
                "observed_at": stamp(15, 31) if post_seal_capture else stamp(*at),
            })
    prices = tmp_path / "price.csv"
    write_csv(prices, ["market", "symbol", "close_at", "adjusted_close",
                       "adjustment", "instrument_kind", "provider", "observed_at"], price_rows)
    sha = hashlib.sha256(prices.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "minute_source_v1", "sources": [{
            "filename": "price.csv", "sha256": sha,
            "market": "XKRX", "provider": "fixture-provider",
            "source_kind": "STOCK",
        }],
    }), encoding="utf-8")
    halts = None
    if halted:
        halts = tmp_path / "halts.csv"
        write_csv(halts, ["market", "symbol", "start_at", "end_at"], [{
            "market": "XKRX", "symbol": "EQUITY",
            "start_at": stamp(14, 10), "end_at": stamp(14, 20),
        }])
    return evaluate_sources(pred, manifest, sessions, halts_csv=halts)


def test_real_source_shape_is_not_independent_authenticity(tmp_path: Path) -> None:
    report = experiment(tmp_path)
    assert report["rows"] == 1
    item = report["items"][0]
    assert item["status"] == "OBSERVED"
    assert item["observation"]["actual_direction"] == "DOWN"
    assert item["observation"]["direction_match"] is True
    assert item["observation"]["absolute_return"] == 102 / 110 - 1
    assert item["forecastBarsAvailableAtSeal"] is True
    assert report["sourceAudit"]["source_origin_authenticated"] is False
    assert report["marketSourceIndependentlyVerified"] is False
    assert report["trainingEligible"] is False
    assert item["prospectiveScoreCertified"] is False


def test_reprint_is_not_a_new_caused_event(tmp_path: Path) -> None:
    item = experiment(tmp_path, reprint=True)["items"][0]
    assert item["familyId"] == "story-42"
    assert item["novelty"] == "REPRINT"
    assert item["causalAttribution"] == "NOT_ESTABLISHED"


def test_halt_excludes_suspect_price_observation(tmp_path: Path) -> None:
    item = experiment(tmp_path, halted=True)["items"][0]
    assert item["status"] == "TRADING_HALT_OVERLAP"
    assert "observation" not in item


def test_missing_market_benchmark_excludes_score(tmp_path: Path) -> None:
    item = experiment(tmp_path, missing_benchmark=True)["items"][0]
    assert item["status"] == "MARKET_MAPPING_UNVERIFIED"
    assert "observation" not in item


def test_retroactive_price_download_is_not_pit_price_at_seal(tmp_path: Path) -> None:
    item = experiment(tmp_path, post_seal_capture=True)["items"][0]
    assert item["status"] == "OBSERVED"
    assert item["forecastBarsAvailableAtSeal"] is False
    assert item["prospectiveScoreCertified"] is False


def test_unverified_news_source_is_not_scored(tmp_path: Path) -> None:
    item = experiment(tmp_path, unverified=True)["items"][0]
    assert item["status"] == "SOURCE_FIRST_PUBLIC_UNVERIFIED"
    assert "observation" not in item


def test_incomplete_calendar_is_not_invented(tmp_path: Path) -> None:
    from tris_market_event_intelligence.point_in_time_source_validation_cli import (
        evaluate_sources,
    )

    experiment(tmp_path)
    session = tmp_path / "sessions.csv"
    write_csv(session, ["market", "date", "status", "open_at", "close_at"], [{
        "market": "XKRX", "date": "2026-11-24",
        "status": "CLOSED", "open_at": "", "close_at": "",
    }])
    report = evaluate_sources(
        tmp_path / "predictions.csv", tmp_path / "manifest.json", session,
    )
    assert report["items"][0]["status"] == "SESSION_UNVERIFIED"
    assert "observation" not in report["items"][0]


def test_after_close_requires_explicit_prior_and_next_sessions(tmp_path: Path) -> None:
    from tris_market_event_intelligence.point_in_time_source_validation_cli import (
        evaluate_sources,
    )

    experiment(tmp_path)
    pred = tmp_path / "predictions.csv"
    with pred.open(encoding="utf-8", newline="") as handle:
        original = list(csv.DictReader(handle))
    original[0].update({
        "published_at": stamp(16, 0),
        "first_public_at": stamp(16, 0),
        "detected_at": stamp(16, 1),
        "forecast_at": stamp(16, 2),
        "horizon_end": datetime(2026, 11, 24, 15, 30, tzinfo=KST).isoformat(),
    })
    write_csv(pred, list(original[0]), original)
    sessions = tmp_path / "sessions.csv"
    write_csv(sessions, ["market", "date", "status", "open_at", "close_at"], [
        {"market": "XKRX", "date": "2026-11-23", "status": "OPEN",
         "open_at": stamp(9, 0), "close_at": stamp(15, 30)},
        {"market": "XKRX", "date": "2026-11-24", "status": "OPEN",
         "open_at": datetime(2026, 11, 24, 9, tzinfo=KST).isoformat(),
         "close_at": datetime(2026, 11, 24, 15, 30, tzinfo=KST).isoformat()},
    ])
    prices = tmp_path / "price.csv"
    with prices.open(encoding="utf-8", newline="") as handle:
        original_prices = list(csv.DictReader(handle))
    original_prices = [
        r for r in original_prices if r["close_at"] == stamp(15, 30)
    ]
    for symbol, price in (("EQUITY", "98"), ("INDEX", "101")):
        original_prices.append({
            "market": "XKRX", "symbol": symbol,
            "close_at": datetime(2026, 11, 24, 15, 30, tzinfo=KST).isoformat(),
            "adjusted_close": price, "adjustment": "adjusted",
            "instrument_kind": "STOCK", "provider": "fixture-provider",
            "observed_at": datetime(2026, 11, 24, 15, 30, tzinfo=KST).isoformat(),
        })
    write_csv(prices, ["market", "symbol", "close_at", "adjusted_close",
                       "adjustment", "instrument_kind", "provider", "observed_at"],
              original_prices)
    manifest = tmp_path / "manifest.json"
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["sources"][0]["sha256"] = hashlib.sha256(prices.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    short = evaluate_sources(pred, manifest, sessions)
    assert short["items"][0]["status"] == "STALE_PRICE"
    long = evaluate_sources(
        pred, manifest, sessions, max_anchor_age_minutes=40,
    )
    assert long["items"][0]["status"] == "OBSERVED"
    assert long["items"][0]["observation"]["absolute_return"] == 98 / 102 - 1
    assert long["trainingEligible"] is False


def test_conflicting_documentary_chronology_blocks_market_scoring(tmp_path: Path) -> None:
    from tris_market_event_intelligence.point_in_time_source_validation_cli import (
        evaluate_sources,
    )

    experiment(tmp_path)
    source = tmp_path / "evidence.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "article_id", "evidence_url", "evidence_type", "evidence_published_at",
            "captured_at", "sha256", "source_url", "first_public_at", "family_id",
            "relationship", "family_evidence_url",
        ])
        writer.writeheader()
        writer.writerow({
            "article_id": "1", "evidence_url": "https://archive.example/1",
            "evidence_type": "ARCHIVE_SNAPSHOT",
            "evidence_published_at": stamp(13, 55),
            "captured_at": stamp(15, 31),
            "sha256": hashlib.sha256(b"fixture-doc").hexdigest(),
            "source_url": "https://example.com/a",
            "first_public_at": stamp(13, 55),
            "family_id": "story-42", "relationship": "NEW",
            "family_evidence_url": "",
        })
    report = evaluate_sources(
        tmp_path / "predictions.csv", tmp_path / "manifest.json",
        tmp_path / "sessions.csv", evidence_csv=source,
    )
    assert report["documentaryEvidenceSupplied"] is True
    assert report["items"][0]["status"] == "EVIDENCE_CONFLICT"
    assert "observation" not in report["items"][0]
    assert report["trainingEligible"] is False


def test_consistent_operator_evidence_is_not_independent_certification(tmp_path: Path) -> None:
    from tris_market_event_intelligence.point_in_time_source_validation_cli import (
        evaluate_sources,
    )

    experiment(tmp_path)
    source = tmp_path / "evidence.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "article_id", "evidence_url", "evidence_type", "evidence_published_at",
            "captured_at", "sha256", "source_url", "first_public_at", "family_id",
            "relationship", "family_evidence_url",
        ])
        writer.writeheader()
        writer.writerow({
            "article_id": "1", "evidence_url": "https://archive.example/1",
            "evidence_type": "PUBLISHER_METADATA",
            "evidence_published_at": stamp(14, 0),
            "captured_at": stamp(15, 31),
            "sha256": hashlib.sha256(b"fixture-doc").hexdigest(),
            "source_url": "https://example.com/a",
            "first_public_at": stamp(14, 0),
            "family_id": "story-42", "relationship": "NEW",
            "family_evidence_url": "",
        })
    report = evaluate_sources(
        tmp_path / "predictions.csv", tmp_path / "manifest.json",
        tmp_path / "sessions.csv", evidence_csv=source,
    )
    result = report["items"][0]
    assert result["status"] == "OBSERVED"
    assert result["chronology"]["chronologyStatus"] == "DOCUMENTARY_CONSISTENT_UNATTESTED"
    assert result["independentFamilyScoringEligible"] is False
    assert result["learningEligible"] is False
    assert result["chronology"]["independentlyCertified"] is False
