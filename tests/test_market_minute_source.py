"""Offline contract regression fixtures; no provider credentials or live prices."""
import csv
import hashlib
import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.market_minute_source import (
    MarketSourceContractError,
    audit_market_source,
)


def make_source(
    directory: Path, *, market: str = "XNAS",
    kind: str = "STOCK", symbol: str = "NVDA",
    adjustment: str = "adjusted", timestamp: str = "2026-11-16T15:16:00+00:00",
    filename: str = "minute.csv", observed: str = "2026-11-16T15:17:00+00:00",
    provider: str = "reviewed-example",
) -> dict:
    path = directory / filename
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "market", "symbol", "close_at", "adjusted_close",
            "adjustment", "instrument_kind", "provider", "observed_at",
        ])
        writer.writeheader()
        writer.writerow({
            "market": market, "symbol": symbol, "close_at": timestamp,
            "adjusted_close": 100.0, "adjustment": adjustment,
            "instrument_kind": kind, "provider": provider, "observed_at": observed,
        })
    return {
        "filename": filename,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "market": market, "provider": provider, "source_kind": kind,
    }


def write_manifest(directory: Path, entries: list[dict]) -> Path:
    path = directory / "manifest.json"
    path.write_text(
        json.dumps({"schema_version": "minute_source_v1", "sources": entries}),
        encoding="utf-8",
    )
    return path


def test_korean_and_us_cross_market_stock_and_benchmark_sources(tmp_path: Path) -> None:
    records = [
        make_source(tmp_path, filename="kr-stock.csv", market="XKRX", symbol="005930"),
        make_source(tmp_path, filename="kr-index.csv", market="XKRX",
                    symbol="KOSPI", kind="BENCHMARK"),
        make_source(tmp_path, filename="us-stock.csv"),
        make_source(tmp_path, filename="us-bench.csv", symbol="SPY", kind="BENCHMARK"),
    ]
    manifest = write_manifest(tmp_path, records)
    original = [p.read_bytes() for p in tmp_path.iterdir()]
    audit = audit_market_source(manifest)
    assert audit["markets_inspected"] == ["XKRX", "XNAS"]
    assert len(audit["sources"]) == 4
    assert all(s["status"] == "SCHEMA_COMPATIBLE_SOURCE_UNVERIFIED"
               for s in audit["sources"])
    assert audit["source_files_hashed"] is True
    assert audit["source_origin_authenticated"] is False
    assert audit["real_market_training_ready"] is False
    assert audit["prediction_model_trained"] is False
    assert original == [p.read_bytes() for p in tmp_path.iterdir()]


def test_daily_krx_unadjusted_close_is_not_marked_as_minute_ready(tmp_path: Path) -> None:
    entry = make_source(tmp_path, adjustment="unadjusted", market="XKRX",
                        symbol="005930")
    r = audit_market_source(write_manifest(tmp_path, [entry]))
    assert r["sources"][0]["status"] == "ADJUSTED_PRICE_PROVENANCE_REQUIRED"
    assert r["real_market_training_ready"] is False


def test_missing_adjustment_is_reported_not_assumed(tmp_path: Path) -> None:
    entry = make_source(tmp_path, adjustment="unknown")
    audit = audit_market_source(write_manifest(tmp_path, [entry]))
    assert audit["sources"][0]["adjustments"] == ["unknown"]
    assert audit["sources"][0]["status"] == "ADJUSTED_PRICE_PROVENANCE_REQUIRED"


def test_tampered_file_fingerprint_is_refused(tmp_path: Path) -> None:
    entry = make_source(tmp_path)
    path = tmp_path / "minute.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(MarketSourceContractError, match="fingerprint"):
        audit_market_source(write_manifest(tmp_path, [entry]))


def test_future_observation_timestamp_and_naive_timestamps_refused(tmp_path: Path) -> None:
    entry = make_source(tmp_path, observed="2026-11-16T15:15:00+00:00")
    with pytest.raises(MarketSourceContractError, match="row 1"):
        audit_market_source(write_manifest(tmp_path, [entry]))
    entry = make_source(tmp_path, observed="2026-11-16T15:17:00+00:00",
                        timestamp="2026-11-16T15:16:00")
    with pytest.raises(MarketSourceContractError, match="row 1"):
        audit_market_source(write_manifest(tmp_path, [entry]))


def test_wrong_market_invalid_instrument_and_provider_mismatch_refused(
    tmp_path: Path,
) -> None:
    entry = make_source(tmp_path)
    entry["market"] = "XKRX"
    with pytest.raises(MarketSourceContractError, match="row 1"):
        audit_market_source(write_manifest(tmp_path, [entry]))
    entry = make_source(tmp_path)
    entry["source_kind"] = "BENCHMARK"
    with pytest.raises(MarketSourceContractError, match="row 1"):
        audit_market_source(write_manifest(tmp_path, [entry]))
    entry = make_source(tmp_path)
    entry["provider"] = "different"
    with pytest.raises(MarketSourceContractError, match="row 1"):
        audit_market_source(write_manifest(tmp_path, [entry]))


def test_duplicate_source_and_cross_file_minute_collision_refused(tmp_path: Path) -> None:
    entry = make_source(tmp_path)
    with pytest.raises(MarketSourceContractError, match="Repeated source"):
        audit_market_source(write_manifest(tmp_path, [entry, entry]))
    repeated = make_source(tmp_path, filename="duplicate.csv")
    with pytest.raises(MarketSourceContractError, match="Duplicate minute"):
        audit_market_source(write_manifest(tmp_path, [entry, repeated]))


def test_filename_traversal_and_symlink_are_refused(tmp_path: Path) -> None:
    entry = make_source(tmp_path)
    entry["filename"] = "../private.csv"
    with pytest.raises(MarketSourceContractError, match="plain local"):
        audit_market_source(write_manifest(tmp_path, [entry]))
    entry = make_source(tmp_path)
    source = tmp_path / "minute.csv"
    pointer = tmp_path / "alias.csv"
    pointer.symlink_to(source)
    entry["filename"] = pointer.name
    with pytest.raises(MarketSourceContractError, match="non-symlink"):
        audit_market_source(write_manifest(tmp_path, [entry]))


def test_bad_manifest_and_bounded_input_validation(tmp_path: Path) -> None:
    empty = write_manifest(tmp_path, [])
    with pytest.raises(MarketSourceContractError, match="1..12"):
        audit_market_source(empty)
    item = make_source(tmp_path)
    item["filename"] = ["bad"]
    with pytest.raises(MarketSourceContractError, match="filename"):
        audit_market_source(write_manifest(tmp_path, [item]))
    item = make_source(tmp_path)
    item["sha256"] = "bad"
    with pytest.raises(MarketSourceContractError, match="SHA-256"):
        audit_market_source(write_manifest(tmp_path, [item]))


def test_bar_is_minute_close_not_a_daily_or_second_observation(tmp_path: Path) -> None:
    entry = make_source(tmp_path, timestamp="2026-11-16T15:16:35+00:00")
    with pytest.raises(MarketSourceContractError, match="row 1"):
        audit_market_source(write_manifest(tmp_path, [entry]))


def test_empty_csv_is_reported_without_imputing_bars(tmp_path: Path) -> None:
    entry = make_source(tmp_path)
    path = tmp_path / entry["filename"]
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "market", "symbol", "close_at", "adjusted_close",
            "adjustment", "instrument_kind", "provider", "observed_at",
        ])
    entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = audit_market_source(write_manifest(tmp_path, [entry]))
    assert audit["sources"][0]["status"] == "NO_MINUTE_BARS"
    assert audit["sources"][0]["rows"] == 0
