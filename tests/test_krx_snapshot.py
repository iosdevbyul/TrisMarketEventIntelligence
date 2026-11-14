"""The Donghak KRX adapter must fail closed without mutating source storage."""
import hashlib
import json
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from tris_market_event_intelligence.krx_snapshot import (
    OPERATIONAL_APPLICATION_ID,
    MarketSnapshotError,
    PinnedKRXSnapshot,
)
from tris_market_event_intelligence.krx_snapshot_cli import inspect_snapshot


def sample_db(path: Path, *, adjustment: str = "unadjusted",
              operational: bool = False, no_trade: bool = False) -> str:
    with sqlite3.connect(path) as db:
        if operational:
            db.execute(f"PRAGMA application_id={OPERATIONAL_APPLICATION_ID}")
            db.execute("PRAGMA user_version=1")
            db.execute(
                "CREATE TABLE market_observations (market TEXT, ticker TEXT,"
                " trading_date TEXT, payload TEXT, validation_status TEXT)"
            )
        else:
            db.execute(
                "CREATE TABLE bars (ticker TEXT, trading_date TEXT, payload TEXT)"
            )
        for day, close in [("2026-11-12", 100), ("2026-11-13", 110)]:
            bar = {
                "ticker": "005930", "trading_date": day, "market": "KOSPI",
                "provider": "krx", "adjustment": adjustment,
                "close": close, "volume": 0 if no_trade else 100,
            }
            if operational:
                db.execute(
                    "INSERT INTO market_observations VALUES (?,?,?,?,?)",
                    ("KOSPI", "005930", day, json.dumps(bar), "UNREVIEWED"),
                )
            else:
                db.execute("INSERT INTO bars VALUES (?,?,?)",
                           ("005930", day, json.dumps(bar)))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_krx_unadjusted_bars_are_not_usable_as_adjusted_labels(
    tmp_path: Path,
) -> None:
    db = tmp_path / "research.sqlite3"
    pin = sample_db(db)
    original = db.read_bytes()
    reader = PinnedKRXSnapshot(db, expected_sha256=pin)
    info = reader.series("005930")
    assert info["source_kind"] == "research_snapshot"
    assert info["source_sha256"] == pin
    assert info["observation_count"] == 2
    assert info["status"] == "CORPORATE_ACTION_ADJUSTMENT_REQUIRED"
    assert info["event_study_price_eligible"] is False
    with pytest.raises(MarketSnapshotError, match="unqualified"):
        reader.adjusted_closes("005930")
    report = inspect_snapshot(db, pin, ("005930",))
    assert report["training_ready"] is False
    assert report["benchmark_ready"] is False
    assert report["quality_verified"] is False
    assert db.read_bytes() == original


def test_adjusted_research_series_can_be_read_explicitly(
    tmp_path: Path,
) -> None:
    db = tmp_path / "research.sqlite3"
    pin = sample_db(db, adjustment="adjusted")
    original = db.read_bytes()
    reader = PinnedKRXSnapshot(db, expected_sha256=pin)
    info = reader.series("005930")
    assert info["status"] == "ADJUSTED_SERIES_AVAILABLE"
    bars = reader.adjusted_closes("005930")
    assert [bar.day for bar in bars] == [date(2026, 11, 12), date(2026, 11, 13)]
    assert [bar.adjusted_close for bar in bars] == [100.0, 110.0]
    assert db.read_bytes() == original


def test_operational_snapshot_requires_correct_file_name_and_never_promotes(
    tmp_path: Path,
) -> None:
    wrong = tmp_path / "live.sqlite3"
    pin = sample_db(wrong, adjustment="adjusted", operational=True)
    with pytest.raises(MarketSnapshotError, match="offline"):
        PinnedKRXSnapshot(wrong, expected_sha256=pin)
    good = tmp_path / "offline.snapshot.sqlite3"
    wrong.rename(good)
    pin = hashlib.sha256(good.read_bytes()).hexdigest()
    reader = PinnedKRXSnapshot(good, expected_sha256=pin)
    assert reader.series("005930")["status"] == "OPERATIONAL_SOURCE_NOT_RESEARCH_APPROVED"
    with pytest.raises(MarketSnapshotError, match="unqualified"):
        reader.adjusted_closes("005930")


def test_unadjusted_operational_snapshot_still_not_eligible(tmp_path: Path) -> None:
    path = tmp_path / "archive.snapshot.sqlite3"
    pin = sample_db(path, operational=True)
    info = PinnedKRXSnapshot(path, expected_sha256=pin).series("005930")
    assert info["event_study_price_eligible"] is False
    assert info["adjustment"] == "unadjusted"


def test_no_trade_and_missing_stock_are_not_fabricated(tmp_path: Path) -> None:
    db = tmp_path / "market.sqlite3"
    pin = sample_db(db, adjustment="adjusted", no_trade=True)
    snapshot = PinnedKRXSnapshot(db, expected_sha256=pin)
    assert snapshot.series("005930")["status"] == "NO_TRADE_BARS_REQUIRE_REVIEW"
    assert snapshot.series("000660")["status"] == "NO_OBSERVATIONS"
    assert snapshot.series("000660")["observation_count"] == 0


def test_missing_pin_modified_file_and_sqlite_sidecars_are_rejected(tmp_path: Path) -> None:
    db = tmp_path / "market.sqlite3"
    pin = sample_db(db)
    with pytest.raises(MarketSnapshotError, match="fingerprint"):
        PinnedKRXSnapshot(db, expected_sha256="0" * 64)
    with pytest.raises(MarketSnapshotError, match="lowercase"):
        PinnedKRXSnapshot(db, expected_sha256="ABC")
    reader = PinnedKRXSnapshot(db, expected_sha256=pin)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO bars VALUES (?,?,?)", ("005930", "2026-11-14", "{}"))
    with pytest.raises(MarketSnapshotError, match="fingerprint"):
        reader.series("005930")
    sidecar = Path(str(db) + "-wal")
    sidecar.write_bytes(b"unsafe")
    with pytest.raises(MarketSnapshotError, match="sidecar"):
        PinnedKRXSnapshot(db, expected_sha256=hashlib.sha256(db.read_bytes()).hexdigest())


def test_invalid_ticker_and_limit_rejected_without_sql_injection(tmp_path: Path) -> None:
    db = tmp_path / "market.sqlite3"
    pin = sample_db(db)
    snap = PinnedKRXSnapshot(db, expected_sha256=pin)
    with pytest.raises(MarketSnapshotError, match="ticker"):
        snap.series("005930' OR 1=1")
    with pytest.raises(ValueError, match="limit"):
        snap.series("005930", limit=2501)
    with pytest.raises(ValueError, match="distinct"):
        inspect_snapshot(db, pin, ("005930", "005930"))


def test_malformed_persisted_bar_is_never_accepted(tmp_path: Path) -> None:
    db = tmp_path / "market.sqlite3"
    sample_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE bars SET payload=? WHERE trading_date='2026-11-13'",
            (json.dumps({
                "ticker": "005930", "trading_date": "2026-11-13",
                "provider": "krx", "market": "KOSPI", "adjustment": "adjusted",
                "close": 0, "volume": 3,
            }),),
        )
    pin = hashlib.sha256(db.read_bytes()).hexdigest()
    with pytest.raises(MarketSnapshotError, match="invalid persisted"):
        PinnedKRXSnapshot(db, expected_sha256=pin).series("005930")
