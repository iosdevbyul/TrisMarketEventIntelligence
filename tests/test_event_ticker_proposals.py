"""Point-in-time KR/US candidate matching refuses invented ticker identity."""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from tris_market_event_intelligence.event_ticker_proposals import (
    _matches,
    load_reviewed_issuers,
    propose_event_tickers,
)
from tris_market_event_intelligence.event_ticker_proposals_cli import main

HEADERS = (
    "market", "symbol", "canonical_name", "aliases", "valid_from",
    "valid_until", "evidence_seen_at", "source_id",
)
EARLY = "2026-01-01T00:00:00+00:00"
LATE = "2026-12-01T00:00:00+00:00"
RECEIVED = "2026-11-14T12:00:00+00:00"
OBSERVED = datetime(2026, 11, 14, 13, 0, tzinfo=UTC)


def record(
    market: str, symbol: str, name: str, aliases: str = "", *,
    observed: str = EARLY, valid_from: str = EARLY, until: str = "",
) -> dict[str, str]:
    return dict(zip(HEADERS, (
        market, symbol, name, aliases, valid_from, until, observed, "reviewed-list",
    ), strict=True))


def registry(path: Path, *records: dict[str, str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(records)
    return path


def database(path: Path, *titles: str, seen: str = RECEIVED) -> Path:
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT,"
            " summary TEXT, detected_at TEXT)"
        )
        db.execute(
            "CREATE TABLE economic_events (id INTEGER PRIMARY KEY,"
            " article_id INTEGER, status TEXT, source TEXT, source_url TEXT)"
        )
        for index, title in enumerate(titles, 1):
            db.execute("INSERT INTO articles VALUES (?,?,?,?)", (
                index, title, "summary with Microsoft", seen,
            ))
            db.execute("INSERT INTO economic_events VALUES (?,?,?,?,?)", (
                index, index, "CANDIDATE", "bbc_business",
                f"https://example.org/article/{index}",
            ))
    return path


def test_two_market_names_are_proposed_without_autoapproval(tmp_path: Path) -> None:
    listed = load_reviewed_issuers(registry(
        tmp_path / "list.csv",
        record("XKRX", "005930", "삼성전자", "Samsung Electronics"),
        record("XNAS", "MSFT", "Microsoft Corporation", "Microsoft"),
    ))
    db = database(tmp_path / "events.db", "삼성전자 and Microsoft announce partnership")
    before = db.read_bytes()
    report = propose_event_tickers(db, listed, as_of=OBSERVED)
    assert report["status"] == "REVIEW_REQUIRED"
    assert report["proposals"] == 2
    assert {p["symbol"] for p in report["items"][0]["proposals"]} == {"005930", "MSFT"}
    assert report["training_dataset_eligible"] is False
    assert report["database_modified"] is False
    assert all(p["human_review_required"] for p in report["items"][0]["proposals"])
    assert db.read_bytes() == before


def test_future_listing_evidence_never_retroactively_links(tmp_path: Path) -> None:
    listed = load_reviewed_issuers(registry(
        tmp_path / "list.csv",
        record("XNAS", "MSFT", "Microsoft", observed=LATE),
    ))
    db = database(tmp_path / "events.db", "Microsoft reports earnings")
    report = propose_event_tickers(db, listed, as_of=OBSERVED)
    assert report["proposals"] == 0
    assert report["items"][0]["status"] == "FUTURE_REGISTRY_EVIDENCE"


def test_inactive_listing_and_unknown_name_dont_link(tmp_path: Path) -> None:
    listed = load_reviewed_issuers(registry(
        tmp_path / "list.csv",
        record("XNAS", "MSFT", "Microsoft", until="2026-01-02T00:00:00+00:00"),
    ))
    db = database(tmp_path / "events.db", "Microsoft reports earnings", "Unknown company")
    report = propose_event_tickers(db, listed, as_of=OBSERVED)
    assert report["proposals"] == 0
    assert all(item["status"] == "NO_VERIFIED_NAME_MATCH" for item in report["items"])


def test_ambiguous_alias_blocks_both_issuers(tmp_path: Path) -> None:
    listed = load_reviewed_issuers(registry(
        tmp_path / "list.csv",
        record("XNAS", "TEST", "Example Holdings", "Orbit"),
        record("XKRX", "000001", "Example Korea", "Orbit"),
    ))
    db = database(tmp_path / "events.db", "Orbit announces plans")
    report = propose_event_tickers(db, listed, as_of=OBSERVED)
    assert report["proposals"] == 0
    assert report["items"][0]["status"] == "AMBIGUOUS_ALIAS"


def test_no_ticker_only_or_summary_only_link(tmp_path: Path) -> None:
    listed = load_reviewed_issuers(registry(
        tmp_path / "list.csv", record("XNAS", "MSFT", "Microsoft"),
    ))
    db = database(tmp_path / "events.db", "MSFT gained 5 percent")
    report = propose_event_tickers(db, listed, as_of=OBSERVED)
    assert report["proposals"] == 0
    assert _matches("Microsoft's quarterly statement", "Microsoft")
    assert not _matches("Microsoftease stock", "Microsoft")


@pytest.mark.parametrize("bad", [
    record("XNAS", "MSFT", "Microsoft", observed="2026-01-01"),
    record("XNAS", "MSFT", "M", observed=EARLY),
    record("NASDAQ", "MSFT", "Microsoft"),
    record("XNAS", "msft", "Microsoft"),
    record("XNAS", "MSFT", "Microsoft", valid_from=LATE, until=EARLY),
])
def test_invalid_registry_rejected(tmp_path: Path, bad: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="row 1"):
        load_reviewed_issuers(registry(tmp_path / "bad.csv", bad))


def test_overlapping_symbol_validity_rejected(tmp_path: Path) -> None:
    path = registry(
        tmp_path / "list.csv",
        record("XNAS", "MSFT", "Microsoft"),
        record("XNAS", "MSFT", "Microsoft New", valid_from="2026-06-01T00:00:00+00:00"),
    )
    with pytest.raises(ValueError, match="Overlapping"):
        load_reviewed_issuers(path)


def test_missing_registry_and_missing_db_fail_without_creating_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="registry"):
        load_reviewed_issuers(tmp_path / "missing.csv")
    listed = load_reviewed_issuers(registry(
        tmp_path / "list.csv", record("XNAS", "MSFT", "Microsoft"),
    ))
    with pytest.raises(ValueError, match="database"):
        propose_event_tickers(tmp_path / "nonexistent.db", listed)
    assert not (tmp_path / "nonexistent.db").exists()


def test_cli_json_output_is_read_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv_path = registry(tmp_path / "list.csv", record("XNAS", "MSFT", "Microsoft"))
    db = database(tmp_path / "events.db", "Microsoft reports results")
    before = db.read_bytes()
    with patch.object(sys, "argv", [
        "market-event-ticker-proposals", "--registry", str(csv_path),
        "--db", str(db),
    ]):
        main()
    result = json.loads(capsys.readouterr().out)
    assert result["proposals"] == 1
    assert result["training_dataset_eligible"] is False
    assert db.read_bytes() == before
