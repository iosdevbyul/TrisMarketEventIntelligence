"""Offline listing import does not imply historical issuer coverage."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from tris_market_event_intelligence.event_ticker_proposals import (
    load_reviewed_issuers,
    propose_event_tickers,
)
from tris_market_event_intelligence.listing_snapshot_import import (
    export_snapshot_registry,
    preview_listing_snapshot,
)
from tris_market_event_intelligence.listing_snapshot_import_cli import main
from test_event_ticker_proposals import database

OBSERVED = "2026-11-14T15:00:00+00:00"
NASDAQ = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|"
    "Round Lot Size|ETF|NextShares\n"
    "MSFT|Microsoft Corporation - Common Stock|Q|N|N|100|N|N\n"
    "AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N\n"
    "FUND|Example ETF|Q|N|N|100|Y|N\n"
    "TEST|Fake Company - Common Stock|Q|Y|N|100|N|N\n"
    "WARNT|Warrant Issuer - Warrant|Q|N|N|100|N|N\n"
    "File Creation Time: 111420261500|||||||\n"
)
OTHER = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|"
    "Round Lot Size|Test Issue|NASDAQ Symbol\n"
    "IBM|International Business Machines Corporation - Common Stock|N|IBM|N|100|N|IBM\n"
    "ARCA|Unrelated Security - Common Stock|P|ARCA|N|100|N|ARCA\n"
    "File Creation Time: 111420261500|||||||\n"
)
KRX = (
    "market,symbol,company_name,security_type\n"
    "XKRX,005930,삼성전자,COMMON\n"
    "XKRX,000660,SK하이닉스,COMMON\n"
    "XKRX,123456,가상ETF,ETF\n"
)


def source(path: Path, content: str, *, kind: str) -> dict[str, str]:
    data = content.encode("utf-8")
    path.write_bytes(data)
    return {
        "kind": kind, "filename": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "observed_at": OBSERVED, "source_id": "operator-reviewed-snapshot",
    }


def manifest(root: Path, *records: dict[str, str]) -> Path:
    path = root / "manifest.json"
    path.write_text(json.dumps({
        "schema_version": "listing_snapshot_v1", "sources": records,
    }), encoding="utf-8")
    return path


def test_three_sources_preview_and_export_stays_snapshot_only(tmp_path: Path) -> None:
    m = manifest(tmp_path,
        source(tmp_path / "nasdaqlisted.txt", NASDAQ, kind="NASDAQ_DIRECTORY"),
        source(tmp_path / "otherlisted.txt", OTHER, kind="NYSE_DIRECTORY"),
        source(tmp_path / "krx.csv", KRX, kind="KRX_REVIEWED_EXPORT"),
    )
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    report, issuer_rows = preview_listing_snapshot(m)
    assert report["issuer_rows"] == 5
    assert report["source_origin_authenticated"] is False
    assert report["historical_backfill_allowed"] is False
    assert {(r["market"], r["symbol"]) for r in issuer_rows} == {
        ("XNAS", "MSFT"), ("XNAS", "AAPL"), ("XNYS", "IBM"),
        ("XKRX", "005930"), ("XKRX", "000660"),
    }
    assert report["source_reports"][0]["excluded"]["etf"] == 1
    assert report["source_reports"][0]["excluded"]["test"] == 1
    assert report["source_reports"][0]["excluded"]["unsupported"] == 1
    assert report["source_reports"][1]["excluded"]["non_nyse"] == 1
    assert all(r["valid_from"] == OBSERVED for r in issuer_rows)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    target = tmp_path / "candidate-issuers.csv"
    written = export_snapshot_registry(m, target, write=True)
    assert written["registry_created"] == str(target)
    issuers = load_reviewed_issuers(target)
    assert len(issuers) == 5
    assert all(i.aliases == (i.canonical_name,) for i in issuers)
    with pytest.raises(FileExistsError):
        export_snapshot_registry(m, target, write=True)
    assert len(load_reviewed_issuers(target)) == 5


def test_snapshot_only_never_backdates_news_links(tmp_path: Path) -> None:
    m = manifest(tmp_path,
        source(tmp_path / "nasdaqlisted.txt", NASDAQ, kind="NASDAQ_DIRECTORY"),
    )
    csv_path = tmp_path / "review.csv"
    export_snapshot_registry(m, csv_path, write=True)
    issuers = load_reviewed_issuers(csv_path)
    db = database(tmp_path / "news.db", "Microsoft reports quarterly results")
    older = propose_event_tickers(db, issuers, as_of=datetime(2026, 11, 14, 15, 5, tzinfo=UTC))
    assert older["proposals"] == 0
    assert older["items"][0]["status"] == "FUTURE_REGISTRY_EVIDENCE"


def test_invalid_hash_and_malformed_source_rejected(tmp_path: Path) -> None:
    entry = source(tmp_path / "nasdaqlisted.txt", NASDAQ, kind="NASDAQ_DIRECTORY")
    m = manifest(tmp_path, entry)
    entry["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        preview_listing_snapshot(m)
    entry["sha256"] = hashlib.sha256(NASDAQ.encode()).hexdigest()
    (tmp_path / "nasdaqlisted.txt").write_text(NASDAQ.replace(
        "File Creation Time: 111420261500|||||||\n", "",
    ))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        preview_listing_snapshot(m)
    raw = (tmp_path / "nasdaqlisted.txt").read_bytes()
    entry["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="footer"):
        preview_listing_snapshot(m)


def test_unsupported_krx_equities_and_duplicate_symbol_fail_closed(tmp_path: Path) -> None:
    krx = KRX + "XKRX,005930,삼성전자,COMMON\n"
    m = manifest(tmp_path,
        source(tmp_path / "krx.csv", krx, kind="KRX_REVIEWED_EXPORT"),
    )
    with pytest.raises(ValueError, match="Duplicate market-symbol"):
        preview_listing_snapshot(m)


def test_invalid_manifest_and_symlink_refused(tmp_path: Path) -> None:
    m = manifest(tmp_path,
        source(tmp_path / "krx.csv", KRX, kind="KRX_REVIEWED_EXPORT"),
    )
    broken = json.loads(m.read_text())
    broken["sources"][0]["observed_at"] = "2026-11-14T15:00:00"
    m.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="Timezone"):
        preview_listing_snapshot(m)
    broken["sources"][0]["observed_at"] = OBSERVED
    broken["sources"][0]["filename"] = "../krx.csv"
    m.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="basename"):
        preview_listing_snapshot(m)
    m.write_text(json.dumps({
        "schema_version": "listing_snapshot_v1",
        "sources": [{
            **broken["sources"][0], "filename": "krx.csv",
        }],
    }))
    link = tmp_path / "other.json"
    link.symlink_to(m)
    with pytest.raises(ValueError, match="manifest"):
        preview_listing_snapshot(link)


def test_cli_default_never_writes_and_explicit_write_creates_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    m = manifest(tmp_path,
        source(tmp_path / "krx.csv", KRX, kind="KRX_REVIEWED_EXPORT"),
    )
    target = tmp_path / "review.csv"
    with patch.object(sys, "argv", ["market-event-listing-snapshot", "--manifest", str(m)]):
        main()
    data = json.loads(capsys.readouterr().out)
    assert data["issuer_rows"] == 2
    assert "registry_created" not in data
    assert not target.exists()
    with patch.object(sys, "argv", [
        "market-event-listing-snapshot", "--manifest", str(m),
        "--write", "--output", str(target),
    ]):
        main()
    data = json.loads(capsys.readouterr().out)
    assert data["registry_created"] == str(target)
    with target.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
