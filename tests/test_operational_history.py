"""Per-cycle audit trail and SQLite sidecar diagnostics."""
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from tris_market_event_intelligence.operational_diagnostics import diagnose
from tris_market_event_intelligence.operations import run_cycle

FEED = b"""<rss><channel><item><guid>test</guid><title>Fed interest rate decision</title>
<link>https://example.net/fed</link><description>Fed policy announcement.</description>
</item></channel></rss>"""


def test_history_retains_two_cycles_without_article_duplication(tmp_path: Path) -> None:
    db, status = tmp_path / "news.sqlite3", tmp_path / "last-run.json"
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=FEED)
    )) as client:
        first = run_cycle(db, status_path=status, selected_sources=("fed_all",), client=client)
        second = run_cycle(db, status_path=status, selected_sources=("fed_all",), client=client)
    ledger = tmp_path / "run-history.jsonl"
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(records) == 2
    assert first["status"] == second["status"] == "healthy"
    assert records[0]["new_articles"] == 1
    assert records[1]["new_articles"] == 0
    assert records[0]["source_statuses"]["fed_all"] == "ok"
    assert "https://example.net" not in ledger.read_text()
    assert "started_at" in records[0]
    assert json.loads(status.read_text())["finished_at"] == records[1]["finished_at"]


def test_failed_cycle_is_preserved_in_history(tmp_path: Path) -> None:
    db, status = tmp_path / "news.sqlite3", tmp_path / "last-run.json"
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(403)
    )) as client:
        outcome = run_cycle(db, status_path=status, selected_sources=("fed_all",), client=client)
    record = json.loads((tmp_path / "run-history.jsonl").read_text().splitlines()[0])
    assert outcome["status"] == "failed"
    assert record["status"] == "failed"
    assert record["source_statuses"]["fed_all"] == "HTTP_403"


def test_doctor_includes_real_wal_and_shm_footprint(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    status = tmp_path / "last-run.json"
    now = datetime(2026, 11, 14, tzinfo=UTC)
    status.write_text(json.dumps({
        "status": "healthy",
        "finished_at": (now - timedelta(minutes=5)).isoformat(),
    }))
    with sqlite3.connect(db) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE sample (id INTEGER)")
        writer.commit()
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO sample VALUES (1)")
        result = diagnose(db, status, now=now, warning_bytes=1)
        assert result["status"] == "warning"
        sizes = result["checks"]["database"]["file_bytes"]
        assert sizes["wal"] > 0 and sizes["shm"] > 0
        assert result["checks"]["database"]["total_storage_bytes"] == sum(sizes.values())
        writer.rollback()
