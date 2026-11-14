"""Narrow, bounded SQLite WAL initialization retry for macOS temp files."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from tris_market_event_intelligence.collector import NewsStore


class FakeConnection:
    def __init__(self, *, wal_failure: str | None = None) -> None:
        self.wal_failure = wal_failure
        self.closed = False
        self.statements: list[str] = []

    def execute(self, sql: str) -> None:
        self.statements.append(sql)
        if sql == "PRAGMA journal_mode=WAL" and self.wal_failure:
            raise sqlite3.OperationalError(self.wal_failure)

    def close(self) -> None:
        self.closed = True


def test_existing_temp_database_retries_only_transient_wal_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "news.sqlite3"
    path.touch()
    first = FakeConnection(wal_failure="unable to open database file")
    second = FakeConnection()
    with (
        patch("tris_market_event_intelligence.collector.sqlite3.connect",
              side_effect=[first, second]) as connect,
        patch("tris_market_event_intelligence.collector.time.sleep") as sleep,
    ):
        returned = NewsStore(path).connect()
    assert returned is second
    assert first.closed is True
    assert second.closed is False
    assert second.statements == [
        "PRAGMA busy_timeout=30000",
        "PRAGMA journal_mode=WAL",
        "PRAGMA foreign_keys=ON",
    ]
    assert connect.call_count == 2
    sleep.assert_called_once_with(0.05)


@pytest.mark.parametrize("message", [
    "database is locked", "attempt to write a readonly database",
])
def test_sqlite_connection_does_not_retry_other_operational_errors(
    tmp_path: Path, message: str,
) -> None:
    path = tmp_path / "news.sqlite3"
    path.touch()
    failed = FakeConnection(wal_failure=message)
    with (
        patch("tris_market_event_intelligence.collector.sqlite3.connect",
              return_value=failed) as connect,
        patch("tris_market_event_intelligence.collector.time.sleep") as sleep,
        pytest.raises(sqlite3.OperationalError, match=message),
    ):
        NewsStore(path).connect()
    assert failed.closed is True
    connect.assert_called_once()
    sleep.assert_not_called()


def test_sqlite_missing_file_does_not_retry_failed_wal_initialization(
    tmp_path: Path,
) -> None:
    failed = FakeConnection(wal_failure="unable to open database file")
    with (
        patch("tris_market_event_intelligence.collector.sqlite3.connect",
              return_value=failed) as connect,
        patch("tris_market_event_intelligence.collector.time.sleep") as sleep,
        pytest.raises(sqlite3.OperationalError, match="unable to open database file"),
    ):
        NewsStore(tmp_path / "not-created.sqlite3").connect()
    assert failed.closed is True
    connect.assert_called_once()
    sleep.assert_not_called()


def test_sqlite_transient_failure_retries_at_most_three_times(
    tmp_path: Path,
) -> None:
    path = tmp_path / "news.sqlite3"
    path.touch()
    failed = [FakeConnection(wal_failure="unable to open database file") for _ in range(3)]
    with (
        patch("tris_market_event_intelligence.collector.sqlite3.connect",
              side_effect=failed) as connect,
        patch("tris_market_event_intelligence.collector.time.sleep") as sleep,
        pytest.raises(sqlite3.OperationalError, match="unable to open database file"),
    ):
        NewsStore(path).connect()
    assert all(conn.closed for conn in failed)
    assert connect.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [0.05, 0.1]
