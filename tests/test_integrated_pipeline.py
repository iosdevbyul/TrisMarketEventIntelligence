"""End-to-end bounded one-shot local pipeline, no real network/model calls."""
from __future__ import annotations

import fcntl
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from tris_market_event_intelligence.impact import ImpactResult
from tris_market_event_intelligence.integrated_pipeline import run_integrated_pass
from tris_market_event_intelligence.integrated_pipeline_cli import main


def news_rss(*, age_seconds: int = 10) -> bytes:
    published = format_datetime(
        datetime.now(UTC) - timedelta(seconds=age_seconds),
        usegmt=True,
    )
    return (
        "<rss><channel><item><title>Federal funds rate</title>"
        "<link>https://www.federalreserve.gov/notice/1</link><guid>notice-1</guid>"
        "<description>The Committee raised the interest rate by 25 basis points.</description>"
        f"<pubDate>{published}</pubDate></item></channel></rss>"
    ).encode()


class EmptyLocalModel:
    model_id = "fake-local-gemma-contract"
    def __init__(self) -> None:
        self.calls = 0
    def analyze(self, event: object) -> ImpactResult:
        self.calls += 1
        return ImpactResult(impacts=[])


def fake_client(*, age_seconds: int = 10) -> httpx.Client:
    body = news_rss(age_seconds=age_seconds)
    return httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=body),
    ))


def test_collect_extract_and_local_model_in_one_pass_then_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "news.sqlite3"
    status = tmp_path / "reports" / "pass.json"
    model = EmptyLocalModel()
    with fake_client() as client:
        first = run_integrated_pass(
            path, selected_sources=("fed_monetary",),
            status_path=status, client=client, model=model,
            enable_local_analysis=True,
        )
    assert first["status"] == "healthy"
    assert first["collection"]["new_articles"] == 1
    assert first["event_extraction"]["events"] >= 1
    assert first["analysis"]["processed"] == 1
    assert len(first["analysis"]["selected_event_ids"]) == 1
    assert first["analysis"]["failed"] == first["analysis"]["deferred"] == 0
    assert first["model_invoked"] is True
    assert model.calls == 1
    assert json.loads(status.read_text())["analysis"]["processed"] == 1
    with fake_client() as client:
        second = run_integrated_pass(
            path, selected_sources=("fed_monetary",),
            client=client, model=model, enable_local_analysis=True,
        )
    assert second["collection"]["new_articles"] == 0
    assert second["analysis"]["selected_event_ids"] == []
    assert model.calls == 1


def test_default_pipeline_extracts_but_does_not_use_ai(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    with fake_client() as client:
        result = run_integrated_pass(
            path, selected_sources=("fed_monetary",), client=client,
        )
    assert result["status"] == "healthy"
    assert result["event_extraction"]["events"] >= 1
    assert result["analysis"]["enabled"] is False
    assert result["model_invoked"] is False
    assert result["market_coverage_verified"] is False


def test_old_publisher_date_defers_model_but_keeps_event_pending(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    model = EmptyLocalModel()
    with fake_client(age_seconds=3600) as client:
        result = run_integrated_pass(
            path, selected_sources=("fed_monetary",),
            client=client, model=model, enable_local_analysis=True,
            live_freshness_seconds=300,
        )
    assert result["analysis"]["deferred"] == 1
    assert result["analysis"]["processed"] == 0
    assert model.calls == 0
    assert result["model_invoked"] is False
    assert result["status"] == "degraded"
    import sqlite3
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM impact_runs").fetchone()[0] == 0


def test_failed_collect_skips_extraction_and_model(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    model = EmptyLocalModel()
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(503),
    )) as client:
        result = run_integrated_pass(
            path, selected_sources=("fed_monetary",),
            client=client, model=model, enable_local_analysis=True,
        )
    assert result["status"] == "blocked"
    assert result["analysis"]["selected_event_ids"] == []
    assert result["event_extraction"]["processed"] == 0
    assert model.calls == 0


def test_integrated_lock_denies_overlapping_pass_without_external_calls(
    tmp_path: Path,
) -> None:
    path = tmp_path / "news.sqlite3"
    lockpath = path.with_name(path.name + ".pipeline.lock")
    lockpath.touch()
    with lockpath.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with httpx.Client(transport=httpx.MockTransport(
            lambda _: pytest.fail("No network if locked"),
        )) as client:
            result = run_integrated_pass(
                path, selected_sources=("fed_monetary",), client=client,
            )
        assert result["status"] == "already_running"
        assert result["model_invoked"] is False


def test_pipeline_rejects_implicit_model_and_invalid_limits(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    with pytest.raises(ValueError, match="explicit local model"):
        run_integrated_pass(path, enable_local_analysis=True)
    with pytest.raises(ValueError, match="without explicit"):
        run_integrated_pass(path, model=EmptyLocalModel())
    with pytest.raises(ValueError, match="1..1000"):
        run_integrated_pass(path, max_event_extraction=5000)


def test_cli_dry_run_never_writes_files_or_contacts_ollama(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    db = tmp_path / "untouched.sqlite3"
    status = tmp_path / "unwritten.json"
    with (
        patch.object(sys, "argv", [
            "market-event-integrated", "--db", str(db), "--status", str(status),
            "--sources", "fed_monetary", "--dry-run", "--enable-local-analysis",
        ]),
        patch("tris_market_event_intelligence.integrated_pipeline_cli.httpx.get",
              side_effect=AssertionError("No Ollama on dry run")),
    ):
        main()
    result = json.loads(capsys.readouterr().out)
    assert result["network_requests"] == 0
    assert result["filesystem_writes"] == 0
    assert result["model_invoked"] is False
    assert not db.exists() and not status.exists()


def test_cli_refuses_unknown_sources_before_network_and_model(tmp_path: Path) -> None:
    with patch.object(sys, "argv", [
        "market-event-integrated", "--sources", "fake",
        "--db", str(tmp_path / "db.sqlite3"), "--enable-local-analysis",
    ]):
        with pytest.raises(SystemExit):
            main()
    assert list(tmp_path.iterdir()) == []
