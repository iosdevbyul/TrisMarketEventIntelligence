"""Operator-gated macOS scheduling and read-only freshness watchdog tests."""
from __future__ import annotations

import json
import plistlib
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from test_integrated_pipeline import fake_client

from tris_market_event_intelligence.integrated_launchd_config import (
    INTEGRATED_LABEL,
    integrated_launchd_configuration,
    main as schedule_main,
)
from tris_market_event_intelligence.integrated_pipeline import run_integrated_pass
from tris_market_event_intelligence.integrated_watchdog import audit_integrated_schedule
from tris_market_event_intelligence.integrated_watchdog_cli import main as watchdog_main
from tris_market_event_intelligence.launchd_config import LABEL as LEGACY_LABEL

SOURCES = ("fed_monetary", "fed_all")


def fixture_config(**kwargs: object) -> dict[str, object]:
    return integrated_launchd_configuration(
        Path("/usr/local/bin/python3"), db_path=Path("/tmp/market-events/news.sqlite3"),
        status_path=Path("/tmp/market-events/integrated-last-run.json"),
        sources=SOURCES, **kwargs,
    )


def test_default_schedule_has_no_model_or_automatic_start() -> None:
    config = plistlib.loads(plistlib.dumps(fixture_config()))
    assert config["Label"] == INTEGRATED_LABEL
    assert config["StartInterval"] == 900
    assert config["RunAtLoad"] is False
    assert config["ProgramArguments"][2] == (
        "tris_market_event_intelligence.integrated_pipeline_cli"
    )
    assert config["ProgramArguments"][-1] == "fed_monetary,fed_all"
    assert "--enable-local-analysis" not in config["ProgramArguments"]
    assert "EnvironmentVariables" not in config
    assert "secret" not in repr(config).lower()
    assert LEGACY_LABEL != INTEGRATED_LABEL


def test_schedule_model_requires_two_explicit_approvals_and_known_model() -> None:
    with pytest.raises(ValueError, match="confirmation"):
        fixture_config(enable_local_analysis=True, model_name="gemma2:latest")
    with pytest.raises(ValueError, match="opt-in"):
        fixture_config(model_name="gemma2:latest")
    with pytest.raises(ValueError, match="identifier"):
        fixture_config(
            enable_local_analysis=True, confirmed_background_inference=True,
            model_name="gemma2:latest;cat /private/secret",
        )
    config = fixture_config(
        enable_local_analysis=True, confirmed_background_inference=True,
        model_name="gemma2:latest",
    )
    assert "--enable-local-analysis" in config["ProgramArguments"]
    assert config["EnvironmentVariables"] == {"MARKET_EVENT_LLM_MODEL": "gemma2:latest"}
    assert not any("API_KEY" in str(value) for value in config.values())


@pytest.mark.parametrize("interval", [0, 30, 299, 86401])
def test_interval_is_bounded(interval: int) -> None:
    with pytest.raises(ValueError, match="Interval"):
        fixture_config(interval_seconds=interval)


def test_sources_are_whitelisted_and_paths_are_absolute() -> None:
    with pytest.raises(ValueError, match="RSS source"):
        fixture_config(sources=("gdelt_doc",))
    with pytest.raises(ValueError, match="RSS source"):
        fixture_config(sources=("fed_all", "fed_all"))
    with pytest.raises(ValueError, match="absolute"):
        integrated_launchd_configuration(
            Path("python"), db_path=Path("/tmp/news.sqlite3"),
            status_path=Path("/tmp/status.json"),
        )


def test_schedule_write_refuses_legacy_collector_plist(tmp_path: Path) -> None:
    agents = tmp_path / "Library" / "LaunchAgents"
    agents.mkdir(parents=True)
    legacy = agents / (LEGACY_LABEL + ".plist")
    legacy.write_text("existing production configuration")
    with (
        patch.object(Path, "home", return_value=tmp_path),
        patch.object(sys, "argv", ["market-event-integrated-launchd", "--write"]),
        pytest.raises(SystemExit) as exc,
    ):
        schedule_main()
    assert exc.value.code == 2
    assert legacy.read_text() == "existing production configuration"
    assert not (agents / (INTEGRATED_LABEL + ".plist")).exists()


def test_schedule_preview_never_creates_launch_agent(tmp_path: Path, capsys: object) -> None:
    with (
        patch.object(Path, "home", return_value=tmp_path),
        patch.object(sys, "argv", ["market-event-integrated-launchd"]),
    ):
        schedule_main()
    assert "RunAtLoad" in capsys.readouterr().out
    assert not (tmp_path / "Library").exists()


def test_schedule_create_only_never_overwrites_existing(tmp_path: Path) -> None:
    agents = tmp_path / "Library" / "LaunchAgents"
    agents.mkdir(parents=True)
    target = agents / (INTEGRATED_LABEL + ".plist")
    target.write_text("operator file")
    with (
        patch.object(Path, "home", return_value=tmp_path),
        patch.object(sys, "argv", ["market-event-integrated-launchd", "--write"]),
        pytest.raises(SystemExit) as exc,
    ):
        schedule_main()
    assert exc.value.code == 2
    assert target.read_text() == "operator file"


def test_watchdog_checks_actual_mocked_integrated_cycle(tmp_path: Path) -> None:
    status_path = tmp_path / "integrated-last-run.json"
    with fake_client() as client:
        result = run_integrated_pass(
            tmp_path / "news.db", selected_sources=SOURCES,
            status_path=status_path, client=client,
        )
    assert result["status"] == "healthy"
    audit = audit_integrated_schedule(
        status_path, as_of=datetime.now(UTC),
        expected_sources=SOURCES,
    )
    assert audit["status"] == "OBSERVED_RECENT_SUCCESS"
    assert audit["model_enabled_in_last_run"] is False
    assert audit["model_inference_verified"] is False
    assert audit["new_articles_in_last_run"] == 2
    assert audit["market_news_coverage_verified"] is False
    stale = audit_integrated_schedule(
        status_path, as_of=datetime.now(UTC) + timedelta(hours=1),
        expected_sources=SOURCES, max_age_seconds=900,
    )
    assert stale["reason"] == "SCHEDULE_MISSED_OR_STALE"


def test_watchdog_missing_legacy_corrupt_and_wrong_sources(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    clock = datetime.now(UTC)
    assert audit_integrated_schedule(path, as_of=clock, expected_sources=SOURCES)[
        "reason"
    ] == "NO_COMPLETED_INTEGRATED_RUN"
    path.write_text("{")
    assert audit_integrated_schedule(path, as_of=clock, expected_sources=SOURCES)[
        "reason"
    ] == "INVALID_STATUS_FILE"
    path.write_text(json.dumps({"status": "healthy", "finished_at": clock.isoformat()}))
    assert audit_integrated_schedule(path, as_of=clock, expected_sources=SOURCES)[
        "reason"
    ] == "NOT_AN_INTEGRATED_RUN"
    path.write_text(json.dumps({
        "schema_version": "integrated_news_pass_v1",
        "status": "healthy", "finished_at": clock.isoformat(),
        "collection": {
            "status": "healthy",
            "source_results": {"fed_all": {"status": "ok"}},
        }, "event_extraction": {"processed": 0},
    }))
    assert audit_integrated_schedule(path, as_of=clock, expected_sources=SOURCES)[
        "reason"
    ] == "SOURCE_SET_MISMATCH"


def test_watchdog_rejects_failed_or_future_stale_integrated_run(tmp_path: Path) -> None:
    path = tmp_path / "integrated.json"
    now = datetime.now(UTC)
    for finish, status, expected in [
        (now + timedelta(minutes=1), "healthy", "FUTURE_COMPLETION_TIME"),
        (now - timedelta(minutes=1), "degraded", "LAST_RUN_NOT_HEALTHY"),
        (now - timedelta(hours=1), "healthy", "SCHEDULE_MISSED_OR_STALE"),
    ]:
        path.write_text(json.dumps({
            "schema_version": "integrated_news_pass_v1",
            "status": status, "finished_at": finish.isoformat(),
        }))
        assert audit_integrated_schedule(
            path, as_of=now, expected_sources=SOURCES,
        )["reason"] == expected


def test_watchdog_cli_blocked_exits_nonzero(tmp_path: Path, capsys: object) -> None:
    path = tmp_path / "not-installed.json"
    with (
        patch.object(sys, "argv", [
            "market-event-integrated-watchdog", "--status", str(path),
            "--sources", "fed_monetary,fed_all",
        ]),
        pytest.raises(SystemExit) as exc,
    ):
        watchdog_main()
    assert exc.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED"
    assert not path.exists()
