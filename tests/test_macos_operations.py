"""Opt-in phase-6 cycle does not need an active macOS daemon to test."""
import json
import plistlib
from pathlib import Path

import httpx
import pytest

from tris_market_event_intelligence.launchd_config import launchd_configuration
from tris_market_event_intelligence.operations import OPERATIONAL_SOURCES, run_cycle


FEED = b"""<rss><channel><item><guid>x</guid>
<title>Producer Price Index PPI released</title>
<link>https://example.org/ppi</link>
<description>PPI increased.</description></item></channel></rss>"""


def test_one_shot_cycle_is_idempotent_and_creates_atomic_status(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "bad.example":
            return httpx.Response(503)
        return httpx.Response(200, content=FEED)

    # No live external requests.
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        db = tmp_path / "news.db"
        status_path = tmp_path / "status" / "last.json"
        first = run_cycle(
            db, status_path=status_path,
            selected_sources=("fed_monetary", "fed_all"), client=client,
        )
        second = run_cycle(
            db, status_path=status_path,
            selected_sources=("fed_monetary", "fed_all"), client=client,
        )
    assert first["status"] == "healthy"
    assert first["new_articles"] == 2
    assert first["new_claims"] >= 2
    assert second["new_articles"] == 0
    assert second["new_claims"] == 0
    assert json.loads(status_path.read_text())["status"] == "healthy"
    assert not first["ai_used"] and not first["scraping_used"]


def test_cycle_marks_source_failures_degraded_or_failed(tmp_path: Path) -> None:
    def handle(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("press_all.xml"):
            return httpx.Response(403)
        return httpx.Response(200, content=FEED)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = run_cycle(
            tmp_path / "news.db",
            selected_sources=("fed_monetary", "fed_all"), client=client,
        )
    assert result["status"] == "degraded"
    assert result["successful_sources"] == 1
    with httpx.Client(transport=httpx.MockTransport(
        lambda req: httpx.Response(403)
    )) as client:
        result = run_cycle(
            tmp_path / "failed.db", selected_sources=("fed_all",), client=client,
        )
    assert result["status"] == "failed"


def test_mac_launchd_plist_is_opt_in_and_bounded() -> None:
    base = Path("/tmp/market-event-test")
    config = launchd_configuration(
        Path("/usr/local/bin/python3"),
        db_path=base / "db.sqlite3",
        status_path=base / "status.json",
        interval_seconds=1800,
    )
    saved = plistlib.loads(plistlib.dumps(config))
    assert saved["RunAtLoad"] is False
    assert saved["StartInterval"] == 1800
    assert saved["ProgramArguments"][2] == "tris_market_event_intelligence.cycle_cli"
    assert not any("NAVER_CLIENT_SECRET" in str(value) for value in saved.values())
    assert set(OPERATIONAL_SOURCES).isdisjoint({"bls_ppi", "gdelt_doc"})
    with pytest.raises(ValueError, match="Interval"):
        launchd_configuration(
            Path("/usr/local/bin/python3"),
            db_path=base / "db.sqlite3",
            status_path=base / "status.json", interval_seconds=30,
        )


def test_unknown_source_is_rejected_before_network(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        run_cycle(tmp_path / "news.db", selected_sources=("unknown",))
