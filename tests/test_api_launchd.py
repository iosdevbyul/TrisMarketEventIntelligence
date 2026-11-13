"""Phase 6-D: opt-in API service and independent claims readiness."""

import plistlib
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tris_market_event_intelligence.api import create_app
from tris_market_event_intelligence.api_launchd_config import (
    API_LABEL,
    api_launchd_configuration,
)
from tris_market_event_intelligence.api_service import main as api_main
from tris_market_event_intelligence.claim_store import ClaimProcessor


def test_claim_readiness_is_independent_from_impact_tables(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    with TestClient(create_app(path)) as client:
        assert client.get("/ready/claims").status_code == 503
        ClaimProcessor(path).initialize()
        assert client.get("/ready/claims").json() == {"status": "ready"}
        assert client.get("/ready").status_code == 503
        claims = client.get("/api/v1/event-claims").json()
        assert claims["items"] == []
        assert claims["nextCursor"] == 0
        assert claims["schemaVersion"] == "1.0"


def test_claim_readiness_does_not_create_missing_database(tmp_path: Path) -> None:
    path = tmp_path / "missing.sqlite3"
    with TestClient(create_app(path)) as client:
        assert client.get("/ready/claims").status_code == 503
    assert not path.exists()


def test_api_plist_is_separate_loopback_and_opt_in(tmp_path: Path) -> None:
    python = tmp_path / "python3"
    python.write_text("stub")
    db = tmp_path / "news.sqlite3"
    config = api_launchd_configuration(python, db_path=db, port=8100)
    data = plistlib.loads(plistlib.dumps(config))
    assert data["Label"] == API_LABEL
    assert data["RunAtLoad"] is True
    assert data["KeepAlive"] == {"SuccessfulExit": False}
    assert data["ProgramArguments"][2] == "tris_market_event_intelligence.api_service"
    assert data["ProgramArguments"][-2:] == ["--port", "8100"]
    assert data["StandardErrorPath"].endswith("/api-stderr.log")
    assert "com.tris.market-event-intelligence.plist" not in str(data)
    with pytest.raises(ValueError, match="Port"):
        api_launchd_configuration(python, db_path=db, port=0)


def test_local_api_runner_binds_loopback_and_explicit_db(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    with (
        patch("sys.argv", ["market-event-api-local", "--db", str(db), "--port", "8181"]),
        patch("tris_market_event_intelligence.api_service.uvicorn.run") as serve,
    ):
        api_main()
    assert serve.call_args.kwargs["host"] == "127.0.0.1"
    assert serve.call_args.kwargs["port"] == 8181
    assert serve.call_args.kwargs["proxy_headers"] is False
    assert serve.call_args.args[0].title == "TrisMarketEventIntelligence"
    assert not db.exists()


def test_api_is_read_only_on_existing_claim_database(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    ClaimProcessor(db).initialize()
    with sqlite3.connect(db) as connection:
        prior = connection.execute("PRAGMA data_version").fetchone()[0]
    with TestClient(create_app(db)) as client:
        for _ in range(3):
            assert client.get("/ready/claims").status_code == 200
            assert client.get("/api/v1/event-claims").status_code == 200
    with sqlite3.connect(db) as connection:
        after = connection.execute("PRAGMA data_version").fetchone()[0]
    assert prior == after
