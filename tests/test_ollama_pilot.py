"""Bounded real-collection pilot preflight and eligible input selection."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from tris_market_event_intelligence.claim_event_bridge import ClaimEventBridge
from tris_market_event_intelligence.claim_store import ClaimProcessor
from tris_market_event_intelligence.collector import NewsStore
from tris_market_event_intelligence.impact import EventInput, ImpactResult
from tris_market_event_intelligence.impact_cli import main as analyze_main
from tris_market_event_intelligence.impact_store import ImpactProcessor


class EmptyProvider:
    model_id = "pilot-offline"

    def __init__(self) -> None:
        self.ids: list[int] = []

    def analyze(self, event: EventInput) -> ImpactResult:
        self.ids.append(event.event_id)
        return ImpactResult(impacts=[])


def _claim_fixture(db: Path) -> None:
    ClaimProcessor(db).initialize()
    with sqlite3.connect(db) as connection:
        connection.execute(
            "INSERT INTO captures (source, feed_url, fetched_at, sha256, payload) "
            "VALUES ('bok_all', 'https://example.org/feed', 'now', 'hash', X'31')"
        )
        connection.execute(
            "INSERT INTO articles "
            "(source, external_id, url, title, detected_at, capture_id) "
            "VALUES ('bok_all', 'fixture', 'https://example.org/a', 'title', 'now', 1)"
        )
        for ordinal, (category, kind) in enumerate([
            ("MONETARY_POLICY", "REPORTED"),
            ("ECONOMIC_INDICATOR", "FORECAST"),
            ("ECONOMIC_INDICATOR", "REPORTED"),
            ("CORPORATE_EVENT", "REPORTED"),
        ]):
            connection.execute(
                "INSERT INTO article_event_claims (article_id, extractor_version, "
                "ordinal, classification, tag, claim_kind, polarity, temporal_context, "
                "status, evidence, verification_note, source, source_url, detected_at, "
                "processed_at, capture_sha256) VALUES "
                "(1, 'claims-v1', ?, ?, 'INFLATION', ?, 'AFFIRMED', 'UNSPECIFIED', "
                "'UNVERIFIED', 'Reported market news.', 'THIRD_PARTY_REPORT', "
                "'bok_all', 'https://example.org/a', 'now', 'now', 'hash')",
                (ordinal, category, kind),
            )
    assert ClaimEventBridge(db).process_pending().created == 4


def test_eligible_selection_skips_initial_ineligible_rows_with_batch_one(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    _claim_fixture(db)
    provider = EmptyProvider()
    first = ImpactProcessor(db, provider).process_pending(batch_size=1)
    assert first.processed == 1
    assert provider.ids == [3]
    second = ImpactProcessor(db, provider).process_pending(batch_size=1)
    assert second.processed == 1
    assert provider.ids == [3, 4]
    assert ImpactProcessor(db, provider).process_pending(batch_size=1).processed == 0
    with NewsStore(db).connect() as connection:
        rows = connection.execute(
            "SELECT event_id, outcome FROM impact_runs ORDER BY event_id"
        ).fetchall()
    assert rows == [(3, "NO_HYPOTHESIS"), (4, "NO_HYPOTHESIS")]


def test_cli_checks_local_model_before_database_changes(tmp_path: Path) -> None:
    db = tmp_path / "untouched.sqlite3"
    response = httpx.Response(200, json={"models": [{"name": "other:latest"}]})
    with (
        patch("sys.argv", [
            "market-event-analyze", "--provider", "local", "--db", str(db),
            "--batch-size", "1",
        ]),
        patch.dict("os.environ", {"MARKET_EVENT_LLM_MODEL": "gemma2:latest"}),
        patch("tris_market_event_intelligence.impact_cli.httpx.get", return_value=response),
        pytest.raises(SystemExit) as exc,
    ):
        analyze_main()
    assert exc.value.code == 2
    assert not db.exists()


def test_cli_uses_selected_local_model_and_explicit_database(tmp_path: Path, capsys) -> None:
    db = tmp_path / "existing.sqlite3"
    _claim_fixture(db)
    model = EmptyProvider()
    with (
        patch("sys.argv", [
            "market-event-analyze", "--provider", "local", "--db", str(db),
            "--batch-size", "1",
        ]),
        patch.dict("os.environ", {"MARKET_EVENT_LLM_MODEL": "gemma2:latest"}),
        patch("tris_market_event_intelligence.impact_cli.httpx.get", return_value=httpx.Response(
            200, json={"models": [{"name": "gemma2:latest"}]}
        )),
        patch("tris_market_event_intelligence.impact_cli.OpenAICompatibleProvider",
              return_value=model),
    ):
        analyze_main()
    result = json.loads(capsys.readouterr().out)
    assert result == {"processed": 1, "impacts": 0, "failed": 0}
    assert model.ids == [3]
