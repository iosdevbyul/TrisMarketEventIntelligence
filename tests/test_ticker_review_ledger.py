"""An isolated review ledger must preserve research-only decisions and news DB."""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from test_ticker_human_review import fixture, write_decision

from tris_market_event_intelligence.ticker_review_ledger import (
    import_review_decisions,
    list_review_decisions,
)
from tris_market_event_intelligence.ticker_review_ledger_cli import main


def test_import_list_idempotent_and_event_filter(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals)
    ledger = tmp_path / "reviews.sqlite3"
    assert not ledger.exists()
    first = import_review_decisions(ledger, proposals, decisions)
    assert first["inserted"] == 1
    assert first["identical_replays"] == 0
    assert first["point_in_time_training_eligible"] is False
    assert first["news_database_modified"] is False
    assert first["approved_ticker_links_created"] is False
    second = import_review_decisions(ledger, proposals, decisions)
    assert second["inserted"] == 0
    assert second["identical_replays"] == 1
    found = list_review_decisions(ledger, event_id=7)
    assert found["returned"] == 1
    assert found["items"][0]["decision"] == "APPROVE"
    assert found["items"][0]["eligible_for_point_in_time_training"] is False
    assert "benchmark_symbol" not in found["items"][0]
    assert list_review_decisions(ledger, event_id=8)["items"] == []


def test_rejected_decision_retained_not_training_ready(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals, decision="REJECT")
    ledger = tmp_path / "reviews.sqlite3"
    assert import_review_decisions(ledger, proposals, decisions)["inserted"] == 1
    result = list_review_decisions(ledger)
    assert result["items"][0]["review_status"] == "REJECTED"
    assert result["point_in_time_training_eligible"] is False


def test_conflicting_review_rejected_and_original_unchanged(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals)
    ledger = tmp_path / "reviews.sqlite3"
    import_review_decisions(ledger, proposals, decisions)
    write_decision(decisions, proposals, decision="REJECT")
    with pytest.raises(ValueError, match="immutable"):
        import_review_decisions(ledger, proposals, decisions)
    assert list_review_decisions(ledger)["items"][0]["decision"] == "APPROVE"


def test_news_database_refused_before_any_write(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals)
    news_db = tmp_path / "news.sqlite3"
    with sqlite3.connect(news_db) as connection:
        connection.execute("CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT)")
        connection.execute("INSERT INTO articles VALUES (1, 'Original')")
    before = news_db.read_bytes()
    with pytest.raises(ValueError, match="ledger identity"):
        import_review_decisions(news_db, proposals, decisions)
    assert news_db.read_bytes() == before
    with pytest.raises(ValueError, match="review"):
        list_review_decisions(news_db)
    assert news_db.read_bytes() == before


def test_missing_and_symlinked_ledger_refused(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals)
    missing = tmp_path / "missing.db"
    with pytest.raises(ValueError, match="initialized"):
        list_review_decisions(missing)
    assert not missing.exists()
    ledger = tmp_path / "reviews.sqlite3"
    import_review_decisions(ledger, proposals, decisions)
    alias = tmp_path / "alias.db"
    alias.symlink_to(ledger)
    with pytest.raises(ValueError, match="non-symlink"):
        list_review_decisions(alias)
    with pytest.raises(ValueError, match="non-symlink"):
        import_review_decisions(alias, proposals, decisions)


def test_record_integrity_catches_manual_mutation(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals)
    ledger = tmp_path / "reviews.sqlite3"
    import_review_decisions(ledger, proposals, decisions)
    with sqlite3.connect(ledger) as connection:
        connection.execute("UPDATE review_decisions SET record_sha256=? WHERE id=1", ("0" * 64,))
    with pytest.raises(ValueError, match="integrity"):
        list_review_decisions(ledger)


def test_invalid_review_does_not_initialize_db(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals, proposal_sha256="f" * 64)
    ledger = tmp_path / "review.sqlite3"
    with pytest.raises(ValueError, match="row 1"):
        import_review_decisions(ledger, proposals, decisions)
    assert not ledger.exists()


def test_cli_import_and_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals)
    ledger = tmp_path / "reviews.sqlite3"
    with patch.object(sys, "argv", [
        "market-event-ticker-ledger", "--ledger", str(ledger),
        "--import-reviewed", "--proposals", str(proposals),
        "--decisions", str(decisions),
    ]):
        main()
    assert json.loads(capsys.readouterr().out)["inserted"] == 1
    with patch.object(sys, "argv", [
        "market-event-ticker-ledger", "--ledger", str(ledger),
        "--list", "--event-id", "7",
    ]):
        main()
    assert json.loads(capsys.readouterr().out)["returned"] == 1


def test_review_evidence_hash_remains_pinned(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals)
    ledger = tmp_path / "reviews.sqlite3"
    imported = import_review_decisions(ledger, proposals, decisions)
    assert imported["source_proposal_sha256"] == hashlib.sha256(
        proposals.read_bytes()
    ).hexdigest()
    with decisions.open(encoding="utf-8", newline="") as stream:
        record = next(csv.DictReader(stream))
    assert record["proposal_sha256"] == imported["source_proposal_sha256"]
