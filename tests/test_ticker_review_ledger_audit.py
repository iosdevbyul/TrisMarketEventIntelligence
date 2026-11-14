"""Whole-ledger audit must verify all reviews without rewriting evidence."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from test_ticker_human_review import fixture, write_decision

from tris_market_event_intelligence.ticker_review_ledger import import_review_decisions
from tris_market_event_intelligence.ticker_review_ledger_audit import (
    audit_review_ledger,
)
from tris_market_event_intelligence.ticker_review_ledger_audit_cli import main


def stored_ledger(tmp_path: Path) -> Path:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals)
    ledger = tmp_path / "reviews.sqlite3"
    import_review_decisions(ledger, proposals, decisions)
    return ledger


def test_full_audit_readonly_and_no_training_export(tmp_path: Path) -> None:
    ledger = stored_ledger(tmp_path)
    before = ledger.read_bytes()
    result = audit_review_ledger(ledger)
    assert result["status"] == "VERIFIED_POST_HOC_RECORD_INTEGRITY_ONLY"
    assert result["records_checked"] == 1
    assert result["approved_post_hoc"] == 1
    assert result["point_in_time_training_eligible"] is False
    assert result["approved_ticker_links_created"] is False
    assert result["external_evidence_authenticated"] is False
    assert ledger.read_bytes() == before


def test_corrupt_indexed_column_detected_even_with_correct_json_digest(tmp_path: Path) -> None:
    ledger = stored_ledger(tmp_path)
    with sqlite3.connect(ledger) as db:
        db.execute("UPDATE review_decisions SET symbol='FAKE' WHERE id=1")
    before = ledger.read_bytes()
    with pytest.raises(ValueError, match="integrity audit failed"):
        audit_review_ledger(ledger)
    assert ledger.read_bytes() == before


def test_corrupt_review_data_detected_even_if_digest_recomputed(tmp_path: Path) -> None:
    from tris_market_event_intelligence.ticker_review_ledger import _digest, _payload

    ledger = stored_ledger(tmp_path)
    with sqlite3.connect(ledger) as db:
        raw = db.execute("SELECT record_json FROM review_decisions").fetchone()[0]
        value = json.loads(raw)
        value["eligible_for_point_in_time_training"] = True
        modified = _payload(value)
        db.execute(
            "UPDATE review_decisions SET record_json=?,record_sha256=? WHERE id=1",
            (modified, _digest(modified)),
        )
    with pytest.raises(ValueError, match="integrity audit failed"):
        audit_review_ledger(ledger)


def test_missing_and_production_db_fail_without_creating_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(ValueError):
        audit_review_ledger(missing)
    assert not missing.exists()
    news = tmp_path / "news.sqlite3"
    with sqlite3.connect(news) as db:
        db.execute("CREATE TABLE articles(id INTEGER)")
    before = news.read_bytes()
    with pytest.raises(ValueError):
        audit_review_ledger(news)
    assert news.read_bytes() == before


def test_cli_success_and_failure_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    ledger = stored_ledger(tmp_path)
    with patch.object(sys, "argv", ["market-event-ticker-ledger-audit", "--ledger", str(ledger)]):
        main()
    assert json.loads(capsys.readouterr().out)["records_checked"] == 1
    with patch.object(sys, "argv", [
        "market-event-ticker-ledger-audit", "--ledger", str(tmp_path / "missing.db"),
    ]), pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
