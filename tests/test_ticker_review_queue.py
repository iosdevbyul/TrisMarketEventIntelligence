"""Exact-bundle review reconciliation must not promote post-hoc decisions."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from test_ticker_human_review import fixture, write_decision

from tris_market_event_intelligence.ticker_review_ledger import import_review_decisions
from tris_market_event_intelligence.ticker_review_queue import reconcile_ticker_review_queue
from tris_market_event_intelligence.ticker_review_queue_cli import main


def test_pending_before_review_and_approved_after_review(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals)
    ledger = tmp_path / "reviews.sqlite3"
    import_review_decisions(ledger, proposals, decisions)
    result = reconcile_ticker_review_queue(proposals, ledger)
    assert result["pending"] == 0
    assert result["approved_post_hoc"] == 1
    assert result["rejected"] == 0
    assert result["items"][0]["review_status"] == "APPROVED_POST_HOC"
    assert result["point_in_time_training_eligible"] is False
    assert result["approved_ticker_links_created"] is False


def test_partially_reviewed_bundle_reports_pending(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    document = json.loads(proposals.read_text())
    other = dict(document["items"][0]["proposals"][0])
    other["symbol"] = "AAPL"
    document["items"][0]["proposals"].append(other)
    proposals.write_text(json.dumps(document), encoding="utf-8")
    write_decision(decisions, proposals)
    ledger = tmp_path / "reviews.sqlite3"
    import_review_decisions(ledger, proposals, decisions)
    before = ledger.read_bytes()
    report = reconcile_ticker_review_queue(proposals, ledger)
    assert report["pending"] == 1
    assert report["approved_post_hoc"] == 1
    assert report["items"][0]["symbol"] == "AAPL"
    assert report["items"][0]["ledger_id"] is None
    assert ledger.read_bytes() == before


def test_new_proposal_bundle_does_not_reuse_older_review(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals)
    ledger = tmp_path / "reviews.sqlite3"
    import_review_decisions(ledger, proposals, decisions)
    document = json.loads(proposals.read_text())
    document["note"] = "new independently generated proposal bundle"
    proposals.write_text(json.dumps(document), encoding="utf-8")
    report = reconcile_ticker_review_queue(proposals, ledger)
    assert report["pending"] == 1
    assert report["approved_post_hoc"] == 0


def test_rejected_decisions_not_promoted(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals, decision="REJECT")
    ledger = tmp_path / "reviews.sqlite3"
    import_review_decisions(ledger, proposals, decisions)
    result = reconcile_ticker_review_queue(proposals, ledger)
    assert result["rejected"] == 1
    assert result["approved_post_hoc"] == 0


def test_bad_ledger_and_future_evidence_fail_closed(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals)
    ledger = tmp_path / "reviews.sqlite3"
    import_review_decisions(ledger, proposals, decisions)
    with sqlite3.connect(ledger) as db:
        db.execute("UPDATE review_decisions SET symbol='AAPL' WHERE id=1")
    with pytest.raises(ValueError, match="integrity"):
        reconcile_ticker_review_queue(proposals, ledger)
    document = json.loads(proposals.read_text())
    document["items"][0]["proposals"][0][
        "registry_evidence_seen_at"
    ] = "2027-01-01T00:00:00+00:00"
    proposals.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Future listing evidence"):
        reconcile_ticker_review_queue(proposals, ledger)


def test_refuse_news_db_and_missing_db_without_writes(tmp_path: Path) -> None:
    proposals, _ = fixture(tmp_path)
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(ValueError):
        reconcile_ticker_review_queue(proposals, missing)
    assert not missing.exists()
    news = tmp_path / "news.sqlite3"
    with sqlite3.connect(news) as db:
        db.execute("CREATE TABLE articles (id INTEGER)")
    before = news.read_bytes()
    with pytest.raises(ValueError):
        reconcile_ticker_review_queue(proposals, news)
    assert news.read_bytes() == before


def test_duplicate_candidates_and_training_flags_rejected(tmp_path: Path) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals)
    ledger = tmp_path / "reviews.sqlite3"
    import_review_decisions(ledger, proposals, decisions)
    document = json.loads(proposals.read_text())
    document["items"][0]["proposals"].append(document["items"][0]["proposals"][0])
    proposals.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Duplicate"):
        reconcile_ticker_review_queue(proposals, ledger)
    document["items"][0]["proposals"].pop()
    document["training_dataset_eligible"] = True
    proposals.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="safety flags"):
        reconcile_ticker_review_queue(proposals, ledger)


def test_cli_read_only_success_and_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    proposals, decisions = fixture(tmp_path)
    write_decision(decisions, proposals)
    ledger = tmp_path / "reviews.sqlite3"
    import_review_decisions(ledger, proposals, decisions)
    with patch.object(sys, "argv", [
        "market-event-ticker-review-queue", "--ledger", str(ledger),
        "--proposals", str(proposals),
    ]):
        main()
    assert json.loads(capsys.readouterr().out)["approved_post_hoc"] == 1
    with patch.object(sys, "argv", [
        "market-event-ticker-review-queue", "--ledger", str(tmp_path / "absent.sqlite3"),
        "--proposals", str(proposals),
    ]), pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
