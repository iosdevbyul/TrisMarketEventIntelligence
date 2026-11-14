"""Real stored listing evidence -> mock RSS -> extracted events -> review output.

These tests do not contact publishers or the operator's Mac database.
"""
from __future__ import annotations

import fcntl
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from tris_market_event_intelligence.impact import ImpactResult
from tris_market_event_intelligence.integrated_pipeline import run_integrated_pass
from tris_market_event_intelligence.listing_research_workflow import (
    acquire_and_reconcile_listings,
)


def snapshot(tmp_path: Path) -> Path:
    reviewed = tmp_path / "actual-review.csv"
    reviewed.write_text(
        "market,symbol,company_name,security_type\n"
        "XKRX,005930,삼성전자,COMMON\n", encoding="utf-8",
    )
    bundle = tmp_path / "observed-issuers"
    acquire_and_reconcile_listings(
        bundle, krx_reviewed=reviewed,
        now=datetime.now(UTC) - timedelta(hours=1),
    )
    return bundle


def mock_feed() -> httpx.Client:
    content = (
        "<rss><channel><item><guid>news-1</guid>"
        "<title>삼성전자 서비스 장애</title>"
        "<link>https://example.org/real-event</link>"
        "</item></channel></rss>"
    ).encode()
    return httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=content),
    ))


def test_opt_in_collect_extract_match_and_report_in_one_locked_pass(
    tmp_path: Path,
) -> None:
    bundle = snapshot(tmp_path)
    root = tmp_path / "reports"
    root.mkdir()
    db = tmp_path / "news.sqlite3"
    with mock_feed() as client:
        first = run_integrated_pass(
            db, selected_sources=("bbc_technology",),
            client=client, listing_bundle=bundle, review_output_root=root,
        )
    assert first["status"] == "healthy"
    assert first["collection"]["new_articles"] == 1
    assert first["event_extraction"]["events"] == 1
    assert first["ticker_review"]["status"] == "REVIEW_REQUIRED"
    assert first["ticker_review"]["candidates"] == 1
    assert first["ticker_review"]["automatic_approvals"] == 0
    assert first["model_invoked"] is False
    summary = json.loads(
        Path(first["ticker_review"]["summary_path"]).read_text()
    )
    assert summary["post_observation_news_events"] == 1
    assert summary["pending_human_review"] == 1
    assert summary["point_in_time_training_eligible"] is False
    assert summary["news_pipeline_evidence"]["articles_total"] == 1
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM economic_events").fetchone()[0] == 1
    with mock_feed() as client:
        second = run_integrated_pass(
            db, selected_sources=("bbc_technology",),
            client=client, listing_bundle=bundle, review_output_root=root,
        )
    assert second["status"] == "healthy"
    assert second["collection"]["new_articles"] == 0
    assert second["event_extraction"]["events"] == 0
    assert second["ticker_review"]["candidates"] == 1
    assert second["ticker_review"]["summary_path"] != first["ticker_review"]["summary_path"]


def test_invalid_listing_bundle_prevents_network_and_new_database(tmp_path: Path) -> None:
    bundle = snapshot(tmp_path)
    root = tmp_path / "reports"
    root.mkdir()
    (bundle / "krx-reviewed.csv").write_text("tampered")
    db = tmp_path / "must-not-create.sqlite3"
    with mock_feed() as client, pytest.raises(ValueError, match="SHA-256"):
        run_integrated_pass(
            db, selected_sources=("bbc_technology",),
            client=client, listing_bundle=bundle, review_output_root=root,
        )
    assert not db.exists()
    assert list(root.iterdir()) == []


def test_locked_integrated_pass_cannot_mutate_or_collect(tmp_path: Path) -> None:
    bundle = snapshot(tmp_path)
    root = tmp_path / "reports"
    root.mkdir()
    db = tmp_path / "news.sqlite3"
    lockpath = tmp_path / "news.sqlite3.pipeline.lock"
    lockpath.touch()
    with lockpath.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with mock_feed() as client:
            report = run_integrated_pass(
                db, selected_sources=("bbc_technology",),
                client=client, listing_bundle=bundle, review_output_root=root,
            )
    assert report["status"] == "already_running"
    assert not db.exists()
    assert list(root.iterdir()) == []


class ForbiddenModel:
    model_id = "never-use"
    def analyze(self, event: object) -> ImpactResult:
        raise AssertionError("Ticker capture must not call a model")


def test_listing_review_rejects_implicit_model_and_wrong_folders(tmp_path: Path) -> None:
    bundle = snapshot(tmp_path)
    root = tmp_path / "private"
    root.mkdir()
    db = tmp_path / "no-db.sqlite3"
    with pytest.raises(ValueError, match="must not invoke"):
        run_integrated_pass(
            db, listing_bundle=bundle, review_output_root=root,
            model=ForbiddenModel(), enable_local_analysis=True,
        )
    with pytest.raises(ValueError, match="both"):
        run_integrated_pass(db, listing_bundle=bundle)
    with pytest.raises(ValueError, match="folder"):
        run_integrated_pass(
            db, listing_bundle=bundle, review_output_root=tmp_path / "missing",
        )
    assert not db.exists()


def test_reviewed_listing_mode_does_not_process_pre_snapshot_article(
    tmp_path: Path,
) -> None:
    bundle = snapshot(tmp_path)
    root = tmp_path / "reports"
    root.mkdir()
    db = tmp_path / "news.sqlite3"
    from tris_market_event_intelligence.collector import NewsStore, parse_rss

    store = NewsStore(db)
    store.initialize()
    raw = (
        "<rss><channel><item><guid>old</guid><title>"
        "삼성전자 서비스 장애"
        "</title><link>https://example.org/old</link></item></channel></rss>"
    ).encode()
    old_at = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    store.save(
        "bbc_technology", "https://example.org/feed", raw,
        parse_rss("bbc_technology", raw, old_at),
        old_at, None, None,
    )
    with mock_feed() as client:
        result = run_integrated_pass(
            db, selected_sources=("bbc_technology",), client=client,
            listing_bundle=bundle, review_output_root=root,
        )
    assert result["event_extraction"]["events"] == 1
    with sqlite3.connect(db) as connection:
        rows = connection.execute(
            "SELECT a.detected_at FROM economic_events e "
            "JOIN articles a ON a.id=e.article_id"
        ).fetchall()
    assert len(rows) == 1
    assert datetime.fromisoformat(rows[0][0]) > datetime.now(UTC) - timedelta(hours=1)


def test_cli_requires_explicit_review_opt_in(tmp_path: Path) -> None:
    import sys

    from tris_market_event_intelligence.integrated_pipeline_cli import main
    with patch.object(sys, "argv", [
        "market-event-integrated", "--listing-bundle", str(tmp_path),
        "--dry-run",
    ]), pytest.raises(SystemExit):
        main()
    with patch.object(sys, "argv", [
        "market-event-integrated", "--enable-ticker-review",
        "--listing-bundle", str(tmp_path), "--review-output-root", str(tmp_path),
        "--enable-local-analysis", "--dry-run",
    ]), pytest.raises(SystemExit):
        main()
