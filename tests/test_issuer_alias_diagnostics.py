"""Review-only alias coverage and verified post-capture news diagnostics."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.event_ticker_proposals import ListedIssuer
from tris_market_event_intelligence.issuer_alias_diagnostics import (
    derived_alias_candidates,
    inspect_alias_coverage,
)
from tris_market_event_intelligence.listing_news_validation import (
    validate_existing_listing_news,
)
from tris_market_event_intelligence.listing_research_workflow import (
    acquire_and_reconcile_listings,
)

SNAPSHOT = datetime(2026, 11, 15, 0, 0, tzinfo=UTC)
CUTOFF = datetime(2026, 11, 15, 5, 0, tzinfo=UTC)


def issuer(symbol: str, name: str, market: str = "XNAS") -> ListedIssuer:
    return ListedIssuer(
        market=market, symbol=symbol, canonical_name=name,
        aliases=(name,), valid_from=SNAPSHOT, valid_until=None,
        evidence_seen_at=SNAPSHOT, source_id="actual-source",
    )


def news(path: Path, rows: list[tuple[str, str]]) -> None:
    with sqlite3.connect(path) as db:
        db.executescript(
            """CREATE TABLE articles (
                id INTEGER PRIMARY KEY, title TEXT, summary TEXT, detected_at TEXT
            );
            CREATE TABLE economic_events (
                id INTEGER PRIMARY KEY, article_id INTEGER, status TEXT,
                source TEXT, source_url TEXT
            );"""
        )
        for index, (title, received) in enumerate(rows, start=1):
            db.execute("INSERT INTO articles VALUES (?,?,'',?)", (index, title, received))
            db.execute(
                "INSERT INTO economic_events VALUES (?,?,'VERIFIED','test',?)",
                (index, index, f"https://example.org/{index}"),
            )


def test_derivation_never_creates_approved_aliases() -> None:
    original = (
        issuer("MSFT", "Microsoft Corporation"),
        issuer("AAPL", "Apple Inc."),
        issuer("ABC", "ABC Inc."),
        issuer("HLD", "Example Holdings"),
    )
    aliases = derived_alias_candidates(original)
    assert "microsoft" in aliases
    assert "apple" not in aliases  # single generic word too short
    assert "abc" not in aliases
    assert "example" not in aliases  # no holdings removal
    assert original[0].aliases == ("Microsoft Corporation",)


def test_reports_old_new_ambiguous_and_future_alias_mentions(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    news(db, [
        ("Microsoft unveils product", "2026-11-14T23:00:00+00:00"),
        ("Microsoft releases update", "2026-11-15T02:00:00+00:00"),
        ("Microsoft announces plan", "2026-11-15T06:00:00+00:00"),
        ("Unknown industry report", "2026-11-15T02:00:00+00:00"),
    ])
    original = (
        issuer("MSFT", "Microsoft Corporation"),
        issuer("MSFT2", "Microsoft Inc.", "XNYS"),
    )
    before = db.read_bytes()
    report = inspect_alias_coverage(
        db, original, [1, 2, 3, 4],
        first_seen=SNAPSHOT, as_of=CUTOFF,
    )
    assert report["hints_total"] == 3
    assert report["counts"] == {
        "AMBIGUOUS_IDENTITY": 2,
        "FUTURE_NEWS_TIMESTAMP": 1,
    }
    assert len(report["hints"]) == 3
    assert len(report["hints"][1]["possible_issuers"]) == 2
    assert all(h["approved_ticker_link"] is False for h in report["hints"])
    assert all(h["training_dataset_eligible"] is False for h in report["hints"])
    assert db.read_bytes() == before


def test_unique_issuer_is_only_human_review_hint(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    news(db, [
        ("Microsoft unveils product", "2026-11-14T23:00:00+00:00"),
        ("Microsoft releases update", "2026-11-15T02:00:00+00:00"),
    ])
    report = inspect_alias_coverage(
        db, (issuer("MSFT", "Microsoft Corporation"),), [1, 2],
        first_seen=SNAPSHOT, as_of=CUTOFF,
    )
    assert report["counts"] == {
        "NEEDS_HUMAN_ALIAS_REVIEW": 1,
        "PRE_OBSERVATION_ONLY": 1,
    }
    assert report["automatic_links_created"] is False
    assert report["hints"][0]["diagnostic_status"] == "PRE_OBSERVATION_ONLY"
    assert report["hints"][1]["diagnostic_status"] == "NEEDS_HUMAN_ALIAS_REVIEW"


def test_actual_krx_bundle_forward_new_news_and_no_alias_autoapproval(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "reviewed.csv"
    raw.write_text(
        "market,symbol,company_name,security_type\n"
        "XKRX,005930,삼성전자,COMMON\n",
        encoding="utf-8",
    )
    bundle = tmp_path / "observed-source"
    acquire_and_reconcile_listings(bundle, krx_reviewed=raw, now=SNAPSHOT)
    db = tmp_path / "news.sqlite3"
    news(db, [
        ("삼성전자 first news", "2026-11-14T23:00:00+00:00"),
        ("삼성전자 new results", "2026-11-15T01:00:00+00:00"),
        ("No issuer mentioned", "2026-11-15T02:00:00+00:00"),
    ])
    full = validate_existing_listing_news(
        bundle, db, tmp_path / "full", as_of=CUTOFF,
    )
    assert full["events_inspected"] == 3
    assert full["candidate_count"] == 1
    assert full["post_observation_news_events"] == 2
    assert full["diagnostic_alias_mentions"] == 0
    fresh = validate_existing_listing_news(
        bundle, db, tmp_path / "fresh", as_of=CUTOFF,
        post_snapshot_only=True,
    )
    assert fresh["events_inspected"] == 2
    assert fresh["post_observation_news_events"] == 2
    assert fresh["pre_observation_news_events"] == 0
    assert fresh["candidate_count"] == 1
    assert fresh["new_news_available"] is True
    assert fresh["point_in_time_training_eligible"] is False
    assert fresh["diagnostic_aliases_approved"] == 0
    details = json.loads(
        (tmp_path / "fresh" / "alias-diagnostics-page-001.json").read_text()
    )
    assert details["automatic_links_created"] is False
    assert fresh["pages"][0]["alias_diagnostics_sha256"]


def test_fresh_news_mode_returns_empty_without_synthetic_events(tmp_path: Path) -> None:
    raw = tmp_path / "reviewed.csv"
    raw.write_text(
        "market,symbol,company_name,security_type\n"
        "XKRX,005930,삼성전자,COMMON\n",
        encoding="utf-8",
    )
    bundle = tmp_path / "observed-source"
    acquire_and_reconcile_listings(bundle, krx_reviewed=raw, now=SNAPSHOT)
    db = tmp_path / "news.sqlite3"
    news(db, [("삼성전자 older report", "2026-11-14T23:00:00+00:00")])
    result = validate_existing_listing_news(
        bundle, db, tmp_path / "empty", as_of=CUTOFF,
        post_snapshot_only=True,
    )
    assert result["events_inspected"] == 0
    assert result["candidate_count"] == 0
    assert result["new_news_available"] is False
    assert result["has_more_events"] is False
    assert result["pages"] == []
    assert result["diagnostic_alias_mentions"] == 0
